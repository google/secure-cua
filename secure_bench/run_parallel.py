
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Same-site-parallel launcher: run a batch of WebArena tasks with K workers PER SITE.

Each site's tasks are split across K workers; every worker runs its OWN container on a derived port
(base + 10000*i) via run_suite (per-task reset + mutation isolation intact). Worker 0 uses the site's base
container/port; workers 1..K-1 use extra containers. Cross-site suites also run concurrently, so total
concurrency = (#sites) * K. Verified integrity-equivalent to the sequential per-site path (same task ->
same verdict on base vs worker container).

  python run_parallel.py --workers 3 --mode secure --model claude-opus-4-8 --out-root vm_results_x --tasks <ids...>

All workers write into ONE --out-root (tasks are partitioned, so <out-root>/<task>_secure never collides).
`--skip-done` is forwarded (resumable). Extra worker containers (i>0) are removed on exit unless --keep.

GRACEFUL SHUTDOWN (no orphans, no dangling ports/locks). The tree is launcher -> N run_suite workers (each its
own session) -> per task a nested DRIVER session (xvfb-run + Xvfb + chromium). A plain kill would orphan the
nested driver, so:
 - on SIGTERM/SIGINT the launcher SIGTERMs each worker (run_suite's own handler then tears down its in-flight
   driver + reset-phase children), waits, SIGKILLs any survivor's group, reaps any driver session recorded in a
   worker sidecar, and removes the extra containers — teardown runs FIRST and each step is guarded so nothing
   can abort it; a SECOND signal hard-escalates to SIGKILL;
 - the whole spawn+register is done under a brief signal block so a signal can never land mid-spawn and orphan a
   not-yet-tracked worker;
 - a run manifest `<out-root>/_parallel_run.json` (parent + worker pids/pgids + start-times) plus per-worker
   `_driverpg_<pid>` sidecars make an already-orphaned run recoverable via `python run_parallel.py --stop
   <out-root>` — which identity-validates every pid (process start-time) before signalling, so a recycled PID
   is never collateral-killed.
"""
import argparse
import errno
import glob
import json
import os
import paths
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import wa_workers as ww  # noqa: E402

PYEXE = sys.executable
MANIFEST_NAME = "_parallel_run.json"
_SIGSET = {signal.SIGTERM, signal.SIGINT}

# ---- module state so the signal handler can tear the run down cleanly (populated by main/_run_realbench) ----
_POPENS = []                 # live worker Popen objects
_WORKER_CONTAINERS = []      # extra (i>0) containers to remove on exit
_DOCKER_HOST = None
_OUT_ROOT = None
_MANIFEST_PATH = None
_SHUTTING_DOWN = False


def _pgid(pid):
    try:
        return os.getpgid(pid)
    except Exception:
        return pid


def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except Exception:
        pass


def _proc_start_time(pid):
    """/proc/<pid>/stat field 22 (starttime, clock ticks since boot) — a stable per-process identity token that,
    with the pid, distinguishes THIS process from a later one that recycled the pid. None if unavailable."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            data = f.read()
        rhs = data[data.rindex(")") + 1:].split()          # comm can contain spaces/parens -> split after the last ')'
        return int(rhs[19])                                # field 22 == rhs[19] (rhs[0] is field 3, 'state')
    except Exception:
        return None


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM
    except Exception:
        return False


def _pgid_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM
    except Exception:
        return False


def _same_proc(pid, start_time):
    """True if pid is alive AND (when a start_time was recorded) is still the SAME process — guards --stop
    against pid reuse. With no recorded start_time, fall back to a bare liveness check."""
    if not pid:
        return False
    if start_time is None:
        return _pid_alive(pid)
    st = _proc_start_time(pid)
    return st is not None and st == start_time


def _child_unblock_signals():
    """preexec in a spawned worker: undo the parent's brief SIGTERM/SIGINT block so the worker can be signalled
    (its own run_suite handler must receive SIGTERM to tear its driver down)."""
    try:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGSET)
    except Exception:
        pass


def _kill_driver_sidecars(out_root):
    """SIGKILL every driver session a worker published in `<out-root>/_driverpg_<pid>` — reaps an orphaned
    driver+xvfb+chromium even when its worker died uncooperatively (kill -9 / OOM) and never ran its handler."""
    if not out_root:
        return
    for f in glob.glob(os.path.join(out_root, "_driverpg_*")):
        try:
            pg = int((open(f, encoding="utf-8").read() or "").strip())
        except Exception:
            pg = None
        if pg and _pgid_alive(pg):
            _killpg(pg, signal.SIGKILL)


def _terminate(popens, grace=20.0):
    """Graceful stop: SIGTERM each worker (run_suite) so ITS handler kills its own driver+Chromium+reset children
    and exits; wait up to `grace`, then SIGKILL any survivor's whole process group."""
    alive = [p for p in popens if p and p.poll() is None]
    for p in alive:
        try:
            p.terminate()                  # SIGTERM -> run_suite._install_signal_handlers cascades to the driver
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < grace and any(p.poll() is None for p in alive):
        time.sleep(0.5)
    for p in alive:
        if p.poll() is None:
            _killpg(_pgid(p.pid), signal.SIGKILL)


def _cleanup_containers(containers, docker_host):
    for c in containers:
        try:
            subprocess.run(["docker", "rm", "-f", c],
                           env=dict(os.environ, DOCKER_HOST=(docker_host or paths.DOCKER_HOST)),
                           capture_output=True, timeout=60)
        except Exception:
            pass


def _remove_manifest(path=None):
    path = path or _MANIFEST_PATH
    if path:
        try:
            os.remove(path)
        except Exception:
            pass


def _write_manifest(path, out_root, mode, model, workers, containers):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"parent_pid": os.getpid(), "parent_start_time": _proc_start_time(os.getpid()),
                       "out_root": os.path.abspath(out_root), "mode": mode, "model": model,
                       "workers": workers, "worker_containers": containers,
                       "stop_hint": f"python run_parallel.py --stop {out_root}"}, f, indent=2)
    except Exception as e:
        try:
            os.write(2, f"[parallel] WARN: could not write manifest {path}: {e}\n".encode())
        except Exception:
            pass


def _on_signal(signum, _frame):
    """Async-signal-safe teardown: no buffered `print` (a reentrant BufferedWriter would abort teardown), each
    step guarded, teardown BEFORE any I/O; a second signal hard-escalates to SIGKILL."""
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:                                      # second signal -> skip grace, SIGKILL everything, die
        for p in _POPENS:
            try:
                if p.poll() is None:
                    _killpg(_pgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        _kill_driver_sidecars(_OUT_ROOT)
        os._exit(137)
    _SHUTTING_DOWN = True
    try:
        _terminate(_POPENS)
    except Exception:
        pass
    try:
        _kill_driver_sidecars(_OUT_ROOT)                   # reap any driver whose worker died uncooperatively
    except Exception:
        pass
    try:
        _cleanup_containers(_WORKER_CONTAINERS, _DOCKER_HOST)
    except Exception:
        pass
    try:
        _remove_manifest()
    except Exception:
        pass
    try:
        os.write(2, f"[parallel] signal {signum}: stopped {len(_POPENS)} worker(s)\n".encode())
    except Exception:
        pass
    os._exit(130 if signum == signal.SIGINT else 143)


def _install_handlers():
    for s in _SIGSET:
        try:
            signal.signal(s, _on_signal)
        except Exception:
            pass


def _stop(out_root):
    """External clean stop of a (possibly already-orphaned) run: read <out-root>/_parallel_run.json, SIGTERM each
    LIVE, IDENTITY-VALIDATED worker (so run_suite tears its driver+Chromium down), wait, SIGKILL survivors'
    groups, reap orphaned driver sidecars, then the parent and the extra containers. pid-reuse safe."""
    path = os.path.join(out_root, MANIFEST_NAME)
    try:
        man = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        sys.exit(f"[parallel --stop] no readable run manifest at {path}: {e}")
    docker_host = os.environ.get("DOCKER_HOST", paths.DOCKER_HOST)
    workers = man.get("workers", [])
    for w in workers:                                      # 1) SIGTERM live workers (identity-checked) -> cascade to their drivers
        if _same_proc(w.get("pid"), w.get("start_time")):
            try:
                os.kill(w["pid"], signal.SIGTERM)
            except Exception:
                pass
    t0 = time.time()                                       # 2) wait for cooperative exit
    while time.time() - t0 < 20 and any(_same_proc(w.get("pid"), w.get("start_time")) for w in workers):
        time.sleep(0.5)
    for w in workers:                                      # 3) SIGKILL survivor groups (still identity-checked)
        pg = w.get("pgid") or w.get("pid")
        if _same_proc(w.get("pid"), w.get("start_time")) and pg and _pgid_alive(pg):
            _killpg(pg, signal.SIGKILL)
    _kill_driver_sidecars(out_root)                        # 4) reap orphaned driver sessions (worker already dead)
    ppid = man.get("parent_pid")                           # 5) the parent launcher, if still the same process
    if _same_proc(ppid, man.get("parent_start_time")):
        try:
            os.kill(ppid, signal.SIGTERM)
        except Exception:
            pass
    _cleanup_containers(man.get("worker_containers", []), docker_host)   # 6) extra containers + manifest + sidecars
    _remove_manifest(path)
    for f in glob.glob(os.path.join(out_root, "_driverpg_*")):
        try:
            os.remove(f)
        except Exception:
            pass
    print(f"[parallel --stop] stopped {len(workers)} worker(s) and cleaned up ({out_root})", flush=True)


def _by_site(tasks):
    d = {}
    for t in tasks:
        d.setdefault(ww.site_of(t), []).append(t)
    return d


def _partition(items, k):
    """Round-robin split into k lists (balances uneven task durations better than contiguous chunks)."""
    out = [[] for _ in range(k)]
    for i, it in enumerate(items):
        out[i % k].append(it)
    return [p for p in out if p]                      # drop empty partitions (fewer tasks than workers)


def _spawn(cmd, env, logf):
    """Popen a worker under the parent's brief signal-block (so a signal can't land between fork and register),
    unblocking signals in the child. Returns the Popen; caller registers it while still blocked."""
    fh = open(logf, "w")
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True, preexec_fn=_child_unblock_signals)


def _run_realbench(a):
    """realbench has no site containers: partition tasks K-way and run each partition as its own run_suite
    subprocess (--benchmark realbench --no-reset). run_suite gives each task its own xvfb-run display, and
    RealBenchmark allocates each a fresh --remote-debugging-port, so N-way parallelism is isolation-safe."""
    global _MANIFEST_PATH
    if not a.tasks:
        sys.exit("--benchmark realbench requires --tasks (e.g. --tasks omnizon-3 omnizon-9); --site is webarena-only")
    k = max(1, min(a.workers, len(a.tasks)))
    parts = _partition(a.tasks, k)
    os.makedirs(a.out_root, exist_ok=True)
    env0 = dict(os.environ,
                PLAYWRIGHT_BROWSERS_PATH=os.environ.get("PLAYWRIGHT_BROWSERS_PATH", paths.PLAYWRIGHT_BROWSERS_PATH))
    manifest_workers = []
    _blocked = signal.pthread_sigmask(signal.SIG_BLOCK, _SIGSET)
    try:
        for wi, part in enumerate(parts):
            cmd = [PYEXE, os.path.join(HERE, "run_suite.py"),
                   "--benchmark", "realbench", "--no-reset",
                   "--mode", a.mode, "--model", a.model, "--out-root", a.out_root,
                   "--max-steps", str(a.max_steps), "--per-task-timeout", str(a.per_task_timeout),
                   "--seeds", str(a.seeds), "--tasks", *part]
            if a.version:
                cmd += ["--version", a.version]
            if a.skip_done:
                cmd.append("--skip-done")
            logf = os.path.join(a.out_root, f"_parallel_realbench_w{wi}.log")
            print(f"[parallel] realbench worker{wi}  {len(part)} tasks -> {part}", flush=True)
            p = _spawn(cmd, env0, logf)
            _POPENS.append(p)
            manifest_workers.append({"site": "realbench", "lane": wi, "port": "", "pid": p.pid,
                                     "pgid": _pgid(p.pid), "start_time": _proc_start_time(p.pid), "log": logf})
        _MANIFEST_PATH = os.path.join(a.out_root, MANIFEST_NAME)
        _write_manifest(_MANIFEST_PATH, a.out_root, a.mode, a.model, manifest_workers, [])
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, _blocked)
    print(f"[parallel] launched {len(_POPENS)} realbench worker(s) ({k} lanes)", flush=True)
    for wi, p in enumerate(_POPENS):
        p.wait()
        print(f"[parallel] realbench worker{wi} exited rc={p.returncode}", flush=True)
    _remove_manifest()
    print("[parallel] done", flush=True)


def main():
    global _DOCKER_HOST, _MANIFEST_PATH, _OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", metavar="OUT_ROOT", default=None,
                    help="gracefully stop a running parallel run (reads <OUT_ROOT>/_parallel_run.json) + clean up, then exit")
    ap.add_argument("--workers", type=int, default=2, help=f"parallel workers PER SITE (<= {ww.MAX_WORKERS})")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--tasks", nargs="+", help="explicit task ids, e.g. reddit-404 shopping-21")
    src.add_argument("--site", choices=["reddit", "shopping", "shopping_admin", "gitlab"],
                     help="run ALL single-site WebArena tasks for this site (instead of listing --tasks)")
    ap.add_argument("--mode", default="secure", choices=["secure", "baseline", "noquery"])
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-hist", type=int, default=None,
                    help="cap the LLM history to the last N turns. secure/noquery -> SECURE_MAX_HIST, "
                         "baseline -> BASELINE_MAX_TRAJ. Default: secure/noquery = FULL history, baseline = 3.")
    ap.add_argument("--per-task-timeout", type=float, default=1800)
    ap.add_argument("--seeds", type=int, default=1, help="run each task across N independent seeds (repeats) -> pass@k")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--keep", action="store_true", help="don't remove extra worker containers on exit")
    ap.add_argument("--benchmark", default="webarena", choices=["webarena", "realbench"])
    ap.add_argument("--version", default=None, help="realbench task version (default v2); ignored for webarena")
    a = ap.parse_args()

    if a.stop:                                          # stop mode: no run, just tear an existing one down
        _stop(a.stop)
        return
    if not a.out_root:
        ap.error("--out-root is required")
    if not (a.tasks or a.site):
        ap.error("one of --tasks / --site is required")

    _OUT_ROOT = a.out_root
    _install_handlers()                                 # SIGTERM/SIGINT -> graceful teardown of the whole run tree
    if a.max_hist is not None:                          # arg -> env var the actuator/adapters read (both benchmarks)
        _hv = "BASELINE_MAX_TRAJ" if a.mode == "baseline" else "SECURE_MAX_HIST"
        os.environ[_hv] = str(a.max_hist)
        print(f"[parallel] --max-hist {a.max_hist} -> {_hv}", flush=True)
    if a.benchmark == "realbench":                      # realbench: no site containers -> partition tasks into run_suite lanes
        _run_realbench(a); return
    if a.site:
        from wa_site_tasks import site_task_ids
        a.tasks = site_task_ids(a.site)
        if not a.tasks:
            sys.exit(f"no tasks found for site {a.site!r} (is config_files/ generated? run setup/20)")
        print(f"[parallel] --site {a.site}: {len(a.tasks)} tasks", flush=True)

    k = max(1, min(a.workers, ww.MAX_WORKERS))
    bysite = _by_site(a.tasks)
    _DOCKER_HOST = os.environ.get("DOCKER_HOST", paths.DOCKER_HOST)
    os.makedirs(a.out_root, exist_ok=True)

    # Spawn + register + write the manifest under a signal block: a SIGTERM/SIGINT landing mid-spawn is queued and
    # delivered only AFTER every worker is in _POPENS and the manifest is on disk, so no worker can be orphaned or
    # left undiscoverable. Workers unblock signals in their own preexec (_spawn).
    manifest_workers = []
    _blocked = signal.pthread_sigmask(signal.SIG_BLOCK, _SIGSET)
    try:
        for site, stasks in bysite.items():
            parts = _partition(stasks, k)
            ports = ww.worker_ports(site, len(parts))
            for wi, (part, port) in enumerate(zip(parts, ports)):
                env = dict(os.environ, WA_SITE=site, WA_SITE_PORT=str(port), DOCKER_HOST=_DOCKER_HOST,
                           PLAYWRIGHT_BROWSERS_PATH=os.environ.get("PLAYWRIGHT_BROWSERS_PATH", paths.PLAYWRIGHT_BROWSERS_PATH))
                cmd = [PYEXE, os.path.join(HERE, "run_suite.py"),
                       "--mode", a.mode, "--model", a.model, "--out-root", a.out_root,
                       "--max-steps", str(a.max_steps), "--per-task-timeout", str(a.per_task_timeout),
                       "--seeds", str(a.seeds), "--tasks", *part]
                if a.skip_done:
                    cmd.append("--skip-done")
                logf = os.path.join(a.out_root, f"_parallel_{site}_p{port}.log")
                print(f"[parallel] {site} worker{wi} :{port}  {len(part)} tasks -> {part}", flush=True)
                p = _spawn(cmd, env, logf)
                _POPENS.append(p)
                manifest_workers.append({"site": site, "port": port, "pid": p.pid, "pgid": _pgid(p.pid),
                                         "start_time": _proc_start_time(p.pid), "log": logf})
                if wi > 0:                              # base container (worker 0) is kept; extras get cleaned up
                    _WORKER_CONTAINERS.append(f"{_BASE_NAME[site]}-p{port}")
        _MANIFEST_PATH = os.path.join(a.out_root, MANIFEST_NAME)
        _write_manifest(_MANIFEST_PATH, a.out_root, a.mode, a.model, manifest_workers, _WORKER_CONTAINERS)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, _blocked)  # unblock: a queued signal now fires against the full tree

    print(f"[parallel] launched {len(_POPENS)} worker suites across {len(bysite)} site(s), {k} workers/site "
          f"(stop cleanly with: python run_parallel.py --stop {a.out_root})", flush=True)

    for w, p in zip(manifest_workers, _POPENS):
        p.wait()
        print(f"[parallel] {w['site']}:{w['port']} exited rc={p.returncode}", flush=True)

    if not a.keep:
        _cleanup_containers(_WORKER_CONTAINERS, _DOCKER_HOST)
    _remove_manifest()
    print("[parallel] done", flush=True)


# base container names (mirror wa_reset SITE_RESET; kept here to avoid importing the heavy reset module)
_BASE_NAME = {"reddit": "webarena-verified-reddit", "shopping": "webarena-shopping",
              "shopping_admin": "webarena-shopping-admin", "gitlab": "webarena-verified-gitlab"}

if __name__ == "__main__":
    main()
