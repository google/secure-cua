
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

"""Batch-run a suite of tasks sequentially, one driver subprocess per task, and aggregate the verdicts.

Parameterized by `--benchmark {webarena,realbench}` via the shared `Benchmark` object (benchmarks.py) — the
same object the drivers use, so the output-dir name, verdict key, task resolution and reset all match:
 - WebArena: each task mutates shared state (votes / reply / posts / edit / subscribe / create-forum) and its
   scorer reads the LIVE database, so before each task we reset the site container to its pristine baked DB
   (`docker stop/remove` + `docker run` fresh; ~12s on the verified image), then score with the official harness.
 - realbench: the site is a public hosted URL scored by agisdk over CDP, so there is no reset (--no-reset is
   implied); each task's driver allocates its own debug port.

Each driver run is its own subprocess under its own Xvfb display (crash/hang isolation + a fresh Playwright per
task), writing `<out-root>/<out_dir_name>/result.json`, which this collects into the suite aggregate. Resumable
via --skip-done. Multi-seed (--seeds K) repeats each task into `seed<k>/` subdirs for pass@k.

Run:
  python run_suite.py --tasks reddit-714 reddit-719                     # WebArena (default)
  python run_suite.py --include-404                                     # + the reddit-404 smoke task
  python run_suite.py --benchmark realbench --tasks omnizon-3 omnizon-9 # realbench (no reset)
  python run_suite.py --dry-run                                         # reset + plumbing only, SKIP the driver
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import paths
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_secure_webarena import OUT_ROOT_DEFAULT, MODEL  # noqa: E402
from benchmarks import get_benchmark  # noqa: E402  (WebArena | realbench seam — same object the drivers use)

PYEXE = sys.executable  # the venv python running this script
# The 10-task representative suite (reddit-404 is the separate smoke task, added via --include-404).
SUITE_10 = ["reddit-29", "reddit-719", "reddit-725", "reddit-714", "reddit-409",
            "reddit-605", "reddit-635", "reddit-731", "reddit-595", "reddit-580"]


def _op_mix(out_dir):
    """{query,compute,act,return,...} counts from op_trace.json (best-effort)."""
    mix = {}
    try:
        trace = json.load(open(os.path.join(out_dir, "op_trace.json"), encoding="utf-8"))
        for e in trace:
            k = (e.get("op") or {}).get("op", "?")
            mix[k] = mix.get(k, 0) + 1
    except Exception:
        pass
    return mix


_DRIVERS = {"secure": "run_secure_webarena.py", "baseline": "run_baseline_webarena.py",
            "noquery": "run_noquery_webarena.py"}

_CUR_DRIVER = None                       # driver Popen currently running (its OWN session); for graceful shutdown
_DRIVERPG_FILE = None                    # sidecar publishing the current driver's pgid so an external --stop can reap it


def _child_unblock_signals():
    """preexec in the driver child: undo the parent's brief SIGTERM/SIGINT block (below) so the driver subtree
    can be signalled normally — Playwright's own chromium teardown relies on delivering SIGTERM."""
    try:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    except Exception:
        pass


def _publish_driver_pg(driver_pid):
    """Write the current driver's process-group id to the sidecar (empty when idle). Lets `run_parallel.py
    --stop` SIGKILL an orphaned driver session even if THIS worker was killed uncooperatively (kill -9 / OOM)."""
    if not _DRIVERPG_FILE:
        return
    try:
        with open(_DRIVERPG_FILE, "w") as f:
            f.write("" if driver_pid is None else str(os.getpgid(driver_pid)))
    except Exception:
        pass


def _install_signal_handlers():
    """On SIGTERM/SIGINT (parent launcher tearing the run down, or a manual stop): kill this worker's WHOLE
    subtree so nothing orphans — the in-flight DRIVER session (xvfb-run + Xvfb + chromium, its own session) AND
    any reset-phase children (auto_login's chromium, docker execs) that live in OUR OWN process group — then
    exit. Per-task aggregates are already flushed, so partial results survive."""
    def _shutdown(signum, _frame):
        for _s in (signal.SIGTERM, signal.SIGINT):           # ignore re-entry: we're about to signal our OWN group
            try:
                signal.signal(_s, signal.SIG_IGN)
            except Exception:
                pass
        p = _CUR_DRIVER
        try:
            if p is not None and p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)     # the driver's separate session
        except Exception:
            pass
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)              # our group: reset-phase children (auto_login, docker exec)
        except Exception:
            pass
        time.sleep(1.0)
        try:
            if p is not None and p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)              # SIGKILL our group -> reset children AND this worker
        except Exception:
            pass
        os._exit(128 + (signum or 15))
    for _s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_s, _shutdown)
        except Exception:
            pass


def _run_driver(task, max_steps, model, per_task_timeout, screen, log, out_root,
                driver="run_secure_webarena.py", idle_timeout=300.0, seed=None,
                benchmark="webarena", version=None):
    """Run the chosen driver under Xvfb as an isolated subprocess. Returns (returncode, killed)."""
    cmd = ["xvfb-run", "-a", "-s", f"-screen 0 {screen}", PYEXE,
           os.path.join(HERE, driver),
           "--task", task, "--max-steps", str(max_steps), "--model", model, "--out-root", out_root,
           "--benchmark", benchmark]
    if version:
        cmd += ["--version", version]
    if seed is not None:                                     # multi-seed: driver writes <out-root>/<task>_.../seed<N>/
        cmd += ["--seed", str(seed)]
    log(f"[suite] $ {' '.join(cmd)}")
    env = dict(os.environ)
    env.setdefault("DOCKER_HOST", paths.DOCKER_HOST)
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", paths.PLAYWRIGHT_BROWSERS_PATH)
    global _CUR_DRIVER
    _blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})   # atomic spawn+publish
    try:
        proc = subprocess.Popen(cmd, env=env, start_new_session=True, preexec_fn=_child_unblock_signals,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _CUR_DRIVER = proc                                    # publish BEFORE unblocking -> handler always sees the driver
        _publish_driver_pg(proc.pid)                         # sidecar so an external --stop can reap this driver session
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, _blocked)  # unblock: a signal queued during the block fires now, driver known
    # TWO-TIER WATCHDOG (both kill the whole process group -> closing the pipe writer ends the readline loop below).
    #  `for line in proc.stdout` BLOCKS on a silent driver, so the checks MUST live in a separate thread.
    #  (1) IDLE cap (idle_timeout): no stdout line for that long. A healthy driver logs every turn (act/observe/
    #      feedback) and even LLM-retry heartbeats (stderr is merged in), so prolonged silence == a genuine hang.
    #      This catches EVERY hang class generically -- a stalled Vertex/LLM call, or a Playwright `page.evaluate`
    #      blocking indefinitely on a pathological page (gitlab's Monaco file-editor SPA, a stuck download/dialog;
    #      `evaluate` is the one Playwright call with no timeout). Bounds waste to ~idle_timeout, not per_task_timeout.
    #  (2) HARD cap (per_task_timeout): absolute wall-time ceiling for a task that keeps chattering but never ends.
    st = {"killed": False, "last": time.monotonic(), "start": time.monotonic(), "done": False}
    lock = threading.Lock()

    def _kill(reason):
        with lock:
            if st["done"]:
                return
            st["killed"] = True
        log(f"[suite] WATCHDOG: {task} {reason} -> kill process group")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    def _watch():
        while True:
            time.sleep(5)
            with lock:
                if st["done"]:
                    return
                now = time.monotonic()
                idle, elapsed = now - st["last"], now - st["start"]
            if elapsed > per_task_timeout:
                _kill(f"exceeded hard cap {per_task_timeout:.0f}s"); return
            if idle > idle_timeout:
                _kill(f"idle {idle:.0f}s > {idle_timeout:.0f}s (driver silent/hung)"); return

    wt = threading.Thread(target=_watch, daemon=True)
    wt.start()
    try:
        for line in proc.stdout:  # stream the driver's log through so a background suite stays observable
            with lock:
                st["last"] = time.monotonic()
            sys.stdout.write("    | " + line)
            sys.stdout.flush()
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
    finally:
        with lock:
            st["done"] = True
        _CUR_DRIVER = None
        _publish_driver_pg(None)                             # driver gone -> clear the sidecar
    return proc.returncode, st["killed"]


def run_suite(tasks, max_steps, model, per_task_timeout, screen, do_reset, skip_done, out_root, mode="secure",
              idle_timeout=300.0, seeds=1, benchmark="webarena", version=None):
    bm = get_benchmark(benchmark, version)               # same object the drivers use (out_dir/verdict/resolve/reset)
    do_reset = do_reset and bm.needs_reset               # realbench: remote immutable site -> never reset
    vkey = bm.result_verdict_key                         # "webarena_verdict" | "corrected_verdict"
    driver = _DRIVERS[mode]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_dir = os.path.join(out_root, "_suite")
    os.makedirs(suite_dir, exist_ok=True)
    global _DRIVERPG_FILE
    _DRIVERPG_FILE = os.path.join(out_root, f"_driverpg_{os.getpid()}")   # sidecar: this worker's current driver pgid
    logpath = os.path.join(suite_dir, f"suite_{mode}_{stamp}.log")
    logf = open(logpath, "w", encoding="utf-8")

    def log(m):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"SUITE start  mode={mode}  driver={driver}  tasks={tasks}  model={model}  max_steps={max_steps}  "
        f"reset={do_reset}  per_task_timeout={per_task_timeout}s")
    units = [(t, s) for t in tasks for s in range(seeds)]    # (task, seed) repeats; seeds==1 -> legacy single run
    results = []
    for i, (task, seed) in enumerate(units, 1):
        seed_arg = seed if seeds > 1 else None               # seeds==1 -> flat dir, no --seed (back-compat)
        base = os.path.join(out_root, bm.out_dir_name(task, model, mode))   # MUST mirror the driver's out_dir
        out_dir = os.path.join(base, f"seed{seed}") if seed_arg is not None else base
        rec = {"task": task, "seed": seed, "reset": None, "verdict": None, "turns": None,
               "op_mix": None, "seconds": None, "error": None}
        t0 = time.time()
        try:
            _goal = bm.resolve(task)["goal"][:80]
        except Exception:
            _goal = task
        log(f"===== [{i}/{len(units)}] {task} seed{seed} :: {_goal} =====")

        if skip_done:
            try:
                prev = json.load(open(os.path.join(out_dir, "result.json"), encoding="utf-8"))
                if prev.get(vkey) in ("SUCCESS", "FAILURE"):
                    log(f"[suite] --skip-done: {task} already has verdict={prev[vkey]} -> skip")
                    rec.update(verdict=prev[vkey], turns=prev.get("turns"),
                               op_mix=_op_mix(out_dir), skipped=True)
                    results.append(rec); continue
            except Exception:
                pass

        if do_reset:
            rep = bm.reset(task, log=log)                            # per-benchmark reset (webarena: container swap)
            rec["reset"] = {"ok": rep.get("ok"), "seconds": round(rep.get("seconds", 0), 1)}
            if not rep.get("ok"):
                rec["error"] = f"reset failed: {rep.get('error')}"
                log(f"[suite] RESET FAILED for {task}: {rep.get('error')} -> skipping task")
                results.append(rec); continue

        if args_dry:
            log(f"[suite] --dry-run: skipping driver for {task}")
            rec["verdict"] = "DRYRUN"; rec["seconds"] = round(time.time() - t0, 1)
            results.append(rec); continue

        rc, killed = _run_driver(task, max_steps, model, per_task_timeout, screen, log, out_root, driver=driver,
                                 idle_timeout=idle_timeout, seed=seed_arg, benchmark=benchmark, version=version)
        rec["seconds"] = round(time.time() - t0, 1)
        try:
            res = json.load(open(os.path.join(out_dir, "result.json"), encoding="utf-8"))
            rec["verdict"] = res.get(vkey)
            rec["turns"] = res.get("turns")
            rec["final_url"] = res.get("final_url")
            rec["metrics"] = res.get("metrics")          # per-task token/time cost (actuator/shadow|baseline)
            if res.get("error"):
                rec["error"] = res["error"]
        except Exception as e:
            rec["error"] = f"no result.json (rc={rc}, killed={killed}): {e}"
            rec["verdict"] = "ERROR"
        if killed:
            rec["error"] = (rec["error"] or "") + " [TIMEOUT-KILLED]"
            rec["verdict"] = rec["verdict"] or "TIMEOUT"
        rec["op_mix"] = _op_mix(out_dir)
        log(f"[suite] {task} -> verdict={rec['verdict']} turns={rec['turns']} "
            f"op_mix={rec['op_mix']} ({rec['seconds']}s)")
        results.append(rec)

        # write the aggregate after EVERY task (so a background/interrupted suite still has partial results)
        _write_aggregate(suite_dir, stamp, tasks, results, model, max_steps, do_reset, log, mode)

    _write_aggregate(suite_dir, stamp, tasks, results, model, max_steps, do_reset, log, mode)
    _print_table(results, log)
    logf.close()
    try:                                                     # clean exit -> remove the driver-pgid sidecar
        if _DRIVERPG_FILE and os.path.exists(_DRIVERPG_FILE):
            os.remove(_DRIVERPG_FILE)
    except Exception:
        pass
    return results


def _write_aggregate(suite_dir, stamp, tasks, results, model, max_steps, do_reset, log, mode="secure"):
    passed = sum(1 for r in results if r.get("verdict") == "SUCCESS")
    done = sum(1 for r in results if r.get("verdict") in ("SUCCESS", "FAILURE"))
    # per-task (n samples, c successes) across seeds -> unbiased pass@k (see passk.py)
    from collections import defaultdict
    from passk import pass_at_k
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r.get("verdict") == "SUCCESS")
    task_nc = {t: (len(v), sum(v)) for t, v in by_task.items()}
    kmax = max((n for n, _ in task_nc.values()), default=1)
    passk = {}
    for k in range(1, kmax + 1):
        vals = [pass_at_k(n, c, k) for n, c in task_nc.values()]
        vals = [x for x in vals if x is not None]
        passk[k] = (sum(vals) / len(vals)) if vals else None
    agg = {"stamp": stamp, "mode": mode, "model": model, "max_steps": max_steps, "reset_per_task": do_reset,
           "n_tasks": len(tasks), "n_done": done, "n_passed": passed,
           "pass_rate": (passed / done) if done else None,
           "pass_at_k": passk, "per_task": {t: {"n": n, "c": c} for t, (n, c) in task_nc.items()},
           "results": results}
    for name in (f"suite_{mode}_{stamp}.json", f"suite_{mode}_latest.json"):
        with open(os.path.join(suite_dir, name), "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, default=str)


def _print_table(results, log):
    log("================ SUITE SUMMARY ================")
    log(f"{'task':<14}{'verdict':<10}{'turns':<7}{'secs':<8}op_mix / error")
    for r in results:
        mix = r.get("op_mix") or {}
        mixs = " ".join(f"{k}:{v}" for k, v in mix.items()) if mix else ""
        tail = mixs + (f"  ERR:{r['error'][:60]}" if r.get("error") else "")
        log(f"{r['task']:<14}{str(r.get('verdict')):<10}{str(r.get('turns') or ''):<7}"
            f"{str(r.get('seconds') or ''):<8}{tail}")
    passed = sum(1 for r in results if r.get("verdict") == "SUCCESS")
    done = sum(1 for r in results if r.get("verdict") in ("SUCCESS", "FAILURE"))
    log(f"PASSED {passed}/{done} scored ({len(results)} attempted)")


args_dry = False


def main():
    global args_dry
    _install_signal_handlers()                               # graceful teardown of the in-flight driver on SIGTERM/SIGINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=None, help="task ids (default: the 10-task suite)")
    ap.add_argument("--include-404", action="store_true", help="prepend reddit-404 (smoke task)")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--per-task-timeout", type=float, default=1800, help="seconds before killing a run")
    ap.add_argument("--screen", default="1920x1200x24", help="Xvfb screen geometry")
    ap.add_argument("--no-reset", action="store_true", help="do NOT reset the container before each task")
    ap.add_argument("--skip-done", action="store_true", help="skip tasks that already have a SUCCESS/FAILURE result.json")
    ap.add_argument("--dry-run", action="store_true", help="reset + plumbing only; skip the (Opus) driver")
    ap.add_argument("--idle-timeout", type=float, default=300.0,
                    help="kill a driver that emits NO output for this long (catches any hang class); < per-task-timeout")
    ap.add_argument("--mode", default="secure", choices=list(_DRIVERS),
                    help="secure = masked/typed-contract loop; baseline = no-defense actuator loop (run_baseline_webarena)")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--seeds", type=int, default=1, help="run each task across N independent seeds (repeats) -> pass@k")
    ap.add_argument("--benchmark", default="webarena", choices=["webarena", "realbench"])
    ap.add_argument("--version", default=None, help="realbench task version (default v2); ignored for webarena")
    a = ap.parse_args()
    args_dry = a.dry_run
    bm = get_benchmark(a.benchmark, a.version)

    if a.benchmark == "realbench" and not a.tasks:
        sys.exit("--benchmark realbench requires explicit --tasks (e.g. --tasks omnizon-3 omnizon-9); no default suite")
    tasks = a.tasks if a.tasks else list(SUITE_10)           # SUITE_10 is the webarena reddit default
    if a.include_404 and "reddit-404" not in tasks:
        tasks = ["reddit-404"] + tasks
    unknown = []                                             # accept ANY id the benchmark can resolve
    for t in tasks:
        try:
            bm.resolve(t)
        except Exception:
            unknown.append(t)
    if unknown:
        print(f"unresolvable tasks: {unknown}", file=sys.stderr); sys.exit(2)

    run_suite(tasks, a.max_steps, a.model, a.per_task_timeout, a.screen,
              do_reset=not a.no_reset, skip_done=a.skip_done, out_root=a.out_root, mode=a.mode,
              idle_timeout=a.idle_timeout, seeds=a.seeds, benchmark=a.benchmark, version=a.version)


if __name__ == "__main__":
    main()
