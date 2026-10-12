
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

"""Unified capped parallel orchestrator for CaMeL-CUA across BOTH benchmarks, with MULTI-ARM support.

CaMeL's entry is `main.py` (cyclopts), which runs a suite's tasks sequentially in ONE process. This
orchestrator gives it real parallelism with a rolling cap, reusing the proven WebArena harness pieces
(wa_workers for worker ports, wa_reset for pristine per-task container resets). It schedules a set of
(task, Q-LLM-arm) JOBS across per-site worker containers — so all 3 arms × 40 tasks = 120 jobs churn
through the container slots (arms are mixed on the same worker containers with a per-job reset between).
It fans out per-worker SUBPROCESSES of ITSELF (`--_lane`), each running its job list sequentially and
shelling to `main.py` per job under `xvfb-run` (headed Chromium + pyautogui both need a display).

  # WebArena, all 3 arms, 4 workers/site (=16-way, fits ~99GB RAM; 6/site OOMs on heavy gitlab):
  python run_camel_orchestrator.py --benchmark webarena --workers 4 --arms opus gemini gpt
  # realbench / omnizon:
  python run_camel_orchestrator.py --benchmark realbench --workers 5 --arms opus gemini gpt

Run in the activated primary conda env (playwright + wa_reset importable) so wa_reset.reset() works in-process.
CaMeL is UNMASKED (its own defense). Fair-eval config: planner azure:gpt-5, grounder claude-sonnet-4-5@20250929,
--system-defense-level 1 --model-defense-level 0 --max-attempts 1 (STOCK prompt = faithful).
"""
import argparse
import os
import paths
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wa_workers as ww  # noqa: E402

CAMEL = paths.CAMEL_ROOT
CAMEL_SRC = os.path.join(CAMEL, "src")          # import camel / camel_cua (added to the lane PYTHONPATH in _lane_env)
VM_RESULTS_WEBARENA = os.path.join(paths.RESULTS, "runs", "camel_webarena")
VM_RESULTS = os.path.join(paths.RESULTS, "runs", "camel_realbench")
OMNIZON_TASKS = ["omnizon-3", "omnizon-9", "omnizon-13", "omnizon-15", "omnizon-17",
                 "omnizon-24", "omnizon-25", "omnizon-28", "omnizon-29", "omnizon-30"]
_BASE_NAME = {"reddit": "webarena-verified-reddit", "shopping": "webarena-shopping",
              "shopping_admin": "webarena-shopping-admin", "gitlab": "webarena-verified-gitlab"}
DOCKER_HOST = os.environ.get("DOCKER_HOST", paths.DOCKER_HOST)

# The Q-LLM arms (grounder/planner are FIXED; only the quarantined Q-LLM varies).
ARM_QLLM = {"opus": "anthropic:claude-opus-4-8",
            "gemini": "google:gemini-3.1-pro-preview",
            "gpt": "azure:gpt-5",
            "sonnet": "anthropic:claude-sonnet-4-5@20250929"}


def _qtag(q_llm):
    m = (q_llm or "").lower()
    if "gemini" in m or "google" in m:
        return "qgemini"
    if "gpt" in m or "openai" in m or "azure" in m or m.startswith(("o1", "o3", "o4")):
        return "qgpt"
    if "sonnet" in m:                       # BEFORE the opus/claude branch: a Sonnet q-llm is its own arm,
        return "qsonnet"                    # not Opus. Must stay identical to webarena_tasks._qtag.
    if "opus" in m or "claude" in m or "anthropic" in m:
        return "qopus"
    return "qunknown"


def _partition(items, k):
    out = [[] for _ in range(k)]
    for i, it in enumerate(items):
        out[i % k].append(it)
    return [p for p in out if p]


def _result_path(benchmark, task, qllm, seed=None):
    if benchmark == "webarena":
        d = os.path.join(VM_RESULTS_WEBARENA, _qtag(qllm), f"{task}_camel")
    else:
        d = os.path.join(VM_RESULTS, f"{task}_v2_camel_{_qtag(qllm)}")
    if seed is not None:                                    # multi-seed -> per-seed subdir (mirror webarena_tasks._out_dir)
        d = os.path.join(d, f"seed{seed}")
    return os.path.join(d, "result.json")


def _done(benchmark, task, qllm, seed=None):
    import json
    try:
        j = json.load(open(_result_path(benchmark, task, qllm, seed)))
        return j.get("webarena_verdict") in ("SUCCESS", "FAILURE") or \
            j.get("corrected_verdict") in ("SUCCESS", "FAILURE")
    except Exception:
        return False


def _load_dotenv(env):
    """Inject .env (creds) into the subprocess env — no-build path has no `uv run --env-file` to do it,
    and main.py doesn't call load_dotenv itself. Already-set vars (e.g. exported by env.sh) win."""
    try:
        for line in open(paths.ENV_FILE):
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _camel_cmd(a, task, qllm):
    suites = "webarena" if a.benchmark == "webarena" else "realbench"
    return [
        "xvfb-run", "-a", "-s", "-screen 0 1920x1080x24",
        sys.executable, "main.py",          # the primary venv interpreter (no uv needed on the no-build path)
        "--model", a.model, "--reasoning-effort", a.reasoning_effort,
        "--path-to-vlm", a.path_to_vlm, "--q-llm", qllm,
        "--suites", suites, "--user-tasks", task,
        "--system-defense-level", str(a.system_defense_level),
        "--model-defense-level", str(a.model_defense_level),
        "--pllm-exposure-level", str(a.pllm_exposure_level),
        "--max-attempts", str(a.max_attempts),
    ]


def _lane_env(a, qllm, site=None, port=None, seed=None):
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    env["DOCKER_HOST"] = DOCKER_HOST
    env["PLAYWRIGHT_BROWSERS_PATH"] = env.get("PLAYWRIGHT_BROWSERS_PATH", paths.PLAYWRIGHT_BROWSERS_PATH)
    # all our code on the path (no-build): secure_bench + camel/src; keep any inherited PYTHONPATH
    _pp = [HERE, CAMEL_SRC]
    if env.get("PYTHONPATH"):
        _pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(_pp)
    _load_dotenv(env)                        # creds for main.py (replaces `uv run --env-file`)
    env["_CAMEL_PLANNER"] = a.model
    env["CAMEL_FORCE_RERUN"] = "1"
    env["CAMEL_NO_RESET_BETWEEN_ATTEMPTS"] = "1"
    env["CAMEL_STEP_LOG"] = "1"
    env["CAMEL_QLLM"] = qllm
    env["CAMEL_VLM"] = a.path_to_vlm
    if seed is not None:                                    # camel's WebArena/omnizon UserTask._out_dir reads CAMEL_SEED
        env["CAMEL_SEED"] = str(seed)
    # Per-SEED log dir. privileged_llm writes its run artifacts to <CAMEL_LOG_DIR>/.../<base_ui.ID>/none, and
    # base_ui.ID is "user_task_<n>" for EVERY seed (agentdojo rewrites the task ID), so with a shared
    # CAMEL_LOG_DIR concurrent seeds of the same task interleave step_log.jsonl / retry_*_step_*.png /
    # plan_0.txt / *_token_count.txt into ONE directory — and _collect() then copies that mixed dir into every
    # seed's result dir. Only the RESULT dir was seed-aware (webarena_tasks._out_dir reads CAMEL_SEED); scope
    # the LOG dir the same way so each seed's trajectory stays its own. seeds=1 -> unchanged path.
    _log_root = "logs_camel_webarena" if a.benchmark == "webarena" else "logs_realbench_omnizon"
    if seed is not None:
        _log_root = os.path.join(_log_root, f"seed{seed}")
    if a.benchmark == "webarena":
        env["WA_SITE"], env["WA_SITE_PORT"] = site, str(port)
    env["CAMEL_LOG_DIR"] = _log_root
    return env


# ------------------------------------------------------------------ lane (per-worker) ---------------
def run_lane(a):
    """Run a JOB list (each job = 'task|qllm') sequentially: (webarena) reset the worker container, then main.py."""
    site, port = a._lane_site, a._lane_port
    tag = f"{a.benchmark}/{site or 'rb'}:{port or '-'}"
    for spec in a._lane_jobs:
        # task|qllm[|seed]. The separator MUST NOT be '@': Vertex Claude ids carry an '@' version suffix
        # (anthropic:claude-sonnet-4-5@20250929), which an '@' split tore into a bogus qllm + seed.
        bits = spec.split("|")
        task, qllm = bits[0], bits[1]
        seed = int(bits[2]) if len(bits) > 2 else None
        stag = f" seed{seed}" if seed is not None else ""
        if a.skip_done and _done(a.benchmark, task, qllm, seed):
            print(f"[lane {tag}] skip-done {task}{stag} [{_qtag(qllm)}]", flush=True)
            continue
        if a.benchmark == "webarena":
            try:
                import wa_reset
                os.environ["WA_SITE"], os.environ["WA_SITE_PORT"] = site, str(port)
                rep = wa_reset.reset(site=site, log=lambda m: None)
                if not rep.get("ok"):
                    print(f"[lane {tag}] RESET FAILED {task}: {rep.get('error')} -> skip", flush=True)
                    continue
            except Exception as e:
                print(f"[lane {tag}] reset EXC {task}: {e} -> continuing", flush=True)
        cmd = _camel_cmd(a, task, qllm)
        env = _lane_env(a, qllm, site, port, seed)
        print(f"[lane {tag}] START {task}{stag} [{_qtag(qllm)}]", flush=True)
        t0 = time.time()
        try:
            p = subprocess.Popen(cmd, cwd=CAMEL, env=env, start_new_session=True)
            try:
                p.wait(timeout=a.per_task_timeout)
            except subprocess.TimeoutExpired:
                print(f"[lane {tag}] TIMEOUT {task} [{_qtag(qllm)}] -> kill", flush=True)
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
                p.wait()
        except Exception as e:
            print(f"[lane {tag}] EXC {task}: {e}", flush=True)
        print(f"[lane {tag}] END {task} [{_qtag(qllm)}] ({int(time.time()-t0)}s)", flush=True)


# ------------------------------------------------------------------ top-level orchestration ---------
def orchestrate(a):
    qllms = [ARM_QLLM.get(x, x) for x in a.arms]
    jobs = ([f"{t}|{q}|{s}" for q in qllms for t in a.tasks for s in range(a.seeds)] if a.seeds > 1
            else [f"{t}|{q}" for q in qllms for t in a.tasks])  # task x Q-LLM arm [x seed]  ('|' — see run_lane)
    procs, worker_containers = [], []
    print(f"[orch] {a.benchmark}: {len(jobs)} jobs ({len(a.tasks)} tasks x {len(qllms)} arms x {a.seeds} seeds) "
          f"= {[_qtag(q) for q in qllms]}", flush=True)
    if a.benchmark == "webarena":
        k = max(1, min(a.workers, ww.MAX_WORKERS))
        bysite = {}
        for spec in jobs:
            t = spec.split("|", 1)[0]
            bysite.setdefault(ww.site_of(t), []).append(spec)
        for site, sjobs in bysite.items():
            parts = _partition(sjobs, k)
            ports = ww.worker_ports(site, len(parts))
            for wi, (part, port) in enumerate(zip(parts, ports)):
                cmd = [sys.executable, os.path.join(HERE, "run_camel_orchestrator.py"), "--_lane",
                       "--benchmark", "webarena", "--_lane-site", site, "--_lane-port", str(port),
                       "--path-to-vlm", a.path_to_vlm, "--model", a.model,
                       "--per-task-timeout", str(a.per_task_timeout),
                       "--system-defense-level", str(a.system_defense_level),
                       "--model-defense-level", str(a.model_defense_level),
                       "--_lane-jobs", *part]
                if a.skip_done:
                    cmd.append("--skip-done")
                # WA_SITE/WA_SITE_PORT MUST be in the lane's env AT SPAWN: wa_reset.SITE_RESET is a module-level
                # dict (ww.worker_container is baked at `import wa_reset`), so a worker's container name is only
                # per-port if the env is set BEFORE the lane imports wa_reset. Mirrors run_parallel.py:65.
                lane_env = dict(os.environ, WA_SITE=site, WA_SITE_PORT=str(port), DOCKER_HOST=DOCKER_HOST,
                                PLAYWRIGHT_BROWSERS_PATH=os.environ.get("PLAYWRIGHT_BROWSERS_PATH",
                                                                        paths.PLAYWRIGHT_BROWSERS_PATH))
                print(f"[orch] webarena {site} worker{wi} :{port} {len(part)} jobs", flush=True)
                procs.append(subprocess.Popen(cmd, start_new_session=True, env=lane_env))
                if wi > 0:
                    worker_containers.append(f"{_BASE_NAME[site]}-p{port}")
            time.sleep(2)
    else:
        stop_evt = threading.Event()             # teardown signal; the .set() at the end is a harmless no-op (no background thread waits on it)
        for part in _partition(jobs, max(1, a.workers)):
            cmd = [sys.executable, os.path.join(HERE, "run_camel_orchestrator.py"), "--_lane",
                   "--benchmark", "realbench", "--path-to-vlm", a.path_to_vlm, "--model", a.model,
                   "--per-task-timeout", str(a.per_task_timeout),
                   "--system-defense-level", str(a.system_defense_level),
                   "--model-defense-level", str(a.model_defense_level), "--_lane-jobs", *part]
            if a.skip_done:
                cmd.append("--skip-done")
            procs.append(subprocess.Popen(cmd, start_new_session=True))
            time.sleep(8)

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
    if a.benchmark == "realbench":
        stop_evt.set()
    if a.benchmark == "webarena" and not a.keep:
        for c in worker_containers:
            subprocess.run(["docker", "rm", "-f", c], env=dict(os.environ, DOCKER_HOST=DOCKER_HOST), capture_output=True)
        if worker_containers:
            print(f"[orch] removed {len(worker_containers)} extra worker container(s)", flush=True)
    print("[orch] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["webarena", "realbench"], required=True)
    ap.add_argument("--workers", type=int, default=3, help="webarena: workers/site (<=MAX_WORKERS, tested=3=12-way); realbench: parallel (Chromium) lanes")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--site", choices=["reddit", "shopping", "shopping_admin", "gitlab"], default=None,
                    help="webarena: run ALL single-site tasks for this site (instead of --tasks / the default 40)")
    ap.add_argument("--seeds", type=int, default=1, help="run each (task,arm) across N independent seeds (repeats) -> pass@k")
    ap.add_argument("--arms", nargs="*", default=["opus", "gemini", "gpt"], help="Q-LLM arms (opus/gemini/gpt or a model id)")
    ap.add_argument("--path-to-vlm", default="claude-sonnet-4-5@20250929", help="the FIXED Sonnet grounder")
    ap.add_argument("--model", default="azure:gpt-5", help="the planner (P-LLM)")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--system-defense-level", type=int, default=1)
    ap.add_argument("--model-defense-level", type=int, default=0)
    ap.add_argument("--pllm-exposure-level", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=1)
    ap.add_argument("--per-task-timeout", type=float, default=2400)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--_lane", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_lane-site", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_lane-port", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_lane-jobs", nargs="*", default=[], help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.site:
        if a.benchmark != "webarena":
            sys.exit("--site is WebArena-only")
        from wa_site_tasks import site_task_ids
        a.tasks = site_task_ids(a.site)
        if not a.tasks:
            sys.exit(f"no tasks found for site {a.site!r} (run setup/20 to generate config_files)")
        print(f"[orch] --site {a.site}: {len(a.tasks)} tasks", flush=True)
    if a.tasks is None:
        if a.benchmark == "webarena":
            from wa_camel_helpers import TASK_IDS
            a.tasks = TASK_IDS
        else:
            a.tasks = OMNIZON_TASKS
    if a._lane:
        run_lane(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
