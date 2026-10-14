"""Register the 10 omnizon REAL-v2 shopping tasks as first-class CaMeL-CUA suite tasks, so CaMeL runs
them through its real pipeline (planner + interpreter + ClaudeVLM) and we score them with the SAME agisdk
`finisheval` + `corrected_eval` as the baseline and secure arms — an apples-to-apples 3-way utility study.

Isolation: these tasks are registered at indices 9000+, and their examples are built in-memory from
`agisdk taskinfo` (no on-disk JSON), so they never collide with any other registered tasks. The agisdk
bridge (`/config` after reset, `finisheval` in `utility`) runs in real_bridge's own
`.agisdk_venv` via subprocess — agisdk is not importable in CaMeL's venv.
"""
import sys
import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from agentdojo.base_tasks import TaskDifficulty
from camel_cua.ui_benchmark.suites.realbench.user_tasks import UIUserTask
from camel_cua.ui_benchmark.suites.realbench.base_ui_task_suite_webarena import build_ui_webarena

# ---- real_bridge paths (the agisdk bridge + catalog-corrected eval live there) ----
BRIDGE = os.environ.get("SGE_ROOT") and os.path.join(os.environ["SGE_ROOT"], "secure_bench")
if not (BRIDGE and os.path.isdir(BRIDGE)):                       # else walk up to the artifact root
    _d = os.path.dirname(os.path.abspath(__file__))
    while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "secure_bench")):
        _d = os.path.dirname(_d)
    BRIDGE = os.path.join(_d, "secure_bench")
if BRIDGE not in sys.path:
    sys.path.insert(0, BRIDGE)
import paths  # noqa: E402
AGISDK_PY = paths.AGISDK_VENV_PY
AGISDK_EVAL = os.path.join(BRIDGE, "agisdk_eval.py")
CORRECTED_EVAL = os.path.join(BRIDGE, "eval_fixed", "corrected_eval.py")
VM_RESULTS = os.path.join(paths.RESULTS, "runs", "camel_realbench")

OMNIZON_TASKS = ["omnizon-3", "omnizon-9", "omnizon-13", "omnizon-15", "omnizon-17",
                 "omnizon-24", "omnizon-25", "omnizon-28", "omnizon-29", "omnizon-30"]
VERSION = "v2"

# Same GENERAL completion guidance the baseline + secure agents get (fair comparison; de-overfit).
COMPLETION_NOTE = (
    "\n\nCompleting the task: a task is done only when its goal state actually holds — not when you have "
    "merely set it up. If it asks you to buy / order / purchase / book / submit / send something, carry it "
    "through the final committing action (open the cart, proceed to checkout, and PLACE THE ORDER; or "
    "submit/send) and confirm it took effect. Do NOT finish (DONE) after only staging it — adding items to "
    "the cart or filling a form is NOT completion.")

# Factual environment context: the task site is ALREADY OPEN. This prevents the planner from typing the
# site name into the address bar (which Chrome resolves to a Google search, navigating OFF the task site
# to the real web — invisible to the evaluator). Baseline/secure never hit this because they click the
# on-page search directly; this note levels that navigation stumble without any capability boost.
SITE_NOTE = (
    "\n\nIMPORTANT — the target shopping site is ALREADY OPEN in the browser. Do NOT type a site name or URL "
    "into the address bar, and do NOT use an external search engine such as Google — doing so navigates AWAY "
    "from the task site and nothing you do there will count. Use the site's OWN on-page search box, category "
    "menus, and links, and stay on this site for the entire task.")


def _agisdk(*args, timeout=180):
    """Run real_bridge/agisdk_eval.py in its own venv; parse the last stdout line as JSON."""
    r = subprocess.run([AGISDK_PY, AGISDK_EVAL, *args], capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "").strip()
    line = out.splitlines()[-1] if out else ""
    try:
        return json.loads(line)
    except Exception:
        return {"error": "could not parse agisdk_eval output", "stdout": out[-800:], "stderr": (r.stderr or "")[-800:]}


def _corrected(task, env_state_path):
    try:
        r = subprocess.run([AGISDK_PY, CORRECTED_EVAL, task, env_state_path],
                           capture_output=True, text=True, timeout=60)
        return (r.stdout or "").strip().splitlines()[-1] if r.stdout else "ERROR"
    except Exception as e:
        return f"ERROR {e}"


def _build_example(task):
    info = _agisdk("taskinfo", task, VERSION)
    if "goal" not in info or "url" not in info:
        raise RuntimeError(f"agisdk taskinfo failed for {task}: {info}")
    return {"instruction": info["goal"] + COMPLETION_NOTE + SITE_NOTE,   # Playwright spec for build_ui_webarena(realbench=True)
            "start_url": info["url"], "webarena_task": task, "site": "omnizon"}


def _qtag(q_llm):
    """Tag the CaMeL result dir by the Q-LLM (the VARIED model) now that the ACTION VLM is FIXED at the
    paper's grounder (Claude Sonnet 4.5). New scheme `_camel_q<model>` — deliberately distinct from the
    earlier action-VLM-keyed dirs (_camel / _camel_gemini / _camel_gpt) so those (flawed) runs are kept
    for comparison rather than clobbered."""
    m = (q_llm or "").lower()
    if "gemini" in m or "google" in m:
        return "_qgemini"
    if "gpt" in m or "openai" in m or "azure" in m or m.startswith(("o1", "o3", "o4")):
        return "_qgpt"
    if "opus" in m or "claude" in m or "anthropic" in m or "sonnet" in m:
        return "_qopus"
    return "_qunknown"


def _variant_tag():
    """Optional dir suffix (naming only, no behavior change) to keep prompt-variant runs separate.
    e.g. CAMEL_VARIANT_TAG=stock -> _camel_q<model>_stock, so the FAITHFUL (stock-prompt) run does
    not clobber the ASSISTED (_camel_q<model>) results. Empty by default."""
    v = os.environ.get("CAMEL_VARIANT_TAG", "").strip()
    return f"_{v}" if v else ""


def _collect_camel_result(task, run_dir, meta, result, env_state):
    """Write vm_results/<task>_v2_camel[_<vlm>]/ so the existing viewer + scoreboard pick up the CaMeL arm.
    Also convert CaMeL's traj.json + screenshots into the baseline-style trajectory the viewer renders."""
    out = os.path.join(VM_RESULTS, f"{task}_{VERSION}_camel{_qtag(meta.get('q_llm'))}{_variant_tag()}")
    seed = os.getenv("CAMEL_SEED")                           # per-seed subdir (set by run_camel_orchestrator._lane_env)
    if seed:
        out = os.path.join(out, f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, default=str)
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=2, default=str)
    json.dump(env_state, open(os.path.join(out, "env_state.json"), "w"), indent=2, default=str)
    # copy gpt-5's plan artifacts here (task-named, real-bench-organized — not buried under user_task_900N)
    parent = os.path.dirname(run_dir)                         # .../user_task_900N ; run_dir is its 'none/'
    for src, dst in ((os.path.join(run_dir, "plan_0.txt"), "plan.txt"),
                     (os.path.join(parent, "model_output_0.txt"), "plan_annotated.txt"),
                     (os.path.join(run_dir, "highlighted_plan_0.html"), "plan_highlighted.html")):
        if os.path.exists(src):
            try:
                shutil.copyfile(src, os.path.join(out, dst))
            except Exception:
                pass
    # per-tool-call step log (opt-in via CAMEL_STEP_LOG) + the screenshots it references (raw retry_* names,
    # so the viewer can show the plan line that ran at each step with its result/reasoning + screenshot)
    _sl = os.path.join(run_dir, "step_log.jsonl")
    if os.path.exists(_sl):
        try:
            shutil.copyfile(_sl, os.path.join(out, "step_log.jsonl"))
            for _png in glob.glob(os.path.join(run_dir, "retry_*_step_*_attempt_*.png")):
                shutil.copyfile(_png, os.path.join(out, os.path.basename(_png)))
        except Exception as e:
            print(f"[omnizon:{task}] step_log copy failed: {e}")
    # convert CaMeL trajectory (retry/step/function/args/screenshot) -> viewer's baseline steps
    steps = []
    try:
        traj = json.load(open(os.path.join(run_dir, "traj.json")))
        for i, e in enumerate(traj, 1):
            src = os.path.join(run_dir, e.get("screenshot", ""))
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(out, f"step_{i - 1}.png"))
            steps.append({"step": i,
                          "model_response": f"{e.get('function')}({json.dumps(e.get('args', {}), default=str)[:400]})"
                                            f"  -> {str(e.get('result_status'))[:120]}",
                          "parsed_actions": [str(e.get("args", {}).get("cmd", e.get("function")))[:300]]})
        # a final screenshot for the last state
        last = max((int(''.join(filter(str.isdigit, e.get("screenshot", "0") or "0")) or 0) for e in traj), default=0)
    except Exception:
        pass
    json.dump({"task": task, "version": VERSION, "goal": meta.get("goal"), "model": meta.get("model"),
               "steps": steps, "num_steps": len(steps)},
              open(os.path.join(out, "trajectory.json"), "w"), indent=2, default=str)
    return out


class OmnizonUIUserTask(UIUserTask):
    """One omnizon task run through the CaMeL pipeline, scored by agisdk. Set as class attrs by
    register_omnizon_tasks: EXAMPLE (the task_config), OMNIZON_TASK (the real 'omnizon-N' — needed because
    agentdojo overwrites ID -> user_task_<n>)."""
    EXAMPLE: dict = {}
    OMNIZON_TASK: str = ""

    def __init__(self):
        # Skip UIUserTask.__init__ (which reads self.task_file); build from the in-memory EXAMPLE instead.
        super(UIUserTask, self).__init__()
        self.example = type(self).EXAMPLE
        self.PROMPT = self.example["instruction"]
        self.run_id = None
        self.cdp = None
        self._result_dir = None

    def _out_dir(self):
        d = os.path.join(VM_RESULTS, f"{self.OMNIZON_TASK}_{VERSION}_camel{_qtag(self.q_llm)}{_variant_tag()}")
        seed = os.getenv("CAMEL_SEED")                       # per-seed subdir (matches _collect_camel_result + orchestrator)
        if seed:
            d = os.path.join(d, f"seed{seed}")
        return d

    def init_environment(self, env):
        # plan-analysis-only stub (no browser) — mirrors user_tasks / webarena_tasks
        if getattr(self, "plan_analysis_only", False):
            from types import SimpleNamespace
            env.base_ui = SimpleNamespace(ID=getattr(self, "ID", self.OMNIZON_TASK),
                                          OSW_CLI=getattr(self, "OSW_CLI", None),
                                          task_config=self.example, path=None,
                                          token_count_file=self.token_count_file, env=None)
            return env
        # Build the Playwright backend here directly (base UIUserTask has no init_environment to chain to),
        # mirroring webarena_tasks.WebArenaUIUserTask.init_environment, with realbench=True.
        self._result_dir = self._out_dir()
        os.makedirs(self._result_dir, exist_ok=True)
        ui = build_ui_webarena(
            defense_level=self.model_defense_level,
            q_llm=self.q_llm,
            q_llm_second_check=self.q_llm_second_check,
            path_to_vlm=self.path_to_vlm,
            token_count_file=self.token_count_file,
            task_spec=self.example,
            out_dir=self._result_dir,
            realbench=True,
        )
        env.base_ui = ui
        env.base_ui.ID = getattr(self, "ID", self.OMNIZON_TASK)
        env.base_ui.OSW_CLI = getattr(self, "OSW_CLI", None)
        env.base_ui.task_config = self.example
        # prime the agisdk run over the Playwright browser's CDP endpoint (replaces de.vm_ip:chromium_port)
        try:
            self.cdp = ui.env.cdp_http
            self.run_id = "run-%d" % int(time.time())
            cfg = _agisdk("config", self.cdp, self.OMNIZON_TASK, VERSION, self.run_id)
            print(f"[omnizon:{self.OMNIZON_TASK}] agisdk config -> {cfg}")
        except Exception as e:
            print(f"[omnizon:{self.OMNIZON_TASK}] agisdk config FAILED: {e}")
        # gate on a11y readiness (CDP-AXTree TSV length) — config just reloaded the tab; wait before the blind
        # plan reads it.
        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                if ui.env.a11y_ready():
                    print(f"[omnizon:{self.OMNIZON_TASK}] a11y ready after {time.time()-t0:.1f}s")
                    break
            except Exception:
                pass
            time.sleep(1.5)
        return env

    def utility(self, model_output, pre_environment, post_environment, strict=True):
        base_ui = getattr(post_environment, "base_ui", None)
        run_dir = getattr(base_ui, "path", None)
        model = getattr(getattr(base_ui, "vlm", None), "model", "?")
        if not (self.cdp and self.run_id):
            print(f"[omnizon:{self.OMNIZON_TASK}] no agisdk run primed; cannot score")
            return False
        # write the agent's last output for finisheval, then score
        mr_file = os.path.join(run_dir or "/tmp", "model_response.txt")
        try:
            open(mr_file, "w").write(str(model_output or ""))
        except Exception:
            mr_file = "/tmp/omnizon_mr.txt"; open(mr_file, "w").write(str(model_output or ""))
        # finisheval reads the run's final Redux state over CDP. It intermittently comes back EMPTY (flaky/
        # dropped CDP on long runs) — an empty {} is a FAILED READ, not
        # "agent did nothing" (that yields a populated dict with empty cart). Retry so a real success is
        # never false-negatived by a flaky read. First sleep also settles a last-action order commit.
        ev, env_state = {}, {}
        for _try in range(3):
            time.sleep(8 if _try == 0 else 5)
            ev = _agisdk("finisheval", self.cdp, self.OMNIZON_TASK, VERSION, self.run_id, mr_file, timeout=240)
            env_state = ev.pop("env_state", {}) if isinstance(ev, dict) else {}
            if env_state:
                break
            print(f"[omnizon:{self.OMNIZON_TASK}] finisheval try {_try+1}: empty env_state (ev={str(ev)[:200]}) — retrying")
        official = bool(ev.get("passed"))
        es_path = os.path.join(run_dir or "/tmp", "env_state.json")
        json.dump(env_state, open(es_path, "w"), default=str)
        if self.OMNIZON_TASK.startswith("omnizon"):
            corrected = _corrected(self.OMNIZON_TASK, es_path)   # omnizon-only catalog-corrected eval
        else:
            corrected = "SUCCESS" if official else "FAILURE"     # other REAL sites: trust the official agisdk verdict
        corrected_pass = corrected == "SUCCESS"
        print(f"[realbench:{self.OMNIZON_TASK}] OFFICIAL passed={official} | verdict={corrected}")
        q_llm = getattr(base_ui, "q_llm", None) or os.getenv("CAMEL_QLLM", "?")
        meta = {"task": self.OMNIZON_TASK, "version": VERSION, "goal": self.PROMPT.split("\n\n")[0],
                "provider": "camel", "model": model, "q_llm": q_llm,
                "planner": os.getenv("_CAMEL_PLANNER", "azure:gpt-5"), "mode": "camel-cua"}
        result = {"task": self.OMNIZON_TASK, "official_passed": official, "corrected_verdict": corrected,
                  "corrected_passed": corrected_pass, "message": ev.get("message"), "eval_info": ev.get("info")}
        try:
            _collect_camel_result(self.OMNIZON_TASK, run_dir, meta, result, env_state)
        except Exception as e:
            print(f"[omnizon:{self.OMNIZON_TASK}] collect failed: {e}")
        # Close the Playwright/CDP backend so it doesn't leak. CaMeL/agentdojo never closes base_ui.env, so a
        # completed run otherwise leaves the headed-Chromium process running -> RAM/swap exhaustion across a
        # batch. Close it now that scoring + artifacts are done.
        try:
            de = getattr(base_ui, "env", None)
            if de is not None and hasattr(de, "close"):
                de.close()
                print(f"[omnizon:{self.OMNIZON_TASK}] env closed (backend closed)")
        except Exception as e:
            print(f"[omnizon:{self.OMNIZON_TASK}] env close failed: {e}")
        return corrected_pass          # the reliable metric (matches our 'corrected' column)


def register_omnizon_tasks(suite, task_ids=None):
    """Register REAL-benchmark tasks (ANY agisdk site — omnizon/udriver/staynb/...) as CaMeL suite tasks.
    Each is built site-agnostically from `agisdk taskinfo` (goal/url) and scored via agisdk `finisheval`; the
    catalog-corrected eval applies to omnizon ONLY (see utility()). `task_ids` defaults to the 10 omnizon ids
    (back-compat); pass any REAL ids to run other sites. Returns a CLI->suite-id map to merge into main.py's
    cli_to_suite_id so `--user-tasks <site>-N` (or `<site>/<site>-N`) resolves."""
    ids = task_ids or OMNIZON_TASKS
    cli_map = {}
    for i, task in enumerate(ids):
        try:
            ex = _build_example(task)
        except Exception as e:
            print(f"[realbench] skipping {task}: {e}")
            continue
        n = 9000 + i                    # class name must be UserTask<digits>; 9000+ avoids the <=369 real indices
        site = task.split("-")[0]       # 'omnizon-9' -> 'omnizon'; 'udriver-6' -> 'udriver'
        cls = type(f"UserTask{n}", (OmnizonUIUserTask,), {
            "DIFFICULTY": TaskDifficulty.MEDIUM,
            "EXAMPLE": ex,
            "OMNIZON_TASK": task,       # (name kept for back-compat; holds ANY REAL task id)
            "ID": task,
            "OSW_CLI": f"{site}/{task}",
            "task_file": Path(task),      # never opened (our __init__ skips the read); only .stem is used
        })
        suite.register_user_task(cls)   # agentdojo sets cls.ID = f"user_task_{n}"
        sid = f"user_task_{n}"
        cli_map[task] = sid
        cli_map[f"{site}/{task}"] = sid
    print(f"[realbench] registered {len(ids)} tasks -> {sorted(set(cli_map.values()))}")
    return cli_map
