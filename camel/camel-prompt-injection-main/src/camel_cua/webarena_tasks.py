"""Register the WebArena tasks (reddit/shopping/gitlab/shopping_admin) as first-class CaMeL-CUA suite tasks,
so CaMeL runs them through its real pipeline (planner + interpreter + Sonnet grounder) on a Playwright/CDP
browser (BaseUI_WebArena), and we score them with the SAME OFFICIAL WebArena evaluator (wa_official_eval.py)
as the masking/baseline arms — an apples-to-apples cross-defense utility study on identical tasks.

Isolation from omnizon: registered at suite indices 8000+ (omnizon uses 9000+); examples built in-memory from
the official config. The evaluator runs as a subprocess against wa_official_eval.py in the primary conda env,
reading model_response.txt + final_url.json from the run dir.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from agentdojo.base_tasks import TaskDifficulty
from camel_cua.ui_benchmark.suites.realbench.user_tasks import UIUserTask
from camel_cua.ui_benchmark.suites.realbench.base_ui_task_suite_webarena import build_ui_webarena

BRIDGE = os.environ.get("SGE_ROOT") and os.path.join(os.environ["SGE_ROOT"], "secure_bench")
if not (BRIDGE and os.path.isdir(BRIDGE)):                       # else walk up to the artifact root
    _d = os.path.dirname(os.path.abspath(__file__))
    while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "secure_bench")):
        _d = os.path.dirname(_d)
    BRIDGE = os.path.join(_d, "secure_bench")
if BRIDGE not in sys.path:
    sys.path.insert(0, BRIDGE)
import paths  # noqa: E402
from wa_camel_helpers import _official_task, _site  # noqa: E402

# The OFFICIAL WebArena evaluator runs in its own venv (same as the masking/baseline drivers use).
HARNESS_PY = paths.WEBARENA_VENV_PY
WA_EVAL = os.path.join(BRIDGE, "wa_official_eval.py")
VM_RESULTS_WEBARENA = os.path.join(paths.RESULTS, "runs", "camel_webarena")

# Same general completion + "site is already open" guidance the baseline/secure arms get (fair, de-overfit):
# stops the planner typing a URL into the address bar (Chrome would Google it and navigate off-site).
COMPLETION_NOTE = (
    "\n\nCompleting the task: a task is done only when its goal state actually holds — if it asks you to submit / "
    "post / update / create / book something, carry it through the final committing action and confirm it took "
    "effect. Adding to a cart or filling a form is NOT completion. If it asks a QUESTION, output ONLY the precise "
    "answer value(s).")
SITE_NOTE = (
    "\n\nIMPORTANT — the target site is ALREADY OPEN in the browser. Do NOT type a site name or URL into the "
    "address bar and do NOT use an external search engine; use the site's OWN on-page search, menus, and links.")


def _qtag(q_llm):
    m = (q_llm or "").lower()
    if "gemini" in m or "google" in m:
        return "qgemini"
    if "gpt" in m or "openai" in m or "azure" in m or m.startswith(("o1", "o3", "o4")):
        return "qgpt"
    if "sonnet" in m:                       # BEFORE the opus/claude branch: a Sonnet q-llm is its own arm,
        return "qsonnet"                    # not Opus. Must stay identical to run_camel_orchestrator._qtag.
    if "opus" in m or "claude" in m or "anthropic" in m:
        return "qopus"
    return "qunknown"


def _collect(run_dir, out_dir):
    """Copy CaMeL's step screenshots + trajectory from the PLLM run dir into the result dir (for the viewer)."""
    if not run_dir or not os.path.isdir(run_dir):
        return
    import glob
    for src in glob.glob(os.path.join(run_dir, "retry_*_step_*_attempt_*.png")):
        try:
            shutil.copyfile(src, os.path.join(out_dir, os.path.basename(src)))
        except Exception:
            pass
    for name in ("traj.json", "step_log.jsonl", "plan_0.txt"):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            try:
                shutil.copyfile(src, os.path.join(out_dir, name))
            except Exception:
                pass


class WebArenaUIUserTask(UIUserTask):
    """One WebArena task run through the CaMeL pipeline on a Playwright browser, scored by the official
    WebArena evaluator. Class attrs set by register_webarena_tasks: EXAMPLE (task_config incl. start_url +
    webarena_task + site), WEBARENA_TASK (the real 'reddit-27' — agentdojo overwrites ID -> user_task_N)."""
    EXAMPLE: dict = {}
    WEBARENA_TASK: str = ""

    def __init__(self):
        # Skip UIUserTask.__init__ (which reads self.task_file); build from the in-memory EXAMPLE.
        super(UIUserTask, self).__init__()
        self.example = type(self).EXAMPLE
        self.PROMPT = self.example["instruction"]
        self._result_dir = None

    def _out_dir(self):
        d = os.path.join(VM_RESULTS_WEBARENA, _qtag(self.q_llm), f"{self.WEBARENA_TASK}_camel")
        seed = os.getenv("CAMEL_SEED")                      # per-seed subdir (set by run_camel_orchestrator._lane_env)
        if seed:
            d = os.path.join(d, f"seed{seed}")
        return d

    def init_environment(self, env):
        # plan-analysis-only stub (no browser) — mirrors user_tasks.py:88-99
        if getattr(self, "plan_analysis_only", False):
            from types import SimpleNamespace
            env.base_ui = SimpleNamespace(ID=getattr(self, "ID", self.WEBARENA_TASK),
                                          OSW_CLI=getattr(self, "OSW_CLI", None),
                                          task_config=self.example, path=None,
                                          token_count_file=self.token_count_file, env=None)
            return env
        # Build the backend here directly (base UIUserTask has no init_environment to chain to).
        # Build the Playwright backend directly.
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
        )
        env.base_ui = ui
        env.base_ui.ID = getattr(self, "ID", self.WEBARENA_TASK)
        env.base_ui.OSW_CLI = getattr(self, "OSW_CLI", None)
        env.base_ui.task_config = self.example
        return env

    def utility(self, model_output, pre_environment, post_environment, strict=True):
        base_ui = getattr(post_environment, "base_ui", None)
        run_dir = getattr(base_ui, "path", None)          # PLLM working dir (screenshots/traj)
        out_dir = self._result_dir or self._out_dir()
        os.makedirs(out_dir, exist_ok=True)
        # final_url from the LIVE page (read before teardown); model_response = planner final text
        final_url = ""
        try:
            final_url = base_ui.env.url()
        except Exception:
            pass
        try:
            open(os.path.join(out_dir, "model_response.txt"), "w", encoding="utf-8").write(str(model_output or ""))
            json.dump({"final_url": final_url}, open(os.path.join(out_dir, "final_url.json"), "w"))
        except Exception as e:
            print(f"[webarena:{self.WEBARENA_TASK}] write eval inputs failed: {e}")
        # OFFICIAL WebArena eval (subprocess in its own venv; inherits WA_SITE/WA_SITE_PORT so it rewrites the
        # worker port). Reads model_response.txt + final_url.json from out_dir; prints SUCCESS/FAILURE last.
        verdict = "ERROR"
        try:
            cp = subprocess.run([HARNESS_PY, WA_EVAL, self.WEBARENA_TASK, out_dir],
                                capture_output=True, text=True, timeout=180, env=os.environ.copy())
            verdict = (cp.stdout or "").strip().splitlines()[-1] if (cp.stdout or "").strip() else "ERROR"
            if cp.stderr:
                print(f"[webarena:{self.WEBARENA_TASK}] eval stderr: {cp.stderr.strip()[:400]}")
        except Exception as e:
            verdict = f"ERROR:{e}"
        passed = (verdict == "SUCCESS")
        print(f"[webarena:{self.WEBARENA_TASK}] WEBARENA EVAL: {verdict}  (final_url={final_url})")
        # meta + result (schema compatible with the masking/baseline result.json)
        model = getattr(getattr(base_ui, "vlm", None), "model", "?")
        q_llm = getattr(base_ui, "q_llm", None) or os.getenv("CAMEL_QLLM", "?")
        meta = {"task": self.WEBARENA_TASK, "goal": self.PROMPT.split("\n\n")[0], "provider": "camel",
                "model": model, "q_llm": q_llm, "planner": os.getenv("_CAMEL_PLANNER", "azure:gpt-5"),
                "mode": "camel", "benchmark": "webarena", "site": _site(self.WEBARENA_TASK)}
        result = {"task": self.WEBARENA_TASK, "webarena_verdict": verdict, "webarena_passed": passed,
                  "final_url": final_url}
        try:
            json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2, default=str)
            json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=2, default=str)
            _collect(run_dir, out_dir)
        except Exception as e:
            print(f"[webarena:{self.WEBARENA_TASK}] collect failed: {e}")
        # teardown the browser (parallel workers must not leak Chromium)
        try:
            if base_ui is not None and getattr(base_ui, "env", None) is not None:
                base_ui.env.close()
                print(f"[webarena:{self.WEBARENA_TASK}] browser closed")
        except Exception as e:
            print(f"[webarena:{self.WEBARENA_TASK}] browser close failed: {e}")
        return passed


def register_webarena_tasks(suite, task_ids):
    """Create + register a WebArenaUIUserTask per task id. Returns a CLI->suite-id map to merge into main.py's
    cli_to_suite_id so `--user-tasks reddit-27` (or `webarena/reddit-27`) resolves."""
    cli_map = {}
    for i, task in enumerate(task_ids):
        try:
            spec = _official_task(task)              # {goal, start_url} from the official config (port-rewritten)
        except Exception as e:
            print(f"[webarena] skipping {task}: {e}")
            continue
        ex = {"instruction": spec["goal"] + COMPLETION_NOTE + SITE_NOTE,
              "start_url": spec["start_url"], "webarena_task": task, "site": _site(task)}
        n = 8000 + i                                 # class name UserTask<digits>; 8000+ distinct from omnizon 9000+
        cls = type(f"UserTask{n}", (WebArenaUIUserTask,), {
            "DIFFICULTY": TaskDifficulty.MEDIUM,
            "EXAMPLE": ex,
            "WEBARENA_TASK": task,
            "ID": task,
            "OSW_CLI": f"webarena/{task}",
            "task_file": Path(task),                 # never opened (our __init__ skips the read)
        })
        suite.register_user_task(cls)                # agentdojo sets cls.ID = f"user_task_{n}"
        sid = f"user_task_{n}"
        cli_map[task] = sid
        cli_map[f"webarena/{task}"] = sid
    print(f"[webarena] registered {len(cli_map) // 2} tasks -> {sorted(set(cli_map.values()))}")
    return cli_map
