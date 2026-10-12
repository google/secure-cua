
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

"""Benchmark backends for the unified Playwright driver.

The WebArena and REAL/omnizon (realbench) benchmarks run through ONE browser pipeline (headed Chromium under
Xvfb, injected-page Smask, CDP-AXTree a11y, viewport Playwright actions). The ONLY per-benchmark differences —
where the site URL comes from, the per-task browser prepare, the scorer, and the output-dir/meta naming — live
here behind a small `Benchmark` object. Everything else (launch skeleton, observe, the agent loop, do_*) is
shared and lives in the driver.

- WebArenaBenchmark: local Docker site containers; goal/start_url from webarena_official config; per-task
  container reset (wa_reset); scored by the official WebArena harness (wa_official_eval.py).
- RealBenchmark: the public omnizon Vercel site (URL from `agisdk taskinfo`); no container/reset; the browser is
  launched with `--remote-debugging-port` so agisdk (CDP-generic) can `config`/`finisheval` over it; scored by
  agisdk + eval_fixed/corrected_eval.py. agisdk runs as a subprocess in the dedicated .agisdk_venv.
"""
import json
import os
import socket
import subprocess
import sys
import time

import paths

_HERE = os.path.dirname(os.path.abspath(__file__))
AGISDK_PY = paths.AGISDK_VENV_PY
AGISDK_EVAL = os.path.join(_HERE, "agisdk_eval.py")
CORRECTED_EVAL = os.path.join(_HERE, "eval_fixed", "corrected_eval.py")
WA_EVAL = os.path.join(_HERE, "wa_official_eval.py")


def _agisdk(*args, timeout=240):
    """Subprocess bridge into the .agisdk_venv; returns the last JSON line agisdk_eval.py prints."""
    r = subprocess.run([AGISDK_PY, AGISDK_EVAL, *args], capture_output=True, text=True, timeout=timeout)
    for line in reversed((r.stdout or "").strip().splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    return {"error": "agisdk: no JSON output", "stdout": (r.stdout or "")[-1500:], "stderr": (r.stderr or "")[-1500:]}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def model_tag(model):
    """Output-dir suffix per model family: claude='', gemini/gpt tagged."""
    ml = (model or "").lower()
    if ml.startswith("gemini"):
        return "_gemini"
    if ml.startswith(("gpt", "o1", "o3", "o4")):
        return "_gpt"
    return ""


class WebArenaBenchmark:
    """Local Docker WebArena sites + the official harness scorer. Wraps the driver's existing helpers verbatim."""
    name = "webarena"
    needs_reset = True
    result_verdict_key = "webarena_verdict"

    def __init__(self, version=None):
        pass

    def _d(self):                              # lazy import avoids a circular import at module load
        import run_secure_webarena as d
        return d

    def resolve(self, task):
        return self._d().resolve_task(task)

    def site_of(self, task):
        return self._d()._site(task)

    def env_label(self, task):
        return f"webarena-{self.site_of(task)}"

    def meta_extra(self, task, info=None):
        return {"a11y": "cdp-axtree", "coord_frame": "viewport-native"}

    def out_dir_name(self, task, model, mode):
        return f"{task}_{mode}"

    def pre_cookies(self, task, log):
        return self._d()._login_cookies(task, log)

    def extra_launch_args(self, task, ctx):
        return []

    def prepare(self, pg, cdp, ctx, task, run_id, log):
        pass                                   # WebArena: no per-task browser setup (cookies + goto already done)

    def reset(self, task, log=print):
        import wa_reset
        return wa_reset.reset(site=self.site_of(task), log=log)

    def score(self, task, out_dir, final_url, user_answer, feedback, ctx, run_id, log):
        try:
            cp = subprocess.run([paths.WEBARENA_VENV_PY, WA_EVAL, task, out_dir],
                                capture_output=True, text=True, timeout=120)
            verdict = (cp.stdout or "").strip().splitlines()[-1] if cp.stdout else "ERROR"
            if cp.stderr:
                log(f"[eval stderr] {cp.stderr.strip()[:400]}")
        except Exception as e:
            verdict = f"ERROR:{e}"
        return {"webarena_verdict": verdict, "webarena_passed": (verdict == "SUCCESS")}


class RealBenchmark:
    """The public omnizon Vercel site + agisdk-over-CDP scorer. No container, no reset."""
    name = "realbench"
    needs_reset = False
    result_verdict_key = "corrected_verdict"

    def __init__(self, version="v2"):
        self.version = version
        self._info = {}

    def resolve(self, task):
        info = _agisdk("taskinfo", task, self.version)
        if "goal" not in info:
            raise RuntimeError(f"agisdk taskinfo failed for {task} {self.version}: {info}")
        self._info = info
        return {"goal": info["goal"], "start_url": info["url"]}

    def site_of(self, task):
        return task.split("-", 1)[0]

    def env_label(self, task):
        return f"realbench-{self.site_of(task)}"

    def meta_extra(self, task, info=None):
        return {"version": self.version, "url": (self._info or {}).get("url"),
                "evals": (self._info or {}).get("evals"), "a11y": "cdp-axtree", "coord_frame": "viewport-native"}

    def out_dir_name(self, task, model, mode):
        return f"{task}_{self.version}{model_tag(model)}_{mode}"

    def pre_cookies(self, task, log):
        return []                              # anonymous public site

    def extra_launch_args(self, task, ctx):
        ctx["cdp_port"] = _free_port()
        ctx["cdp_http"] = f"http://127.0.0.1:{ctx['cdp_port']}"
        return [f"--remote-debugging-port={ctx['cdp_port']}"]

    def prepare(self, pg, cdp, ctx, task, run_id, log):
        cfg = _agisdk("config", ctx["cdp_http"], task, self.version, run_id)
        log(f"[agisdk config] {cfg}")
        if isinstance(cfg, dict) and cfg.get("error"):
            raise RuntimeError(f"agisdk config failed: {cfg}")

    def reset(self, task, log=print):
        return {"ok": True}                    # remote immutable site + agisdk run_id scoping -> nothing to reset

    def score(self, task, out_dir, final_url, user_answer, feedback, ctx, run_id, log):
        time.sleep(5)                          # extra settle before reading /finish state (the original secure loop used ~8s total)
        mr_file = os.path.join(out_dir, "model_response.txt")
        ev = _agisdk("finisheval", ctx["cdp_http"], task, self.version, run_id, mr_file)
        env_state = ev.pop("env_state", {}) if isinstance(ev, dict) else {}
        with open(os.path.join(out_dir, "env_state.json"), "w", encoding="utf-8") as f:
            json.dump(env_state, f, indent=2, default=str, ensure_ascii=False)
        out = {k: v for k, v in ev.items() if k != "info"}
        out["eval_info"] = ev.get("info")
        out["official_passed"] = out.get("passed")
        try:                                   # catalog-corrected verdict (omnizon), runs in THIS venv
            cp = subprocess.run([sys.executable, CORRECTED_EVAL, task, os.path.join(out_dir, "env_state.json")],
                                capture_output=True, text=True, timeout=30)
            cv = (cp.stdout or "").strip().splitlines()[-1] if cp.stdout else "ERROR"
        except Exception as e:
            cv = f"ERROR:{e}"
        out["corrected_verdict"] = cv
        out["corrected_passed"] = (cv == "SUCCESS")
        log(f"[agisdk] official_passed={out.get('official_passed')} corrected={cv} reward={out.get('reward')}")
        return out


_BENCHMARKS = {"webarena": WebArenaBenchmark, "realbench": RealBenchmark}


def get_benchmark(name, version=None):
    if name not in _BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r} (choose from {list(_BENCHMARKS)})")
    return _BENCHMARKS[name](version=(version or "v2"))
