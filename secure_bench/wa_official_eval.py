
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

"""Glue adapter — score a run with WebArena's OFFICIAL `evaluation_harness` (UNMODIFIED).

Runs in the PRIMARY conda env under the same interpreter as the spawning runner (the openai.error shim below
lets the unmodified harness import under a modern openai); invoked as a subprocess by the runners with
the SAME CLI contract as the previous local scorer (now retired):

    python wa_official_eval.py <task e.g. reddit-67> <run_dir>   -> prints SUCCESS/FAILURE (last line)

This file authors NO scoring logic. It imports the official `evaluator_router` / `EvaluatorComb` /
`create_stop_action` from `webarena_official/` and only supplies what the harness consumes:
  - the resolved per-task config `config_files/<task_id>.json` (task_id == index; verified),
  - a Playwright `Page` opened as the eval user (site `storage_state`),
  - a `trajectory` whose last action carries the agent's answer (`model_response.txt`),
  - a `CDPSession` client.
The official `EvaluatorComb` returns 1.0 (pass) / 0.0 (fail).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # real_bridge (for wa_workers)
import wa_workers as ww                                                    # same-site parallelism (base run => identity)
import paths

WEBARENA = paths.WEBARENA_OFFICIAL
sys.path.insert(0, WEBARENA)
CONFIG_DIR = os.path.join(WEBARENA, "config_files")


def _shim_openai_error():
    """WebArena's `llms/providers/openai_utils.py` does `import openai.error` at module load (the pre-1.0 API).
    Under openai>=1.0 that submodule is gone, so the harness won't even import. We never make those calls (the
    fuzzy judge's transport is monkeypatched to Azure), so we only need the import + the `openai.error.X`
    references to resolve — inject a stub module with the exception names the harness touches. This lets the
    official harness run under a MODERN openai, so the eval can share the agents' venv (no openai<1 pin)."""
    import openai
    if getattr(openai, "error", None) is not None:
        return                                                   # pre-1.0 openai already provides it
    import types
    err = types.ModuleType("openai.error")

    class OpenAIError(Exception):
        pass
    for _n in ("OpenAIError", "RateLimitError", "APIError", "APIConnectionError", "Timeout",
               "ServiceUnavailableError", "InvalidRequestError", "AuthenticationError", "TryAgain"):
        setattr(err, _n, OpenAIError if _n == "OpenAIError" else type(_n, (OpenAIError,), {}))
    openai.error = err
    sys.modules["openai.error"] = err


_shim_openai_error()          # must run before the harness (openai_utils) is imported

# Site -> eval-user storage_state (cookies). All 4 sites wired for the 2026-07-24 re-run campaign.
STORAGE_STATE = dict(paths.WA_STATE)

# The resolved config's URLs must be reachable as env for `func:` helpers / the judge; harmless for the
# string_match/program_html tasks we currently score. Defaults match generate_test_data.py.
_URL_DEFAULTS = {"REDDIT": "http://localhost:9999", "SHOPPING": "http://localhost:7770",
                 "SHOPPING_ADMIN": "http://localhost:7780/admin", "GITLAB": "http://localhost:8023",
                 "WIKIPEDIA": "http://localhost:8888", "MAP": "http://localhost:3000",
                 "HOMEPAGE": "http://localhost:4399"}
for k, v in _URL_DEFAULTS.items():
    os.environ.setdefault(k, v)


def _patch_fuzzy_judge():
    """Redirect the OFFICIAL fuzzy/ua LLM judge's TRANSPORT to a working Azure deployment WITHOUT modifying the
    harness. The harness hardcodes model 'gpt-4-1106-preview' (inaccessible here) and any judge exception is
    swallowed into FAILURE, so EVERY fuzzy_match task false-fails regardless of the answer. We monkeypatch ONLY
    the transport fn the judge calls (helper_functions.generate_from_openai_chat_completion); the judge PROMPT +
    parsing stay official/unmodified. Plain HTTP (the harness venv has openai<1, no AzureOpenAI class).
    Deployment gpt-4.1 (accepts the judge's classic temperature/max_tokens); override via FUZZY_JUDGE_DEPLOYMENT."""
    ep = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    dep = os.getenv("FUZZY_JUDGE_DEPLOYMENT", "gpt-4.1")
    if not (ep and key):
        print("[wa_official_eval] WARN no Azure creds in env — fuzzy judge left on the (broken) official transport",
              file=sys.stderr)
        return
    import requests
    import evaluation_harness.helper_functions as hf

    def _judge(messages, model, temperature, max_tokens, top_p, context_length, stop_token=None):
        url = f"{ep}/openai/deployments/{dep}/chat/completions?api-version={ver}"
        body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        r = requests.post(url, headers={"api-key": key, "Content-Type": "application/json"}, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    hf.generate_from_openai_chat_completion = _judge
    print(f"[wa_official_eval] fuzzy/ua judge -> Azure deployment {dep!r}", file=sys.stderr)


def _site(task):      # "reddit-67" -> "reddit"; "shopping_admin-243" -> "shopping_admin"
    return task.rsplit("-", 1)[0]


def _task_id(task):   # "reddit-67" -> 67
    return int(task.rsplit("-", 1)[1])


def _resolve_config(task_id, site):
    """Return a config path pointing at the LIVE container. Configs use the site's CANONICAL port; for a
    same-site-parallel WORKER (WA_SITE/WA_SITE_PORT), rewrite that site's canonical port -> the worker's
    port into a temp copy so the OFFICIAL evaluator scores against the worker's own container. Pure URL
    resolution (what generate_test_data already does) — no eval content is touched. Base run -> original."""
    cfg = os.path.join(CONFIG_DIR, f"{task_id}.json")
    if not ww.is_worker(site) or not os.path.exists(cfg):
        return cfg
    d, p = ww.SITE_DEFAULT_PORT[site], ww.worker_port(site)
    raw = open(cfg, encoding="utf-8").read().replace(f"localhost:{d}", f"localhost:{p}")
    fd, tmp = tempfile.mkstemp(suffix=f"_{task_id}.json"); os.close(fd)
    open(tmp, "w", encoding="utf-8").write(raw)
    return tmp


def evaluate(task, run_dir):
    from evaluation_harness import evaluator_router          # OFFICIAL, unmodified
    from browser_env.actions import create_stop_action       # OFFICIAL, unmodified
    from playwright.sync_api import sync_playwright

    _patch_fuzzy_judge()                                     # route the fuzzy/ua judge to a working Azure model

    tid = _task_id(task)
    cfg = _resolve_config(tid, _site(task))
    if not os.path.exists(cfg):
        print(f"[wa_official_eval] no config for {task} (expected {CONFIG_DIR}/{tid}.json — run "
              f"scripts/generate_test_data.py)", file=sys.stderr)
        return "FAILURE"
    conf = json.load(open(cfg, encoding="utf-8"))
    if conf.get("task_id") != tid:
        print(f"[wa_official_eval] WARN config task_id {conf.get('task_id')} != {tid}", file=sys.stderr)

    try:
        answer = open(os.path.join(run_dir, "model_response.txt"), encoding="utf-8").read()
    except Exception:
        answer = ""

    state = STORAGE_STATE.get(_site(task))
    if state:
        state = ww.worker_state_path(_site(task), state)                 # eval user's per-worker auth (base => unchanged)
    if state and not os.path.exists(state):
        print(f"[wa_official_eval] WARN storage_state missing for {_site(task)}: {state}", file=sys.stderr)
        state = None

    trajectory = [create_stop_action(answer)]                # last action carries the answer (StringEvaluator)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        ctx = b.new_context(storage_state=state) if state else b.new_context()
        page = ctx.new_page()
        client = ctx.new_cdp_session(page)
        # Put the eval page on the agent's FINAL url so the official URLEvaluator (which reads page.url) sees it.
        # StringEvaluator ignores the page and program_html re-navigates to its own targets, so this is safe for
        # every eval type; without it the eval page sits at about:blank and url_match tasks false-fail.
        try:
            _fu = json.load(open(os.path.join(run_dir, "final_url.json"), encoding="utf-8"))
            _final_url = _fu if isinstance(_fu, str) else (_fu.get("final_url") or _fu.get("url"))
        except Exception:
            _final_url = None
        if _final_url:
            try:
                page.goto(_final_url, wait_until="load", timeout=30000)
            except Exception as e:
                print(f"[wa_official_eval] WARN goto final_url failed ({_final_url}): {e}", file=sys.stderr)
        try:
            score = evaluator_router(cfg)(trajectory, cfg, page, client)   # OFFICIAL scoring
            print(f"[wa_official_eval] {task}: official score = {score}", file=sys.stderr)
        finally:
            b.close()
    return "SUCCESS" if float(score) == 1.0 else "FAILURE"


def main():
    if len(sys.argv) < 3:
        print("usage: python wa_official_eval.py <task> <run_dir>", file=sys.stderr)
        print("FAILURE"); return
    try:
        print(evaluate(sys.argv[1], sys.argv[2]))
    except Exception:
        import traceback
        traceback.print_exc()
        print("FAILURE")


if __name__ == "__main__":
    main()
