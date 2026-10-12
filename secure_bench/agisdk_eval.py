
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

"""Bridge to agisdk's OFFICIAL REAL evaluation — used as-is (no reimplementation).

Runs in the dedicated .agisdk_venv. The scoring is agisdk's own
`WebCloneEvaluator.evaluate()` (which executes the bundled `eval_*.py` scripts);
we only drive the documented `/config` (baseline) and `/finish` (env_state)
endpoints against the already-running Chromium over CDP.

Subcommands (all print one JSON line to stdout):
  taskinfo   <task> <version>
  config     <cdp_url> <task> <version> <run_id>
  finisheval <cdp_url> <task> <version> <run_id> <model_response_file|->
"""
import json
import sys

from playwright.sync_api import sync_playwright
from agisdk.REAL.browsergym.webclones.task_config import TaskConfig
from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator


def _tc(task, version):
    return TaskConfig(task, version=version)


def _canonical(tc):
    return getattr(tc, "canonical_id", None) or f"{tc.version}.{tc.task_name}"


def _config_task_id(tc, run_id):
    # mirrors AbstractWebCloneTask.setup()
    if tc.version == "v1" and str(run_id) != "0":
        return tc.task_name
    return _canonical(tc)


def cmd_taskinfo(task, version):
    tc = _tc(task, version)
    print(json.dumps({
        "task_name": tc.task_name,
        "version": tc.version,
        "goal": tc.get_goal(),
        "url": tc.get_start_url(),
        "evals": [(e.type, getattr(e, "script", None)) for e in tc.get_evals()],
    }))


def cmd_config(cdp, task, version, run_id):
    tc = _tc(task, version)
    url = tc.get_start_url().rstrip("/")
    ctid = _config_task_id(tc, run_id)
    cfg = f"{url}/config?run_id={run_id}&task_id={ctid}&latency=0"
    host = url.split("//")[-1].split("/")[0]
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp(cdp)
        try:
            ctx = b.contexts[0]
            bg = ctx.new_page()
            bg.goto(cfg, wait_until="networkidle", timeout=30000)
            bg.wait_for_timeout(1000)
            bg.goto(f"{url}/finish", wait_until="networkidle", timeout=30000)  # prime (mirrors setup)
            bg.wait_for_timeout(500)
            bg.close()
            # reload the agent's app tab so it reflects the configured run (mirrors setup's self.page.goto)
            agent_pg = None
            for pg in ctx.pages:
                if host in (pg.url or ""):
                    agent_pg = pg
                    break
            if agent_pg is None and ctx.pages:
                agent_pg = ctx.pages[0]
            if agent_pg:
                agent_pg.bring_to_front()
                agent_pg.goto(url, wait_until="domcontentloaded", timeout=30000)
        finally:
            b.close()
    print(json.dumps({"config_url": cfg, "config_task_id": ctid}))


def _fetch_env_state(ctx, url):
    pg = ctx.new_page()
    try:
        pg.goto(f"{url.rstrip('/')}/finish", wait_until="networkidle", timeout=30000)
        pg.wait_for_timeout(800)
        # the /finish <pre> is CSS-hidden, so inner_text() returns "" -> use text_content()
        el = pg.wait_for_selector("pre", state="attached", timeout=10000)
        txt = el.text_content() if el else ""
        return json.loads(txt) if txt and txt.strip() else {}
    finally:
        pg.close()


def cmd_finisheval(cdp, task, version, run_id, mr_file):
    tc = _tc(task, version)
    url = tc.get_start_url()
    model_response = ""
    if mr_file and mr_file != "-":
        try:
            with open(mr_file, encoding="utf-8") as f:
                model_response = f.read()
        except Exception:
            pass
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp(cdp)
        try:
            env_state = _fetch_env_state(b.contexts[0], url)
        finally:
            b.close()
    ev = WebCloneEvaluator(task_config=tc)
    reward, done, message, info = ev.evaluate(env_state=env_state, model_response=model_response)
    results = info.get("results") if isinstance(info, dict) else None
    passed = bool(results) and all(bool(r[0]) for r in results)
    print(json.dumps({
        "reward": reward, "done": done, "passed": passed, "message": message,
        "info": info, "env_state": env_state,
    }))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "taskinfo":
        cmd_taskinfo(sys.argv[2], sys.argv[3])
    elif cmd == "config":
        cmd_config(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "finisheval":
        cmd_finisheval(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                       sys.argv[6] if len(sys.argv) > 6 else "-")
    else:
        print(json.dumps({"error": f"unknown cmd {cmd}"}))
        sys.exit(1)
