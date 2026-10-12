
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

"""BASELINE (no-defense) driver on a WebArena or realbench task — the utility control for run_secure_webarena.py.

The unified baseline web-agent for BOTH benchmarks: observe (screenshot + a11y tree) -> return ONE UI
command -> execute -> repeat, last-3 trajectory, Opus 4.8, the same completion guidance. Commands are JSON
coordinate commands native to this Playwright harness; the per-benchmark seams (site source, prepare, scorer,
output naming) live behind the `Benchmark` object in benchmarks.py, exactly as in the secure driver.

Differences from the secure reddit run are ONLY the defense: no masking (raw screenshot + UNMASKED a11y), no
query/compute indirection, no egress/verify gate. Everything else is shared and identical:
  a11y      : wa_a11y.get_masked_a11y(cdp, []) — the SAME tree/format as secure, just unmasked
  execution : the SAME bare Playwright primitives as run_secure_webarena.do_act (pg.mouse.click / keyboard /
              wheel; _pwkey reused) — verified ~100% reliable in the loop
  LLM path  : secure_llm Vertex client + adaptive thinking (byte-identical to secure)
  scoring   : official WebArena harness (wa_official_eval.py) ; per-task pristine reset via run_suite.py --mode baseline
So baseline-vs-secure isolates the defense's utility cost. The agent is coordinate-grounded (matching the
secure agent's modality) so the comparison is controlled.

Run:  xvfb-run -s "-screen 0 1920x1200x24" python run_baseline_webarena.py --task reddit-29
"""
import argparse
import base64
import datetime
import json
import os
import paths
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(paths.ENV_FILE)
except Exception:
    pass

import secure_llm as L  # noqa: E402
from wa_a11y import get_masked_a11y  # noqa: E402  (SAME a11y as the secure run; empty manifest + no ext = unmasked)
from run_secure_webarena import TASKS, resolve_task, _pwkey, _site, _login_cookies, CHROMIUM_EXE  # noqa: E402  (goals/start_urls + SAME key map + per-site login as secure)
import metrics as M  # noqa: E402
from metrics import Meter  # noqa: E402
from benchmarks import get_benchmark  # noqa: E402  (WebArena | realbench seam)

MODEL = "claude-opus-4-8"
ROOT = paths.SGE_ROOT
REDDIT_STATE = paths.WA_STATE["reddit"]
OUT_ROOT_DEFAULT = os.path.join(paths.RESULTS, "runs")
VIEWPORT = (1920, 1080)
MAX_TRAJ = int(os.getenv("BASELINE_MAX_TRAJ")) if (os.getenv("BASELINE_MAX_TRAJ") or "").strip() else 3  # last-3 obs = WebArena-standard agent (full history piles up raw screenshots -> confuses the vision model on multi-step tasks, e.g. shopping-511: full=fail@21turns vs last-3=SUCCESS@5). Secure actuator keeps FULL history (its design); env override available.
A11Y_MAX = 8000
SETTLE = 0.6                                              # match secure do_act's post-action settle

_SYS = """You are a web agent completing a task in a browser. Each turn you get a screenshot of the current
page (1920x1080 viewport) plus its accessibility tree, and you take ONE action.

The accessibility tree lists elements, one per line, as:  role "name" @x,y wXh  — where @x,y is the element's
top-left viewport coordinate and wXh is its width×height (given for clickable controls: links, buttons, inputs,
vote arrows). To click a control precisely, target its CENTER: (x + w/2, y + h/2). Lines without @x,y are
read-only context; read page content from the element names and the screenshot.

You control ONLY the web PAGE — the screenshot shows page content, NOT the browser toolbar/tabs/address bar.
There is NO address bar: do NOT type URLs. NAVIGATE by CLICKING on-page links and buttons (e.g. category or
section nav links, a listing or title link, a sort control such as "Sort by"). You can also use the site's own
on-page SEARCH box (usually in the top bar): click it, type a query, and press Enter to find items or pages.
To act on every item by or under one ENTITY (all posts by a user, all items in a category, all reviews of a
product), go to that entity's OWN page — its profile / category / product page — which lists exactly those
items, instead of scrolling the whole site. For "the cheapest / newest / highest-rated one", SORT by that
trusted-visible field and take the top rather than comparing every candidate one by one.

COMPLETENESS & VERBATIM. If the goal asks for ALL / every / the full list of items meeting a condition
(everyone who…, all reviews / products / results that…), you are not done until you have examined EVERY
candidate: page through all results and reviews (load the next page until none remain) and check each one —
a partial list is a wrong answer. When the goal asks you to EXTRACT or QUOTE specific text (sentences, names,
quotes, criticisms), reproduce it EXACTLY as written — verbatim, not paraphrased, shortened, or corrected.

TASK: {goal}

Each turn, output EXACTLY ONE action as a single JSON object in a ```json code block:
- {{"action":"click","x":<x>,"y":<y>}} — click a point (usually a control's center).
- {{"action":"type","text":"<text>"}} — type into the currently focused field.
- {{"action":"press","key":"<key>"}} — press a key on the focused field, e.g. "enter", "tab", "ctrl+a".
- {{"action":"scroll","dy":<n>}} — scroll at screen center; NEGATIVE dy scrolls DOWN toward more content,
  positive scrolls up.
- {{"action":"go_back"}} — return to the PREVIOUS page (browser history). Use it to escape a dead-end page
  (e.g. a raw image file opened directly, or a page with no useful controls).
- {{"action":"stop","status":"success"|"give_up","answer":"<answer / what you accomplished>"}} — finish. If the
  task asks a QUESTION (a count, a name, a value), put ONLY the precise ANSWER in `answer` — the exact
  name / number / email / URL / date / title asked for, the value(s) ALONE, with NO surrounding words,
  restated question, units, explanation, and NO leading label or entity noun. ANY extra word can FAIL exact
  scoring: write "42", not "There are 42"; "Hannah Lim", not "The customer is Hannah Lim" or "customer,
  Hannah Lim". Several values → just those values, comma- or newline-separated. (ACTION task → short confirmation ok.)

COMPLETING THE TASK: a task is done only when its GOAL STATE actually holds — not when you have merely set it
up. Almost every task has ONE final committing action that makes the outcome real (vote / post / reply /
subscribe / submit / save); the steps before it only STAGE the outcome. Work out that committing action for
THIS task, carry it through, and confirm it took effect before you stop with "success". A typed message is not a
sent message; a filled form is not a submitted form; a chosen item is not a committed one.

RECOVER, DON'T REPEAT: if an action leaves the screen unchanged, do NOT re-issue the same action — read the
result and change course (a different / more precise target, scroll to find the control, or a different step).

Think briefly, then output ONE action."""

_OBS_Q = ("Given the screenshot and the accessibility tree below:\n{a11y}\n\n"
          "What is your next action?")


def parse_command(response):
    """Extract the one JSON action object from the model text. Returns a dict (with 'action') or None."""
    if not response:
        return None
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", response, re.S) or re.findall(r"```\s*(\{.*?\})\s*```", response, re.S)
    cand = blocks[-1] if blocks else None
    if cand is None:                                     # fall back: last balanced {...} mentioning "action"
        starts = [i for i, c in enumerate(response) if c == "{"]
        for i in reversed(starts):
            depth = 0
            for j in range(i, len(response)):
                depth += (response[j] == "{") - (response[j] == "}")
                if depth == 0:
                    blob = response[i:j + 1]
                    if '"action"' in blob:
                        cand = blob
                    break
            if cand:
                break
    if cand is None:
        return None
    try:
        op = json.loads(cand)
        return op if isinstance(op, dict) and op.get("action") else None
    except Exception:
        return None


def do_act(op, pg, log):
    """Execute one coordinate UI-command via the SAME bare Playwright primitives as secure do_act (no gates)."""
    action = op.get("action")
    try:
        if action == "click":
            x, y = int(op["x"]), int(op["y"])
            x = max(0, min(VIEWPORT[0] - 1, x)); y = max(0, min(VIEWPORT[1] - 1, y))
            pg.mouse.click(x, y); log(f"click ({x},{y})")
        elif action == "type":
            text = op.get("text", "")
            if isinstance(text, str) and text:
                pg.keyboard.type(text, delay=30); log(f"type {text[:40]!r}")
            else:
                log("type: empty text, skipped")
        elif action == "press":
            key = op.get("key", "enter")
            pg.keyboard.press(_pwkey(key)); log(f"press {key}")
        elif action == "scroll":
            dy = int(op.get("dy", -400))
            pg.mouse.move(VIEWPORT[0] // 2, VIEWPORT[1] // 2)
            pg.mouse.wheel(0, -dy); log(f"scroll {dy}")   # op dy negative=DOWN -> negate for Playwright wheel
        elif action == "go_back":
            pg.go_back(wait_until="load", timeout=8000); log("go_back")
        else:
            log(f"unknown action {action!r}, skipped")
    except Exception as e:
        log(f"[act] failed: {e} :: {json.dumps(op, default=str)[:120]}")


def _make_llm(provider, model):
    """Build (client, model_id, gtypes) for the baseline's single-text-call path, per provider. Mirrors
    secure_shadow's provider setup so the baseline calls gpt-5 (Azure) / gemini (Vertex) the same way the
    secure loop does — a fair no-defense comparison for each model family (was Anthropic-only before)."""
    if provider == "gemini":
        from google import genai
        from google.genai import types
        _ho = {"timeout": 120000}                                        # ms; bound a black-holed Gemini call
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"), http_options=_ho)
        else:
            client = genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                                  location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"), http_options=_ho)
        return client, model, types
    if provider == "openai":
        import openai
        client = openai.AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                                    timeout=120.0, max_retries=2)                 # bound a black-holed GPT call
        return client, (model or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5"), None
    return L.vertex_client(), model, None


def run(task, max_steps, out_root, model=MODEL, seed=None, benchmark="webarena", version=None):
    bm = get_benchmark(benchmark, version)
    spec = bm.resolve(task)
    goal, start_url = spec["goal"], spec["start_url"]
    run_id = "run-%d" % int(time.time())
    out_dir = os.path.join(out_root, bm.out_dir_name(task, model, "baseline"))
    if seed is not None:                                     # multi-seed run -> per-seed subdir (mirror in run_suite.py)
        out_dir = os.path.join(out_dir, f"seed{seed}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    profile_dir = os.path.join(out_dir, "_chrome_profile")
    logf = open(os.path.join(out_dir, "runtime.log"), "w", encoding="utf-8")

    def log(m):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    def save(name, data, mode="wb"):
        with open(os.path.join(out_dir, name), mode, encoding=None if "b" in mode else "utf-8") as f:
            f.write(data)

    system = _SYS.format(goal=goal)

    log(f"BASELINE[{benchmark}] task={task} model={model}")
    log(f"goal: {goal}"); log(f"start_url: {start_url}")
    save("meta.json", json.dumps({"task": task, "goal": goal, "start_url": start_url, "model": model,
         "mode": "baseline", "benchmark": benchmark, "env": bm.env_label(task), "run_id": run_id,
         "max_steps": max_steps, "screen_size": list(VIEWPORT),
         "observation_type": "screenshot_a11y_tree", "action_space": "json-coordinate-commands",
         "agent": "native coordinate web agent (screenshot+a11y -> JSON UI commands; unmasked)",
         **bm.meta_extra(task)}, indent=2), "w")

    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    ctx = pg = cdp = client = None
    result = {"task": task, bm.result_verdict_key: None}
    ctx_state = {"cdp_port": None, "cdp_http": None}         # realbench: filled by extra_launch_args (--remote-debugging-port)
    t_task0 = time.perf_counter()
    meter = Meter(("baseline",))
    traj = []                                             # last-MAX_TRAJ (a11y, png_b64, thought)
    last_response = ""
    stop_answer = None
    turn = 0

    def observe():
        try:
            pg.wait_for_load_state("load", timeout=6000)
        except Exception:
            pass
        png = pg.screenshot()
        a11y = ""
        try:
            a11y = get_masked_a11y(cdp, [], viewport=VIEWPORT, max_chars=A11Y_MAX)   # empty manifest = unmasked
        except Exception as e:
            log(f"[a11y] failed: {e}")
        return png, a11y

    def call(system_text, hist):
        """One model call over neutral history [(role, text, b64|None)] -> text. Provider-branched so
        gpt-5 (Azure) and gemini (Vertex) work, not just Anthropic; identical prompt content across all."""
        t0 = time.perf_counter()
        usage = None
        try:
            if provider == "gemini":
                contents = []
                for role, text, b64 in hist:
                    parts = [_gtypes.Part.from_text(text=text)]
                    if b64:
                        parts.append(_gtypes.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png"))
                    contents.append(_gtypes.Content(role=("user" if role == "user" else "model"), parts=parts))
                r = client.models.generate_content(
                    model=model_id, contents=contents,
                    config=_gtypes.GenerateContentConfig(max_output_tokens=8000, system_instruction=system_text))
                usage = M.norm_gemini(r.usage_metadata)
                return r.text or ""
            if provider == "openai":
                msgs = [{"role": "system", "content": system_text}]
                for role, text, b64 in hist:
                    if role == "user":
                        c = [{"type": "text", "text": text}]
                        if b64:
                            c.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                        msgs.append({"role": "user", "content": c})
                    else:
                        msgs.append({"role": "assistant", "content": text})
                rea = model_id.lower().startswith(("gpt-5", "o1", "o3", "o4"))
                kw = dict(model=model_id, messages=msgs, max_completion_tokens=(16000 if rea else 8000))
                if rea:
                    kw["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "medium")
                r = client.chat.completions.create(**kw)
                usage = M.norm_openai(r.usage)
                return r.choices[0].message.content or ""
            msgs = []
            for role, text, b64 in hist:
                if role == "user":
                    msgs.append({"role": "user",
                                 "content": [{"type": "text", "text": text}] + ([L.img_block(b64)] if b64 else [])})
                else:
                    msgs.append({"role": "assistant", "content": text})
            r = client.messages.create(model=model, max_tokens=8000, system=system_text,
                                       messages=msgs, extra_body={"thinking": {"type": "adaptive"}})
            usage = M.norm_anthropic(r.usage)
            return next((b.text for b in r.content if getattr(b, "type", None) == "text"), "") or ""
        except Exception as e:
            log(f"[llm] call failed: {e}"); return ""
        finally:
            meter.add_time("baseline", time.perf_counter() - t0)
            if usage:
                meter.add_tokens("baseline", *usage)

    try:
        log("launch headed Chromium (Xvfb) + cookies + goto (NO extension / no masking) ...")
        ctx = pw.chromium.launch_persistent_context(
            profile_dir, headless=False, executable_path=CHROMIUM_EXE,   # newest chromium (1187 crashes on gitlab's editor)
            args=["--no-sandbox", "--disable-dev-shm-usage", *bm.extra_launch_args(task, ctx_state)],
            viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]}, device_scale_factor=1)
        cookies = bm.pre_cookies(task, log)                 # WebArena: per-site login; realbench: [] (anonymous)
        if cookies:
            ctx.add_cookies(cookies)
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        cdp = ctx.new_cdp_session(pg)
        ml = (model or "").lower()
        provider = ("gemini" if ml.startswith("gemini")
                    else "openai" if ml.startswith(("gpt", "o1", "o3", "o4")) else "anthropic")
        client, model_id, _gtypes = _make_llm(provider, model)
        log(f"[llm] provider={provider} model={model_id}")
        pg.goto(start_url, wait_until="load"); time.sleep(1.5)
        bm.prepare(pg, cdp, ctx_state, task, run_id, log)   # realbench: agisdk config; WebArena: no-op

        done = False
        no_action = 0
        while not done and turn < max_steps:
            png, a11y = observe()
            b64 = base64.b64encode(png).decode()
            turn += 1
            save(f"step_{turn}.png", png)
            save(f"step_{turn}.axtree.txt", a11y, "w")

            hist = []
            for (h_a11y, h_b64, h_thought) in (traj if MAX_TRAJ is None else traj[-MAX_TRAJ:]):   # FULL history (parity w/ secure) unless BASELINE_MAX_TRAJ set
                hist.append(("user", _OBS_Q.format(a11y=h_a11y), h_b64))
                hist.append(("assistant", h_thought or "(no action)", None))
            hist.append(("user", _OBS_Q.format(a11y=a11y), b64))

            response = call(system, hist)
            last_response = response or last_response
            op = parse_command(response)
            save(f"step_{turn}.response.txt", response or "", "w")
            traj.append((a11y, b64, response))
            log(f"--- step {turn}: {op.get('action') if op else 'no-op'} --- {str(response)[:180]}")

            if not op:
                no_action += 1
                if no_action >= 3:
                    log("3 consecutive turns with no valid action -> stopping"); break
                continue
            no_action = 0
            if op["action"] == "stop":
                stop_answer = op.get("answer", "") or ""
                log(f"agent: stop [{op.get('status')}] {stop_answer[:140]}"); done = True; break
            do_act(op, pg, log)
            time.sleep(SETTLE)

        final_url = pg.url
        save("final_url.json", json.dumps({"final_url": final_url}), "w")
        # model_response.txt feeds string_match tasks: prefer the explicit stop answer, else the last reply.
        save("model_response.txt", (stop_answer if stop_answer is not None else last_response) or "", "w")
        result.update({"turns": turn, "final_url": final_url})
        log(f"settling 3s, then scoring [{benchmark}] ...")
        time.sleep(3)
        result.update(bm.score(task, out_dir, final_url, stop_answer, last_response, ctx_state, run_id, log))
        result["metrics"] = meter.snapshot(time.perf_counter() - t_task0)
        save("result.json", json.dumps(result, indent=2, default=str), "w")
        log(f"EVAL[{benchmark}]: {result.get(bm.result_verdict_key)}  (turns={turn}, final_url={final_url})")
    except Exception as e:
        import traceback
        log(f"EXCEPTION: {e}\n{traceback.format_exc()}")
        result["error"] = str(e)
        result["metrics"] = meter.snapshot(time.perf_counter() - t_task0)
        save("result.json", json.dumps(result, indent=2, default=str), "w")
    finally:
        try:
            ctx and ctx.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass
        logf.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="reddit-29")   # any <site>-<N> WebArena id (curated or full-site)
    ap.add_argument("--max-steps", type=int, default=35)
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=None, help="multi-seed repeat index -> <out-root>/<task>_.../seed<N>/")
    ap.add_argument("--benchmark", default="webarena", choices=["webarena", "realbench"])
    ap.add_argument("--version", default=None, help="realbench task version (default v2); ignored for webarena")
    a = ap.parse_args()
    r = run(a.task, a.max_steps, a.out_root, model=a.model, seed=a.seed, benchmark=a.benchmark, version=a.version)
    print("RESULT:", r.get("webarena_verdict") or r.get("corrected_verdict"),
          "turns=", r.get("turns"), "final_url=", r.get("final_url"))


if __name__ == "__main__":
    main()
