
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

"""NO-QUERY ablation loop on a WebArena task (headed Chromium/Xvfb via Playwright).

A parallel twin of run_secure_webarena.py with the READ CHANNEL removed. Same masking (V⊤ + content-free
marks + masked a11y), same egress-CFI gate, same act executor, same scorer — but NO `query`, NO `compute`,
NO K/G memory, and NO verify-before-action gate. The actuator (noquery_actuator.Actuator) can emit only
`act` and `return`, so on query-necessary tasks it must guess. Running the same tasks through this and
through the secure pipeline measures how much the `query` capability is worth.

run_secure_webarena.py / secure_actuator.py / tool_spec.py are UNTOUCHED. This module swaps
tool_spec.TOOLS -> noquery_tool_spec.TOOLS at import (this process only) so native-tool adapters advertise
just act+return; run_secure_webarena runs in its own process and is unaffected.

Run:  xvfb-run -s "-screen 0 1920x1200x24" python run_noquery_webarena.py --task reddit-67
"""
import argparse
import base64
import datetime
import json
import os
import paths
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

# Swap the advertised op set to act+return BEFORE any adapter is built (adapters read tool_spec.TOOLS in
# their __init__). This process never uses the secure tools; the secure runner is a separate process.
import tool_spec as _TS            # noqa: E402
import noquery_tool_spec as _NQ    # noqa: E402
_TS.TOOLS = _NQ.TOOLS
_TS.TOOL_BY_NAME = _NQ.TOOL_BY_NAME

from smask_runtime import Smask                       # noqa: E402
from wa_a11y import get_masked_a11y                   # noqa: E402
from noquery_actuator import Actuator                 # noqa: E402  (act+return only)
from secure_diff import frac_changed                  # noqa: E402
import secure_protocol as P                           # noqa: E402
from benchmarks import get_benchmark                  # noqa: E402  (WebArena | realbench seam)
# Task table + eval are shared, unchanged, with the secure runner.
from run_secure_webarena import (TASKS, resolve_task, _pwkey, act_sig, _site, _login_cookies,  # noqa: E402
                                 _seed_hosts, CHROMIUM_EXE)

MODEL = "claude-opus-4-8"
ROOT = paths.SGE_ROOT
EXT_DIR = paths.EXT_DIR
OUT_ROOT_DEFAULT = os.path.join(paths.RESULTS, "runs")
MASK_SETTLE = 0.4
VIEWPORT = (1920, 1080)


def run(task, max_steps, pause, out_root, provider="anthropic", model=MODEL, seed=None,
        benchmark="webarena", version=None):
    bm = get_benchmark(benchmark, version)
    spec = bm.resolve(task)
    goal, start_url = spec["goal"], spec["start_url"]

    out_dir = os.path.join(out_root, bm.out_dir_name(task, model, "noquery"))
    if seed is not None:                                     # multi-seed run -> per-seed subdir (mirror in run_suite.py)
        out_dir = os.path.join(out_dir, f"seed{seed}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    profile_dir = os.path.join(out_dir, "_chrome_profile")           # FRESH profile -> seeds mask rules from seed.js
    logf = open(os.path.join(out_dir, "runtime.log"), "w", encoding="utf-8")

    def log(m):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    def save(name, png):
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(png)

    def save_json(name, obj):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str, ensure_ascii=False)

    def save_text(name, s):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(s or "")

    def b64(png):
        return base64.b64encode(png).decode()

    run_id = "run-%d" % int(time.time())
    log(f"NOQUERY[{benchmark}]  task={task} model={model} provider={provider}")
    log(f"goal: {goal}"); log(f"start_url: {start_url}")
    save_json("meta.json", {"task": task, "goal": goal, "start_url": start_url, "run_id": run_id,
                            "model": model, "provider": provider,
                            "mode": "noquery", "benchmark": benchmark, "env": bm.env_label(task),
                            "max_steps": max_steps, "screen_size": list(VIEWPORT), **bm.meta_extra(task),
                            "ablation": "no query/compute/K/G/verify-gate; masking+egress+act+return kept"})

    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    ctx = pg = smask = None
    result = {"task": task, bm.result_verdict_key: None}
    ctx_state = {"cdp_port": None, "cdp_http": None}         # realbench: filled by extra_launch_args (--remote-debugging-port)
    op_trace = []
    user_answer = None

    try:
        log("launch headed Chromium (Xvfb) + extension + cookies + goto ...")
        ctx = pw.chromium.launch_persistent_context(
            profile_dir, headless=False, executable_path=CHROMIUM_EXE,   # newest chromium (fix: was Playwright default)
            args=["--no-sandbox", "--disable-dev-shm-usage",
                  f"--disable-extensions-except={EXT_DIR}", f"--load-extension={EXT_DIR}",
                  *bm.extra_launch_args(task, ctx_state)],               # realbench: --remote-debugging-port for agisdk
            viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]}, device_scale_factor=1)
        cookies = bm.pre_cookies(task, log)                          # WebArena: per-site login cookies; realbench: [] (anon)
        if cookies:
            ctx.add_cookies(cookies)                                 # login BEFORE first navigation
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        cdp = ctx.new_cdp_session(pg)
        pg.goto(start_url, wait_until="load"); time.sleep(1.5)
        bm.prepare(pg, cdp, ctx_state, task, run_id, log)            # realbench: agisdk config; WebArena: no-op

        smask = Smask(page=pg, host_substr="", log=log)             # injected-page mode
        if not smask.wait_ready(timeout=20):                        # PREFLIGHT: extension MUST be live, else we'd screenshot UNMASKED
            raise RuntimeError("mask extension not live (#__smask_overlay absent) — refusing to run (would screenshot an unmasked page)")
        _host = pg.evaluate("() => location.host")                  # verify rules are SEEDED for this host (NOT inferred from page content)
        if _host not in _seed_hosts():
            raise RuntimeError(f"mask rules NOT seeded for host {_host} (absent from seed.js) — refusing to run (would screenshot an unmasked page)")
        smask.mask(); time.sleep(MASK_SETTLE)                       # apply masking (count is INFORMATIONAL — 0 is fine on a content-empty start page)
        _stamped = pg.eval_on_selector_all('[data-trust]', "e=>e.length")
        log(f"[preflight] mask extension live; rules seeded for {_host}; {_stamped} data-trust stamped on start page")
        log(f"[actuator] provider={provider} model={model}  (no-query: act+return only)")
        actuator = Actuator(goal, provider=provider, model=model, log=log)
        A11Y_MAX = 5000

        def a11y(man, reveal=None):
            try:
                return get_masked_a11y(cdp, man, reveal_mark=reveal, viewport=VIEWPORT, max_chars=A11Y_MAX)
            except Exception as e:
                log(f"[a11y] failed: {e}"); return ""

        def _visible_untrusted():
            try:
                return pg.eval_on_selector_all(
                    '[data-trust="untrusted"],[data-trust="frame"],[data-trust="obscured"]',
                    "els=>els.filter(e=>{const r=e.getBoundingClientRect();"
                    "return r.width>3&&r.height>3&&r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth;}).length")
            except Exception:
                return 0

        def observe_vtop():
            # Same mask-then-verify-then-screenshot guarantee as the secure runner (V⊤ fully masked before it
            # reaches the actuator). No K store, so no set_marks.
            try:
                pg.wait_for_load_state("load", timeout=6000)
            except Exception:
                pass
            rdy = smask.wait_ready(timeout=8)                        # a DEAD extension must NOT masquerade as an empty (vis==0) page
            man = []; vis = 0
            for _ in range(6):
                if not rdy:                                         # extension may finish (re)injecting mid-loop (heavy pages / post-nav)
                    rdy = smask.wait_ready(timeout=1)
                smask.mask(); time.sleep(MASK_SETTLE)
                man = smask.manifest()
                vis = _visible_untrusted()
                if rdy and (vis == 0 or len(man) >= vis):
                    break
                time.sleep(MASK_SETTLE)
            else:                                                   # never converged in 6 tries
                _u = (pg.url or "").strip()                         # ONLY blank/browser-internal pages carry no maskable content
                _benign = _u in ("", "about:blank") or _u.startswith(("about:", "chrome://", "chrome-error://", "edge://"))
                if not rdy and not _benign:                        # live CONTENT page (http/https/data/file) w/o mask -> leak risk
                    raise RuntimeError(f"observe_vtop: mask extension not live at {_u} — refusing to screenshot an unmasked content page")
                if not rdy:                                         # blank page (e.g. go_back off the front of history to about:blank):
                    log(f"[observe_vtop] off-site/blank page {_u!r} — no maskable content; screenshotting as-is so the agent "  # allow it
                        "can leave the site and recover instead of hard-aborting")
                else:                                               # extension LIVE but coverage short (frame-collapse/late render)
                    log(f"[observe_vtop] WARN mask unverified at {pg.url} ({vis} untrusted visible, {len(man)} boxed after 6 tries) "
                        "-- best-effort screenshot (frame-collapse or stubborn late render)")
            for r in man:                                           # VIEWPORT-native coords
                r["sx"], r["sy"] = r["x"], r["y"]
            png = pg.screenshot()
            return png, man, a11y(man)

        def do_act(op, turn, manifest, cur_url):
            action = op.get("action")
            on_aff = True
            if action == "click":
                target = op.get("target") or {}
                if "mark" in target:
                    r = next((x for x in manifest if x["n"] == int(target["mark"])), None)
                    if not r:
                        return {"feedback": f"act refused: mark {target['mark']} not on screen.", "advanced": False, "reason": "mark-not-on-screen"}
                    px, py = r["sx"] + r["w"] // 2, r["sy"] + r["h"] // 2
                elif "x" in target and "y" in target:
                    px, py = int(target["x"]), int(target["y"])
                else:
                    return {"feedback": "act refused: click needs target {x,y} or {mark}.", "advanced": False, "reason": "no-target"}
                v = smask.egress(px, py, viewport=True)              # CFI gate (KEPT — the security invariant)
                if not v.get("admit"):
                    rsn = v.get("reason", "refused")
                    hint = {
                        "off-viewport": "that coordinate is outside the 1920x1080 page — re-read the screen and click "
                                        "a control you can actually see.",
                        "untrusted-non-affordance": "that point is masked/untrusted content, which is not itself "
                                        "clickable — to open a masked item, click it by its {mark:N}; to click a "
                                        "control, give the coordinates of a trusted button/link.",
                        "off-origin": "that target navigates OFF this site — pick an on-page (same-site) control.",
                    }.get(rsn, "click a first-party control on the page.")
                    return {"feedback": f"act REFUSED by egress ({rsn}) at ({px},{py}): {hint}",
                            "advanced": False, "reason": rsn}
                # NO verify-before-action gate in the no-query pipeline: the agent has no way to query a mark,
                # so it clicks masked items blind (an egress-admitted first-party route).
                on_aff = (v.get("reason") == "first-party-affordance")
                pg.mouse.click(px, py)
                acted = f"click ({px},{py}) [{v.get('tag')}, {v.get('origin')}] admitted"
            elif action == "type":
                text = op.get("text", "")
                if not isinstance(text, str) or not text:
                    return {"feedback": "act refused: type needs non-empty text (a trusted literal you author).", "advanced": False, "reason": "type-empty"}
                pg.keyboard.type(text, delay=30)
                acted = f"typed {text!r} into focused field"
            elif action == "press":
                key = op.get("key", "enter")
                pg.keyboard.press(_pwkey(key))
                acted = f"pressed {key}"
            elif action == "scroll":
                dy = int(op.get("dy", -400))
                tg = op.get("target") or {}
                sx, sy = int(tg.get("x", VIEWPORT[0] // 2)), int(tg.get("y", VIEWPORT[1] // 2))
                pg.mouse.move(sx, sy)
                pg.mouse.wheel(0, -dy)
                acted = f"scrolled {dy} at ({sx},{sy})"
            elif action == "go_back":
                try:
                    pg.go_back(wait_until="load", timeout=8000)
                except Exception as e:
                    log(f"[act] go_back: {e}")
                acted = "went back (browser history)"
            else:
                return {"feedback": f"act refused: unknown action {action!r}.", "advanced": False, "reason": "unknown-action"}
            log(f"[act] {acted}")
            time.sleep(0.6)
            vtop, man, a11y_new = observe_vtop()
            new_url = pg.url
            navigated = new_url != cur_url
            save(f"turn_{turn}_after.png", vtop)
            return {"feedback": acted + (" — NAVIGATED." if navigated else " — screen re-observed."),
                    "advanced": True, "vtop": vtop, "manifest": man, "a11y": a11y_new, "new_url": new_url,
                    "on_affordance": on_aff}

        def do_return(op):
            """Terminal. No payload/G in the no-query pipeline — `answer` is the agent's own literal text."""
            answer = op.get("answer", "")
            status = op.get("status", "success")
            if not isinstance(answer, str) or not answer.strip():
                return {"ok": False, "err": "return rejected: need a non-empty `answer` (your own text)."}
            return {"ok": True, "answer": answer, "status": status}

        # ---- the loop (act/return only) ----
        vtop_png, manifest, a11y_top = observe_vtop()
        cur_url = pg.url
        feedback, turn, done = "", 0, False
        act_log = []
        nochange_streak = 0
        engaged = 0          # acts that actually changed/advanced the page (genuine exploration)
        act_attempts = 0     # total act ops emitted
        GIVEUP_FLOOR = 3     # must genuinely engage this many times before a give_up is allowed
        while not done and turn < max_steps:
            turn += 1
            op, raw = actuator.turn(b64(vtop_png), P.render_marks(manifest), a11y_top, feedback)
            save(f"turn_{turn}_vtop.png", vtop_png); save_json(f"turn_{turn}_op.json", op); save_text(f"turn_{turn}_raw.txt", raw)
            save_text(f"turn_{turn}_a11y.txt", a11y_top)
            kind = op.get("op")
            log(f"--- turn {turn}: {kind} --- {json.dumps(op, default=str)[:220]}")
            if kind == "act":
                act_attempts += 1
                prev_vtop = vtop_png
                res = do_act(op, turn, manifest, cur_url)
                feedback = res["feedback"]
                if res.get("advanced"):
                    new_url = res.get("new_url", cur_url)
                    navigated = new_url != cur_url
                    changed = navigated or frac_changed(prev_vtop, res["vtop"]) > 0.002
                    if changed:
                        engaged += 1                       # a real, page-changing attempt
                    on_aff = res.get("on_affordance", True)
                    vtop_png, manifest, cur_url = res["vtop"], res["manifest"], new_url
                    a11y_top = res.get("a11y", a11y_top)
                    if changed:
                        outcome = "ok"; nochange_streak = 0
                    elif on_aff:
                        outcome = "ok"
                    else:
                        nochange_streak += 1
                        outcome = "nochange"
                        if nochange_streak >= 5:
                            feedback += ("\n\nNOTE: your last several clicks changed nothing and did not land on any "
                                         "interactive control — re-read the page and click an actual button/link, or "
                                         "open a masked item by its {mark:N}.")
                else:
                    outcome = "refused:" + str(res.get("reason", "refused"))
                sig = act_sig(op)
                window = act_log[-6:]
                nonprog_same = sum(1 for s, o in window if s == sig and o != "ok")
                total_same = sum(1 for s, _ in window if s == sig) + 1
                act_log.append((sig, outcome))
                banner = None
                if outcome != "ok" and (nonprog_same + 1) >= 2:
                    what = "being refused" if outcome.startswith("refused") else "having no effect"
                    banner = (f"⚠ You have issued this SAME action {nonprog_same + 1} times and it keeps {what} — "
                              f"repeating it will NOT work. Do something DIFFERENT (a different target/op).")
                elif op.get("action") == "click" and total_same >= 3:
                    banner = (f"⚠ You have targeted this same spot {total_same} times. If it is not advancing the "
                              f"task, it is not the right control — try a DIFFERENT element.")
                if banner:
                    feedback = banner + "\n\nDetails:\n" + feedback
            elif kind == "return":                              # the sole terminal op
                # Engagement floor (fairness): a no-query agent must GENUINELY TRY before it may concede, so
                # its FAILUREs reflect real attempts, not premature surrender. Block give_up until it has made
                # a few page-changing attempts (with a cap so a truly stuck agent isn't looped forever).
                if op.get("status") == "give_up" and engaged < GIVEUP_FLOOR and act_attempts < 8:
                    feedback = ("Not yet — before you may GIVE UP you must genuinely try. So far only "
                                f"{engaged} of your actions changed the page. ACTUALLY ATTEMPT the task now: "
                                "scroll through the relevant items, use the search / sort / filter, open a "
                                "candidate, take the action. Give up only after a real attempt with no path left.")
                    log(f"[give_up floor] rejected: engaged={engaged} act_attempts={act_attempts}")
                    op_trace.append({"turn": turn, "op": op, "feedback": feedback[:500]})
                    save_json("op_trace.json", op_trace)
                    continue
                rr = do_return(op)
                if rr["ok"]:
                    user_answer = rr["answer"]; result["end_status"] = rr.get("status", "success")
                    op_trace.append({"turn": turn, "op": op})
                    save_text("model_response.txt", user_answer)
                    log(f"[return] status={rr.get('status')} answer ({len(user_answer)} chars) -> user sink")
                    done = True
                    break
                feedback = rr["err"]
            else:
                feedback = f"Your output was not a valid op ({op.get('error')}). Emit exactly ONE op (act|return)."
            log(f"  feedback: {feedback[:200]}")
            op_trace.append({"turn": turn, "op": op, "feedback": feedback[:500]})
            save_json("op_trace.json", op_trace)

        # ---- score via the benchmark scorer (WebArena harness OR agisdk over CDP) ----
        final_url = pg.url
        save_json("final_url.json", {"final_url": final_url})
        save_text("model_response.txt", user_answer if user_answer is not None else feedback)
        result["turns"] = turn
        result["final_url"] = final_url
        result["end_answer"] = user_answer
        log(f"settling 3s, then scoring [{benchmark}] ...")
        time.sleep(3)
        result.update(bm.score(task, out_dir, final_url, user_answer, feedback, ctx_state, run_id, log))
        save_json("result.json", result)
        log(f"EVAL[{benchmark}]: {result.get(bm.result_verdict_key)}  (turns={turn}, final_url={final_url})")

    except Exception as e:
        import traceback
        log(f"EXCEPTION: {e}\n{traceback.format_exc()}")
        result["error"] = str(e)
        save_json("result.json", result)
    finally:
        save_json("op_trace.json", op_trace)
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
    ap.add_argument("--task", default="reddit-404")   # any <site>-<N> WebArena id (curated or full-site)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--provider", default=None, choices=["text", "anthropic", "openai", "gemini"])
    ap.add_argument("--seed", type=int, default=None, help="multi-seed repeat index -> <out-root>/<task>_.../seed<N>/")
    ap.add_argument("--benchmark", default="webarena", choices=["webarena", "realbench"])
    ap.add_argument("--version", default=None, help="realbench task version (default v2); ignored for webarena")
    a = ap.parse_args()
    ml = a.model.lower()
    provider = a.provider or ("gemini" if ml.startswith("gemini")
                              else "openai" if ml.startswith(("gpt", "o1", "o3", "o4")) else "anthropic")
    r = run(a.task, a.max_steps, a.pause, a.out_root, provider=provider, model=a.model, seed=a.seed,
            benchmark=a.benchmark, version=a.version)
    print("RESULT:", r.get("webarena_verdict") or r.get("corrected_verdict"),
          "turns=", r.get("turns"), "final_url=", r.get("final_url"))


if __name__ == "__main__":
    main()
