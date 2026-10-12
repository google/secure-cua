
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

"""Secure actuator loop on a WebArena or realbench task, driven by headed Chromium (Xvfb) via Playwright.

The unified secure driver for BOTH benchmarks: the trusted ACTUATOR sees only V⊤ (masked
screenshot) + a content-free mark list + a masked a11y tree + K's schema; reads untrusted content only as
closed typed answers from an isolated SHADOW (one call per mark over a V_g that reveals only that mark);
reduces with deterministic `compute`; and acts by COORDINATE gated by an egress check. The only per-benchmark
seams (site source, per-task prepare, scorer, output naming) live behind the `Benchmark` object in
benchmarks.py — WebArena resets a docker site + scores via the official harness (wa_official_eval.py);
realbench attaches to a public hosted URL over CDP + scores via the agisdk venv.
Screenshot: page.screenshot()  |  input: page.mouse/keyboard  |  a11y: wa_a11y.get_masked_a11y (CDP AXTree).
Coordinates are VIEWPORT-native (page.screenshot / page.mouse frame); egress is called with viewport=True.

Run (needs Xvfb + a fresh ADC):  xvfb-run -s "-screen 0 1920x1200x24" python run_secure_webarena.py --task reddit-404
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
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(paths.ENV_FILE)
except Exception:
    pass

from smask_runtime import Smask  # noqa: E402
from wa_a11y import get_masked_a11y  # noqa: E402
from secure_store import KStore, GStore, compute  # noqa: E402
from secure_actuator import Actuator  # noqa: E402
from secure_shadow import Shadow  # noqa: E402
from secure_diff import frac_changed  # noqa: E402
import secure_protocol as P  # noqa: E402
from metrics import Meter  # noqa: E402
from benchmarks import get_benchmark  # noqa: E402  (WebArena | realbench seam)

MODEL = "claude-opus-4-8"
ROOT = paths.SGE_ROOT
EXT_DIR = paths.EXT_DIR
REDDIT_STATE   = paths.WA_STATE["reddit"]
SHOPPING_STATE = paths.WA_STATE["shopping"]
GITLAB_STATE         = paths.WA_STATE["gitlab"]
SHOPPING_ADMIN_STATE = paths.WA_STATE["shopping_admin"]
SITE_STATE = {"reddit": REDDIT_STATE, "shopping": SHOPPING_STATE,     # per-site agent login cookies
              "gitlab": GITLAB_STATE, "shopping_admin": SHOPPING_ADMIN_STATE}
OUT_ROOT_DEFAULT = os.path.join(paths.RESULTS, "runs")
import wa_workers as ww                                               # same-site parallelism (base run => identity)
MASK_SETTLE = 0.4
VIEWPORT = (1920, 1080)


def _chromium_exe():
    """Pick the NEWEST installed Playwright chromium build. Playwright's current default (chromium-1187) has a
    renderer-crash bug: gitlab's Monaco file-editor SIGSEGVs the renderer in ~3s (deterministic, GPU-independent);
    chromium-1228 renders it fine. Return the highest-numbered chromium-<N> binary; None => Playwright default."""
    import re
    base = os.getenv("PLAYWRIGHT_BROWSERS_PATH", "")
    best = (-1, None)
    if base and os.path.isdir(base):
        for d in os.listdir(base):
            m = re.fullmatch(r"chromium-(\d+)", d)
            if not m:
                continue
            for sub in ("chrome-linux64/chrome", "chrome-linux/chrome"):
                exe = os.path.join(base, d, sub)
                if os.path.exists(exe) and int(m.group(1)) > best[0]:
                    best = (int(m.group(1)), exe)
                    break
    return best[1]


CHROMIUM_EXE = _chromium_exe()


def _site(task):                        # "shopping-21" -> "shopping"; "reddit-67" -> "reddit"
    return task.rsplit("-", 1)[0]


def _login_cookies(task, log=print):
    """Login cookies for the task's SITE (added BEFORE first nav). Missing storage_state -> [] (anonymous)."""
    path = SITE_STATE.get(_site(task))
    if path:
        path = ww.worker_state_path(_site(task), path)               # per-worker auth file (base run => unchanged)
    if not path or not os.path.exists(path):
        log(f"[login] no storage_state for site '{_site(task)}' ({path}) -> running ANONYMOUS")
        return []
    return json.load(open(path, encoding="utf-8")).get("cookies", [])


def _seed_hosts():
    """Hosts the mask extension has rules for (seed 'site' keys). Lets the preflight assert rules WILL seed for
    this host WITHOUT inferring it from page content (a valid start page may legitimately have no untrusted
    content, e.g. reddit's empty homepage)."""
    try:
        txt = open(os.path.join(EXT_DIR, "seed.js"), encoding="utf-8").read()
        body = txt.split("__SMASK_SEED", 1)[1].split("=", 1)[1].strip().rstrip(";").strip()
        return {e.get("site") for e in json.loads(body)}
    except Exception:
        return set()

# The ONLY task data we author is this LIST of WebArena task ids to run. goal + start_url are loaded
# VERBATIM from the OFFICIAL resolved config (webarena_official/config_files/{task_id}.json), never
# hand-written — so nothing eval-relevant (goal / start_url / eval spec) originates with us.
WEBARENA_OFFICIAL = paths.WEBARENA_OFFICIAL
# Clean re-run campaign (post-masking-fix, 2026-07-24): 10 representative tasks per site (diverse task-kinds,
# spanning string_match=info / program_html=action / url_match=nav; chosen to exercise untrusted content).
TASK_IDS = [
    # reddit (:9999) — specs/reddit.rules.json
    "reddit-27", "reddit-66", "reddit-399", "reddit-404", "reddit-409",
    "reddit-580", "reddit-595", "reddit-605", "reddit-719", "reddit-725",
    # shopping storefront (:7770) — specs/shopping.rules.json
    "shopping-21", "shopping-163", "shopping-188", "shopping-269", "shopping-279",
    "shopping-324", "shopping-384", "shopping-465", "shopping-511", "shopping-585",
    # gitlab (:8023) — specs/gitlab.rules.json
    "gitlab-102", "gitlab-132", "gitlab-168", "gitlab-308", "gitlab-314",
    "gitlab-339", "gitlab-389", "gitlab-411", "gitlab-441", "gitlab-784",
    # shopping_admin (:7780) — specs/shopping_admin.rules.json
    "shopping_admin-0", "shopping_admin-112", "shopping_admin-198", "shopping_admin-208", "shopping_admin-243",
    "shopping_admin-458", "shopping_admin-464", "shopping_admin-470", "shopping_admin-694", "shopping_admin-704",
]


def _official_task(task):
    """goal + start_url loaded verbatim from the official resolved config; site port from WA_SITE_PORT."""
    cfg_path = os.path.join(WEBARENA_OFFICIAL, "config_files", f"{int(task.rsplit('-', 1)[1])}.json")
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"missing {cfg_path} — run webarena_official/scripts/generate_test_data.py first")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    return {"goal": cfg["intent"],                                   # per-site canonical port -> this worker's port (base => identity)
            "start_url": ww.rewrite_port(cfg["start_url"], _site(task))}


TASKS = {t: _official_task(t) for t in TASK_IDS}


def resolve_task(task):
    """Spec (goal/start_url) for a task id. Curated ids come from TASKS; ANY other valid WebArena id
    (<site>-<N>, e.g. produced by wa_site_tasks.py) resolves live from webarena_official/config_files/{N}.json."""
    spec = TASKS.get(task)
    if spec:
        return spec
    try:
        return _official_task(task)
    except Exception as e:
        raise RuntimeError(f"cannot resolve task {task!r} (not a curated id and no official config): {e}")


_KEYMAP = {"enter": "Enter", "return": "Enter", "tab": "Tab", "backspace": "Backspace", "esc": "Escape",
           "escape": "Escape", "space": "Space", " ": "Space", "delete": "Delete", "del": "Delete",
           "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown", "up": "ArrowUp",
           "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
           "arrowup": "ArrowUp", "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight"}
_MODS = {"ctrl": "Control", "control": "Control", "cmd": "Meta", "command": "Meta", "meta": "Meta",
         "alt": "Alt", "option": "Alt", "shift": "Shift"}


def _pwkey(k):
    """pyautogui-style key/combo -> Playwright key string (e.g. 'enter'->'Enter', 'ctrl+a'->'Control+a')."""
    k = str(k).strip()
    if "+" in k:
        out = []
        for p in [p.strip() for p in k.split("+") if p.strip()]:
            pl = p.lower()
            out.append(_MODS.get(pl, _KEYMAP.get(pl, p if len(p) == 1 else p.capitalize())))
        return "+".join(out)
    kl = k.lower()
    return _KEYMAP.get(kl, k if len(k) == 1 else k.capitalize())


def act_sig(op):
    """Coarse signature of an `act` op for repeat detection (carried over from the original secure loop)."""
    a = op.get("action")
    t = op.get("target") or {}
    try:
        if a == "click":
            if "mark" in t:
                return ("click", "mark", int(t["mark"]))
            if "x" in t and "y" in t:
                return ("click", round(float(t["x"]) / 25), round(float(t["y"]) / 25))
            return ("click", "?")
        if a == "type":
            return ("type", str(op.get("text", ""))[:40])
        if a == "press":
            return ("press", str(op.get("key", "enter")))
        if a == "scroll":
            return ("scroll", "down" if int(op.get("dy", -400)) < 0 else "up")
    except Exception:
        pass
    return (str(a),)


def run(task, max_steps, pause, out_root, provider="anthropic", model=MODEL, seed=None,
        benchmark="webarena", version=None):
    bm = get_benchmark(benchmark, version)                   # WebArena (Docker sites + harness) | realbench (Vercel + agisdk)
    spec = bm.resolve(task)
    goal, start_url = spec["goal"], spec["start_url"]

    _ml = (model or "").lower()
    shadow_provider = ("gemini" if _ml.startswith("gemini")
                       else "openai" if _ml.startswith(("gpt", "o1", "o3", "o4")) else "anthropic")
    out_dir = os.path.join(out_root, bm.out_dir_name(task, model, "secure"))
    if seed is not None:                                     # multi-seed run -> per-seed subdir (mirror in run_suite.py)
        out_dir = os.path.join(out_dir, f"seed{seed}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    profile_dir = os.path.join(out_dir, "_chrome_profile")           # FRESH profile -> seeds reddit rules from seed.js
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
    log(f"SECURE[{benchmark}]  task={task} model={model} provider={provider} shadow={shadow_provider}")
    log(f"goal: {goal}"); log(f"start_url: {start_url}")
    save_json("meta.json", {"task": task, "goal": goal, "start_url": start_url, "run_id": run_id,
                            "model": model, "provider": provider, "shadow_provider": shadow_provider,
                            "mode": "secure", "benchmark": benchmark, "env": bm.env_label(task),
                            "max_steps": max_steps, "screen_size": list(VIEWPORT), **bm.meta_extra(task)})

    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    ctx = pg = smask = None
    result = {"task": task, bm.result_verdict_key: None}
    ctx_state = {"cdp_port": None, "cdp_http": None}         # realbench: filled by extra_launch_args (--remote-debugging-port)
    t_task0 = time.perf_counter()
    meter = Meter(("actuator", "shadow"))
    op_trace = []
    user_answer = None                 # set by the terminal `return` op: the materialized answer for the user sink

    try:
        log("launch headed Chromium (Xvfb) + extension + cookies + goto ...")
        ctx = pw.chromium.launch_persistent_context(
            profile_dir, headless=False, executable_path=CHROMIUM_EXE,   # newest chromium (1187 crashes on gitlab's editor)
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
        bm.prepare(pg, cdp, ctx_state, task, run_id, log)            # realbench: agisdk config (seed the run); WebArena: no-op

        smask = Smask(page=pg, host_substr="", log=log)             # injected-page mode (no CDP connect)
        if not smask.wait_ready(timeout=20):                        # PREFLIGHT: extension MUST be live, else we'd screenshot UNMASKED
            raise RuntimeError("mask extension not live (#__smask_overlay absent) — refusing to run (would screenshot an unmasked page)")
        _host = pg.evaluate("() => location.host")                  # verify rules are SEEDED for this host (NOT inferred from page content:
        if _host not in _seed_hosts():                              #   a valid start page may legitimately have no untrusted content, e.g. reddit's empty homepage)
            raise RuntimeError(f"mask rules NOT seeded for host {_host} (absent from seed.js) — refusing to run (would screenshot an unmasked page)")
        smask.mask(); time.sleep(MASK_SETTLE)                       # apply masking (count is INFORMATIONAL — 0 is fine on a content-empty start page)
        _stamped = pg.eval_on_selector_all('[data-trust]', "e=>e.length")
        log(f"[preflight] mask extension live; rules seeded for {_host}; {_stamped} data-trust stamped on start page")
        log(f"[actuator] provider={provider} model={model}")
        actuator = Actuator(goal, provider=provider, model=model, log=log, meter=meter)
        shadow = Shadow(provider=shadow_provider, model=model, log=log, meter=meter)
        store = KStore()
        gstore = GStore()              # per-task opaque-payload store (name-addressed, survives navigation)
        pstate = {"queried": False, "warned": set()}
        A11Y_MAX = 5000

        def a11y(man, reveal=None):
            try:
                return get_masked_a11y(cdp, man, reveal_mark=reveal, viewport=VIEWPORT, max_chars=A11Y_MAX)
            except Exception as e:
                log(f"[a11y] failed: {e}"); return ""

        def _visible_untrusted():
            """# of untrusted/frame/obscured elements with a VISIBLE rect >3px — the population the extension
            actually boxes. Uses the SAME clip as content.js `visibleRect`: intersect the element's rect with
            every overflow-clipping ancestor AND the viewport. A raw getBoundingClientRect over-counts elements
            that an overflow container clips away, or that straddle the viewport edge with <3px actually showing —
            which the extension correctly does NOT box (they carry no visible pixels; they're a11y-masked
            regardless). Matching the clip keeps `covered` a true check (all visually-leakable untrusted IS boxed)
            without the spurious 'mask unverified' WARN + wasted settle iterations on dynamic/late-render grids.
            (Note: this only gates the settle loop; it never drives masking, so it cannot leak content.)"""
            try:
                return pg.eval_on_selector_all(
                    '[data-trust="untrusted"],[data-trust="frame"],[data-trust="obscured"]',
                    "els=>els.filter(e=>{const r=e.getBoundingClientRect();"
                    "let L=r.left,T=r.top,R=r.right,B=r.bottom;"
                    "for(let p=e.parentElement,d=0;p&&p!==document.documentElement&&d<24;p=p.parentElement,d++){"
                    "const s=getComputedStyle(p);"
                    "if(s.overflowX!=='visible'||s.overflowY!=='visible'){const pr=p.getBoundingClientRect();"
                    "L=Math.max(L,pr.left);T=Math.max(T,pr.top);R=Math.min(R,pr.right);B=Math.min(B,pr.bottom);}}"
                    "L=Math.max(L,0);T=Math.max(T,0);R=Math.min(R,innerWidth);B=Math.min(B,innerHeight);"
                    "return (R-L)>3&&(B-T)>3;}).length")
            except Exception:
                return 0

        def observe_vtop():
            # The runtime MUST guarantee V⊤ is fully masked before the frame reaches the actuator. The leak we
            # fix here: masking runs while the page/rules are not yet in place -> recompute() stamps 0 elements
            # -> 0 boxes AND 0 aria-hides (BOTH channels blank, since both branch off the same stamp) -> the old
            # `vis==0` break read that as a clean page and screenshotted the now-fully-loaded UNMASKED content
            # (this is what happened on reddit's subreddit listings). Deterministic order:
            #   (1) wait for the page to be LOAD-COMPLETE (load + networkidle) so its content is actually there;
            #   (2) confirm the engine's RULES are loaded -- a STATE check (data-smask-rules), NOT a "does the
            #       content look untrusted?" check (impossible pre-mask). A real content page with 0 loaded
            #       rules is a seed/load failure: reload once to re-inject, else fail-closed;
            #   (3) only THEN mask + verify + screenshot, and never trust `vis==0` until BOTH hold.
            def _load_complete():
                for _st, _to in (("load", 8000), ("networkidle", 5000)):
                    try: pg.wait_for_load_state(_st, timeout=_to)
                    except Exception: pass
                # networkidle fires BEFORE client-side JS paints (gitlab's Vue file-tree/commits, React SPAs), so
                # masking here would race the render -> late content lands UNMASKED in V⊤. Wait for the DOM to go
                # mutation-quiet (no changes for `q`ms, capped at `m`ms) so late content is present before we mask.
                try:
                    pg.evaluate("""(cfg)=>new Promise(res=>{let t=setTimeout(()=>{o.disconnect();res()},cfg.q);
                        const o=new MutationObserver(()=>{clearTimeout(t);t=setTimeout(()=>{o.disconnect();res()},cfg.q)});
                        o.observe(document.documentElement,{childList:true,subtree:true,characterData:true});
                        setTimeout(()=>{try{o.disconnect()}catch(e){};res()},cfg.m)})""", {"q": 700, "m": 4000})
                except Exception: pass
            def _benign():
                _u = (pg.url or "").strip()
                return _u in ("", "about:blank") or _u.startswith(("about:", "chrome://", "chrome-error://", "edge://"))
            _load_complete()
            rdy = smask.wait_ready(timeout=8)                        # a DEAD extension must NOT masquerade as an empty (vis==0) page
            if rdy and not _benign() and smask.rules_count() <= 0:   # engine live but 0 rules on a real content page = seed/load fail
                log(f"[observe_vtop] mask engine loaded 0 rules at {pg.url} — reloading to re-inject before screenshotting")
                try: pg.reload(wait_until="load", timeout=15000)
                except Exception: pass
                _load_complete(); rdy = smask.wait_ready(timeout=8)
                if smask.rules_count() <= 0:
                    raise RuntimeError(f"observe_vtop: 0 mask rules loaded at {pg.url} — refusing to screenshot a content page the masker can't cover")
            man = []; vis = 0; prev_vis = -1; prev_dom = -1
            for _ in range(10):
                if not rdy:                                         # extension may finish (re)injecting mid-loop (heavy pages / post-nav)
                    rdy = smask.wait_ready(timeout=1)
                smask.mask(); time.sleep(MASK_SETTLE)
                man = smask.manifest()
                vis = _visible_untrusted()
                try: dom = pg.evaluate("() => document.getElementsByTagName('*').length")   # total DOM size: detects still-ARRIVING
                except Exception: dom = -1                          # async content (Magento reviews AJAX-load on tab-click; admin grids
                rules_ok = _benign() or smask.rules_count() > 0     # are async Knockout) that vis (stamped-visible) hasn't reflected yet
                covered = (vis == 0 or len(man) >= vis)
                settled = (vis == prev_vis and dom == prev_dom)    # settle only when BOTH the untrusted set AND the DOM stopped growing
                prev_vis = vis; prev_dom = dom                     # => never break while lazy AJAX content is still loading in
                if rdy and rules_ok and covered and settled:       # break only when covered AND settled — never on the FIRST "covered"
                    break
                time.sleep(MASK_SETTLE)                              # late-rendered content -> loop re-masks/re-stamps
            else:                                                   # never converged in 10 tries
                _u = (pg.url or "").strip()                         # ONLY blank/browser-internal pages carry no maskable content
                if not rdy and not _benign():                      # live CONTENT page (http/https/data/file) w/o mask -> leak risk
                    raise RuntimeError(f"observe_vtop: mask extension not live at {_u} — refusing to screenshot an unmasked content page")
                if not rdy:                                         # blank page (e.g. go_back off the front of history to about:blank):
                    log(f"[observe_vtop] off-site/blank page {_u!r} — no maskable content; screenshotting as-is so the agent "  # allow it
                        "can leave the site and recover instead of hard-aborting")
                else:                                               # extension LIVE but coverage short (frame-collapse/late render)
                    log(f"[observe_vtop] WARN mask unverified at {pg.url} ({vis} untrusted visible, {len(man)} boxed after 10 tries) "
                        "-- best-effort screenshot (frame-collapse or stubborn late render)")
            smask.mask(); time.sleep(MASK_SETTLE)                   # FINAL re-stamp IMMEDIATELY before capture: closes the race where
            man = smask.manifest()                                  # async content (AJAX reviews / admin grid) appeared during the settle,
            for r in man:                                           # so both the screenshot AND the a11y reflect a freshly-masked DOM.
                r["sx"], r["sy"] = r["x"], r["y"]                   # VIEWPORT-native: render_marks + mark-click use x,y
            store.set_marks([r["n"] for r in man])
            png = pg.screenshot()
            return png, man, a11y(man)

        def do_query(op, turn, manifest, cur_url):
            rho, over = op.get("rho", ""), op.get("over", [])
            T, bind, enum_values = op.get("return_type"), op.get("bind"), op.get("enum_values")
            if isinstance(over, (str, int)):
                over = [over]
            try:
                over = [int(m) for m in (over or [])]
            except (TypeError, ValueError):
                return "query rejected: `over` must be a list of mark numbers, e.g. [3,9,15]."
            if T not in ("bool", "enum", "int", "decimal", "payload") or not bind or not over:
                return "query rejected: need rho, over (mark #s), return_type (bool|enum|int|decimal|payload), bind."
            if T == "enum" and not enum_values:
                return "query rejected: enum requires enum_values."
            if T == "payload":
                gstore.declare(bind)            # opaque strings accumulate in a named G handle, not K
            else:
                store.declare(bind, T)
            valid = {r["n"] for r in manifest}
            vgs, a11y_vg = {}, {}
            for m in over:
                m = int(m)
                if m not in valid:
                    continue
                try:
                    smask.isolate(m); time.sleep(MASK_SETTLE)
                    vgs[m] = pg.screenshot()
                    a11y_vg[m] = a11y(manifest, reveal=m)            # single-reveal masked a11y for the shadow
                    save(f"turn_{turn}_mark{m}_vg.png", vgs[m])
                    save_text(f"turn_{turn}_mark{m}_a11y.txt", a11y_vg[m])
                except Exception as e:
                    log(f"[query] isolate {m} failed: {e}")
            smask.remask(); time.sleep(MASK_SETTLE)

            def probe(m):
                return m, shadow.ask(b64(vgs[m]), rho, T, enum_values, a11y_vg.get(m, ""))
            tags = {}
            if vgs:
                with ThreadPoolExecutor(max_workers=8) as ex:
                    for m, res in ex.map(probe, list(vgs.keys())):
                        tags[m] = res["outcome"]
                        if res["outcome"] == P.SUCCESS:
                            if T == "payload":
                                gstore.append(bind, res["value"], {"url": cur_url, "mark": m, "rho": rho})
                            else:
                                store.write(bind, m, res["value"])
            miss = [int(m) for m in over if int(m) not in valid]
            parts = ", ".join(f"#{int(m)}:{tags.get(int(m), 'error')}" for m in over if int(m) in valid)
            note = f" (skipped not-on-screen: {miss})" if miss else ""
            pstate["queried"] = True
            if T == "payload":
                return (f"payload query '{bind}' done. Per-mark outcomes: {parts}{note}. Each 'answered' mark "
                        f"stored an OPAQUE string into handle '{bind}' (in G) — you cannot read it or compute on "
                        f"it; route it to the user with `return`. Reuse bind='{bind}' to add more marks to it.")
            return (f"query '{bind}' ({T}) done. Per-mark outcomes: {parts}{note}. "
                    f"'answered' only means a value was stored (true OR false) — it does NOT mean yes. The values "
                    f"are hidden; you MUST `compute read`/`filter` on column '{bind}' to see what they actually are "
                    f"before deciding or acting.")

        def do_compute(op):
            args = op.get("args")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return ("compute rejected: `args` was not valid JSON — pass a JSON object like "
                            "{\"eq\":true} or {\"op\":\"<\",\"v\":100}, then re-issue the compute.")
            if args is not None and not isinstance(args, dict):
                return "compute rejected: `args` must be a JSON object, e.g. {\"eq\":true}. Re-issue the compute."
            try:
                res = compute(store, op.get("select"), op.get("operation"), args)
            except Exception as e:
                return f"compute rejected: {e}. Check select (a K column), operation, and args, then re-issue."
            if res.get("kind") == "error":
                return f"compute error: {res['error']}"
            return f"compute {op.get('operation')}({op.get('select')}) -> {json.dumps(res, default=str)}"

        def do_act(op, turn, manifest, cur_url):
            action = op.get("action")
            on_aff = True
            if action == "click":
                target = op.get("target") or {}
                if "mark" in target:
                    r = next((x for x in manifest if x["n"] == int(target["mark"])), None)
                    if not r:
                        return {"feedback": f"act refused: mark {target['mark']} not on screen.", "advanced": False, "reason": "mark-not-on-screen"}
                    px, py = r["sx"] + r["w"] // 2, r["sy"] + r["h"] // 2   # sx,sy remapped to viewport in observe_vtop
                elif "x" in target and "y" in target:
                    px, py = int(target["x"]), int(target["y"])
                else:
                    return {"feedback": "act refused: click needs target {x,y} or {mark}.", "advanced": False, "reason": "no-target"}
                v = smask.egress(px, py, viewport=True)              # CFI gate (viewport coords)
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
                cc = smask.click_context(px, py, viewport=True)      # verify-before-action (graduated) gate
                near = (([int(cc["self"])] if cc.get("self") is not None else [])
                        + [int(m) for m in (cc.get("context") or [])])
                if near:
                    _q = lambda m: any(m in c["cells"] for c in store.columns.values())
                    if not any(_q(m) for m in near):
                        if not pstate["queried"]:
                            return {"feedback": f"act REFUSED (verify-before-action): you have queried NOTHING on this "
                                                f"screen, so you would be acting BLIND on masked content (marks {near}). "
                                                f"Query at least one relevant mark first (e.g. `query over=[{near[0]}], "
                                                f"return_type bool, rho='is this <what you expect>?'`) and `compute read` "
                                                f"it, then act.", "advanced": False, "reason": "verify-before-action"}
                        sig = act_sig(op)
                        if sig not in pstate["warned"]:
                            pstate["warned"].add(sig)
                            return {"feedback": f"act WARNING (verify): this click touches masked content (marks {near}) "
                                                f"you have not specifically queried — you HAVE queried elsewhere on this "
                                                f"screen. If what that content IS matters here, query one of {near} and "
                                                f"`compute read` it first. If you are confident it is right, repeat this "
                                                f"EXACT click to proceed.", "advanced": False, "reason": "verify-warning"}
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
                dy = int(op.get("dy", -400))                        # negative = DOWN (pyautogui convention)
                tg = op.get("target") or {}
                sx, sy = int(tg.get("x", VIEWPORT[0] // 2)), int(tg.get("y", VIEWPORT[1] // 2))
                pg.mouse.move(sx, sy)
                pg.mouse.wheel(0, -dy)                              # Playwright wheel: +deltaY = down -> negate dy
                acted = f"scrolled {dy} at ({sx},{sy})"
            elif action == "go_back":                               # browser history back (no egress: agent-initiated nav to a prior first-party page)
                try:
                    pg.go_back(wait_until="load", timeout=8000)
                except Exception as e:
                    log(f"[act] go_back: {e}")
                acted = "went back (browser history)"
            else:
                return {"feedback": f"act refused: unknown action {action!r}.", "advanced": False, "reason": "unknown-action"}
            log(f"[act] {acted}")
            time.sleep(0.6)
            vtop, man, a11y_new = observe_vtop()                    # re-observe (settle+remask) THEN read url
            new_url = pg.url
            navigated = new_url != cur_url
            if navigated:
                store.discard()
                pstate["queried"] = False; pstate["warned"].clear()
            save(f"turn_{turn}_after.png", vtop)
            return {"feedback": acted + (" — NAVIGATED (K reset)." if navigated else " — screen re-observed."),
                    "advanced": True, "vtop": vtop, "manifest": man, "a11y": a11y_new, "new_url": new_url,
                    "on_affordance": on_aff}

        def do_return(op):
            """Materialize opaque payload handles into the user answer (declassification, display-only).
            Substitutes {slot}->handle bytes AT THE USER SINK; the bytes NEVER enter feedback / the actuator.
            Terminal on success. Rejects unknown / non-G handles."""
            answer = op.get("answer", "")
            use = op.get("use") or []
            status = op.get("status", "success")
            if isinstance(use, str):                         # some providers serialize `use` as a JSON string
                try:
                    use = json.loads(use)
                except Exception:
                    use = None
            if not isinstance(answer, str) or not answer.strip():
                return {"ok": False, "err": "return rejected: need a non-empty `answer` template (your own text)."}
            if use is None or not isinstance(use, list):
                return {"ok": False, "err": "return rejected: `use` must be a JSON array of {slot,handle}."}
            subs = {}
            for u in use:
                if not isinstance(u, dict) or not u.get("slot") or not u.get("handle"):
                    return {"ok": False, "err": "return rejected: each `use` item must be {\"slot\":..,\"handle\":..}."}
                h = u["handle"]
                if not gstore.has(h):
                    return {"ok": False, "err": f"return rejected: no payload handle '{h}' in G — only "
                            f"return_type=payload results can be returned. Available handles: "
                            f"{[c['name'] for c in gstore.schema()]}."}
                subs[u["slot"]] = gstore.materialize(h) or ""    # runtime-only; bytes never surface to feedback
            final = answer
            for slot, val in subs.items():
                final = final.replace("{" + slot + "}", val)
            return {"ok": True, "answer": final, "status": status}

        # ---- the loop ----
        vtop_png, manifest, a11y_top = observe_vtop()
        cur_url = pg.url
        feedback, turn, done = "", 0, False
        act_log = []
        nochange_streak = 0
        while not done and turn < max_steps:
            turn += 1
            op, raw = actuator.turn(b64(vtop_png), P.render_marks(manifest), a11y_top,
                                    P.render_ktable_schema(store), P.render_gstore_schema(gstore), feedback)
            save(f"turn_{turn}_vtop.png", vtop_png); save_json(f"turn_{turn}_op.json", op); save_text(f"turn_{turn}_raw.txt", raw)
            save_text(f"turn_{turn}_a11y.txt", a11y_top)
            kind = op.get("op")
            log(f"--- turn {turn}: {kind} --- {json.dumps(op, default=str)[:220]}")
            if kind == "query":
                feedback = do_query(op, turn, manifest, cur_url)
            elif kind == "compute":
                feedback = do_compute(op)
            elif kind == "act":
                prev_vtop = vtop_png
                res = do_act(op, turn, manifest, cur_url)
                feedback = res["feedback"]
                if res.get("advanced"):
                    new_url = res.get("new_url", cur_url)
                    navigated = new_url != cur_url
                    changed = navigated or frac_changed(prev_vtop, res["vtop"]) > 0.002
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
                              f"repeating it will NOT work. Do something DIFFERENT (a different target/op, or query "
                              f"the mark named below).")
                elif op.get("action") == "click" and total_same >= 3:
                    banner = (f"⚠ You have targeted this same spot {total_same} times. If it is not advancing the "
                              f"task, it is not the right control — try a DIFFERENT element (a different control, or "
                              f"open a masked item by its {{mark:N}}).")
                if banner:
                    feedback = banner + "\n\nDetails:\n" + feedback
            elif kind == "return":                              # the sole terminal op
                rr = do_return(op)
                if rr["ok"]:
                    user_answer = rr["answer"]; result["end_status"] = rr.get("status", "success")
                    op_trace.append({"turn": turn, "op": op})
                    save_text("model_response.txt", user_answer)      # user sink (materialized bytes); NEVER feedback
                    log(f"[return] status={rr.get('status')} answer ({len(user_answer)} chars) -> user sink")
                    done = True
                    break
                feedback = rr["err"]
            else:
                feedback = f"Your output was not a valid op ({op.get('error')}). Emit exactly ONE JSON op (query|compute|act|return)."
            log(f"  feedback: {feedback[:200]}")
            op_trace.append({"turn": turn, "op": op, "feedback": feedback[:500]})
            save_json(f"K_turn_{turn}.json", store.dump())
            save_json(f"G_turn_{turn}.json", gstore.dump())          # payload bytes — artifacts only, never to actuator
            save_json("op_trace.json", op_trace)

        # ---- score via the benchmark scorer (WebArena harness OR agisdk over the CDP endpoint) ----
        final_url = pg.url
        save_json("final_url.json", {"final_url": final_url})
        # `return` (the sole terminal op) already wrote the materialized answer inline; fall back to the last
        # feedback only if the agent never returned (e.g. hit the step budget).
        save_text("model_response.txt", user_answer if user_answer is not None else feedback)
        result["turns"] = turn
        result["final_url"] = final_url
        result["end_answer"] = user_answer
        log(f"settling 3s, then scoring [{benchmark}] ...")
        time.sleep(3)
        result.update(bm.score(task, out_dir, final_url, user_answer, feedback, ctx_state, run_id, log))
        result["metrics"] = meter.snapshot(time.perf_counter() - t_task0)
        save_json("result.json", result)
        log(f"EVAL[{benchmark}]: {result.get(bm.result_verdict_key)}  (turns={turn}, final_url={final_url})")

    except Exception as e:
        import traceback
        log(f"EXCEPTION: {e}\n{traceback.format_exc()}")
        result["error"] = str(e)
        result["metrics"] = meter.snapshot(time.perf_counter() - t_task0)
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
