
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

"""(OPTIONAL) READ-ONLY masking coverage probe — dumps masked screenshots of each task's pages so a human can
confirm the answer-bearing untrusted content is actually covered by the shipped rules.

Generalizes preflight_webarena.py to any site (per-task login + start_url from the official config). It ONLY
observes: applies the ALREADY-SHIPPED masks, counts data-trust stamping, checks the masked a11y for leaks, and
saves a masked PNG. It NEVER modifies any marking rule (does not touch specs/*.rules.json or seed.js). Anything
left unmasked in a PNG is a labeling gap to DECIDE ON manually — this script changes nothing.

   xvfb-run -a -s "-screen 0 1920x1200x24" python coverage_webarena.py --tasks reddit-67 shopping-21 shopping-387
"""
import argparse
import os
import paths
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playwright.sync_api import sync_playwright  # noqa: E402
from smask_runtime import Smask  # noqa: E402
from wa_a11y import get_masked_a11y  # noqa: E402
from run_secure_webarena import TASKS, _login_cookies, _site, EXT_DIR, VIEWPORT  # noqa: E402

ROOT = paths.SGE_ROOT
SHOTS = os.path.join(paths.RESULTS, "_probe_shots", "coverage")
os.makedirs(SHOTS, exist_ok=True)


def _mask_report(pg, cdp, sm, label):
    sm.mask(); pg.wait_for_timeout(600)
    sm.mask(); pg.wait_for_timeout(400)                                  # 2nd pass catches AJAX/late-rendered content
    man = sm.manifest()
    for r in man:
        r["sx"], r["sy"] = r["x"], r["y"]
    stamped = pg.eval_on_selector_all('[data-trust]', "e=>e.length")
    vis = pg.eval_on_selector_all(
        '[data-trust="untrusted"],[data-trust="frame"],[data-trust="obscured"]',
        "els=>els.filter(e=>{const r=e.getBoundingClientRect();"
        "return r.width>3&&r.height>3&&r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth;}).length")
    png = pg.screenshot()
    path = os.path.join(SHOTS, f"{label}.png")
    open(path, "wb").write(png)
    a11 = get_masked_a11y(cdp, man, viewport=VIEWPORT) if man else ""
    masked_txt = pg.eval_on_selector_all('[data-trust="untrusted"]',
        "els=>[...new Set(els.map(e=>(e.textContent||'').trim()).filter(s=>s.length>=5))]")
    leaked = [s for s in masked_txt if s in a11][:5]
    flag = "" if len(man) >= vis and not leaked else "  <-- REVIEW"
    print(f"  [{label}] stamped={stamped} visible_untrusted={vis} boxes={len(man)} a11y_leaks={len(leaked)} -> {path}{flag}")
    if leaked:
        print(f"    a11y leak sample: {leaked}")


def probe(task, log=print):
    spec = TASKS[task]
    site = _site(task)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            f"/tmp/wa_coverage_{task}", headless=False,
            args=["--no-sandbox", "--disable-dev-shm-usage",
                  f"--disable-extensions-except={EXT_DIR}", f"--load-extension={EXT_DIR}"],
            viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]}, device_scale_factor=1)
        cookies = _login_cookies(task, log)
        if cookies:
            ctx.add_cookies(cookies)
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        cdp = ctx.new_cdp_session(pg)
        pg.goto(spec["start_url"], wait_until="load"); pg.wait_for_timeout(1500)
        sm = Smask(page=pg, host_substr="")
        if not sm.wait_ready(timeout=20):
            print(f"  [{task}] EXTENSION NOT LIVE — cannot probe"); ctx.close(); return
        _mask_report(pg, cdp, sm, f"{task}_start")
        if site == "shopping":                                          # open the AJAX Reviews tab so review masks appear
            try:
                tab = pg.query_selector("#tab-label-reviews-title") or pg.query_selector("a[href='#reviews']")
                if tab:
                    tab.scroll_into_view_if_needed(); tab.click(); pg.wait_for_timeout(2500)
                    _mask_report(pg, cdp, sm, f"{task}_reviews")
                else:
                    print(f"  [{task}] no Reviews tab on start page (search/homepage task) — start PNG only")
            except Exception as e:
                print(f"  [{task}] reviews-tab step skipped: {e}")
        ctx.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    a = ap.parse_args()
    print(f"COVERAGE PROBE (READ-ONLY — changes NO rules) -> PNGs in {SHOTS}")
    for t in a.tasks:
        if t not in TASKS:
            print(f"  unknown task {t} (known: {list(TASKS)})"); continue
        print(f"--- {t}  ({_site(t)}) ---")
        probe(t)
    print("\nReview the masked PNGs by eye. Anything answer-bearing that is NOT blacked out is a labeling "
          "gap to decide on manually — this probe changed nothing.")


if __name__ == "__main__":
    main()
