
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

"""No-ADC pre-flight for the WebArena secure driver (checks #3/#4/#5). Run under Xvfb:
   xvfb-run -s "-screen 0 1920x1200x24" python preflight_webarena.py
Validates: launch+ext+cookies+mask+manifest+screenshot dims (#3); egress viewport-flag admit/refuse/
off-viewport (#4); masked-a11y non-empty + 0-leak + a clickable control present (#5). No LLM/ADC used.
"""
import io
import json
import os
import paths
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playwright.sync_api import sync_playwright  # noqa: E402
from PIL import Image  # noqa: E402
from smask_runtime import Smask  # noqa: E402
from wa_a11y import get_masked_a11y  # noqa: E402

ROOT = paths.SGE_ROOT
EXT = paths.EXT_DIR
STATE = paths.WA_STATE["reddit"]
POST = "http://localhost:9999/f/books/124260/adults-reading-to-each-other-out-loud"
SHOTS = os.path.join(paths.RESULTS, "_probe_shots")
VP = (1920, 1080)

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        "/tmp/wa_preflight", headless=False,
        args=["--no-sandbox", "--disable-dev-shm-usage",
              f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"],
        viewport={"width": VP[0], "height": VP[1]}, device_scale_factor=1)
    ctx.add_cookies(json.load(open(STATE))["cookies"])
    pg = ctx.pages[0] if ctx.pages else ctx.new_page()
    cdp = ctx.new_cdp_session(pg)
    pg.goto(POST, wait_until="load"); pg.wait_for_timeout(1500)

    sm = Smask(page=pg, host_substr="")
    check("#3 extension wait_ready", sm.wait_ready(timeout=20))
    sm.mask(); pg.wait_for_timeout(500)
    man = sm.manifest()
    for r in man:
        r["sx"], r["sy"] = r["x"], r["y"]
    check("#3 manifest non-empty", len(man) > 0, f"({len(man)} marks)")
    check("#3 manifest rows have x,y+sx,sy", all(all(k in r for k in ("x", "y", "sx", "sy", "w", "h")) for r in man))
    png = pg.screenshot(); open(os.path.join(SHOTS, "preflight_masked.png"), "wb").write(png)
    dim = Image.open(io.BytesIO(png)).size
    check("#3 screenshot is 1920x1080", dim == VP, str(dim))
    ntrust = pg.eval_on_selector_all('[data-trust="untrusted"]', "e=>e.length")
    check("#3 data-trust untrusted count > 0 (rules seeded)", ntrust > 0, f"({ntrust})")

    # ---- #4 egress viewport-flag ----
    vote = pg.evaluate("""()=>{const el=document.querySelector('div.submission__vote button')||document.querySelector('div.submission__vote a');
      if(!el)return null;const r=el.getBoundingClientRect();return {x:Math.round(r.left+r.width/2),y:Math.round(r.top+r.height/2)};}""")
    if vote:
        e = sm.egress(vote["x"], vote["y"], viewport=True)
        check("#4 egress ADMITS the vote control", bool(e.get("admit")) and e.get("reason") == "first-party-affordance",
              f"({e.get('reason')}, box={e.get('box')})")
    else:
        check("#4 vote control found", False)
    noaff = pg.evaluate("""()=>{const AFF='a[href],button,[role=button],[role=link],input,select,textarea';
      const els=[...document.querySelectorAll('[data-trust=\"untrusted\"]')].filter(e=>!e.closest(AFF));
      for(const e of els){const r=e.getBoundingClientRect();if(r.width>6&&r.height>6&&r.top>=0&&r.top<1070)
        return {x:Math.round(r.left+r.width/2),y:Math.round(r.top+r.height/2)};}return null;}""")
    if noaff:
        e = sm.egress(noaff["x"], noaff["y"], viewport=True)
        check("#4 egress REFUSES a masked non-affordance", (not e.get("admit")) and e.get("reason") == "untrusted-non-affordance",
              f"({e.get('reason')})")
    else:
        check("#4 masked non-affordance found", False)
    off = sm.egress(500, 1200, viewport=True)
    check("#4 egress refuses off-viewport (y=1200)", (not off.get("admit")) and off.get("reason") == "off-viewport", f"({off.get('reason')})")

    # ---- #5 masked a11y: non-empty, 0-leak, a control present ----
    a11 = get_masked_a11y(cdp, man, viewport=VP)
    open(os.path.join(SHOTS, "preflight_a11y.txt"), "w").write(a11)
    check("#5 masked a11y non-empty", len(a11) > 50, f"({len(a11)} chars)")
    # 0-leak: no MASKED text content (>=5 chars) may appear in the a11y string
    masked_txt = pg.eval_on_selector_all('[data-trust="untrusted"]',
        "els=>[...new Set(els.map(e=>(e.textContent||'').trim()).filter(s=>s.length>=5))]")
    leaked = [s for s in masked_txt if s in a11][:5]
    check("#5 masked a11y has ZERO untrusted-content leaks", len(leaked) == 0, f"leaked={leaked}")
    check("#5 masked a11y exposes clickable controls (coords present)", "@" in a11 and ("button" in a11.lower() or "link" in a11.lower()))
    ctx.close()

print("\nPREFLIGHT #3/#4/#5:", "ALL PASS" if not fails else f"{len(fails)} FAILED -> {fails}")
sys.exit(1 if fails else 0)
