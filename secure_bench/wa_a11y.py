
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

"""Masked, linearized accessibility tree for the WebArena/Playwright secure driver. Source = Chrome CDP
AXTree (`Accessibility.getFullAXTree`), matching omnizon's "feed the actuator a masked a11y tree" design.

Masking has TWO layers:
  (1) The extension's `maskAria` sets aria-hidden (+ blanks aria-label) on every untrusted region in the
      CURRENT mask/isolate view, so those nodes come back `ignored` from getFullAXTree and are dropped here.
  (2) Belt-and-suspenders: any kept node whose viewport box is CONTAINED in a masked mark's box (from the
      manifest) is scrubbed — catching any residual name that survived aria-hidden name-computation.

The CALLER sets the view: `smask.mask()` -> trusted V⊤ a11y (all untrusted hidden); `smask.isolate(m)` ->
only mark m un-hidden = the shadow's single-reveal a11y. `reveal_mark` here just keeps that one mark's box
out of the scrub set. Boxes are `getBoundingClientRect` (VIEWPORT px; with device_scale_factor=1 this
equals page.screenshot() pixels and the page.mouse click frame — no coordinate desync).
"""

# roles whose element the actuator may click -> we resolve a viewport box so it can ground on the a11y center
INTERACTIVE = {"button", "link", "textbox", "searchbox", "checkbox", "radio", "combobox", "listbox", "tab",
               "menuitem", "menuitemcheckbox", "menuitemradio", "switch", "slider", "spinbutton", "option"}
_SKIP = {"", "generic", "none", "presentation", "InlineTextBox", "LineBreak"}

_RECT_FN = ("function(){var r=this.getBoundingClientRect();"
            "return [Math.round(r.x),Math.round(r.y),Math.round(r.width),Math.round(r.height)];}")


def _rect(cdp, backend_id):
    """Viewport-frame getBoundingClientRect of a backend DOM node, via CDP resolve + callFunctionOn."""
    try:
        ro = cdp.send("DOM.resolveNode", {"backendNodeId": backend_id})
        oid = (ro.get("object") or {}).get("objectId")
        if not oid:
            return None
        r = cdp.send("Runtime.callFunctionOn",
                     {"objectId": oid, "functionDeclaration": _RECT_FN, "returnByValue": True})
        v = (r.get("result") or {}).get("value")
        return v if (v and len(v) == 4 and v[2] > 0 and v[3] > 0) else None
    except Exception:
        return None


def _inside(b, m, tol=2):
    return (b[0] >= m[0] - tol and b[1] >= m[1] - tol
            and b[0] + b[2] <= m[0] + m[2] + tol and b[1] + b[3] <= m[1] + m[3] + tol)


_MASKED_TXT_FN = """() => {
  const out = [];
  document.querySelectorAll('[data-trust="untrusted"],[data-trust="obscured"]').forEach(e => {
    const n = parseInt(e.getAttribute('data-smask-n') || '0', 10);
    const t = (e.textContent || '').trim();
    if (t.length >= 5) out.push([n, t]);
    const al = (e.getAttribute('aria-label') || '').trim(); if (al.length >= 5) out.push([n, al]);
  });
  return out;
}"""


def _masked_strings(cdp, reveal_mark):
    """The untrusted text (the 'lexicon') to scrub from the a11y.
    A trusted node whose ACCESSIBLE NAME is computed from a masked child (e.g. an <h1> over a masked <a>)
    survives aria-hidden, so we drop any node whose name CONTAINS a masked string. Excludes the revealed
    mark (V_g) so the shadow still sees that one region's content."""
    try:
        r = cdp.send("Runtime.evaluate", {"expression": f"({_MASKED_TXT_FN})()", "returnByValue": True})
        pairs = (r.get("result") or {}).get("value") or []
    except Exception:
        return []
    rs = reveal_mark if isinstance(reveal_mark, set) else ({int(reveal_mark)} if reveal_mark is not None else set())
    return [t for (n, t) in pairs if int(n) not in rs and t]


def get_masked_a11y(cdp, manifest, reveal_mark=None, viewport=(1920, 1080), max_chars=5000, cap_rects=160):
    """Return a masked, linearized a11y string for the CURRENT (already-masked) view.
    cdp: a Playwright CDPSession (page.context.new_cdp_session(page)). manifest: viewport-coord mark rows."""
    try:
        cdp.send("DOM.enable")
        cdp.send("Accessibility.enable")
        tree = cdp.send("Accessibility.getFullAXTree")
    except Exception:
        return ""
    VW, VH = viewport
    rm = int(reveal_mark) if reveal_mark is not None else None
    reveal_set = set()                                       # V_g: reveal isolated mark + any mask SUBSUMING it (mirrors content.js; over-reveal only in untrusted view)
    if rm is not None:
        reveal_set.add(rm)
        rmbox = next(((r["x"], r["y"], r["w"], r["h"]) for r in (manifest or []) if int(r.get("n", -1)) == rm), None)
        if rmbox is not None:
            for r in (manifest or []):
                n = int(r.get("n", -1))
                if n != rm and _inside(rmbox, (r["x"], r["y"], r["w"], r["h"])):
                    reveal_set.add(n)
    masked = [(r["x"], r["y"], r["w"], r["h"]) for r in (manifest or []) if int(r.get("n", -1)) not in reveal_set]
    masked_txt = _masked_strings(cdp, reveal_set)            # lexicon to scrub, minus the whole reveal set
    lines, nrect = [], 0
    for nd in tree.get("nodes", []):
        if nd.get("ignored"):
            continue                                          # aria-hidden (incl. masked untrusted) -> excluded
        role = (nd.get("role") or {}).get("value") or ""
        name = ((nd.get("name") or {}).get("value") or "").strip()
        interactive = role in INTERACTIVE
        if role in _SKIP and not interactive:
            continue
        if not name and not interactive:
            continue
        if name and any(ms in name for ms in masked_txt):     # name leaks masked content (e.g. h1 over a masked link) -> drop
            continue
        box = None
        bid = nd.get("backendDOMNodeId")
        if interactive and bid and nrect < cap_rects:        # resolve a box only for clickable controls
            nrect += 1
            box = _rect(cdp, bid)
            if box and not (0 <= box[0] < VW and -40 < box[1] < VH + 40):
                box = None                                    # off-screen -> keep the line, drop the coord
            if box and any(_inside(box, m) for m in masked):
                continue                                      # belt-and-suspenders: inside a masked mark -> scrub
        coord = f" @{box[0]},{box[1]} {box[2]}x{box[3]}" if box else ""
        nm = name[:90].replace("\n", " ")
        lines.append((f'{role} "{nm}"' if name else role) + coord)
        if sum(len(x) + 1 for x in lines) > max_chars:
            break
    return "\n".join(lines)[:max_chars]
