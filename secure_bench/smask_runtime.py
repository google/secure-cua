
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

"""Runtime control of the Secure-Mask browser extension, driven from the HOST over CDP.

The extension (mask_extension/, loaded into the headed Chromium) stamps `data-trust` on the live
DOM from the committed omnizon rules and, when masking is ON, paints black set-of-marks boxes
over every untrusted element (+ hides them in the a11y tree). Masking is controlled through a
KEY-GATED DOM channel that only the runtime can reach:

  * The runtime sets  <html data-smask-key="R">  and dispatches a `__smask_cmd` CustomEvent
    carrying the same R. content.js (extension, isolated world) validates d.key against the
    <html> attribute before acting. Both live in the shared DOM, so the isolated-world listener
    can read the key the main-world runtime set.
  * The AGENT emits only pixels (pyautogui) — no JS — so it can neither set the attribute nor
    dispatch the event. Untrusted page CONTENT (injected text/images) has no script execution
    either. Hence only the Playwright/CDP runtime drives mask/reveal. (Residual: untrusted
    MAIN-world JS could read the attribute — out of scope; pages are the immutable first-party
    build and the threat is content injection, not XSS.)

Actions:
  mask()   -> 'maskAll'   : masked=true, then RE-STAMP (catches lazy-loaded / post-action DOM)
                            and re-render. Call right before every ACTUATOR screenshot.
  reveal() -> 'revealAll' : masked=false, drop every mask -> clean unmasked view. Call right
                            before a SHADOW screenshot (untrusted read).
  remask() -> 'maskAll'   : restore masking after the shadow call.

Runs in-process in the primary conda env (playwright present); the injected page-side masking is driven
over CDP against the live Chromium.
"""
import secrets
import time
import xml.etree.ElementTree as ET

# playwright is imported lazily in connect() so this module (pure stdlib) is importable without it


# main-world JS: stamp the session key + action on <html> (read cross-world by content.js) and fire
# the trigger event. Kept tiny and idempotent so it is safe to call before every screenshot.
_DISPATCH_JS = """([k, action]) => {
  const h = document.documentElement;
  h.setAttribute('data-smask-key', k);
  h.setAttribute('data-smask-cmd', action);
  document.dispatchEvent(new CustomEvent('__smask_cmd'));
  return { overlay: !!document.getElementById('__smask_overlay') };
}"""

# extension readiness: content.js appends #__smask_overlay to <html> (visible from main world).
_READY_JS = "() => !!document.getElementById('__smask_overlay')"

# the numbered-region manifest content.js stamps on <html> each render (no untrusted content —
# only contract placeholders + viewport & SCREEN rects), used to mask the a11y tree host-side.
_MANIFEST_JS = "() => { const m = document.documentElement.getAttribute('data-smask-manifest'); return m || '[]'; }"

# V_g isolation: reveal ONLY mark n (content.js 'isolate' view masks every other region, both channels).
_ISOLATE_JS = """([k, n]) => {
  const h = document.documentElement;
  h.setAttribute('data-smask-key', k);
  h.setAttribute('data-smask-mark', String(n));
  h.setAttribute('data-smask-cmd', 'isolateOne');
  document.dispatchEvent(new CustomEvent('__smask_cmd'));
  return { overlay: !!document.getElementById('__smask_overlay') };
}"""

# egress / CFI gate: at a SCREEN coordinate, resolve the affordance and decide admit/refuse. Self-
# contained main-world fn (it cannot see content.js's closure). The mask overlay is pointer-events:none,
# so elementFromPoint reads the real page affordance underneath. Same AFF set as content.js.
_EGRESS_JS = """([x, y, vp]) => {
  const AFF = 'button,a[href],input,select,textarea,[role=button],[role=link],[role=tab],[role=checkbox],[role=radio],[role=combobox],[role=textbox],[role=menuitem],[role=switch],[role=slider],[role=option]';
  const ox = vp ? 0 : (window.screenX || 0), oy = vp ? 0 : ((window.screenY || 0) + (window.outerHeight - window.innerHeight));
  const vx = x - ox, vy = y - oy;                                 // vp=true: (x,y) already viewport; else screen->viewport (mirror content.js scrOX/scrOY)
  if (vx < 0 || vy < 0 || vx > window.innerWidth || vy > window.innerHeight)
    return { admit: false, reason: 'off-viewport', vx, vy };
  const el = document.elementFromPoint(vx, vy);
  if (!el) return { admit: false, reason: 'no-element', vx, vy };
  if (el.closest('[data-trust="frame"]')) return { admit: false, reason: 'regime2-frame' };   // attacker-authored structure
  const a = el.closest('a[href]');                                // OFF-origin nav is refused regardless of author
  let origin = 'SAME', href = null;
  if (a) { href = a.getAttribute('href'); try { origin = new URL(href, location.href).origin === location.origin ? 'SAME' : 'OFF'; } catch (e) { origin = 'OFF'; } }
  if (origin === 'OFF') return { admit: false, reason: 'off-origin', href, vx, vy };
  const aff = el.closest(AFF);
  const untrusted = el.closest('[data-trust="untrusted"],[data-trust="obscured"]');
  // Admit iff the click lands on (or inside) a first-party affordance — the SAME `el.closest(AFF)` the
  // labeler uses to tag a masked region CLICKABLE (content.js). So egress and the mask agree by
  // construction: a masked region is a valid click target exactly when the mask marks it CLICKABLE; other
  // masked content is not a click target. (Off-origin nav and data-trust=frame are already refused above.)
  if (!aff && untrusted) return { admit: false, reason: 'untrusted-non-affordance', vx, vy };
  const target = aff || el;
  const r = target.getBoundingClientRect();
  const role = target.getAttribute && target.getAttribute('role');
  const tag = (target.tagName || '').toLowerCase() + (role ? '[role=' + role + ']' : '');
  return {
    admit: true, reason: aff ? 'first-party-affordance' : 'first-party-element',
    origin, tag, href, vx, vy,
    box: { sx: Math.round(r.left + ox), sy: Math.round(r.top + oy), w: Math.round(r.width), h: Math.round(r.height) }
  };
}"""

# verify-before-action context: at a SCREEN coord, returns {self, context}.
#  self    = the masked (black) region the click LANDS ON (its mark), or null. Clicking a masked region
#            requires THAT exact mark to have been queried — you can't click a black box you never looked at.
#  context = the masked marks of the bounded UNIT the click sits in (the nearest ancestor holding masked
#            content, if small enough to be one unit; a large container is a group, not a unit, and adds
#            nothing). For a click on a TRUSTED control, querying ANY ONE of these establishes the unit it
#            acts on.
# Domain-agnostic — no notion of "commit". Marks are read from data-smask-n (stamped by content.js).
_CONTEXT_JS = """([x, y, vp]) => {
  const AFF = 'button,a[href],input,select,textarea,[role=button],[role=link],[role=tab],[role=checkbox],[role=radio],[role=combobox],[role=textbox],[role=menuitem],[role=switch],[role=slider],[role=option]';
  const UNTRUST = '[data-trust="untrusted"],[data-trust="obscured"]';
  const ox = vp ? 0 : (window.screenX || 0), oy = vp ? 0 : ((window.screenY || 0) + (window.outerHeight - window.innerHeight));
  const el = document.elementFromPoint(x - ox, y - oy);
  if (!el) return { self: null, context: [] };
  const target = (el.closest && el.closest(AFF)) || el;
  // self: the masked region the click actually LANDED ON (its mark). Mandatory to query before clicking it.
  let self_mark = null;
  const selfU = el.closest && el.closest(UNTRUST);
  if (selfU) { const n = parseInt(selfU.getAttribute('data-smask-n'), 10); if (n) self_mark = n; }
  // context: masked marks the control ACTS ON — its own masked descendants plus its DIRECTLY masked
  // siblings (the item's fields co-located with the control in one box, e.g. a card where the button sits
  // alongside the item's name/image). Masked content boxed in a SEPARATE nested sub-region that the control
  // merely shares a broader ancestor with (a summary / navigation control, e.g. a cart or checkout button)
  // is NOT what the control acts on, so it stays ungated. Domain-general — no labels, no "commit" notion.
  const context = new Set(), ITEM_MAX = 12;
  const add = e => { const n = parseInt(e.getAttribute('data-smask-n'), 10); if (n) context.add(n); };
  target.querySelectorAll(UNTRUST).forEach(add);                              // masked content INSIDE the control
  const parent = target.parentElement;                                       // masked fields directly BESIDE it
  if (parent) for (const sib of parent.children) if (sib !== target && sib.matches && sib.matches(UNTRUST)) add(sib);
  return { self: self_mark, context: (context.size <= ITEM_MAX ? [...context] : []) };
}"""


class Smask:
    def __init__(self, cdp_url=None, key=None, host_substr="vercel.app", log=print, page=None):
        self.cdp_url = cdp_url
        self.key = key or secrets.token_hex(12)     # per-session; unknown to agent + page content
        self.host_substr = host_substr
        self.log = log
        self._pw = None
        self._browser = None
        self._page = page                            # injected Playwright page (WebArena driver): bypass connect()/CDP

    # -- lifecycle -----------------------------------------------------------
    def connect(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.connect_over_cdp(self.cdp_url)
        return self

    def close(self):
        for fn in (lambda: self._browser and self._browser.close(),
                   lambda: self._pw and self._pw.stop()):
            try:
                fn()
            except Exception:
                pass

    # -- target pages --------------------------------------------------------
    def _pages(self):
        if self._page is not None:                   # injected-page mode: the driver owns the one page
            return [self._page]
        pages = []
        for ctx in self._browser.contexts:
            for pg in ctx.pages:
                try:
                    if self.host_substr in (pg.url or ""):
                        pages.append(pg)
                except Exception:
                    pass
        return pages

    def page(self):
        """The first omnizon Playwright page on THIS browser (reuse it — do not open a second
        sync_playwright(), which Playwright forbids while this driver is running)."""
        ps = self._pages()
        return ps[0] if ps else None

    def ready(self):
        """True once the extension's overlay exists on at least one omnizon page."""
        for pg in self._pages():
            try:
                if pg.evaluate(_READY_JS):
                    return True
            except Exception:
                pass
        return False

    def wait_ready(self, timeout=12.0, poll=0.5):
        end = time.time() + timeout
        while time.time() < end:
            if self.ready():
                return True
            time.sleep(poll)
        return False

    # -- commands ------------------------------------------------------------
    def _dispatch(self, action):
        n = 0
        for pg in self._pages():
            try:
                pg.evaluate(_DISPATCH_JS, [self.key, action])
                n += 1
            except Exception as e:
                self.log(f"[smask] {action} failed on {getattr(pg, 'url', '?')}: {e}")
        return n

    def mask(self):
        """maskAll — trusted V⊤: black boxes + contract + number, untrusted content hidden from a11y.
        Re-stamps (catches lazy-loaded DOM). Call before every actuator screenshot."""
        return self._dispatch("maskAll")

    def marks(self):
        """marksAll — untrusted/shadow view: numbers kept, black fill removed, content visible,
        a11y intact. Call before a shadow screenshot so the untrusted model can answer 'what is #N?'."""
        return self._dispatch("marksAll")

    def reveal(self):
        """revealAll — drop every overlay entirely (fully clean screenshot)."""
        return self._dispatch("revealAll")

    def remask(self):
        """Restore the trusted masked view after a shadow call."""
        return self._dispatch("maskAll")

    def isolate(self, n):
        """V_g: reveal ONLY mark n (mask every other region, both channels). Capture the screenshot for
        the per-mark shadow, then call remask() to restore V⊤. Only the runtime can drive this."""
        cnt = 0
        for pg in self._pages():
            try:
                pg.evaluate(_ISOLATE_JS, [self.key, int(n)])
                cnt += 1
            except Exception as e:
                self.log(f"[smask] isolate({n}) failed on {getattr(pg, 'url', '?')}: {e}")
        return cnt

    def egress(self, x, y, viewport=False):
        """CFI gate at (x,y): resolve the affordance under it → admit iff first-party / SAME-origin, else
        refuse (OFF-origin, regime-2 frame, no affordance, off-viewport). viewport=True: (x,y) are already
        VIEWPORT coords (Playwright/page.screenshot frame) so skip the screen→viewport offset (ox=oy=0);
        default False keeps the omnizon SCREEN-coord behavior. Returns {admit, reason, origin?, tag?, href?, box?}."""
        pg = self.page()
        if pg is None:
            return {"admit": False, "reason": "no-page"}
        try:
            return pg.evaluate(_EGRESS_JS, [float(x), float(y), bool(viewport)])
        except Exception as e:
            return {"admit": False, "reason": f"egress-error: {e}"}

    def click_context(self, x, y, viewport=False):
        """Verify-before-action context of a click at (x,y): {"self": <mark|None>, "context": [<mark#s>]}.
        self = the masked region the click LANDS ON (must be queried before clicking it); context = the
        bounded unit's masked marks (a click on a trusted control needs >=1 of them queried)."""
        pg = self.page()
        if pg is None:
            return {"self": None, "context": []}
        try:
            return pg.evaluate(_CONTEXT_JS, [float(x), float(y), bool(viewport)])
        except Exception as e:
            return {"self": None, "context": [], "error": str(e)}

    def lexicon(self, page=None):
        """The untrusted lexicon: text + alt/aria-label/title of every masked (data-trust) element
        and its descendants, read from the live DOM. Passed to the a11y masker for the content pass.
        (Reads DOM attributes set by the isolated-world content script — visible cross-world.)"""
        js = """() => {
          const out = new Set();
          const push = s => { if (s && s.trim().length >= 4) out.add(s.trim()); };
          document.querySelectorAll('[data-trust="untrusted"],[data-trust="frame"],[data-trust="obscured"]').forEach(el => {
            push(el.textContent);
            for (const a of ['aria-label','alt','title']) push(el.getAttribute(a));
            el.querySelectorAll('[aria-label],[alt],[title]').forEach(d => {
              for (const a of ['aria-label','alt','title']) push(d.getAttribute(a));
            });
          });
          return [...out];
        }"""
        pages = [page] if page is not None else self._pages()
        for pg in pages:
            try:
                v = pg.evaluate(js)
                if v:
                    return v
            except Exception:
                pass
        return []

    def manifest(self, page=None):
        """Return the current numbered-region manifest (list of {n, tag, x,y,w,h, sx,sy}) from the
        first omnizon page. Content-free (contract placeholders only). Used to mask the a11y tree."""
        import json
        pages = [page] if page is not None else self._pages()
        for pg in pages:
            try:
                raw = pg.evaluate(_MANIFEST_JS)
                data = json.loads(raw)
                if data:
                    return data
            except Exception:
                pass
        return []

    def rules_count(self, page=None):
        """How many mask rules the extension has LOADED for this host (from localStorage or the bundled
        seed), read off the shared DOM attribute the engine stamps at inject time. Returns -1 if the
        attribute is absent (extension not injected / attribute unsupported). Lets the runtime tell
        'rules loaded, page legitimately has no untrusted content' (>0) from 'seed/load failed' (0),
        so a 0-manifest on a real content page is never mistaken for a clean page."""
        pg = page or self.page()
        if not pg:
            return -1
        try:
            v = pg.evaluate("() => document.documentElement.getAttribute('data-smask-rules')")
            return int(v) if v is not None else -1
        except Exception:
            return -1
