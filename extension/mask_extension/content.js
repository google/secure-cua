// Copyright 2026 Google LLC
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Secure Mask — engine (no UI). Console-driven via the main-world shim (content_main.js → window.mark).
   Rules (per host, template-scoped) stamp data-trust onto the live DOM each render; masking blacks out
   [data-trust=untrusted|frame] in both channels (overlay pixels + reversible aria). Runtime unmask is
   secret-gated (agent has no JS, no UI). */
(function () {
  if (window.__smaskLoaded) return; window.__smaskLoaded = true;
  const HOST = location.host;                                    // host:PORT — so localhost sites on different ports (WebArena) don't collide
  const RKEY = '__smask_rules__' + HOST;
  let rules = JSON.parse(localStorage.getItem(RKEY) || 'null');   // {sel, trust:'⊥'|'⊤'|'OBSCURED', regime?, page, part?}
  if (!rules || !rules.length) {                                 // fresh install (VM): seed from bundled export (seed.js -> self.__SMASK_SEED)
    const seed = self.__SMASK_SEED;
    rules = [];
    // seed may be a MULTI-site array [{site,templates}...] (current build_seed) or a map {sites:{host:templates}},
    // or a legacy single {site,templates}. Pick the entry whose site === this host:port.
    let tmpl = null;
    if (Array.isArray(seed)) { const e = seed.find(s => s && s.site === HOST); tmpl = e && e.templates; }
    else if (seed && seed.sites) tmpl = seed.sites[HOST];
    else if (seed && seed.site === HOST) tmpl = seed.templates;
    if (tmpl) for (const page in tmpl) for (const r of tmpl[page]) rules.push(Object.assign({ page }, r));
  }
  // expose the loaded rule count on the shared DOM so the runtime can gate on "rules actually loaded"
  // (distinct from "content script injected"): a page with content but 0 loaded rules => seed/load failure,
  // NOT a legitimately-empty mask. Kept current in saveRules() too.
  try { document.documentElement.setAttribute('data-smask-rules', String(rules.length)); } catch (e) {}
  let view = 'off', lastTestSel = null, isolateTarget = null;   // view: 'off' | 'mask' (V⊤) | 'marks' (shadow, all content) | 'isolate' (V_g: one mark revealed)
  const touched = new Map();
  // --- persistent mark numbers: an element keeps its number across scroll / re-render / re-mask on the
  // SAME url (so K rows stay addressable); reset only when the url (path+query) changes = new page. ---
  let numEpoch = null, markNum = new WeakMap(), nextNum = 0;
  function ensureEpoch() { const k = location.pathname + location.search; if (k !== numEpoch) { numEpoch = k; markNum = new WeakMap(); nextNum = 0; } }
  function numberOf(el) { let n = markNum.get(el); if (!n) { n = ++nextNum; markNum.set(el, n); } return n; }
  const AFF = 'button,a[href],input,select,textarea,[role=button],[role=link],[role=tab],[role=checkbox],[role=radio],[role=combobox],[role=textbox],[role=menuitem],[role=switch],[role=slider],[role=option]';
  const saveRules = () => localStorage.setItem(RKEY, JSON.stringify(rules));
  const routeKey = () => location.pathname.replace(/\/\d+/g, '/#').replace(/\/+$/, '') || '/';
  // A rule's `page` scope is matched against routeKey. Besides exact match and '*' (whole site),
  // a pattern may use per-segment wildcards so a template with variable path segments (e.g. Postmill
  // /f/<forum>/<id>/<slug>) is still scoped, not blanket site-wide:  '*' = any ONE segment,
  // '**' = the rest (tail; must be last), '#' = a numeric segment (routeKey already collapses ids).
  //   /f/*         -> any forum listing        /user/*      -> any profile
  //   /f/*/#/**    -> any submission + comments (slug/extra segments optional)
  function pageMatches(pat, rk) {
    if (!pat || pat === '*') return true;                     // whole site (also the {all:true} default)
    if (pat === rk) return true;                              // exact routeKey (fast path / omnizon)
    if (pat.indexOf('*') === -1) return false;                // literal, already knew it's != rk
    const P = pat.split('/'), K = rk.split('/');
    for (let i = 0; i < P.length; i++) {
      if (P[i] === '**') return true;                         // tail wildcard: matches the rest (incl. none)
      if (i >= K.length) return false;
      if (P[i] !== '*' && P[i] !== K[i]) return false;        // '*' matches any one segment
    }
    return P.length === K.length;                             // no tail wildcard -> segment count must match
  }

  const overlay = document.createElement('div'); overlay.id = '__smask_overlay'; overlay.setAttribute('aria-hidden', 'true');
  overlay.style.cssText = 'position:fixed;left:0;top:0;width:0;height:0;pointer-events:none;z-index:2147483640';   // above any page content
  const testLayer = document.createElement('div'); testLayer.id = '__smask_test'; testLayer.setAttribute('aria-hidden', 'true');
  testLayer.style.cssText = 'position:fixed;left:0;top:0;width:0;height:0;pointer-events:none;z-index:2147483645';
  document.documentElement.appendChild(overlay); document.documentElement.appendChild(testLayer);
  const inUI = el => el && el.closest && (el.closest('#__smask_overlay') || el.closest('#__smask_test'));

  // ---- rules -> data-trust (template-scoped so nothing bleeds across templates) ----
  let stamped = new Set();
  function stamp() {
    restoreAria();                                  // undo any aria masking FIRST, else aria-label rules
    stamped.forEach(el => { if (el.isConnected) delete el.dataset.trust; }); stamped.clear();   // fail to re-match on re-stamp (maskAria blanks aria-label)
    const rk = routeKey();
    for (const r of rules) {                       // later rule wins (precedence)
      if (r.part) continue;                        // part rules (value/text) -> render-time text-rect masking, not data-trust
      if (!pageMatches(r.page, rk)) continue;
      let els = []; try { els = document.querySelectorAll(r.sel); } catch (e) {}
      els.forEach(el => { el.dataset.trust = r.trust === '⊥' ? (r.regime === 2 ? 'frame' : 'untrusted') : (r.trust === 'OBSCURED' ? 'obscured' : 'trusted'); stamped.add(el); });
    }
  }

  // ---- content-free placeholder (MARKING_CONTRACT §4) ----
  const round5 = n => Math.round(n / 5) * 5;
  function aspect(w, h) { if (!h) return w + ':1'; const r = w / h; for (const [a, b] of [[1, 1], [4, 3], [3, 2], [16, 9], [2, 1], [3, 1], [1, 2], [2, 3], [3, 4], [9, 16]]) if (Math.abs(r - a / b) < .1) return a + ':' + b; return (Math.round(r * 10) / 10) + ':1'; }
  const isActionable = el => { try { return el.matches(AFF); } catch (e) { return false; } };
  function kindOf(el) { const t = el.tagName.toLowerCase(), role = el.getAttribute('role'); if (t === 'img' || role === 'img') return 'IMG'; if (!el.textContent.trim() && el.querySelector('img, svg[role=img], picture')) return 'IMG'; if (t === 'button' || role === 'button') return 'BUTTON'; if (t === 'a' || role === 'link') return 'LINK'; if (t === 'input' || t === 'textarea' || t === 'select' || role === 'textbox' || role === 'combobox') return 'INPUT'; if (t === 'svg') return 'ICON'; return 'TEXT'; }
  function clickable(el) { try { return !!el.closest(AFF); } catch (e) { return false; } }        // itself or an ancestor is a link/button
  function originOf(el) {                                   // SAME (internal) / OFF (cross-origin), or null
    const a = el.closest && el.closest('a[href]');
    if (a) { try { return new URL(a.getAttribute('href'), location.href).origin === location.origin ? 'SAME' : 'OFF'; } catch (e) {} }
    if (el.closest && el.closest('[role=link],[role=button]')) return 'SAME';                       // client-side nav = internal route
    const img = el.tagName === 'IMG' ? el : (el.querySelector && el.querySelector('img'));
    if (img && img.getAttribute('src')) { try { return new URL(img.getAttribute('src'), location.href).origin === location.origin ? 'SAME' : 'OFF'; } catch (e) {} }
    return null;
  }
  const contentLen = el => {                                // form controls hold their text in .value, not textContent
    const t = el.tagName.toLowerCase();
    if (t === 'input' || t === 'textarea') return (el.value || '').trim().length;
    if (t === 'select') { const o = el.options && el.options[el.selectedIndex]; return ((o && o.text) || '').trim().length; }
    return (el.textContent || '').trim().length;
  };
  function placeholder(el, r, special) {
    if (special === 'OBSCURED') return '<OBSCURED>';
    const parts = [];
    if (special === 'FRAME') { parts.push('FRAME', Math.round(r.width) + '×' + Math.round(r.height)); }
    else {
      const k = kindOf(el); parts.push(k);
      if (k === 'IMG') parts.push(aspect(r.width, r.height));
      else if (k !== 'ICON') parts.push('~' + round5(contentLen(el)) + 'ch');
    }
    const o = originOf(el); if (o) parts.push(o);
    if (special !== 'FRAME' && clickable(el)) parts.push('CLICKABLE');
    return '<' + parts.join(' | ') + '>';
  }

  // ---- a11y channel (reversible) ----
  function restoreAria() { touched.forEach((v, el) => { if (v.h === null) el.removeAttribute('aria-hidden'); else el.setAttribute('aria-hidden', v.h); if (v.l !== null) el.setAttribute('aria-label', v.l); }); touched.clear(); }
  function maskAria(el) { if (touched.has(el)) return; touched.set(el, { h: el.getAttribute('aria-hidden'), l: el.getAttribute('aria-label') }); el.setAttribute('aria-hidden', 'true'); if (el.hasAttribute('aria-label')) el.setAttribute('aria-label', ' '); }

  function visibleRect(el) {                                 // box clipped to clipping-ancestors (overflow) + viewport
    const r = el.getBoundingClientRect();
    let L = r.left, T = r.top, R = r.right, B = r.bottom;
    for (let p = el.parentElement, d = 0; p && p !== document.documentElement && d < 24; p = p.parentElement, d++) {
      const s = getComputedStyle(p);
      if (s.overflowX !== 'visible' || s.overflowY !== 'visible') { const pr = p.getBoundingClientRect(); L = Math.max(L, pr.left); T = Math.max(T, pr.top); R = Math.min(R, pr.right); B = Math.min(B, pr.bottom); }
    }
    L = Math.max(L, 0); T = Math.max(T, 0); R = Math.min(R, innerWidth); B = Math.min(B, innerHeight);
    return { left: L, top: T, width: R - L, height: B - T };
  }
  function clipToView(r) { const L = Math.max(r.left, 0), T = Math.max(r.top, 0), R = Math.min(r.right, innerWidth), B = Math.min(r.bottom, innerHeight); return { left: L, top: T, width: R - L, height: B - T }; }
  function subsumes(a, b, tol) {                             // does box a fully contain box b? (V_g: reveal a when isolating b)
    tol = tol == null ? 2 : tol;
    return b.left >= a.left - tol && b.top >= a.top - tol &&
           b.left + b.width <= a.left + a.width + tol && b.top + b.height <= a.top + a.height + tol;
  }
  function lastTextNode(el) { let t = null; for (const nd of el.childNodes) if (nd.nodeType === 3 && nd.nodeValue.trim()) t = nd; return t; }
  function textRect(tn) { try { const rg = document.createRange(); rg.selectNodeContents(tn); const c = rg.getBoundingClientRect(); return { left: c.left, top: c.top, right: c.right, bottom: c.bottom, width: c.width, height: c.height }; } catch (e) { return null; } }
  function textLineRects(el) {                               // per-LINE client rects of ALL text in el (viewport coords); empty gaps
    try { const rg = document.createRange(); rg.selectNodeContents(el);        // (padding, a floated trusted neighbor) fall BETWEEN lines -> never covered
      return [...rg.getClientRects()].map(c => ({ left: c.left, top: c.top, right: c.right, bottom: c.bottom, width: c.width, height: c.height })); } catch (e) { return []; }
  }
  function unionRect(rs) {                                   // bounding box of rects (derive right/bottom from w/h -> works on clipToView output, which drops them)
    let L = 1e9, T = 1e9, R = -1e9, B = -1e9;
    for (const r of rs) { L = Math.min(L, r.left); T = Math.min(T, r.top); R = Math.max(R, r.left + r.width); B = Math.max(B, r.top + r.height); }
    return { left: L, top: T, right: R, bottom: B, width: R - L, height: B - T };
  }
  function linesClip(lines, r) {                             // clip-path (coords relative to box r) that shows ONLY the text-line rects
    let d = '';
    for (const L of lines) {
      const x0 = Math.max(0, L.left - r.left), y0 = Math.max(0, L.top - r.top), x1 = Math.min(r.width, L.left + L.width - r.left), y1 = Math.min(r.height, L.top + L.height - r.top);
      if (x1 > x0 && y1 > y0) d += `M${x0} ${y0}L${x1} ${y0}L${x1} ${y1}L${x0} ${y1}Z`;
    }
    return d ? `path('${d}')` : 'path("M0 0Z")';             // no visible line -> empty region (box paints nothing, but a11y stays masked)
  }

  // ---- render (pixels). view: 'off' | 'mask' (trusted V⊤: black + contract + number, a11y hidden)
  //                              | 'marks' (untrusted/shadow: content visible, number only, a11y intact)
  let maskBoxes = [];
  const scrOX = () => (window.screenX || 0);                                 // viewport origin in SCREEN coords
  const scrOY = () => (window.screenY || 0) + (window.outerHeight - window.innerHeight);   // (for a11y masking)
  // Largest monospace font (px) at which `txt` — WRAPPED — fits a w×h box. Walks down from 12; models
  // chars-per-line from the width so multi-line tags no longer overflow short boxes (the old formula only
  // checked one line's height). CW/LH = DejaVu Sans Mono advance / line-height as a fraction of font-size.
  function fitFont(txt, w, h) {
    const CW = 0.62, LH = 1.12, n = Math.max(1, (txt || '').length);
    for (let fs = 12; fs > 4; fs--) {
      const cpl = Math.max(1, Math.floor(w / (CW * fs)));    // chars that fit on one line
      if (Math.ceil(n / cpl) * LH * fs <= h) return fs;      // wrapped height fits the box?
    }
    return 4;                                                // floor; overflow:hidden clips the rare too-small box
  }
  function mkBox(r, txt, frame, num) {                        // filled black mask box (trusted view)
    const b = document.createElement('div');
    b.style.cssText = `position:absolute;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;box-sizing:border-box;background:${frame ? '#0b0f1a' : '#1a1a22'};border:1px solid #5b6472;overflow:hidden`;
    const padR = r.width > 70 ? 32 : 2;                       // reserve room for the number badge on wider boxes
    const fs = fitFont(txt, Math.max(6, r.width - 2 - padR), Math.max(6, r.height - 2));   // fit the WRAPPED tag to the usable box
    const t = document.createElement('div'); t.textContent = txt;
    t.style.cssText = `position:absolute;left:2px;top:1px;right:${padR}px;bottom:1px;color:#fff;font:${fs}px/1.1 "DejaVu Sans Mono",monospace;word-break:break-word;overflow:hidden`;
    b.appendChild(t);
    return b;                                                 // number badge is a SEPARATE overlay element (mkBadge) so occlusion clipping never eats it
  }
  function mkMark(r) {                                        // transparent outline (untrusted view)
    const b = document.createElement('div');
    b.style.cssText = `position:absolute;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;box-sizing:border-box;outline:1.5px dashed rgba(217,70,239,.95);background:rgba(217,70,239,.06)`;
    return b;
  }
  const BW = 17, BH = 14;                                     // number-badge footprint
  function mkBadge(num) {                                     // standalone number badge (positioned at a VISIBLE corner by reposition)
    const nb = document.createElement('div'); nb.textContent = num;
    // MUST use an explicitly-installed font ("DejaVu Sans Mono") — generic families (sans-serif/monospace)
    // do NOT resolve in the headless Xvfb Chromium, so the digits painted BLANK (looked like an empty badge).
    nb.style.cssText = `position:absolute;height:${BH}px;min-width:${BW}px;box-sizing:border-box;background:#d946ef;color:#fff;font:700 11px/14px "DejaVu Sans Mono",monospace;padding:0 3px;text-align:center;border-radius:2px`;
    return nb;
  }
  function mkTag(txt) {                                       // standalone type-tag for text (per-line-clipped) masks — the box's OWN inner
    const t = document.createElement('div'); t.textContent = txt;   // label would be clipped away, so draw the tag as a separate un-clipped overlay
    t.style.cssText = `position:absolute;height:${BH}px;max-width:70vw;box-sizing:border-box;background:#1a1a22;color:#fff;font:700 10px/14px "DejaVu Sans Mono",monospace;padding:0 3px;border:1px solid #5b6472;border-radius:2px;white-space:nowrap;overflow:hidden`;
    return t;
  }
  function render() {
    overlay.textContent = ''; restoreAria(); maskBoxes = [];
    if (view === 'off') { document.documentElement.removeAttribute('data-smask-manifest'); return; }
    ensureEpoch();                                            // reset numbers only when the url changed
    const marksOnly = view === 'marks';
    const isolating = view === 'isolate';
    let targetBox = null;                                    // V_g: also reveal masks that SUBSUME the isolated mark (untrusted view only; V⊤ untouched)
    if (isolating) {                                          // find the isolated mark's box via the stable per-element number (not a DOM attr)
      document.querySelectorAll('[data-trust="untrusted"],[data-trust="obscured"],[data-trust="frame"]').forEach(el => {
        if (!inUI(el) && numberOf(el) === isolateTarget) { const tb = visibleRect(el); if (tb.width > 0 && tb.height > 0) targetBox = tb; } }); }
    const manifest = [];
    const add = (el, r, txt, frame, tn, lines) => {           // one region: draw + (masked) hide a11y + record
      const num = numberOf(el);                              // persistent per-url number (stable across scroll/re-mask)
      try { el.dataset.smaskN = num; } catch (e) {}          // expose the mark number to the runtime (commit gate)
      const reveal = marksOnly || (isolating && (num === isolateTarget || (targetBox && subsumes(r, targetBox))));   // V_g: the isolated mark + any mask that covers it
      const b = reveal ? mkMark(r) : mkBox(r, lines ? '' : txt, frame);   // text masks: suppress the box's inner label (per-line clip would eat it)
      const badge = mkBadge(num);                            // separate element -> never clipped by the box's occlusion carve
      const tag = (lines && !reveal) ? mkTag(txt) : null;    // ...and carry the type-tag on its own un-clipped element instead
      overlay.appendChild(b); overlay.appendChild(badge); if (tag) overlay.appendChild(tag);
      maskBoxes.push({ el, b, badge, tag, tn, lines, marksOnly: reveal });
      if (!reveal) maskAria(el);                              // masked regions hidden from a11y; a revealed region is left intact
      manifest.push({ n: num, tag: txt, x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height),
                      sx: Math.round(r.left + scrOX()), sy: Math.round(r.top + scrOY()) });
    };
    document.querySelectorAll('[data-trust="frame"]').forEach(el => {                         // regime-2: one block
      if (inUI(el)) return; const r = el.getBoundingClientRect(); if (r.width < 3 || r.height < 3) return;
      add(el, r, placeholder(el, r, 'FRAME'), true, null);
    });
    document.querySelectorAll('[data-trust="untrusted"],[data-trust="obscured"]').forEach(el => {
      if (inUI(el) || el.closest('[data-trust="frame"]')) return;
      const r = visibleRect(el); if (r.width < 3 || r.height < 3) return;
      add(el, r, placeholder(el, r, el.dataset.trust === 'obscured' ? 'OBSCURED' : null), false, null);
    });
    const rk = routeKey();                                    // part rules mask a text SUB-RECT, not the whole element box:
    for (const rr of rules) {                                 //   'value' = last text node ("Label: Value" -> Value); 'text' = the element's WHOLE text, per-line
      if (!rr.part || rr.trust !== '⊥' || !pageMatches(rr.page, rk)) continue;
      let els = []; try { els = document.querySelectorAll(rr.sel); } catch (e) {}
      els.forEach(el => {
        if (inUI(el)) return;
        if (rr.part === 'value') {
          const tn = lastTextNode(el); if (!tn) return;
          const tr = textRect(tn); if (!tr) return;
          const v = clipToView(tr); if (v.width < 2 || v.height < 2) return;
          add(el, v, '<TEXT | ~' + round5(tn.nodeValue.trim().length) + 'ch>', false, tn);
        } else {                                              // 'text': cover every text LINE so a trusted element floated into the
          const lines = textLineRects(el).map(clipToView).filter(x => x.width >= 2 && x.height >= 2);   // box (e.g. a review's star rating) stays visible AND no text node leaks
          if (!lines.length) return;
          const v = unionRect(lines); if (v.width < 2 || v.height < 2) return;
          add(el, v, '<TEXT | ~' + round5((el.textContent || '').trim().length) + 'ch>', false, null, lines);
        }
      });
    }
    // a11y masking must cover ALL untrusted content, including BELOW-THE-FOLD: a pixel box is only drawn for
    // in-viewport rects (visibleRect >= 3px), but getFullAXTree returns the WHOLE page, so off-viewport prose
    // would otherwise leak to the actuator's a11y channel. So aria-hide every untrusted region here, decoupled
    // from the box render (idempotent via `touched`; restoreAria() undoes it). Skip only the revealed mark
    // (V_g isolate) and the 'marks' authoring view (which intentionally shows content).
    if (!marksOnly) {
      document.querySelectorAll('[data-trust="untrusted"],[data-trust="frame"],[data-trust="obscured"]').forEach(el => {
        if (inUI(el)) return;
        if (isolating && (String(el.dataset.smaskN) === String(isolateTarget) || (targetBox && subsumes(visibleRect(el), targetBox)))) return;   // keep the isolated region + any mask that subsumes it (V_g)
        maskAria(el);
      });
      for (const rr of rules) {                              // part rules (value/text) don't set data-trust, so aria-mask ALL their matches here
        if (!rr.part || rr.trust !== '⊥' || !pageMatches(rr.page, rk)) continue;   // (add() only aria-masks the in-view ones; the AXTree spans the whole page)
        let els = []; try { els = document.querySelectorAll(rr.sel); } catch (e) {}
        els.forEach(el => {
          if (inUI(el)) return;
          if (isolating && (String(el.dataset.smaskN) === String(isolateTarget) || (targetBox && subsumes(visibleRect(el), targetBox)))) return;   // keep the isolated (V_g) mark + subsuming masks readable
          maskAria(el);
        });
      }
    }
    document.documentElement.setAttribute('data-smask-manifest', JSON.stringify(manifest));
    reposition();                                            // apply occlusion (hide/carve boxes under opaque overlays) on this fresh draw
  }
  // --- occlusion: keep the pixel mask off OPAQUE overlays (dropdowns/menus/modals) in FRONT of the masked
  //     element. Pixels ONLY — maskAria never depends on visual stacking, so the untrusted content stays
  //     hidden from the a11y tree regardless. FULL opaque cover -> hide the box. PARTIAL opaque cover ->
  //     carve each overlay's rect out, flush (inset only by the overlay's own corner-radius so its unpainted
  //     rounded corners aren't carved). A see-through cover, any point where the element is still visible, or
  //     any point where el paints ON TOP of the cover (elPts) keeps the mask there (fail-safe). ---
  function bgAlpha(c) { const m = /^rgba?\(([^)]+)\)/.exec(c || ''); if (!m) return 0; const p = m[1].split(',').map(parseFloat); return (p.length === 4 ? p[3] : 1) || 0; }
  function undimmed(n) {                                     // n's paint is NOT thinned by group-opacity/filter on n or any ancestor
    try {
      let o = 1;
      for (let a = n, d = 0; a && a !== document.documentElement && d < 24; a = a.parentElement, d++) {
        const s = getComputedStyle(a);
        if (/\b(blur|opacity)\(/.test(s.filter || '')) return false;    // filter blur()/opacity() shows content through; drop-shadow/color filters keep it solid
        o *= (parseFloat(s.opacity) || 0);
        if (o < 0.85) return false;                                     // element/group opacity (incl. compounding) fades it -> keep mask
      }
    } catch (e) { return false; }
    return true;
  }
  function coverInset(el) {                                  // px to shrink the carve hole so an overlay's ROUNDED corners (which the bg does NOT paint)
    try { const s = getComputedStyle(el);                   // aren't carved over untrusted content. Inset = the unpainted corner-wedge depth R*(1-1/√2)≈0.293R (scales
      const r = Math.max(parseFloat(s.borderTopLeftRadius) || 0, parseFloat(s.borderTopRightRadius) || 0, parseFloat(s.borderBottomLeftRadius) || 0, parseFloat(s.borderBottomRightRadius) || 0);   // with ANY radius); 0 for a SQUARE cover -> flush carve, no line.
      return Math.ceil(0.3 * r);
    } catch (e) { return 2; }
  }
  // Rect to carve for an opaque hit `n`: expand to the NEAREST positioned (absolute/fixed) ancestor that is itself
  // opaque AND whose border-box ENCLOSES n. Carves a complex overlay (a mini-cart whose rows/columns carry the bg
  // with transparent gutters between) as ONE rect that covers the gutters. The ENCLOSURE test is ESSENTIAL: a nested
  // absolute child positioned OUTSIDE its parent — a Magento sub-flyout is `left:100%`, to the RIGHT of its parent
  // submenu — is NOT enclosed by that parent, so we must NOT expand to the parent (whose box is elsewhere) and instead
  // carve the sub-flyout's OWN box. Bounded below the LCA; the container fully paints (bg>=0.85, undimmed) -> all-opaque. Else n.
  function coverNode(n, el) {
    let nr; try { nr = n.getBoundingClientRect(); } catch (e) { return n; }
    try {
      for (let a = n.parentElement, d = 0; a && a !== document.body && a !== document.documentElement && !a.contains(el) && d < 16; a = a.parentElement, d++) {
        const s = getComputedStyle(a);
        if ((s.position === 'absolute' || s.position === 'fixed') && bgAlpha(s.backgroundColor) >= 0.85 && undimmed(a)) {
          const ar = a.getBoundingClientRect();
          if (ar.left <= nr.left + 1 && ar.top <= nr.top + 1 && ar.right >= nr.right - 1 && ar.bottom >= nr.bottom - 1) return a;   // encloses n -> the overlay container
        }
      }
    } catch (e) {}
    return n;
  }
  // Opaque surface painted IN FRONT of `el` at hit `h` -> the cover NODE (truthy) | null. Caller guarantees h is
  // neither el, a descendant, nor an ancestor of el. Walk UP from h and STOP at the lowest common ancestor of h
  // and el (that node — and <body>/<html> above it — is el's BACKDROP: it paints BEHIND el, never as an overlay
  // in front). A cover counts as opaque ONLY via a near-solid background-COLOR that fills its whole border-box
  // AND is not thinned by opacity/filter. A background-IMAGE is deliberately NOT trusted (gradients / alpha PNGs
  // carry transparent stops, so the border-box isn't actually filled -> would carve/hide over see-through pixels).
  // The caller carves that node's border-box; background-color fills it entirely. Uncertain -> null -> keep mask.
  function isOpaque(h, el) {
    try {
      for (let n = h, d = 0; n && d < 24; n = n.parentElement, d++) {
        if (n === el || n.contains(el) || n === document.body || n === document.documentElement) return null;
        const s = getComputedStyle(n);
        if (s.visibility === 'hidden') continue;                          // not painting -> no surface here
        if (bgAlpha(s.backgroundColor) >= 0.85) return undimmed(n) ? coverNode(n, el) : null;   // near-solid fill; carve its opaque container
      }
    } catch (e) {}
    return null;                                                          // transparent / uncertain -> keep the mask
  }
  function occlusionOf(el, r) {                              // null=clear · {full:true} · {coverRects:[...]}=partial
    // Sample a GRID (~48px spacing) — a few fixed points MISS a narrow overlay that falls between them on a wide
    // box. Collect EVERY distinct opaque cover node (a masked box can sit behind a dropdown AND a nested sub-flyout).
    const STEP = 48;
    const nx = Math.max(3, Math.min(24, Math.ceil(r.width / STEP)));
    const ny = Math.max(2, Math.min(10, Math.ceil(r.height / STEP)));
    let visible = false;
    const covers = new Map();                                // opaque cover node -> its client rect (dedup by node)
    const elPts = [];                                        // points where el (or a descendant) actually PAINTS on top
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const px = r.left + r.width * (i + 0.5) / nx, py = r.top + r.height * (j + 0.5) / ny;
      const h = document.elementFromPoint(px, py);
      if (h === el || el.contains(h)) { visible = true; elPts.push([px, py]); continue; }   // el (or descendant) paints on top here
      if (!h || h.contains(el)) { visible = true; continue; }   // el's own empty area (backdrop through an L-gap) / off-screen — NOT a front overlay, el not painting
      const cov = isOpaque(h, el);
      if (cov) { if (!covers.has(cov)) { const cr = cov.getBoundingClientRect(); covers.set(cov, { left: cr.left, top: cr.top, right: cr.right, bottom: cr.bottom, in: coverInset(cov) }); } }
      else visible = true;                                   // see-through cover -> el could show through -> keep mask here
    }
    if (!covers.size) return null;
    // Drop any cover rect that CONTAINS a point where el paints on top (el is higher-z than that cover there) —
    // carving it would re-expose el. This bounds coverNode's container expansion to genuinely-occluding rects.
    const rects = [...covers.values()].filter(o => !elPts.some(([x, y]) => x >= o.left && x <= o.right && y >= o.top && y <= o.bottom));
    if (!rects.length) return null;
    // FULL (hide box + border + badge) ONLY when a SINGLE opaque rect GEOMETRICALLY spans the whole box AND no
    // sample saw el. Geometry (not "every sample happened to be covered") so a see-through sliver hiding BETWEEN
    // the 48px grid points can never force a full hide -> worst case it downgrades to a partial carve (sliver stays masked).
    if (!visible) for (const o of rects)
      if (o.left <= r.left && o.top <= r.top && o.right >= r.left + r.width && o.bottom >= r.top + r.height)
        return { full: true };
    return { coverRects: rects };
  }
  function placeBadge(m, r) {                                // put the number badge at a corner where the element is still visible
    const x0 = r.left, y0 = r.top, x1 = r.left + r.width - BW, y1 = r.top + r.height - BH;
    const corners = [[x1, y0], [x0, y0], [x1, y1], [x0, y1]];   // prefer top-right, then TL, BR, BL
    let pick = m.marksOnly ? corners[0] : null, seeThrough = null;
    if (!m.marksOnly) {
      for (const [cx, cy] of corners) {
        const h = document.elementFromPoint(cx + BW / 2, cy + BH / 2);
        if (h && (h === m.el || m.el.contains(h) || h.contains(m.el))) { pick = [cx, cy]; break; }   // best: a corner on the element (or its own area)
        else if (!seeThrough && h && !isOpaque(h, m.el)) seeThrough = [cx, cy];   // acceptable: over a SEE-THROUGH overlay (the number still shows)
      }
      if (!pick) pick = seeThrough;                            // no element-corner -> a see-through corner is fine (don't lose the number under a transparent backdrop)
      if (!pick) { m.badge.style.display = 'none'; return; }    // every corner behind an OPAQUE overlay -> hide the number, don't plant it on the menu
    }
    m.badge.style.display = '';                                 // a prior frame may have hidden it -> restore
    m.badge.style.left = Math.max(0, pick[0]) + 'px'; m.badge.style.top = Math.max(0, pick[1]) + 'px';
  }
  function reposition() {
    maskBoxes.forEach(m => {
      const b = m.b, badge = m.badge;
      let r, curLines = null;
      if (m.lines) {                                          // text mode: recompute the per-line rects (they move with scroll/reflow)
        curLines = textLineRects(m.el).map(clipToView).filter(x => x.width >= 2 && x.height >= 2);
        if (curLines.length) r = unionRect(curLines);
      } else if (m.tn) { const tr = textRect(m.tn); if (!tr) { b.style.display = 'none'; badge.style.display = 'none'; return; } r = clipToView(tr); }
      else { r = visibleRect(m.el); }
      if (!r || r.width < 3 || r.height < 3) { b.style.display = 'none'; badge.style.display = 'none'; if (m.tag) m.tag.style.display = 'none'; return; }
      const occ = m.marksOnly ? null : occlusionOf(m.el, r);
      if (occ && occ.full) { b.style.display = 'none'; badge.style.display = 'none'; if (m.tag) m.tag.style.display = 'none'; return; }   // fully behind an opaque overlay
      b.style.display = ''; badge.style.display = '';
      b.style.left = r.left + 'px'; b.style.top = r.top + 'px'; b.style.width = r.width + 'px'; b.style.height = r.height + 'px';
      if (m.lines) {                                          // clip the box to exactly the text lines. A PARTIAL opaque overlay is left masked here
        b.style.clipPath = linesClip(curLines, r);            // (fail-safe: never expose text protruding past a narrow overlay); occ.full above already hides a full cover.
        if (m.tag) { const f = curLines[0]; m.tag.style.display = ''; m.tag.style.left = f.left + 'px'; m.tag.style.top = f.top + 'px'; }   // tag on the first line (always text, never a top-left float)
      } else if (occ && occ.coverRects) {                     // partial -> carve EACH confirmed opaque cover's rect out, inset by ITS OWN corner-radius
        let holes = '';                                        // (0 for a square cover -> FLUSH, no line on the overlay; >0 only for a rounded overlay,
        for (const o of occ.coverRects) {                      // so its unpainted corners are never carved over untrusted content)
          const IN = o.in || 0;
          const hx = Math.max(r.left, o.left) - r.left + IN, hy = Math.max(r.top, o.top) - r.top + IN;
          const hx2 = Math.min(r.left + r.width, o.right) - r.left - IN, hy2 = Math.min(r.top + r.height, o.bottom) - r.top - IN;
          if (hx2 > hx && hy2 > hy) holes += `M${hx} ${hy}L${hx2} ${hy}L${hx2} ${hy2}L${hx} ${hy2}Z`;
        }
        b.style.clipPath = holes ? `path(evenodd,'M0 0L${r.width} 0L${r.width} ${r.height}L0 ${r.height}Z${holes}')` : '';
      } else b.style.clipPath = '';
      placeBadge(m, r);
    });
    if (lastTestSel) showTest(lastTestSel);
  }
  let raf = 0; const schedule = () => { if (raf) return; raf = requestAnimationFrame(() => { raf = 0; reposition(); }); };
  addEventListener('scroll', schedule, true); addEventListener('resize', schedule);
  addEventListener('click', () => { setTimeout(schedule, 50); setTimeout(schedule, 260); }, true);   // a click may open a dropdown -> re-check occlusion
  let hov = 0;                                                                                       // a HOVER may open a flyout submenu (attr change, missed by the MutationObserver) ->
  addEventListener('mouseover', () => { schedule(); clearTimeout(hov); hov = setTimeout(schedule, 350); }, true);  // re-check occlusion now + once after the menu's hover-intent delay
  function recompute() { stamp(); render(); if (lastTestSel) showTest(lastTestSel); }
  let deb = 0; const mo = new MutationObserver(() => { clearTimeout(deb); deb = setTimeout(recompute, 350); });

  // ---- verify: highlight every match of a selector ----
  function clearTest() { testLayer.textContent = ''; lastTestSel = null; }
  function showTest(sel) {
    testLayer.textContent = ''; lastTestSel = sel; let els = [];
    try { els = [...document.querySelectorAll(sel)]; } catch (e) { return 0; }
    let n = 0;
    els.forEach(el => {
      if (inUI(el)) return; const r = visibleRect(el); if (r.width < 2 || r.height < 2) return; n++;
      const b = document.createElement('div');
      b.style.cssText = `position:absolute;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;outline:2px solid #d946ef;background:rgba(217,70,239,.14);box-sizing:border-box`;
      const lab = document.createElement('div'); lab.textContent = n;
      lab.style.cssText = 'position:absolute;right:0;top:0;background:#d946ef;color:#fff;font:700 10px/13px sans-serif;padding:0 3px;white-space:nowrap';   // top-right, inside the box
      b.appendChild(lab); testLayer.appendChild(b);
    });
    return n;
  }

  function exportRules() {
    const byT = {};                                   // group per template for review + transfer
    for (const r of rules) { const k = r.page || '*'; const e = r.regime === 2 ? { sel: r.sel, trust: r.trust, regime: 2 } : { sel: r.sel, trust: r.trust }; if (r.part) e.part = r.part; (byT[k] = byT[k] || []).push(e); }
    const blob = new Blob([JSON.stringify({ site: HOST, templates: byT }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = HOST.split('.')[0].replace(/[^a-z0-9]+/gi, '_') + '.rules.json'; a.click();
    console.log('[mark] exported ' + rules.length + ' rule(s) across ' + Object.keys(byT).length + ' template(s)');
  }

  // ---- console API (driven by the main-world shim window.mark) ----
  document.addEventListener('__smask_api', ev => {
    const d = ev.detail || {}; const page = d.page ? d.page : (d.all ? '*' : routeKey());   // explicit pattern > {all} > current template
    const TR = { untrust: '⊥', trust: '⊤', frame: '⊥', obscure: 'OBSCURED' };
    if (d.fn === 'test') { const n = showTest(d.sel); console.log('%c[mark] highlighted ' + n + ': ' + d.sel, 'color:#d946ef'); }
    else if (d.fn === 'hideTest') clearTest();
    else if (TR[d.fn]) { rules.push({ sel: d.sel, trust: TR[d.fn], regime: d.fn === 'frame' ? 2 : undefined, page, part: d.part }); saveRules(); clearTest(); recompute(); console.log('%c[mark] ' + TR[d.fn] + (d.part ? ' ' + d.part : '') + (d.fn === 'frame' ? '(frame)' : '') + ' [' + page + '] ' + d.sel, 'color:#22c55e'); }
    else if (d.fn === 'rules') console.table(rules.map((r, i) => ({ '#': i, template: r.page || 'all', trust: r.trust + (r.part ? '·' + r.part.slice(0, 3) : '') + (r.regime === 2 ? '(frame)' : ''), sel: r.sel })).sort((a, b) => String(a.template).localeCompare(String(b.template))));
    else if (d.fn === 'rm') { const g = rules.splice(d.i, 1); saveRules(); recompute(); console.log('[mark] removed', g[0] && g[0].sel); }
    else if (d.fn === 'clear') { const rk = routeKey(), before = rules.length; rules = rules.filter(r => (r.page || '*') !== rk); saveRules(); clearTest(); recompute(); console.log('[mark] cleared ' + (before - rules.length) + ' rule(s) for template ' + rk + ' (other templates + [all] kept)'); }
    else if (d.fn === 'clearAll') { rules.length = 0; saveRules(); clearTest(); recompute(); console.log('[mark] cleared ALL rules for this site'); }
    else if (d.fn === 'masks') { view = d.on !== false ? 'mask' : 'off'; recompute(); console.log('[mark] masks ' + (view === 'mask' ? 'ON (trusted V⊤)' : 'off')); }
    else if (d.fn === 'marks') { view = d.on !== false ? 'marks' : 'off'; recompute(); console.log('[mark] marks ' + (view === 'marks' ? 'ON (untrusted view: numbers, content visible)' : 'off')); }
    else if (d.fn === 'export') exportRules();
  });

  // ---- runtime control channel (key-gated) --------------------------------------------------
  // The runtime (Playwright, main world) sets <html data-smask-key="R"> and dispatches __smask_cmd
  // carrying the same key. Both worlds read the key from the shared DOM attribute. The AGENT emits
  // only pixels (no JS) and untrusted page CONTENT can't set attributes or dispatch events, so neither
  // can drive this channel — only the runtime can. (Residual: untrusted MAIN-world JS could read the
  // attribute; out of scope here — pages are the immutable first-party build, threat is content injection.)
  function forceMask() { view = 'mask'; recompute(); }                 // trusted V⊤: re-stamp (catch lazy-loaded) + render
  function revealAll() { view = 'off'; render(); }                     // drop every mask -> clean unmasked view
  function showMarks() { view = 'marks'; recompute(); }                // untrusted view: numbers kept, content visible
  // key + action are read from shared <html> attributes (guaranteed cross-world; CustomEvent.detail
  // is not, in Chrome). The event is only a synchronous trigger. Gate: a session key must be present
  // (only the runtime can set attributes / dispatch — the agent has no JS, content can't script).
  document.addEventListener('__smask_cmd', () => {
    const h = document.documentElement;
    if (!h.getAttribute('data-smask-key')) return;                    // no armed session -> ignore
    const action = h.getAttribute('data-smask-cmd');
    if (action === 'maskAll' || action === 'remask') forceMask();      // before actuator screenshot / after shadow
    else if (action === 'revealAll' || action === 'unmaskAll') revealAll();   // fully clean screenshot
    else if (action === 'marksAll' || action === 'numbers') showMarks();      // untrusted/shadow view (numbers, content visible)
    else if (action === 'isolateOne') { isolateTarget = parseInt(h.getAttribute('data-smask-mark'), 10); view = 'isolate'; recompute(); }   // V_g: reveal ONLY this mark, mask the rest
    else if (action === 'unmask') {                                   // (future) reveal ONE numbered element
      const sel = h.getAttribute('data-smask-sel'); let el = null;
      try { el = sel && document.querySelector(sel); } catch (e) {}
      if (el) { const t = touched.get(el); if (t) { if (t.h === null) el.removeAttribute('aria-hidden'); else el.setAttribute('aria-hidden', t.h); if (t.l !== null) el.setAttribute('aria-label', t.l); touched.delete(el); } const mb = maskBoxes.find(x => x.el === el); if (mb) mb.b.remove(); }
    }
  });

  recompute();
  mo.observe(document.body, { childList: true, subtree: true, attributes: false });
  ['pushState', 'replaceState'].forEach(m => { const o = history[m]; history[m] = function () { const r = o.apply(this, arguments); setTimeout(recompute, 60); return r; }; });
  addEventListener('popstate', () => recompute());
  window.__smask = { recompute };
})();
