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

/* Main-world shim: exposes window.mark to the devtools console. It does DOM queries itself (so
   mark.test returns a live NodeList you can inspect), and relays state changes to the isolated
   engine (content.js) over a CustomEvent. Keeping the engine isolated preserves the runtime's
   security; this shim is only the authoring console surface. */
(function () {
  if (window.mark) return;
  const send = detail => document.dispatchEvent(new CustomEvent('__smask_api', { detail }));
  const q = sel => { try { return document.querySelectorAll(sel); } catch (e) { console.error('[mark] bad selector:', e.message); return null; } };
  const commit = (fn, color) => (sel, o) => { if (!q(sel)) return; send({ fn, sel, all: !!(o && o.all), page: o && o.page }); };
  const uniqueSel = el => { if (el.id) return '#' + CSS.escape(el.id); const parts = []; for (let n = el; n && n.nodeType === 1 && n !== document.body; n = n.parentElement) { let p = n.tagName.toLowerCase(); const par = n.parentElement; if (par) { const sib = [...par.children].filter(c => c.tagName === n.tagName); if (sib.length > 1) p += ':nth-of-type(' + (sib.indexOf(n) + 1) + ')'; } if (n.id) { parts.unshift('#' + CSS.escape(n.id)); break; } parts.unshift(p); } return parts.join(' > '); };

  window.mark = {
    // --- verify ---
    test(sel) { const els = q(sel); if (!els) return; console.log('%c[mark] ' + sel + ' → ' + els.length + ' match(es)', 'color:#d946ef;font-weight:bold', els); send({ fn: 'test', sel }); return els; },
    count(sel) { const els = q(sel); return els ? els.length : 0; },
    hideTest() { send({ fn: 'hideTest' }); },
    // --- find a selector: inspect an element, then mark.suggest($0) ---
    suggest(el) {
      if (!el || el.nodeType !== 1) { console.warn('[mark] inspect an element in the Elements panel, then run: mark.suggest($0)'); return; }
      const tag = el.tagName.toLowerCase(), cands = [], esc = s => String(s).replace(/"/g, '\\"');
      const add = (sel, hint) => { let c = 0; try { c = document.querySelectorAll(sel).length; } catch (e) { return; } cands.push({ selector: sel, matches: c, hint }); };
      if (el.id) add('#' + CSS.escape(el.id), 'id — unique');
      const tid = el.getAttribute('data-testid'); if (tid) add('[data-testid="' + esc(tid) + '"]', 'testid — stable');
      const al = el.getAttribute('aria-label');
      if (al) { add(tag + '[aria-label^="' + esc(al.split(/\s+/).slice(0, 2).join(' ')) + '"]', 'aria prefix — stable, generalizes across data'); add('[aria-label="' + esc(al) + '"]', 'exact aria — just this one'); }
      const role = el.getAttribute('role'); if (role) add(tag + '[role="' + esc(role) + '"]', 'role — stable');
      // class selectors — self-hosted/benchmark apps (Postmill etc.) use stable, semantic (BEM) classes; rank descriptive first, sink hashed.
      const hashy = c => /^(css|sc|emotion|jsx|mui)-/i.test(c) || (/[0-9]/.test(c) && !c.includes('-') && !c.includes('__') && c.length <= 10);
      const cscore = c => (hashy(c) ? -9 : 0) + (c.includes('__') ? 3 : 0) + (c.includes('-') ? 1 : 0) + Math.min(2, Math.floor(c.length / 6));
      (el.getAttribute('class') || '').split(/\s+/).filter(Boolean).sort((a, b) => cscore(b) - cscore(a)).forEach(c =>
        add('.' + CSS.escape(c), 'class' + (c.includes('__') ? ' — BEM, descriptive' : (hashy(c) ? ' — looks hashed, avoid' : ' — stable'))));
      add(tag, 'tag / kind — broad');
      const cont = el.closest('[aria-label],[role=region],[role=main],nav,[id]');
      if (cont && cont !== el) { const cs = cont.id ? '#' + CSS.escape(cont.id) : (cont.getAttribute('aria-label') ? cont.tagName.toLowerCase() + '[aria-label="' + esc(cont.getAttribute('aria-label')) + '"]' : null); if (cs) add(cs + ' ' + tag, 'inside "' + (cont.getAttribute('aria-label') || cont.id) + '"'); }
      console.log('%c[mark] selectors for <' + tag + '> — prefer a stable one; check the count:', 'color:#22c55e;font-weight:bold');
      console.table(cands);
      console.log('Then: mark.test("…") to highlight/verify, then mark.untrust("…") / mark.trust("…").');
      return cands;
    },
    // --- commit (template-scoped by default; pass {all:true} for every page, e.g. nav/footer) ---
    untrust: commit('untrust'),
    frame: commit('frame'),
    obscure: commit('obscure'),
    // untrustValue(sel): mask ONLY the last text node of each match — the value in a "Label: Value"
    // element (e.g. <p>Sold by: <#text>Owala</#text></p> → hides "Owala", keeps "Sold by:").
    untrustValue(sel, o) { if (!q(sel)) return; send({ fn: 'untrust', sel, part: 'value', all: !!(o && o.all), page: o && o.page }); console.log('%c[mark] ⊥ value-only (last text node) ' + sel, 'color:#b45309'); },
    // untrustText(sel): mask each match's TEXT, per line (getClientRects), NOT its layout box — so a trusted
    // element floated/positioned inside the box (e.g. a review's star-rating in the body's top-left) stays
    // visible, and EVERY text node is covered (multi-node / <br>-split bodies don't leak the way untrustValue
    // — last-node-only — can). Use when a trusted region sits inside the untrusted element's box; else use untrust.
    untrustText(sel, o) { if (!q(sel)) return; send({ fn: 'untrust', sel, part: 'text', all: !!(o && o.all), page: o && o.page }); console.log('%c[mark] ⊥ text-content, per-line ' + sel, 'color:#b45309'); },
    // trust(x): override untrust for x. x can be a SELECTOR (re-trusts all it matches) or the inspected
    // ELEMENT $0 (re-trusts just that one — handy for peeling off an over-selected extra).
    trust(x, o) {
      if (x && x.nodeType === 1) { const s = uniqueSel(x); send({ fn: 'trust', sel: s, all: !!(o && o.all), page: o && o.page }); console.log('%c[mark] ⊤ (this element) ' + s, 'color:#2563eb'); return s; }
      if (typeof x === 'string') { if (!q(x)) return; send({ fn: 'trust', sel: x, all: !!(o && o.all), page: o && o.page }); return x; }
      console.warn('[mark] pass a selector, or the inspected element: mark.trust($0)');
    },
    exclude(x, o) { return this.trust(x, o); },   // alias — identical to trust; reads better when subtracting an extra
    // --- review what's been marked on THIS page ---
    marked() {
      const u = [...document.querySelectorAll('[data-trust="untrusted"]')];
      const f = [...document.querySelectorAll('[data-trust="frame"]')];
      const o = [...document.querySelectorAll('[data-trust="obscured"]')];
      console.log('%c[mark] this page → untrusted: ' + u.length + ' · frame: ' + f.length + ' · obscured: ' + o.length, 'color:#b45309;font-weight:bold');
      console.log('  untrusted elements:', u);
      if (f.length) console.log('  frame (regime-2):', f);
      send({ fn: 'test', sel: '[data-trust="untrusted"],[data-trust="frame"],[data-trust="obscured"]' });   // highlight them all (magenta)
      console.log('  → highlighted in magenta. mark.masks(true) = black masked view · mark.hideTest() = clear highlights');
      return u.concat(f, o);
    },
    // --- scope: see this page's routeKey so you can pattern a template rule ---
    route() {
      const rk = (location.pathname.replace(/\/\d+/g, '/#').replace(/\/+$/, '') || '/');
      console.log('%c[mark] routeKey: ' + rk + '   (pathname ' + location.pathname + ')', 'color:#0891b2;font-weight:bold');
      console.log('  Scope a rule to a TEMPLATE: mark.untrust(sel, {page:"/f/*/#/**"})  — * = one segment, ** = the rest, # = numeric id.');
      console.log('  Default (no opts) scopes to THIS exact routeKey; {all:true} = every page.');
      return rk;
    },
    // --- manage ---
    rules() { send({ fn: 'rules' }); },
    rm(i) { send({ fn: 'rm', i }); },
    clear() { send({ fn: 'clear' }); },          // clears only THIS template's rules
    clearAll() { send({ fn: 'clearAll' }); },    // clears every rule for this site
    masks(on) { send({ fn: 'masks', on: on !== false }); },       // trusted V⊤: black boxes + contract + number
    marks(on) { send({ fn: 'marks', on: on !== false }); },       // untrusted view: numbers kept, content visible
    export() { send({ fn: 'export' }); },
    help() {
      console.log('%cSecure Mask — console API', 'font-weight:bold;font-size:13px;color:#22c55e');
      console.log('%c0) Find a selector: inspect an element (right-click → Inspect), then:', 'font-weight:bold');
      console.log('   mark.suggest($0)     ranked stable selectors ([id]/[testid]/[aria]/[role]) + match counts');
      console.log('%c1) Verify a selector picks exactly what you intend:', 'font-weight:bold');
      console.log('   mark.test("h2[role=link]")     highlights every match (magenta) + count; returns the NodeList');
      console.log('   mark.count(sel)   mark.hideTest()');
      console.log('%c2) Commit (default = only this template; {all:true} = every page; {page:"pat"} = a template pattern):', 'font-weight:bold');
      console.log('   mark.untrust(sel)   mark.trust(sel | $0)   mark.frame(sel)   mark.obscure(sel)');
      console.log('   e.g.  mark.untrust("h2[role=link]")');
      console.log('   Variable-path templates (e.g. Postmill /f/<forum>/<id>/<slug>): scope with a pattern —');
      console.log('     mark.untrust(".submission__title", {page:"/f/*/#/**"})   (* = one segment, ** = the rest, # = numeric id)');
      console.log('   mark.route()  shows this page\'s routeKey so you can build the pattern.');
      console.log('   Subtract an over-selected extra: inspect it, then mark.trust($0)   (mark.exclude = alias)');
      console.log('   "Label: Value" in one element, mask only the value: mark.untrustValue(sel)');
      console.log('%c3) Review this page:', 'font-weight:bold');
      console.log('   mark.marked()          list + highlight every untrusted element on this page');
      console.log('   mark.masks(true)       show the black masked view (V⊤);  mark.masks(false) to hide');
      console.log('%c4) Manage:', 'font-weight:bold');
      console.log('   mark.rules()   mark.rm(i)   mark.clear() [this template]   mark.clearAll()   mark.export()');
    },
  };
  console.log('%c[mark] console API ready — run mark.help()', 'color:#22c55e;font-weight:bold');
})();
