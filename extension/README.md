# Secure-Mask browser extension

The masking defense's front end. A Chromium MV3 extension that, on the live page, stamps every
untrusted region (`data-trust=untrusted|frame|obscured|trusted`) and paints reversible black overlays,
so the agent's screenshot **V⊤** (trusted view) hides third-party-controllable content while the isolated
**V_g** view can reveal one region at a time. The `secure` and `noquery` WebArena runners load it; the
`baseline` runner does **not** (that's the unmasked control).

## Files
| file | role |
|---|---|
| `mask_extension/manifest.json` | MV3 manifest (content scripts, host matches: localhost / 127.0.0.1) |
| `mask_extension/content.js` | the masking **engine** — loads per-host rules, stamps `data-trust`, paints masks, exposes the key-gated `maskAll`/`revealAll`/`isolate` control channel used to produce V⊤ vs V_g |
| `mask_extension/content_main.js` | MAIN-world shim exposing the `window.mark` console API used to author rules |
| `mask_extension/specs/<site>.rules.json` | **the checked-in source of truth** — per-site rules `{site, templates:{routeKey:[{sel,trust}]}}` for reddit / shopping / shopping_admin / gitlab / omnizon |
| `mask_extension/build_seed.js` | build tool: merges `specs/*.rules.json` → `seed.js` |
| `mask_extension/seed.js` | **generated** bundle (`self.__SMASK_SEED`) — do not hand-edit |

## Build
```bash
cd extension/mask_extension && node build_seed.js      # or: bash ../../setup/40_extension.sh
```
`build_seed.js` merges the specs and replicates each `localhost:PORT` site's rules to the derived worker
ports (`base + 10000*i`), so same-site parallel containers mask identically.

## How the runners use it
`run_secure_webarena.py` / `run_noquery_webarena.py` launch headed Chromium with
`--load-extension=<mask_extension>` and drive it over CDP via `secure_bench/smask_runtime.py`
(`Smask.mask()/isolate()/manifest()`), taking a masked **V⊤** screenshot for the actuator and, on a
`query`, an isolated **V_g** screenshot for the single region being read.

## Authoring rules for a new site/page (only if you add tasks)
1. Open the page, DevTools console:
   - `mark.suggest($0)` — get a stable selector for the selected element
   - `mark.test(sel)` — highlight what it matches
   - `mark.untrust(sel, {page:"<routeKey>"})` — commit an untrusted rule (page-scoped)
   - `mark.export()` — dump the rule JSON
2. Save the JSON into `specs/<site>.rules.json`, then rebuild with `node build_seed.js`.
