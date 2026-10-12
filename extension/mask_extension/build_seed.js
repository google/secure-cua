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

/* Merge every specs/*.rules.json into a multi-site seed the extension bundles.
   Regenerate after editing/exporting any spec:  node build_seed.js
   Emits  self.__SMASK_SEED = [{site, templates}, ...]  (content.js picks the entry matching location.host). */
const fs = require("fs"), path = require("path");
const dir = "specs";
const files = fs.readdirSync(dir).filter(f => f.endsWith(".rules.json")).sort();
const specs = files.map(f => JSON.parse(fs.readFileSync(path.join(dir, f), "utf8")));

// content.js does seed.find(s => s.site === HOST) -> ONE entry per host. So when a host's page
// set is a UNION of two rule sources (e.g. localhost:7780 serves BOTH the Magento admin AND the
// same storefront theme as 7770), merge at build time. A spec may list extra hosts via "also_sites":
// its templates are appended to those hosts' entries too — storefront rules stay defined once (DRY).
const bySite = {};                       // host -> {site, templates}
function addSpec(host, templates) {
  if (!bySite[host]) bySite[host] = { site: host, templates: {} };
  const dst = bySite[host].templates;
  for (const [k, arr] of Object.entries(templates || {})) dst[k] = (dst[k] || []).concat(arr);
}
for (const s of specs) {
  addSpec(s.site, s.templates);
  for (const extra of (s.also_sites || [])) addSpec(extra, s.templates);
}

// SAME-SITE PARALLELISM: a worker task runs its OWN container on a derived port (base + 10000*i) so K tasks
// on one site run concurrently, each with its own per-task reset + mutation isolation. The mask engine keys
// rules by location.host (host:PORT), so replicate each localhost site's ALREADY-MERGED rules to its worker
// ports — masking is then identical on every worker. The port formula (base + 10000*i, i in 1..MAX_WORKERS-1)
// MUST match the Python harness (real_bridge worker-port derivation). Non-localhost specs (omnizon) untouched.
const MAX_WORKERS = 6;
for (const host of Object.keys(bySite)) {            // snapshot: worker hosts added below are not re-processed
  const m = host.match(/^localhost:(\d+)$/);
  if (!m) continue;
  const base = parseInt(m[1], 10);
  const tmpl = bySite[host].templates;               // the merged (base ∪ also_sites) ruleset for this site
  for (let i = 1; i < MAX_WORKERS; i++) {
    const wport = base + 10000 * i;
    if (wport > 65535) break;
    addSpec("localhost:" + wport, tmpl);             // concat => deep copy of the rule arrays (no shared refs)
  }
}
const sites = Object.values(bySite);

fs.writeFileSync("seed.js",
  "/* GENERATED from specs/*.rules.json — do not hand-edit. Regenerate: node build_seed.js */\n" +
  "self.__SMASK_SEED = " + JSON.stringify(sites) + ";\n");
console.log("seed.js regenerated from " + files.length + " spec(s): " + files.map(f => f.replace(".rules.json", "")).join(", "));
for (const s of sites) console.log("  " + s.site + "  (" + (s.templates["*"] || []).length + " rules in *)");
