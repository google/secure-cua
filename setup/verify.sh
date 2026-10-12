#!/usr/bin/env bash

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

# Preflight — check every piece is in place before running. Prints PASS/FAIL per component.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; . "$ROOT/env.sh"
pass=0; fail=0
P(){ echo "  [PASS] $1"; pass=$((pass+1)); }
F(){ echo "  [FAIL] $1"; fail=$((fail+1)); }

echo "### host tools ###"
for t in python3 docker git node Xvfb; do command -v "$t" >/dev/null 2>&1 && P "$t" || F "$t (missing — setup/00_prereqs.sh)"; done

echo "### venvs import their key packages ###"
"$SGE_VENV" -c "import playwright, anthropic, openai" 2>/dev/null && P "primary env" || F "primary env (setup/10_venvs.sh)"
# Mirror the runtime import path: evaluation_harness asserts the site URLs at import and uses the pre-1.0
# `openai.error` API, so set the localhost URLs and go through wa_official_eval (it installs the openai.error shim).
REDDIT=http://localhost:9999 SHOPPING=http://localhost:7770 SHOPPING_ADMIN=http://localhost:7780/admin \
GITLAB=http://localhost:8023 WIKIPEDIA=http://localhost:8888 HOMEPAGE=http://localhost:4399 MAP=http://localhost:3000 \
  "$SGE_VENV" -c "import wa_official_eval, browser_env, evaluation_harness" 2>/dev/null \
  && P "WebArena harness (in primary env)" || F "WebArena harness in primary env (setup/10_venvs.sh)"
# agisdk is realbench-ONLY (opt-in via SETUP_REALBENCH=1), so absence is a warn, not a fail — like the realbench
# scorer check below. A WebArena-only setup should reach "READY TO RUN" without it.
if [ -x "$SGE_FETCHED/.agisdk_venv/bin/python" ] && "$SGE_FETCHED/.agisdk_venv/bin/python" -c "import agisdk" 2>/dev/null; then
  P "agisdk env"
else
  echo "  [warn] agisdk env absent (realbench only — SETUP_REALBENCH=1 bash setup.sh)"
fi

echo "### secure_bench imports + relative paths resolve ###"
"$SGE_VENV" -c "import paths, tool_spec, wa_workers, secure_protocol; print(paths.SGE_ROOT)" >/dev/null 2>&1 \
  && P "secure_bench importable, paths resolve" || F "secure_bench import"

echo "### masking extension ###"
[ -s "$ROOT/extension/mask_extension/seed.js" ] && P "mask_extension/seed.js built" || F "seed.js empty (setup/40_extension.sh)"

echo "### WebArena site containers reachable ###"
for p in 9999 7770 7780 8023; do
  curl -s -o /dev/null -m 5 "http://localhost:$p" && P "site on :$p" || F "site on :$p (setup/20_webarena_docker.sh)"
done

echo "### credentials ###"
[ -f "$ROOT/.env" ] && P ".env present" || F ".env missing (cp .env.example .env)"
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
  P "ADC json present"
else
  echo "  [warn] ADC not set — run: gcloud auth application-default login  (Vertex arms only)"
fi

echo "### realbench (optional) ###"
# realbench = the same Playwright pipeline + the agisdk scorer venv (checked above). No VM image.
[ -x "$SGE_FETCHED/.agisdk_venv/bin/python" ] \
  && echo "  [ok]  realbench ready (agisdk scorer venv present; runs on the shared Playwright pipeline)" \
  || echo "  [warn] agisdk scorer venv absent (realbench only — SETUP_REALBENCH=1 bash setup.sh)"

echo
echo "PASS=$pass  FAIL=$fail"
[ "$fail" -eq 0 ] && echo ">>> READY TO RUN" || { echo ">>> fix the [FAIL] items above"; exit 1; }
