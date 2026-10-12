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

# Step 1 — install the Python environments with conda + pip (no venv, no uv). All from requirements.txt.
#   (a) primary env = the conda env you ACTIVATED: the four arms (secure/baseline/noquery/camel) + the official
#       WebArena eval, ALL from the single requirements.txt. No-build: the bundled code (secure_bench, camel,
#       the WebArena harness) runs straight from PYTHONPATH (set by env.sh), so pip installs only deps.
#   (b) agisdk env = the REAL-benchmark scorer — a SEPARATE conda env at $FETCHED/.agisdk_venv (agisdk needs
#       gymnasium 1.x + numpy 2.x, which conflict with the primary env's numpy<2). Realbench arms only, opt-in
#       via SETUP_REALBENCH=1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FETCHED="${SGE_FETCHED:-$ROOT/fetched}"     # relocate fetched DATA off the code drive: set SGE_FETCHED
mkdir -p "$FETCHED"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$FETCHED/.pw-browsers}"   # where the runners look

echo "### clone the WebArena eval harness (runs from PYTHONPATH; env.sh puts it on the import path) ###"
WA="$FETCHED/webarena_official"
if [ ! -d "$WA/.git" ]; then
  git clone https://github.com/web-arena-x/webarena "$WA"
  git -C "$WA" checkout dce04686a56253aefba7b18a4fa0937cf1dc987b 2>/dev/null || \
    echo "  (pinned commit unavailable; using default HEAD)"
fi

# primary env = the conda env you activated (a 3.10-3.12 env). Install the deps into it with pip.
if [ -z "${CONDA_PREFIX:-}" ] || [ "${CONDA_DEFAULT_ENV:-base}" = "base" ]; then
  echo "activate the primary conda env first:  conda create -y -n sge python=3.12 && conda activate sge" >&2
  exit 1
fi
echo "### (a) primary env: conda '$CONDA_DEFAULT_ENV' ($CONDA_PREFIX) ###"
"$CONDA_PREFIX/bin/pip" install --upgrade pip
"$CONDA_PREFIX/bin/pip" install -r "$ROOT/requirements.txt"
"$CONDA_PREFIX/bin/python" -m playwright install chromium

if [ "${SETUP_REALBENCH:-0}" = "1" ]; then
  echo "### (b) agisdk scorer env (separate conda env; realbench only) ###"
  AG="$FETCHED/.agisdk_venv"
  [ -x "$AG/bin/python" ] || conda create -y -p "$AG" python=3.11
  "$AG/bin/pip" install --upgrade pip
  "$AG/bin/pip" install agisdk==0.3.5
  "$AG/bin/python" -m playwright install chromium
else
  echo ">>> skipped agisdk env (realbench only; set SETUP_REALBENCH=1 to include it)"
fi

echo "envs ready: conda '$CONDA_DEFAULT_ENV' (primary)$( [ "${SETUP_REALBENCH:-0}" = 1 ] && echo " + $FETCHED/.agisdk_venv (agisdk)" )"
