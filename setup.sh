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

# One-shot environment build for the artifact. Each step is also runnable on its own (setup/NN_*.sh).
# Realbench (REAL/omnizon) runs through the SAME Playwright pipeline as WebArena; SETUP_REALBENCH=1 just adds
# the separate agisdk scorer venv (built by setup/10). No KVM VM.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
echo "########## secure-gui-eval :: setup ##########"
bash "$D/setup/00_prereqs.sh"
bash "$D/setup/10_venvs.sh"      # SETUP_REALBENCH=1 also builds the .agisdk_venv scorer
bash "$D/setup/40_extension.sh"
bash "$D/setup/20_webarena_docker.sh"
echo
echo ">>> credentials:  cp .env.example .env  &&  edit .env   (Vertex: gcloud auth application-default login)"
echo ">>> verifying ..."
bash "$D/setup/verify.sh" || true
