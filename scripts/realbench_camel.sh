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

# realbench (REAL / omnizon) · CaMeL-CUA. Orchestrates per-(task,Q-LLM-arm) jobs on the shared Playwright pipeline
# (public hosted site — no VM).
# Usage:  scripts/realbench_camel.sh --workers 3 --arms opus gemini gpt --tasks omnizon-3 omnizon-9
set -euo pipefail
. "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
exec "$SGE_VENV" "$SGE_ROOT/secure_bench/run_camel_orchestrator.py" --benchmark realbench "$@"
