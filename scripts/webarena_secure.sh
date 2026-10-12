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

# WebArena · SECURE — masked query/compute/act actuator loop (our defense).
# Usage:  scripts/webarena_secure.sh --model claude-opus-4-8 --tasks reddit-404 shopping-21 [--workers N] [--seeds K] [--skip-done]
#   --model {claude-opus-4-8 | gpt-5 | gemini-3.1-pro-preview}
set -euo pipefail
. "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
exec "$SGE_VENV" "$SGE_ROOT/secure_bench/run_parallel.py" --mode secure --benchmark webarena \
     --out-root "$SGE_ROOT/results/runs" "$@"
