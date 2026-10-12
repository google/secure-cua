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

# Convenience: run the three masking-family arms (SECURE, BASELINE, NO-QUERY) for a benchmark, then
# print the CaMeL command (it takes different flags: --arms instead of --model).
#
# Both benchmarks take the SAME flags (--model / --tasks / --workers / --seeds); only the ids differ.
# Usage:
#   scripts/run_all.sh webarena  --model claude-opus-4-8 --tasks reddit-404 shopping-21 --workers 3
#   scripts/run_all.sh realbench --model claude-opus-4-8 --tasks omnizon-3 omnizon-9
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
B="${1:?usage: run_all.sh <webarena|realbench> [args...]}"; shift || true
case "$B" in
  webarena|realbench) ;;
  *) echo "unknown benchmark '$B' (use: webarena | realbench)"; exit 1 ;;
esac
for arm in secure baseline noquery; do
  echo "########## $B · $arm ##########"
  "$D/${B}_${arm}.sh" "$@"
done
echo "########## $B · camel — run separately (uses --arms): ##########"
echo "  $D/${B}_camel.sh --arms opus gemini gpt $*"
