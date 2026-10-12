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

# Step 4 — (re)build the masking extension's bundled seed from the per-site rule specs.
# `node build_seed.js` merges extension/mask_extension/specs/*.rules.json into seed.js (with the
# worker-port replication that keeps parallel same-site containers masking identically).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/extension/mask_extension"
node build_seed.js
echo "built seed.js ($(wc -c < seed.js) bytes) from specs: $(ls specs/*.rules.json | xargs -n1 basename | tr '\n' ' ')"
