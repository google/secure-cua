
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

"""Summarize the secure-loop batch: official vs corrected verdict + the actual ordered items per task."""
import json
import os
import paths
import sys

BR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BR, "eval_fixed"))
import corrected_eval as CE  # noqa: E402

ROOT = paths.RESULTS
ORDER = ["omnizon-3", "omnizon-9", "omnizon-13", "omnizon-15", "omnizon-17",
         "omnizon-24", "omnizon-25", "omnizon-28", "omnizon-29", "omnizon-30"]

print(f"{'task':12} {'turns':>5} {'official':>9} {'corrected':>10}   ordered items (id:name xqty)")
print("-" * 100)
off = corr = n = 0
for t in ORDER:
    d = os.path.join(ROOT, f"{t}_v2_secure")
    rp = os.path.join(d, "result.json")
    if not os.path.exists(rp):
        print(f"{t:12} {'--':>5} {'(no run)':>9}")
        continue
    r = json.load(open(rp))
    n += 1
    o = r.get("official_passed")
    c, items = None, "(no order placed)"
    esp = os.path.join(d, "env_state.json")
    if os.path.exists(esp):
        try:
            es = json.load(open(esp))
            c = (CE.evaluate(t, es) == "SUCCESS")     # recompute LIVE (reflects catalog fixes)
            its = CE.order_items(es)
            if its:
                items = "; ".join(f"{i['id']}:{i['name'][:26]} x{i['qty']}" for i in its[:3])
        except Exception:
            pass
    off += 1 if o else 0
    corr += 1 if c else 0
    print(f"{t:12} {str(r.get('turns', '?')):>5} {str(o):>9} {str(c):>10}   {items}")
print("-" * 100)
print(f"official pass: {off}/{n}    corrected pass: {corr}/{n}")
