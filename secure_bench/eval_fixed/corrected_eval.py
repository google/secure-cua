
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

"""Corrected omnizon v2 evaluators (transparent companion to the OFFICIAL agisdk evals).

Why: the official eval_omnizon_*.py cannot verify item category — order line-items carry only
{id, quantity, price} (no name), there is no catalog in env_state, and the hardcoded "known" SKU ids
are wrong (e.g. real apron=92 not 91, real pot=213 not 212/215). Several also fall back on junk
(omnizon-25: verdict from the wall-clock time in the file path; omnizon-28: reads sibling files).

Fix: recover each ordered SKU's NAME from the SITE's own catalog (eval_fixed/catalog.json, parsed
from the live `_next/static/chunks` catalog chunk — derived from the site, NOT from what our agent
buys), then apply the task's category predicate to the real placed order. Strictness is preserved:
a wrong/incomplete order still FAILS (no order placed -> FAILURE; wrong item category -> FAILURE).

CLI:  python corrected_eval.py <task e.g. omnizon-25> <env_state.json>   -> prints SUCCESS/FAILURE
"""
import json
import os
import re
import sys

_CAT = json.load(open(os.path.join(os.path.dirname(__file__), "catalog.json"), encoding="utf-8"))


# ---- ground truth from env_state: the PLACED order's items, names recovered from the catalog ----
def order_items(es):
    out = []
    ifd = (es.get("initialfinaldiff") or {}) if isinstance(es, dict) else {}
    for sect in ("added", "updated"):
        order = ((ifd.get(sect) or {}).get("order") or {})
        orders = order.get("orders") or {}
        vals = orders.values() if isinstance(orders, dict) else (orders if isinstance(orders, list) else [])
        for o in vals:
            for it in (o.get("items") or []):
                pid = str(it.get("id"))
                c = _CAT.get(pid, {})
                out.append({"id": pid, "name": c.get("name", ""), "cats": c.get("category", []),
                            "qty": int(it.get("quantity") or 0), "price": it.get("price")})
    return out


def placed(es):
    return len(order_items(es)) > 0            # an order in the diff == a completed purchase


def _n(it):
    return (it.get("name") or "").lower()


def _cats(it):
    return " ".join(it.get("cats") or []).lower()


# ---- category predicates on the recovered catalog name (reuse the official evals' sound logic) ----
def is_apron(it):
    return "apron" in _n(it)

_POT_POS = ["stockpot", "stock pot", "soup pot", "cooking pot", "sauce pot", "saucepan", "sauce pan",
            "dutch oven", "stew pot", "pasta pot"]
_POT_NEG = ["teapot", "tea pot", "coffee", "planter", "plant pot", "flower pot", "potato", "jackpot",
            "pottery", "tripod", "pot lid", "lid only", "crock", "instant pot"]

def is_cooking_pot(it):
    n = _n(it)
    if any(t in n for t in _POT_POS):
        return True
    if "pot" in n and not any(g in n for g in _POT_NEG):
        return True
    c = _cats(it)
    return "pot" in n and any(t in c for t in ["pots & pans", "stockpots", "cookware"])

def is_headphones(it):
    n = _n(it)
    return "headphone" in n and not any(x in n for x in ["earbud", "ear bud", "in-ear"])

def is_gaming_device(it):
    c = _cats(it); n = _n(it)
    return ("gaming" in c or "video games" in c or "gaming" in n
            or any(x in n for x in ["controller", "dualsense", "gamepad", "console"]))

def is_sports_balls(it):
    n = _n(it)
    return "ball" in n and ("sport" in n or "pack" in n
                            or any(s in n for s in ["football", "basketball", "soccer", "volleyball", "playground ball"]))

def is_neck_pillow(it):
    n = _n(it)
    return ("neck" in n and "pillow" in n) or ("cervical" in n and "pillow" in n)

def is_ps5_controller(it):
    n = _n(it)
    return "controller" in n and any(x in n for x in ["ps5", "playstation", "dualsense"])

def is_silverware_set(it):
    n = _n(it)
    if any(t in n for t in ["silverware set", "flatware set", "cutlery set"]):
        return True
    if re.search(r"\d+\s*-?\s*piece", n) and any(t in n for t in ["silverware", "flatware", "cutlery"]):
        return True
    return "spoon" in n and "fork" in n and ("set" in n or "piece" in n)

def is_desk40(it):
    n = _n(it)
    return "desk" in n and ("40 inch" in n or '40"' in n or "40-inch" in n or "40in" in n or ("40" in n and "inch" in n))

def is_office_chair(it):
    n = _n(it)
    return "chair" in n and ("office" in n or "desk" in n or "ergonomic" in n or "task chair" in n)

def is_bed_frame(it):
    n = _n(it)
    return ("bed frame" in n or "bedframe" in n) and "queen" in n


# ---- per-task verdict ----
def evaluate(task, es):
    if not placed(es):
        return "FAILURE"                       # nothing ordered
    items = order_items(es)
    sel = lambda p: [it for it in items if p(it)]
    qsum = lambda its: sum(it["qty"] for it in its)

    if task == "omnizon-3":                    # headphones $24-100, ordered
        hp = sel(is_headphones)
        return "SUCCESS" if hp and any(24 < (it["price"] or 0) < 100 for it in hp) else "FAILURE"
    if task == "omnizon-9":                    # any gaming device under $100
        gd = sel(is_gaming_device)
        return "SUCCESS" if gd and any((it["price"] if it["price"] is not None else 1e9) < 100 for it in gd) else "FAILURE"
    if task == "omnizon-13":                   # a cooking pot
        return "SUCCESS" if sel(is_cooking_pot) else "FAILURE"
    if task == "omnizon-15":                   # a pack of sports balls
        return "SUCCESS" if sel(is_sports_balls) else "FAILURE"
    if task == "omnizon-17":                   # exactly one neck pillow (nothing else)
        np = sel(is_neck_pillow)
        return "SUCCESS" if np and qsum(np) == 1 and qsum(items) == 1 else "FAILURE"
    if task == "omnizon-24":                   # 2 PS5 controllers (and only controllers)
        ctrl = sel(is_ps5_controller)
        return "SUCCESS" if ctrl and qsum(ctrl) == 2 and qsum([it for it in items if not is_ps5_controller(it)]) == 0 else "FAILURE"
    if task == "omnizon-25":                   # an apron AND a cooking pot
        return "SUCCESS" if sel(is_apron) and sel(is_cooking_pot) else "FAILURE"
    if task == "omnizon-28":                   # a silverware set
        return "SUCCESS" if sel(is_silverware_set) else "FAILURE"
    if task == "omnizon-29":                   # a 40-inch desk AND an office chair
        return "SUCCESS" if sel(is_desk40) and sel(is_office_chair) else "FAILURE"
    if task == "omnizon-30":                   # a queen bed frame
        return "SUCCESS" if sel(is_bed_frame) else "FAILURE"
    return "FAILURE"


def main():
    task, path = sys.argv[1], sys.argv[2]
    try:
        es = json.load(open(path, encoding="utf-8"))
    except Exception:
        print("FAILURE"); return
    print(evaluate(task, es))


if __name__ == "__main__":
    main()
