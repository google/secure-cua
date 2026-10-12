
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

"""K — the per-screen, mark-keyed typed fact table (trusted-addressed, closed untrusted values) +
the deterministic single-column `compute` reducer. See SECURE_PIPELINE.md §5/§6.

Rows = marks (persistent per-URL numbers from the mask extension). Columns = each query's `bind`
(name + closed type T). Cells = closed answers (bool|enum|int|decimal), possibly wrong. `schema()`
returns names+types+coverage ONLY — never values (the actuator programs over the schema). `compute`
reduces exactly ONE column in trusted Python — never the shadow, never model arithmetic.
"""
import math
import operator

_CMP = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
        "==": operator.eq, "!=": operator.ne}
_ORDERED = {">", ">=", "<", "<="}


def _num(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _truthy(v):
    if isinstance(v, bool):
        return v
    try:
        return float(v) != 0
    except Exception:
        return bool(v)


class KStore:
    def __init__(self):
        self.columns = {}          # bind -> {"type": T, "cells": {mark:int -> value}}
        self.marks_present = []    # marks currently on-screen (from the manifest)

    def set_marks(self, marks):
        self.marks_present = [int(m) for m in marks]

    def declare(self, bind, T):
        if bind in self.columns:
            if self.columns[bind]["type"] != T:
                raise ValueError(f"column {bind!r} already typed {self.columns[bind]['type']}, not {T}")
        else:
            self.columns[bind] = {"type": T, "cells": {}}

    def write(self, bind, mark, value):
        self.columns[bind]["cells"][int(mark)] = value

    def schema(self):
        """Names + types + coverage — NEVER values. This is all the actuator ever sees of K."""
        return {
            "marks": list(self.marks_present),
            "columns": [
                {"name": b, "type": c["type"], "coverage": sorted(c["cells"].keys()), "n_filled": len(c["cells"])}
                for b, c in self.columns.items()
            ],
        }

    def dump(self):
        """Full contents incl. values — for artifacts only, NEVER sent to the actuator."""
        return {b: {"type": c["type"], "cells": {str(m): v for m, v in c["cells"].items()}}
                for b, c in self.columns.items()}

    def discard(self):
        """K is per-screen — clear on navigation (marks reset in the extension via ensureEpoch)."""
        self.columns = {}


def _make_pred(args, T):
    args = args or {}
    if "eq" in args:
        val = args["eq"]
        return lambda v: v == val
    if "in" in args:
        s = set(args["in"])
        return lambda v: v in s
    if "op" in args and "v" in args:
        o, n = args["op"], args["v"]
        f = _CMP.get(o, operator.eq)
        if o in _ORDERED:
            return lambda v: f(_num(v), _num(n))
        return lambda v: f(v, n)
    return lambda v: _truthy(v)     # default: truthy (bool columns)


def compute(store, select, op, args=None):
    """Reduce ONE K column deterministically. Returns a tagged result dict:
      read   -> {"kind":"values","type":T,"values":[{mark,value}...]}   (closed column surfaced)
      filter -> {"kind":"marks","marks":[...]}
      argmin/argmax -> {"kind":"mark","mark":m}    (extreme cell; tie -> lowest mark)
      min/max/count -> {"kind":"scalar","value":..}
      any/all -> {"kind":"scalar","value":bool}
      compare -> {"kind":"scalar","value":-1|0|1}
    """
    col = store.columns.get(select)
    if col is None:
        return {"kind": "error", "error": f"no K column {select!r}"}
    T, cells = col["type"], col["cells"]
    items = sorted(cells.items())      # [(mark, value)] ascending by mark (determinism)

    if op == "read":
        return {"kind": "values", "type": T, "values": [{"mark": m, "value": v} for m, v in items]}

    if op == "filter":
        pred = _make_pred(args, T)
        return {"kind": "marks", "marks": [m for m, v in items if pred(v)]}

    if op in ("argmin", "argmax"):
        if not items:
            return {"kind": "empty"}
        rev = (op == "argmax")
        best_m, best_v = None, None
        for m, v in items:
            nv = _num(v)
            if best_m is None:
                best_m, best_v = m, nv
            elif (nv > best_v) if rev else (nv < best_v):
                best_m, best_v = m, nv
            # ties keep the lower mark (items already ascending) -> no update on ==
        return {"kind": "mark", "mark": best_m}

    if op in ("min", "max"):
        if not items:
            return {"kind": "empty"}
        vals = [_num(v) for _, v in items]
        return {"kind": "scalar", "value": min(vals) if op == "min" else max(vals)}

    if op == "count":
        return {"kind": "scalar", "value": len(items)}

    if op == "any":
        return {"kind": "scalar", "value": any(_truthy(v) for _, v in items)}

    if op == "all":
        return {"kind": "scalar", "value": len(items) > 0 and all(_truthy(v) for _, v in items)}

    if op == "compare":
        args = args or {}
        if "a" in args and "b" in args:
            va, vb = cells.get(int(args["a"])), cells.get(int(args["b"]))
            if va is None or vb is None:
                return {"kind": "error", "error": "compare: missing cell"}
            na, nb = _num(va), _num(vb)
            return {"kind": "scalar", "value": (na > nb) - (na < nb)}
        if "mark" in args and "v" in args:
            v = cells.get(int(args["mark"]))
            if v is None:
                return {"kind": "error", "error": "compare: missing cell"}
            nv, nn = _num(v), _num(args["v"])
            return {"kind": "scalar", "value": (nv > nn) - (nv < nn)}
        return {"kind": "error", "error": "compare needs {a,b} or {mark,v}"}

    return {"kind": "error", "error": f"unknown op {op!r}"}


class GStore:
    """G — the per-task, name-addressed OPAQUE-PAYLOAD store (SECURE_PIPELINE.md §5, global scope).

    Handles hold untrusted STRING payloads the actuator must MOVE but never READ. Unlike K it is
    - NAME-addressed (the agent names the handle), NOT mark-keyed — marks reset per URL, so they can't
      key a cross-screen store; each entry keeps its mark/url only as PROVENANCE;
    - session-persistent — NOT discarded on navigation; a handle accumulates entries across screens.

    `schema()` exposes names + counts + source ONLY (never bytes) — all the actuator ever sees of G.
    `materialize()` joins a WHOLE handle's bytes and is called ONLY by do_return at the user sink.
    `dump()` includes bytes — for artifacts only, NEVER sent to the actuator. `compute` never touches
    G, so an opaque payload has no path back into the actuator's context.
    """

    def __init__(self):
        self.handles = {}          # name -> {"source": str, "entries": [{"entry_id","value","provenance"}]}
        self._next_id = 0

    def declare(self, name, source=""):
        h = self.handles.get(name)
        if h is None:
            self.handles[name] = {"source": source, "entries": []}
        elif source and not h["source"]:
            h["source"] = source

    def append(self, name, value, provenance=None):
        """Add one opaque payload entry to a (declared-on-demand) handle. Returns its entry_id."""
        self.declare(name)
        eid = self._next_id
        self._next_id += 1
        self.handles[name]["entries"].append(
            {"entry_id": eid, "value": str(value), "provenance": dict(provenance or {})})
        return eid

    def has(self, name):
        return name in self.handles

    def schema(self):
        """Names + entry count + source descriptor — NEVER the bytes. All the actuator sees of G."""
        return [{"name": n, "source": h["source"], "n_entries": len(h["entries"])}
                for n, h in self.handles.items()]

    def materialize(self, name, sep=", "):
        """Runtime-only: join the WHOLE handle's payload bytes (dedup-on-equality, order-preserving).
        Returns the joined string, or None if the handle is unknown. Called ONLY by do_return at the
        user sink — never surfaced to the actuator or K."""
        h = self.handles.get(name)
        if h is None:
            return None
        seen, out = set(), []
        for e in h["entries"]:
            v = e["value"]
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return sep.join(out)

    def dump(self):
        """Full contents incl. bytes — for artifacts only, NEVER sent to the actuator."""
        return {n: {"source": h["source"], "entries": h["entries"]} for n, h in self.handles.items()}
