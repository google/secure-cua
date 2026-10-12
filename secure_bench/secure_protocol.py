
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

"""Actuator↔runtime protocol: parse the actuator's one-op-per-turn JSON, bound-typecheck shadow
outputs, and render the content-free views the actuator is allowed to see (mark list + K schema).

Security: `coerce` is the type gate on untrusted shadow output — only a closed bool|enum|int|decimal
(or a rejection) ever escapes to K. The render_* helpers emit NO untrusted content — only mark
numbers, content-free contract placeholders, and K's schema (names+types+coverage, never values).
"""
import json
import math
import re

# per-mark query outcome tags (the bounded reply the actuator reads). NOTE: SUCCESS is the NEUTRAL word
# "answered" — it means the reader RETURNED a typed value (which may be true OR false), NOT that the answer
# is yes. The value is hidden in K; the actuator must `compute read`/`filter` to see it before deciding.
SUCCESS, UNKNOWN, ERROR = "answered", "unknown", "error"
OUTCOMES = {SUCCESS, UNKNOWN, ERROR}

OPS = {"query", "compute", "act", "return"}

# Upper bound on one opaque payload's length (chars). Bounds artifact/sink size; DoS hygiene.
PAYLOAD_MAX = 4000


def _last_json_obj(text):
    """Return the last top-level JSON object in text (scans all balanced {...}), or None."""
    dec = json.JSONDecoder()
    best, i = None, 0
    while True:
        j = text.find("{", i)
        if j == -1:
            break
        try:
            obj, end = dec.raw_decode(text, j)
            if isinstance(obj, dict):
                best = obj
            i = end
        except Exception:
            i = j + 1
    return best


def parse_op(text):
    """Extract the actuator's op dict. Prefers a fenced ```json block, else the last JSON object.
    Malformed / no-op → {"op":"error", ...} (a re-prompt cue, never a crash)."""
    if not text or not text.strip():
        return {"op": "error", "error": "empty response"}
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for chunk in ([fenced[-1]] if fenced else []) + [text]:
        obj = _last_json_obj(chunk)
        if obj and obj.get("op") in OPS:
            return obj
    return {"op": "error", "error": "no valid op JSON found", "raw": text[:300]}


def coerce(value, T, enum_values=None):
    """Bounded typecheck of a shadow value against T ∈ {bool,enum,int,decimal}.
    Returns (ok, coerced). Rejects free-form strings, out-of-set enums, non-numeric, NaN/inf."""
    try:
        if T == "bool":
            if isinstance(value, bool):
                return True, value
            if isinstance(value, str):
                s = value.strip().lower()
                if s in ("true", "yes"):
                    return True, True
                if s in ("false", "no"):
                    return True, False
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
                return True, bool(value)
            return False, None
        if T == "int":
            if isinstance(value, bool):
                return False, None
            iv = int(value) if not isinstance(value, str) else int(round(float(value.strip())))
            return True, iv
        if T == "decimal":
            if isinstance(value, bool):
                return False, None
            s = value.strip().replace("$", "").replace(",", "") if isinstance(value, str) else value
            fv = float(s)
            if math.isnan(fv) or math.isinf(fv):
                return False, None
            return True, fv
        if T == "enum":
            if not enum_values:
                return False, None
            if value in enum_values:
                return True, value
            if isinstance(value, str):
                for e in enum_values:
                    if str(e).strip().lower() == value.strip().lower():
                        return True, e
            return False, None
    except Exception:
        return False, None
    return False, None


def bound_payload(value, cap=PAYLOAD_MAX):
    """Bound an OPAQUE string payload from the shadow (return_type=payload). Unlike `coerce`, this
    ACCEPTS free-form text — the payload is stored in G and the actuator NEVER reads it (it is only
    materialized at the user sink), so it need not be a closed value. We only sanitize: coerce to str,
    strip control chars (keep \\n and \\t), cap length. Returns (ok, cleaned)."""
    if value is None:
        return False, None
    s = value if isinstance(value, str) else str(value)
    s = "".join(ch for ch in s if ch in "\n\t" or (32 <= ord(ch) != 127)).strip()
    if not s:
        return False, None
    return True, s[:cap]


def render_marks(manifest):
    """Content-free mark list for the actuator: #n <kind|size|origin|clickable> at screen(sx,sy) size."""
    if not manifest:
        return "marks: (none masked on this screen)"
    lines = ["marks (content-free — you never see the content behind a mask):"]
    for r in sorted(manifest, key=lambda x: x.get("n", 0)):
        lines.append(f"  #{r['n']} {r['tag']} @screen({r['sx']},{r['sy']}) size {r['w']}x{r['h']}")
    return "\n".join(lines)


def render_ktable_schema(store):
    """K's schema for the actuator — names + types + coverage, NEVER cell values."""
    s = store.schema()
    lines = [f"K rows (marks with facts): screen has marks {s['marks']}"]
    if not s["columns"]:
        lines.append("K columns: (none yet — issue a query to create one)")
    else:
        lines.append("K columns  (name : type : marks answered):")
        for c in s["columns"]:
            lines.append(f"  {c['name']} : {c['type']} : answered={c['coverage']} ({c['n_filled']})")
    return "\n".join(lines)


def render_gstore_schema(gstore):
    """G's schema for the actuator — handle NAMES + entry counts + source, NEVER the payload bytes.
    Opaque payloads the actuator may MOVE (route to the user via `return`) but never READ."""
    cols = gstore.schema()
    if not cols:
        return ("G payload handles (opaque strings you can MOVE but never READ): (none yet — a "
                "`query return_type=payload` creates one; `return` routes it to the user)")
    lines = ["G payload handles (opaque strings — you can route them to the user with `return`, "
             "but NEVER read them and CANNOT `compute` on them):"]
    for c in cols:
        src = f", from {c['source']}" if c.get("source") else ""
        n = c["n_entries"]
        lines.append(f"  {c['name']} : payload (untrusted{src}) : {n} entr" + ("y" if n == 1 else "ies"))
    return "\n".join(lines)
