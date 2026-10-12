
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

"""No-query op set for the ABLATION pipeline: only `act` and `return`.

These two entries are copied VERBATIM from tool_spec.py (the `act` and `return` schemas) so the runtime's
op-dict fields stay identical — the no-query fork reuses the same `do_act` / `do_return` and the same
secure_protocol parsing. The `query` and `compute` entries are deliberately absent: this actuator has no
read channel. tool_spec.py itself is left untouched; the no-query runner swaps `tool_spec.TOOLS` for the
list below before building its adapter.
"""

TOOLS = [
    {
        "name": "act",
        "description": (
            "Do ONE thing in the browser (the only step that changes the page), by COORDINATE on the trusted "
            "frame or a mark. The runtime egress-checks it must land on a first-party control/route, "
            "else it is refused. `type` text must be a literal you author, never content read off the screen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["click", "type", "press", "scroll", "go_back"]},
                "target": {
                    "type": "object",
                    "description": "for click: {x,y} screen pixels OR {mark:N}",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "mark": {"type": "integer"}},
                    "additionalProperties": False,
                },
                "text": {"type": "string", "description": "literal to type (for action=type)"},
                "key": {"type": "string", "description": "key/combo for action=press, e.g. enter or ctrl+a"},
                "dy": {"type": "integer", "description": "scroll amount for action=scroll: NEGATIVE scrolls DOWN "
                       "(toward content further down the page), positive scrolls up. Scrolls at screen center "
                       "unless you also give target {x,y}."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "return",
        "description": (
            "END the task (the sole terminal op). `answer` is your terminal message — a SUMMARY of what you "
            "accomplished (for an action task) OR the ANSWER to report (for a question); it is written to the "
            "user. You have no way to read masked content, so `answer` is always just your literal text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "give_up"]},
                "answer": {"type": "string", "description": "your terminal message to the user"},
            },
            "required": ["status", "answer"],
        },
    },
]

TOOL_BY_NAME = {t["name"]: t for t in TOOLS}


def op_from_toolcall(name, args):
    """Normalize a provider tool-call into the runtime's op dict (identical to tool_spec.op_from_toolcall)."""
    return {"op": name, **(args or {})}
