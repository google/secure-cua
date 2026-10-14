"""Per-tool-call execution log for CaMeL UI runs (opt-in via env CAMEL_STEP_LOG=1).

The interpreter's single call chokepoint (_eval_call) is where the plan LINE, the tool, and the RESULT
are all known; the base_ui's screenshot save is where the SCREEN is captured. Neither side can see the
other, so both write here:

  interpreter._eval_call:   begin(lineno, src, tool)  ... then  end(result)
  base_ui.take_and_save_*:  attach_screenshot(path)      (lands on the currently-open call)

-> writes one {seq,line,src,tool,result,shots[]} record per tool-call to <run_dir>/step_log.jsonl, so
the viewer can show "the plan line that ran at each step, its result/reasoning, and its screenshot(s)".
Zero effect unless start() ran (which only activates when CAMEL_STEP_LOG is set)."""

import json
import os
import threading

_lock = threading.Lock()
_S = {"dir": None, "seq": 0, "cur": None}

# Noise: constructors / builtins that aren't meaningful "steps" (Instruction() wraps a find arg, etc.).
_SKIP = {"Instruction", "len", "str", "int", "float", "range", "list", "dict", "set", "tuple",
         "bool", "abs", "min", "max", "sum", "sorted", "enumerate", "zip", "no_op"}


def start(d):
    """Enable logging into run dir `d` (only if CAMEL_STEP_LOG is set). Resets the log."""
    if not os.environ.get("CAMEL_STEP_LOG"):
        return
    try:
        os.makedirs(str(d), exist_ok=True)
        open(os.path.join(str(d), "step_log.jsonl"), "w").close()
        _S["dir"], _S["seq"], _S["cur"] = str(d), 0, None
    except Exception:
        _S["dir"] = None


def _on():
    return _S["dir"] is not None


def begin(lineno, src, tool):
    if not _on() or (tool in _SKIP):
        _S["cur"] = None
        return
    _S["seq"] += 1
    _S["cur"] = {"seq": _S["seq"], "line": lineno, "src": (src or "")[:400], "tool": tool, "shots": []}


def attach_screenshot(path):
    if not _on() or _S["cur"] is None or not path:
        return
    _S["cur"]["shots"].append(os.path.basename(str(path)))


def end(result):
    if not _on() or _S["cur"] is None:
        return
    c = _S["cur"]
    _S["cur"] = None
    try:
        c["result"] = str(result)[:600]
        with _lock, open(os.path.join(_S["dir"], "step_log.jsonl"), "a") as f:
            f.write(json.dumps(c, default=str) + "\n")
    except Exception:
        pass
