
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

"""Per-task token + wall-time metering for the secure/baseline WebArena drivers.

A LEAF module (no project imports) so secure_llm / llm_adapters / secure_shadow can all import
it without a cycle. One Meter is created per task in each driver's run() and injected into the
Actuator/Shadow (secure) or captured by the baseline call() closure. Recording is split by
dimension so it is correct across retries, errors, and 8 concurrent shadow threads:

  - TIME + call-count are recorded at the provider-neutral boundary (Actuator.turn / Shadow.ask /
    baseline call()) inside a `finally`, so they cover every call_text retry, the SDK's own
    max_retries, backoff sleeps, and any error/early-return path. An errored call still counts as
    1 call with real time and 0 tokens.
  - TOKENS are recorded deep, at each successful create()/call_text return, tagged with the
    caller's component name ("actuator" | "shadow" | "baseline"). This resolves the shared-
    call_text attribution problem: call_text records nothing; it just RETURNS normalized usage,
    and the caller (which statically knows its component) records under the right bucket. Tokens
    are counted once per logical call (only the returning attempt), never per retry.

Common per-component shape: {"calls","input","output","thinking","time"}.

The three normalizers map each provider's asymmetric usage object to (input, output, thinking).
They are null-safe (a missing/None usage object or field yields zeros, never a crash):
  - Anthropic: thinking is folded into output_tokens (not separable)      -> thinking = 0
  - OpenAI:    reasoning_tokens is a SUBSET of completion_tokens           -> do NOT add to output
  - Gemini:    thoughts_token_count is SEPARATE from candidates_token_count-> billed on top of output
The stored shape is raw/uninterpreted; the cost math (collate_metrics.py) applies the two
semantic asymmetries above (billable_out = output + thinking only for Gemini).
"""
import copy
import threading


def _zeros():
    return {"calls": 0, "input": 0, "output": 0, "thinking": 0, "time": 0.0}


class Meter:
    """Thread-safe per-task accumulator. Pre-register the arm's components so the emitted schema is
    stable even when a component makes zero calls (e.g. a secure task with no `query` still emits a
    zeroed "shadow")."""

    def __init__(self, components=()):
        self._c = {name: _zeros() for name in components}
        self._lock = threading.Lock()

    def add_time(self, component, dt):
        with self._lock:
            c = self._c.setdefault(component, _zeros())
            c["time"] += float(dt)
            c["calls"] += 1

    def add_tokens(self, component, inp, out, think=0):
        with self._lock:
            c = self._c.setdefault(component, _zeros())
            c["input"] += int(inp or 0)
            c["output"] += int(out or 0)
            c["thinking"] += int(think or 0)

    def snapshot(self, total_time=None):
        with self._lock:
            out = copy.deepcopy(self._c)
        for c in out.values():
            c["time"] = round(c["time"], 3)
        if total_time is not None:
            out["task_time"] = round(float(total_time), 3)
        return out


def _g(obj, *names):
    """First present, non-None attribute among names, else 0 (null-safe)."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return 0


def norm_anthropic(usage):
    """Anthropic .usage -> (input, output, thinking=0). Thinking is folded into output_tokens."""
    if usage is None:
        return (0, 0, 0)
    return (int(_g(usage, "input_tokens") or 0), int(_g(usage, "output_tokens") or 0), 0)


def norm_openai(usage):
    """OpenAI/Azure .usage -> (prompt, completion, reasoning). reasoning ⊆ completion."""
    if usage is None:
        return (0, 0, 0)
    inp = int(_g(usage, "prompt_tokens") or 0)
    out = int(_g(usage, "completion_tokens") or 0)
    det = getattr(usage, "completion_tokens_details", None)
    think = int(getattr(det, "reasoning_tokens", 0) or 0) if det is not None else 0
    return (inp, out, think)


def norm_gemini(um):
    """Gemini .usage_metadata -> (prompt, candidates, thoughts). thoughts are billed on top."""
    if um is None:
        return (0, 0, 0)
    inp = int(_g(um, "prompt_token_count") or 0)
    out = int(_g(um, "candidates_token_count") or 0)
    think = int(_g(um, "thoughts_token_count") or 0)
    return (inp, out, think)
