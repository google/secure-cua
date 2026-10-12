
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

"""Provider-agnostic actuator backends. One interface, four implementations.

    Adapter.step(text, image_b64=None) -> (op_dict, reasoning_text)

`op_dict` is the runtime's op ({"op": <name>, ...args}) — identical whether the model produced it via
a native tool-call (Anthropic / OpenAI / Gemini) or as text-JSON. The adapter owns the provider's
conversation state + tool wiring so the Actuator (secure_actuator.py) stays provider-neutral.

Providers:
  - "text"     TextProtocolAdapter  — the portable path: no tools, model emits ```json, parse_op().
                                       Byte-for-byte the behavior validated in the 10-task batch. DEFAULT.
  - "anthropic" AnthropicToolAdapter — AnthropicVertex native tools (input_schema), adaptive thinking.
  - "openai"    OpenAIToolAdapter    — OpenAI function tools (tool_calls / role:tool).
  - "gemini"    GeminiToolAdapter    — google.genai function_declarations (function_call/response).

The 4 tools come from tool_spec.TOOLS (defined once, portable JSON Schema). Only the Anthropic path is
credential-testable in this environment; OpenAI/Gemini follow each SDK's documented function-calling
pattern and need OPENAI_API_KEY / GEMINI creds to smoke-test.
"""
import json
import os

import secure_protocol as P
import tool_spec as TS

import metrics as M

_MAX_TOKENS = 8000


def _is_reasoning(model):
    """OpenAI reasoning models (gpt-5, o-series): take reasoning_effort, reject temperature/top_p."""
    m = (model or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _secure_max_hist(_cache={}):
    """SECURE_MAX_HIST=N -> each actuator API call carries only the last N (observation, response) TURNS
    of history (mirrors run_baseline_webarena.BASELINE_MAX_TRAJ). Unset/empty -> None -> FULL history:
    byte-for-byte today's default behavior. Windowing is applied non-destructively at call time (the
    adapter still accumulates the full episode internally) and at TURN boundaries, de-orphaning the
    leading tool_result/function_response so provider tool-pairing stays valid. Read once per process."""
    if "v" not in _cache:
        s = (os.getenv("SECURE_MAX_HIST") or "").strip()
        try:
            _cache["v"] = int(s) if s else None
        except ValueError:
            _cache["v"] = None
    return _cache["v"]


class Adapter:
    provider = "base"

    def step(self, text, image_b64=None):
        """Send one observation (feedback text + optional screenshot); return (op_dict, reasoning)."""
        raise NotImplementedError

    @staticmethod
    def _img_data_uri(b64):
        return f"data:image/png;base64,{b64}"


# ------------------------------------------------------------------ text (portable, default) --------
class TextProtocolAdapter(Adapter):
    """No native tools. System folded into the first user message; model emits one JSON op; parse_op().
    Kept identical to the original Actuator so provider='text' reproduces the batch-validated behavior."""
    provider = "text"

    def __init__(self, system, model=None, max_tokens=_MAX_TOKENS, log=print, meter=None):
        import secure_llm as L
        self._L = L
        self.meter = meter
        self.client = L.vertex_client()
        self.model = model or L.MODEL
        self.max_tokens = max_tokens
        self.log = log
        self.system = system
        self.messages = []
        self._first = True
        self._maxhist = _secure_max_hist()
        if self._maxhist is not None:
            log(f"[text-adapter] SECURE_MAX_HIST={self._maxhist} — history windowed to last {self._maxhist} turns")

    def _call_messages(self):
        """Full self.messages, or (SECURE_MAX_HIST set) messages[0] (carries the folded system prompt — the
        anchor, always kept) + the last N turns. messages alternate user/assistant and end with the current
        user; the tail slice [-(2N):] starts on an assistant, so [msgs[0]] + tail alternates cleanly."""
        n = self._maxhist
        if n is None or len(self.messages) <= 2 * n + 1:
            return self.messages
        return self.messages[:1] + self.messages[-(2 * n):]

    def step(self, text, image_b64=None):
        parts = []
        if self._first:
            parts.append({"type": "text", "text": self.system})
            self._first = False
        parts.append({"type": "text", "text": text})
        if image_b64:
            parts.append(self._L.img_block(image_b64))
        self.messages.append({"role": "user", "content": parts})
        try:
            raw, usage = self._L.call_text(self.client, self._call_messages(), self.model, self.max_tokens)
        except Exception as e:
            self.log(f"[text-adapter] call failed: {e}")
            raw, usage = "", None
        self.messages.append({"role": "assistant", "content": raw or "(no response)"})
        if self.meter is not None and usage:
            self.meter.add_tokens("actuator", *usage)
        return P.parse_op(raw), raw


# ------------------------------------------------------------------ Anthropic native tools ----------
class AnthropicToolAdapter(Adapter):
    """AnthropicVertex with native tools + adaptive thinking. tool_use -> op; the next observation is
    delivered as a tool_result (feedback text + screenshot). Thinking blocks are preserved in history
    (required for replay). tool_choice defaults to auto (forced choice is incompatible with thinking);
    a text-only turn falls back to parse_op."""
    provider = "anthropic"

    def __init__(self, system, model=None, max_tokens=_MAX_TOKENS, log=print, meter=None):
        import secure_llm as L
        self.client = L.vertex_client()
        self.model = model or L.MODEL
        self.system = system
        self.max_tokens = max_tokens
        self.log = log
        self.meter = meter
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in TS.TOOLS]
        self.messages = []
        self._pending = []          # tool_use ids owed a tool_result next turn
        self._maxhist = _secure_max_hist()
        if self._maxhist is not None:
            log(f"[anthropic-adapter] SECURE_MAX_HIST={self._maxhist} — history windowed to last {self._maxhist} turns")

    def _img(self, b64):
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}

    def _call_messages(self):
        """Full self.messages, or (if SECURE_MAX_HIST set) the last N turns + current obs. self.messages
        alternates user/assistant with no system in the list and ends with the just-appended user obs, so
        the last (2N+1) entries = N complete turns + the current obs, and slot 0 is always a user message.
        If that leading user leads with a tool_result (its owning assistant is now dropped), unwrap it to the
        plain obs content so no tool_result is orphaned."""
        n = self._maxhist
        if n is None or len(self.messages) <= 2 * n + 1:
            return self.messages
        w = self.messages[-(2 * n + 1):]
        c = w[0].get("content")
        if w[0].get("role") == "user" and isinstance(c, list) and c and isinstance(c[0], dict) \
                and c[0].get("type") == "tool_result":
            w = [{"role": "user", "content": list(c[0].get("content", []))}] + w[1:]
        return w

    def step(self, text, image_b64=None):
        if self._pending:
            content = []
            for i, tid in enumerate(self._pending):
                if i == 0:
                    inner = [{"type": "text", "text": text}]
                    if image_b64:
                        inner.append(self._img(image_b64))
                    content.append({"type": "tool_result", "tool_use_id": tid, "content": inner})
                else:                # only one op runs per turn; stub the extras so the API stays consistent
                    content.append({"type": "tool_result", "tool_use_id": tid,
                                    "content": [{"type": "text", "text": "(ignored — one op per turn)"}]})
        else:
            content = [{"type": "text", "text": text}]
            if image_b64:
                content.append(self._img(image_b64))
        self.messages.append({"role": "user", "content": content})
        try:
            r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                            system=self.system, tools=self.tools, messages=self._call_messages(),
                                            extra_body={"thinking": {"type": "adaptive"}})
        except Exception as e:
            self.log(f"[anthropic-adapter] call failed: {e}")
            self._pending = []
            return {"op": "error", "reason": str(e)}, ""
        self.messages.append({"role": "assistant", "content": r.content})
        if self.meter is not None:
            self.meter.add_tokens("actuator", *M.norm_anthropic(r.usage))
        tool_uses = [b for b in r.content if getattr(b, "type", None) == "tool_use"]
        reasoning = " ".join(b.text for b in r.content if getattr(b, "type", None) == "text").strip()
        self._pending = [b.id for b in tool_uses]
        if tool_uses:
            b = tool_uses[0]
            return TS.op_from_toolcall(b.name, dict(b.input or {})), reasoning
        return P.parse_op(reasoning), reasoning     # text-only turn (rare) — portable fallback


# ------------------------------------------------------------------ OpenAI native tools -------------
class OpenAIToolAdapter(Adapter):
    """OpenAI / Azure-OpenAI chat.completions with function tools. tool_calls -> op; the next observation
    is a role:tool result (feedback) followed by a user message carrying the screenshot (tool messages
    can't hold images). Uses AzureOpenAI when AZURE_OPENAI_ENDPOINT is set (model = the Azure DEPLOYMENT
    name, e.g. gpt-5). Reasoning models (gpt-5/o*) get reasoning_effort and omit temperature; tool_choice
    is 'required' so the model emits a native function call each turn."""
    provider = "openai"

    def __init__(self, system, model=None, max_tokens=_MAX_TOKENS, log=print, meter=None):
        import openai
        self.meter = meter
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.client = openai.AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                             api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                                             timeout=120.0, max_retries=2)          # bound a black-holed GPT call
            self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5"
        else:
            self.client = openai.OpenAI(timeout=120.0, max_retries=2)
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.log = log
        self._reasoning = _is_reasoning(self.model)
        self.tools = [{"type": "function",
                       "function": {"name": t["name"], "description": t["description"],
                                    "parameters": t["parameters"]}} for t in TS.TOOLS]
        self.messages = [{"role": "system", "content": system}]
        self._pending = []          # tool_call ids owed a role:tool reply next turn
        self.last_via = None        # "tool" if the last op arrived via a native tool_call, else "text"
        self._maxhist = _secure_max_hist()
        if self._maxhist is not None:
            log(f"[openai-adapter] SECURE_MAX_HIST={self._maxhist} — history windowed to last {self._maxhist} turns")

    def _call_messages(self):
        """system + (if SECURE_MAX_HIST set) the last N turns. Turn boundaries = assistant messages; the
        window opens right after an assistant, so its leading role:tool obs is orphaned (owning assistant
        dropped) -> re-express as a user text message. Interior tool/assistant pairs stay intact."""
        n = self._maxhist
        if n is None:
            return self.messages
        body = self.messages[1:]
        asst = [i for i, m in enumerate(body) if m.get("role") == "assistant"]
        if len(asst) <= n:
            return self.messages
        win = body[asst[-(n + 1)] + 1:]
        out, lead = [], True
        for m in win:
            if lead and m.get("role") == "tool":
                out.append({"role": "user", "content": m.get("content", "")})   # orphaned obs -> user text
                continue
            lead = False
            out.append(m)
        return self.messages[:1] + out

    def step(self, text, image_b64=None):
        for i, tid in enumerate(self._pending):
            self.messages.append({"role": "tool", "tool_call_id": tid,
                                  "content": text if i == 0 else "(ignored — one op per turn)"})
        user_content = []
        if not self._pending:
            user_content.append({"type": "text", "text": text})
        if image_b64:
            user_content.append({"type": "image_url", "image_url": {"url": self._img_data_uri(image_b64)}})
        if user_content:
            self.messages.append({"role": "user", "content": user_content})
        mct = max(self.max_tokens, 16000) if self._reasoning else self.max_tokens  # reasoning tokens count too
        kw = dict(model=self.model, messages=self._call_messages(), tools=self.tools,
                  tool_choice="required", max_completion_tokens=mct)
        if self._reasoning and getattr(self, "_send_reff", True):
            kw["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "medium")
        try:
            r = self.client.chat.completions.create(**kw)
        except Exception as e:
            # some reasoning models (e.g. gpt-5.5+) reject reasoning_effort alongside function tools — drop it,
            # remember not to send it again (avoids re-erroring every turn), and retry once.
            if "reasoning_effort" in kw and "reasoning_effort" in str(e):
                self._send_reff = False
                kw.pop("reasoning_effort", None)
                try:
                    r = self.client.chat.completions.create(**kw)
                except Exception as e2:
                    self.log(f"[openai-adapter] call failed (retry w/o reasoning_effort): {e2}")
                    self._pending = []
                    return {"op": "error", "reason": str(e2)}, ""
            else:
                self.log(f"[openai-adapter] call failed: {e}")
                self._pending = []
                return {"op": "error", "reason": str(e)}, ""
        msg = r.choices[0].message
        self.messages.append(msg.model_dump(exclude_none=True))   # replay-safe dict (preserves tool_calls)
        if self.meter is not None:
            self.meter.add_tokens("actuator", *M.norm_openai(r.usage))
        tcs = msg.tool_calls or []
        reasoning = msg.content or ""
        self._pending = [tc.id for tc in tcs]
        if tcs:
            self.last_via = "tool"
            tc = tcs[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            return TS.op_from_toolcall(tc.function.name, args), reasoning
        self.last_via = "text"
        return P.parse_op(reasoning), reasoning


# ------------------------------------------------------------------ Gemini native tools -------------
class GeminiToolAdapter(Adapter):
    """google.genai with function_declarations. function_call -> op; the next observation is a
    function_response part plus an inline screenshot. Model via GEMINI_MODEL (default gemini-2.5-pro).
    Creds: GEMINI_API_KEY/GOOGLE_API_KEY, else Vertex (GOOGLE_CLOUD_PROJECT/LOCATION)."""
    provider = "gemini"

    def __init__(self, system, model=None, max_tokens=_MAX_TOKENS, log=print, meter=None):
        from google import genai
        from google.genai import types
        self._types = types
        self.meter = meter
        _ho = {"timeout": 120000}                                        # ms; bound a black-holed Gemini call
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"), http_options=_ho)
        else:
            self.client = genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                                       location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"), http_options=_ho)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.log = log
        self._as_string = {}        # tool -> set(param names re-encoded as JSON strings for Gemini)
        decls = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                           parameters=self._schema(t["name"], t["parameters"])) for t in TS.TOOLS]
        # mode=ANY forces the model to answer with a function CALL (never free text) — this is what
        # makes Gemini use the native tool path instead of narrating JSON. Thinking still runs (the
        # thought precedes the forced call; thought_signatures are preserved in `contents`).
        tool_cfg = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
        self.config = types.GenerateContentConfig(system_instruction=system,
                                                  tools=[types.Tool(function_declarations=decls)],
                                                  tool_config=tool_cfg,
                                                  max_output_tokens=max_tokens)
        self.contents = []
        self._pending = []          # (name) of function_calls owed a function_response next turn
        self.last_via = None        # "tool" if the last op arrived via a native function_call, else "text"
        self._maxhist = _secure_max_hist()
        if self._maxhist is not None:
            log(f"[gemini-adapter] SECURE_MAX_HIST={self._maxhist} — history windowed to last {self._maxhist} turns")

    def _call_contents(self):
        """Full self.contents, or (SECURE_MAX_HIST set) the last N turns + current obs. contents alternate
        user/model and end with the current user obs; the last (2N+1) = N turns + current, slot 0 a user.
        Unwrap a leading function_response (its function_call is dropped) into a plain text part."""
        n = self._maxhist
        if n is None or len(self.contents) <= 2 * n + 1:
            return self.contents
        types = self._types
        w = list(self.contents[-(2 * n + 1):])
        first = w[0]
        parts = list(getattr(first, "parts", None) or [])
        if getattr(first, "role", None) == "user" and parts and getattr(parts[0], "function_response", None):
            try:
                txt = (parts[0].function_response.response or {}).get("result", "") or ""
            except Exception:
                txt = ""
            newparts = [types.Part.from_text(text=txt or "(prior observation)")]
            newparts += [p for p in parts[1:] if not getattr(p, "function_response", None)]
            w = [types.Content(role="user", parts=newparts)] + w[1:]
        return w

    # JSON Schema -> Gemini Schema dict (uppercase types; free-form objects become JSON strings)
    _T = {"object": "OBJECT", "string": "STRING", "array": "ARRAY",
          "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}

    def _schema(self, tool, s, _top=None):
        top = tool if _top is None else _top
        t = s.get("type", "string")
        if t == "object":
            props = s.get("properties") or {}
            if not props or s.get("additionalProperties"):    # free-form -> STRING (model emits JSON text)
                if _top is None:                              # can't stringify the whole tool; keep object
                    pass
                else:
                    return {"type": "STRING", "description": (s.get("description", "") + " (JSON object)").strip()}
            out = {"type": "OBJECT", "properties": {}}
            for k, v in props.items():
                if (v.get("type") == "object") and (not v.get("properties") or v.get("additionalProperties")):
                    self._as_string.setdefault(top, set()).add(k)
                    out["properties"][k] = {"type": "STRING",
                                            "description": (v.get("description", "") + " (JSON object)").strip()}
                else:
                    out["properties"][k] = self._schema(tool, v, top)
            if s.get("required"):
                out["required"] = s["required"]
            if s.get("description"):
                out["description"] = s["description"]
            return out
        if t == "array":
            items = s.get("items") or {"type": "string"}
            return {"type": "ARRAY", "items": self._schema(tool, items, top),
                    **({"description": s["description"]} if s.get("description") else {})}
        out = {"type": self._T.get(t, "STRING")}
        for k in ("description", "enum"):
            if s.get(k):
                out[k] = s[k]
        return out

    def _normalize(self, name, args):
        args = dict(args or {})
        for k in self._as_string.get(name, ()):               # decode the JSON strings back to objects
            if isinstance(args.get(k), str):
                try:
                    args[k] = json.loads(args[k])
                except Exception:
                    pass
        return TS.op_from_toolcall(name, args)

    def step(self, text, image_b64=None):
        import base64
        types = self._types
        parts = []
        if self._pending:
            for i, nm in enumerate(self._pending):
                resp = {"result": text} if i == 0 else {"result": "(ignored — one op per turn)"}
                parts.append(types.Part.from_function_response(name=nm, response=resp))
        else:
            parts.append(types.Part.from_text(text=text))
        if image_b64:
            parts.append(types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/png"))
        self.contents.append(types.Content(role="user", parts=parts))
        try:
            r = self.client.models.generate_content(model=self.model, contents=self._call_contents(), config=self.config)
        except Exception as e:
            self.log(f"[gemini-adapter] call failed: {e}")
            self._pending = []
            return {"op": "error", "reason": str(e)}, ""
        cand = (r.candidates or [None])[0]
        cparts = (cand.content.parts if cand and cand.content else []) or []
        self.contents.append(types.Content(role="model", parts=cparts))
        if self.meter is not None:
            self.meter.add_tokens("actuator", *M.norm_gemini(r.usage_metadata))
        calls = [p.function_call for p in cparts if getattr(p, "function_call", None)]
        reasoning = " ".join(p.text for p in cparts if getattr(p, "text", None)).strip()
        self._pending = [c.name for c in calls]
        if calls:
            self.last_via = "tool"
            c = calls[0]
            return self._normalize(c.name, dict(c.args or {})), reasoning
        self.last_via = "text"
        return P.parse_op(reasoning), reasoning


_ADAPTERS = {"text": TextProtocolAdapter, "anthropic": AnthropicToolAdapter,
             "openai": OpenAIToolAdapter, "gemini": GeminiToolAdapter}


def make_adapter(provider, system, model=None, max_tokens=_MAX_TOKENS, log=print, meter=None):
    provider = (provider or "text").lower()
    if provider not in _ADAPTERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(_ADAPTERS)}")
    return _ADAPTERS[provider](system, model=model, max_tokens=max_tokens, log=log, meter=meter)
