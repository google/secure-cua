
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

"""The isolated SHADOW — an untrusted reader called ONCE PER MARK. It sees only the V_g image
(exactly one region revealed, everything else masked) + the question + the required closed type.
No task, no memory, no K. It returns ONLY a bounded outcome {tag, coerced value}; the raw text is
discarded and never reaches the actuator or K. This is the untrusted→trusted type gate. (§4)
"""
import base64
import os
import time

import metrics as M
import secure_llm as L
import secure_protocol as P

_INSTR = """You are shown ONE small highlighted region of a web page. Everything else is blacked out.
Answer ONLY about that one visible region. You may also be given an accessibility-tree snippet for that
same region (structure only; everything outside your region is scrubbed to `[N]`/`[~]`) — use it to read
the content more precisely, but still answer only about this one region.

Question: {rho}
Return type: {typedesc}

Respond with ONLY this JSON object and nothing else:
{{"outcome": "answered|unknown", "value": <the {T} value, or null>}}
- "answered": you can determine the answer -> put the typed value (true OR false, whichever is correct).
- "unknown": you cannot answer — either it is not that kind of content, or you cannot tell. value null.
- If the value is TEXT you are transcribing (a name, sentence, quote, or passage), copy it VERBATIM — exactly
  as it appears in the region, word for word and character for character; do NOT paraphrase, summarize,
  translate, shorten, re-order, or fix spelling / spacing / punctuation.
No explanation. JSON only."""


_A11Y_NOTE = ("Accessibility tree for the visible region "
              "(structure only; everything outside your region is scrubbed):\n")


class Shadow:
    """Provider-neutral untrusted reader. provider='anthropic' (AnthropicVertex) or 'gemini'
    (google.genai on Vertex). Either way it makes ONE image+text call and returns only a bounded
    {outcome, value}; the raw text never escapes."""

    def __init__(self, provider="anthropic", model=None, max_tokens=3000, log=print, meter=None):
        self.provider = (provider or "anthropic").lower()
        self.max_tokens, self.log, self.meter = max_tokens, log, meter
        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            self._types = types
            _ho = {"timeout": 120000}                                    # ms; bound a black-holed Gemini call
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"), http_options=_ho)
            else:
                self.client = genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                                           location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"), http_options=_ho)
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        elif self.provider == "openai":
            import openai
            if os.getenv("AZURE_OPENAI_ENDPOINT"):
                self.client = openai.AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                                 azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                                 api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                                                 timeout=120.0, max_retries=2)      # bound a black-holed GPT call
                self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5"
            else:
                self.client = openai.OpenAI(timeout=120.0, max_retries=2)
                self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        else:
            self.client = L.vertex_client()
            self.model = model or L.MODEL

    def ask(self, vg_png_b64, rho, T, enum_values=None, a11y_text=""):
        """Return {"outcome": tag, "value": coerced|None}. Only these two bounded fields ever escape.
        a11y_text is the accessibility tree masked to reveal ONLY this region (mirrors the V_g image)."""
        if T == "payload":
            # opaque string extraction: the reader returns free text (a name, a one-sentence summary);
            # it is stored in G and NEVER read by the actuator, so it is not a closed type.
            typedesc = "a short free-text string answering the question (e.g. a name, or a one-sentence summary)"
            t_word = "text"
        else:
            typedesc = f"one of {enum_values}" if (T == "enum" and enum_values) else T
            t_word = T
        instr = _INSTR.format(rho=rho, typedesc=typedesc, T=t_word)
        t0 = time.perf_counter()
        usage = None
        try:
            if self.provider == "gemini":
                text, usage = self._ask_gemini(vg_png_b64, instr, a11y_text)
            elif self.provider == "openai":
                text, usage = self._ask_openai(vg_png_b64, instr, a11y_text)
            else:
                content = [L.img_block(vg_png_b64), {"type": "text", "text": instr}]
                if a11y_text:
                    content.append({"type": "text", "text": _A11Y_NOTE + a11y_text})
                text, usage = L.call_text(self.client, [{"role": "user", "content": content}],
                                          self.model, self.max_tokens)
        except Exception as e:
            self.log(f"[shadow] call failed: {e}")
            return {"outcome": P.ERROR, "value": None}
        finally:
            if self.meter is not None:
                self.meter.add_time("shadow", time.perf_counter() - t0)
                if usage:
                    self.meter.add_tokens("shadow", *usage)
        return self._bound(text, T, enum_values)

    def _ask_gemini(self, vg_png_b64, instr, a11y_text):
        types = self._types
        parts = [types.Part.from_bytes(data=base64.b64decode(vg_png_b64), mime_type="image/png"),
                 types.Part.from_text(text=instr)]
        if a11y_text:
            parts.append(types.Part.from_text(text=_A11Y_NOTE + a11y_text))
        r = self.client.models.generate_content(
            model=self.model, contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(max_output_tokens=self.max_tokens))
        return (r.text or ""), M.norm_gemini(r.usage_metadata)

    def _ask_openai(self, vg_png_b64, instr, a11y_text):
        content = [{"type": "text", "text": instr},
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{vg_png_b64}"}}]
        if a11y_text:
            content.append({"type": "text", "text": _A11Y_NOTE + a11y_text})
        rea = self.model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
        kw = dict(model=self.model, messages=[{"role": "user", "content": content}],
                  max_completion_tokens=(max(self.max_tokens, 8000) if rea else self.max_tokens))
        if rea:
            kw["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "medium")
        r = self.client.chat.completions.create(**kw)
        return (r.choices[0].message.content or ""), M.norm_openai(r.usage)

    @staticmethod
    def _bound(text, T, enum_values):
        obj = P._last_json_obj(text or "") or {}
        outcome = obj.get("outcome")
        if outcome not in P.OUTCOMES:
            return {"outcome": P.ERROR, "value": None}
        if outcome != P.SUCCESS:
            return {"outcome": outcome, "value": None}
        if T == "payload":               # opaque free string: sanitize/cap, never coerce to closed
            ok, val = P.bound_payload(obj.get("value"))
        else:
            ok, val = P.coerce(obj.get("value"), T, enum_values)
        if not ok:                       # unbounded / wrong-type value -> error; nothing escapes
            return {"outcome": P.ERROR, "value": None}
        return {"outcome": P.SUCCESS, "value": val}
