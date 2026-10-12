
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

"""Shared Vertex/Anthropic call helpers for the secure loop (actuator + isolated shadow).

Mirrors agent.py's Claude/Vertex branch exactly: AnthropicVertex(project_id, region) +
messages.create(..., extra_body={"thinking":{"type":"adaptive"}}), first `text` block extracted.
Env (from camel-prompt-injection-main/.env): GOOGLE_CLOUD_PROJECT, ANTHROPIC_VERTEX_REGION or
GOOGLE_CLOUD_LOCATION, GOOGLE_APPLICATION_CREDENTIALS (ADC), GOOGLE_GENAI_USE_VERTEXAI.
"""
import os
import sys
import time

import anthropic

import metrics as M

MODEL = "claude-opus-4-8"


def vertex_client():
    region = os.getenv("ANTHROPIC_VERTEX_REGION") or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    proj = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not proj:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set (is .env loaded?)")
    # HANG FIX (root cause: a Vertex call occasionally black-holes -> hits the per-request timeout; with a long
    # timeout * many retries a single stall became a 25-40 min hang that stalled the whole run). timeout=120
    # bounds each attempt (generous: real adaptive-thinking calls are ~10-30 s); max_retries=2 caps the SDK's
    # own backoff loop. call_text() adds an OUTER retry with a FRESH call, so a transient stall is abandoned in
    # ~2 min and recovers, and a persistent outage fails the turn in ~6 min instead of ~40. Shared by the
    # actuator adapters AND the shadow, so this hardens all three call paths at once.
    return anthropic.AnthropicVertex(project_id=proj, region=region, max_retries=2, timeout=120.0)


def img_block(png_b64):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64}}


def call_text(client, messages, model=MODEL, max_tokens=8000):
    """One Vertex call with adaptive thinking; returns the first text block (skips the thinking block).
    Retries a FRESH call on (a) a 200-with-empty-content stall AND (b) an EXCEPTION — the SDK having exhausted
    its own (now small) retry budget against a black-holed/overloaded endpoint. This abandons a stuck call and
    tries again instead of compounding one stall into a long hang; each create() is time-bounded by the client
    timeout. Returns a (text, usage) tuple: usage is norm_anthropic(r.usage)=(input,output,thinking) on success,
    or None after all attempts fail so the caller degrades (logs an empty op) rather than crashing."""
    for attempt in range(3):
        try:
            r = client.messages.create(model=model, max_tokens=max_tokens, messages=messages,
                                       extra_body={"thinking": {"type": "adaptive"}})
            txt = next((b.text for b in r.content if getattr(b, "type", None) == "text"), "") or ""
            if txt:
                return txt, M.norm_anthropic(r.usage)
        except Exception as e:                                            # SDK exhausted retries -> stuck endpoint
            sys.stderr.write(f"[secure_llm] create attempt {attempt + 1}/3 failed: {type(e).__name__}: {str(e)[:140]}\n")
            sys.stderr.flush()
            time.sleep(1.5)
    return "", None
    
