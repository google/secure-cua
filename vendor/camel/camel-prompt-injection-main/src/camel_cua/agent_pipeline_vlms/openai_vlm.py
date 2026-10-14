"""
OpenAI / Azure-OpenAI-backed action VLM adapter (e.g. GPT-5 vision), used as the CaMeL-CUA
`self.vlm` for screenshot -> pixel-coordinate grounding.

Design mirrors ClaudeVLM: the base_ui `find()` builds Anthropic-style messages
    [{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":..,"data":<b64>}},
                               {"type":"text","text":..}]}]
and calls `self.vlm.query(messages=...)`. This adapter TRANSLATES those messages into the OpenAI
chat-completions vision format (`image_url` data-URIs), calls the model, and returns a plain assistant
message whose text lives in ["content"] (a string). base_ui's `_blocks_from_response` is adapted (in the
OpenAI base_ui copy) to accept that string, so the existing text-coordinate grounding path is reused
unchanged. Reasoning models (gpt-5 / o-series) take `reasoning_effort` + `max_completion_tokens` and
reject `temperature`.
"""

import logging
from typing import Any, List, Optional, Sequence, Tuple

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from camel_cua.agent_pipeline_vlms.extended_types import ChatMessage

logger = logging.getLogger(__name__)

# gpt-5 / o-series take reasoning_effort + max_completion_tokens and reject temperature.
_REASONING = ("gpt-5", "o1", "o3", "o4", "codex")


def _is_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return any(k in m for k in _REASONING)


def _block_to_openai(blk: Any) -> Optional[dict]:
    """Translate one Anthropic-style content block into an OpenAI chat content part."""
    if not isinstance(blk, dict):
        return {"type": "text", "text": str(blk)}
    t = blk.get("type")
    if t == "text":
        return {"type": "text", "text": blk.get("text", "")}
    if t == "image":
        src = blk.get("source", {}) or {}
        if src.get("type") == "base64":
            mt = src.get("media_type", "image/png")
            return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{src.get('data','')}"}}
        # already a data/http URI (Anthropic 'url' source) or a bare data field
        uri = src.get("url") or blk.get("url") or blk.get("data")
        if uri:
            return {"type": "image_url", "image_url": {"url": uri}}
    if t == "image_url":                                   # already OpenAI-shaped
        return blk
    # tool_result / thinking / unknown -> fold any text into a text part (best effort; find() never hits this)
    inner = blk.get("content") or blk.get("text")
    if isinstance(inner, str) and inner:
        return {"type": "text", "text": inner}
    return None


def _messages_to_openai(system: Optional[str], messages: Sequence[ChatMessage]) -> List[dict]:
    out: List[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if role == "system":                              # a system msg already in history -> keep as system
            text = content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in (content or []) if isinstance(b, dict))
            out.append({"role": "system", "content": text})
            continue
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        parts = [p for p in (_block_to_openai(b) for b in (content or [])) if p is not None]
        out.append({"role": role, "content": parts or ""})
    return out


class OpenAIVLM(BasePipelineElement):
    """OpenAI/Azure vision model as the CaMeL action VLM. `query()` returns an assistant message whose
    text is in ["content"] (string)."""

    def __init__(
        self,
        client: Any,                       # openai.OpenAI or openai.AzureOpenAI (built by build_ui_openai)
        model: str,                        # Azure deployment name or OpenAI model id (e.g. "gpt-5")
        system: Optional[str] = None,
        *,
        enable_tools: bool = False,        # unused here (grounding is text-coordinate); kept for base_ui parity
        enable_thinking: bool = True,      # unused; base_ui toggles these attrs on self.vlm
        display_width_px: int = 1920,
        display_height_px: int = 1080,
        reasoning_effort: str = "medium",
        max_completion_tokens: int = 16000,
    ) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.enable_tools = enable_tools
        self.enable_thinking = enable_thinking
        self.display_width_px = display_width_px
        self.display_height_px = display_height_px
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
        token_count_file: str | None = None,
        function_name: str = "",
    ) -> Tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        oai_messages = _messages_to_openai(self.system, messages)
        kw: dict = {"model": self.model, "messages": oai_messages,
                    "max_completion_tokens": self.max_completion_tokens}
        if _is_reasoning(self.model):
            kw["reasoning_effort"] = self.reasoning_effort   # reasoning models reject temperature
        else:
            kw["temperature"] = 0.0
        response = self.client.chat.completions.create(**kw)

        text = (response.choices[0].message.content or "") if getattr(response, "choices", None) else ""

        usage = getattr(response, "usage", None)
        if usage and token_count_file:
            try:
                with open(token_count_file, "a") as f:
                    f.write(f"Model {self.model} Function {function_name or 'unknown'} - "
                            f"Input Tokens: {getattr(usage,'prompt_tokens',0)}, "
                            f"Output Tokens: {getattr(usage,'completion_tokens',0)}\n")
            except Exception:
                pass

        assistant_msg = {"role": "assistant", "content": text}
        return query, runtime, env, [*messages, assistant_msg], extra_args
