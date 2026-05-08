"""Anthropic Messages client with tool-use, prompt caching, and retry."""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic
from anthropic import APIConnectionError, APIStatusError, RateLimitError


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False
    raw: Any = None

    def first_tool_input(self) -> Optional[dict[str, Any]]:
        return self.tool_calls[0]["input"] if self.tool_calls else None


class LLMClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 4,
        base_backoff: float = 0.6,
    ):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.model = model
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._client = anthropic.Anthropic()

    def call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[dict[str, Any]] = None,
        max_tokens: int = 1024,
        cache_system: bool = False,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if cache_system:
            sys_payload: Any = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            sys_payload = system

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=sys_payload,
            messages=messages,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        resp = self._with_retry(lambda: self._client.messages.create(**kwargs))

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {"name": block.name, "input": block.input, "id": block.id}
                )

        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(
                resp.usage, "cache_creation_input_tokens", 0
            ),
        }
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
            usage=usage,
            cached=usage["cache_read_input_tokens"] > 0,
            raw=resp,
        )

    def _with_retry(self, fn):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except RateLimitError as e:
                last_exc = e
            except APIConnectionError as e:
                last_exc = e
            except APIStatusError as e:
                if e.status_code not in RETRYABLE_STATUS:
                    raise
                last_exc = e
            if attempt == self.max_retries:
                break
            sleep_s = self.base_backoff * (2 ** attempt) + random.uniform(0, 0.25)
            time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc
