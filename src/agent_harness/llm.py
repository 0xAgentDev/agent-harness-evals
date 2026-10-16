"""Thin wrapper around an OpenAI-compatible chat-completions endpoint.

Deliberately minimal: one call shape in, content + tool calls + usage out.
Token usage is tracked so evals can report latency and cost honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as returned by the API


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatClient:
    def __init__(self, settings: Settings) -> None:
        settings.require_api_key()
        self.settings = settings
        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage
        return ChatResponse(
            content=msg.content,
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
