"""The agent loop: a ReAct-style tool-calling harness.

One decision per iteration: the model either calls tools or finishes with a
final message. The full transcript (every model message, tool call and tool
result) is kept so evals and humans can audit exactly what happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .llm import ChatClient
from .tools import Tool, ToolContext, default_tools, run_tool_call

SYSTEM_PROMPT = """\
You are a careful coding agent working inside a project directory.
You have file and shell tools. Think step by step, verify your work with
the tools (read files you wrote, run tests), and only finish when the task
is actually done. If a tool errors, read the error and try a different
approach. Keep final messages short and factual.\
"""


@dataclass
class StepRecord:
    index: int
    model_text: str | None
    tool_calls: list[str]  # "name(args)" summaries
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class AgentResult:
    final_message: str | None
    steps: list[StepRecord] = field(default_factory=list)
    finished: bool = False  # True if the model stopped calling tools on its own
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def estimated_cost_usd(self, settings: Settings) -> float:
        return (
            self.total_prompt_tokens / 1000 * settings.price_in_per_1k
            + self.total_completion_tokens / 1000 * settings.price_out_per_1k
        )


class Agent:
    def __init__(
        self,
        client: ChatClient,
        settings: Settings,
        workdir: str | Path,
        tools: list[Tool] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.settings = settings
        self.ctx = ToolContext(root=Path(workdir).resolve(), shell_timeout_s=settings.shell_timeout_s)
        self.tools = tools if tools is not None else default_tools()
        self.system_prompt = system_prompt

    def run(self, task: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        tool_schemas = [t.schema() for t in self.tools]
        result = AgentResult(final_message=None)

        for i in range(self.settings.max_steps):
            t0 = time.monotonic()
            resp = self.client.chat(messages, tools=tool_schemas)
            latency = time.monotonic() - t0

            result.total_prompt_tokens += resp.prompt_tokens
            result.total_completion_tokens += resp.completion_tokens
            result.total_latency_s += latency

            summaries: list[str] = []
            if resp.content:
                messages.append({"role": "assistant", "content": resp.content})

            if not resp.tool_calls:
                result.final_message = resp.content
                result.finished = True
                result.steps.append(
                    StepRecord(i, resp.content, [], latency, resp.prompt_tokens, resp.completion_tokens)
                )
                break

            # Execute tool calls, feed results back in.
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in resp.tool_calls
            ]
            messages.append(assistant_msg)

            for tc in resp.tool_calls:
                output = run_tool_call(self.ctx, self.tools, tc.name, tc.arguments)
                summaries.append(f"{tc.name}({tc.arguments[:80]})")
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )

            result.steps.append(
                StepRecord(i, resp.content, summaries, latency, resp.prompt_tokens, resp.completion_tokens)
            )
        return result


def run_task(task: str, workdir: str | Path, settings: Settings | None = None) -> AgentResult:
    """One-shot convenience: build client + agent, run, return the result."""
    settings = settings or Settings()
    return Agent(ChatClient(settings), settings, workdir).run(task)
