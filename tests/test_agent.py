"""Tests for the agent loop with a scripted fake model. No API calls."""

import json

from agent_harness.agent import Agent
from agent_harness.config import Settings
from agent_harness.llm import ChatClient, ChatResponse, ToolCall


class FakeClient(ChatClient):
    """Replays a script of responses instead of calling an API."""

    def __init__(self, script: list[ChatResponse]):
        self._script = list(script)

    def chat(self, messages, tools=None):
        return self._script.pop(0)


def make_settings(**kw):
    kw.setdefault("api_key", "test-key")  # never used by FakeClient
    return Settings(**kw)


def test_agent_writes_file_then_finishes(tmp_path):
    script = [
        ChatResponse(
            content="I'll create the file.",
            tool_calls=[ToolCall(id="1", name="write_file",
                                 arguments=json.dumps({"path": "out.txt", "content": "done"}))],
            prompt_tokens=10, completion_tokens=5,
        ),
        ChatResponse(content="File created.", prompt_tokens=10, completion_tokens=5),
    ]
    agent = Agent(FakeClient(script), make_settings(), tmp_path)
    result = agent.run("create out.txt")
    assert result.finished
    assert (tmp_path / "out.txt").read_text() == "done"
    assert len(result.steps) == 2
    assert result.total_tokens == 30


def test_agent_stops_at_max_steps(tmp_path):
    script = [
        ChatResponse(
            content="looping",
            tool_calls=[ToolCall(id=str(i), name="list_dir", arguments="{}")],
            prompt_tokens=1, completion_tokens=1,
        )
        for i in range(10)
    ]
    agent = Agent(FakeClient(script), make_settings(max_steps=3), tmp_path)
    result = agent.run("list forever")
    assert not result.finished
    assert len(result.steps) == 3


def test_tool_error_does_not_crash_loop(tmp_path):
    script = [
        ChatResponse(
            content="trying a bad tool",
            tool_calls=[ToolCall(id="1", name="nope", arguments="{}")],
            prompt_tokens=1, completion_tokens=1,
        ),
        ChatResponse(content="Recovered.", prompt_tokens=1, completion_tokens=1),
    ]
    agent = Agent(FakeClient(script), make_settings(), tmp_path)
    result = agent.run("do something")
    assert result.finished
    assert result.final_message == "Recovered."
