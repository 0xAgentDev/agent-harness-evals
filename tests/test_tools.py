"""Tests for the tool layer. No API calls — pure local behavior."""

import pytest

from agent_harness.tools import ToolContext, default_tools, run_tool_call


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(root=tmp_path)


@pytest.fixture
def tools():
    return default_tools()


def test_write_and_read_file(ctx, tools):
    assert run_tool_call(ctx, tools, "write_file",
                         '{"path": "a.txt", "content": "hi"}').startswith("wrote")
    out = run_tool_call(ctx, tools, "read_file", '{"path": "a.txt"}')
    assert out == "hi"


def test_path_traversal_rejected(ctx, tools):
    out = run_tool_call(ctx, tools, "read_file", '{"path": "../secret.txt"}')
    assert out.startswith("error:")
    assert "escapes" in out


def test_unknown_tool(ctx, tools):
    out = run_tool_call(ctx, tools, "rm_rf", "{}")
    assert out.startswith("error: unknown tool")


def test_shell_allowlist(ctx, tools):
    out = run_tool_call(ctx, tools, "run_shell", '{"command": "echo hello"}')
    assert "exit=0" in out and "hello" in out


def test_shell_blocked_command(ctx, tools):
    out = run_tool_call(ctx, tools, "run_shell", '{"command": "curl example.com"}')
    assert out.startswith("error: command not allowed")


def test_shell_confined_to_root(ctx, tools, tmp_path):
    run_tool_call(ctx, tools, "run_shell", '{"command": "touch marker.txt"}')
    assert (tmp_path / "marker.txt").exists()


def test_list_dir(ctx, tools):
    run_tool_call(ctx, tools, "write_file", '{"path": "x.txt", "content": "1"}')
    out = run_tool_call(ctx, tools, "list_dir", '{"path": "."}')
    assert "x.txt" in out
