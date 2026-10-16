"""Tool definitions for the agent.

Safety model (documented, not bulletproof):
- Every file/shell operation is confined to a per-run working directory.
  Path traversal (`..`) outside the root is rejected.
- The shell tool only runs commands from an explicit allowlist and always
  runs with cwd=root and a timeout. No pipes to interpreters outside the
  allowlist, no network tools.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Commands the agent may run. Keep this list boring on purpose.
SHELL_ALLOWLIST = {
    "ls", "cat", "echo", "grep", "find", "wc", "head", "tail",
    "mkdir", "touch", "cp", "mv", "python", "python3", "pytest",
    "node", "npm", "git", "diff",
}


@dataclass
class ToolContext:
    """Per-run sandbox root. All paths resolve inside it."""
    root: Path
    shell_timeout_s: int = 30

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        root = self.root.resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"path escapes working directory: {rel!r}")
        return p


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[ToolContext, dict[str, Any]], str]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _read_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    p = ctx.resolve(args["path"])
    if not p.is_file():
        return f"error: not a file: {args['path']}"
    text = p.read_text()
    max_chars = int(args.get("max_chars", 20000))
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
    return text


def _write_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    p = ctx.resolve(args["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    return f"wrote {len(args['content'])} chars to {args['path']}"


def _list_dir(ctx: ToolContext, args: dict[str, Any]) -> str:
    p = ctx.resolve(args.get("path", "."))
    if not p.is_dir():
        return f"error: not a directory: {args.get('path', '.')}"
    entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
    return "\n".join(entries) if entries else "(empty)"


def _run_shell(ctx: ToolContext, args: dict[str, Any]) -> str:
    cmd = args["command"].strip()
    if not cmd:
        return "error: empty command"
    argv = shlex.split(cmd)
    if argv[0] not in SHELL_ALLOWLIST:
        return f"error: command not allowed: {argv[0]} (allowlist: {sorted(SHELL_ALLOWLIST)})"
    try:
        proc = subprocess.run(
            argv,
            cwd=ctx.root,
            capture_output=True,
            text=True,
            timeout=ctx.shell_timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"error: timed out after {ctx.shell_timeout_s}s"
    out = (proc.stdout or "") + (proc.stderr or "")
    out = out.strip() or "(no output)"
    if len(out) > 8000:
        out = out[:8000] + "\n... [truncated]"
    return f"exit={proc.returncode}\n{out}"


def default_tools() -> list[Tool]:
    str_prop = {"type": "string"}
    return [
        Tool(
            name="read_file",
            description="Read a text file relative to the working directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": str_prop,
                    "max_chars": {"type": "integer", "description": "max chars to return"},
                },
                "required": ["path"],
            },
            fn=_read_file,
        ),
        Tool(
            name="write_file",
            description="Write (create or overwrite) a text file relative to the working directory.",
            parameters={
                "type": "object",
                "properties": {"path": str_prop, "content": str_prop},
                "required": ["path", "content"],
            },
            fn=_write_file,
        ),
        Tool(
            name="list_dir",
            description="List files in a directory relative to the working directory.",
            parameters={
                "type": "object",
                "properties": {"path": str_prop},
                "required": [],
            },
            fn=_list_dir,
        ),
        Tool(
            name="run_shell",
            description=(
                "Run a shell command in the working directory. Only allowlisted "
                "commands (ls, cat, grep, python, pytest, git, ...) are permitted."
            ),
            parameters={
                "type": "object",
                "properties": {"command": str_prop},
                "required": ["command"],
            },
            fn=_run_shell,
        ),
    ]


def run_tool_call(ctx: ToolContext, tools: list[Tool], name: str, arguments_json: str) -> str:
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return f"error: unknown tool: {name}"
    try:
        args = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"error: bad JSON arguments: {e}"
    try:
        return tool.fn(ctx, args)
    except Exception as e:  # tools must never crash the loop
        return f"error: {type(e).__name__}: {e}"
