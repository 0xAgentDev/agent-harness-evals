"""Agent harness + evals: a small, honest coding-agent loop with tool calling
and a task-success / reliability / cost eval suite.

Works against any OpenAI-compatible chat-completions endpoint
(OpenAI, Ollama, OmniRoute, ...) via environment variables.
"""

from .agent import Agent, AgentResult, run_task
from .config import Settings
from .evals.runner import EvalRunner
from .llm import ChatClient
from .tools import Tool, ToolContext, default_tools

__all__ = [
    "Agent",
    "AgentResult",
    "ChatClient",
    "EvalRunner",
    "Settings",
    "Tool",
    "ToolContext",
    "default_tools",
    "run_task",
]
