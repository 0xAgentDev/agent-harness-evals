"""Eval runner: executes tasks against the agent, each in a fresh temp dir,
and collects honest metrics — pass/fail plus steps, tokens, latency, cost.

Nothing here uses an LLM judge. A task passes only if its check function
says the produced files are actually correct.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from ..agent import Agent, AgentResult
from ..config import Settings
from ..llm import ChatClient
from .tasks import Task


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    check_message: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    finished: bool
    error: str | None = None


@dataclass
class EvalSummary:
    results: list[TaskResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def total_latency_s(self) -> float:
        return sum(r.latency_s for r in self.results)

    @property
    def avg_steps(self) -> float:
        return sum(r.steps for r in self.results) / self.total if self.total else 0.0


class EvalRunner:
    def __init__(self, settings: Settings | None = None, client: ChatClient | None = None) -> None:
        self.settings = settings or Settings()
        self.client = client or ChatClient(self.settings)

    def run_task(self, task: Task) -> TaskResult:
        t0 = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix=f"eval-{task.id}-") as tmp:
                root = Path(tmp)
                task.setup(root)
                agent = Agent(self.client, self.settings, root)
                agent_result: AgentResult = agent.run(task.prompt)
                passed, message = task.check(root)
                return TaskResult(
                    task_id=task.id,
                    passed=passed,
                    check_message=message,
                    steps=len(agent_result.steps),
                    prompt_tokens=agent_result.total_prompt_tokens,
                    completion_tokens=agent_result.total_completion_tokens,
                    latency_s=agent_result.total_latency_s,
                    cost_usd=agent_result.estimated_cost_usd(self.settings),
                    finished=agent_result.finished,
                )
        except Exception:
            return TaskResult(
                task_id=task.id,
                passed=False,
                check_message="runner exception",
                steps=0,
                prompt_tokens=0,
                completion_tokens=0,
                latency_s=time.monotonic() - t0,
                cost_usd=0.0,
                finished=False,
                error=traceback.format_exc(limit=3),
            )

    def run(self, tasks: list[Task]) -> EvalSummary:
        summary = EvalSummary()
        for task in tasks:
            result = self.run_task(task)
            summary.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"[{status}] {result.task_id:20s} "
                f"steps={result.steps:2d} "
                f"tokens={result.prompt_tokens + result.completion_tokens:5d} "
                f"latency={result.latency_s:5.1f}s "
                f"cost=${result.cost_usd:.4f} "
                f"— {result.check_message}"
            )
        return summary
