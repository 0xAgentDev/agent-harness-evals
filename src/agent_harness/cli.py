"""Run the built-in eval suite and print a markdown report.

Usage:
    python -m agent_harness.cli            # all tasks
    python -m agent_harness.cli fix_bug    # one task

Environment: OPENAI_API_KEY (+ optional OPENAI_BASE_URL, AGENT_MODEL).
"""

from __future__ import annotations

import sys

from .config import Settings
from .evals import EvalRunner, builtin_tasks, render_markdown


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    settings = Settings()
    tasks = builtin_tasks()
    if argv:
        wanted = set(argv)
        tasks = [t for t in tasks if t.id in wanted]
        if not tasks:
            print(f"unknown task(s): {sorted(wanted)}")
            print("available:", ", ".join(t.id for t in builtin_tasks()))
            return 2
    runner = EvalRunner(settings)
    summary = runner.run(tasks)
    print()
    print(render_markdown(summary, settings.model))
    return 0 if summary.passed == summary.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
