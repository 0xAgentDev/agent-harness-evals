"""Single-task demo: fix a bug in a temp directory and show the transcript.

Usage:  OPENAI_API_KEY=... python examples/demo.py
"""

import tempfile
from pathlib import Path

from agent_harness import Settings, run_task

TASK = (
    "The file calc.py has a bug: add(a, b) subtracts instead of adding. "
    "Fix it, then run `python3 test_calc.py` to verify it prints ok."
)

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="demo-") as tmp:
        root = Path(tmp)
        (root / "calc.py").write_text("def add(a, b):\n    return a - b  # BUG\n")
        (root / "test_calc.py").write_text(
            "from calc import add\nassert add(2, 3) == 5\nprint('ok')\n"
        )
        result = run_task(TASK, root, Settings())
        print(f"finished={result.finished} steps={len(result.steps)} "
              f"tokens={result.total_tokens}")
        print("final:", result.final_message)
        for s in result.steps:
            print(f"--- step {s.index} ({s.latency_s:.1f}s): {s.tool_calls}")
