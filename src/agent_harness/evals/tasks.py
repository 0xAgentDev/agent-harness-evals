"""Eval tasks: each task sets up a fixture workdir, gives the agent a prompt,
and checks the outcome with a real assertion — not an LLM judge.

Checks run against the actual files the agent produced, so a "pass" means
the work is verifiably done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Task:
    id: str
    description: str
    prompt: str
    setup: Callable[[Path], None] = lambda root: None
    check: Callable[[Path], tuple[bool, str]] = lambda root: (False, "no check defined")
    tags: tuple[str, ...] = ()


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# --- checks ---------------------------------------------------------------

def _check_exact_file(rel: str, expected: str):
    def check(root: Path) -> tuple[bool, str]:
        p = root / rel
        if not p.is_file():
            return False, f"missing file {rel}"
        actual = p.read_text()
        if actual == expected:
            return True, "content matches exactly"
        return False, f"content mismatch: got {actual!r}, want {expected!r}"
    return check


# --- tasks ----------------------------------------------------------------

def task_create_file() -> Task:
    return Task(
        id="create_file",
        description="Create a file with exact content.",
        prompt="Create a file named greeting.txt in the working directory containing exactly: hello world",
        check=_check_exact_file("greeting.txt", "hello world"),
        tags=("file-ops", "single-step"),
    )


def task_fix_bug() -> Task:
    buggy = 'def add(a, b):\n    return a - b  # BUG: should add\n'
    test = (
        "from calc import add\n"
        "assert add(2, 3) == 5, 'add is broken'\n"
        "assert add(-1, 1) == 0\n"
        "print('ok')\n"
    )

    def setup(root: Path) -> None:
        _write(root, "calc.py", buggy)
        _write(root, "test_calc.py", test)

    def check(root: Path) -> tuple[bool, str]:
        import subprocess
        proc = subprocess.run(
            ["python3", "test_calc.py"], cwd=root, capture_output=True, text=True, timeout=30
        )
        if proc.returncode == 0 and "ok" in proc.stdout:
            return True, "test_calc.py passes"
        return False, f"tests fail: {(proc.stdout + proc.stderr).strip()[:300]}"

    return Task(
        id="fix_bug",
        description="Find and fix a bug, then verify with the provided test.",
        prompt=(
            "The file calc.py has a bug: add(a, b) subtracts instead of adding. "
            "Fix it, then run `python3 test_calc.py` to verify it prints ok."
        ),
        setup=setup,
        check=check,
        tags=("debugging", "verify"),
    )


def task_read_and_answer() -> Task:
    def setup(root: Path) -> None:
        _write(root, "notes/alice.txt", "Alice's favorite number is 42.\n")
        _write(root, "notes/bob.txt", "Bob's favorite number is 17.\n")

    def check(root: Path) -> tuple[bool, str]:
        p = root / "answer.txt"
        if not p.is_file():
            return False, "missing answer.txt"
        text = p.read_text()
        if "59" in text:
            return True, "answer contains 59"
        return False, f"answer.txt does not contain 59: {text!r}"

    return Task(
        id="read_and_answer",
        description="Read two files and write the sum of the numbers found.",
        prompt=(
            "In the notes/ directory there are files with people's favorite numbers. "
            "Read them, add the numbers together, and write the total into answer.txt "
            "(just the number is fine)."
        ),
        setup=setup,
        check=check,
        tags=("reading", "multi-step"),
    )


def task_multi_file_edit() -> Task:
    def setup(root: Path) -> None:
        _write(root, "src/a.py", "VERSION = '1.0.0'\nprint('a', VERSION)\n")
        _write(root, "src/b.py", "VERSION = '1.0.0'\nprint('b', VERSION)\n")

    def check(root: Path) -> tuple[bool, str]:
        for rel in ("src/a.py", "src/b.py"):
            text = (root / rel).read_text()
            if "2.0.0" not in text:
                return False, f"{rel} was not bumped to 2.0.0"
        return True, "both files bumped to 2.0.0"

    return Task(
        id="multi_file_edit",
        description="Make the same edit consistently across two files.",
        prompt="Bump VERSION from '1.0.0' to '2.0.0' in every Python file under src/.",
        setup=setup,
        check=check,
        tags=("editing", "consistency"),
    )


def task_shell_pipeline() -> Task:
    def setup(root: Path) -> None:
        _write(root, "data.txt", "apple\nbanana\napple\ncherry\nbanana\napple\n")

    def check(root: Path) -> tuple[bool, str]:
        p = root / "counts.txt"
        if not p.is_file():
            return False, "missing counts.txt"
        text = p.read_text()
        # expect apple:3, banana:2, cherry:1 in some readable form
        ok = all(word in text for word in ("apple", "banana", "cherry", "3", "2"))
        return (True, "counts look right") if ok else (False, f"unexpected content: {text!r}")

    return Task(
        id="shell_pipeline",
        description="Use shell tools to count word frequencies.",
        prompt=(
            "data.txt contains one word per line. Count how many times each word "
            "appears and write the results to counts.txt (one line per word is fine)."
        ),
        setup=setup,
        check=check,
        tags=("shell", "multi-step"),
    )


def builtin_tasks() -> list[Task]:
    return [
        task_create_file(),
        task_fix_bug(),
        task_read_and_answer(),
        task_multi_file_edit(),
        task_shell_pipeline(),
    ]
