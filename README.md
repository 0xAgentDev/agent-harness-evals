# agent-harness-evals

A small, honest coding-agent harness with tool calling and a task-success eval suite.

The harness is a ReAct-style loop: the model gets file and shell tools, works
step by step inside a sandboxed working directory, and finishes when the task
is done. The eval suite runs the agent against fixture tasks and checks the
**actual files it produced** — no LLM judge, a pass means the work is
verifiably done.

## Why this exists

Most agent demos stop at "look, it called a tool." The interesting questions
are the boring ones: did it finish? How many steps did it take? How many
tokens did it burn? This project answers those with real numbers.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

export OPENAI_API_KEY=sk-...
# Optional: point at any OpenAI-compatible endpoint (Ollama, OmniRoute, ...)
export OPENAI_BASE_URL=https://api.openai.com/v1
export AGENT_MODEL=gpt-4o-mini

# run the full eval suite
.venv/bin/python -m agent_harness.cli
# or a single task
.venv/bin/python -m agent_harness.cli fix_bug
```

Tests (no API calls needed):

```bash
.venv/bin/python -m pytest tests/ -q
```

## Architecture

```
src/agent_harness/
  agent.py      # the harness loop: model -> tool calls -> results -> ...
  llm.py        # thin OpenAI-compatible chat client + usage tracking
  tools.py      # read_file, write_file, list_dir, run_shell (allowlisted)
  config.py     # everything from env vars
  cli.py        # `python -m agent_harness.cli`
  evals/
    tasks.py    # fixture tasks: create_file, fix_bug, read_and_answer,
                #   multi_file_edit, shell_pipeline
    runner.py   # runs each task in a fresh temp dir, collects metrics
    report.py   # markdown report: pass rate, steps, tokens, cost
```

A typical eval run prints per-task results and a summary:

```
[PASS] create_file          steps= 2 tokens=  812 latency=  3.2s cost=$0.0006 — content matches exactly
[FAIL] fix_bug              steps= 7 tokens= 4210 latency= 18.4s cost=$0.0031 — tests fail: ...
```

## Safety model

- All file/shell operations are confined to a per-run working directory.
  `..` traversal outside the root is rejected.
- `run_shell` only allows an explicit command allowlist
  (`ls`, `cat`, `grep`, `python`, `pytest`, `git`, …), always with
  `cwd=<workdir>` and a timeout.
- Tool exceptions are caught and returned to the model as error text —
  tools never crash the loop.

This is documented and tested, not bulletproof: don't point it at
production machines or secrets.

## Design decisions

- **OpenAI-compatible, not OpenAI-locked.** `OPENAI_BASE_URL` + `AGENT_MODEL`
  means the same harness runs against a local Ollama, a self-hosted gateway,
  or OpenAI with zero code changes.
- **Real assertions, not LLM judges.** Every eval check inspects produced
  files or runs produced tests. A pass is a fact, not an opinion.
- **Full transcripts.** Every model message, tool call, and tool result is
  recorded per step, so failures are debuggable instead of mysterious.
- **Cost is a first-class metric.** Token usage is tracked per step and
  converted to USD estimates — the number that decides whether an agent
  ships.

## Roadmap

- [ ] More tasks: multi-file refactor, dependency install, test generation
- [ ] Parallel eval runs with a worker pool
- [ ] Trajectory diffing: compare two models on the same task set
- [ ] MCP client tool (call external MCP servers as tools)
- [ ] Guardrail evals: prompt-injection resistance, tool-misuse detection
