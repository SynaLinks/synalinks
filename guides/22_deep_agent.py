# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Deep Agent

[Guide 6](https://synalinks.github.io/synalinks/guides/Agents/) introduced the agent loop — decide, act, observe, repeat.
That loop is general: as long as you give it a set of typed tools,
the model can reason about *anything*. This guide picks a specific
useful shape for those tools: a workspace on disk. The result is a
**deep agent** — an agent that treats a directory as its environment,
reading and editing files and running Python inside a sandboxed,
copy-on-write copy of it.

In other words: where `FunctionCallingAgent` calls a few narrow
helpers (a calculator, a clock), a deep agent has a small but
complete toolbox for *working on code* — the same shape of tools
that drive AI coding assistants like Claude Code, Devin, Aider.

## When Do You Want One?

The agent loop has the same four phases everywhere — think, decide,
act, observe — but the *act* phase looks very different depending on
what tools you give it:

- A RAG agent acts by searching documents.
- An SQL agent acts by writing SELECT queries.
- A deep agent acts by **changing the filesystem** — reading source
  files, patching them, running them, looking at the output.

That makes deep agents a fit for tasks where the answer cannot be
expressed in a single output: it has to *exist* somewhere in a
workspace by the time the loop ends. Bug fixes, refactors, scaffolds,
data-wrangling pipelines, exploratory code reviews — anything where
"the result is some new state in the workspace" is the right ending.

## The Seven Tools

`synalinks.DeepAgent` wraps a `FunctionCallingAgent` and pre-wires
seven tools, all bound to a single working directory you supply:

| Tool | Purpose |
|------|---------|
| `list_files(pattern)` | List files matching a glob (e.g. `**/*.py`). |
| `search_files(pattern, glob)` | Grep file *contents* by regex across files matching a glob. Returns `(path, line_number, line)` matches. |
| `read_file(path, offset, limit)` | Line-paginated file read. Output carries 1-based `start_line` / `end_line`, so the LM can cite line numbers later without re-reading. |
| `write_file(path, content)` | Create or overwrite a file (in the overlay). |
| `edit_file(path, old, new)` | Replace an exact substring; rejects 0 or 2+ occurrences so the LM has to add surrounding context to disambiguate (pass `replace_all=True` to override). |
| `run_python_code(code)` | Run a Python snippet directly in the sandbox interpreter. |
| `run_python_file(path)` | Run a self-contained script the agent wrote into the overlay (it cannot import other overlay files). |

Two design choices are worth highlighting:

1. **Line-paginated reads, line-numbered output.** `read_file` does
   not return character-offset slices. It returns lines — with 1-based
   `start_line` / `end_line` markers. The model thinks in terms of
   lines, edits are usually targeted at a line, so the read tool's
   output should match. The line numbers also mean a model can
   `read_file` once and then `edit_file` referencing the exact text
   it just saw, without re-reading to confirm. Page through long files
   with `offset` / `limit`.

2. **`list_files` globs, `search_files` greps.** `list_files` answers
   "what files are there?" (a glob over paths); `search_files` answers
   "where does this pattern appear?" — a regex over file *contents*,
   restricted to a glob, returning `(path, line_number, line)` matches.
   Two narrow tools instead of one overloaded one keeps the LM's choice
   shallow: it picks by intent, not by guessing what an empty argument
   means.

## The Security Model

Here is the single idea that makes the deep agent safe to point at a
real directory: **the tools operate on a sandboxed, copy-on-write
copy of the workdir, never the workdir itself.**

### Reads fall through, writes stay in the overlay

`DeepAgent` mounts the workdir in a `MontySandbox`. Reads
(`read_file`, `list_files`, `search_files`) fall through to the real
files on disk, so the agent sees your project as it is. But every
mutation — `write_file`, `edit_file`, and anything `run_python_code` /
`run_python_file` writes — lands in an **in-memory overlay** layered on
top. The real workspace on disk is never modified. (Paths are still
rooted at the workdir — `..` cannot escape it — so reads stay inside
the directory you mounted.)

That means there is no capability to gate: the agent simply *cannot*
damage the host through its tools, because nothing it does is ever
written back to disk. All seven tools are therefore always available.

### Code runs in the Monty interpreter, not the host

`run_python_code` and `run_python_file` do not shell out. They execute
in **Monty**, a restricted Python interpreter embedded in the sandbox.
There is no `open()` (file access goes through the overlay-backed
tools), no `subprocess`, no arbitrary host I/O; `json` exposes only
`loads` / `dumps`. `timeout` (default 30s) bounds each execution.

This is why there is no `run_bash` and no `allow_write` / `allow_bash`
switches that earlier designs had: the containment is structural, not a
set of gates you can forget to turn on.

### Inspecting and persisting what the agent did

Because changes live in the overlay, you read them back through the
sandbox rather than off disk. The `DeepAgent` instance exposes it as
`agent.sandbox`:

- `agent.sandbox.changes()` → `{"written": [...], "deleted": [...]}`,
  a summary of the final overlay state.
- `agent.sandbox.journal()` → an ordered log of every mutation.
- `agent.sandbox.read_overlay(path)` → the effective bytes for a path
  (overlay value if written, else the base file).

If you want any of it on disk, persist it yourself from those reads.

## Building the Agent

The constructor signature mirrors `FunctionCallingAgent` exactly —
every parameter on that class is accepted with identical semantics.
The additions are workspace-specific:

| Param | Required | Default | Notes |
|-------|----------|---------|-------|
| `workdir` | yes | — | Must exist and be a directory. Mounted read-through in the sandbox; the agent's writes/edits stay in the overlay and never touch it. (Omit it for an empty in-memory workspace.) |
| `timeout` | no | `30.0` | Per-execution budget in seconds for `run_python_code` / `run_python_file`. |
| `tools` | no | `None` | Extra `Tool` instances or async functions to append to the built-ins. Same name-collision and no-leading-underscore rules as `FunctionCallingAgent`. |

```python
import synalinks

lm = synalinks.LanguageModel(model="ollama/mistral")

inputs = synalinks.Input(data_model=synalinks.ChatMessages)
outputs = await synalinks.DeepAgent(
    workdir="/tmp/my_project",
    language_model=lm,
)(inputs)
agent = synalinks.Program(inputs=inputs, outputs=outputs)
```

You don't need a "read-only mode" to point this at a precious
directory: the copy-on-write overlay already guarantees the real
workdir is never modified, so even when the agent writes, edits, and
runs code, your files on disk stay exactly as they were. To review
what it *would* have changed, read `agent.sandbox.changes()` after the
run.

## A Worked Example

A small end-to-end task: scaffold a Python file, ask the agent to
extend it, and verify the agent's change actually runs. Keep a
reference to the `DeepAgent` instance — its `sandbox` is where the
overlay changes live, so that is where you read the result (the file
on disk is never touched).

```python
import asyncio, os, tempfile
import synalinks

workdir = tempfile.mkdtemp(prefix="deep_demo_")
with open(os.path.join(workdir, "calculator.py"), "w") as f:
    f.write("def add(a, b):\n    return a + b\n")

lm = synalinks.LanguageModel(model="ollama/mistral")
inputs = synalinks.Input(data_model=synalinks.ChatMessages)
deep = synalinks.DeepAgent(
    workdir=workdir,
    language_model=lm,
    max_iterations=10,
)
outputs = await deep(inputs)
agent = synalinks.Program(inputs=inputs, outputs=outputs)

task = synalinks.ChatMessages(messages=[synalinks.ChatMessage(
    role="user",
    content=(
        "Open calculator.py, add a `multiply(a, b)` function, "
        "and run `run_python_code` with "
        "`from calculator import multiply; print(multiply(6, 7))` "
        "to confirm it prints 42."
    ),
)])
result = await agent(task)

# Changes live in the overlay, not on disk — inspect them via the sandbox:
print(deep.sandbox.changes())                       # {'written': ['/calculator.py'], ...}
print(deep.sandbox.read_overlay("calculator.py").decode())
```

What the agent will typically do:

1. `read_file("calculator.py")` — see the current source, with line
   markers.
2. `edit_file("calculator.py", "def add(a, b):\n    return a + b\n", "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n")`
   — or write the whole file with `write_file`. The model picks.
3. `run_python_code("from calculator import multiply; print(multiply(6, 7))")`
   — verify the change against the overlay.
4. Stop calling tools; produce the final assistant message.

If the verify step fails (syntax error, wrong number printed), the
agent reads the result, edits again, re-runs. The iteration cap
is the only thing that stops it from looping forever; in practice a
correct change usually lands in 2-3 tool calls.

## Compared to Other Agents

`DeepAgent` is one of several specialized agents that all wrap a
`FunctionCallingAgent` with a workload-specific tool set:

| Agent | Bound to | Tools |
|-------|----------|-------|
| `FunctionCallingAgent` | nothing | whatever you pass in |
| `SQLAgent` | a `KnowledgeBase` | schema discovery, table sample, read-only SQL |
| `VectorRAGAgent` | a `KnowledgeBase` | schema discovery, semantic/keyword search, get-by-id |
| `DeepAgent` | a workdir | list, search, read, write, edit, run Python |

All four accept the same `FunctionCallingAgent` parameters plus
their domain-specific extras. The same `tools=` slot is available on
all of them for layering extra tools on top of the built-ins.

When to pick which:

- **SQLAgent**: data lives in a structured DB; the answer is a
  computation over rows.
- **VectorRAGAgent**: data lives as documents; the answer is a
  synthesis grounded in retrieval.
- **DeepAgent**: the answer is a *change* to a workspace, or a
  diagnosis that requires running code.
- **FunctionCallingAgent**: you have a bespoke tool set that doesn't
  match any of the above.

## API References

- [DeepAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/DeepAgent%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [SQLAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/SQLAgent%20module/)
- [VectorRAGAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/VectorRAGAgent%20module/)
- [Tool](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Tool%20module/)
- [ChatMessages (Base DataModels)](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)
"""

import asyncio
import os
import shutil
import tempfile

from dotenv import load_dotenv

import synalinks


def populate_workspace(workdir: str) -> None:
    """Seed the workdir with a tiny Python project for the agent to extend."""
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "calculator.py"), "w") as f:
        f.write('def add(a, b):\n    """Return a + b."""\n    return a + b\n')


async def main():
    load_dotenv()
    synalinks.clear_session()

    workdir = tempfile.mkdtemp(prefix="deep_agent_guide_")
    try:
        populate_workspace(workdir)

        # The LM that drives the agent. Any function-calling-capable model
        # works; pick a small local model for fast iteration during dev.
        language_model = synalinks.LanguageModel(model="ollama/mistral")

        # Build the agent. The workdir is the only thing DeepAgent adds on
        # top of FunctionCallingAgent's signature; everything else is
        # forwarded with identical semantics.
        # Keep a reference to the DeepAgent: its `sandbox` holds the
        # copy-on-write overlay where the agent's edits land (the workdir
        # on disk is never modified), so that is where we read results.
        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        deep = synalinks.DeepAgent(
            workdir=workdir,
            language_model=language_model,
            max_iterations=10,  # coding tasks tend to need a few rounds
        )
        outputs = await deep(inputs)

        agent = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="deep_agent",
            description="A coding agent with sandboxed file and Python tools.",
        )

        # The task: read the file, add a function, verify by running Python.
        task = synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content=(
                        "Open calculator.py, add a `multiply(a, b)` function "
                        "in the same style as `add`, and use run_python_code "
                        "to run `from calculator import multiply; "
                        "print(multiply(6, 7))` to confirm it prints 42."
                    ),
                )
            ]
        )
        result = await agent(task)

        # The trajectory captures every tool call and its result; the final
        # assistant message is the model's natural-language answer.
        final = [m for m in result.get("messages", []) if m.get("role") == "assistant"]
        if final:
            print("Agent:", final[-1].get("content"))

        # The agent's edits live in the overlay, not on disk. Inspect them
        # through the sandbox: `changes()` summarizes what was written, and
        # `read_overlay(...)` returns the effective file content.
        print("\nOverlay changes:", deep.sandbox.changes())
        overlay = deep.sandbox.read_overlay("calculator.py")
        if overlay is not None:
            print("\n--- calculator.py (overlay) ---")
            print(overlay.decode())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
