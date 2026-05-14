# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Deep Agent

[Guide 5](Agents.md) introduced the agent loop — decide, act, observe, repeat.
That loop is general: as long as you give it a set of typed tools,
the model can reason about *anything*. This guide picks a specific
useful shape for those tools: a workspace on disk. The result is a
**deep agent** — an agent that treats a directory as its environment,
reading and editing files and running shell commands inside it.

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
  files, patching them, running tests, looking at the output.

That makes deep agents a fit for tasks where the answer cannot be
expressed in a single output: it has to *exist* somewhere in a
workspace by the time the loop ends. Bug fixes, refactors, scaffolds,
data-wrangling pipelines, exploratory code reviews — anything where
"the result is some new state on disk" is the right ending.

## The Six Tools

`synalinks.DeepAgent` wraps a `FunctionCallingAgent` and pre-wires
six tools, all bound to a single working directory you supply:

| Tool | Purpose |
|------|---------|
| `list_directory(path)` | List entries with name / type / size. |
| `search_files(file_pattern, content_pattern)` | Glob for files; optionally grep their contents (regex). One call, two modes. |
| `read_file(path, offset, limit)` | Line-paginated file read. Output is prefixed with line numbers (`cat -n` style), so the LM can cite line numbers later without re-reading. |
| `write_file(path, content)` | Create or overwrite a file. Gated by `allow_write`. |
| `edit_file(path, old_string, new_string)` | Replace an exact substring; rejects 0 or 2+ occurrences so the LM has to add surrounding context to disambiguate. Gated by `allow_write`. |
| `run_bash(command)` | Run a shell command with timeout. Gated by `allow_bash`. |

Two design choices are worth highlighting:

1. **Line-paginated reads, line-numbered output.** `read_file` does
   not return character-offset slices. It returns lines — each
   prefixed with a 1-based line number. The model thinks in terms of
   lines, edits are usually targeted at a line, so the read tool's
   output should match. The line numbers also mean a model can
   `read_file` once and then `edit_file` referencing the exact line
   it just saw, without re-reading to confirm.

2. **`search_files` is glob + grep in one call.** Passing
   `content_pattern=""` gives you pure glob (just file paths); a
   non-empty regex turns it into grep, returning
   `(path, line_number, line)` triples. Two physical tools collapsed
   into one keeps the LM's choice tree shallow: it picks `glob` or
   `grep` by what it writes in the second argument, not by guessing
   between two similar tool names.

## The Security Model

Of the six tools, five touch the filesystem and one runs a shell. The
agent enforces two different containment guarantees, and the
distinction matters:

### File tools are sandboxed to the workdir

`read_file`, `write_file`, `edit_file`, and `list_directory` run
every path through a single resolver; `search_files` uses similar
glob + containment-check logic against the same resolved workdir.
The resolver is roughly:

```python
def _resolve_inside_workdir(workdir, user_path):
    resolved_workdir = workdir.resolve()
    candidate = (
        resolved_workdir / user_path
        if not os.path.isabs(user_path)
        else Path(user_path)
    )
    resolved = candidate.resolve()
    resolved.relative_to(resolved_workdir)   # raises if outside
    return resolved
```

`Path.resolve()` does two important things:

- It flattens `..` (so `subdir/../../etc/passwd` becomes
  `/etc/passwd`, which then fails the prefix check).
- It follows existing symlinks (so a symlink named `shortcut` inside
  the workdir pointing to `/etc/passwd` resolves to `/etc/passwd`,
  which also fails the prefix check).

`Path.relative_to` rejects any path that doesn't start with the
resolved workdir. File opens use `O_NOFOLLOW` where the OS supports
it as defense in depth against TOCTOU races (an attacker replacing a
file with a symlink between the path check and the actual `open`).

This is a *real* boundary. The deep agent cannot read or write
anything outside the workdir through its file tools.

### Bash is NOT sandboxed

`run_bash` runs with `cwd=workdir`, but that's just the shell's
starting directory. The shell itself can read or write anything the
host process can read or write — there's no way to make `cat
/etc/passwd` fail at the Python layer. The agent merely passes the
command string to `asyncio.create_subprocess_shell`.

If you're going to expose this to untrusted input (e.g. an agent
that takes user instructions), you have two options:

1. **OS-level isolation.** Run the host Python process inside a
   container, a `chroot`, a user namespace, or another mechanism
   that limits what the *process* can see. The Python layer cannot
   provide this.
2. **Disable bash.** `allow_bash=False` removes the tool entirely.
   You still have a capable read/edit agent — just no shell.

The guide is explicit about this because it's easy to assume a
"workdir" parameter implies a real sandbox. It doesn't, for the bash
tool specifically.

### Output capping and time bounds

A few smaller controls keep the agent from drowning the conversation
in output:

- `max_output_chars` (default 10000) caps each stream returned by
  `run_bash` (stdout and stderr counted separately) and truncates
  each individual line returned by `read_file` and each matching
  line from `search_files`. A long file therefore comes back with
  many lines, each individually bounded, rather than the read
  itself being capped to a single 10000-char slice.
- `max_search_results` (default 100) caps the entries returned by
  `search_files`.
- `timeout` (default 30s) is the per-command bash budget.
  `asyncio.wait_for` kills the process if it overruns.
- `_SEARCH_MAX_FILE_BYTES` (1 MB) is a hardcoded skip threshold so
  grepping never reads a single huge file end-to-end. Binary files
  are skipped silently via `UnicodeDecodeError`.

## Building the Agent

The constructor signature mirrors `FunctionCallingAgent` exactly —
every parameter on that class is accepted with identical semantics.
The additions are workspace-specific:

| Param | Required | Default | Notes |
|-------|----------|---------|-------|
| `workdir` | yes | — | Must exist and be a directory. Resolved once at construction; subsequent file paths are checked against this. |
| `allow_write` | no | `True` | When `False`, `write_file` and `edit_file` are omitted from the tool set. |
| `allow_bash` | no | `True` | When `False`, `run_bash` is omitted. |
| `timeout` | no | `30.0` | Per-bash-command timeout in seconds. |
| `max_output_chars` | no | `10000` | Per-stream output cap. |
| `max_search_results` | no | `100` | Cap on `search_files` results. |
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

Want an inspector that can read code but not change it or run
anything?

```python
outputs = await synalinks.DeepAgent(
    workdir="/path/to/repo",
    language_model=lm,
    allow_write=False,
    allow_bash=False,
)(inputs)
```

That gives the LM three tools: `list_directory`, `search_files`,
`read_file`. Enough to summarize, audit, or explain — nothing to
modify with.

## A Worked Example

A small end-to-end task: scaffold a Python file, ask the agent to
extend it, and verify the agent's change actually runs.

```python
import asyncio, os, shutil, tempfile
import synalinks

workdir = tempfile.mkdtemp(prefix="deep_demo_")
with open(os.path.join(workdir, "calculator.py"), "w") as f:
    f.write("def add(a, b):\n    return a + b\n")

lm = synalinks.LanguageModel(model="ollama/mistral")
inputs = synalinks.Input(data_model=synalinks.ChatMessages)
outputs = await synalinks.DeepAgent(
    workdir=workdir,
    language_model=lm,
    max_iterations=10,
)(inputs)
agent = synalinks.Program(inputs=inputs, outputs=outputs)

task = synalinks.ChatMessages(messages=[synalinks.ChatMessage(
    role="user",
    content=(
        "Open calculator.py, add a `multiply(a, b)` function, "
        "and confirm with `python -c 'from calculator import multiply; "
        "print(multiply(6, 7))'`."
    ),
)])
result = await agent(task)
```

What the agent will typically do:

1. `read_file("calculator.py", 0, 100)` — see the current source,
   line numbers included.
2. `edit_file("calculator.py", "def add(a, b):", "def add(a, b):\n    return a + b\n\ndef multiply(a, b):")`
   — or write the whole file with `write_file`. The model picks.
3. `run_bash("python -c 'from calculator import multiply; print(multiply(6, 7))'")`
   — verify the change.
4. Stop calling tools; produce the final assistant message.

If the verify step fails (syntax error, wrong number printed), the
agent reads the bash result, edits again, re-runs. The iteration cap
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
| `DeepAgent` | a workdir | list, search, read, write, edit, bash |

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
        f.write(
            "def add(a, b):\n"
            "    \"\"\"Return a + b.\"\"\"\n"
            "    return a + b\n"
        )


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
        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        outputs = await synalinks.DeepAgent(
            workdir=workdir,
            language_model=language_model,
            max_iterations=10,  # coding tasks tend to need a few rounds
        )(inputs)

        agent = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="deep_agent",
            description="A coding agent with file and shell access.",
        )

        # The task: read the file, add a function, verify with the shell.
        task = synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content=(
                        "Open calculator.py, add a `multiply(a, b)` function "
                        "in the same style as `add`, and run "
                        "`python -c 'from calculator import multiply; "
                        "print(multiply(6, 7))'` to confirm."
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

        # Verify the file actually got the new function.
        with open(os.path.join(workdir, "calculator.py")) as f:
            print("\n--- calculator.py ---")
            print(f.read())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
