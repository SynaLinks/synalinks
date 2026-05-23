"""
# Deep Agent

A **Deep Agent** is a coding-style agent that treats a working directory
as its environment — read files, grep them, write or patch them, run a
bit of Python to check the result. Where a `FunctionCallingAgent`
typically calls a few narrow tools (search, calculate, fetch), a deep
agent edits a workspace in place.

In Synalinks the deep agent's tools **are** the methods of a
`MontySandbox` mounted on the workdir. That makes it host-safe by
construction: reads fall through to the real directory, but writes,
edits and code execution are captured in an in-memory copy-on-write
overlay that never modifies the disk.

## What Deep Agents Can Do

```mermaid
graph LR
    A[User Task] --> B[DeepAgent]
    B --> C{Pick Tool}
    C -->|Discover| D[list_files]
    C -->|Find/grep| E[search_files]
    C -->|Read code| F[read_file]
    C -->|Patch| G[edit_file]
    C -->|Create| H[write_file]
    C -->|Run Python| I[run_python_code / run_python_file]
    D --> B
    E --> B
    F --> B
    G --> B
    H --> B
    I --> B
    B --> J[Final Answer]
```

Typical use cases:

1. **Code review / explanation**: read a repo, summarize what each
   file does.
2. **Bug fix loop**: find the failing code, read the relevant module,
   patch it, iterate.
3. **Project bootstrapping**: scaffold a small project in the overlay,
   then persist the result yourself if you're happy with it.
4. **Data wrangling**: search a directory of CSVs / logs, extract
   what's relevant, write a report into the overlay.

## The Seven Built-in Tools

| Tool | Purpose |
|------|---------|
| `list_files` | List files matching a glob (e.g. `**/*.py`). |
| `search_files` | Glob for files and grep their contents (regex). |
| `read_file` | Read a file by 1-based line range (paginated). |
| `write_file` | Create or overwrite a file (in the overlay). |
| `edit_file` | Exact-string replacement; rejects 0 or 2+ occurrences (pass `replace_all=True` to override). |
| `run_python_code` | Run an inline Python snippet (arg: `code`). |
| `run_python_file` | Run a self-contained script you wrote into the overlay (arg: `path`). |

## Host Safety

Every tool is backed by the sandbox's copy-on-write overlay, so the
agent **cannot damage the real workspace**:

- **Reads** fall through to the real `workdir`.
- **Writes / edits** land in an in-memory overlay; the files on disk
  are never modified. Inspect what changed via
  `agent.sandbox.changes()` / `agent.sandbox.journal()`, and persist
  any of it yourself if you want it on disk.
- **Paths** are rooted at the workdir; `..` cannot escape it, and a
  symlink in the base pointing outside it is refused.
- **`run_python_code`** (inline snippet) and **`run_python_file`** (a
  script the agent wrote into the overlay) both run inside Monty — a
  restricted Python interpreter with a small stdlib subset, no
  third-party imports and no network — so model-authored code can't
  reach the host either. (A `run_python_file` script must be
  self-contained: Monty cannot import other overlay files.)

## Building the Agent

`DeepAgent` mirrors `FunctionCallingAgent` — every parameter on that
class is accepted with identical semantics. The only additions are
`workdir` (required) and the sandbox `timeout` (the per-snippet budget
for `run_python_code` / `run_python_file`).

```python
import synalinks

agent = synalinks.DeepAgent(
    workdir="/tmp/my_project",
    language_model=lm,
    timeout=30,         # per-run_python_code budget, seconds
    max_iterations=10,  # coding tasks tend to need more rounds than RAG/SQL
)
```

There is no "read-only" switch to set: the copy-on-write overlay
already guarantees the real workdir is never modified, so even an agent
that writes, edits, and runs code leaves your files on disk untouched.
Review what it *would* have changed with `agent.sandbox.changes()`.

User-supplied extra tools (e.g. a date helper, a web search) are
passed via `tools=` and merged with the built-ins. The same name-
collision and leading-underscore rules apply as for every other
FunctionCallingAgent-derived class.

## Example Usage

This example creates a small Python project, asks the agent to add a
function, and then inspects the agent's overlay to see the edit — while
the file on disk stays untouched.

```python
result = await agent(ChatMessages(messages=[
    ChatMessage(
        role="user",
        content=(
            "Open calculator.py and add a `multiply(a, b)` function "
            "next to `add`, then read it back to confirm."
        ),
    )
]))
```

The agent will typically:

1. `list_files("**/*")` to see what's in the workdir.
2. `read_file("/calculator.py")` to read existing code.
3. `edit_file(...)` to insert the new function (or `write_file` for a
   full rewrite if the file is short).
4. `read_file(...)` again to confirm, then stop and answer.

## Key Takeaways

- **One module, seven tools**: `synalinks.DeepAgent` bundles file IO,
  search, and a Python sandbox into a single ready-to-use agent.
- **Host-safe by construction**: the tools are a copy-on-write overlay,
  so the agent can explore and edit freely without touching the disk —
  there are no capability gates to set, because nothing it does is ever
  written back.
- **Inspect & persist**: `agent.sandbox.changes()` / `.journal()` /
  `.read_overlay(path)` show exactly what the agent did; you decide what
  (if anything) to write to disk.
- **Token-aware**: `read_file` / `list_files` / `search_files` paginate
  with `offset` / `limit` (1-based, grep convention).

## API References

- [DeepAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/DeepAgent%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [ChatMessages (Base DataModels)](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)
"""

# --8<-- [start:source]
import asyncio
import os
import shutil
import tempfile

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Workspace setup
# =============================================================================


def populate_workspace(workdir: str) -> None:
    """Seed the workdir with a tiny Python project for the agent to work on."""
    os.makedirs(workdir, exist_ok=True)

    # A starter module the agent will extend.
    with open(os.path.join(workdir, "calculator.py"), "w") as f:
        f.write('def add(a, b):\n    """Return a + b."""\n    return a + b\n')

    # A README the agent can read to understand context.
    with open(os.path.join(workdir, "README.md"), "w") as f:
        f.write(
            "# Calculator\n"
            "\n"
            "A tiny module. Currently supports `add(a, b)`.\n"
            "Planned: `multiply(a, b)`.\n"
        )


def print_messages(result) -> None:
    """Pretty-print the agent's tool-call trajectory."""
    messages = result.get("messages", [])
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                args = call.get("arguments", {})
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                print(f"  Tool call: {call['name']}({args_str})")
        elif role == "tool":
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = str(content)
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"  Tool result: {content}")
        elif role == "assistant" and msg.get("content"):
            print(f"  Assistant: {msg['content']}")


# =============================================================================
# Main example
# =============================================================================


async def main():
    """Run a small end-to-end task with the deep agent."""
    load_dotenv()
    synalinks.clear_session()

    workdir = tempfile.mkdtemp(prefix="deep_agent_demo_")
    print(f"Workdir: {workdir}\n")

    try:
        populate_workspace(workdir)

        lm = synalinks.LanguageModel(model="gemini/gemini-3.1-flash-lite-preview")

        # Keep a reference to the DeepAgent so we can inspect its sandbox
        # (the copy-on-write overlay) after the run.
        deep_agent = synalinks.DeepAgent(
            workdir=workdir,
            language_model=lm,
            max_iterations=10,
        )
        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        outputs = await deep_agent(inputs)

        agent = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="deep_agent",
            description="A coding agent with a sandboxed copy of a workdir.",
        )

        agent.summary()

        # ---------------------------------------------------------------------
        # Task 1: explore the workdir
        # ---------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Task 1: Explore")
        print("=" * 60)

        question = synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content=(
                        "What's in this directory? List the files and tell "
                        "me what the project is about. Be brief."
                    ),
                )
            ]
        )
        result = await agent(question)
        print_messages(result)

        # ---------------------------------------------------------------------
        # Task 2: extend the calculator (in the overlay, host-safe)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Task 2: Extend (overlay edit)")
        print("=" * 60)

        task = synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content=(
                        "Open calculator.py and add a `multiply(a, b)` function "
                        "that returns a * b, in the same style as `add`. Then "
                        "read the file back to confirm the change."
                    ),
                )
            ]
        )
        result = await agent(task)
        print_messages(result)

        # The edit lives in the sandbox overlay — show it, and show that the
        # file on disk is untouched.
        print("\n--- calculator.py (sandbox overlay) ---")
        overlay = await deep_agent.sandbox.read_file("/calculator.py")
        print(overlay.get("content", overlay))

        print("\n--- calculator.py (on disk, unchanged) ---")
        with open(os.path.join(workdir, "calculator.py")) as f:
            print(f.read())

        print("--- overlay changes ---")
        print(deep_agent.sandbox.changes())

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
