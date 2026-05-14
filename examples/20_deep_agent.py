"""
# Deep Agent

A **Deep Agent** is a coding-style agent with direct access to the
filesystem and a shell, scoped to a single working directory. Where a
`FunctionCallingAgent` typically calls a few narrow tools (search,
calculate, fetch), a deep agent treats a workspace as its environment
and edits it in place — read files, grep them, write or patch them,
then run a command to validate the result.

## What Deep Agents Can Do

```mermaid
graph LR
    A[User Task] --> B[DeepAgent]
    B --> C{Pick Tool}
    C -->|Discover| D[list_directory]
    C -->|Find/grep| E[search_files]
    C -->|Read code| F[read_file]
    C -->|Patch| G[edit_file]
    C -->|Create| H[write_file]
    C -->|Run| I[run_bash]
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
2. **Bug fix loop**: find the failing test, read the relevant module,
   patch it, re-run the test, iterate.
3. **Project bootstrapping**: scaffold a small project from scratch
   (write files, run `python -m pytest` to confirm).
4. **Data wrangling**: search a directory of CSVs / logs, extract
   what's relevant, write a report.

## The Six Built-in Tools

| Tool | Purpose |
|------|---------|
| `list_directory` | Enumerate a directory (name, type, size). |
| `search_files` | Glob for files and/or grep their contents (regex). |
| `read_file` | Line-paginated file read, output prefixed with line numbers (`cat -n` style). |
| `write_file` | Create or overwrite a file. Gated by `allow_write`. |
| `edit_file` | Exact-string replacement; rejects 0 or 2+ occurrences. Gated by `allow_write`. |
| `run_bash` | Execute a shell command with timeout. Gated by `allow_bash`. |

## Security Model

The deep agent enforces two different containment guarantees, and
it's important to know which is real and which is *not*.

**File tools** (`read_file` / `write_file` / `edit_file` /
`list_directory` / `search_files`) refuse any path that resolves
outside the workdir, including:

- `..` traversal (`subdir/../../etc/passwd`)
- Absolute paths (`/etc/passwd`)
- Symlinks pointing outside the workdir (the symlink is resolved
  during the path check, so the target's location is what's
  compared)

Paths are canonicalized with `Path.resolve()` (which flattens `..`
and follows existing symlinks) and then prefix-checked against the
resolved workdir. File opens use `O_NOFOLLOW` where the OS supports
it as defense in depth against TOCTOU symlink-swap races. This is a
robust boundary.

**Bash is not sandboxed.** The shell runs with `cwd=workdir`, but
that's just where its prompt starts — the LM can write `cat
/etc/passwd` and the shell will happily read it. The Python layer
cannot make `run_bash` safe on its own. If you're running this on
untrusted input, run the host process inside a container or other
OS-level isolation, or disable bash with `allow_bash=False`.

## Building the Agent

`DeepAgent` mirrors `FunctionCallingAgent` — every parameter on that
class is accepted with identical semantics. The only additions are
`workdir` (required), the per-tool gates (`allow_write`, `allow_bash`),
and a few output-shaping knobs (`timeout`, `max_output_chars`,
`max_search_results`).

```python
import synalinks

agent = synalinks.DeepAgent(
    workdir="/tmp/my_project",
    language_model=lm,
    allow_write=True,   # default
    allow_bash=True,    # default
    timeout=30,         # per-bash-command timeout, seconds
    max_iterations=10,  # coding tasks tend to need more rounds than RAG/SQL
)
```

Read-only mode for code review without write privileges:

```python
agent = synalinks.DeepAgent(
    workdir="/path/to/repo",
    language_model=lm,
    allow_write=False,  # only read_file/list_directory/search_files (+ run_bash if allowed)
    allow_bash=False,   # purely static inspection
)
```

User-supplied extra tools (e.g. a date helper, a web search) are
passed via `tools=` and merged with the built-ins. The same name-
collision and leading-underscore rules apply as for every other
FunctionCallingAgent-derived class.

## Example Usage

This example creates a small Python project, asks the agent to add a
function, then verifies the addition by running a quick check.

```python
result = await agent(ChatMessages(messages=[
    ChatMessage(
        role="user",
        content=(
            "Open the calculator.py file, add a `multiply(a, b)` "
            "function next to `add`, and run `python -c 'from "
            "calculator import multiply; print(multiply(6, 7))'` "
            "to confirm it works."
        ),
    )
]))
```

The agent will typically:

1. `list_directory(".")` to see what's in the workdir.
2. `read_file("calculator.py")` to read existing code.
3. `edit_file(...)` to insert the new function (or `write_file` for a
   full rewrite if the file is short).
4. `run_bash("python -c '...'")` to validate.
5. Stop and answer.

## Key Takeaways

- **One module, six tools**: `synalinks.DeepAgent` bundles file IO,
  search, and shell execution into a single ready-to-use agent.
- **Real path-traversal defense**: file tools refuse anything outside
  the workdir, including symlink escapes.
- **Bash is not sandboxed**: containerize if you need true isolation.
- **Read-only mode**: set `allow_write=False` (and optionally
  `allow_bash=False`) for inspector agents that audit code without
  changing it.
- **Token-aware**: `read_file` returns line-numbered output (LMs cite
  line numbers, no re-reading needed); `search_files` caps results;
  `run_bash` truncates stdout/stderr.

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
        f.write(
            "def add(a, b):\n"
            "    \"\"\"Return a + b.\"\"\"\n"
            "    return a + b\n"
        )

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

        # Build the agent. autonomous=True runs the tool loop end-to-end;
        # max_iterations=10 gives the LM enough rounds for a multi-step task.
        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        outputs = await synalinks.DeepAgent(
            workdir=workdir,
            language_model=lm,
            max_iterations=10,
        )(inputs)

        agent = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="deep_agent",
            description="A coding agent with file and shell access.",
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
        # Task 2: extend the calculator
        # ---------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Task 2: Extend & verify")
        print("=" * 60)

        task = synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content=(
                        "Open calculator.py and add a `multiply(a, b)` function "
                        "that returns a * b, in the same style as `add`. "
                        "Then run `python -c 'from calculator import multiply; "
                        "print(multiply(6, 7))'` and tell me what it printed."
                    ),
                )
            ]
        )
        result = await agent(task)
        print_messages(result)

        # Show the final file content so we can verify ourselves.
        print("\n--- Final calculator.py ---")
        with open(os.path.join(workdir, "calculator.py")) as f:
            print(f.read())

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
