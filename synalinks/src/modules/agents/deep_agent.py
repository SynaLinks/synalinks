# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import os
import re
from pathlib import Path
from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


class PathTraversalError(ValueError):
    """Raised when a tool argument resolves outside the configured workdir."""


def _resolve_inside_workdir(workdir: Path, user_path: str) -> Path:
    """Resolve ``user_path`` relative to ``workdir`` and verify containment.

    Defense layers:
    1. ``Path.resolve()`` normalizes ``..`` and follows existing symlinks,
       so any path that escapes the workdir surfaces as an absolute path
       outside it (rather than as a literal string we'd have to parse).
    2. ``is_relative_to`` compares the resolved absolute paths — if the
       caller-supplied path lands anywhere outside the workdir, the
       check fails and we refuse the operation.
    """
    resolved_workdir = workdir.resolve()
    candidate = (resolved_workdir / user_path) if not os.path.isabs(user_path) else Path(user_path)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_workdir)
    except ValueError:
        raise PathTraversalError(
            f"Path {user_path!r} resolves outside the agent workdir."
        )
    return resolved


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Cap a string to ``max_chars``. Returns ``(text, truncated)``."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def get_default_instructions(
    workdir: str,
    allow_write: bool,
    allow_bash: bool,
) -> str:
    """Default system instructions for the deep agent.

    Args:
        workdir: Absolute path of the agent's working directory.
            Embedded in the prompt so the LM knows where it's
            operating.
        allow_write: Whether write/edit tools are enabled.
        allow_bash: Whether the bash tool is enabled.

    Returns:
        A prompt string describing the tool plan and the safety
        constraints currently in effect.
    """
    capabilities = ["read_file", "list_directory", "search_files"]
    if allow_write:
        capabilities.extend(["write_file", "edit_file"])
    if allow_bash:
        capabilities.append("run_bash")

    extras = []
    if not allow_write:
        extras.append("Write/edit tools are DISABLED — this is a read-only session.")
    if not allow_bash:
        extras.append("Shell execution is DISABLED.")
    constraints = ("\n".join(f"- {line}" for line in extras) + "\n") if extras else ""

    return f"""
You are a software engineering assistant with filesystem and shell access
scoped to a single working directory.

Workdir: {workdir}
Available tools: {capabilities}

Plan:
1. Use `list_directory` to discover what's in the workdir.
2. Use `search_files` to locate files by glob and/or grep their contents.
3. Use `read_file` to read files. Output is line-numbered (``cat -n``
   style). Pages of lines via `offset` / `limit`; raise `offset` to read
   further into the file.
4. {"Use `edit_file` for surgical changes (preferred over `write_file`)." if allow_write else "Reads only — do not propose write operations."}
5. {"Use `run_bash` for builds, tests, and other shell work." if allow_bash else "Shell is disabled — solve tasks with file tools only."}
6. Once you have the answer, stop calling tools and respond.

Constraints:
- All paths must stay inside the workdir. ``..`` traversal and absolute
  paths that escape the workdir are rejected.
{constraints}""".strip()


_SEARCH_MAX_FILE_BYTES = 1_000_000  # Skip files larger than this when grepping.
_SEARCH_MAX_FILES_SCANNED = 5_000  # Hard cap on the glob result set.


def _build_tools(
    workdir: Path,
    *,
    allow_write: bool = True,
    allow_bash: bool = True,
    timeout: float = 30.0,
    max_output_chars: int = 10_000,
    max_search_results: int = 100,
):
    """Build the deep-agent tools bound to a workdir.

    Args:
        workdir: The directory all file operations are scoped to. All
            user-supplied paths are resolved relative to it and any
            path that escapes it is rejected with
            :class:`PathTraversalError`.
        allow_write: Include ``write_file`` and ``edit_file``.
        allow_bash: Include ``run_bash``.
        timeout: Per-command bash timeout, in seconds.
        max_output_chars: Cap on bytes returned from
            ``read_file`` / ``run_bash`` (per stream). Excess content
            is truncated and the result reports ``truncated=True``.
        max_search_results: Cap on entries returned by ``search_files``
            (matching files or matching lines, depending on mode).

    Returns:
        A list of plain async functions. Tools are filtered by the
        ``allow_*`` flags; callers wrap with :class:`Tool` themselves.
    """

    async def read_file(path: str, offset: int, limit: int):
        """Read a file with line-based pagination.

        Returns the requested lines prefixed with 1-based line numbers
        in ``cat -n`` style (``{lineno}\\t{content}``), so the LM can
        cite line numbers in subsequent ``edit_file`` calls without
        re-reading.

        Args:
            path (str): Relative path within the workdir.
            offset (int): 0-indexed line number to start reading from.
                ``0`` reads from the top; ``100`` skips the first 100
                lines.
            limit (int): Maximum number of lines to return. Each line
                is also truncated to the agent's ``max_output_chars``
                cap so a binary or minified file can't blow up the
                response.
        """
        try:
            resolved = _resolve_inside_workdir(workdir, path)
        except PathTraversalError as e:
            return {"error": str(e), "path": path}
        if not resolved.exists():
            return {"error": f"File not found: {path}", "path": path}
        if not resolved.is_file():
            return {"error": f"Not a regular file: {path}", "path": path}
        try:
            # O_NOFOLLOW so a symlink swapped in after the path check
            # can't redirect us outside the workdir.
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved, flags)
        except OSError as e:
            return {"error": str(e), "path": path}
        offset = max(0, offset)
        limit = max(1, limit)
        rendered_lines: List[str] = []
        total_lines = 0
        try:
            with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, start=1):
                    total_lines = lineno
                    if lineno <= offset:
                        continue
                    if len(rendered_lines) < limit:
                        stripped = line.rstrip("\n")
                        if len(stripped) > max_output_chars:
                            stripped = stripped[:max_output_chars]
                        rendered_lines.append(f"{lineno}\t{stripped}")
                    # Keep iterating so total_lines reflects the full file —
                    # cheap on small files, the only honest answer on big ones.
        except OSError as e:
            return {"error": str(e), "path": path}
        has_more = total_lines > offset + len(rendered_lines)
        return {
            "path": path,
            "content": "\n".join(rendered_lines),
            "offset": offset,
            "lines_returned": len(rendered_lines),
            "total_lines": total_lines,
            "has_more": has_more,
        }

    async def list_directory(path: str):
        """List the contents of a directory.

        Args:
            path (str): Relative path within the workdir. Use ``"."``
                for the workdir root.
        """
        try:
            resolved = _resolve_inside_workdir(workdir, path)
        except PathTraversalError as e:
            return {"error": str(e), "path": path}
        if not resolved.exists():
            return {"error": f"Directory not found: {path}", "path": path}
        if not resolved.is_dir():
            return {"error": f"Not a directory: {path}", "path": path}
        entries = []
        for child in sorted(resolved.iterdir()):
            entries.append(
                {
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return {"path": path, "entries": entries, "entry_count": len(entries)}

    async def search_files(file_pattern: str, content_pattern: str):
        """Find files by glob, optionally grepping their contents.

        Two modes selected by ``content_pattern``:

        - **Glob-only** (``content_pattern=""``): returns the list of
          files matching ``file_pattern`` under the workdir.
        - **Glob + grep** (``content_pattern`` non-empty): same file
          set, then each file is scanned line-by-line and matching
          ``(path, line_number, line)`` triples are returned.

        Files larger than 1 MB and files that aren't valid UTF-8 are
        skipped silently. Symlinks pointing outside the workdir are
        skipped. The total result set is capped at the agent's
        ``max_search_results`` setting; long matching lines are
        truncated to ``max_output_chars``.

        Args:
            file_pattern (str): Glob pattern relative to the workdir.
                Use ``"**/*.py"`` for "all .py files recursively",
                ``"*.md"`` for top-level markdown, ``"**/*"`` for
                everything. Standard ``fnmatch`` syntax.
            content_pattern (str): Python regex to match within each
                file. Pass an empty string for glob-only mode.
        """
        resolved_workdir = workdir.resolve()
        try:
            matched_paths = list(resolved_workdir.glob(file_pattern))
        except (ValueError, OSError) as e:
            return {"error": f"Invalid file_pattern: {e}", "file_pattern": file_pattern}
        # Filter to files inside the workdir, skipping symlink escapes
        # and capping the scan at _SEARCH_MAX_FILES_SCANNED.
        files: List[Path] = []
        for p in matched_paths:
            if len(files) >= _SEARCH_MAX_FILES_SCANNED:
                break
            try:
                rp = p.resolve()
                rp.relative_to(resolved_workdir)
            except (OSError, ValueError):
                continue
            if rp.is_file():
                files.append(rp)

        # Glob-only mode: return paths sorted, capped at max_search_results.
        if not content_pattern:
            files.sort()
            truncated = len(files) > max_search_results
            rel_paths = [
                str(p.relative_to(resolved_workdir))
                for p in files[:max_search_results]
            ]
            return {
                "file_pattern": file_pattern,
                "content_pattern": "",
                "files": rel_paths,
                "match_count": len(rel_paths),
                "truncated": truncated,
            }

        # Grep mode: compile regex once, scan files line-by-line.
        try:
            regex = re.compile(content_pattern)
        except re.error as e:
            return {
                "error": f"Invalid content_pattern regex: {e}",
                "content_pattern": content_pattern,
            }

        matches = []
        truncated = False
        for fp in sorted(files):
            if len(matches) >= max_search_results:
                truncated = True
                break
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if size > _SEARCH_MAX_FILE_BYTES:
                continue
            try:
                with fp.open("r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, start=1):
                        if regex.search(line):
                            stripped = line.rstrip("\n")
                            if len(stripped) > max_output_chars:
                                stripped = stripped[:max_output_chars]
                            matches.append(
                                {
                                    "path": str(fp.relative_to(resolved_workdir)),
                                    "line_number": lineno,
                                    "line": stripped,
                                }
                            )
                            if len(matches) >= max_search_results:
                                truncated = True
                                break
            except (OSError, UnicodeDecodeError):
                # Binary or unreadable file — skip.
                continue
        return {
            "file_pattern": file_pattern,
            "content_pattern": content_pattern,
            "matches": matches,
            "match_count": len(matches),
            "truncated": truncated,
        }

    async def write_file(path: str, content: str):
        """Write content to a file, creating or overwriting it.

        Args:
            path (str): Relative path within the workdir.
            content (str): The full file content to write.
        """
        try:
            resolved = _resolve_inside_workdir(workdir, path)
        except PathTraversalError as e:
            return {"error": str(e), "path": path}
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            existed = resolved.exists()
            resolved.write_text(content, encoding="utf-8")
        except OSError as e:
            return {"error": str(e), "path": path}
        return {
            "path": path,
            "bytes_written": len(content.encode("utf-8")),
            "created": not existed,
        }

    async def edit_file(path: str, old_string: str, new_string: str):
        """Replace an exact string in a file with another.

        The ``old_string`` must appear exactly once in the file or
        the edit is rejected. This avoids accidental multi-site
        substitutions; for renames, call ``edit_file`` once per
        occurrence with surrounding context.

        Args:
            path (str): Relative path within the workdir.
            old_string (str): The exact substring to find. Must
                appear exactly once.
            new_string (str): The replacement substring.
        """
        try:
            resolved = _resolve_inside_workdir(workdir, path)
        except PathTraversalError as e:
            return {"error": str(e), "path": path}
        if not resolved.is_file():
            return {"error": f"File not found: {path}", "path": path}
        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as e:
            return {"error": str(e), "path": path}
        occurrences = content.count(old_string)
        if occurrences == 0:
            return {"error": "old_string not found in file", "path": path}
        if occurrences > 1:
            return {
                "error": (
                    f"old_string appears {occurrences} times; expected exactly 1. "
                    f"Add surrounding context to disambiguate."
                ),
                "path": path,
            }
        new_content = content.replace(old_string, new_string, 1)
        try:
            resolved.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return {"error": str(e), "path": path}
        return {
            "path": path,
            "old_length": len(old_string),
            "new_length": len(new_string),
        }

    async def run_bash(command: str):
        """Run a shell command and return its output.

        The command runs with the workdir as its cwd, but the shell
        itself is NOT sandboxed — it can read any file the host
        process can read. Use OS-level isolation (containers,
        namespaces) if running on untrusted input.

        Args:
            command (str): A shell command to execute.
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workdir),
            )
        except OSError as e:
            return {"error": str(e), "command": command}
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return {
                "error": f"Command timed out after {timeout}s",
                "command": command,
                "timeout": timeout,
            }
        stdout, stdout_trunc = _truncate(
            stdout_b.decode("utf-8", errors="replace"), max_output_chars
        )
        stderr, stderr_trunc = _truncate(
            stderr_b.decode("utf-8", errors="replace"), max_output_chars
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_trunc,
            "stderr_truncated": stderr_trunc,
        }

    tools = [read_file, list_directory, search_files]
    if allow_write:
        tools.extend([write_file, edit_file])
    if allow_bash:
        tools.append(run_bash)
    return tools


@synalinks_export(
    [
        "synalinks.modules.DeepAgent",
        "synalinks.DeepAgent",
    ]
)
class DeepAgent(Module):
    """A coding agent with filesystem and shell access scoped to a workdir.

    DeepAgent is a thin specialization of :class:`FunctionCallingAgent`
    that pre-wires up to six workspace tools:

    - ``read_file``: read a file with line-based pagination, output
      prefixed with 1-based line numbers (``cat -n`` style).
    - ``list_directory``: list entries in a directory.
    - ``search_files``: glob for files and optionally grep their
      contents (regex). Combines find and grep in one call.
    - ``write_file``: overwrite or create a file (gated by ``allow_write``).
    - ``edit_file``: exact-string replacement, one occurrence at a time
      (gated by ``allow_write``).
    - ``run_bash``: run a shell command (gated by ``allow_bash``).

    The constructor mirrors :class:`FunctionCallingAgent` — every
    parameter on that class is accepted here with identical semantics.
    The only additions are ``workdir`` (required) and the safety
    knobs: ``allow_write``, ``allow_bash``, ``timeout``,
    ``max_output_chars``. User-supplied ``tools`` are appended to the
    built-in ones.

    ## Security model

    File tools (``read_file`` / ``write_file`` / ``edit_file`` /
    ``list_directory``) refuse any path that resolves outside the
    workdir, including ``..`` traversal and absolute paths. Paths are
    canonicalized via ``Path.resolve()`` (which flattens ``..`` and
    follows existing symlinks) and then prefix-checked against the
    resolved workdir, so a symlink-inside-workdir pointing to
    ``/etc/passwd`` is also caught. File opens use ``O_NOFOLLOW``
    where the OS supports it as defense in depth against TOCTOU
    symlink swaps.

    The bash tool is **NOT sandboxed**. Its ``cwd`` is the workdir,
    but the shell can still read or write any path the host process
    can. If you're running this on untrusted input, run the host
    process inside a container or other OS-level isolation; the
    Python layer cannot make ``run_bash`` safe on its own. Disable
    it with ``allow_bash=False`` when you don't need it.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():
        lm = synalinks.LanguageModel(model="ollama/mistral")

        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        outputs = await synalinks.DeepAgent(
            workdir="/tmp/my_project",
            language_model=lm,
        )(inputs)
        agent = synalinks.Program(inputs=inputs, outputs=outputs)

        messages = synalinks.ChatMessages(messages=[
            synalinks.ChatMessage(
                role="user",
                content="What's in this directory?",
            )
        ])
        result = await agent(messages)
        print(result.get("messages")[-1].get("content"))

    asyncio.run(main())
    ```

    Args:
        workdir (str): Working directory the agent operates in.
            Required. Must exist. All file paths supplied by the LM
            are resolved relative to it and rejected if they escape.
        allow_write (bool): When ``False``, ``write_file`` and
            ``edit_file`` are omitted from the tool set. Defaults to
            ``True``.
        allow_bash (bool): When ``False``, ``run_bash`` is omitted.
            Defaults to ``True``.
        timeout (float): Per-command bash timeout in seconds.
            Defaults to 30.
        max_output_chars (int): Cap on characters returned per
            stream from ``read_file`` (single stream) and ``run_bash``
            (stdout and stderr each). Also caps the length of each
            matching line returned by ``search_files``. Defaults to
            10000.
        max_search_results (int): Cap on entries returned by
            ``search_files`` (matching files or matching lines).
            Defaults to 100.
        tools (list): Additional :class:`Tool` instances (or plain
            async functions) to expose alongside the built-in tools.
            Names must not start with ``_`` or collide with built-ins.
        schema (dict): JSON schema for the final answer.
        data_model (DataModel): DataModel for the final answer.
            Mutually exclusive with ``schema``.
        language_model (LanguageModel): The language model that drives
            the agent loop.
        prompt_template (str): Forwarded to the tool-call generator.
        examples (list): Few-shot examples for the tool-call generator.
        instructions (str): Override the default system instructions.
            When omitted, the default is built from the workdir and
            the configured permissions.
        final_instructions (str): Instructions for the final-answer
            generator. Defaults to ``instructions``.
        temperature (float): LM sampling temperature. Defaults to 0.0.
        use_inputs_schema (bool): Include the input schema in the
            prompt.
        use_outputs_schema (bool): Include the output schema in the
            prompt.
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When ``True``, the tool-call
            generator emits a ``thinking`` field per round.
        autonomous (bool): When ``True`` (default), the agent runs
            the tool loop end-to-end. When ``False``, returns one
            step at a time for human-in-the-loop workflows.
        return_inputs_with_trajectory (bool): When ``True`` (default),
            the full message trajectory is included alongside the
            final answer.
        max_iterations (int): Maximum number of tool-call rounds.
            Defaults to 10 (coding tasks tend to need more rounds
            than RAG / SQL).
        streaming (bool): Stream the final answer when no ``schema``
            is set. Defaults to ``False``.
        name (str): Module name.
        description (str): Module description.
    """

    def __init__(
        self,
        *,
        workdir: str,
        allow_write: bool = True,
        allow_bash: bool = True,
        timeout: float = 30.0,
        max_output_chars: int = 10_000,
        max_search_results: int = 100,
        tools: Optional[List] = None,
        schema=None,
        data_model=None,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions: Optional[str] = None,
        final_instructions: Optional[str] = None,
        temperature: float = 0.0,
        use_inputs_schema: bool = False,
        use_outputs_schema: bool = False,
        reasoning_effort: Optional[str] = None,
        use_chain_of_thought: bool = False,
        autonomous: bool = True,
        return_inputs_with_trajectory: bool = True,
        max_iterations: int = 10,
        streaming: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(name=name, description=description)

        if not workdir:
            raise ValueError("`workdir` is required")
        resolved_workdir = Path(workdir).resolve()
        if not resolved_workdir.exists():
            raise ValueError(f"workdir does not exist: {workdir}")
        if not resolved_workdir.is_dir():
            raise ValueError(f"workdir is not a directory: {workdir}")
        self.workdir = str(resolved_workdir)

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"`timeout` must be a positive number, got {timeout!r}")
        self.timeout = float(timeout)

        if not isinstance(max_output_chars, int) or max_output_chars < 1:
            raise ValueError(
                f"`max_output_chars` must be a positive integer, got {max_output_chars!r}"
            )
        self.max_output_chars = max_output_chars

        if not isinstance(max_search_results, int) or max_search_results < 1:
            raise ValueError(
                f"`max_search_results` must be a positive integer, "
                f"got {max_search_results!r}"
            )
        self.max_search_results = max_search_results

        self.allow_write = bool(allow_write)
        self.allow_bash = bool(allow_bash)

        self.language_model = _get_lm(language_model)

        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        if instructions is None:
            instructions = get_default_instructions(
                self.workdir, self.allow_write, self.allow_bash
            )
        self.instructions = instructions
        self.final_instructions = final_instructions

        self.prompt_template = prompt_template
        self.examples = examples
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.reasoning_effort = reasoning_effort
        self.use_chain_of_thought = use_chain_of_thought
        self.autonomous = autonomous
        self.return_inputs_with_trajectory = return_inputs_with_trajectory
        self.max_iterations = max_iterations
        self.streaming = streaming

        builtin_tools = [
            Tool(fn)
            for fn in _build_tools(
                resolved_workdir,
                allow_write=self.allow_write,
                allow_bash=self.allow_bash,
                timeout=self.timeout,
                max_output_chars=self.max_output_chars,
                max_search_results=self.max_search_results,
            )
        ]
        builtin_names = {t.name for t in builtin_tools}

        self.extra_tools = list(tools) if tools else []
        merged_tools = list(builtin_tools)
        for extra in self.extra_tools:
            extra_tool = extra if isinstance(extra, Tool) else Tool(extra)
            if extra_tool.name in builtin_names:
                raise ValueError(
                    f"Tool name {extra_tool.name!r} collides with a built-in "
                    f"deep-agent tool. Rename the additional tool."
                )
            merged_tools.append(extra_tool)
        # Leading-underscore check is centralized in FunctionCallingAgent.

        self.agent = FunctionCallingAgent(
            schema=self.schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            final_instructions=self.final_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            reasoning_effort=self.reasoning_effort,
            use_chain_of_thought=self.use_chain_of_thought,
            tools=merged_tools,
            autonomous=self.autonomous,
            return_inputs_with_trajectory=self.return_inputs_with_trajectory,
            max_iterations=self.max_iterations,
            streaming=self.streaming,
            name="agent_" + self.name,
        )

    async def call(self, inputs, training=False):
        return await self.agent(inputs, training=training)

    async def compute_output_spec(self, inputs, training=False):
        return await self.agent.compute_output_spec(inputs, training=training)

    def get_config(self):
        config = {
            "workdir": self.workdir,
            "allow_write": self.allow_write,
            "allow_bash": self.allow_bash,
            "timeout": self.timeout,
            "max_output_chars": self.max_output_chars,
            "max_search_results": self.max_search_results,
            "schema": self.schema,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "final_instructions": self.final_instructions,
            "temperature": self.temperature,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "reasoning_effort": self.reasoning_effort,
            "use_chain_of_thought": self.use_chain_of_thought,
            "autonomous": self.autonomous,
            "return_inputs_with_trajectory": self.return_inputs_with_trajectory,
            "max_iterations": self.max_iterations,
            "streaming": self.streaming,
            "name": self.name,
            "description": self.description,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(
                    t if isinstance(t, Tool) else Tool(t)
                )
                for t in self.extra_tools
            ]
        }
        return {**config, **language_model_config, **tools_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        tools = [
            serialization_lib.deserialize_synalinks_object(t)
            for t in config.pop("tools", [])
        ]
        return cls(
            language_model=language_model,
            tools=tools,
            **config,
        )
