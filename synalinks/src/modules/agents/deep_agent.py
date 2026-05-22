# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from pathlib import Path
from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.sandboxes.monty_sandbox import MontySandbox
from synalinks.src.saving import serialization_lib


def get_default_instructions(workdir: Optional[str]) -> str:
    """Default system instructions for the deep agent.

    Args:
        workdir: Absolute path of the agent's working directory, or
            ``None`` for an empty in-memory workspace. Embedded in the
            prompt so the LM knows where it's operating.

    Returns:
        A prompt string describing the tool plan.
    """
    if workdir:
        workdir_line = f"Workdir: {workdir}"
    else:
        workdir_line = (
            "Workdir: (none) — an empty in-memory workspace; "
            "create files with `write_file`."
        )
    return f"""
You are a software engineering assistant working inside a sandboxed,
copy-on-write filesystem.

{workdir_line}
Available tools: read_file, list_files, search_files, write_file,
edit_file, run_python_code, run_python_file

Plan:
1. Use `list_files` to discover files (glob, e.g. `**/*.py`).
2. Use `search_files` to grep file contents by regex across a glob.
3. Use `read_file` to read a file; it returns the requested lines with
   1-based `start_line` / `end_line`. Page through long files with
   `offset` / `limit` (raise `offset` to read further in).
4. Use `edit_file` for surgical changes (preferred over `write_file`).
5. Use `run_python_code` to run a Python snippet directly, or
   `write_file` a self-contained script into the overlay then
   `run_python_file(path)` to execute it (a script cannot import other
   overlay files).
6. Once you have the answer, stop calling tools and respond.

Notes:
- The filesystem is copy-on-write: edits and new files land in an
  in-memory overlay and never modify the real workspace on disk.
- Paths are rooted at the workdir; `..` cannot escape it.""".strip()


@synalinks_export(
    [
        "synalinks.modules.DeepAgent",
        "synalinks.DeepAgent",
    ]
)
class DeepAgent(Module):
    """A coding agent whose tools are a sandboxed copy of a workdir.

    DeepAgent is a thin specialization of :class:`FunctionCallingAgent`
    that mounts the workdir in a :class:`MontySandbox` and exposes the
    sandbox's tool methods to the LM:

    - ``read_file``: read a file by 1-based line range (paginated).
    - ``list_files``: list files matching a glob.
    - ``search_files``: glob for files and grep their contents (regex).
    - ``write_file``: create/overwrite a file.
    - ``edit_file``: exact-string replacement.
    - ``run_python_code``: run a Python snippet directly in the sandbox.
    - ``run_python_file``: run a self-contained script the agent wrote
      into the overlay.

    Every tool is backed by the sandbox's copy-on-write overlay and the
    Monty interpreter, so the agent is **host-safe by construction**:
    reads fall through to the real ``workdir`` but writes, edits and code
    execution can never modify it or reach the host — so there is nothing
    to gate, and all tools are always available. Inspect what the agent
    did through ``agent.sandbox`` — ``changes()``, ``journal()``,
    ``read_overlay()`` — and persist any of it yourself if desired.

    The constructor mirrors :class:`FunctionCallingAgent` — every
    parameter on that class is accepted here with identical semantics.
    The additions are ``workdir`` (required) and the sandbox ``timeout``.
    User-supplied ``tools`` are appended to the built-in ones.

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
        workdir (str): Optional working directory the agent operates on.
            When given it must exist and is mounted read-through in the
            sandbox (the LM's writes/edits stay in the overlay and never
            touch it). When omitted, the sandbox starts as an empty
            in-memory filesystem.
        timeout (float): Per-snippet execution budget in seconds for
            ``run_python_code`` / ``run_python_file``. Defaults to 30.
        tools (list): Additional :class:`Tool` instances (or plain async
            functions) to expose alongside the built-in tools. Names must
            not start with ``_`` or collide with built-ins.
        schema (dict): JSON schema for the final answer.
        data_model (DataModel): DataModel for the final answer.
            Mutually exclusive with ``schema``.
        language_model (LanguageModel): The language model that drives
            the agent loop.
        prompt_template (str): Forwarded to the tool-call generator.
        examples (list): Few-shot examples for the tool-call generator.
        instructions (str): Override the default system instructions.
            When omitted, the default is built from the workdir.
        final_instructions (str): Instructions for the final-answer
            generator. Defaults to ``instructions``.
        temperature (float): LM sampling temperature. Defaults to 0.0.
        use_inputs_schema (bool): Include the input schema in the prompt.
        use_outputs_schema (bool): Include the output schema in the prompt.
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When ``True``, the tool-call
            generator emits a ``thinking`` field per round.
        autonomous (bool): When ``True`` (default), the agent runs the
            tool loop end-to-end. When ``False``, returns one step at a
            time for human-in-the-loop workflows.
        return_inputs_with_trajectory (bool): When ``True`` (default),
            the full message trajectory is included alongside the final
            answer.
        max_iterations (int): Maximum number of tool-call rounds.
            Defaults to 10 (coding tasks tend to need more rounds than
            RAG / SQL).
        streaming (bool): Stream the final answer when no ``schema`` is
            set. Defaults to ``False``.
        name (str): Module name.
        description (str): Module description.
    """

    def __init__(
        self,
        *,
        workdir: Optional[str] = None,
        timeout: float = 30.0,
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

        # `workdir` is optional: when omitted the sandbox is a pure
        # in-memory filesystem. A provided path must exist and be a dir.
        if workdir:
            resolved_workdir = Path(workdir).resolve()
            if not resolved_workdir.exists():
                raise ValueError(f"workdir does not exist: {workdir}")
            if not resolved_workdir.is_dir():
                raise ValueError(f"workdir is not a directory: {workdir}")
            self.workdir = str(resolved_workdir)
        else:
            self.workdir = None

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"`timeout` must be a positive number, got {timeout!r}")
        self.timeout = float(timeout)

        self.language_model = _get_lm(language_model)

        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        if instructions is None:
            instructions = get_default_instructions(self.workdir)
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

        # The sandbox IS the filesystem the tools operate on: reads fall
        # through to the workdir, writes/edits/code stay host-safe in the
        # overlay. Inspect via `self.sandbox.changes()` / `.journal()`.
        self.sandbox = MontySandbox(workdir=self.workdir, timeout=self.timeout)
        builtin_fns = [
            self.sandbox.read_file,
            self.sandbox.list_files,
            self.sandbox.search_files,
            self.sandbox.write_file,
            self.sandbox.edit_file,
            self.sandbox.run_python_code,
            self.sandbox.run_python_file,
        ]
        builtin_tools = [Tool(fn) for fn in builtin_fns]
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
            "timeout": self.timeout,
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
