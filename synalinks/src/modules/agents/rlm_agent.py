# Modified from: dspy/predict/rlm.py
# Original authors: Alex L. Zhang, Tim Kraska, Omar Khattab (DSPy Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio

import jsonschema
from jsonschema import ValidationError

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_chat_messages
from synalinks.src.modules.agents.agent_utils import InputsSummary
from synalinks.src.modules.agents.agent_utils import summarize_inputs
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.sandboxes.monty_sandbox import MontySandbox
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.object_registration import get_registered_name
from synalinks.src.saving.object_registration import get_registered_object


def get_default_instructions():
    """Default instructions for non-recursive Python-snippet reasoning."""
    return """
You solve the user task by calling a SINGLE tool,
`run_python_code(code=...)`, which runs your snippet in a
persistent sandbox and returns `{"stdout": ..., "stderr": ..., "error": ...}`.
State persists across calls: variables, imports and function definitions stay
defined.

IMPORTANT: This is ITERATIVE. Each snippet runs, you read the output, then you
decide what to do next. Do NOT try to solve everything in one step.

The user input is bound as a dict named `inputs` in the sandbox, the full,
untruncated value. In the prompt you only see an `InputsSummary` with previews
and sizes; always read the real values through `inputs[field]` inside your
code, never re-type them from the preview.

Use `print(...)` to log intermediate observations. `submit` and any tools
bound to the agent are async callables *inside* the sandbox (see the tools
catalog), not separate tool calls — call them inside `async def main():` and
drive with `asyncio.run(main())`; calling without `await` returns a coroutine
object, not the value. Reach them only from the code you pass to
`run_python_code`.

Termination: call `submit(result={...})` from inside your snippet, with
`result` matching its schema. `submit` is async, so `await` it inside
`async def main(): ...`. It captures the answer and ends the run in one step.
If the payload fails schema validation you'll see the error on the next turn
and can retry.

`submit` is the only termination path; calling `run_python_code` with an
empty snippet is a no-op and you'll be reminded to call `submit`. Don't run
out of iterations without calling it.
""".strip()


def get_recursive_instructions():
    """Default instructions for recursive (sub-LM) Python-snippet reasoning.

    The ``{max_llm_calls}`` placeholder is substituted at construction time.
    """
    return """
You solve the user task by calling a SINGLE tool,
`run_python_code(code=...)`, with Python that programmatically
explores the inputs and recursively delegates semantic work to a sub-LM. It
runs your snippet in a persistent sandbox and returns
`{"stdout": ..., "stderr": ..., "error": ...}`; state persists across calls
(variables, imports and function definitions stay defined).

IMPORTANT: This is ITERATIVE. Each snippet runs, you read the output, then you
decide what to do next. Do NOT try to solve everything in one step.

Treat the inputs as an *external environment*, not as text in your
prompt. Long documents and large collections live in the sandbox and
are read with `inputs[field]`. The prompt only shows an
`InputsSummary` with previews and sizes, never re-type values from
the preview.

Two recursive helpers are always exposed in the tools catalog:

- `llm_query(prompt)`, query a sub-LM with one prompt; returns
  `{"result": <text>}`. Use it for semantic work on snippets you've
  already carved out with code (search, classification, summarization,
  reformatting). Pass *only the relevant snippet*, the sub-LM has its
  own context budget.
- `llm_query_batched(prompts)`, same, but takes a list and runs the
  prompts concurrently. Returns `{"result": [<text-or-error>, ...]}`
  preserving input order; failed prompts come back as strings prefixed
  with `[error] <ExceptionType>: <message>`, filter them before
  aggregating. Strongly preferred over a Python loop of `llm_query`
  calls, sequential calls waste wall time.

You have a hard budget of {max_llm_calls} sub-LM calls per run; the
counter is shared between `llm_query` and `llm_query_batched`. When
exhausted, both helpers short-circuit with
`{"result": <empty>, "error": "<msg>"}` without consuming any quota,
check `error` before trusting `result`. Plan recursion accordingly:
prefer code-side aggregation (regex, set ops, sorting,
dict-comprehension counting) over re-querying.

Use `print(...)` to log intermediate observations. All sandbox tools
are async, call them inside `async def main():` and drive with
`asyncio.run(main())`; calling without `await` yields a coroutine
object, not the value.

Working rules:

1. EXPLORE FIRST. Print sample values, lengths, types, and shapes of
   `inputs[field]` before slicing or batching. A cheap probe turn
   prevents wasted sub-LM calls on the wrong field or shape.
2. CODE FOR STRUCTURE, `llm_query` FOR MEANING. Regex, slicing, and
   set ops find WHERE things are; the sub-LM understands WHAT they
   mean. Don't burn `llm_query` budget on aggregation a one-liner can
   do.
3. MINIMIZE RETYPING. When values are long, precise, or error-prone
   (IDs, numbers, quoted text, code), re-access them via
   `inputs[field]` and compute in Python. Never copy from the
   `InputsSummary` preview into a sub-LM prompt, the preview is
   truncated.
4. VERIFY BEFORE SUBMITTING. If results look wrong (empty, zeros,
   unexpected shape), inspect them on a separate turn. Don't submit a
   guess.
5. `submit` IS TERMINAL. The snippet runs to completion (so a
   `print(...)` next to `submit(...)` is captured into the
   observation), but a successful submit ends the loop with no
   follow-up turn — you never get to read that print. Inspect on one
   turn, submit on the next.

Termination: call `submit(result={...})` from inside your snippet, with
`result` matching its schema. `submit`, `llm_query` and `llm_query_batched`
are async callables *inside* the sandbox (advertised in the tools catalog),
not separate tool calls — reach them only from the code you pass to
`run_python_code`. `submit` is the only termination path; an empty
snippet is a no-op and you'll be reminded to call `submit`. Don't run out of
iterations without calling it.
""".strip()


class ToolSpec(DataModel):
    """Description of one tool exposed in the sandbox."""

    name: str = Field(
        description=(
            "The async callable's name in the sandbox. Invoke with "
            "`await {name}(**kwargs)`."
        )
    )
    description: str = Field(
        description="What the tool does (from the Python docstring).",
    )
    parameters: dict = Field(
        description=(
            "JSON Schema for keyword arguments: `properties` maps each "
            "parameter name to its `{type, description}`, and `required` "
            "lists the parameters that must be passed."
        ),
    )


class ToolsCatalog(DataModel):
    """Catalog of tools bound to the sandbox."""

    tools: list[ToolSpec] = Field(
        default=[],
        description=(
            "Tools callable inside the sandbox as global async functions. "
            "Every tool returns a dict, a tool wrapping `async def f(x) "
            "-> int` yields `{'result': <value>}`; a tool already returning "
            "a dict yields that dict directly. Call with `await` inside "
            "`async def main(): ...` and drive with `asyncio.run(main())`."
        ),
    )


def _build_tools_catalog(tools: dict):
    """Build a ``ToolsCatalog`` from a ``{name: Tool}`` mapping.

    Returns ``None`` when ``tools`` is empty (the agent's trajectory
    skips the concat in that case).
    """
    if not tools:
        return None
    return ToolsCatalog(
        tools=[
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                parameters={
                    "properties": tool._params_schema,
                    "required": tool._required_params,
                },
            )
            for tool in tools.values()
        ]
    )


def _build_submit_tool(schema, holder: dict, tool_name: str = "submit"):
    """Build a per-call ``Tool`` that captures the final payload and
    signals end-of-run.

    ``submit`` is always exposed to the sandbox, it's the canonical
    termination signal for the agent. When the LM calls
    ``submit(result={...})`` inside a snippet, the payload lands in
    ``holder["value"]`` and the agent stops iterating.

    If ``schema`` is provided, the ``result`` parameter's JSON schema is
    overridden with it so the LM discovers the expected shape directly
    in ``ToolsCatalog``; the agent then validates the payload against
    the same schema and feeds validation errors back as retry
    observations. If ``schema`` is ``None`` (schemaless mode) the
    payload is accepted as-is and appended to the trajectory.
    """

    async def submit(result: dict) -> dict:
        """Submit the final answer and end the run.

        Args:
            result (dict): The final payload. With a target schema, ``result``
                must match it (validation errors come back as an observation on
                the next turn). Schemaless, pass ``{"answer": "..."}`` — the
                ``answer`` string becomes the content of the final assistant
                message.
        """
        holder["value"] = dict(result)
        return {"submitted": dict(result)}

    tool = Tool(submit, name=tool_name)
    if schema:
        tool._params_schema["result"] = {
            **schema,
            "title": "Result",
            "description": (
                "The final answer. Must match the target output schema: "
                "all required fields present with matching types."
            ),
        }
    else:
        tool._params_schema["result"] = {
            "type": "object",
            "title": "Result",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer message.",
                }
            },
            "required": ["answer"],
            "description": (
                'The final answer, as `{"answer": "..."}`. The `answer` string '
                "becomes the content of the final assistant message appended "
                "to the trajectory."
            ),
        }
    return tool


def _build_llm_query_tool(sub_language_model, max_llm_calls, counter, lock):
    """Build a fresh ``llm_query`` Tool bound to a per-call quota counter."""

    async def llm_query(prompt: str) -> dict:
        """Query the sub-language-model with a single prompt.

        Args:
            prompt (str): The prompt sent to the sub-LM. Pass only the
                snippet you actually need analyzed, the sub-LM has its
                own context budget.
        """
        async with lock:
            if counter["value"] >= max_llm_calls:
                return {
                    "result": "",
                    "error": (
                        f"sub-LM call budget exhausted "
                        f"({counter['value']}/{max_llm_calls}). "
                        "Aggregate remaining work in Python instead."
                    ),
                }
            counter["value"] += 1

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content=prompt)]
        )
        response = await sub_language_model(messages)
        text = response.get("content", "") if response else ""
        return {"result": text}

    return Tool(llm_query, name="llm_query")


def _build_llm_query_batched_tool(sub_language_model, max_llm_calls, counter, lock):
    """Build a fresh ``llm_query_batched`` Tool sharing the quota counter."""

    async def llm_query_batched(prompts: list[str]) -> dict:
        """Query the sub-language-model with multiple prompts concurrently.

        Args:
            prompts (list[str]): The prompts to dispatch in parallel.
                Strongly preferred over a Python loop of `llm_query`
                calls when the queries are independent.

        Returns:
            dict: ``{"result": [...]}`` where each element is the
                sub-LM response text, or, for prompts that failed —
                an error string prefixed with
                ``"[error] <ExceptionType>: <message>"``. Order matches
                the input prompts. Inspect each entry before aggregating
                so error strings aren't silently treated as data. When
                the call would exceed the shared budget, the helper
                short-circuits with ``{"result": [], "error": "..."}``
                and does not consume any quota.
        """
        async with lock:
            n = len(prompts)
            remaining = max_llm_calls - counter["value"]
            if n > remaining:
                return {
                    "result": [],
                    "error": (
                        f"sub-LM call budget would be exceeded: "
                        f"{counter['value']} + {n} > {max_llm_calls}. "
                        "Aggregate remaining work in Python instead."
                    ),
                }
            counter["value"] += n

        async def _query_one(prompt):
            try:
                messages = ChatMessages(
                    messages=[ChatMessage(role=ChatRole.USER, content=prompt)]
                )
                response = await sub_language_model(messages)
                return response.get("content", "") if response else ""
            except Exception as e:
                return f"[error] {type(e).__name__}: {e}"

        results = await asyncio.gather(*(_query_one(p) for p in prompts))
        return {"result": results}

    return Tool(llm_query_batched, name="llm_query_batched")


def _adapt_tool_for_sandbox(tool):
    """Route a sandbox tool call through the `Tool` Module (preserving
    observability/retry) and return a plain dict Monty can marshal back.
    """

    async def adapter(**kwargs):
        result = await tool(**kwargs)
        if result is None:
            return None
        if hasattr(result, "get_json"):
            return result.get_json()
        return result

    return adapter


@synalinks_export(
    [
        "synalinks.modules.RecursiveLanguageModelAgent",
        "synalinks.RecursiveLanguageModelAgent",
        "synalinks.modules.RLM",
        "synalinks.RLM",
    ]
)
class RecursiveLanguageModelAgent(Module):
    """A recursive-language-model agent.

    Each turn the LM calls a single tool,
    ``run_python_code(code: str)``, which runs the snippet in a
    persistent `Monty <https://github.com/pydantic/monty>`_ REPL sandbox and
    returns ``{"stdout", "stderr", "error"}``. State (variables, imports,
    function definitions) accumulates across turns so the agent can build up
    intermediate values, probe data, and iterate. ``submit``, the recursive
    helpers and any user tools are **not** exposed to the LM as tools — they
    live *inside* the sandbox as async callables (advertised through the
    tools catalog), reachable only from the code passed to
    ``run_python_code``.

    When ``recursive=True`` (the default), two extra helpers are exposed
    inside the sandbox: ``llm_query(prompt)`` and
    ``llm_query_batched(prompts)``. The agent then treats long inputs as
    an *external environment*, it writes Python that slices, filters, and
    aggregates the data, and recursively delegates semantic work to a
    sub-LM only on the snippets it cares about. Compared to feeding a
    long document straight into the primary LM, this trades a single
    huge context for many small ones, which both fits inside provider
    limits and reduces the chance of long-context regressions.

    When ``recursive=False``, the agent runs without the sub-LM helpers,
    useful when the task is purely computational and recursion would
    only add cost.

    Bound user tools (if any) appear inside the sandbox as global
    **async** callables; scripts must ``await`` them inside an
    ``async def`` and drive with ``asyncio.run(...)``.

    Termination: the snippet calls the in-sandbox ``submit(result=...)``
    callable with the final payload. If ``max_iterations`` is reached
    without ``submit``, a final inference step formats the accumulated
    trajectory into the target ``schema`` / ``data_model``. Empty snippets
    are not termination signals, the loop feeds back a reminder and keeps
    going.

    The ``llm_query`` quota is per-call: every invocation of this agent
    gets a fresh budget of ``max_llm_calls`` sub-LM queries, and
    concurrent invocations of the *same* agent instance each get an
    independent budget — the counter and lock are built inside
    ``call()`` and never shared across runs.

    Example:

    ```python
    import synalinks
    import asyncio

    class Doc(synalinks.DataModel):
        text: str

    class Answer(synalinks.DataModel):
        answer: str

    async def main():
        primary = synalinks.LanguageModel(model="openai/gpt-4o")
        cheap = synalinks.LanguageModel(model="openai/gpt-4o-mini")
        inputs = synalinks.Input(data_model=Doc)
        outputs = await synalinks.RLM(
            data_model=Answer,
            language_model=primary,
            sub_language_model=cheap,
            max_iterations=8,
            max_llm_calls=20,
        )(inputs)
        agent = synalinks.Program(inputs=inputs, outputs=outputs)
        long_text = open("book.txt").read()
        result = await agent(Doc(text=long_text))
        print(result.prettify_json())

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    References:
        - [Recursive Language Models](https://arxiv.org/abs/2512.24601)

    Args:
        schema (dict): Optional. The target JSON schema for the final
            structured answer. If not provided, use ``data_model`` to
            infer it. When both are omitted, the agent runs in
            **schemaless** mode, the final generator emits a
            ``ChatMessage`` that is appended to the trajectory, and
            ``call`` returns the ``ChatMessages`` trajectory directly.
        data_model (DataModel | SymbolicDataModel | JsonDataModel):
            Optional. The target data model for the final answer.
        language_model (LanguageModel): The language model driving the
            per-turn code generator and the final-formatting step.
        sub_language_model (LanguageModel): Optional. The language
            model used by ``llm_query`` and ``llm_query_batched`` when
            ``recursive=True``. Defaults to ``language_model``, pass a
            cheaper / smaller model here when the recursive sub-queries
            don't need the primary LM's full capability. Ignored when
            ``recursive=False``.
        recursive (bool): Optional. If ``True`` (default), expose
            ``llm_query`` and ``llm_query_batched`` inside the sandbox
            and use the recursive instructions. If ``False``, run
            without the sub-LM helpers.
        tools (list): Optional. Extra :class:`Tool` instances exposed to
            the sandbox in addition to ``submit`` (and ``llm_query`` /
            ``llm_query_batched`` when ``recursive=True``). The names
            ``submit``, ``llm_query``, and ``llm_query_batched`` are
            always reserved at construction time, even when
            ``recursive=False``, so tool naming stays stable across the
            two modes.

            **Naming gotcha**: each tool is registered under
            ``tool.name == tool._func.__name__``. ``Tool(_my_helper)``
            shows up inside the script as ``_my_helper``. Rename the
            function rather than relying on an alias.
        prompt_template (str): Optional. Prompt template forwarded to
            the per-turn code generator.
        examples (list): Optional. Examples forwarded to the per-turn
            code generator.
        instructions (str): Optional. Instructions for the per-turn
            code generator. Defaults to either
            :func:`get_recursive_instructions` (when ``recursive=True``,
            with the ``{max_llm_calls}`` placeholder substituted) or
            :func:`get_default_instructions` otherwise.
        final_instructions (str): Optional. Instructions for the final
            answer generator. Defaults to ``instructions``.
        temperature (float): Optional. Sampling temperature
            (Default 0.0).
        use_inputs_schema (bool): Optional. Feed the input schema to
            the generator prompt (Default False).
        use_outputs_schema (bool): Optional. Feed the output schema to
            the generator prompt (Default False).
        reasoning_effort (str): Optional. One of ``'minimal'``,
            ``'low'``, ``'medium'``, ``'high'``, ``'disable'``,
            ``'none'``, ``None``. Default ``None``.
        use_chain_of_thought (bool): Optional. Wrap the per-turn
            generator in ChainOfThought so it emits a ``thinking``
            field alongside the tool call. Default ``False``.
        autonomous (bool): Optional. If ``True`` (default), run the
            full code/execute/observe loop until the LM calls
            ``submit`` or ``max_iterations`` is reached, then produce a
            structured final answer. If ``False``, require a
            ``ChatMessages`` input and execute a single code turn per
            call, returning the updated trajectory, suitable for
            human-in-the-loop use. For cross-call REPL state in
            interactive mode, hand a ``Sandbox`` to ``call`` via the
            ``sandbox`` kwarg; the agent itself stays stateless.
        timeout (int): Per-turn execution budget in seconds
            (Default 60). Recursive sub-LM calls dominate per-turn wall
            time; ``llm_query_batched`` of even a handful of prompts
            can take several seconds. Snippets that exceed the budget
            turn into an observation so the LM can recover on the next
            turn.
        max_iterations (int): Maximum number of code-execution turns
            before forcing the final answer step (Default 20).
        max_llm_calls (int): Hard cap on sub-LM calls per agent
            invocation, shared between ``llm_query`` and
            ``llm_query_batched`` (Default 50). Once the budget is
            spent, further calls return an error string instead of a
            response so the LM can fall back to code-side aggregation.
            Ignored when ``recursive=False``.
        max_output_chars (int): Maximum characters to include from
            REPL output in the per-turn observation (Default 10_000).
            Anything beyond is truncated with a
            ``… (truncated, N chars omitted)`` marker so a single
            noisy turn cannot blow up the trajectory.
        return_inputs_with_trajectory (bool): Optional. Whether to
            return the full trajectory alongside the final answer
            (Default ``True``).
        sandbox (Sandbox): Optional. A pre-built ``Sandbox`` instance to
            reuse across calls. When supplied, the agent will not build
            its own sandbox at ``call()`` time and ``sandbox_type`` is
            derived from ``type(sandbox)``. Pass this when the caller
            owns the sandbox lifecycle (e.g. interactive sessions where
            REPL state must persist across calls). When omitted, a
            fresh sandbox of ``sandbox_type`` is built per call.
        sandbox_type (type): Optional. The ``Sandbox`` subclass to
            instantiate when no sandbox is supplied (here or to
            ``call()``). Defaults to ``MontySandbox``, or to
            ``type(sandbox)`` when ``sandbox`` is given. Any ``Sandbox``
            subclass whose ``__init__`` accepts
            ``(timeout=..., name=...)`` works; register custom
            subclasses with ``@register_synalinks_serializable`` so
            they round-trip through ``get_config`` / ``from_config``.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
    """

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        language_model=None,
        sub_language_model=None,
        recursive=True,
        tools=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        final_instructions=None,
        temperature=0.0,
        use_inputs_schema=False,
        use_outputs_schema=False,
        reasoning_effort=None,
        use_chain_of_thought=False,
        autonomous=True,
        timeout=60,
        max_iterations=20,
        max_llm_calls=50,
        max_output_chars=10_000,
        return_inputs_with_trajectory=True,
        sandbox=None,
        sandbox_type=None,
        name=None,
        description=None,
    ):
        super().__init__(name=name, description=description)

        if not schema and data_model:
            schema = data_model.get_schema()
        # `schema` is optional, when omitted, the agent operates in
        # "schemaless" mode and returns a ChatMessages trajectory (with
        # a final assistant message appended) instead of a typed answer.
        self.schema = schema

        self.language_model = _get_lm(language_model)
        # `sub_language_model` defaults to the primary LM when omitted.
        # ``get(None)`` would raise, so resolve only when a value is given.
        self.sub_language_model = (
            _get_lm(sub_language_model)
            if sub_language_model is not None
            else self.language_model
        )
        self.recursive = recursive
        self.prompt_template = prompt_template
        self.examples = examples

        if not instructions:
            if recursive:
                instructions = get_recursive_instructions().replace(
                    "{max_llm_calls}",
                    str(max_llm_calls),
                )
            else:
                instructions = get_default_instructions()
        self.instructions = instructions
        self.final_instructions = final_instructions or instructions

        # Sandbox handling: if a concrete sandbox is supplied at
        # construction, reuse it across calls and derive sandbox_type
        # from its class. Otherwise fall back to sandbox_type (default
        # MontySandbox) and build one fresh per `call()`. Set early so
        # the sandbox-specific prompt text can be composed into the
        # code generator's instructions below.
        self.sandbox = sandbox
        if sandbox is not None:
            self.sandbox_type = type(sandbox)
        else:
            self.sandbox_type = sandbox_type or MontySandbox
        sandbox_description = self.sandbox_type.description
        if sandbox_description:
            self.instructions = self.instructions + "\n\n" + sandbox_description

        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.reasoning_effort = reasoning_effort
        self.use_chain_of_thought = use_chain_of_thought
        self.autonomous = autonomous

        self.timeout = timeout
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.return_inputs_with_trajectory = return_inputs_with_trajectory

        reserved = self._reserved_tool_names()
        self.tools = {}
        if tools:
            for tool in tools:
                if tool.name.startswith("_"):
                    raise ValueError(
                        f"Tool name {tool.name!r} starts with an underscore. "
                        f"Tools exposed to the LM must have public names — "
                        f"rename the function or pass an explicit `name=` "
                        f"to Tool(...)."
                    )
                if tool.name in reserved:
                    raise ValueError(
                        f"Tool name '{tool.name}' is reserved by {type(self).__name__}."
                    )
                self.tools[tool.name] = tool

        self.tools_catalog = _build_tools_catalog(self.tools)

        # The per-turn generator is schemaless: the loop calls it with
        # `tools=[run_python_code]`, so each snippet arrives as a native
        # tool call. The `run_python_code` tool is per-call (it closes
        # over the sandbox) and passed at call time.
        generator_cls = ChainOfThought if use_chain_of_thought else Generator
        self.code_generator = generator_cls(
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            reasoning_effort=self.reasoning_effort,
            language_model=self.language_model,
            name="code_generator_" + self.name,
        )

        self.final_generator = Generator(
            schema=self.schema,
            language_model=self.language_model,
            instructions=self.final_instructions,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            return_inputs=False,
            name="final_generator_" + self.name,
        )

    def _reserved_tool_names(self) -> frozenset:
        """Names a user tool cannot collide with at construction time.

        ``submit``, ``llm_query``, and ``llm_query_batched`` are always
        reserved, even when ``recursive=False``, so that tool naming
        stays stable across the two modes and a user tool can't quietly
        shadow a helper name that would reappear if ``recursive`` is
        flipped back on.
        """
        return frozenset({"submit", "llm_query", "llm_query_batched"})

    def _build_extra_call_tools(self) -> dict:
        """Build the per-call recursive helpers when ``recursive=True``.

        A fresh counter+lock is built on every invocation so concurrent
        calls into the same agent instance get independent budgets.
        Returns ``{}`` when ``recursive=False``.
        """
        if not self.recursive:
            return {}
        counter = {"value": 0}
        lock = asyncio.Lock()
        return {
            "llm_query": _build_llm_query_tool(
                self.sub_language_model,
                self.max_llm_calls,
                counter,
                lock,
            ),
            "llm_query_batched": _build_llm_query_batched_tool(
                self.sub_language_model,
                self.max_llm_calls,
                counter,
                lock,
            ),
        }

    def _build_run_python_code_tool(self, sandbox):
        """Build the lone tool the LM can call.

        ``run_python_code`` is the *only* tool exposed to the LM. It wraps
        the sandbox's :meth:`MontySandbox.run_python_code` (which runs the
        snippet in the persistent sandbox) and clips the captured streams
        to ``max_output_chars``. The tools (``submit``, ``llm_query`` and
        the user tools) and the ``inputs`` payload are not passed here —
        :meth:`call` binds them onto the sandbox before each run (via
        ``bind_functions`` and a persisted ``inputs`` variable). The
        sandbox is closed over, so a fresh tool is built per call. For
        ``compute_output_spec`` the closure is built with ``sandbox=None``
        and never executed; only its signature/docstring drive the schema.
        """
        max_output_chars = self.max_output_chars

        def _clip(text):
            if not text:
                return ""
            text = text.rstrip()
            if max_output_chars is not None and len(text) > max_output_chars:
                omitted = len(text) - max_output_chars
                return text[:max_output_chars] + (
                    f"\n… (truncated, {omitted} chars omitted)"
                )
            return text

        async def run_python_code(code: str) -> dict:
            """Execute one async Python snippet in the persistent sandbox.

            State persists across calls — variables, imports and function
            definitions stay defined. The user input is bound as a dict named
            `inputs`; read full values via `inputs[field]`. Other tools
            (`submit`, `llm_query`, ...) are pre-imported async callables —
            call them inside `async def main(): ...` and drive with
            `asyncio.run(main())`. Call `submit(result={...})` to end the run.

            Args:
                code (str): The Python snippet to execute in the
                    sandbox.
            """
            result = await sandbox.run_python_code(code)
            observation = {"stdout": _clip(result.get("stdout", ""))}
            stderr = _clip(result.get("stderr", ""))
            if stderr:
                observation["stderr"] = stderr
            if result.get("error"):
                observation["error"] = result["error"]
            return observation

        return Tool(run_python_code, name="run_python_code")

    async def call(self, inputs, training=False, sandbox=None):
        if not inputs:
            return None

        if not self.autonomous and not is_chat_messages(inputs):
            raise ValueError(
                f"In interactive mode, the {type(self).__name__} needs a "
                "ChatMessages-like data model as inputs"
            )

        # Per-call tool set: user tools plus a fresh `submit` bound to a
        # private holder, plus any per-call recursive helpers. submit is
        # the canonical termination signal, always exposed, schema'd or
        # not, and everything in this set is built fresh per call so
        # concurrent invocations don't share holders, counters, or locks.
        call_tools = dict(self.tools)
        submit_holder = {"value": None}
        call_tools["submit"] = _build_submit_tool(self.schema, submit_holder)
        call_tools.update(self._build_extra_call_tools())
        call_tools_catalog = _build_tools_catalog(call_tools)

        if is_chat_messages(inputs):
            trajectory = inputs
            inputs_json = {}
        else:
            inputs_json = inputs.get_json()
            # The LM prompt only sees a metadata summary of the inputs —
            # previews and sizes, never the full value. The sandbox gets the
            # complete `inputs_json` rebound on every `run_python_code`
            # call, so `inputs[field]` is always reachable.
            base = summarize_inputs(inputs_json)
            if call_tools_catalog is not None:
                base = await ops.concat(
                    base,
                    call_tools_catalog,
                    name="inputs_with_tools_" + self.name,
                )
            trajectory = await ops.concat(
                base,
                ChatMessages(),
                name="trajectory_" + self.name,
            )

        agent_messages = trajectory.get("messages")

        # Sandbox resolution order: per-call kwarg > constructor-supplied
        # sandbox > fresh sandbox of `sandbox_type`. The first two cases
        # let the caller (or the agent's owner) keep REPL state alive
        # across calls; the third is the stateless-per-call default.
        if sandbox is None:
            sandbox = self.sandbox or self.sandbox_type(timeout=self.timeout)

        # The per-turn snippet is delivered as a native `run_python_code`
        # tool call — the only tool the LM can call, wrapping the sandbox's
        # own `run_python_code`. The sandbox-side tools (submit, llm_query,
        # user tools) are NOT exposed to the LM; they live inside the
        # sandbox as async callables (advertised via the tools catalog).
        # Bind them onto the sandbox so every `run_python_code` snippet can
        # reach them.
        external_functions = {
            name: _adapt_tool_for_sandbox(t) for name, t in call_tools.items()
        }
        sandbox.bind_functions(external_functions)
        # Persist the full input payload as `inputs` in the sandbox
        # namespace. Monty's per-run `inputs=` binding does not persist, so
        # copy it into a real variable once; every snippet then reads it via
        # `inputs[field]`.
        await sandbox.run("inputs = _rlm_inputs", inputs={"_rlm_inputs": inputs_json})
        run_tool = self._build_run_python_code_tool(sandbox)

        iterations = self.max_iterations if self.autonomous else 1
        submitted_final = None

        for _ in range(iterations):
            tool_calls = await self.code_generator(trajectory, tools=[run_tool])
            if not tool_calls:
                break
            # The generator returns an assistant ChatMessage with native
            # `tool_calls` ({id, name, arguments}); append it as-is.
            agent_messages.append(tool_calls.get_json())
            native_tool_calls = tool_calls.get("tool_calls") or []
            if not native_tool_calls:
                # No tool call this turn — nudge toward run_python_code /
                # submit and spend another iteration.
                agent_messages.append(
                    ChatMessage(
                        role=ChatRole.USER,
                        content=(
                            "(no tool call) Call `run_python_code` with a "
                            "snippet, and call `submit(result={...})` inside "
                            "it to terminate the run."
                        ),
                    ).get_json()
                )
                trajectory.update({"messages": agent_messages})
                continue

            for tool_call in native_tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments") or {}
                tool_call_id = tool_call.get("id")
                if tool_name != "run_python_code":
                    content = {
                        "error": (
                            f"Unknown tool '{tool_name}'. The only callable "
                            "tool is 'run_python_code'; everything else is a "
                            "sandbox function reachable from your snippet."
                        )
                    }
                else:
                    try:
                        result = await run_tool(**tool_args)
                        content = (
                            result.get_json() if hasattr(result, "get_json") else result
                        )
                    except Exception as e:
                        content = {"error": f"{type(e).__name__}: {e}"}
                    # submit() inside the snippet wrote to the holder. Clear it
                    # either way so a stale payload can't short-circuit a later
                    # retry.
                    submitted = submit_holder["value"]
                    submit_holder["value"] = None
                    if submitted is not None:
                        content = dict(content)
                        if self.schema:
                            try:
                                jsonschema.validate(submitted, self.schema)
                            except ValidationError as ve:
                                content["submit"] = (
                                    f"validation failed: {ve.message}. "
                                    "Revise the payload and call submit again."
                                )
                            else:
                                submitted_final = submitted
                                content["submit"] = "accepted"
                        else:
                            submitted_final = submitted
                            content["submit"] = "accepted"
                agent_messages.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        tool_call_id=tool_call_id,
                        content=content,
                    ).get_json()
                )
                if submitted_final is not None:
                    break
            trajectory.update({"messages": agent_messages})
            if submitted_final is not None:
                break

        # Interactive mode: only invoke the final generator when the LM itself
        # signalled completion via submit. Otherwise return the updated
        # trajectory so the caller can decide when to continue.
        if not self.autonomous and submitted_final is None:
            validated_messages = ChatMessages(
                messages=[ChatMessage(**msg) for msg in agent_messages]
            )
            return JsonDataModel(
                json=validated_messages.get_json(),
                schema=ChatMessages.get_schema(),
                name=self.name,
            )

        # submit short-circuit: the LM already produced the final payload
        # inside the sandbox, so we skip the final-formatting LM call.
        # Schemaless mode treats the payload as the content of a final
        # assistant ChatMessage appended to the trajectory.
        if submitted_final is not None:
            if self.schema:
                final_result = JsonDataModel(
                    json=submitted_final,
                    schema=self.schema,
                    name="final_generator_" + self.name,
                )
            else:
                # Schemaless: the answer is submitted as {"answer": "..."}; use
                # that string as the final assistant message content (falling
                # back to the whole payload if `answer` is absent).
                agent_messages.append(
                    ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=submitted_final.get("answer", submitted_final),
                    ).get_json()
                )
                validated_messages = ChatMessages(
                    messages=[ChatMessage(**msg) for msg in agent_messages]
                )
                return JsonDataModel(
                    json=validated_messages.get_json(),
                    schema=ChatMessages.get_schema(),
                    name=self.name,
                )
        else:
            final_result = await self.final_generator(trajectory)
            if not self.schema:
                # Schemaless fallback: the final generator emits a
                # ChatMessage. Append it to the trajectory and return.
                if final_result:
                    agent_messages.append(final_result.get_json())
                validated_messages = ChatMessages(
                    messages=[ChatMessage(**msg) for msg in agent_messages]
                )
                return JsonDataModel(
                    json=validated_messages.get_json(),
                    schema=ChatMessages.get_schema(),
                    name=self.name,
                )

        if self.return_inputs_with_trajectory:
            validated_messages = ChatMessages(
                messages=[ChatMessage(**msg) for msg in agent_messages]
            )
            return await ops.concat(
                JsonDataModel(
                    json=validated_messages.get_json(),
                    schema=ChatMessages.get_schema(),
                    name=self.name,
                ),
                final_result,
                name=self.name,
            )
        return final_result

    async def compute_output_spec(self, inputs, training=False, sandbox=None):
        if not self.autonomous and not is_chat_messages(inputs):
            raise ValueError(
                f"In interactive mode, the {type(self).__name__} needs a "
                "ChatMessages-like data model as inputs"
            )
        # Mirror the runtime: the code generator sees a summary of the
        # input plus the tool catalog, not the raw input DataModel.
        if is_chat_messages(inputs):
            generator_inputs = inputs
        else:
            generator_inputs = SymbolicDataModel(
                schema=InputsSummary.get_schema(),
                name="inputs_summary_" + self.name,
            )
            if self.tools_catalog is not None:
                generator_inputs = await ops.concat(
                    generator_inputs,
                    self.tools_catalog,
                    name="inputs_with_tools_" + self.name,
                )
        # The closure placeholders are never executed during spec tracing;
        # only the tool's signature/docstring shape the prompt.
        spec_tool = self._build_run_python_code_tool(None)
        _ = await self.code_generator(generator_inputs, tools=[spec_tool])
        if not self.autonomous:
            # Interactive mode: the common case is returning the trajectory.
            # When the LM emits empty code the runtime returns the final
            # answer instead; we pick the common-case spec.
            return SymbolicDataModel(
                schema=ChatMessages.get_schema(),
                name=self.name,
            )
        if not self.schema:
            # Schemaless autonomous: final generator produces a ChatMessage
            # appended to the trajectory; output spec is ChatMessages.
            _ = await self.final_generator(inputs)
            return SymbolicDataModel(
                schema=ChatMessages.get_schema(),
                name=self.name,
            )
        if self.return_inputs_with_trajectory:
            return await ops.logical_and(
                SymbolicDataModel(
                    schema=ChatMessages.get_schema(),
                    name=self.name,
                ),
                SymbolicDataModel(
                    schema=self.schema,
                    name="final_generator_" + self.name,
                ),
                name=self.name,
            )
        return await self.final_generator(inputs)

    def get_config(self):
        config = {
            "schema": self.schema,
            "recursive": self.recursive,
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
            "timeout": self.timeout,
            "max_iterations": self.max_iterations,
            "max_llm_calls": self.max_llm_calls,
            "max_output_chars": self.max_output_chars,
            "return_inputs_with_trajectory": self.return_inputs_with_trajectory,
            "sandbox_type": get_registered_name(self.sandbox_type),
            "name": self.name,
            "description": self.description,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            ),
            "sub_language_model": serialization_lib.serialize_synalinks_object(
                self.sub_language_model,
            ),
        }
        sandbox_config = {
            "sandbox": (
                serialization_lib.serialize_synalinks_object(self.sandbox)
                if self.sandbox is not None
                else None
            )
        }
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(tool)
                for tool in self.tools.values()
            ]
        }
        return {**config, **language_model_config, **sandbox_config, **tools_config}

    @classmethod
    def from_config(cls, config):
        tools = [
            serialization_lib.deserialize_synalinks_object(tool)
            for tool in config.pop("tools", [])
        ]
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        sub_language_model = None
        if "sub_language_model" in config:
            sub_language_model = serialization_lib.deserialize_synalinks_object(
                config.pop("sub_language_model")
            )
        sandbox = None
        if "sandbox" in config:
            sandbox_serialized = config.pop("sandbox")
            if sandbox_serialized is not None:
                sandbox = serialization_lib.deserialize_synalinks_object(
                    sandbox_serialized
                )
        sandbox_type_name = config.pop("sandbox_type", None)
        sandbox_type = (
            get_registered_object(sandbox_type_name) if sandbox_type_name else None
        )
        return cls(
            language_model=language_model,
            sub_language_model=sub_language_model,
            tools=tools or None,
            sandbox=sandbox,
            sandbox_type=sandbox_type,
            **config,
        )
