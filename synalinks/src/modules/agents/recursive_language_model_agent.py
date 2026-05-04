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
    """Default instructions for non-recursive code-mode reasoning."""
    return """
You solve the user task by writing and executing Python snippets inside a
persistent sandbox. Each turn you emit ONE snippet via the `python_code`
field; state persists across turns (variables stay defined, imports stay
loaded).

IMPORTANT: This is ITERATIVE. Each code block you write will execute,
you'll see the output, then you decide what to do next. Do NOT try to
solve everything in one step.

The user input is bound as a dict named `inputs` in the sandbox, the
full, untruncated value. In the prompt you only see an `InputsSummary`
with previews and sizes; always read the real values through
`inputs[field]` inside your code, never re-type them from the preview.

Each turn carries an `IterationInfo.iteration` field like `3/5`, that's
your progress out of the hard iteration cap. Budget accordingly: early
turns can explore, later ones should converge. If few turns remain,
batch work into a single snippet instead of spreading it out.

Use `print(...)` to log intermediate observations. Any tools bound to the
agent are exposed as async callables, call them inside `async def main():`
and drive with `asyncio.run(main())`; calling them without `await` returns
a coroutine object, not the value.

Termination: call the `submit` tool (always present in the tools
catalog) with `result={...}` matching its `result` parameter schema.
`submit` is async, so drive it inside `async def main(): ...` with
`asyncio.run(main())` like any other tool. It captures the answer
and ends the run in one step. If the payload fails schema validation
you'll see the validation error on the next turn and can retry.

`submit` is the only termination path, emitting an empty `python_code`
string is treated as a no-op and you'll be reminded to call `submit`.
Don't run out of iterations without calling it.
""".strip()


def get_recursive_instructions():
    """Default instructions for recursive (sub-LM) code-mode reasoning.

    The ``{max_llm_calls}`` placeholder is substituted at construction time.
    """
    return """
You solve the user task by writing Python that programmatically explores
the inputs and recursively delegates semantic work to a sub-LM. Each
turn you emit ONE snippet via the `python_code` field; state persists
across turns (variables stay defined, imports stay loaded).

IMPORTANT: This is ITERATIVE. Each code block you write will execute,
you'll see the output, then you decide what to do next. Do NOT try to
solve everything in one step.

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

Each turn carries an `IterationInfo.iteration` field like `3/5` —
your progress against the iteration cap. Early turns can explore;
later ones should converge. When few turns remain, batch work into
fewer snippets.

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

Termination: call the always-present `submit` tool with `result={...}`
matching its `result` parameter schema. `submit` is the only
termination path, empty `python_code` strings are no-ops and you'll
be reminded to call `submit`. Don't run out of iterations without
calling it.
""".strip()


class CodeStep(DataModel):
    """One turn of code-mode reasoning: a Python snippet to execute next."""

    python_code: str = Field(
        description=(
            "A Python snippet to execute in the persistent sandbox. The user "
            "input is bound as a dict named `inputs`. State persists across "
            "turns, so variables, functions and imports stay defined. Call the "
            "`submit` tool to terminate the run with the final answer"
        )
    )


class ToolSpec(DataModel):
    """Description of one tool exposed in the code-mode sandbox."""

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
    """Catalog of tools bound to the code-mode sandbox."""

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


class InputsSummary(DataModel):
    """Metadata-only view of the user input bound as ``inputs`` in the sandbox.

    Only per-field previews and sizes are surfaced here to keep the prompt
    small when the input contains long documents or large collections. Read
    the **full** values through ``inputs[field_name]`` inside your code —
    the sandbox namespace holds the untruncated data.
    """

    fields: list[dict] = Field(
        default=[],
        description=(
            "One entry per top-level input field, each with `name`, `type`, "
            "`size` (len of string/list/dict, else null), `preview`, and "
            "`truncated` (true when preview omits part of the value). Read "
            "the complete value from the sandbox via `inputs[name]`."
        ),
    )


class IterationInfo(DataModel):
    """Budget info visible to the code generator on each turn."""

    iteration: str = Field(
        description=(
            "Current turn position as `<current>/<max>`. Budget your "
            "remaining turns accordingly, batch work into fewer snippets "
            "when turns are running out."
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
            result (dict): The final payload. When a target schema is
                configured, ``result`` must match it, validation errors
                come back as an observation on the next turn.
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
            "description": (
                "The final answer as a free-form dict. No schema is "
                "configured, contents are appended to the trajectory "
                "as an assistant ChatMessage."
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


def _summarize_inputs(
    inputs_json,
    preview_chars: int = 200,
    preview_items: int = 5,
) -> InputsSummary:
    """Build a compact ``InputsSummary`` from a raw input JSON dict.

    Small values are previewed in full. Long strings, lists and dicts show
    only a head, the full value remains accessible inside the sandbox at
    ``inputs[name]``.
    """
    import json as _json

    fields = []
    for name, value in inputs_json.items():
        type_name = type(value).__name__
        size = None
        preview = None
        truncated = False

        if isinstance(value, str):
            size = len(value)
            if size > preview_chars:
                preview = value[:preview_chars]
                truncated = True
            else:
                preview = value
        elif isinstance(value, list):
            size = len(value)
            head = value[:preview_items]
            truncated = size > preview_items
            preview = _json.dumps(head)
            if len(preview) > preview_chars:
                preview = preview[:preview_chars] + "…"
                truncated = True
        elif isinstance(value, dict):
            size = len(value)
            head = dict(list(value.items())[:preview_items])
            truncated = size > preview_items
            preview = _json.dumps(head)
            if len(preview) > preview_chars:
                preview = preview[:preview_chars] + "…"
                truncated = True
        else:
            preview = _json.dumps(value)

        fields.append(
            {
                "name": name,
                "type": type_name,
                "size": size,
                "preview": preview,
                "truncated": truncated,
            }
        )

    return InputsSummary(fields=fields)


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


def _format_observation(stdout, stderr, result, error, max_output_chars=None):
    parts = []
    if stdout:
        parts.append(f"stdout:\n{stdout.rstrip()}")
    if stderr:
        parts.append(f"stderr:\n{stderr.rstrip()}")
    if result is not None:
        parts.append(f"result: {result!r}")
    if error:
        parts.append(f"error: {error}")
    observation = "\n".join(parts) or "(no output)"
    if max_output_chars is not None and len(observation) > max_output_chars:
        omitted = len(observation) - max_output_chars
        observation = (
            observation[:max_output_chars] + f"\n… (truncated, {omitted} chars omitted)"
        )
    return observation


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

    A code-mode reasoning agent that emits Python snippets each turn
    and executes them in a persistent
    `Monty <https://github.com/pydantic/monty>`_ REPL sandbox. State
    (variables, imports, function definitions) accumulates across turns
    so the agent can build up intermediate values, probe data, and
    iterate.

    When ``recursive=True`` (the default), two extra helpers are exposed
    inside the sandbox: ``llm_query(prompt)`` and
    ``llm_query_batched(prompts)``. The agent then treats long inputs as
    an *external environment*, it writes Python that slices, filters, and
    aggregates the data, and recursively delegates semantic work to a
    sub-LM only on the snippets it cares about. Compared to feeding a
    long document straight into the primary LM, this trades a single
    huge context for many small ones, which both fits inside provider
    limits and reduces the chance of long-context regressions.

    When ``recursive=False``, the agent runs as a plain code-mode agent
    with no sub-LM helpers, useful when the task is purely
    computational and recursion would only add cost.

    Bound user tools (if any) appear inside the sandbox as global
    **async** callables; scripts must ``await`` them inside an
    ``async def`` and drive with ``asyncio.run(...)``.

    Termination: the LM calls the always-present ``submit`` tool with
    the final payload. If ``max_iterations`` is reached without
    ``submit``, a final inference step formats the accumulated
    trajectory into the target ``schema`` / ``data_model``. Empty
    ``python_code`` snippets are not termination signals, the loop feeds
    back a reminder and keeps going.

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
            and use the recursive instructions. If ``False``, run as a
            plain code-mode agent with no sub-LM helpers.
        tools (list): Optional. Extra :class:`Tool` instances exposed to
            the sandbox in addition to ``submit`` (and ``llm_query`` /
            ``llm_query_batched`` when ``recursive=True``). The names
            ``submit``, ``llm_query``, and ``llm_query_batched`` are
            reserved when their corresponding helpers are active.

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
            field alongside ``code``. Default ``False``.
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
                if tool.name in reserved:
                    raise ValueError(
                        f"Tool name '{tool.name}' is reserved by "
                        f"{type(self).__name__}."
                    )
                self.tools[tool.name] = tool

        self.tools_catalog = _build_tools_catalog(self.tools)

        code_step_schema = CodeStep.get_schema()
        generator_cls = ChainOfThought if use_chain_of_thought else Generator
        self.code_generator = generator_cls(
            schema=code_step_schema,
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

        Computed from ``self.recursive`` so non-recursive runs free up
        ``llm_query`` / ``llm_query_batched`` for user tools.
        """
        names = {"submit"}
        if self.recursive:
            names.update({"llm_query", "llm_query_batched"})
        return frozenset(names)

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

    async def _execute_turn(
        self,
        sandbox,
        code,
        inputs_json,
        tools=None,
    ):
        """Run one code snippet in the sandbox, return a formatted observation.

        ``tools`` overrides ``self.tools`` for the duration of this call —
        used at call time to layer in per-call built-ins like ``submit``.
        """
        active_tools = tools if tools is not None else self.tools
        external_functions = (
            {name: _adapt_tool_for_sandbox(t) for name, t in active_tools.items()}
            if active_tools
            else None
        )
        # `inputs` is rebound every turn so the snippet can always read
        # `inputs[field]` regardless of what prior turns did to the name.
        run_kwargs = {"inputs": {"inputs": inputs_json}}
        if external_functions is not None:
            run_kwargs["external_functions"] = external_functions

        execution = await sandbox.run(code, **run_kwargs)
        return _format_observation(
            execution.stdout,
            execution.stderr,
            execution.result,
            execution.error,
            max_output_chars=self.max_output_chars,
        )

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
            # previews and sizes, never the full value. The sandbox gets
            # the complete `inputs_json` rebound on every turn (see
            # `_execute_turn`), so `inputs[field]` is always reachable.
            base = _summarize_inputs(inputs_json)
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

        iterations = self.max_iterations if self.autonomous else 1
        submitted_final = None

        for n in range(iterations):
            # The iteration counter is concat'd in per turn so the code
            # generator can pace itself ("2/5" => half the budget left,
            # plan accordingly).
            iteration_info = IterationInfo(
                iteration=f"{n + 1}/{iterations}",
            )
            turn_input = await ops.concat(
                trajectory,
                iteration_info,
                name=f"turn_{n}_{self.name}",
            )
            code_step = await self.code_generator(turn_input)
            if not code_step:
                break

            code = code_step.get("python_code") or ""
            thinking = code_step.get("thinking", "")

            content_parts = []
            if thinking:
                content_parts.append(thinking)
            if code.strip():
                content_parts.append(f"```python\n{code}\n```")
            assistant_message = ChatMessage(
                role=ChatRole.ASSISTANT,
                content="\n\n".join(content_parts) if content_parts else "",
            )
            agent_messages.append(assistant_message.get_json())

            # Empty code is no longer a termination signal, submit is the
            # canonical path. Feed a reminder back as an observation and
            # let the loop run another turn.
            if not code.strip():
                agent_messages.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        content=(
                            "(no code emitted) Call the `submit` tool with "
                            "the final result to terminate the run."
                        ),
                    ).get_json()
                )
                trajectory.update({"messages": agent_messages})
                continue

            observation = await self._execute_turn(
                sandbox,
                code,
                inputs_json,
                tools=call_tools,
            )

            # If submit was called, validate (when a schema is set) and
            # decide whether to end the loop. Clear the holder either way
            # so a subsequent retry isn't short-circuited by a stale
            # captured payload.
            submitted = submit_holder["value"]
            submit_holder["value"] = None
            if submitted is not None:
                if self.schema:
                    try:
                        jsonschema.validate(submitted, self.schema)
                    except ValidationError as ve:
                        observation = (
                            observation
                            + f"\nsubmit validation failed: {ve.message}. "
                            + "Revise the payload and call submit again."
                        )
                    else:
                        submitted_final = submitted
                        observation = observation + "\nsubmit accepted."
                else:
                    # Schemaless: any dict is accepted.
                    submitted_final = submitted
                    observation = observation + "\nsubmit accepted."

            agent_messages.append(
                ChatMessage(
                    role=ChatRole.TOOL,
                    content=observation,
                ).get_json()
            )
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
                agent_messages.append(
                    ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=submitted_final,
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
        # input plus the tool catalog plus an IterationInfo, not the raw
        # input DataModel.
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
        generator_inputs = await ops.concat(
            generator_inputs,
            SymbolicDataModel(
                schema=IterationInfo.get_schema(),
                name="iteration_info_" + self.name,
            ),
            name="turn_input_" + self.name,
        )
        _ = await self.code_generator(generator_inputs)
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
