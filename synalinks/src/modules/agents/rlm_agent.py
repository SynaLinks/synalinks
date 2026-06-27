# Modified from: dspy/predict/rlm.py
# Original authors: Alex L. Zhang, Tim Kraska, Omar Khattab (DSPy Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import json
from typing import List
from typing import Optional

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
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.agents.utils.agents_utils import InputsSummary
from synalinks.src.modules.agents.utils.agents_utils import summarize_inputs
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox
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
bound to the agent are functions available *inside* the sandbox (see the tools
catalog), not separate tool calls — call them directly, e.g.
`out = submit(...)`. Reach them only from the code you pass to
`run_python_code`.

A snippet looks like this (note the variable is `inputs`, plural — it is a
dict; `input` is something else):

    text = inputs["some_field"]          # read the full value via the binding
    submit(result={"answer": text[:200]})

Termination: call `submit(result={...})` from inside your snippet, with
`result` matching its schema. It captures the answer and ends the run in one
step. If the payload fails schema validation you'll see the error on the next
turn and can retry.

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

Use `print(...)` to log intermediate observations. Call sandbox tools
directly, e.g. `out = llm_query(prompt)`.

A snippet looks like this (note the variable is `inputs`, plural — it is a
dict; `input` is something else):

    text = inputs["some_field"]          # read the full value via the binding
    out = llm_query(prompt=f"... {text[:500]} ...")   # returns {"result": ...}
    submit(result={"answer": out["result"]})

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
are functions available *inside* the sandbox (advertised in the tools
catalog), not separate tool calls — call them directly and reach them only
from the code you pass to `run_python_code`. `submit` is the only termination
path; an empty snippet is a no-op and you'll be reminded to call `submit`.
Don't run out of iterations without calling it.
""".strip()


class ToolSpec(DataModel):
    """Description of one tool exposed in the sandbox."""

    name: str = Field(
        description=(
            "The function's name in the sandbox. Call it directly as "
            "`{name}(**kwargs)`."
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
            "Tools callable inside the sandbox as global functions: call them "
            "directly, `result = name(**kwargs)`. Every tool returns a dict — a "
            "tool wrapping `def f(x) -> int` yields `{'result': <value>}`; a "
            "tool already returning a dict yields that dict directly."
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
    observability/retry) and return a plain dict the sandbox can marshal back.
    """

    async def adapter(**kwargs):
        result = await tool(**kwargs)
        if result is None:
            return None
        if hasattr(result, "get_json"):
            return result.get_json()
        return result

    return adapter


def get_subagent_tools_guidance() -> str:
    """Guidance appended to the instructions when subagents are enabled."""
    return """
Besides `run_python_code`, you can delegate to parallel subagents, each on an
isolated *fork* of the sandbox that inherits your current REPL state
(variables, functions, imports) AND files:
- `spawn_subagents(tasks)`: launch one subagent per task string. Each runs
  concurrently on its own fork; its REPL/file changes stay on that fork and do
  NOT affect you. Returns a `handle`, the subagent's `result`, and a `patch`
  (its file changes as a git-style unified diff — the actual line-level edits)
  per subagent. Call it as a top-level tool (not from inside a snippet).
- `merge_subagent(handle, paths=None, force=False, adopt_repl=False)`: fold a
  subagent's file changes into your sandbox (paths/force as for files).
  `adopt_repl=True` ALSO adopts that subagent's whole Python namespace
  (variables/functions/imports) — all-or-nothing, and only one subagent's REPL
  can be adopted per batch (a second would overwrite the first).
- `discard_subagent(handle)`: drop a subagent's fork unmerged.
Nothing a subagent does affects your sandbox until you `merge_subagent` it.
""".strip()


def get_subagent_instructions() -> str:
    """Instructions for a spawned RLM subagent (depth >= 1)."""
    return """
You are a subagent working on a private fork of a persistent Python sandbox
that inherited the parent's variables, functions, imports and files. Solve your
task by calling `run_python_code(code=...)` iteratively, reading `inputs` and
using the inherited state as needed, and call `submit(result={"answer": "..."})`
from inside a snippet to finish. Your REPL and file changes stay on your fork;
the parent reviews and decides whether to keep them, so do the work your task
requires and report concisely what you computed and changed.
""".strip()


def _subagent_answer(output) -> str:
    """Extract a subagent's final answer text from its (schemaless) output."""
    if output is None:
        return ""
    data = output.get_json() if hasattr(output, "get_json") else output
    if isinstance(data, dict):
        messages = data.get("messages")
        if messages:
            last = messages[-1]
            content = last.get("content") if isinstance(last, dict) else None
            if isinstance(content, str):
                return content
            return "" if content is None else json.dumps(content, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)
    return str(data)


@synalinks_export(
    [
        "synalinks.modules.RecursiveLanguageModelAgent",
        "synalinks.RecursiveLanguageModelAgent",
        "synalinks.modules.RLM",
        "synalinks.RLM",
    ]
)
class RecursiveLanguageModelAgent(FunctionCallingAgent):
    """A recursive-language-model agent.

    Each turn the LM calls a single tool,
    ``run_python_code(code: str)``, which runs the snippet in a
    persistent REPL sandbox (by default a ``MirageSandbox``) and
    returns ``{"stdout", "stderr", "error"}``. State (variables, imports,
    function definitions) accumulates across turns so the agent can build up
    intermediate values, probe data, and iterate. ``submit``, the recursive
    helpers and any user tools are **not** exposed to the LM as tools — they
    live *inside* the sandbox as plain synchronous functions (advertised
    through the tools catalog), reachable only from the code passed to
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

    Bound user tools (if any) appear inside the sandbox as global functions;
    scripts call them directly, ``result = tool_name(...)``.

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
            ``recursive=True``, and by spawned subagents. Defaults to
            ``language_model``, pass a cheaper / smaller model here when
            the recursive sub-queries don't need the primary LM's full
            capability. Ignored when ``recursive=False``.
        prompt_template (str): Optional. Prompt template forwarded to
            the per-turn code generator.
        examples (list): Optional. Examples forwarded to the per-turn
            code generator.
        instructions (str): Optional. Instructions for the per-turn
            code generator. Defaults to either
            `get_recursive_instructions` (when ``recursive=True``,
            with the ``{max_llm_calls}`` placeholder substituted) or
            `get_default_instructions` otherwise.
        final_instructions (str): Optional. Instructions for the final
            answer generator. Defaults to ``instructions``.
        temperature (float): Optional. Sampling temperature
            (Default 0.0).
        max_tokens (int): Optional. Maximum number of tokens to generate.
            Default None (the model's own default; caps generation length).
        top_p (float): Optional. Nucleus sampling probability. Default None
            (the model's own default).
        top_k (int): Optional. Top-k sampling cutoff. Default None (the
            model's own default).
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
        tools (list): Optional. Extra `Tool` instances exposed to
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
        autonomous (bool): Optional. If ``True`` (default), run the
            full code/execute/observe loop until the LM calls
            ``submit`` or ``max_iterations`` is reached, then produce a
            structured final answer. If ``False``, require a
            ``ChatMessages`` input and execute a single code turn per
            call, returning the updated trajectory, suitable for
            human-in-the-loop use. For cross-call REPL state in
            interactive mode, hand a ``Sandbox`` to ``call`` via the
            ``sandbox`` kwarg; the agent itself stays stateless.
        return_inputs_with_trajectory (bool): Optional. Whether to
            return the full trajectory alongside the final answer
            (Default ``True``).
        max_iterations (int): Maximum number of code-execution turns
            before forcing the final answer step (Default 20).
        timeout (int): Per-turn execution budget in seconds
            (Default 60). Recursive sub-LM calls dominate per-turn wall
            time; ``llm_query_batched`` of even a handful of prompts
            can take several seconds. Snippets that exceed the budget
            turn into an observation so the LM can recover on the next
            turn.
        recursive (bool): Optional. If ``True`` (default), expose
            ``llm_query`` and ``llm_query_batched`` inside the sandbox
            and use the recursive instructions. If ``False``, run
            without the sub-LM helpers.
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
        workdir (str): Optional. Host directory the agent operates on. When
            building its own sandbox (i.e. no ``sandbox`` instance is supplied),
            the workdir seeds the sandbox filesystem. If it contains an
            ``AGENTS.md`` file, its contents are also injected as an additional
            input so the agent follows the declared project conventions (see
            ``read_agents_md``). Must point to an existing directory. Defaults
            to ``None``.
        skills (list): Optional. Folder paths (Agent Skill roots) whose skills
            are listed for the agent as an ``<available_skills>`` context message
            (see `FunctionCallingAgent`). The skill files must also be reachable
            from the agent's sandbox (e.g. under ``workdir``) for their bodies to
            be read on demand. Defaults to ``None``.
        sandbox (Sandbox): Optional. A pre-built ``Sandbox`` instance to
            reuse across calls. When supplied, the agent will not build
            its own sandbox at ``call()`` time and ``sandbox_type`` is
            derived from ``type(sandbox)``. Pass this when the caller
            owns the sandbox lifecycle (e.g. interactive sessions where
            REPL state must persist across calls). When omitted, a
            fresh sandbox of ``sandbox_type`` is built per call.
        sandbox_type (type): Optional. The ``Sandbox`` subclass to
            instantiate when no sandbox is supplied (here or to
            ``call()``). Defaults to ``MirageSandbox``, or to
            ``type(sandbox)`` when ``sandbox`` is given. Any ``Sandbox``
            subclass whose ``__init__`` accepts
            ``(timeout=..., name=...)`` works; register custom
            subclasses with ``@register_synalinks_serializable`` so
            they round-trip through ``get_config`` / ``from_config``.
        max_subagent_depth (int): When ``> 0``, the agent gains
            ``spawn_subagents`` / ``merge_subagent`` / ``discard_subagent``
            tools (called between snippets, with the REPL idle, so they can
            fork it). Each subagent runs in parallel on a
            `Sandbox.fork` that inherits this agent's current REPL
            state (variables, functions, imports) *and* files; its work only
            lands on an explicit ``merge_subagent``. ``1`` (recommended) lets
            this agent spawn subagents that cannot themselves spawn; higher
            values allow nesting. Defaults to ``0`` (disabled).

            Across parallel subagents you can fold back **all** their file
            changes, but only **one** subagent's REPL namespace per
            ``spawn_subagents`` batch (via ``merge_subagent(..., adopt_repl=
            True)``): the REPL serializes only as a whole, so parallel
            namespaces can't be unioned. That is a backend constraint, not a
            design shortcut.
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
        prompt_template=None,
        examples=None,
        instructions=None,
        final_instructions=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        reasoning_effort=None,
        use_chain_of_thought=False,
        tools=None,
        autonomous=True,
        return_inputs_with_trajectory=True,
        max_iterations=20,
        streaming=False,
        timeout=60,
        recursive=True,
        max_llm_calls=50,
        max_output_chars=10_000,
        workdir=None,
        skills=None,
        sandbox=None,
        sandbox_type=None,
        max_subagent_depth=0,
        _subagent_depth=0,
        name=None,
        description=None,
    ):
        if not isinstance(max_subagent_depth, int) or max_subagent_depth < 0:
            raise ValueError(
                "`max_subagent_depth` must be a non-negative int, got "
                f"{max_subagent_depth!r}"
            )
        # Domain attributes set before `super().__init__()`: the base
        # constructor builds the (inherited) step + final generators from the
        # instructions composed here, and `_get_builtin_tools` (called from
        # there) needs none of them — RLM's callable tool is built per call.
        self.max_subagent_depth = max_subagent_depth
        self._subagent_depth = _subagent_depth
        # Subagent delegation is offered only while we may still go one level
        # deeper, so the deepest subagents can't fan out endlessly.
        self._subagents_enabled = self._subagent_depth < self.max_subagent_depth

        self.recursive = recursive
        self.timeout = timeout
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars

        # `sub_language_model` defaults to the primary LM when omitted.
        # ``get(None)`` would raise, so resolve only when a value is given.
        self.sub_language_model = (
            _get_lm(sub_language_model)
            if sub_language_model is not None
            else _get_lm(language_model)
        )

        # Sandbox handling: a concrete sandbox supplied at construction is
        # reused across calls and its class becomes `sandbox_type`; otherwise a
        # fresh `sandbox_type` (default MirageSandbox) is built per `call()`.
        # Resolved here because the sandbox's prompt description is composed
        # into the instructions below.
        self.sandbox = sandbox
        if sandbox is not None:
            self.sandbox_type = type(sandbox)
        else:
            self.sandbox_type = sandbox_type or MirageSandbox

        # Compose instructions before delegating to the base constructor. The
        # final generator keeps the base instructions (recursive/default plus
        # subagent guidance); only the step generator gets the sandbox
        # description appended.
        if not instructions:
            if recursive:
                instructions = get_recursive_instructions().replace(
                    "{max_llm_calls}", str(max_llm_calls)
                )
            else:
                instructions = get_default_instructions()
        if self._subagents_enabled:
            # Idempotent: a serialized agent round-trips its post-append
            # instructions back through __init__, so don't append twice.
            guidance = get_subagent_tools_guidance()
            if guidance not in instructions:
                instructions = instructions + "\n\n" + guidance
        resolved_final_instructions = final_instructions or instructions
        sandbox_description = self.sandbox_type.description
        if sandbox_description:
            instructions = instructions + "\n\n" + sandbox_description

        super().__init__(
            schema=schema,
            data_model=data_model,
            language_model=language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=resolved_final_instructions,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            use_inputs_schema=use_inputs_schema,
            use_outputs_schema=use_outputs_schema,
            reasoning_effort=reasoning_effort,
            use_chain_of_thought=use_chain_of_thought,
            tools=tools,
            autonomous=autonomous,
            return_inputs_with_trajectory=return_inputs_with_trajectory,
            max_iterations=max_iterations,
            streaming=streaming,
            workdir=workdir,
            skills=skills,
            name=name,
            description=description,
        )

        # User tools are stored by the base constructor in `self.tools` but,
        # for RLM, are exposed *inside* the sandbox (advertised via the
        # catalog) rather than as native tool calls. Reject reserved helper
        # names here, after the base's public-name check.
        reserved = self._reserved_tool_names()
        for tool_name in self.tools:
            if tool_name in reserved:
                raise ValueError(
                    f"Tool name '{tool_name}' is reserved by {type(self).__name__}."
                )
        self.tools_catalog = _build_tools_catalog(self.tools)

    def _get_builtin_tools(self):
        # RLM exposes no native tools at construction: its only callable tool,
        # `run_python_code`, closes over a per-call sandbox and is built in
        # `_begin_call`. User tools live inside the sandbox, not as native calls.
        return []

    def _requires_tools(self):
        # `run_python_code` (built per call) is always available, so a user may
        # construct an RLM agent with no tools at all.
        return False

    def _reserved_tool_names(self) -> frozenset:
        """Names a user tool cannot collide with at construction time.

        ``submit``, ``llm_query``, and ``llm_query_batched`` are always
        reserved, even when ``recursive=False``, so that tool naming
        stays stable across the two modes and a user tool can't quietly
        shadow a helper name that would reappear if ``recursive`` is
        flipped back on. The subagent helpers are reserved too, for the
        same stability across ``max_subagent_depth`` settings.
        """
        return frozenset(
            {
                "submit",
                "llm_query",
                "llm_query_batched",
                "run_python_code",
                "spawn_subagents",
                "merge_subagent",
                "discard_subagent",
            }
        )

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
        the sandbox's ``run_python_code`` (which runs the
        snippet in the persistent sandbox) and clips the captured streams
        to ``max_output_chars``. The tools (``submit``, ``llm_query`` and
        the user tools) and the ``inputs`` payload are not passed here —
        `call` binds them onto the sandbox before each run (via
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
            """Execute one Python snippet in the persistent sandbox.

            State persists across calls — variables, imports and function
            definitions stay defined. The user input is bound as a dict named
            `inputs`; read full values via `inputs[field]`. Other tools
            (`submit`, `llm_query`, ...) are pre-imported functions — call them
            directly, e.g. `out = llm_query(prompt)`. Call `submit(result={...})`
            to end the run.

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

    def _build_subagent_tools(self, sandbox, registry, counter, repl_state):
        """Build the per-call subagent tools (spawn / merge / discard).

        These are *native* tools the LM calls directly — never from inside a
        ``run_python_code`` snippet — so the sandbox REPL is idle when they
        fork or merge it (a busy REPL can't be dumped). The closures capture
        the resolved ``sandbox``, a per-call ``registry`` (handle -> fork),
        a handle ``counter``, and ``repl_state`` tracking the single REPL
        adoption allowed per turn.
        """
        from synalinks.src.modules.core.input_module import Input
        from synalinks.src.programs.program import Program

        async def spawn_subagents(tasks: List[str]) -> dict:
            """Run subagents in parallel, each on an isolated fork of the sandbox.

            Each task is handed to a fresh subagent on its own fork that
            inherits your current REPL state (variables, functions, imports)
            and files. Subagents run concurrently; their changes are isolated.
            Review each returned ``patch`` then ``merge_subagent(handle)`` to
            fold a subagent's work into your sandbox.

            Args:
                tasks (list): One instruction string per subagent describing
                    what that subagent should accomplish.

            Returns:
                dict: ``subagents`` — a list of ``{handle, task, result,
                diff, patch}`` per subagent, where ``patch`` is the subagent's
                pending changes as a git-style unified diff (the actual
                line-level edits) and ``diff`` is the structured
                ``{written, deleted}`` summary; or ``{handle, task, error}``
                for a failed subagent; or a top-level ``error`` when ``tasks``
                is empty.
            """
            prompts = [str(t) for t in (tasks or [])]
            if not prompts:
                return {"error": "no tasks provided"}

            async def run_one(index, prompt):
                # Subagents inherit the parent's confinement (``confine=None``):
                # when this sandbox is confined, the subagent is confined to its
                # OWN fork (host hidden, network cut, isolated filesystem, and
                # the parent's egress/mount/seccomp posture); when the parent
                # runs unconfined, so does the subagent.
                fork = sandbox.fork(
                    copy_repl=True, name=f"{self.name}_sub{index}", confine=None
                )
                subagent = RecursiveLanguageModelAgent(
                    language_model=self.sub_language_model,
                    sub_language_model=self.sub_language_model,
                    sandbox=fork,
                    recursive=self.recursive,
                    instructions=get_subagent_instructions(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    reasoning_effort=self.reasoning_effort,
                    use_chain_of_thought=self.use_chain_of_thought,
                    max_iterations=self.max_iterations,
                    max_llm_calls=self.max_llm_calls,
                    max_output_chars=self.max_output_chars,
                    max_subagent_depth=self.max_subagent_depth,
                    _subagent_depth=self._subagent_depth + 1,
                    return_inputs_with_trajectory=False,
                    autonomous=True,
                    name=f"{self.name}_sub{index}",
                )
                # Run through a Program (the canonical path) so the subagent's
                # build is LM-free and it isn't double-invoked.
                inp = Input(data_model=ChatMessages)
                out = await subagent(inp)
                program = Program(
                    inputs=inp, outputs=out, name=f"{self.name}_sub{index}_prog"
                )
                messages = ChatMessages(
                    messages=[ChatMessage(role=ChatRole.USER, content=prompt)]
                )
                result = await program(messages)
                return fork, _subagent_answer(result)

            results = await asyncio.gather(
                *(run_one(i, p) for i, p in enumerate(prompts)),
                return_exceptions=True,
            )
            report = []
            for prompt, res in zip(prompts, results):
                handle = f"subagent_{counter[0]}"
                counter[0] += 1
                if isinstance(res, Exception):
                    report.append(
                        {
                            "handle": handle,
                            "task": prompt,
                            "error": f"{type(res).__name__}: {res}",
                        }
                    )
                    continue
                fork, answer = res
                registry[handle] = fork
                report.append(
                    {
                        "handle": handle,
                        "task": prompt,
                        "result": answer,
                        "diff": fork.diff(),
                        "patch": fork.patch(),
                    }
                )
            return {"subagents": report}

        async def merge_subagent(
            handle: str,
            paths: Optional[List[str]] = None,
            force: bool = False,
            adopt_repl: bool = False,
        ) -> dict:
            """Fold a subagent's changes into your sandbox.

            Merges the subagent's file changes (``paths`` / ``force`` as for
            files). With ``adopt_repl=True``, also adopts the subagent's whole
            Python namespace (variables/functions/imports) — only ONE
            subagent's REPL can be adopted per ``spawn_subagents`` batch (a
            second would overwrite the first).

            Args:
                handle (str): A handle returned by ``spawn_subagents``.
                paths (list): Optional subset of virtual paths to merge.
                force (bool): Apply conflicting file paths instead of
                    refusing them.
                adopt_repl (bool): Also adopt the subagent's whole REPL
                    namespace.

            Returns:
                dict: ``{written, deleted, conflicts, skipped, repl_adopted}``,
                or ``error`` for an unknown handle.
            """
            fork = registry.get(handle)
            if fork is None:
                return {"error": f"unknown subagent handle: {handle!r}"}
            do_repl = bool(adopt_repl)
            note = None
            if do_repl and repl_state["adopted"]:
                do_repl = False
                note = (
                    "REPL already adopted from another subagent this turn; "
                    "merged files only (a second adoption would overwrite the "
                    "first)."
                )
            report = sandbox.merge(fork, paths=paths, force=force, repl=do_repl)
            if do_repl:
                repl_state["adopted"] = True
            if note:
                report = dict(report)
                report["repl_warning"] = note
            return report

        async def discard_subagent(handle: str) -> dict:
            """Drop a subagent's fork without applying any of its changes.

            Args:
                handle (str): A handle returned by ``spawn_subagents``.

            Returns:
                dict: ``{discarded: handle}``, or ``error`` for an unknown
                handle.
            """
            if registry.pop(handle, None) is None:
                return {"error": f"unknown subagent handle: {handle!r}"}
            return {"discarded": handle}

        return {
            "spawn_subagents": Tool(spawn_subagents, name="spawn_subagents"),
            "merge_subagent": Tool(merge_subagent, name="merge_subagent"),
            "discard_subagent": Tool(discard_subagent, name="discard_subagent"),
        }

    async def _begin_call(self, inputs, training, *, sandbox=None):
        # Per-call tool set: user tools plus a fresh `submit` bound to a
        # private holder, plus any per-call recursive helpers. submit is the
        # canonical termination signal, always exposed, schema'd or not, and
        # everything here is built fresh per call so concurrent invocations
        # don't share holders, counters, or locks.
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
            # complete `inputs_json` rebound on every `run_python_code` call,
            # so `inputs[field]` is always reachable.
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

        # Sandbox resolution order: per-call kwarg > constructor-supplied
        # sandbox > fresh sandbox of `sandbox_type`. The first two cases let the
        # caller (or the agent's owner) keep REPL state alive across calls; the
        # third is the stateless-per-call default. A freshly built sandbox is
        # seeded from `workdir` when one is set.
        if sandbox is None:
            if self.sandbox is not None:
                sandbox = self.sandbox
            elif self.workdir is not None:
                sandbox = self.sandbox_type(workdir=self.workdir, timeout=self.timeout)
            else:
                sandbox = self.sandbox_type(timeout=self.timeout)

        # The per-turn snippet is delivered as a native `run_python_code` tool
        # call — the only tool the LM can call, wrapping the sandbox's own
        # `run_python_code`. The sandbox-side tools (submit, llm_query, user
        # tools) are NOT exposed to the LM; they live inside the sandbox as
        # plain synchronous functions (advertised via the tools catalog). Bind
        # them onto the sandbox so every `run_python_code` snippet can reach them.
        external_functions = {
            name: _adapt_tool_for_sandbox(t) for name, t in call_tools.items()
        }
        sandbox.bind_functions(external_functions)
        # Persist the full input payload as `inputs` in the sandbox namespace.
        # The per-run `inputs=` binding does not persist, so copy it into a real
        # variable once; every snippet then reads it via `inputs[field]`.
        await sandbox.run("inputs = _rlm_inputs", inputs={"_rlm_inputs": inputs_json})
        run_tool = self._build_run_python_code_tool(sandbox)

        # Subagent tools (spawn / merge / discard) are *native* tools the LM
        # calls directly alongside run_python_code — never from inside a
        # snippet — so the REPL is idle when they fork/merge it. Built fresh per
        # call with a private fork registry and a single-REPL-adoption guard.
        subagent_registry = {}
        extra_native_tools = {}
        if self._subagents_enabled:
            extra_native_tools = self._build_subagent_tools(
                sandbox, subagent_registry, [0], {"adopted": False}
            )

        ctx = {
            "sandbox": sandbox,
            "run_tool": run_tool,
            "extra_native_tools": extra_native_tools,
            "submit_holder": submit_holder,
            "native_tools": [run_tool, *extra_native_tools.values()],
            "submitted_final": None,
        }
        return trajectory, ctx

    def _native_tools(self, ctx):
        # The LM only ever calls `run_python_code` (plus the subagent tools);
        # user tools are reachable from inside the sandbox, not as native calls.
        return ctx["native_tools"]

    def _requires_tools(self):
        return False

    def _on_empty_generation(self, agent_messages, ctx):
        # RLM simply stops the loop on an empty generation (no message added).
        return None

    async def _on_no_tool_calls(self, tool_calls, agent_messages, ctx):
        # The shared loop only appends the assistant message when there are
        # native tool calls; on an empty turn RLM keeps it and nudges the model
        # toward `run_python_code` / `submit`, then keeps iterating.
        agent_messages.append(tool_calls.get_json())
        agent_messages.append(
            ChatMessage(
                role=ChatRole.USER,
                content=(
                    "(no tool call) Call `run_python_code` with a snippet, and "
                    "call `submit(result={...})` inside it to terminate the run."
                ),
            ).get_json()
        )
        return False

    async def _dispatch_tool_calls(self, native_tool_calls, agent_messages, ctx):
        # Sequential dispatch: a `submit()` inside a snippet writes to the
        # holder, short-circuiting the run. The subagent tools fork/merge the
        # idle REPL, so they must not run concurrently with a snippet.
        run_tool = ctx["run_tool"]
        extra_native_tools = ctx["extra_native_tools"]
        submit_holder = ctx["submit_holder"]
        for tool_call in native_tool_calls:
            function = tool_call.get("function") or {}
            tool_name = function.get("name")
            tool_args = function.get("arguments") or {}
            tool_call_id = tool_call.get("id")
            if tool_name in extra_native_tools:
                # spawn_subagents / merge_subagent / discard_subagent —
                # native tools, invoked with the REPL idle.
                try:
                    result = await extra_native_tools[tool_name](**tool_args)
                    content = result.get_json() if hasattr(result, "get_json") else result
                except Exception as e:
                    content = {"error": f"{type(e).__name__}: {e}"}
            elif tool_name != "run_python_code":
                callable_tools = ", ".join(
                    ["'run_python_code'", *(f"'{n}'" for n in extra_native_tools)]
                )
                content = {
                    "error": (
                        f"Unknown tool '{tool_name}'. Callable tools are "
                        f"{callable_tools}; everything else is a sandbox "
                        "function reachable from your snippet."
                    )
                }
            else:
                try:
                    result = await run_tool(**tool_args)
                    content = result.get_json() if hasattr(result, "get_json") else result
                except Exception as e:
                    content = {"error": f"{type(e).__name__}: {e}"}
                # submit() inside the snippet wrote to the holder. Clear it
                # either way so a stale payload can't short-circuit a later retry.
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
                            ctx["submitted_final"] = submitted
                            content["submit"] = "accepted"
                    else:
                        ctx["submitted_final"] = submitted
                        content["submit"] = "accepted"
            agent_messages.append(
                ChatMessage(
                    role=ChatRole.TOOL,
                    tool_call_id=tool_call_id,
                    content=content,
                ).get_json()
            )
            if ctx["submitted_final"] is not None:
                return True
        return False

    async def _run_interactive(self, trajectory, agent_messages, ctx, training):
        # RLM's interactive mode is a single pass of the same loop.
        return await self._run_loop(
            trajectory, agent_messages, ctx, training, max_steps=1
        )

    def _wrap_trajectory(self, agent_messages):
        return JsonDataModel(
            json=ChatMessages(
                messages=[ChatMessage(**msg) for msg in agent_messages]
            ).get_json(),
            schema=ChatMessages.get_schema(),
            name=self.name,
        )

    async def _finish(self, trajectory, agent_messages, ctx, training):
        submitted_final = ctx["submitted_final"]

        # Interactive mode: only invoke the final generator when the LM itself
        # signalled completion via submit. Otherwise return the updated
        # trajectory so the caller can decide when to continue.
        if not self.autonomous and submitted_final is None:
            return self._wrap_trajectory(agent_messages)

        # submit short-circuit: the LM already produced the final payload inside
        # the sandbox, so skip the final-formatting LM call. Schemaless mode
        # treats the payload as the content of a final assistant message.
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
                        content=submitted_final.get("answer", submitted_final),
                    ).get_json()
                )
                return self._wrap_trajectory(agent_messages)
        else:
            final_result = await self.final_generator(trajectory)
            if not self.schema:
                # Schemaless fallback: the final generator emits a ChatMessage.
                if final_result:
                    agent_messages.append(final_result.get_json())
                return self._wrap_trajectory(agent_messages)

        if self.return_inputs_with_trajectory:
            return await ops.concat(
                self._wrap_trajectory(agent_messages),
                final_result,
                name=self.name,
            )
        return final_result

    async def compute_output_spec(self, inputs, training=False, sandbox=None, **kwargs):
        # See FunctionCallingAgent.compute_output_spec: `call()` takes **kwargs
        # (sandbox is threaded through it), so mirror the signature here.
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
        _ = await self.tool_calls_generator(generator_inputs, tools=[spec_tool])
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
        config = super().get_config()
        config.update(
            {
                "recursive": self.recursive,
                "timeout": self.timeout,
                "max_llm_calls": self.max_llm_calls,
                "max_output_chars": self.max_output_chars,
                "max_subagent_depth": self.max_subagent_depth,
                "sandbox_type": get_registered_name(self.sandbox_type),
                "sub_language_model": serialization_lib.serialize_synalinks_object(
                    self.sub_language_model,
                ),
                "sandbox": (
                    serialization_lib.serialize_synalinks_object(self.sandbox)
                    if self.sandbox is not None
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if config.get("sub_language_model") is not None:
            config["sub_language_model"] = serialization_lib.deserialize_synalinks_object(
                config.pop("sub_language_model")
            )
        else:
            config.pop("sub_language_model", None)
        sandbox_type_name = config.pop("sandbox_type", None)
        config["sandbox_type"] = (
            get_registered_object(sandbox_type_name) if sandbox_type_name else None
        )
        if config.get("sandbox") is not None:
            config["sandbox"] = serialization_lib.deserialize_synalinks_object(
                config.pop("sandbox")
            )
        else:
            config.pop("sandbox", None)
        return super().from_config(config)
