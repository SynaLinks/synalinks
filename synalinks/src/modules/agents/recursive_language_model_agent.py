# Modified from: dspy/predict/rlm.py
# Original authors: Alex L. Zhang, Tim Kraska, Omar Khattab (DSPy Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.modules.agents.code_mode_agent import CodeModeAgent
from synalinks.src.modules.core.tool import Tool
from synalinks.src.saving import serialization_lib


def get_default_instructions():
    """Default instructions for the recursive-language-model agent."""
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


@synalinks_export(
    [
        "synalinks.modules.RecursiveLanguageModelAgent",
        "synalinks.RecursiveLanguageModelAgent",
        "synalinks.modules.RLM",
        "synalinks.RLM",
    ]
)
class RecursiveLanguageModelAgent(CodeModeAgent):
    """A recursive-language-model agent.

    A code-mode agent extended with two always-on recursive helpers:
    ``llm_query(prompt)`` and ``llm_query_batched(prompts)``. The agent
    treats long inputs as an *external environment*, it writes Python
    that slices, filters, and aggregates the data, and recursively
    delegates semantic work to a sub-LM only on the snippets it cares
    about. Compared to feeding a long document straight into the
    primary LM, this trades a single huge context for many small ones,
    which both fits inside provider limits and reduces the chance of
    long-context regressions.

    State (variables, imports, function definitions) accumulates across
    turns in the persistent sandbox, so the agent can build up
    intermediate values, probe data, and iterate. The ``submit`` tool
    is the canonical termination signal, exactly as in
    :class:`CodeModeAgent`.

    The ``llm_query`` quota is per-call: every invocation of this agent
    gets a fresh budget of ``max_llm_calls`` sub-LM queries, and
    concurrent invocations of the *same* agent instance each get an
    independent budget, the counter and lock are built inside
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
        outputs = await synalinks.RecursiveLanguageModelAgent(
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
            ``call`` returns the ``ChatMessages`` trajectory directly
            (mirroring ``FunctionCallingAgent``'s schemaless behaviour).
        data_model (DataModel | SymbolicDataModel | JsonDataModel):
            Optional. The target data model for the final answer.
        language_model (LanguageModel): The language model driving the
            per-turn code generator and the final-formatting step.
        sub_language_model (LanguageModel): Optional. The language
            model used by ``llm_query`` and ``llm_query_batched``.
            Defaults to ``language_model``, pass a cheaper / smaller
            model here when the recursive sub-queries don't need the
            primary LM's full capability.
        tools (list): Optional. Extra :class:`Tool` instances exposed
            to the sandbox in addition to ``submit``, ``llm_query``,
            and ``llm_query_batched``. The names ``llm_query``,
            ``llm_query_batched``, and ``submit`` are reserved.

            **Naming gotcha**: each tool is registered under
            ``tool.name == tool._func.__name__``. ``Tool(_my_helper)``
            shows up inside the script as ``_my_helper``. Rename the
            function rather than relying on an alias.
        prompt_template (str): Optional. Prompt template forwarded to
            the per-turn code generator.
        examples (list): Optional. Examples forwarded to the per-turn
            code generator.
        instructions (str): Optional. Instructions for the per-turn
            code generator. Defaults to ``get_default_instructions()``
            with the ``{max_llm_calls}`` placeholder substituted.
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
            (Default 60). Higher than the parent's 5s default because
            recursive sub-LM calls dominate per-turn wall time —
            ``llm_query_batched`` of even a handful of prompts can
            blow past 5s. Snippets that exceed the budget turn into an
            observation so the LM can recover on the next turn.
        max_iterations (int): Maximum number of code-execution turns
            before forcing the final answer step (Default 20). Higher
            than the parent's 5 because RLM workflows characteristically
            explore → carve → batch-query → aggregate → submit, which
            needs more turns than plain code-mode reasoning.
        max_llm_calls (int): Hard cap on sub-LM calls per agent
            invocation, shared between ``llm_query`` and
            ``llm_query_batched`` (Default 50). Once the budget is
            spent, further calls return an error string instead of a
            response so the LM can fall back to code-side aggregation.
        max_output_chars (int): Maximum characters to include from
            REPL output in the per-turn observation (Default 10_000).
            Anything beyond is truncated with a
            ``… (truncated, N chars omitted)`` marker so a single
            noisy turn cannot blow up the trajectory.
        return_inputs_with_trajectory (bool): Optional. Whether to
            return the full trajectory alongside the final answer
            (Default ``True``).
        sandbox_type (type): Optional. The ``Sandbox`` subclass to
            instantiate when no caller-supplied sandbox is passed to
            ``call()``. Defaults to ``MontySandbox``. Any ``Sandbox``
            subclass whose ``__init__`` accepts
            ``(timeout=..., name=...)`` works; register custom
            subclasses with ``@register_synalinks_serializable`` so
            they round-trip through ``get_config`` / ``from_config``.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
    """

    _RESERVED_TOOL_NAMES = frozenset({"submit", "llm_query", "llm_query_batched"})

    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model=None,
        sub_language_model=None,
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
        sandbox_type=None,
        name=None,
        description=None,
    ):
        if not instructions:
            instructions = get_default_instructions().replace(
                "{max_llm_calls}",
                str(max_llm_calls),
            )
        super().__init__(
            schema=schema,
            data_model=data_model,
            language_model=language_model,
            tools=tools,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            temperature=temperature,
            use_inputs_schema=use_inputs_schema,
            use_outputs_schema=use_outputs_schema,
            reasoning_effort=reasoning_effort,
            use_chain_of_thought=use_chain_of_thought,
            autonomous=autonomous,
            timeout=timeout,
            max_iterations=max_iterations,
            max_output_chars=max_output_chars,
            return_inputs_with_trajectory=return_inputs_with_trajectory,
            sandbox_type=sandbox_type,
            name=name,
            description=description,
        )
        self.sub_language_model = sub_language_model or self.language_model
        self.max_llm_calls = max_llm_calls

    def _build_extra_call_tools(self):
        """Build the per-call ``llm_query`` / ``llm_query_batched`` pair.

        A fresh counter+lock is built on every invocation so concurrent
        calls into the same agent instance get independent budgets.
        """
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

    def get_config(self):
        config = super().get_config()
        config["max_llm_calls"] = self.max_llm_calls
        config["sub_language_model"] = serialization_lib.serialize_synalinks_object(
            self.sub_language_model,
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "sub_language_model" in config:
            config["sub_language_model"] = serialization_lib.deserialize_synalinks_object(
                config["sub_language_model"]
            )
        return super().from_config(config)
