# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_chat_messages
from synalinks.src.backend.common.op_scope import trajectory_scope
from synalinks.src.modules.agents.utils.agents_utils import agents_md_prompt
from synalinks.src.modules.agents.utils.agents_utils import discover_agents_md
from synalinks.src.modules.agents.utils.agents_utils import merge_tools
from synalinks.src.modules.agents.utils.agents_utils import prepend_context_message
from synalinks.src.modules.agents.utils.agents_utils import resolve_workdir
from synalinks.src.modules.agents.utils.skills_utils import discover_skills_in_roots
from synalinks.src.modules.agents.utils.skills_utils import resolve_skills_paths
from synalinks.src.modules.agents.utils.skills_utils import skills_prompt
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.language_models.language_model import StreamingIterator
from synalinks.src.modules.module import Module
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.saving import serialization_lib


def get_default_instructions():
    """The default parallel function calling agent instructions."""
    return """
Think step by step: Use the thinking field to elaborate what you observe and
what do you need to accomplish next.
Reflect on prior steps: Review your previous actions and their outcomes to
avoid unnecessary repetition.
Avoid unnecessary actions: If you already have enough information to complete
the user task, return an empty tool calls array.
""".strip()


@synalinks_export(
    [
        "synalinks.modules.FunctionCallingAgent",
        "synalinks.FunctionCallingAgent",
    ]
)
class FunctionCallingAgent(Module):
    """A trainable parallel function calling agent.

    The agent has 2 different modes:

    - Autonomous: It will execute tools as soon as called.
    - Non-autonomous: It will return the tool arguments as a ChatMessage.

    In *autonomous* mode, the agent accept **any kind of data model input**
    and perform a final inference to eventually format its final answer if a
    `data_model` or `schema` is provided.

    Example:

    ```python
    import synalinks
    import asyncio

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class NumericalFinalAnswer(synalinks.DataModel):
        final_answer: float = synalinks.Field(
            description="The correct final numerical answer",
        )

    async def calculate(expression: str):
        \"""Calculate the result of a mathematical expression.

        Args:
            expression (str): The mathematical expression to calculate, such as
                '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
                parentheses, and spaces.
        \"""
        if not all(char in "0123456789+-*/(). " for char in expression):
            return {
                "result": None,
                "log": (
                        "Error: invalid characters in expression. "
                        "The expression can only contain numbers, operators (+, -, *, /),"
                        " parentheses, and spaces NOT letters."
                    ),
            }
        try:
            # Evaluate the mathematical expression safely
            result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
            return {
                "result": result,
                "log": "Successfully executed",
            }
        except Exception as e:
            return {
                "result": None,
                "log": f"Error: {e}",
            }

    async def main():
        language_model = synalinks.LanguageModel(model="ollama/mistral")

        tools = [
            synalinks.Tool(calculate),
        ]

        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.FunctionCallingAgent(
            data_model=NumericalFinalAnswer,
            tools=tools,
            language_model=language_model,
            max_iterations=5,
            autonomous=True,
        )(inputs)
        agent = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="math_agent",
            description="A math agent",
        )

        input_query = Query(query="How much is 152648 + 485?")
        response = await agent(input_query)

        print(response.prettify_json())

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Result:

    ```json
    {
        "query": "How much is 152648 + 485?",
        "messages": [
            {
            "role": "assistant",
            "content": "Performing simple addition",
            "tool_calls": [
                {
                    "id": "92a3657c-1a45-46e6-8173-df4255b8423b",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": {
                            "expression": "152648 + 485"
                            }
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": {
                    "result": 153133.0,
                    "log": "Successfully executed"
                },
                "tool_call_id": "92a3657c-1a45-46e6-8173-df4255b8423b",
            },
            {
                "role": "assistant",
                "content": "The user has asked for a simple addition "
                "calculation. The assistant used the 'calculate' tool to "
                "perform this task, and the result was returned successfully.",
            }
        ],
        "final_answer": 153133.0
    }
    ```

    In *non-autonomous* mode (also called human in the loop or interactive mode), the
    user needs to validate/edit the tool arguments and send it back to the agent. In this
    mode, the agent requires an `ChatMessages` data model as input and output an
    `ChatMessage` (or `ChatMessages` if `return_inputs_with_trajectory` is true)
    back to the user. In that case, the agent ignore the `max_iterations` argument,
    as it will only perform one **step at a time**.

    Example:

    ```python
    import synalinks
    import asyncio

    MAX_ITERATIONS = 5

    async def calculate(expression: str):
        \"""Calculate the result of a mathematical expression.

        Args:
            expression (str): The mathematical expression to calculate, such as
                '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
                parentheses, and spaces.
        \"""
        if not all(char in "0123456789+-*/(). " for char in expression):
            return {
                "result": None,
                "log": (
                        "Error: invalid characters in expression. "
                        "The expression can only contain numbers, operators (+, -, *, /),"
                        " parentheses, and spaces NOT letters."
                    ),
            }
        try:
            # Evaluate the mathematical expression safely
            result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
            return {
                "result": result,
                "log": "Successfully executed",
            }
        except Exception as e:
            return {
                "result": None,
                "log": f"Error: {e}",
            }

    async def main():

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        tools = [
            synalinks.Tool(calculate),
        ]

        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        outputs = await synalinks.FunctionCallingAgent(
            tools=tools,
            language_model=language_model,
            return_inputs_with_trajectory=True,
            autonomous=False,
        )(inputs)
        agent = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="math_agent",
            description="A math agent",
        )

        input_messages = synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content="How much is 152648 + 485?",
                )
            ]
        )

        for i in range(MAX_ITERATIONS):

            response = await agent(input_messages)

            print("Assistant response (with trajectory):")
            print(response.prettify_json())

            assistant_message = response.get("messages")[-1]

            if not assistant_message.get("tool_calls"):
                break # We stop the loop if the agent didn't call any tool

            # Validate the tool calls arguments (with an UI or CLI)
            # Then re-inject the validated assistant response in the input_messages
            # The corresponding tools will be called by the agent
            # Here we assume everything is okay for the purpose of the demo.

            input_messages.messages.append(assistant_message)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    The FunctionCallingAgent is compatible with MCP tools,
    here is an example on how to use it:

    ```python
    import synalinks
    import asyncio
    import litellm

    class Query(synalinks.DataModel):
        \"""Input query data model\"""

        query: str = synalinks.Field(
            description="The user query",
        )

    class FinalAnswer(synalinks.DataModel):
        \"""Final answer data model\"""

        answer: str = synalinks.Field(
            description="The correct final answer",
        )


    async def main():

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        mcp_client = synalinks.MultiServerMCPClient(
            {
                "math": {
                    "url": "http://localhost:8183/mcp/",
                    "transport": "streamable_http",
                },
            }
        )

        tools = await mcp_client.get_tools()

        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.FunctionCallingAgent(
            data_model=FinalAnswer,
            tools=tools,
            language_model=language_model,
            max_iterations=5,
            autonomous=True,
        )(inputs)

        agent = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="mcp_math_agent",
            description="A math agent that can use an external calculator",
        )

        input_query = Query(query="How much is 152648 + 485?")
        response = await agent(input_query)

        print(response.prettify_json())


    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        schema (dict): The target JSON schema.
            If not provided use the `data_model` to infer it.
        data_model (DataModel | SymbolicDataModel | JsonDataModel): The target data
            model for structured output.
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The jinja2 prompt template.
        examples (list): The default list of examples, the examples
            are a list of tuples containing input/output JSON pairs.
        instructions (str): The default instructions being a string containing
            instructions for the language model.
        final_instructions (str): Optional. The instructions for the final generator
            that produces the structured output. If not provided, use the same
            instructions as the tool calls generator.
        temperature (float): Optional. The temperature for the LM call.
        max_tokens (int): Optional. Maximum number of tokens to generate. Default
            None (the model's own default; caps generation length when set).
        top_p (float): Optional. The nucleus sampling probability for the LM call.
            Default None (the model's own default).
        top_k (int): Optional. The top-k sampling cutoff for the LM call.
            Default None (the model's own default).
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False).
        reasoning_effort (string): Optional. The reasoning effort for the LM call
            between ['minimal', 'low', 'medium', 'high', 'disable', 'none', None].
            Default to None (no reasoning).
        use_chain_of_thought (bool): Optional. Use chain of thought for tool
            calls generator, usefull when using non-reasoning models. Default False.
        tools (list): The list of `Tool` or MCP tools available to the agent.
        autonomous (bool): Optional. Whether the agent runs autonomously
            (executing tools automatically) or in interactive mode where the user
            validates tool arguments before execution (Default to True).
        return_inputs_with_trajectory (bool): Optional. Whether or not to return the
            inputs concatenated with the full message trajectory (Default to True).
        max_iterations (int): Optional. The maximum number of tool calling iterations
            in autonomous mode (Default to 5). Ignored in interactive mode.
        workdir (str): Optional. Path to a working directory. When provided and the
            directory contains an `AGENTS.md` file, its contents are injected as an
            additional input message so the agent follows the project conventions
            declared there (see `read_agents_md`). Must point to an existing
            directory (Default to None).
        skills (list): Optional. A list of folder paths, each a *root* directory
            under which Agent Skills live as ``<root>/<name>/SKILL.md`` (the open
            agentskills.io standard). The discovered skills' names and
            descriptions are injected as an ``<available_skills>`` context
            message so the agent knows what is available; per progressive
            disclosure, each skill's full ``SKILL.md`` body and bundled files are
            read on demand through the agent's own file/bash tools. Each path
            must point to an existing directory (Default to None).
        streaming (bool): Optional. If true, stream the final answer. Only takes
            effect when no `data_model`/`schema` is provided. When streaming,
            the agent returns a `StreamingIterator` instead of a wrapped
            trajectory; the caller iterates it to consume the final response.
            (Default to False).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
    """

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        language_model=None,
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
        max_iterations=5,
        streaming=False,
        workdir=None,
        skills=None,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )

        # `workdir` is optional. When set it must be an existing directory;
        # an `AGENTS.md` inside it is injected as an extra input message at
        # call time (see `read_agents_md`).
        self.workdir = resolve_workdir(workdir)
        # Read AGENTS.md once, here at construction — re-reading it on every
        # `call()` could pick up a changed/corrupted file and would defeat the
        # re-injection guard (which compares against this stable message).
        self.agents_md_message = self.read_agents_md()
        # `skills` are Agent Skill *root* directories. Discover them once at
        # construction (same rationale as AGENTS.md) into a stable
        # `<available_skills>` context message injected at call time.
        self.skills = resolve_skills_paths(skills)
        self.skills_message = self.read_skills()
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        self.prompt_template = prompt_template

        if not instructions:
            instructions = get_default_instructions()
        self.instructions = instructions
        if not final_instructions:
            self.final_instructions = instructions
        else:
            self.final_instructions = final_instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k

        self.examples = examples
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.reasoning_effort = reasoning_effort
        self.use_chain_of_thought = use_chain_of_thought
        self.language_model = _get_lm(language_model)

        # Built-in tools come from the `_get_builtin_tools` hook (empty for the
        # base agent; subclasses inject domain tools). User-supplied `tools`
        # are merged on top, with name-collision and underscore checks.
        self.extra_tools = list(tools) if tools else []
        merged_tools = merge_tools(
            self._get_builtin_tools(),
            self.extra_tools,
            kind=self._builtin_tool_kind(),
        )
        self.tools = {}
        if not merged_tools and self._requires_tools():
            raise ValueError("You must set the `tools` argument")
        for tool in merged_tools:
            if tool.name.startswith("_"):
                raise ValueError(
                    f"Tool name {tool.name!r} starts with an underscore. "
                    f"Tools exposed to the LM must have public names — "
                    f"rename the function or pass an explicit `name=` to "
                    f"Tool(...)."
                )
            self.tools[tool.name] = tool

        self.autonomous = autonomous
        self.return_inputs_with_trajectory = return_inputs_with_trajectory
        self.max_iterations = max_iterations
        # Streaming is only meaningful for the final answer (no schema).
        if self.schema and streaming:
            streaming = False
        self.streaming = streaming

        # Native function-calling: the LM picks tools and emits a
        # ChatMessage with `tool_calls` populated in the synalinks flat
        # shape ({id, name, arguments}). Tool-call IDs are the provider's
        # own — no local uuid4 needed. Tools are passed at call time
        # (see `call`), not at construction.
        tool_calls_generator_cls = ChainOfThought if use_chain_of_thought else Generator
        self.tool_calls_generator = tool_calls_generator_cls(
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            reasoning_effort=self.reasoning_effort,
            language_model=self.language_model,
            name="tool_calls_generator_" + self.name,
        )

        self.final_generator = Generator(
            schema=self.schema,
            language_model=self.language_model,
            instructions=self.final_instructions,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            reasoning_effort=self.reasoning_effort,
            return_inputs=False,
            streaming=self.streaming,
            name="final_generator_" + self.name,
        )

    def _get_builtin_tools(self):
        """Return the agent's built-in tools (hook for subclasses).

        The base `FunctionCallingAgent` has no built-in tools — it exposes only
        the user-supplied `tools`. Subclasses (e.g. `SQLAgent`, `DeepAgent`)
        override this to inject domain tools, which are merged ahead of the
        user's tools and protected against name collisions. Domain attributes
        the hook depends on must be assigned *before* ``super().__init__()``.
        """
        return []

    def _builtin_tool_kind(self):
        """Short noun naming the built-in tool family, used in collision errors."""
        return "agent"

    def _requires_tools(self):
        """Whether at least one tool must be present at construction (hook).

        The base agent calls user tools natively, so it needs at least one.
        Subclasses whose callable tool is built per-call (e.g. RLM's
        `run_python_code`) override this to allow zero construction-time tools.
        """
        return True

    def read_agents_md(self):
        """Convert the workdir's root ``AGENTS.md`` into an input message.

        When a ``workdir`` is configured and contains a non-empty root
        ``AGENTS.md``, its body is taken verbatim (no added framing — the
        agents.md spec prescribes no prompt wording) and wrapped in a user
        ``ChatMessage`` for injection at the front of the trajectory (see
        `discover_agents_md` / `agents_md_prompt`). No sandbox needed.

        Returns:
            (ChatMessage | None): A user message carrying the conventions, or
                ``None`` when no ``workdir`` is set or no non-empty ``AGENTS.md``
                is found.
        """
        if not self.workdir:
            return None
        content = agents_md_prompt(discover_agents_md(self.workdir))
        if not content:
            return None
        return ChatMessage(role=ChatRole.USER, content=content)

    def read_skills(self):
        """Build the ``<available_skills>`` context message from `skills` roots.

        Discovers the Agent Skills under each configured root (deduped by name,
        first root wins) and renders the ``<available_skills>`` XML block — names
        and descriptions only (level 1 / progressive disclosure). The block is
        wrapped in a user ``ChatMessage`` for injection at the front of the
        trajectory (see `prepend_context_message`).

        Returns:
            (ChatMessage | None): A user message listing the available skills, or
                ``None`` when no `skills` roots are set or none contain a skill.
        """
        if not self.skills:
            return None
        skills = discover_skills_in_roots(self.skills)
        if not skills:
            return None
        return ChatMessage(role=ChatRole.USER, content=skills_prompt(skills))

    async def call(self, inputs, training=False, **kwargs):
        if not inputs:
            return None
        if not self.autonomous and not is_chat_messages(inputs):
            raise ValueError(
                f"In interactive mode, the {type(self).__name__} needs a "
                "ChatMessages-like data model as inputs"
            )
        # `call` is a template method: it builds the trajectory, injects
        # AGENTS.md, then runs the autonomous loop or the interactive step.
        # Subclasses (e.g. RLM) specialize behavior by overriding the hooks
        # below (`_begin_call`, `_native_tools`, `_dispatch_tool_calls`,
        # `_final_result`, ...) rather than reimplementing `call`. Extra
        # keyword arguments (e.g. RLM's per-call `sandbox=`) are forwarded to
        # `_begin_call`.
        #
        # Anchor whole-trajectory time-to-first-token here: `trajectory_scope`
        # is set-once, so a nested sub-agent inherits this (outermost) agent's
        # start. The streamed final answer's TTFT is then measured from this
        # point -- including every tool-calling round below.
        with trajectory_scope():
            trajectory, ctx = await self._begin_call(inputs, training, **kwargs)
            agent_messages = trajectory.get("messages")

            # Inject context messages at the front of the trajectory, read once
            # at construction. Skills are prepended first so that AGENTS.md
            # (prepended after) ends up as the very first message — keeping
            # declared project conventions at the top. Both are guarded against
            # re-injection so feeding a returned trajectory back in (e.g. in
            # interactive mode) doesn't stack duplicate copies each turn.
            if prepend_context_message(agent_messages, self.skills_message):
                trajectory.update({"messages": agent_messages})
            if prepend_context_message(agent_messages, self.agents_md_message):
                trajectory.update({"messages": agent_messages})

            if self.autonomous:
                return await self._run_autonomous(
                    trajectory, agent_messages, ctx, training
                )
            return await self._run_interactive(
                trajectory, agent_messages, ctx, training
            )

    async def _begin_call(self, inputs, training, **kwargs):
        """Build the working trajectory and an opaque per-call context (hook).

        The base agent needs no per-call state, so the context is an empty
        dict. Subclasses stash per-call objects in it (e.g. RLM puts its
        sandbox, native tools and submit holder there); the loop threads the
        context through `_native_tools`, `_dispatch_tool_calls`, `_final_result`
        and `_finish`. Extra keyword arguments forwarded from `call` are ignored
        by the base agent.
        """
        if is_chat_messages(inputs):
            trajectory = inputs
        else:
            trajectory = await ops.concat(
                inputs, ChatMessages(), name="trajectory_" + self.name
            )
        return trajectory, {}

    def _native_tools(self, ctx):
        """Tools advertised to the step generator each turn (hook)."""
        return list(self.tools.values())

    def _on_empty_generation(self, agent_messages, ctx):
        """Handle a turn where the generator returned nothing (hook)."""
        agent_messages.append(
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content="Something went wrong while trying to decide the next action.",
            ).get_json()
        )

    async def _on_no_tool_calls(self, tool_calls, agent_messages, ctx):
        """Handle an assistant turn carrying no tool calls (hook).

        Returns True to break the loop (the base agent's behavior), or False to
        keep iterating after, e.g., nudging the model.
        """
        return True

    async def _dispatch_tool_calls(self, native_tool_calls, agent_messages, ctx):
        """Execute the turn's tool calls and append their results (hook).

        The base agent runs every call concurrently. Returns True if the loop
        should terminate early (e.g. a submit signal); the base never does.
        """
        tasks = []
        tool_calls_ids = []
        for tool_call in native_tool_calls:
            function = tool_call.get("function") or {}
            tool_name = function.get("name")
            tools_arguments = function.get("arguments") or {}
            tool_calls_ids.append(tool_call.get("id"))
            tasks.append(self.tools[tool_name](**tools_arguments))

        tool_results = await asyncio.gather(*tasks, return_exceptions=True)
        for j, tool_result in enumerate(tool_results):
            tool_call_id = tool_calls_ids[j]
            if isinstance(tool_result, Exception):
                agent_messages.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        tool_call_id=tool_call_id,
                        content="error: %s" % str(tool_result),
                    ).get_json()
                )
            else:
                # Handle both JsonDataModel and raw dict results
                content = (
                    tool_result.get_json()
                    if hasattr(tool_result, "get_json")
                    else tool_result
                )
                agent_messages.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        tool_call_id=tool_call_id,
                        content=content,
                    ).get_json()
                )
        return False

    async def _final_result(self, trajectory, ctx, training):
        """Produce the final answer data model from the trajectory (hook)."""
        return await self.final_generator(trajectory)

    async def _wrap_final(self, final_result, agent_messages, training):
        """Wrap the final answer into the agent's return value (shared)."""
        if self.schema:
            # With schema: return the structured data model
            if self.return_inputs_with_trajectory:
                # Combine trajectory with structured output
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
        # Without schema: append the final ChatMessage to the trajectory.
        # Interactive streaming returns the lazy StreamingIterator for the
        # caller to consume; inside a batched loop the final generator has
        # already drained it to a concrete ChatMessage, so fall through to the
        # normal trajectory wrapping below (keeping the prediction scorable).
        if isinstance(final_result, StreamingIterator):
            return final_result
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

    async def _finish(self, trajectory, agent_messages, ctx, training):
        """Compute and wrap the final answer after the loop (hook seam)."""
        final_result = await self._final_result(trajectory, ctx, training)
        return await self._wrap_final(final_result, agent_messages, training)

    async def _run_autonomous(self, trajectory, agent_messages, ctx, training):
        return await self._run_loop(
            trajectory, agent_messages, ctx, training, max_steps=self.max_iterations
        )

    async def _run_loop(self, trajectory, agent_messages, ctx, training, *, max_steps):
        """Shared step loop: generate → dispatch tool calls → repeat → finish.

        Runs `max_steps` turns. Autonomous runs use `max_iterations`; subclasses
        whose interactive mode is a single pass of the same loop reuse it with
        `max_steps=1`.
        """
        for _ in range(max_steps):
            tool_calls = await self.tool_calls_generator(
                trajectory, tools=self._native_tools(ctx)
            )
            if not tool_calls:
                self._on_empty_generation(agent_messages, ctx)
                break

            native_tool_calls = tool_calls.get("tool_calls") or []
            if not native_tool_calls:
                if await self._on_no_tool_calls(tool_calls, agent_messages, ctx):
                    break
                trajectory.update({"messages": agent_messages})
                continue

            # The generator returned an assistant ChatMessage with native
            # `tool_calls`; append it as-is rather than rebuilding from scratch.
            agent_messages.append(tool_calls.get_json())
            terminated = await self._dispatch_tool_calls(
                native_tool_calls, agent_messages, ctx
            )
            trajectory.update({"messages": agent_messages})
            if terminated:
                break

        return await self._finish(trajectory, agent_messages, ctx, training)

    async def _run_interactive(self, trajectory, agent_messages, ctx, training):
        # Track new messages generated in this step
        new_messages = []

        if len(agent_messages) > 0:
            if agent_messages[-1].get("role") == ChatRole.ASSISTANT:
                tasks = []
                tool_calls_ids = []

                tool_calls = agent_messages[-1].get("tool_calls") or []
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    tool_name = function.get("name")
                    tools_arguments = function.get("arguments") or {}
                    tool_call_id = tool_call.get("id")
                    tool_calls_ids.append(tool_call_id)
                    tasks.append(self.tools[tool_name](**tools_arguments))

                tool_results = await asyncio.gather(*tasks, return_exceptions=True)
                for j, tool_result in enumerate(tool_results):
                    tool_call_id = tool_calls_ids[j]
                    if isinstance(tool_result, Exception):
                        tool_message = ChatMessage(
                            role=ChatRole.TOOL,
                            tool_call_id=tool_call_id,
                            content="error: %s" % str(tool_result),
                        )
                    else:
                        # Handle both JsonDataModel and raw dict results
                        content = (
                            tool_result.get_json()
                            if hasattr(tool_result, "get_json")
                            else tool_result
                        )
                        tool_message = ChatMessage(
                            role=ChatRole.TOOL,
                            tool_call_id=tool_call_id,
                            content=content,
                        )
                    agent_messages.append(tool_message.get_json())
                    new_messages.append(tool_message)

        trajectory.update({"messages": agent_messages})

        tool_calls = await self.tool_calls_generator(
            trajectory, tools=list(self.tools.values())
        )

        # If no tool calls, call final generator
        # without appending the empty tool calls message
        if not tool_calls or not tool_calls.get("tool_calls"):
            final_result = await self.final_generator(trajectory)
            # Interactive streaming hands the lazy iterator back; in a batched
            # loop the final generator has already drained it to a concrete
            # result, so fall through to the normal wrapping below.
            if isinstance(final_result, StreamingIterator):
                return final_result
            if self.schema:
                # Combine messages with structured output
                if self.return_inputs_with_trajectory:
                    validated_messages = ChatMessages(
                        messages=[ChatMessage(**msg) for msg in agent_messages]
                    )
                else:
                    validated_messages = ChatMessages(messages=new_messages)
                return await ops.concat(
                    JsonDataModel(
                        json=validated_messages.get_json(),
                        schema=ChatMessages.get_schema(),
                        name=self.name,
                    ),
                    final_result,
                    name=self.name,
                )
            else:
                # Append ChatMessage to messages
                if final_result:
                    if self.return_inputs_with_trajectory:
                        agent_messages.append(final_result.get_json())
                        validated_messages = ChatMessages(
                            messages=[ChatMessage(**msg) for msg in agent_messages]
                        )
                    else:
                        new_messages.append(ChatMessage(**final_result.get_json()))
                        validated_messages = ChatMessages(messages=new_messages)
                else:
                    if self.return_inputs_with_trajectory:
                        validated_messages = ChatMessages(
                            messages=[ChatMessage(**msg) for msg in agent_messages]
                        )
                    else:
                        validated_messages = ChatMessages(messages=new_messages)
                return JsonDataModel(
                    json=validated_messages.get_json(),
                    schema=ChatMessages.get_schema(),
                    name=self.name,
                )

        # The generator returned an assistant ChatMessage with native
        # tool_calls already in the synalinks shape; reuse it directly.
        assistant_message = ChatMessage(**tool_calls.get_json())
        agent_messages.append(assistant_message.get_json())
        new_messages.append(assistant_message)
        trajectory.update({"messages": agent_messages})

        if self.return_inputs_with_trajectory:
            # Convert dict messages to ChatMessage objects to avoid Pydantic warnings
            validated_messages = ChatMessages(
                messages=[ChatMessage(**msg) for msg in agent_messages]
            )
            return JsonDataModel(
                json=validated_messages.get_json(),
                schema=ChatMessages.get_schema(),
                name=self.name,
            )
        else:
            return JsonDataModel(
                json=ChatMessages(messages=new_messages).get_json(),
                schema=ChatMessages.get_schema(),
                name=self.name,
            )

    async def compute_output_spec(self, inputs, training=False, **kwargs):
        # `call()` accepts **kwargs, so the auto-build machinery
        # (Module._maybe_build) splats a synthetic `kwargs` entry in here when
        # tracing; mirror that signature so symbolic build doesn't fail with
        # "unexpected keyword argument 'kwargs'".
        if self.autonomous:
            _ = await self.tool_calls_generator(inputs, tools=list(self.tools.values()))
            if self.schema:
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
                else:
                    return await self.final_generator(inputs)
            else:
                # Without schema: return ChatMessages with final message appended
                _ = await self.final_generator(inputs)
                return SymbolicDataModel(
                    schema=ChatMessages.get_schema(),
                    name=self.name,
                )
        else:
            if not is_chat_messages(inputs):
                raise ValueError(
                    "In interactive mode, the FunctionCallingAgent "
                    "needs an ChatMessages-like data model as inputs"
                )

            _ = await self.tool_calls_generator(inputs, tools=list(self.tools.values()))

            # The output can be either the final generator output (when no tool calls)
            # or ChatMessages (when there are tool calls)
            # We use ChatMessages as the output spec since it's the common case
            return SymbolicDataModel(
                schema=ChatMessages.get_schema(),
                name=self.name,
            )

    def get_config(self):
        config = {
            "schema": self.schema,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "final_instructions": self.final_instructions,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "reasoning_effort": self.reasoning_effort,
            "use_chain_of_thought": self.use_chain_of_thought,
            "autonomous": self.autonomous,
            "max_iterations": self.max_iterations,
            "return_inputs_with_trajectory": self.return_inputs_with_trajectory,
            "streaming": self.streaming,
            "workdir": self.workdir,
            "skills": self.skills,
            "name": self.name,
            "description": self.description,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        # Only user-supplied tools are serialized; built-in tools are rebuilt
        # by `_get_builtin_tools` on reload (they may not be serializable, and
        # would otherwise be duplicated against the rebuilt built-ins).
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(tool)
                for tool in self.extra_tools
            ]
        }
        return {**config, **language_model_config, **tools_config}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        tools = [
            serialization_lib.deserialize_synalinks_object(tool)
            for tool in config.pop("tools", [])
        ]
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            language_model=language_model,
            tools=tools or None,
            **config,
        )
