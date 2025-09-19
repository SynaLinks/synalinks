# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import uuid

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import ToolCall
from synalinks.src.backend import is_chat_messages
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_tool_calls
from synalinks.src.backend.common.json_utils import out_mask_json
from synalinks.src.modules.module import Module
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.saving import serialization_lib


def get_default_instructions():
    """The default parallel function calling agent instructions."""
    return """
Think step by step: Use the thinking field to elaborate what you observe and what do you need to accomplish next.
Reflect on prior steps: Review your previous actions and their outcomes to avoid unnecessary repetition.
Avoid unnecessary actions: If you already have enough information to complete the user task, return an empty tool calls array.
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

    In *autonomous* mode, the agent accept **any kind of data model input** and perform a final inference to
    eventually format its final answer if a `data_model` or `schema` is provided.

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
                    "name": "calculate",
                    "arguments": {
                        "expression": "152648 + 485"
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
                "content": "The user has asked for a simple addition calculation. The assistant used the 'calculate' tool to perform this task, and the result was returned successfully.",
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

    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        tools=None,
        autonomous=True,
        return_inputs_with_trajectory=True,
        max_iterations=5,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        self.prompt_template = prompt_template

        if not instructions:
            instructions = get_default_instructions()
        self.instructions = instructions

        self.examples = examples
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.language_model = language_model

        self.tools = {}
        if not tools:
            raise ValueError("You must set the `tools` argument")
        for tool in tools:
            self.tools[tool.name] = tool
        tool_calls_schema = dynamic_tool_calls(tools=tools)

        self.autonomous = autonomous
        self.return_inputs_with_trajectory = return_inputs_with_trajectory
        self.max_iterations = max_iterations

        self.tool_calls_generator = ChainOfThought(
            schema=tool_calls_schema,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            language_model=self.language_model,
            name=self.name + "_tool_calls_generator",
        )

        if self.schema and self.autonomous:
            self.final_generator = ChainOfThought(
                schema=self.schema,
                language_model=self.language_model,
                instructions=self.instructions,
                return_inputs=self.return_inputs_with_trajectory,
                name=self.name + "_final_generator",
            )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        if self.autonomous:
            if not is_chat_messages(inputs):
                trajectory = await ops.concat(
                    inputs,
                    ChatMessages(),
                    name=self.name + "_trajectory",
                )
            else:
                trajectory = inputs
        else:
            if not is_chat_messages(inputs):
                raise ValueError(
                    "In interactive mode, the FunctionCallingAgent "
                    "needs an ChatMessages-like data model as inputs"
                )
            trajectory = inputs

        agent_messages = trajectory.get("messages")

        if self.autonomous:
            for i in range(self.max_iterations):
                tool_calls = await self.tool_calls_generator(trajectory)

                if not tool_calls:
                    assistant_message = ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content="Something went wrong while trying to decide the next action.",
                    )
                    agent_messages.append(assistant_message.get_json())
                    break

                assistant_message = ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=tool_calls.get("thinking", ""),
                )

                if not tool_calls.get("tool_calls"):
                    agent_messages.append(assistant_message.get_json())
                    break

                tasks = []
                tool_calls_ids = []

                for tool_call in tool_calls.get("tool_calls"):
                    tool_name = tool_call.get("tool_name")
                    tools_arguments = out_mask_json(tool_call, mask=["tool_name"])
                    tool_call_id = str(uuid.uuid4())
                    tool_calls_ids.append(tool_call_id)
                    assistant_message.tool_calls.append(
                        ToolCall(
                            id=tool_call_id,
                            name=tool_name,
                            arguments=tools_arguments,
                        )
                    )
                    tasks.append(self.tools[tool_name](**tools_arguments))

                agent_messages.append(assistant_message.get_json())

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
                        agent_messages.append(
                            ChatMessage(
                                role=ChatRole.TOOL,
                                tool_call_id=tool_call_id,
                                content=tool_result,
                            ).get_json()
                        )

                trajectory.update({"messages": agent_messages})
            if self.schema:
                return await self.final_generator(trajectory)
            else:
                return trajectory
        else:
            if len(agent_messages) > 0:
                if agent_messages[-1].get("role") == ChatRole.ASSISTANT:
                    tasks = []
                    tool_calls_ids = []

                    tool_calls = agent_messages[-1].get("tool_calls")
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name")
                        tools_arguments = tool_call.get("arguments")
                        tool_call_id = tool_call.get("id")
                        tool_calls_ids.append(tool_call_id)
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
                            agent_messages.append(
                                ChatMessage(
                                    role=ChatRole.TOOL,
                                    tool_call_id=tool_call_id,
                                    content=tool_result,
                                ).get_json()
                            )

            trajectory.update({"messages": agent_messages})

            tool_calls = await self.tool_calls_generator(trajectory)

            assistant_message = ChatMessage(
                role=ChatRole.ASSISTANT,
                content=tool_calls.get("thinking", ""),
                tool_calls=[],
            )

            for tool_call in tool_calls.get("tool_calls", []):
                tool_name = tool_call.get("tool_name")
                tools_arguments = out_mask_json(tool_call, mask=["tool_name"])
                tool_call_id = str(uuid.uuid4())

                assistant_message.tool_calls.append(
                    ToolCall(
                        id=tool_call_id,
                        name=tool_name,
                        arguments=tools_arguments,
                    )
                )

            agent_messages.append(assistant_message.get_json())
            trajectory.update({"messages": agent_messages})

            if self.return_inputs_with_trajectory:
                return JsonDataModel(
                    json=trajectory.get_json(),
                    schema=ChatMessages.get_schema(),
                    name=self.name,
                )
            else:
                return JsonDataModel(
                    json=assistant_message.get_json(),
                    schema=ChatMessage.get_schema(),
                    name=self.name,
                )

    async def compute_output_spec(self, inputs, training=False):
        if self.autonomous:
            _ = await self.tool_calls_generator(inputs)
            if self.schema:
                return await self.final_generator(inputs)
            else:
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

            _ = await self.tool_calls_generator(inputs)

            if self.return_inputs_with_trajectory:
                return SymbolicDataModel(
                    schema=ChatMessages.get_schema(),
                    name=self.name,
                )
            else:
                return SymbolicDataModel(
                    schema=ChatMessage.get_schema(),
                    name=self.name,
                )

    def get_config(self):
        config = {
            "schema": self.schema,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "autonomous": self.autonomous,
            "max_iterations": self.max_iterations,
            "return_inputs_with_trajectory": self.return_inputs_with_trajectory,
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
                serialization_lib.serialize_synalinks_object(tool)
                for tool in self.tools.values()
            ]
        }
        return {**config, **language_model_config, **tools_config}

    @classmethod
    def from_config(cls, config):
        tools = [
            serialization_lib.deserialize_synalinks_object(tool)
            for tool in config.pop("tools")
        ]
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            language_model=language_model,
            tools=tools,
            **config,
        )
