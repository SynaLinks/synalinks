# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import copy
from typing import List, Dict, Any

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel, Field, JsonDataModel
from synalinks.src.modules.core.action import Action, GenericAction
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.saving.serialization_lib import deserialize_synalinks_object, serialize_synalinks_object
from synalinks.src.utils.nlp_utils import to_singular_property
from synalinks.src.utils.tool_utils import Tool, toolkit_to_static_prompt


class ToolChoice(DataModel):
    name: str
    purpose: str = Field(
        description="A clear, specific explanation of what the tool should accomplish."
    )


class ToolDecision(DataModel):
    reasoning: str = Field(
        description="A step-by-step analysis of the current state, what has been done, and what should be done next."
    )
    choices: List[ToolChoice] = Field(
        description="The array of tool choices to run in parallel with their specific purpose."
    )


class Toolkit(DataModel):
    toolkit: str = Field(description="The description of available toolkit")


class Purpose(DataModel):
    purpose: str = Field(description="The purpose of the chosen tool")


class ToolMessage(DataModel):    
    name: str
    inputs: Dict[str, Any] = Field(description="The inputs")
    outputs: Dict[str, Any] = Field(description="The outputs")


class ToolAction(DataModel):
    """A generic action with tool name and I/O"""

    action: ToolMessage = Field(description="An action already performed")


def dynamic_enum_on_nested_property(schema, property, labels, description=None):
    """Update a JSON schema with a dynamic enum on a nested property.
 
    Args:
        schema (dict): The schema to update.
        property (str): The nested, slash-separated property path
        labels (list): The array of enum options (strings).
        description (str, optional): The optional enum description

    Returns:
        dict: The updated JSON schema with the enum applied to the nested property.
    """
    schema = copy.deepcopy(schema)
    
    if schema.get("$defs"):
        schema = {"$defs": schema.pop("$defs"), **schema}
    else:
        schema = {"$defs": {}, **schema}

    property = property.split("/")
    klass = to_singular_property(property[-1]).title().replace("_", " ")
    
    definition = {
        "enum": labels,
        "title": klass,
        "type": "string",
    }
    
    if description:
        definition["description"] = description
    
    schema["$defs"].update({klass: definition})

    entity = schema["$defs"]
    
    for part in property[:-1]:
        if part == "items":
            entity = entity.setdefault("items", {})
        elif part == "properties":
            entity = entity.setdefault("properties", {})
        else:
            entity = entity.setdefault(part, {})
    
    entity[property[-1]] = {"$ref": f"#/$defs/{klass}"}
    
    return schema


def tool_action_from_generic_action_message(name: str, message: GenericAction, schema: Dict[str, Any]) -> JsonDataModel:
    """Convert a generic action message to a tool action message."""
    message = ToolMessage(name=name, **message.get("action"))
    message = ToolAction(action=message)

    return JsonDataModel(
        json=message.get_json(),
        schema=schema,
    )


def get_default_decision_toolkit(toolkit: List[Tool] = None):
    """The default toolkit prompt to make decisions in a ReAct agent."""
    toolkit = toolkit or []
    toolkit = toolkit_to_static_prompt(toolkit)

    toolkit_static_prompt = Toolkit(toolkit=toolkit)

    return toolkit_static_prompt


def get_default_decision_instructions() -> List[str]:
    """The default mandatory instructions to make decisions in a ReAct agent."""
    return [
        "Analyze the current state: What do you observe? What do you need to accomplish next? Before taking any action, carefully consider context and all available information.",
        "Reflect on prior steps: Review your previous actions and their outcomes to avoid unnecessary repetition.",
        "Reason methodically: Think step-by-step to determine the most appropriate choices with the available toolkit.",
        "Avoid unnecessary actions: If you already have enough information to complete the user task, return an empty choices array.",
        "Keep subgoals atomic: Each choice must focus on accomplishing exactly one specific task. Break down complex objectives into multiple separate choices rather than creating compound subgoals.",
        "Split complex steps across parallel choices: When facing a multi-part objective, create multiple choices using the same tool with different atomic subgoals that can execute simultaneously without dependencies.",
        "Write self-contained subgoals: Each subgoal must be completely self-explanatory without referencing context, other subgoals, or using pronouns. Include all necessary information explicitly within the subgoal description.",
        "Ensure parallel independence: Make tool choices that can execute concurrently without requiring results from each other. Avoid creating dependencies between parallel subgoals.",
    ]


@synalinks_export(["synalinks.modules.Agent", "synalinks.Agent"])
class Agent(Module):
    """
    ReAct agent as a directed acyclic graph that chooses at each step which tools to run.

    Note:
        - Each function must return a JSON object and be asynchrounous.

    References:
        - [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

    Example:
        ```python
        import asyncio

        import synalinks


        class Query(synalinks.DataModel):
            query: str = synalinks.Field(
                description="The user query",
            )


        class Answer(synalinks.DataModel):
            answer: str = synalinks.Field(
                description="The final answer to the query.",
            )
        

        async def main():
            async def websearch(query: str):
                \"""Perform a web search for the given query.
                
                Args:
                    query (str): The search query to perform.
                \"""
                return {
                    "results": [
                        f"Result 1 for {query}",
                        f"Result 2 for {query}",
                        f"Result 3 for {query}",
                    ],
                    "log": "Web search completed successfully."
                }

            language_model = synalinks.LanguageModel(model="ollama/mistral")

            x = synalinks.Input(data_model=Query)
            y = await synalinks.Agent(
                language_model=language_model,
                data_model=Answer,
                toolkit=[websearch],
                return_inputs_with_trajectory=True,
            )(x)

            program = synalinks.Program(
                inputs=x,
                outputs=y,
                name="navigator",
                description="A ReAct agent with websearch",
            )

            result = await program(
                Query(query=(
                    "What are the latest news on the import taxes Trump just imposed on the EU? "
                    "When you have found those... What is the sentiment of European and American people on social media? "
                    "How could this be a blessing in disguise for the EU reinassance?"
                    )
                )
            )

            print(result.prettify_json())

    
        if __name__ == "__main__":
            asyncio.run(main())
        ```

    Args:
        schema (dict): The JSON schema to use for the final answer.
            If not provided, it will use the `output_data_model` argument.
        data_model (DataModel | JsonDataModel | SymbolicDataModel): Optional.
            The data model to use for the final answer.
            If None provided, the Agent will return a ChatMessage-like data model.
        toolkit (list): The available toolkit of functions.
        language_model (LanguageModel): The language model to use, if provided
            it will ignore `decision_language_model` and `action_language_model` argument.
        decision_language_model (LanguageModel): The language model used for
            decision-making.
        action_language_model (LanguageModel): The language model used for actions.
        prompt_template (str): Optional. The jinja2 prompt template to use
            (See `Generator`).
        examples (list): A default list of examples for decision-making (See `Decision`).
        instructions (list): A default list of instructions for decision-making
            (See `Decision`).
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the decision prompt (Default to False) (see `Decision`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the decision prompt (Default to False) (see `Decision`).
        return_inputs_with_trajectory (bool): Optional. Whether or not to concatenate the
            inputs along with the agent trajectory to the outputs (Default to False).
        return_inputs (bool): Optional. Whether or not to concatenate the inputs
            to the outputs (Default to False).
        max_iterations (int): The maximum number of steps to perform.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        toolkit=None,
        language_model=None,
        decision_language_model=None,
        action_language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_inputs_with_trajectory=False,
        return_inputs=False,
        max_iterations=10,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )

        if schema:
            self.schema = schema
        elif data_model:
            self.schema = data_model.get_schema()
        else:
            raise ValueError(
                "You must set either `schema` or `data_model` arguments."
            )

        if language_model:
            self.decision_language_model = language_model
            self.action_language_model = language_model
        elif action_language_model and decision_language_model:
            self.decision_language_model = decision_language_model
            self.action_language_model = action_language_model
        else:
            raise ValueError(
                "You must set either `language_model` "
                " or both `action_language_model` and `decision_language_model` arguments."
            )

        self.prompt_template = prompt_template

        if not examples:
            examples = []

        self.examples = examples

        if not instructions:
            instructions = get_default_decision_instructions()

        self.instructions = instructions

        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
    
        if return_inputs and return_inputs_with_trajectory:
            raise ValueError(
                "You cannot set both `return_inputs` and "
                "`return_inputs_with_trajectory` arguments to true: choose only one."
            )

        self.return_inputs_with_trajectory = return_inputs_with_trajectory
        self.return_inputs = return_inputs

        assert max_iterations > 0, "The agent must perform at least one decision-making step."
        self.max_iterations = max_iterations

        toolkit = toolkit or []
        toolkit = [_ if isinstance(_, Tool) else Tool(_) for _ in toolkit]

        self.toolkit = toolkit

        self.actions = {
            _.name(): Action(
                fn=_._func,
                language_model=self.action_language_model,
                prompt_template=self.prompt_template,
                use_inputs_schema=self.use_inputs_schema,
                use_outputs_schema=self.use_outputs_schema,
                name=self.name + f"_action_{_.name()}",
            )
            for _ in self.toolkit
        }

        self.tool_action_schema = dynamic_enum_on_nested_property(
            ToolAction.get_schema(),
            "ToolMessage/properties/name",
            list(self.actions.keys()),
            description="The name of tool that was run."
        )
  
        decision_schema = dynamic_enum_on_nested_property(
            ToolDecision.get_schema(),
            "ToolChoice/properties/name",
            list(self.actions.keys()),
            description="The name of tool to run from available toolkit."
        )

        self.decision_maker = Generator(
            schema=decision_schema,
            language_model=self.decision_language_model,
            instructions=self.instructions,
            static_system_prompt=toolkit_to_static_prompt(self.toolkit),
            examples=self.examples,
            prompt_template=self.prompt_template,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            name=self.name + "_decision_maker",
        )

        self.response_maker = Generator(
            schema=self.schema,
            language_model=self.action_language_model,
            instructions=["Provide the final response, taking into account all the information gathered."],
            name=self.name + "_response_maker",
        )

    async def call(self, inputs, training=False):
        step = inputs

        for _ in range(self.max_iterations):  
            decision = await self.decision_maker(step, training=training)
            choices = decision.get("choices", [])

            step = await ops.concat(step, decision)

            if not choices:
                break

            futures = []

            for choice in choices:
                name = choice.get("name")
                purpose = choice.get("purpose")

                purpose = Purpose(purpose=purpose)
                purpose = JsonDataModel(
                    json=purpose.get_json(),
                    data_model=Purpose,
                )

                action = self.actions[name]

                futures.append(action(purpose, training=training))

            messages = await asyncio.gather(*futures)

            tool_message = None

            for choice, message in zip(choices, messages):
                name = choice.get("name")

                message = tool_action_from_generic_action_message(
                    name=name, message=message, schema=self.tool_action_schema,
                )

                if not tool_message:
                    tool_message = message
                else:
                    tool_message = await ops.concat(tool_message, message)

            step = await ops.concat(step, tool_message)

        response = await self.response_maker(step, training=training)

        if self.return_inputs_with_trajectory:
            response = await ops.concat(step, response)
            # FIXME
            # response.factorize()

        if self.return_inputs:
            response = await ops.concat(inputs, response)

        return response

    async def compute_output_spec(self, inputs, training=False):
        step = inputs

        for _ in range(self.max_iterations):
            decision_spec = await self.decision_maker.compute_output_spec(step)
            
            if self.actions:
                action_specs = []

                for action in self.actions.values():
                    purpose = Purpose(purpose="mock purpose for output spec computation")
                    purpose = JsonDataModel(
                        json=purpose.get_json(),
                        data_model=Purpose,
                    )

                    spec = await action.compute_output_spec(purpose)
                    action_specs.append(spec)
                
                if len(action_specs) == 1:
                    combined_spec = action_specs[0]
                else:
                    combined_spec = action_specs[0]

                    for i in range(1, len(action_specs)):
                        combined_spec = await ops.concat(
                            combined_spec, action_specs[i],
                        )

                step = await ops.concat(step, combined_spec)

        response = await self.response_maker.compute_output_spec(step)

        if self.return_inputs_with_trajectory:
            response = await ops.concat(step, response)

        if self.return_inputs:
            response = await ops.concat(inputs, response)

        return response

    def get_config(self):
        config = {
            "schema": self.schema,
            "toolkit": self.toolkit,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs_with_trajectory": self.return_inputs_with_trajectory,
            "return_inputs": self.return_inputs,
            "max_iterations": self.max_iterations,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }

        language_model_config = {}
        language_model_config["decision_language_model"] = serialize_synalinks_object(
            self.decision_language_model
        )
        language_model_config["action_language_model"] = serialize_synalinks_object(
            self.action_language_model
        )

        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            language_model=language_model,
            **config,
        )
