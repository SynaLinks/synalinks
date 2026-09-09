# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.backend.pydantic.metrics import Rating20
from synalinks.src.backend.pydantic.metrics import get_score_type
from synalinks.src.backend.pydantic.metrics import normalize_score
from synalinks.src.backend.pydantic.metrics import score_type_description
from synalinks.src.backend.pydantic.metrics import serialize_score_type
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.ttc.self_critique import CritiqueWithReward
from synalinks.src.modules.ttc.self_critique import _normalized_reward_schema
from synalinks.src.modules.ttc.self_critique import critique_with_reward_schema
from synalinks.src.programs import Program
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.saving import serialization_lib


def default_agent_judge_instructions(score_type=Rating20):
    """Return the default instructions of `AgentAsJudge` for a score scale.

    The instructions tell the judge to gather evidence with its tools before
    grading, and spell out the grading scale in plain words so the language
    model knows what the `reward` means even when the output schema is not
    included in the prompt.

    Args:
        score_type (type | str): The score scale (see `get_score_type`).
            Defaults to `Rating20`.

    Returns:
        (str): The instructions string.
    """
    return (
        "You are an impartial judge. Your task is to carefully examine the "
        "provided inputs and evaluate the prediction. Use the available tools to "
        "verify the prediction against reality (run code, inspect files, look "
        "things up) instead of trusting your first impression; when a gold "
        "reference is provided, compare the prediction against it. Once you have "
        "enough evidence, stop calling tools, write an elaborated critique of the "
        "prediction, then grade it with a reward. The reward is "
        f"{score_type_description(score_type)}."
    )


async def judge_with_agent(agent, score_type, inputs):
    """Run an agent judge on a `[y_true, y_pred]` pair and normalize its reward.

    Shared by the agent-based judge programs (`AgentAsJudgeProgram`,
    `RLMAsJudgeProgram`): the gold reference, when given, is prefixed with
    `gold` and concatenated with the prediction, the agent grades the result
    on the `score_type` scale, and the reward is rewritten to a 0.0..1.0 float.
    An empty prediction, or an agent that produces nothing, scores 0.0 without
    raising so training keeps going.

    Args:
        agent (Module): The judge agent; its output schema must carry a
            `critique` and a `reward` on the `score_type` scale.
        score_type (type): The score scale the agent grades on.
        inputs (list): The `[y_true, y_pred]` pair to grade.

    Returns:
        (JsonDataModel | SymbolicDataModel): The critique with a normalized reward.
    """
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("The inputs should be a list or tuple.")
    if len(inputs) != 2:
        raise ValueError("The inputs of the program should have a length of 2.")
    y_true, y_pred = inputs
    if not y_pred:
        return CritiqueWithReward(
            critique="Empty prediction: nothing to evaluate.",
            reward=0.0,
        ).to_json_data_model()
    if y_true:
        y_true = await ops.prefix(y_true, prefix="gold", name="gold_y_true")
        inputs = await ops.concat(y_true, y_pred, name="y_true_with_y_pred")
    else:
        inputs = y_pred
    outputs = await agent(inputs)
    if outputs is None:
        return CritiqueWithReward(
            critique="The judge produced no verdict.",
            reward=0.0,
        ).to_json_data_model()
    schema = _normalized_reward_schema(outputs.get_schema())
    if is_symbolic_data_model(outputs):
        return SymbolicDataModel(schema=schema, name=outputs.name)
    json = dict(outputs.get_json())
    if json.get("reward") is not None:
        json["reward"] = normalize_score(json["reward"], score_type)
    return JsonDataModel(json=json, schema=schema, name=outputs.name)


class AgentAsJudgeProgram(Program):
    """Evaluate the output of a program using a tool-calling agent.

    The judge is a `FunctionCallingAgent` whose final answer is a critique and
    a reward on the `score_type` scale; the reward is normalized to 0.0..1.0.

    Args:
        language_model (LanguageModel): The language model to use.
        tools (list): The tools the judge can call to gather evidence
            (see `FunctionCallingAgent`). At least one tool is required.
        prompt_template (str): The default jinja2 prompt template
            to use (see `FunctionCallingAgent`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (str): The default instructions for the tool-calling turns.
            Defaults to `default_agent_judge_instructions(score_type)`.
        final_instructions (str): Optional. The instructions for the final
            grading turn. Defaults to `instructions`.
        score_type (type | str): Optional. The scale the judge picks the reward
            from (see `SelfCritique`). Defaults to `Rating20`; the reward is
            normalized to 0.0..1.0.
        max_iterations (int): Optional. The maximum number of tool-calling
            turns before the judge must grade (Default to 5).
        use_chain_of_thought (bool): Optional. Whether the tool-calling turns
            think step by step before choosing tools (Default to False).
        workdir (str): Optional. The judge's working directory
            (see `FunctionCallingAgent`).
        skills (list): Optional. Agent Skills roots (see `FunctionCallingAgent`).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        tools=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        final_instructions=None,
        score_type=Rating20,
        max_iterations=5,
        use_chain_of_thought=False,
        workdir=None,
        skills=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.score_type = get_score_type(score_type or Rating20)
        if instructions is None:
            instructions = default_agent_judge_instructions(self.score_type)
        self.agent = FunctionCallingAgent(
            schema=critique_with_reward_schema(self.score_type),
            language_model=language_model,
            tools=tools,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            use_chain_of_thought=use_chain_of_thought,
            autonomous=True,
            return_inputs_with_trajectory=False,
            max_iterations=max_iterations,
            workdir=workdir,
            skills=skills,
            name="agent_" + self.name,
        )
        self.language_model = language_model
        self.tools = tools
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.final_instructions = final_instructions
        self.max_iterations = max_iterations
        self.use_chain_of_thought = use_chain_of_thought
        self.workdir = workdir
        self.skills = skills

    async def call(self, inputs):
        return await judge_with_agent(self.agent, self.score_type, inputs)

    def get_config(self):
        config = {
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "final_instructions": self.final_instructions,
            "score_type": serialize_score_type(self.score_type),
            "max_iterations": self.max_iterations,
            "use_chain_of_thought": self.use_chain_of_thought,
            "workdir": self.workdir,
            "skills": self.skills,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            )
        }
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(tool)
                for tool in self.tools or []
            ]
        }
        return {**language_model_config, **tools_config, **config}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        tools = [
            serialization_lib.deserialize_synalinks_object(tool)
            for tool in config.pop("tools", [])
        ]
        return cls(language_model=language_model, tools=tools or None, **config)


@synalinks_export(
    [
        "synalinks.AgentAsJudge",
        "synalinks.rewards.AgentAsJudge",
    ]
)
class AgentAsJudge(ProgramAsJudge):
    """Evaluate the output of a program using a tool-calling agent.

    Unlike `LMAsJudge`, which grades from a single read of the inputs, this
    judge is a `FunctionCallingAgent`: it can call the given tools (run the
    predicted code, read a file, query a database, fetch a page) to gather
    evidence, then writes a critique and grades the prediction. Use it when
    the correctness of an output cannot be judged from its text alone.

    Example:

    ```python
    import asyncio

    import numpy as np

    import synalinks


    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="The user query")


    class Answer(synalinks.DataModel):
        answer: str = synalinks.Field(description="The answer to the query")


    async def calculate(expression: str):
        \"\"\"Calculate the result of a mathematical expression.

        Use it to check any arithmetic in the prediction instead of computing
        it in your head.

        Args:
            expression (str): The expression to evaluate, such as '2 + 2'. It
                can contain numbers, operators (+, -, *, /), parentheses and
                spaces.
        \"\"\"
        if not all(char in "0123456789+-*/(). " for char in expression):
            return {"result": None, "log": "Error: invalid characters in expression"}
        return {
            "result": eval(expression, {"__builtins__": None}, {}),
            "log": "Successfully executed",
        }


    async def main():
        language_model = synalinks.LanguageModel(model="ollama/mistral")

        # The program to train: a plain generator answering math questions.
        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Generator(
            data_model=Answer,
            language_model=language_model,
        )(x0)
        program = synalinks.Program(inputs=x0, outputs=x1, name="math_qa")

        # The judge re-does the arithmetic with its tool before grading.
        program.compile(
            reward=synalinks.rewards.AgentAsJudge(
                language_model=language_model,
                tools=[synalinks.Tool(calculate)],
            ),
            optimizer=synalinks.optimizers.RandomFewShot(),
        )

        x_train = np.array(
            [
                Query(query="How much is 152648 + 485?"),
                Query(query="What is 12 * 12?"),
                Query(query="Compute (3 + 4) * 5."),
                Query(query="What is 1000 / 8?"),
            ],
            dtype="object",
        )
        y_train = np.array(
            [
                Answer(answer="153133"),
                Answer(answer="144"),
                Answer(answer="35"),
                Answer(answer="125"),
            ],
            dtype="object",
        )

        history = await program.fit(
            x=x_train,
            y=y_train,
            epochs=4,
            validation_split=0.25,
            callbacks=[
                synalinks.callbacks.EarlyStopping(
                    monitor="val_reward",
                    mode="max",
                    patience=2,
                ),
            ],
        )


    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        language_model (LanguageModel): The language model to use.
        tools (list): The tools the judge can call to gather evidence
            (see `FunctionCallingAgent`). At least one tool is required.
        prompt_template (str): The default jinja2 prompt template
            to use (see `FunctionCallingAgent`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (str): The default instructions for the tool-calling turns.
            Defaults to `default_agent_judge_instructions(score_type)`, which
            asks the judge to verify with its tools and spells out the scale.
        final_instructions (str): Optional. The instructions for the final
            grading turn. Defaults to `instructions`.
        score_type (type | str): Optional. The scale the judge picks the reward
            from: `synalinks.Rating20` (default), `synalinks.Score`,
            `synalinks.FineScore`, `synalinks.Rating`, `synalinks.Rating10`, any
            `Enum` whose members are `int` or `float`, or the name of one of them.
            The reward is always normalized to a float between 0.0 and 1.0.
        max_iterations (int): Optional. The maximum number of tool-calling
            turns before the judge must grade (Default to 5).
        use_chain_of_thought (bool): Optional. Whether the tool-calling turns
            think step by step before choosing tools (Default to False).
        workdir (str): Optional. The judge's working directory
            (see `FunctionCallingAgent`).
        skills (list): Optional. Agent Skills roots (see `FunctionCallingAgent`).
        reduction (str): Optional. The reward reduction (Default to `"mean"`).
        name (str): Optional. string name of the reward instance.
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the reward.
        in_mask_pattern (str): Optional. Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): Optional. Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
    """

    def __init__(
        self,
        language_model=None,
        tools=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        final_instructions=None,
        score_type=Rating20,
        max_iterations=5,
        use_chain_of_thought=False,
        workdir=None,
        skills=None,
        reduction="mean",
        name="agent_as_judge",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        program = AgentAsJudgeProgram(
            language_model=language_model,
            tools=tools,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            score_type=score_type,
            max_iterations=max_iterations,
            use_chain_of_thought=use_chain_of_thought,
            workdir=workdir,
            skills=skills,
        )
        super().__init__(
            program=program,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        program = serialization_lib.deserialize_synalinks_object(config.pop("program"))
        return cls(
            language_model=program.language_model,
            tools=program.tools,
            prompt_template=program.prompt_template,
            examples=program.examples,
            instructions=program.instructions,
            final_instructions=program.final_instructions,
            score_type=program.score_type,
            max_iterations=program.max_iterations,
            use_chain_of_thought=program.use_chain_of_thought,
            workdir=program.workdir,
            skills=program.skills,
            **config,
        )
