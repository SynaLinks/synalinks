# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.pydantic.metrics import Rating20
from synalinks.src.backend.pydantic.metrics import get_score_type
from synalinks.src.backend.pydantic.metrics import score_type_bounds
from synalinks.src.backend.pydantic.metrics import serialize_score_type
from synalinks.src.modules.agents.rlm_agent import RecursiveLanguageModelAgent
from synalinks.src.modules.agents.rlm_agent import get_default_instructions
from synalinks.src.modules.agents.rlm_agent import get_recursive_instructions
from synalinks.src.modules.ttc.self_critique import critique_with_reward_schema
from synalinks.src.programs import Program
from synalinks.src.rewards.agent_as_judge import judge_with_agent
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.object_registration import get_registered_name
from synalinks.src.saving.object_registration import get_registered_object


def rlm_reward_description(score_type=Rating20):
    """Return the LM-facing description of the `RLMAsJudge` reward scale.

    The judge computes its reward in Python, so unlike `LMAsJudge` it is not
    limited to the discrete members of `score_type`: any integer or float
    between the lowest and highest member is accepted.

    Args:
        score_type (type | str): The score scale (see `get_score_type`).
            Defaults to `Rating20`.

    Returns:
        (str): The description string.
    """
    lo, hi = score_type_bounds(score_type)
    return (
        f"a number (integer or float) between {lo} and {hi}, "
        f"{lo} being very bad and {hi} very good"
    )


def rlm_critique_schema(score_type=Rating20):
    """Build the `CritiqueWithReward` schema handed to the RLM judge.

    Same as `critique_with_reward_schema`, except the `reward` is a bounded
    number rather than an enum of the members of `score_type`, so a value
    computed inside the sandbox (a pass rate scaled to the range, say)
    passes `submit` validation whether it is an int or a float.

    Args:
        score_type (type | str): The score scale (see `get_score_type`).
            Defaults to `Rating20`.

    Returns:
        (dict): The JSON schema of the judge's final answer.
    """
    lo, hi = score_type_bounds(score_type)
    schema = critique_with_reward_schema(score_type)
    schema["properties"]["reward"] = {
        "title": "Reward",
        "type": "number",
        "minimum": lo,
        "maximum": hi,
        "description": (
            "The reward value corresponding to the critique "
            f"({rlm_reward_description(score_type)})"
        ),
    }
    return schema


def default_rlm_judge_instructions(
    score_type=Rating20,
    recursive=True,
    max_llm_calls=50,
):
    """Return the default instructions of `RLMAsJudge` for a score scale.

    The instructions are the `RecursiveLanguageModelAgent` defaults (how to
    drive the sandbox, and the `llm_query` helpers when `recursive` is set)
    followed by the judging task: where the prediction and the gold reference
    live in the `inputs` dict, how to verify with code, and what to `submit`.
    The grading scale is spelled out in plain words so the language model
    knows what the `reward` means.

    Args:
        score_type (type | str): The score scale (see `get_score_type`).
            Defaults to `Rating20`.
        recursive (bool): Whether the sub-LM helpers are available.
        max_llm_calls (int): The sub-LM call budget rendered in the
            recursive instructions.

    Returns:
        (str): The instructions string.
    """
    if recursive:
        base = get_recursive_instructions().replace(
            "{max_llm_calls}", str(max_llm_calls)
        )
    else:
        base = get_default_instructions()
    verify = (
        "execute the predicted code, re-run the computation, diff the "
        "prediction against the `gold_` fields, delegate semantic comparison "
        "to `llm_query`"
        if recursive
        else "execute the predicted code, re-run the computation, diff the "
        "prediction against the `gold_` fields"
    )
    return (
        base
        + "\n\n"
        + "You are an impartial judge. The `inputs` dict holds the prediction to "
        "grade and, when a reference is available, the expected values under "
        "keys prefixed with `gold_`. Use the sandbox to verify the prediction "
        f"against reality ({verify}) instead of trusting your first impression. "
        "Once you have enough evidence, call "
        '`submit(result={"critique": ..., "reward": ...})` with an elaborated '
        "critique of the prediction and a reward. The reward is "
        f"{rlm_reward_description(score_type)}."
    )


class RLMAsJudgeProgram(Program):
    """Evaluate the output of a program using a `RecursiveLanguageModelAgent`.

    The judge writes Python in a persistent sandbox to verify the prediction,
    then submits a critique and a reward. The reward is any integer or float
    within the bounds of `score_type`, so it can be computed in code; it is
    normalized to 0.0..1.0.

    Args:
        language_model (LanguageModel): The language model to use.
        sub_language_model (LanguageModel): Optional. The sub-LM behind
            `llm_query` (see `RecursiveLanguageModelAgent`). Defaults to
            `language_model`.
        tools (list): Optional. Extra `Tool` instances exposed inside the
            sandbox (see `RecursiveLanguageModelAgent`).
        native_tools (list): Optional. `Tool` instances the judge calls
            directly, alongside `run_python_code`.
        prompt_template (str): The default jinja2 prompt template
            to use (see `FunctionCallingAgent`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (str): The default instructions for the code turns.
            Defaults to `default_rlm_judge_instructions(...)`.
        final_instructions (str): Optional. The instructions for the fallback
            grading turn, used when the judge runs out of iterations without
            calling `submit`. Defaults to `instructions`.
        score_type (type | str): Optional. The scale bounding the reward
            (see `SelfCritique`). Defaults to `Rating20`; any integer or float
            within its bounds is accepted and normalized to 0.0..1.0.
        max_iterations (int): Optional. The maximum number of code turns
            before the fallback grading step (Default to 20).
        use_chain_of_thought (bool): Optional. Whether the code turns think
            step by step before writing a snippet (Default to False).
        timeout (int): Optional. Per-turn execution budget in seconds
            (Default to 60).
        recursive (bool): Optional. Whether to expose `llm_query` and
            `llm_query_batched` inside the sandbox (Default to True).
        max_llm_calls (int): Optional. The sub-LM call budget per evaluation
            (Default to 50).
        max_output_chars (int): Optional. Maximum characters of REPL output
            kept per turn (Default to 10_000).
        workdir (str): Optional. Host directory seeding the sandbox
            (see `RecursiveLanguageModelAgent`).
        skills (list): Optional. Agent Skills roots (see `FunctionCallingAgent`).
        sandbox (Sandbox): Optional. A pre-built sandbox reused across
            evaluations (see `RecursiveLanguageModelAgent`).
        sandbox_type (type): Optional. The `Sandbox` subclass built per
            evaluation when no `sandbox` is given (Default to `MirageSandbox`).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        sub_language_model=None,
        tools=None,
        native_tools=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        final_instructions=None,
        score_type=Rating20,
        max_iterations=20,
        use_chain_of_thought=False,
        timeout=60,
        recursive=True,
        max_llm_calls=50,
        max_output_chars=10_000,
        workdir=None,
        skills=None,
        sandbox=None,
        sandbox_type=None,
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
        self.instructions = instructions
        if instructions is None:
            instructions = default_rlm_judge_instructions(
                self.score_type,
                recursive=recursive,
                max_llm_calls=max_llm_calls,
            )
        self.agent = RecursiveLanguageModelAgent(
            schema=rlm_critique_schema(self.score_type),
            language_model=language_model,
            sub_language_model=sub_language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            use_chain_of_thought=use_chain_of_thought,
            tools=tools,
            native_tools=native_tools,
            autonomous=True,
            return_inputs_with_trajectory=False,
            max_iterations=max_iterations,
            timeout=timeout,
            recursive=recursive,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            workdir=workdir,
            skills=skills,
            sandbox=sandbox,
            sandbox_type=sandbox_type,
            name="agent_" + self.name,
        )
        self.language_model = language_model
        self.sub_language_model = sub_language_model
        self.tools = tools
        self.native_tools = native_tools
        self.prompt_template = prompt_template
        self.examples = examples
        self.final_instructions = final_instructions
        self.max_iterations = max_iterations
        self.use_chain_of_thought = use_chain_of_thought
        self.timeout = timeout
        self.recursive = recursive
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.workdir = workdir
        self.skills = skills
        self.sandbox = sandbox
        self.sandbox_type = self.agent.sandbox_type

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
            "timeout": self.timeout,
            "recursive": self.recursive,
            "max_llm_calls": self.max_llm_calls,
            "max_output_chars": self.max_output_chars,
            "workdir": self.workdir,
            "skills": self.skills,
            "sandbox_type": get_registered_name(self.sandbox_type),
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        models_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            ),
            "sub_language_model": (
                serialization_lib.serialize_synalinks_object(self.sub_language_model)
                if self.sub_language_model is not None
                else None
            ),
            "sandbox": (
                serialization_lib.serialize_synalinks_object(self.sandbox)
                if self.sandbox is not None
                else None
            ),
        }
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(tool)
                for tool in self.tools or []
            ],
            "native_tools": [
                serialization_lib.serialize_synalinks_object(tool)
                for tool in self.native_tools or []
            ],
        }
        return {**models_config, **tools_config, **config}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        for key in ("language_model", "sub_language_model", "sandbox"):
            value = config.pop(key, None)
            config[key] = (
                serialization_lib.deserialize_synalinks_object(value)
                if value is not None
                else None
            )
        for key in ("tools", "native_tools"):
            tools = [
                serialization_lib.deserialize_synalinks_object(tool)
                for tool in config.pop(key, [])
            ]
            config[key] = tools or None
        sandbox_type_name = config.pop("sandbox_type", None)
        config["sandbox_type"] = (
            get_registered_object(sandbox_type_name) if sandbox_type_name else None
        )
        return cls(**config)


@synalinks_export(
    [
        "synalinks.RLMAsJudge",
        "synalinks.rewards.RLMAsJudge",
    ]
)
class RLMAsJudge(ProgramAsJudge):
    """Evaluate the output of a program using a `RecursiveLanguageModelAgent`.

    Where `AgentAsJudge` calls a fixed set of tools, this judge writes Python
    in a persistent sandbox: the gold reference and the prediction are bound
    as the `inputs` dict, so it can execute the predicted code, recompute a
    result, diff long outputs field by field, or delegate semantic comparison
    of carved-out snippets to a sub-LM through `llm_query`. The prompt only
    sees a summary of the inputs, which keeps the judge usable on predictions
    too large to grade in one context window. It ends by calling
    `submit(result={"critique": ..., "reward": ...})`; if it runs out of
    iterations first, a final grading step formats the trajectory instead.

    Because the judge can compute its verdict in Python (a pass rate, a
    field-by-field match ratio), the reward is not restricted to the discrete
    members of `score_type`: any integer or float between its lowest and
    highest member is accepted, then normalized to 0.0..1.0.

    Example:

    ```python
    import asyncio

    import numpy as np

    import synalinks


    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="The user query")


    class Answer(synalinks.DataModel):
        answer: str = synalinks.Field(description="The answer to the query")


    async def main():
        language_model = synalinks.LanguageModel(model="ollama/mistral")

        # The program to train: a plain generator answering math questions.
        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Generator(
            data_model=Answer,
            language_model=language_model,
        )(x0)
        program = synalinks.Program(inputs=x0, outputs=x1, name="math_qa")

        # The judge re-does the arithmetic in its sandbox before grading.
        program.compile(
            reward=synalinks.rewards.RLMAsJudge(
                language_model=language_model,
                # The sub-LM helpers are not needed to check arithmetic.
                recursive=False,
                max_iterations=5,
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
        sub_language_model (LanguageModel): Optional. The sub-LM behind
            `llm_query` (see `RecursiveLanguageModelAgent`). Defaults to
            `language_model`.
        tools (list): Optional. Extra `Tool` instances exposed inside the
            sandbox as plain functions (see `RecursiveLanguageModelAgent`).
        native_tools (list): Optional. `Tool` instances the judge calls
            directly, alongside `run_python_code`.
        prompt_template (str): The default jinja2 prompt template
            to use (see `FunctionCallingAgent`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (str): The default instructions for the code turns.
            Defaults to `default_rlm_judge_instructions(...)`, which combines
            the agent's sandbox instructions with the judging task and spells
            out the scale.
        final_instructions (str): Optional. The instructions for the fallback
            grading turn, used when the judge runs out of iterations without
            calling `submit`. Defaults to `instructions`.
        score_type (type | str): Optional. The scale bounding the reward:
            `synalinks.Rating20` (default), `synalinks.Score`,
            `synalinks.FineScore`, `synalinks.Rating`, `synalinks.Rating10`, any
            `Enum` whose members are `int` or `float`, or the name of one of them.
            Any integer or float between the lowest and highest member is
            accepted; the reward is always normalized to a float between 0.0
            and 1.0.
        max_iterations (int): Optional. The maximum number of code turns
            before the fallback grading step (Default to 20).
        use_chain_of_thought (bool): Optional. Whether the code turns think
            step by step before writing a snippet (Default to False).
        timeout (int): Optional. Per-turn execution budget in seconds
            (Default to 60).
        recursive (bool): Optional. Whether to expose `llm_query` and
            `llm_query_batched` inside the sandbox (Default to True).
        max_llm_calls (int): Optional. The sub-LM call budget per evaluation
            (Default to 50).
        max_output_chars (int): Optional. Maximum characters of REPL output
            kept per turn (Default to 10_000).
        workdir (str): Optional. Host directory seeding the sandbox
            (see `RecursiveLanguageModelAgent`).
        skills (list): Optional. Agent Skills roots (see `FunctionCallingAgent`).
        sandbox (Sandbox): Optional. A pre-built sandbox reused across
            evaluations (see `RecursiveLanguageModelAgent`).
        sandbox_type (type): Optional. The `Sandbox` subclass built per
            evaluation when no `sandbox` is given (Default to `MirageSandbox`).
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
        sub_language_model=None,
        tools=None,
        native_tools=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        final_instructions=None,
        score_type=Rating20,
        max_iterations=20,
        use_chain_of_thought=False,
        timeout=60,
        recursive=True,
        max_llm_calls=50,
        max_output_chars=10_000,
        workdir=None,
        skills=None,
        sandbox=None,
        sandbox_type=None,
        reduction="mean",
        name="rlm_as_judge",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        program = RLMAsJudgeProgram(
            language_model=language_model,
            sub_language_model=sub_language_model,
            tools=tools,
            native_tools=native_tools,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            score_type=score_type,
            max_iterations=max_iterations,
            use_chain_of_thought=use_chain_of_thought,
            timeout=timeout,
            recursive=recursive,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            workdir=workdir,
            skills=skills,
            sandbox=sandbox,
            sandbox_type=sandbox_type,
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
            sub_language_model=program.sub_language_model,
            tools=program.tools,
            native_tools=program.native_tools,
            prompt_template=program.prompt_template,
            examples=program.examples,
            instructions=program.instructions,
            final_instructions=program.final_instructions,
            score_type=program.score_type,
            max_iterations=program.max_iterations,
            use_chain_of_thought=program.use_chain_of_thought,
            timeout=program.timeout,
            recursive=program.recursive,
            max_llm_calls=program.max_llm_calls,
            max_output_chars=program.max_output_chars,
            workdir=program.workdir,
            skills=program.skills,
            sandbox=program.sandbox,
            sandbox_type=program.sandbox_type,
            **config,
        )
