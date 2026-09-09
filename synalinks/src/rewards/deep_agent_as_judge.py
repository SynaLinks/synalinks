# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.backend.pydantic.metrics import Rating20
from synalinks.src.backend.pydantic.metrics import get_score_type
from synalinks.src.backend.pydantic.metrics import score_type_description
from synalinks.src.backend.pydantic.metrics import serialize_score_type
from synalinks.src.modules.agents.deep_agent import DeepAgent
from synalinks.src.modules.agents.deep_agent import get_default_instructions
from synalinks.src.modules.agents.utils.agents_utils import resolve_workdir
from synalinks.src.modules.ttc.self_critique import critique_with_reward_schema
from synalinks.src.programs import Program
from synalinks.src.rewards.agent_as_judge import judge_with_agent
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.saving import serialization_lib


def default_deep_agent_judge_instructions(score_type=Rating20, workdir=None):
    """Return the default instructions of `DeepAgentAsJudge` for a score scale.

    The instructions are the `DeepAgent` defaults (the file and shell tools
    and how to use them) followed by the judging task: where the prediction
    and the gold reference live, how to verify with the tools, and what to
    answer. The grading scale is spelled out in plain words so the language
    model knows what the `reward` means.

    Args:
        score_type (type | str): The score scale (see `get_score_type`).
            Defaults to `Rating20`.
        workdir (str): Optional. The judge's working directory, rendered in
            the tool plan.

    Returns:
        (str): The instructions string.
    """
    return (
        get_default_instructions(resolve_workdir(workdir))
        + "\n\n"
        + "You are an impartial judge. The `InputsSummary` describes the "
        "prediction to grade and, when a reference is available, the expected "
        "values under keys prefixed with `gold_`; the full values are in the "
        "JSON file named in its `inputs_file` field. Use the tools to verify the "
        "prediction against reality (read the inputs file, run the predicted "
        "code or the project's tests with `run_bash`, diff the prediction "
        "against the `gold_` fields) instead of trusting your first impression. "
        "Once you have enough evidence, stop calling tools, write an elaborated "
        "critique of the prediction, then grade it with a reward. The reward is "
        f"{score_type_description(score_type)}."
    )


class DeepAgentAsJudgeProgram(Program):
    """Evaluate the output of a program using a `DeepAgent`.

    The judge works in a sandboxed copy of `workdir` with file and shell
    tools; its final answer is a critique and a reward on the `score_type`
    scale, normalized to 0.0..1.0.

    Args:
        language_model (LanguageModel): The language model to use.
        sub_language_model (LanguageModel): Optional. The language model
            driving spawned subagents (see `DeepAgent`). Defaults to
            `language_model`.
        tools (list): Optional. Extra `Tool` instances exposed alongside the
            built-in file and shell tools (see `DeepAgent`).
        prompt_template (str): The default jinja2 prompt template
            to use (see `FunctionCallingAgent`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (str): The default instructions for the tool-calling turns.
            Defaults to `default_deep_agent_judge_instructions(...)`.
        final_instructions (str): Optional. The instructions for the final
            grading turn. Defaults to `instructions`.
        score_type (type | str): Optional. The scale the judge picks the reward
            from (see `SelfCritique`). Defaults to `Rating20`; the reward is
            normalized to 0.0..1.0.
        max_iterations (int): Optional. The maximum number of tool-calling
            turns before the judge must grade (Default to 10).
        use_chain_of_thought (bool): Optional. Whether the tool-calling turns
            think step by step before choosing tools (Default to False).
        timeout (float): Optional. Per-command budget in seconds for
            `run_bash` (Default to 30).
        workdir (str): Optional. Host directory whose files seed the sandbox
            (see `DeepAgent`). Its tests and sources are what the judge runs
            the prediction against.
        skills (list): Optional. Agent Skills roots (see `FunctionCallingAgent`).
        sandbox (Sandbox): Optional. A ready-made sandbox to grade in instead
            of one built from `workdir` (see `DeepAgent`).
        max_subagent_depth (int): Optional. Lets the judge spawn subagents on
            forks of its filesystem (see `DeepAgent`). Defaults to 0.
        reset_sandbox (bool): Optional. Whether to wipe the sandbox back to
            the `workdir` seed before each evaluation, so samples are graded
            independently. Defaults to True when the judge builds its own
            sandbox and to False when a `sandbox` is supplied.
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        sub_language_model=None,
        tools=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        final_instructions=None,
        score_type=Rating20,
        max_iterations=10,
        use_chain_of_thought=False,
        timeout=30.0,
        workdir=None,
        skills=None,
        sandbox=None,
        max_subagent_depth=0,
        reset_sandbox=None,
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
            instructions = default_deep_agent_judge_instructions(
                self.score_type, workdir=workdir
            )
        self.agent = DeepAgent(
            schema=critique_with_reward_schema(self.score_type),
            language_model=language_model,
            sub_language_model=sub_language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            use_chain_of_thought=use_chain_of_thought,
            tools=tools,
            autonomous=True,
            return_inputs_with_trajectory=False,
            max_iterations=max_iterations,
            timeout=timeout,
            workdir=workdir,
            skills=skills,
            sandbox=sandbox,
            max_subagent_depth=max_subagent_depth,
            name="agent_" + self.name,
        )
        self.language_model = language_model
        self.sub_language_model = sub_language_model
        self.tools = tools
        self.prompt_template = prompt_template
        self.examples = examples
        self.final_instructions = final_instructions
        self.max_iterations = max_iterations
        self.use_chain_of_thought = use_chain_of_thought
        self.timeout = timeout
        self.workdir = workdir
        self.skills = skills
        self.sandbox = sandbox
        self.max_subagent_depth = max_subagent_depth
        if reset_sandbox is None:
            reset_sandbox = sandbox is None
        self.reset_sandbox = reset_sandbox

    async def call(self, inputs):
        y_pred = (
            inputs[1] if isinstance(inputs, (list, tuple)) and len(inputs) == 2 else None
        )
        # A fresh workspace per graded sample; symbolic builds never run tools.
        if self.reset_sandbox and y_pred and not is_symbolic_data_model(y_pred):
            self.agent.sandbox.reset()
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
            "workdir": self.workdir,
            "skills": self.skills,
            "max_subagent_depth": self.max_subagent_depth,
            "reset_sandbox": self.reset_sandbox,
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
            ]
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
        tools = [
            serialization_lib.deserialize_synalinks_object(tool)
            for tool in config.pop("tools", [])
        ]
        return cls(tools=tools or None, **config)


@synalinks_export(
    [
        "synalinks.DeepAgentAsJudge",
        "synalinks.rewards.DeepAgentAsJudge",
    ]
)
class DeepAgentAsJudge(ProgramAsJudge):
    """Evaluate the output of a program using a `DeepAgent`.

    The judge is a coding agent working in a sandboxed, copy-on-write copy
    of `workdir`, with `read_file`, `list_files`, `search_files`,
    `write_file`, `edit_file` and `run_bash`. The gold reference and the
    prediction are written to a JSON file in that sandbox and the prompt only
    carries a summary of them, so the judge can apply a predicted patch, run
    the project's test suite, execute a generated script, or diff long
    outputs before writing its critique and grading. Nothing it does touches
    the real `workdir`. By default the sandbox is wiped back to the `workdir`
    seed before each evaluation, so samples are graded independently.

    Example:

    ```python
    import asyncio

    import numpy as np

    import synalinks


    class Task(synalinks.DataModel):
        task: str = synalinks.Field(description="The coding task to solve")


    class Solution(synalinks.DataModel):
        code: str = synalinks.Field(description="The Python code solving the task")


    async def main():
        language_model = synalinks.LanguageModel(model="ollama/mistral")

        # The program to train: a plain generator writing Python snippets.
        x0 = synalinks.Input(data_model=Task)
        x1 = await synalinks.Generator(
            data_model=Solution,
            language_model=language_model,
        )(x0)
        program = synalinks.Program(inputs=x0, outputs=x1, name="coder")

        # The judge writes the predicted code to a file, runs it in its
        # sandbox and compares the output against the gold reference.
        program.compile(
            reward=synalinks.rewards.DeepAgentAsJudge(
                language_model=language_model,
                max_iterations=5,
                # Optional: grade on a 1..5 integer scale instead of the
                # default 1..20 `Rating20`. The reward is normalized back
                # to 0.0..1.0 automatically.
                score_type=synalinks.Rating,
            ),
            optimizer=synalinks.optimizers.RandomFewShot(),
        )

        x_train = np.array(
            [
                Task(task="Print the sum of 152648 and 485."),
                Task(task="Print the square of 12."),
                Task(task="Print (3 + 4) * 5."),
                Task(task="Print 1000 divided by 8."),
            ],
            dtype="object",
        )
        y_train = np.array(
            [
                Solution(code="print(152648 + 485)"),
                Solution(code="print(12 ** 2)"),
                Solution(code="print((3 + 4) * 5)"),
                Solution(code="print(1000 / 8)"),
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
        sub_language_model (LanguageModel): Optional. The language model
            driving spawned subagents (see `DeepAgent`). Defaults to
            `language_model`.
        tools (list): Optional. Extra `Tool` instances exposed alongside the
            built-in file and shell tools (see `DeepAgent`).
        prompt_template (str): The default jinja2 prompt template
            to use (see `FunctionCallingAgent`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (str): The default instructions for the tool-calling turns.
            Defaults to `default_deep_agent_judge_instructions(...)`, which
            combines the agent's tool plan with the judging task and spells
            out the scale.
        final_instructions (str): Optional. The instructions for the final
            grading turn. Defaults to `instructions`.
        score_type (type | str): Optional. The scale the judge picks the reward
            from: `synalinks.Rating20` (default), `synalinks.Score`,
            `synalinks.FineScore`, `synalinks.Rating`, `synalinks.Rating10`, any
            `Enum` whose members are `int` or `float`, or the name of one of them.
            The reward is always normalized to a float between 0.0 and 1.0.
        max_iterations (int): Optional. The maximum number of tool-calling
            turns before the judge must grade (Default to 10).
        use_chain_of_thought (bool): Optional. Whether the tool-calling turns
            think step by step before choosing tools (Default to False).
        timeout (float): Optional. Per-command budget in seconds for
            `run_bash` (Default to 30).
        workdir (str): Optional. Host directory whose files seed the sandbox
            (see `DeepAgent`). Its tests and sources are what the judge runs
            the prediction against.
        skills (list): Optional. Agent Skills roots (see `FunctionCallingAgent`).
        sandbox (Sandbox): Optional. A ready-made sandbox to grade in instead
            of one built from `workdir` (see `DeepAgent`).
        max_subagent_depth (int): Optional. Lets the judge spawn subagents on
            forks of its filesystem (see `DeepAgent`). Defaults to 0.
        reset_sandbox (bool): Optional. Whether to wipe the sandbox back to
            the `workdir` seed before each evaluation, so samples are graded
            independently. Defaults to True when the judge builds its own
            sandbox and to False when a `sandbox` is supplied.
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
        prompt_template=None,
        examples=None,
        instructions=None,
        final_instructions=None,
        score_type=Rating20,
        max_iterations=10,
        use_chain_of_thought=False,
        timeout=30.0,
        workdir=None,
        skills=None,
        sandbox=None,
        max_subagent_depth=0,
        reset_sandbox=None,
        reduction="mean",
        name="deep_agent_as_judge",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        program = DeepAgentAsJudgeProgram(
            language_model=language_model,
            sub_language_model=sub_language_model,
            tools=tools,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            score_type=score_type,
            max_iterations=max_iterations,
            use_chain_of_thought=use_chain_of_thought,
            timeout=timeout,
            workdir=workdir,
            skills=skills,
            sandbox=sandbox,
            max_subagent_depth=max_subagent_depth,
            reset_sandbox=reset_sandbox,
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
            prompt_template=program.prompt_template,
            examples=program.examples,
            instructions=program.instructions,
            final_instructions=program.final_instructions,
            score_type=program.score_type,
            max_iterations=program.max_iterations,
            use_chain_of_thought=program.use_chain_of_thought,
            timeout=program.timeout,
            workdir=program.workdir,
            skills=program.skills,
            sandbox=program.sandbox,
            max_subagent_depth=program.max_subagent_depth,
            reset_sandbox=program.reset_sandbox,
            **config,
        )
