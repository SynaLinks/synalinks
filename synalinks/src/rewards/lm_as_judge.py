# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.modules import SelfCritique
from synalinks.src.programs import Program
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.saving import serialization_lib


class LMAsJudgeProgram(Program):
    """Evaluate the output of a program using a `LanguageModel`.

    Args:
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (list): The default instructions to use (see `Generator`).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.critique = SelfCritique(
            language_model=language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            name="self_critique_" + self.name,
        )
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions

    async def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("The inputs should be a list or tuple.")
        if len(inputs) != 2:
            raise ValueError("The inputs of the program should have a length of 2.")
        y_true = inputs[0]
        y_pred = inputs[1]
        if y_true:
            y_true = await ops.prefix(
                y_true,
                prefix="gold",
                name="gold_y_true",
            )
            return await self.critique(
                await ops.concat(
                    y_true,
                    y_pred,
                    name="y_true_with_y_pred",
                )
            )
        else:
            return await self.critique(y_pred)

    def get_config(self):
        config = {
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            )
        }
        return {**language_model_config, **config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=language_model, **config)


@synalinks_export(
    [
        "synalinks.LMAsJudge",
        "synalinks.rewards.LMAsJudge",
    ]
)
class LMAsJudge(ProgramAsJudge):
    """Evaluate the output of a program using a `LanguageModel`.

    Example:

    ```python

    async def main():
        # ... program definition

        program.compile(
            reward=synalinks.rewards.LMAsJudge(
                language_model=language_model,
            )
            optimizer=synalinks.optimizers.RandomFewShot(),
        )

        history = await program.fit(...)

    ```

    Args:
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        instructions (list): The default instructions to use (see `Generator`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        name (str): Optional. string name of the reward instance.
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the reward.
    """

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        name="lm_as_judge",
        in_mask=None,
        out_mask=None,
    ):
        program = LMAsJudgeProgram(
            language_model=language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
        )
        super().__init__(
            program=program,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
        )
