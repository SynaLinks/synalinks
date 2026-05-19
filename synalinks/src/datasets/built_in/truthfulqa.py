# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.dataset import split_train_test
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class TruthfulQAQuestion(DataModel):
    question: str = Field(
        description="The question and lettered candidate answers",
    )


class TruthfulQAAnswer(DataModel):
    answer: str = Field(
        description="The letter of the truthful answer",
    )


# Choice count varies, so we don't use a Literal here. We render up to 26
# choices via the alphabet; in practice TruthfulQA tops out at ~13.
_INPUT_TEMPLATE = (
    r'{%- set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}'
    r"{%- set lines = [] -%}"
    r"{%- for i in range(mc1_targets.choices | length) -%}"
    r'{%- set _ = lines.append(letters[i] ~ ") " ~ mc1_targets.choices[i]) -%}'
    r"{%- endfor -%}"
    r'{"question": {{ (question ~ "\n" ~ lines | join("\n")) | tojson }}}'
)
# `mc1_targets.labels` is a one-hot list with a single 1 at the gold index.
_OUTPUT_TEMPLATE = (
    r'{%- set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}'
    r'{"answer": {{ letters[mc1_targets.labels.index(1)] | tojson }}}'
)


@synalinks_export("synalinks.datasets.truthfulqa.get_input_data_model")
def get_input_data_model():
    """Returns TruthfulQA (MC1) input data model."""
    return TruthfulQAQuestion


@synalinks_export("synalinks.datasets.truthfulqa.get_output_data_model")
def get_output_data_model():
    """Returns TruthfulQA (MC1) output data model."""
    return TruthfulQAAnswer


@synalinks_export("synalinks.datasets.truthfulqa.load_data")
def load_data(validation_split=0.2):
    """
    Load TruthfulQA (MC1, ``multiple_choice`` config).

    HF ships only a single ``validation`` split for the MC1 task, so we
    deterministically split it into train / test.

    Args:
        validation_split (float): Fraction held out for evaluation
            (default ``0.2``).

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x, y = load_split(
        path="truthful_qa",
        name="multiple_choice",
        split="validation",
        input_data_model=TruthfulQAQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=TruthfulQAAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )
    return split_train_test(x, y, validation_split=validation_split)


@synalinks_export("synalinks.datasets.truthfulqa.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="validation"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="truthful_qa",
        name="multiple_choice",
        split=split,
        streaming=True,
        input_data_model=TruthfulQAQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=TruthfulQAAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
