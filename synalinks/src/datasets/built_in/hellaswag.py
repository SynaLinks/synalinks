# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Literal

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class HellaSwagQuestion(DataModel):
    question: str = Field(
        description="The context plus the four lettered endings",
    )


class HellaSwagAnswer(DataModel):
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The most plausible continuation letter",
    )


_INPUT_TEMPLATE = (
    r'{"question": {{ (ctx ~ "\nA) " ~ endings[0] ~ "\nB) " ~ endings[1] '
    r'~ "\nC) " ~ endings[2] ~ "\nD) " ~ endings[3]) | tojson }}}'
)
_OUTPUT_TEMPLATE = r'{"answer": {{ ["A", "B", "C", "D"][label | int] | tojson }}}'


@synalinks_export("synalinks.datasets.hellaswag.get_input_data_model")
def get_input_data_model():
    """Returns HellaSwag input data model."""
    return HellaSwagQuestion


@synalinks_export("synalinks.datasets.hellaswag.get_output_data_model")
def get_output_data_model():
    """Returns HellaSwag output data model."""
    return HellaSwagAnswer


def _load(split):
    return load_split(
        path="Rowan/hellaswag",
        split=split,
        input_data_model=HellaSwagQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=HellaSwagAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )


@synalinks_export("synalinks.datasets.hellaswag.load_data")
def load_data():
    """
    Load HellaSwag.

    HF test split has no public labels, so we use ``train`` for training
    and ``validation`` for evaluation.

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x_train, y_train = _load("train")
    x_test, y_test = _load("validation")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.hellaswag.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="Rowan/hellaswag",
        split=split,
        streaming=True,
        input_data_model=HellaSwagQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=HellaSwagAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
