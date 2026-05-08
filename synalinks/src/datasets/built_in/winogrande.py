# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Literal

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class WinoGrandeQuestion(DataModel):
    question: str = Field(
        description="The sentence with a `_` blank and two numbered options",
    )


class WinoGrandeAnswer(DataModel):
    answer: Literal[1, 2] = Field(
        description="The option that fills the blank",
    )


_INPUT_TEMPLATE = (
    r'{"question": {{ (sentence ~ "\n1) " ~ option1 ~ "\n2) " ~ option2) ' r"| tojson }}}"
)
# HF stores ``answer`` as a string; coerce to int so it matches Literal[1, 2].
_OUTPUT_TEMPLATE = r'{"answer": {{ (answer | int) | tojson }}}'


@synalinks_export("synalinks.datasets.winogrande.get_input_data_model")
def get_input_data_model():
    """Returns WinoGrande input data model."""
    return WinoGrandeQuestion


@synalinks_export("synalinks.datasets.winogrande.get_output_data_model")
def get_output_data_model():
    """Returns WinoGrande output data model."""
    return WinoGrandeAnswer


def _load(split):
    return load_split(
        path="allenai/winogrande",
        name="winogrande_xl",
        split=split,
        input_data_model=WinoGrandeQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=WinoGrandeAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )


@synalinks_export("synalinks.datasets.winogrande.load_data")
def load_data():
    """
    Load WinoGrande (XL).

    HF ``test`` split has no public labels, so we use ``train`` for
    training and ``validation`` for evaluation.

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x_train, y_train = _load("train")
    x_test, y_test = _load("validation")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.winogrande.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="allenai/winogrande",
        name="winogrande_xl",
        split=split,
        streaming=True,
        input_data_model=WinoGrandeQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=WinoGrandeAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
