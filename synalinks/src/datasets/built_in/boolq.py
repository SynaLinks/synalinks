# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class BoolQQuestion(DataModel):
    question: str = Field(
        description="The passage followed by the yes/no question",
    )


class BoolQAnswer(DataModel):
    answer: bool = Field(
        description="True for yes, False for no",
    )


_INPUT_TEMPLATE = (
    r'{"question": {{ (passage ~ "\n\nQuestion: " ~ question ~ "?") ' r"| tojson }}}"
)
_OUTPUT_TEMPLATE = r'{"answer": {{ answer | tojson }}}'


@synalinks_export("synalinks.datasets.boolq.get_input_data_model")
def get_input_data_model():
    """Returns BoolQ input data model."""
    return BoolQQuestion


@synalinks_export("synalinks.datasets.boolq.get_output_data_model")
def get_output_data_model():
    """Returns BoolQ output data model."""
    return BoolQAnswer


def _load(split):
    return load_split(
        path="google/boolq",
        split=split,
        input_data_model=BoolQQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=BoolQAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )


@synalinks_export("synalinks.datasets.boolq.load_data")
def load_data():
    """
    Load BoolQ.

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x_train, y_train = _load("train")
    x_test, y_test = _load("validation")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.boolq.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="google/boolq",
        split=split,
        streaming=True,
        input_data_model=BoolQQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=BoolQAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
