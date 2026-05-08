# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class SQuADQuestion(DataModel):
    question: str = Field(
        description="The context followed by the question",
    )


class SQuADAnswer(DataModel):
    answer: str = Field(
        description="A short span from the context",
    )


_INPUT_TEMPLATE = r'{"question": {{ (context ~ "\n\nQuestion: " ~ question) | tojson }}}'
# SQuAD answers are a list of equivalent gold spans — pick the first.
_OUTPUT_TEMPLATE = r'{"answer": {{ answers.text[0] | tojson }}}'


@synalinks_export("synalinks.datasets.squad.get_input_data_model")
def get_input_data_model():
    """Returns SQuAD input data model."""
    return SQuADQuestion


@synalinks_export("synalinks.datasets.squad.get_output_data_model")
def get_output_data_model():
    """Returns SQuAD output data model."""
    return SQuADAnswer


def _load(split):
    return load_split(
        path="rajpurkar/squad",
        split=split,
        input_data_model=SQuADQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=SQuADAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )


@synalinks_export("synalinks.datasets.squad.load_data")
def load_data():
    """
    Load SQuAD v1.1.

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x_train, y_train = _load("train")
    x_test, y_test = _load("validation")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.squad.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="rajpurkar/squad",
        split=split,
        streaming=True,
        input_data_model=SQuADQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=SQuADAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
