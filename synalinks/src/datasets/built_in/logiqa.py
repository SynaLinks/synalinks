# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Literal

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class LogiQAQuestion(DataModel):
    question: str = Field(
        description="Context, question, and lettered options",
    )


class LogiQAAnswer(DataModel):
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The answer letter",
    )


# LogiQA's ``options`` strings already include "A. " / "B. " prefixes.
_INPUT_TEMPLATE = (
    r'{"question": {{ (context ~ "\n\nQuestion: " ~ question ~ "\n" ~ '
    r'(options | join("\n"))) | tojson }}}'
)
_OUTPUT_TEMPLATE = r'{"answer": {{ answer | tojson }}}'


@synalinks_export("synalinks.datasets.logiqa.get_input_data_model")
def get_input_data_model():
    """Returns LogiQA input data model."""
    return LogiQAQuestion


@synalinks_export("synalinks.datasets.logiqa.get_output_data_model")
def get_output_data_model():
    """Returns LogiQA output data model."""
    return LogiQAAnswer


def _load(split):
    return load_split(
        path="fireworks-ai/logiqa",
        split=split,
        input_data_model=LogiQAQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=LogiQAAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )


@synalinks_export("synalinks.datasets.logiqa.load_data")
def load_data():
    """
    Load LogiQA.

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x_train, y_train = _load("train")
    x_test, y_test = _load("test")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.logiqa.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="fireworks-ai/logiqa",
        split=split,
        streaming=True,
        input_data_model=LogiQAQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=LogiQAAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
