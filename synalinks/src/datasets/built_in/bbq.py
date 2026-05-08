# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Literal

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split
from synalinks.src.datasets.huggingface_dataset import split_train_test


class BBQQuestion(DataModel):
    question: str = Field(
        description="The context, question, and three lettered choices",
    )


class BBQAnswer(DataModel):
    answer: Literal["A", "B", "C"] = Field(
        description="The answer letter",
    )


_INPUT_TEMPLATE = (
    r'{"question": {{ (context ~ "\n" ~ question ~ "\nA) " ~ choices[0] '
    r'~ "\nB) " ~ choices[1] ~ "\nC) " ~ choices[2]) | tojson }}}'
)
_OUTPUT_TEMPLATE = r'{"answer": {{ ["A", "B", "C"][answer] | tojson }}}'


@synalinks_export("synalinks.datasets.bbq.get_input_data_model")
def get_input_data_model():
    """Returns BBQ input data model."""
    return BBQQuestion


@synalinks_export("synalinks.datasets.bbq.get_output_data_model")
def get_output_data_model():
    """Returns BBQ output data model."""
    return BBQAnswer


@synalinks_export("synalinks.datasets.bbq.load_data")
def load_data(category="age", validation_split=0.2):
    """
    Load BBQ (Bias Benchmark for QA).

    BBQ on HF is split by *category* (``age``, ``gender_identity``,
    ``race_ethnicity``, ``religion``, ``ses``, ...) rather than train /
    test. We load the requested category and split it deterministically
    into train / test.

    Args:
        category (str): The BBQ category to load. Defaults to ``"age"``.
        validation_split (float): Fraction held out for evaluation
            (default ``0.2``).

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x, y = load_split(
        path="walledai/BBQ",
        split=category,
        input_data_model=BBQQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=BBQAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )
    return split_train_test(x, y, validation_split=validation_split)


@synalinks_export("synalinks.datasets.bbq.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, category="age"):
    """
    Streaming dataset for RL-style training.

    Args:
        category (str): BBQ category to stream (``"age"`` by default).

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="walledai/BBQ",
        split=category,
        streaming=True,
        input_data_model=BBQQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=BBQAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
