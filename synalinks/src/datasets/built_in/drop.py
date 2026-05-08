# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class DROPQuestion(DataModel):
    question: str = Field(
        description="The passage followed by the question",
    )


class DROPAnswer(DataModel):
    answer: str = Field(
        description="A short span (number, date, or phrase) from the passage",
    )


_INPUT_TEMPLATE = r'{"question": {{ (passage ~ "\n\nQuestion: " ~ question) | tojson }}}'
# Pick the first gold span — DROP has multiple equivalent answers.
_OUTPUT_TEMPLATE = r'{"answer": {{ answers_spans.spans[0] | tojson }}}'


@synalinks_export("synalinks.datasets.drop.get_input_data_model")
def get_input_data_model():
    """Returns DROP input data model."""
    return DROPQuestion


@synalinks_export("synalinks.datasets.drop.get_output_data_model")
def get_output_data_model():
    """Returns DROP output data model."""
    return DROPAnswer


def _load(split):
    return load_split(
        path="ucinlp/drop",
        split=split,
        input_data_model=DROPQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=DROPAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )


@synalinks_export("synalinks.datasets.drop.load_data")
def load_data():
    """
    Load DROP (Discrete Reasoning Over Paragraphs).

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x_train, y_train = _load("train")
    x_test, y_test = _load("validation")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.drop.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="ucinlp/drop",
        split=split,
        streaming=True,
        input_data_model=DROPQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=DROPAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
