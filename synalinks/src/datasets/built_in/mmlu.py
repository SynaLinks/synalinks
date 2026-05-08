# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Literal

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset


class MMLUQuestion(DataModel):
    question: str = Field(
        description="The multiple-choice question text",
    )
    choices: List[str] = Field(
        description="The four answer options, in order A, B, C, D",
    )
    subject: str = Field(
        description="The subject domain (e.g. anatomy, abstract_algebra)",
    )


class MMLUAnswer(DataModel):
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The letter of the correct choice",
    )


_INPUT_TEMPLATE = (
    "{"
    '"question": {{ question | tojson }}, '
    '"choices": {{ choices | tojson }}, '
    '"subject": {{ subject | tojson }}'
    "}"
)

# MMLU stores ``answer`` as an int 0-3; map it to the conventional letter.
_OUTPUT_TEMPLATE = '{"answer": {{ ["A", "B", "C", "D"][answer] | tojson }}}'


@synalinks_export("synalinks.datasets.mmlu.get_input_data_model")
def get_input_data_model():
    """
    Returns MMLU input data model for pipeline configurations.

    Returns:
        (DataModel): The MMLU input data model
    """
    return MMLUQuestion


@synalinks_export("synalinks.datasets.mmlu.get_output_data_model")
def get_output_data_model():
    """
    Returns MMLU output data model for pipeline configurations.

    Returns:
        (DataModel): The MMLU output data model
    """
    return MMLUAnswer


def _split_dataset(split):
    """Materialize an MMLU split into a single ``(x, y)`` pair via HuggingFaceDataset."""
    ds = HuggingFaceDataset(
        path="cais/mmlu",
        name="all",
        split=split,
        streaming=False,
        input_data_model=MMLUQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=MMLUAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=None,
    )
    return next(iter(ds))


@synalinks_export("synalinks.datasets.mmlu.load_data")
def load_data():
    """
    Load and format data from HuggingFace.

    MMLU is an evaluation-only benchmark; the conventional split here is
    the ``validation`` set (1.5k examples, useful for few-shot prompt
    tuning) as ``train`` and the ``test`` set (14k examples) as ``test``.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.mmlu.load_data()
    ```

    Returns:
        (tuple): The train and test data ready for training
    """
    x_train, y_train = _split_dataset("validation")
    x_test, y_test = _split_dataset("test")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.mmlu.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="validation"):
    """
    Streaming dataset for RL-style training.

    Default split is ``"validation"`` (1.5k rows) — the conventional
    training pool for MMLU. Pass ``split="auxiliary_train"`` for the
    larger 99k auxiliary corpus, or ``split="test"`` for evaluation.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="cais/mmlu",
        name="all",
        split=split,
        streaming=True,
        input_data_model=MMLUQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=MMLUAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
