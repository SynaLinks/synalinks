# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split
from synalinks.src.datasets.dataset import split_train_test


class BBHBooleanQuestion(DataModel):
    question: str = Field(
        description="The boolean expression to evaluate",
    )


class BBHBooleanAnswer(DataModel):
    answer: bool = Field(
        description="The truth value of the boolean expression",
    )


_INPUT_TEMPLATE = r'{"question": {{ input | tojson }}}'
# BBH ships ``target`` as the literal string "True" / "False" — coerce to bool.
_OUTPUT_TEMPLATE = r'{"answer": {{ (target == "True") | tojson }}}'


@synalinks_export("synalinks.datasets.bbh.get_input_data_model")
def get_input_data_model():
    """Returns BBH (boolean_expressions) input data model."""
    return BBHBooleanQuestion


@synalinks_export("synalinks.datasets.bbh.get_output_data_model")
def get_output_data_model():
    """Returns BBH (boolean_expressions) output data model."""
    return BBHBooleanAnswer


@synalinks_export("synalinks.datasets.bbh.load_data")
def load_data(validation_split=0.2):
    """
    Load BIG-Bench Hard (boolean_expressions task).

    BBH ships only a ``test`` split (~250 rows per task), so we split
    it deterministically into train / test.

    Args:
        validation_split (float): Fraction held out for evaluation
            (default ``0.2``).

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x, y = load_split(
        path="lukaemon/bbh",
        name="boolean_expressions",
        split="test",
        input_data_model=BBHBooleanQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=BBHBooleanAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )
    return split_train_test(x, y, validation_split=validation_split)


@synalinks_export("synalinks.datasets.bbh.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="test"):
    """
    Streaming dataset for RL-style training.

    Args:
        repeat (int): Number of consecutive copies of each row — set
            equal to ``batch_size`` for GRPO-style rollouts.
        batch_size (int): Examples per yielded batch.
        limit (int): Optional cap on raw rows (useful for smoke tests).
        split (str): HF split to stream. Defaults to ``"test"`` (the
            only labeled split BBH ships).

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="lukaemon/bbh",
        name="boolean_expressions",
        split=split,
        streaming=True,
        input_data_model=BBHBooleanQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=BBHBooleanAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
