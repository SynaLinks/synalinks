# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split
from synalinks.src.datasets.dataset import split_train_test


class HumanEvalQuestion(DataModel):
    prompt: str = Field(
        description="The function signature and docstring",
    )


class HumanEvalAnswer(DataModel):
    completion: str = Field(
        description="The indented Python function body",
    )


_INPUT_TEMPLATE = r'{"prompt": {{ prompt | tojson }}}'
_OUTPUT_TEMPLATE = r'{"completion": {{ canonical_solution | tojson }}}'


@synalinks_export("synalinks.datasets.humaneval.get_input_data_model")
def get_input_data_model():
    """Returns HumanEval input data model."""
    return HumanEvalQuestion


@synalinks_export("synalinks.datasets.humaneval.get_output_data_model")
def get_output_data_model():
    """Returns HumanEval output data model."""
    return HumanEvalAnswer


@synalinks_export("synalinks.datasets.humaneval.load_data")
def load_data(validation_split=0.2):
    """
    Load HumanEval.

    HF ships only a ``test`` split (164 problems), so we split it
    deterministically into train / test. Real scoring requires running
    unit tests against the completion — exact match against the
    canonical solution is a placeholder reward.

    Args:
        validation_split (float): Fraction held out for evaluation
            (default ``0.2``).

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x, y = load_split(
        path="openai/openai_humaneval",
        split="test",
        input_data_model=HumanEvalQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=HumanEvalAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )
    return split_train_test(x, y, validation_split=validation_split)


@synalinks_export("synalinks.datasets.humaneval.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="test"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="openai/openai_humaneval",
        split=split,
        streaming=True,
        input_data_model=HumanEvalQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=HumanEvalAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
