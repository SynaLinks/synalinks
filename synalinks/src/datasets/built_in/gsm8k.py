# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset


class MathQuestion(DataModel):
    question: str = Field(
        description="The math word problem",
    )


class NumericalAnswerWithThinking(DataModel):
    thinking: str = Field(
        description="Your step by step thinking",
    )
    answer: float = Field(
        description="The numerical answer",
    )


# Template renders each gsm8k row into JSON matching MathQuestion.
_INPUT_TEMPLATE = '{"question": {{ question | tojson }}}'

# gsm8k stores answers as ``"<chain of thought>\n#### <numeric>"``.
# Split, strip, drop commas, coerce to float so pydantic gets a JSON number.
_OUTPUT_TEMPLATE = (
    "{"
    '"thinking": {{ answer.split("####")[0].strip() | tojson }}, '
    '"answer": {{ answer.split("####")[-1].strip().replace(",", "") | float }}'
    "}"
)


@synalinks_export("synalinks.datasets.gsm8k.get_input_data_model")
def get_input_data_model():
    """
    Returns GSM8K input data_model for pipeline configurations.

    Returns:
        (DataModel): The GSM8K input data_model
    """
    return MathQuestion


@synalinks_export("synalinks.datasets.gsm8k.get_output_data_model")
def get_output_data_model():
    """
    Returns GSM8K output data_model for pipeline configurations.

    Returns:
        (DataModel): The GSM8K output data_model
    """
    return NumericalAnswerWithThinking


def _split_dataset(split):
    """Materialize a gsm8k split into a single ``(x, y)`` pair via HuggingFaceDataset."""
    ds = HuggingFaceDataset(
        path="gsm8k",
        name="main",
        split=split,
        streaming=False,
        input_data_model=MathQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=NumericalAnswerWithThinking,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=None,
    )
    return next(iter(ds))


@synalinks_export("synalinks.datasets.gsm8k.load_data")
def load_data():
    """
    Load and format data from HuggingFace.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
    ```

    Returns:
        (tuple): The train and test data ready for training
    """
    x_train, y_train = _split_dataset("train")
    x_test, y_test = _split_dataset("test")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.gsm8k.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Args:
        repeat (int): Number of consecutive copies of each row — set
            equal to ``batch_size`` for GRPO-style rollouts.
        batch_size (int): Examples per yielded batch.
        limit (int): Optional cap on raw rows (useful for smoke tests).
        split (str): HF split to stream. Defaults to ``"train"``.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="gsm8k",
        name="main",
        split=split,
        streaming=True,
        input_data_model=MathQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=NumericalAnswerWithThinking,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
