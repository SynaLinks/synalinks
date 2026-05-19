# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.dataset import split_train_test
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class LAMBADAQuestion(DataModel):
    question: str = Field(
        description="The passage with a `___` blank for the missing final word",
    )


class LAMBADAAnswer(DataModel):
    answer: str = Field(
        description="The missing final word",
    )


# Take everything except the last whitespace-split token, append a blank.
_INPUT_TEMPLATE = (
    r"{%- set words = text.split() -%}"
    r'{"question": {{ ((words[:-1] | join(" ")) ~ " ___") | tojson }}}'
)
_OUTPUT_TEMPLATE = r'{"answer": {{ text.split()[-1] | tojson }}}'


@synalinks_export("synalinks.datasets.lambada.get_input_data_model")
def get_input_data_model():
    """Returns LAMBADA input data model."""
    return LAMBADAQuestion


@synalinks_export("synalinks.datasets.lambada.get_output_data_model")
def get_output_data_model():
    """Returns LAMBADA output data model."""
    return LAMBADAAnswer


@synalinks_export("synalinks.datasets.lambada.load_data")
def load_data(validation_split=0.2):
    """
    Load LAMBADA (OpenAI variant).

    HF ships only a ``test`` split (~5k passages), so we split it
    deterministically into train / test.

    Args:
        validation_split (float): Fraction held out for evaluation
            (default ``0.2``).

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x, y = load_split(
        path="EleutherAI/lambada_openai",
        name="en",
        split="test",
        input_data_model=LAMBADAQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=LAMBADAAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )
    return split_train_test(x, y, validation_split=validation_split)


@synalinks_export("synalinks.datasets.lambada.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="test"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="EleutherAI/lambada_openai",
        name="en",
        split=split,
        streaming=True,
        input_data_model=LAMBADAQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=LAMBADAAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
