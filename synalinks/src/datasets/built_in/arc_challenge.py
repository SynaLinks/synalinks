# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split


class ARCQuestion(DataModel):
    question: str = Field(
        description="The science question with lettered choices",
    )


class ARCAnswer(DataModel):
    answer: str = Field(
        description="The letter (A-E) or number (1-5) of the correct choice",
    )


# Choice count varies (3-5), so we cannot use a Literal here. Build the
# rendered question text by interleaving choices.label[i] / choices.text[i].
_INPUT_TEMPLATE = (
    r"{%- set lines = [] -%}"
    r"{%- for i in range(choices.text | length) -%}"
    r'{%- set _ = lines.append(choices.label[i] ~ ") " ~ choices.text[i]) -%}'
    r"{%- endfor -%}"
    r'{"question": {{ (question ~ "\n" ~ lines | join("\n")) | tojson }}}'
)
_OUTPUT_TEMPLATE = r'{"answer": {{ answerKey | tojson }}}'


@synalinks_export("synalinks.datasets.arc_challenge.get_input_data_model")
def get_input_data_model():
    """Returns ARC-Challenge input data model."""
    return ARCQuestion


@synalinks_export("synalinks.datasets.arc_challenge.get_output_data_model")
def get_output_data_model():
    """Returns ARC-Challenge output data model."""
    return ARCAnswer


def _load(split):
    return load_split(
        path="allenai/ai2_arc",
        name="ARC-Challenge",
        split=split,
        input_data_model=ARCQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=ARCAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )


@synalinks_export("synalinks.datasets.arc_challenge.load_data")
def load_data():
    """
    Load ARC-Challenge.

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x_train, y_train = _load("train")
    x_test, y_test = _load("test")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.arc_challenge.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="allenai/ai2_arc",
        name="ARC-Challenge",
        split=split,
        streaming=True,
        input_data_model=ARCQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=ARCAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
