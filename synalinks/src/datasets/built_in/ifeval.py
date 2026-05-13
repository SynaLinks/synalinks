# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset
from synalinks.src.datasets.huggingface_dataset import load_split
from synalinks.src.datasets.dataset import split_train_test


class IFEvalQuestion(DataModel):
    prompt: str = Field(
        description="The original instruction the model must follow",
    )


class IFEvalAnswer(DataModel):
    response: str = Field(
        description="Model response (gold = the original prompt for reference)",
    )


_INPUT_TEMPLATE = r'{"prompt": {{ prompt | tojson }}}'
# IFEval has no gold response — the original benchmark scores by rules.
# We mirror the prompt as ``response`` so an LM-as-judge sees both fields.
_OUTPUT_TEMPLATE = r'{"response": {{ prompt | tojson }}}'


@synalinks_export("synalinks.datasets.ifeval.get_input_data_model")
def get_input_data_model():
    """Returns IFEval input data model."""
    return IFEvalQuestion


@synalinks_export("synalinks.datasets.ifeval.get_output_data_model")
def get_output_data_model():
    """Returns IFEval output data model."""
    return IFEvalAnswer


@synalinks_export("synalinks.datasets.ifeval.load_data")
def load_data(validation_split=0.2):
    """
    Load IFEval (Instruction-Following Eval).

    HF ships only a ``train`` split (~541 prompts), so we split it
    deterministically into train / test. The benchmark is rule-based;
    the gold ``response`` is the prompt itself, intended to be used
    with an LM-as-judge reward.

    Args:
        validation_split (float): Fraction held out for evaluation
            (default ``0.2``).

    Returns:
        (tuple): ``(x_train, y_train), (x_test, y_test)``.
    """
    x, y = load_split(
        path="google/IFEval",
        split="train",
        input_data_model=IFEvalQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=IFEvalAnswer,
        output_template=_OUTPUT_TEMPLATE,
    )
    return split_train_test(x, y, validation_split=validation_split)


@synalinks_export("synalinks.datasets.ifeval.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="google/IFEval",
        split=split,
        streaming=True,
        input_data_model=IFEvalQuestion,
        input_template=_INPUT_TEMPLATE,
        output_data_model=IFEvalAnswer,
        output_template=_OUTPUT_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
    )
