# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import numpy as np
from datasets import load_dataset

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field


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


@synalinks_export("synalinks.datasets.gsm8k.load_data")
def load_data():
    """
    Load and format data from HuggingFace

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
    ```

    Returns:
        (tuple): The train and test data ready for training
    """
    dataset = load_dataset("gsm8k", "main")

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for data_point in dataset["train"]:
        question = data_point["question"]
        thinking = data_point["answer"].split("####")[0].strip()
        answer = data_point["answer"].split("####")[-1].strip()
        x_train.append(MathQuestion(question=question))
        y_train.append(
            NumericalAnswerWithThinking(
                thinking=thinking,
                answer=float(answer.replace(",", "")),
            )
        )

    for data_point in dataset["test"]:
        question = data_point["question"]
        thinking = data_point["answer"].split("####")[0].strip()
        answer = data_point["answer"].split("####")[-1].strip()
        x_test.append(MathQuestion(question=question))
        y_test.append(
            NumericalAnswerWithThinking(
                thinking=thinking,
                answer=float(answer.replace(",", "")),
            )
        )

    x_train = np.array(x_train, dtype="object")
    y_train = np.array(y_train, dtype="object")

    x_test = np.array(x_test, dtype="object")
    y_test = np.array(y_test, dtype="object")

    return (x_train, y_train), (x_test, y_test)
