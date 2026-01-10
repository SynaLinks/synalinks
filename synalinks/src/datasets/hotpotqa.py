# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)


import numpy as np
from datasets import load_dataset

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel


class Document(DataModel):
    title: str
    text: str


class Question(DataModel):
    question: str


class Answer(DataModel):
    answer: str


@synalinks_export("synalinks.datasets.hotpotqa.get_knowledge_data_model")
def get_knowledge_data_model():
    """
    Returns HotpotQA knowledge data model for pipeline configurations.

    Returns:
        (DataModel): The HotpotQA knowledge data model
    """
    return Document


@synalinks_export("synalinks.datasets.hotpotqa.get_input_data_model")
def get_input_data_model():
    """
    Returns HotpotQA input data model for pipeline configurations.

    Returns:
        (DataModel): The HotpotQA input data model
    """
    return Question


@synalinks_export("synalinks.datasets.hotpotqa.get_output_data_model")
def get_output_data_model():
    """
    Returns HotpotQA output data model for pipeline configurations.

    Returns:
        (DataModel): The HotpotQA output data model
    """
    return Answer


@synalinks_export("synalinks.datasets.hotpotqa.load_knowledge")
def load_knowledge():
    """
    Load and format data from HuggingFace

    Example:

    ```python
    knowledge = synalinks.datasets.hotpotqa.load_knowledge()
    ```

    Returns:
        (list): The  data ready for knowledge injestion
    """
    documents = []
    train_examples = load_dataset(
        "hotpot_qa", "fullwiki", split="train", trust_remote_code=True
    )
    for raw_example in train_examples:
        context = raw_example.get("context", None)
        if context:
            for i in range(len(context["title"])):
                documents.append(
                    Document(
                        title=context["title"][i],
                        text="\n".join(context["sentences"][i]),
                    )
                )
    documents = np.array(documents, dtype="object")
    return documents


@synalinks_export("synalinks.datasets.hotpotqa.load_data")
def load_data():
    """
    Load and format data from HuggingFace

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.hotpotqa.load_data()
    ```

    Returns:
        (tuple): The train and test data ready for training
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    train_examples = load_dataset(
        "hotpot_qa", "fullwiki", split="train", trust_remote_code=True
    )
    eval_examples = load_dataset(
        "hotpot_qa", "fullwiki", split="validation", trust_remote_code=True
    )

    for raw_example in train_examples:
        x_train.append(Question(question=raw_example["question"]))
        y_train.append(Answer(answer=raw_example["answer"]))

    for raw_example in eval_examples:
        if raw_example["level"] == "hard":
            x_test.append(Question(question=raw_example["question"]))
            y_test.append(Answer(answer=raw_example["answer"]))

    x_train = np.array(x_train, dtype="object")
    y_train = np.array(y_train, dtype="object")

    x_test = np.array(x_test, dtype="object")
    y_test = np.array(y_test, dtype="object")

    return (x_train, y_train), (x_test, y_test)
