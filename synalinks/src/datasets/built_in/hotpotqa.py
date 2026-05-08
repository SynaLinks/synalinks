# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset


class Document(DataModel):
    title: str
    text: str


class Question(DataModel):
    question: str


class Answer(DataModel):
    answer: str


_QUESTION_TEMPLATE = '{"question": {{ question | tojson }}}'
_ANSWER_TEMPLATE = '{"answer": {{ answer | tojson }}}'
_DOCUMENT_TEMPLATE = '{"title": {{ title | tojson }}, "text": {{ text | tojson }}}'


class _HardOnlyHotpotQA(HuggingFaceDataset):
    """Validation split filtered to ``level == "hard"`` rows only."""

    def _iter_rows(self):
        for raw_example in self._dataset:
            if raw_example.get("level") == "hard":
                yield raw_example


class _HotpotKnowledge(HuggingFaceDataset):
    """Explodes each row's ``context`` into one ``Document`` per paragraph."""

    def _iter_rows(self):
        for raw_example in self._dataset:
            context = raw_example.get("context")
            if not context:
                continue
            for i in range(len(context["title"])):
                yield {
                    "title": context["title"][i],
                    "text": "\n".join(context["sentences"][i]),
                }


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
    Load and format data from HuggingFace.

    Example:

    ```python
    knowledge = synalinks.datasets.hotpotqa.load_knowledge()
    ```

    Returns:
        (np.ndarray): The data ready for knowledge ingestion
    """
    ds = _HotpotKnowledge(
        path="hotpot_qa",
        name="fullwiki",
        split="train",
        streaming=False,
        input_data_model=Document,
        input_template=_DOCUMENT_TEMPLATE,
        batch_size=None,
        trust_remote_code=True,
    )
    (documents,) = next(iter(ds))
    return documents


@synalinks_export("synalinks.datasets.hotpotqa.load_data")
def load_data():
    """
    Load and format data from HuggingFace.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.hotpotqa.load_data()
    ```

    Returns:
        (tuple): The train and test data ready for training
    """
    train_ds = HuggingFaceDataset(
        path="hotpot_qa",
        name="fullwiki",
        split="train",
        streaming=False,
        input_data_model=Question,
        input_template=_QUESTION_TEMPLATE,
        output_data_model=Answer,
        output_template=_ANSWER_TEMPLATE,
        batch_size=None,
        trust_remote_code=True,
    )
    test_ds = _HardOnlyHotpotQA(
        path="hotpot_qa",
        name="fullwiki",
        split="validation",
        streaming=False,
        input_data_model=Question,
        input_template=_QUESTION_TEMPLATE,
        output_data_model=Answer,
        output_template=_ANSWER_TEMPLATE,
        batch_size=None,
        trust_remote_code=True,
    )
    x_train, y_train = next(iter(train_ds))
    x_test, y_test = next(iter(test_ds))
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.hotpotqa.iterable_dataset")
def iterable_dataset(repeat=1, batch_size=1, limit=None, split="train"):
    """
    Streaming dataset for RL-style training.

    Returns:
        (HuggingFaceDataset): A streaming, iterable dataset.
    """
    return HuggingFaceDataset(
        path="hotpot_qa",
        name="fullwiki",
        split=split,
        streaming=True,
        input_data_model=Question,
        input_template=_QUESTION_TEMPLATE,
        output_data_model=Answer,
        output_template=_ANSWER_TEMPLATE,
        batch_size=batch_size,
        limit=limit,
        repeat=repeat,
        trust_remote_code=True,
    )
