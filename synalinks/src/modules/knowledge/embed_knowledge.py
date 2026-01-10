# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import warnings

from synalinks.src import ops
from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import Embedding as EmbeddingVector
from synalinks.src.backend import JsonDataModel
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.async_utils import run_maybe_nested


@synalinks_export(
    [
        "synalinks.modules.EmbedKnowledge",
        "synalinks.EmbedKnowledge",
    ]
)
class EmbedKnowledge(Module):
    """Extracts a field of interest and generate the corresponding embedding vector.

    This module is designed to work with any data model structure. It supports to mask the
    entity fields in order to keep **only one** field to embed per data model.

    **Note**: Each data model should have the *same field* to compute the embedding
        from like a `name` or `description` field using `in_mask`.
        **Or** every data model should have *only one field left* after masking using
        `out_mask` argument.

    ```python
    import synalinks
    import asyncio
    from typing import Literal

    class Document(synalinks.DataModel):
        title: str = synalinks.Field(
            description="The document title",
        )
        text: str = synalinks.Field(
            description="The document content",
        )

    async def main():
        inputs = synalinks.Input(data_model=Document)
        outputs = await synalinks.EmbedKnowledge(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)

        program = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="embbed_document",
            description="Embbed the given documents"
        )

        doc = Document(
            title="my title",
            text="my document",
        )

        result = await program(doc)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    If you want to process batch asynchronously
    use `program.predict()` instead, see the [FAQ](https://synalinks.github.io/synalinks/FAQ/#whats-the-difference-between-program-methods-predict-and-__call__)
    to understand the difference between `program()` and `program.predict()`

    Here is an example:

    ```python
    import synalinks
    import asyncio
    import numpy as np
    from typing import Literal

    class Document(synalinks.Entity):
        label: Literal["Document"]
        text: str = synalinks.Field(
            description="The document content",
        )

    async def main():
        inputs = synalinks.Input(data_model=Document)
        outputs = await synalinks.EmbedKnowledge(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)

        program = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="embbed_document",
            description="Embbed the given documents"
        )

        doc1 = Document(label="Document", text="my document 1")
        doc2 = Document(label="Document", text="my document 2")
        doc3 = Document(label="Document", text="my document 3")

        docs = np.array([doc1, doc2, doc3], dtype="object")

        embedded_docs = await program.predict(docs)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        embedding_model (EmbeddingModel): The embedding model to use.
        in_mask (list): A mask applied to keep specific entity fields.
        out_mask (list): A mask applied to remove specific entity fields.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        embedding_model=None,
        in_mask=None,
        out_mask=None,
        name=None,
        description=None,
        trainable=False,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.embedding_model = embedding_model
        self.in_mask = in_mask
        self.out_mask = out_mask

    async def _embed(self, data_model):
        embeddings = data_model.get("embeddings")
        if embeddings:
            warnings.warn(
                "Embeddings already generated for this data model. "
                "Returning original data model."
            )
            return JsonDataModel(
                json=data_model.get_json(),
                schema=data_model.get_schema(),
                name="embedded_" + data_model.name,
            )
        masked_data_model = data_model
        if self.out_mask:
            masked_data_model = await ops.out_mask(
                data_model,
                mask=self.out_mask,
                recursive=False,
                name="out_mask_" + data_model.name,
            )
        elif self.in_mask:
            masked_data_model = await ops.in_mask(
                data_model,
                mask=self.in_mask,
                recursive=False,
                name="in_mask_" + data_model.name,
            )
        embeddings = await ops.embedding(
            masked_data_model,
            embedding_model=self.embedding_model,
            name=data_model.name + "_embedding",
        )
        if not embeddings or not embeddings.get("embeddings"):
            warnings.warn(
                f"No embeddings generated for data model {data_model.name}. "
                "Please check that your schema is correct."
            )
            return None
        embedding_list = embeddings.get("embeddings")
        if len(embedding_list) != 1:
            warnings.warn(
                "Data models can only have one embedding vector per data model, "
                "adjust `EmbedKnowledge` module's `in_mask` or `out_mask` "
                "to keep only one field. Skipping embedding."
            )
            return None
        vector = embedding_list[0]
        return await ops.concat(
            data_model,
            EmbeddingVector(embedding=vector),
            name="embedded_" + data_model.name,
        )

    async def call(self, inputs):
        if not inputs:
            return None
        return tree.map_structure(
            lambda x: run_maybe_nested(self._embed(x)),
            inputs,
        )

    async def compute_output_spec(self, inputs):
        return tree.map_structure(
            lambda x: x.clone(name="embedded_" + x.name),
            inputs,
        )

    def get_config(self):
        config = {
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        embedding_model_config = {
            "embedding_model": serialization_lib.serialize_synalinks_object(
                self.embedding_model
            )
        }
        return {**embedding_model_config, **config}

    @classmethod
    def from_config(cls, config):
        embedding_model = serialization_lib.deserialize_synalinks_object(
            config.pop("embedding_model")
        )
        return cls(embedding_model=embedding_model, **config)
