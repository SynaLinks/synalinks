# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import copy
import warnings

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import Embedding as EmbeddingVector
from synalinks.src.backend import EmbeddingRequest
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_entities
from synalinks.src.backend import is_knowledge_graph
from synalinks.src.backend import is_knowledge_graphs
from synalinks.src.backend import is_relation
from synalinks.src.backend import is_relations
from synalinks.src.modules.embedding_models import get as _get_em
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib

# The JSON-schema fragment for the `embedding` field, taken from the
# `Embedding` data model so the augmented output schema declares the
# vector exactly the way the rest of the framework does.
_EMBEDDING_FIELD_SCHEMA = EmbeddingVector.get_schema()["properties"]["embedding"]


@synalinks_export(
    [
        "synalinks.modules.EmbedKnowledge",
        "synalinks.EmbedKnowledge",
    ]
)
class EmbedKnowledge(Module):
    """Extracts a field of interest and generate the corresponding embedding vector.

    This module works with any data model structure and, crucially, with the
    knowledge primitives: an `Entity`, an `Entities`/`Relations` list, a
    `Relation`, a `KnowledgeGraph` or a `KnowledgeGraphs` batch. It gathers
    every entity that needs an embedding (including the `subj`/`obj`
    endpoints carried by relations), sends them to the embedding model in a
    **single batched** request, then returns the same structure with an
    `embedding` vector reattached in place on each entity.

    The `in_mask` / `out_mask` arguments select which field(s) of each entity
    are concatenated into the text that gets embedded.

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
        *,
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
        self.embedding_model = _get_em(embedding_model)
        self.in_mask = in_mask
        self.out_mask = out_mask

    def _embeds_whole_object(self, data_model):
        """Whether the object is embedded as a single unit.

        True for a plain data model or a bare `Entity` (the top-level
        object carries the vector). False for the container shapes
        (`Entities`, `Relations`, `KnowledgeGraph(s)`) and a `Relation`,
        whose nested entities are embedded individually instead.
        """
        return not (
            is_knowledge_graphs(data_model)
            or is_knowledge_graph(data_model)
            or is_relations(data_model)
            or is_entities(data_model)
            or is_relation(data_model)
        )

    def _text_for(self, entity_json):
        """Mask an entity dict and flatten the kept fields into one string.

        Masking is field-name selection on the top-level keys — the same
        effect as `ops.in_mask` / `ops.out_mask` with ``recursive=False``,
        done directly on the dict so it works on any nested entity without
        depending on a resolvable per-entity schema.
        """
        fields = dict(entity_json)
        if self.out_mask:
            fields = {k: v for k, v in fields.items() if k not in self.out_mask}
        elif self.in_mask:
            fields = {k: v for k, v in fields.items() if k in self.in_mask}
        texts = tree.flatten(tree.map_structure(lambda f: str(f), fields))
        return " ".join(t for t in texts if t)

    def _gather_relation(self, rel_json):
        """Collect the subj/obj entity dicts carried by a relation."""
        units = []
        for endpoint in ("subj", "obj"):
            value = rel_json.get(endpoint)
            if isinstance(value, dict):
                units.append(value)
        return units

    def _gather_kg(self, kg_json):
        """Collect a knowledge graph's entity dicts + relation endpoints."""
        units = list(kg_json.get("entities", []) or [])
        for rel_json in kg_json.get("relations", []) or []:
            units += self._gather_relation(rel_json)
        return units

    def _gather(self, json, data_model):
        """Return the mutable entity dicts to embed, in batch order.

        Each returned dict is a sub-dict of ``json`` — writing
        ``["embedding"]`` onto it lands in the output. Dispatch order
        mirrors `UpdateKnowledge._update` (graphs before lists, relation
        before entity). `KnowledgeGraph(s)` are walked structurally: a
        graph carries no ``label`` discriminator for
        `get_nested_entity_list`, so only the label-bearing leaf entities
        (and relation endpoints) are embedded.
        """
        if is_knowledge_graphs(data_model):
            units = []
            for kg_json in json.get("knowledge_graphs", []) or []:
                units += self._gather_kg(kg_json)
            return units
        if is_knowledge_graph(data_model):
            return self._gather_kg(json)
        if is_relations(data_model):
            units = []
            for rel_json in json.get("relations", []) or []:
                units += self._gather_relation(rel_json)
            return units
        if is_entities(data_model):
            return list(json.get("entities", []) or [])
        if is_relation(data_model):
            return self._gather_relation(json)
        # Bare entity or plain data model: embed the whole object.
        return [json]

    def _augment_schema(self, data_model):
        """Add the `embedding` field to the schema's entity definitions."""
        schema = copy.deepcopy(data_model.get_schema())
        if self._embeds_whole_object(data_model):
            schema.setdefault("properties", {})["embedding"] = _EMBEDDING_FIELD_SCHEMA
        for definition in (schema.get("$defs") or {}).values():
            props = definition.get("properties")
            # Entities carry a `label`; relations (label + subj + obj) are
            # not embedded themselves, only their endpoints.
            if props and "label" in props and not ("subj" in props and "obj" in props):
                props["embedding"] = _EMBEDDING_FIELD_SCHEMA
        return schema

    async def _embed(self, data_model):
        if self._embeds_whole_object(data_model) and data_model.get("embedding"):
            warnings.warn(
                "Embedding already generated for this data model. "
                "Returning original data model."
            )
            return data_model.clone(name="embedded_" + data_model.name)

        json = copy.deepcopy(data_model.get_json())
        units = self._gather(json, data_model)
        if not units:
            return data_model.clone(name="embedded_" + data_model.name)

        texts = [self._text_for(unit) for unit in units]
        embeddings = await self.embedding_model(EmbeddingRequest(texts=texts))
        vectors = embeddings.get("embeddings") if embeddings else None
        if not vectors:
            warnings.warn(
                f"No embeddings generated for data model {data_model.name}. "
                "Please check that your schema is correct."
            )
            return None
        if len(vectors) != len(units):
            warnings.warn(
                f"Expected {len(units)} embedding vectors but got {len(vectors)} "
                f"for data model {data_model.name}. Skipping embedding."
            )
            return None

        for unit, vector in zip(units, vectors):
            unit["embedding"] = vector

        return JsonDataModel(
            json=json,
            schema=self._augment_schema(data_model),
            name="embedded_" + data_model.name,
        )

    async def call(self, inputs):
        if not inputs:
            return None
        # Await each embedding on the *current* event loop. Using
        # `run_maybe_nested` here ran every embedding on a transient thread-loop,
        # binding litellm's process-global httpx client to a loop closed moments
        # later ("Event loop is closed" at teardown). Sequential to preserve the
        # original one-at-a-time ordering. flatten/pack mirrors `map_structure`
        # (data models are tree leaves), so the output structure is unchanged.
        leaves = tree.flatten(inputs)
        embedded = [await self._embed(leaf) for leaf in leaves]
        return tree.pack_sequence_as(inputs, embedded)

    async def compute_output_spec(self, inputs):
        return tree.map_structure(
            lambda x: SymbolicDataModel(
                schema=self._augment_schema(x),
                name="embedded_" + x.name,
            ),
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
