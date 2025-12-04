## `Embedding`

Bases: `Module`

Extracts and updates the embedding vector of entities.

This module is designed to work with `Entity`, `Relation`, `Entities`, `Relations` or `KnowledgeGraph` data models. It supports to mask the entity fields in order to keep **only one** field to embed per entity.

**Note**: Each entity should have the *same field* to compute the embedding from like a `name` or `description` field using `in_mask`. **Or** every entity should have *only one field left* after masking using `out_mask` argument.

```
import synalinks
import asyncio
from typing import Literal

class Document(synalinks.Entity):
    label: Literal["Document"]
    text: str = synalinks.Field(
        description="The document content",
    )

async def main():
    inputs = synalinks.Input(data_model=Document)
    outputs = await synalinks.Embedding(
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
        label="Document",
        text="my document",
    )

    result = await program(doc)

if __name__ == "__main__":
    asyncio.run(main())
```

If you want to process batch asynchronously use `program.predict()` instead, see the [FAQ](https://synalinks.github.io/synalinks/FAQ/#whats-the-difference-between-program-methods-predict-and-__call__) to understand the difference between `program()` and `program.predict()`

Here is an example:

```
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
    outputs = await synalinks.Embedding(
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

Parameters:

| Name              | Type             | Description                                         | Default |
| ----------------- | ---------------- | --------------------------------------------------- | ------- |
| `embedding_model` | `EmbeddingModel` | The embedding model to use.                         | `None`  |
| `in_mask`         | `list`           | A mask applied to keep specific entity fields.      | `None`  |
| `out_mask`        | `list`           | A mask applied to remove specific entity fields.    | `None`  |
| `name`            | `str`            | Optional. The name of the module.                   | `None`  |
| `description`     | `str`            | Optional. The description of the module.            | `None`  |
| `trainable`       | `bool`           | Whether the module's variables should be trainable. | `False` |

Source code in `synalinks/src/modules/knowledge/embedding.py`

````
@synalinks_export(
    [
        "synalinks.modules.Embedding",
        "synalinks.Embedding",
    ]
)
class Embedding(Module):
    """Extracts and updates the embedding vector of entities.

    This module is designed to work with `Entity`, `Relation`, `Entities`,
    `Relations` or `KnowledgeGraph` data models. It supports to mask the
    entity fields in order to keep **only one** field to embed per entity.

    **Note**: Each entity should have the *same field* to compute the embedding
        from like a `name` or `description` field using `in_mask`.
        **Or** every entity should have *only one field left* after masking using
        `out_mask` argument.

    ```python
    import synalinks
    import asyncio
    from typing import Literal

    class Document(synalinks.Entity):
        label: Literal["Document"]
        text: str = synalinks.Field(
            description="The document content",
        )

    async def main():
        inputs = synalinks.Input(data_model=Document)
        outputs = await synalinks.Embedding(
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
            label="Document",
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
        outputs = await synalinks.Embedding(
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

    async def _embed_entity(self, entity):
        # # Check if entity is already embedded and has valid embeddings
        embeddings = entity.get("embeddings")
        if embeddings:
            warnings.warn(
                "Embeddings already generated for entity.Returning original entity."
            )
            return JsonDataModel(
                json=entity.get_json(),
                schema=entity.get_schema(),
                name=entity.name + "_embedded",
            )
        # Apply masking to the entity
        filtered_entity = entity  # Default to original entity

        if self.out_mask:
            filtered_entity = await ops.out_mask(
                entity,
                mask=self.out_mask,
                recursive=False,
                name=entity.name + "_out_mask",
            )
        elif self.in_mask:
            filtered_entity = await ops.in_mask(
                entity,
                mask=self.in_mask,
                recursive=False,
                name=entity.name + "_in_mask",
            )

        # Generate embeddings
        embeddings = await ops.embedding(
            filtered_entity,
            embedding_model=self.embedding_model,
            name=entity.name + "_embedding",
        )

        # Validate embeddings
        if not embeddings or not embeddings.get("embeddings"):
            warnings.warn(
                f"No embeddings generated for entity {entity.name}. "
                "Please check that your schema is correct."
            )
            return None

        embedding_list = embeddings.get("embeddings")
        if len(embedding_list) != 1:
            warnings.warn(
                "Entities can only have one embedding vector per entity, "
                "adjust `Embedding` module's `in_mask` or `out_mask` "
                "to keep only one field. Skipping embedding."
            )
            return None

        # Add embedding to entity
        vector = embedding_list[0]
        return await ops.concat(
            entity,
            EmbeddingVector(embedding=vector),
            name=entity.name + "_embedded",
        )

    async def _embed_relation(self, relation):
        subj = relation.get_nested_entity("subj")
        obj = relation.get_nested_entity("obj")
        if not subj or not obj:
            return None
        embedded_subj = await self._embed_entity(subj)
        embedded_obj = await self._embed_entity(obj)

        relation_json = copy.deepcopy(relation.get_json())
        relation_json.update(
            {
                "subj": embedded_subj.get_json(),
                "obj": embedded_obj.get_json(),
            }
        )
        outputs_schema = copy.deepcopy(relation.get_schema())

        # Update schema definitions for embedded entities
        if outputs_schema.get("$defs"):
            subj_label = subj.get("label")
            obj_label = obj.get("label")

            if subj_label and subj_label in outputs_schema["$defs"]:
                embedded_subj_schema = embedded_subj.get_schema()
                if embedded_subj_schema.get("properties"):
                    outputs_schema["$defs"][subj_label]["properties"].update(
                        embedded_subj_schema["properties"]
                    )

            if obj_label and obj_label in outputs_schema["$defs"]:
                embedded_obj_schema = embedded_obj.get_schema()
                if embedded_obj_schema.get("properties"):
                    outputs_schema["$defs"][obj_label]["properties"].update(
                        embedded_obj_schema["properties"]
                    )

        return JsonDataModel(
            json=relation_json,
            schema=outputs_schema,
            name=relation.name + "_embedded",
        )

    async def call(self, inputs):
        if not inputs:
            return None
        if is_knowledge_graph(inputs):
            entities_json = []
            relations_json = []
            outputs_schema = copy.deepcopy(inputs.get_schema())

            # Process entities
            for entity in inputs.get_nested_entity_list("entities"):
                embedded_entity = await self._embed_entity(entity)
                if embedded_entity:
                    entities_json.append(embedded_entity.get_json())

                    # Update schema definitions
                    if outputs_schema.get("$defs"):
                        entity_label = entity.get("label")
                        if entity_label and entity_label in outputs_schema["$defs"]:
                            embedded_schema = embedded_entity.get_schema()
                            if embedded_schema.get("properties"):
                                outputs_schema["$defs"][entity_label][
                                    "properties"
                                ].update(embedded_schema["properties"])
            # Process relations
            for relation in inputs.get_nested_entity_list("relations"):
                embedded_relation = await self._embed_relation(relation)
                if embedded_relation:
                    relations_json.append(embedded_relation.get_json())

                    embedded_schema = embedded_relation.get_schema()
                    if embedded_schema.get("$defs") and outputs_schema.get("$defs"):
                        for def_key, def_value in embedded_schema["$defs"].items():
                            if def_key in outputs_schema["$defs"]:
                                # Merge properties if they exist
                                if def_value.get("properties") and outputs_schema[
                                    "$defs"
                                ][def_key].get("properties"):
                                    outputs_schema["$defs"][def_key]["properties"].update(
                                        def_value["properties"]
                                    )
                                else:
                                    outputs_schema["$defs"][def_key] = def_value

            # Update output JSON
            outputs_json = inputs.get_json()
            outputs_json.update({"entities": entities_json, "relations": relations_json})
            return JsonDataModel(
                json=outputs_json,
                schema=outputs_schema,
                name=inputs.name + "_embedded",
            )

        elif is_entities(inputs):
            entities_json = []
            outputs_schema = copy.deepcopy(inputs.get_schema())

            # Process all entities and collect schema updates
            for entity in inputs.get_nested_entity_list("entities"):
                embedded_entity = await self._embed_entity(entity)
                if embedded_entity:
                    entities_json.append(embedded_entity.get_json())

                    # Update schema definitions
                    if outputs_schema.get("$defs"):
                        entity_label = entity.get("label")
                        if entity_label and entity_label in outputs_schema["$defs"]:
                            embedded_schema = embedded_entity.get_schema()
                            if embedded_schema.get("properties"):
                                outputs_schema["$defs"][entity_label][
                                    "properties"
                                ].update(embedded_schema["properties"])

            # Update output JSON with embedded entities
            outputs_json = inputs.get_json()
            outputs_json.update({"entities": entities_json})

            return JsonDataModel(
                json=outputs_json,
                schema=outputs_schema,
                name=inputs.name + "_embedded",
            )

        elif is_relations(inputs):
            relations_json = []
            outputs_schema = copy.deepcopy(inputs.get_schema())

            # Process all relations
            for relation in inputs.get_nested_entity_list("relations"):
                embedded_relation = await self._embed_relation(relation)
                if embedded_relation:
                    relations_json.append(embedded_relation.get_json())

                    # Merge schema definitions from embedded relation
                    embedded_schema = embedded_relation.get_schema()
                    if embedded_schema.get("$defs") and outputs_schema.get("$defs"):
                        for def_key, def_value in embedded_schema["$defs"].items():
                            if def_key in outputs_schema["$defs"]:
                                # Merge properties if they exist
                                if def_value.get("properties") and outputs_schema[
                                    "$defs"
                                ][def_key].get("properties"):
                                    outputs_schema["$defs"][def_key]["properties"].update(
                                        def_value["properties"]
                                    )
                                else:
                                    outputs_schema["$defs"][def_key] = def_value

            # Update output JSON
            outputs_json = inputs.get_json()
            outputs_json.update({"relations": relations_json})

            return JsonDataModel(
                json=outputs_json,
                schema=outputs_schema,
                name=inputs.name + "_embedded",
            )
        elif is_relation(inputs):
            return await self._embed_relation(inputs)
        elif is_entity(inputs):
            return await self._embed_entity(inputs)
        else:
            return None

    async def compute_output_spec(self, inputs):
        return inputs.clone(name=inputs.name + "_embedded")

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
````
