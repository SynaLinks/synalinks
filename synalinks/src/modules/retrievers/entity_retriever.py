# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import SimilaritySearch
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum
from synalinks.src.modules import Module
from synalinks.src.modules.core.generator import Generator


def default_entity_retriever_instructions(entity_labels):
    """The default instructions for the entity retriever"""
    return f"""
Your task is to retrive entities among the following entity labels: {entity_labels}.
First, decide step-by-step which entity label you need, then describe the entities you are looking for.
The `similarity search` field should be a short description of the entities to match.
""".strip()


@synalinks_export(
    [
        "synalinks.modules.EntityRetriever",
        "synalinks.EntityRetriever",
    ]
)
class EntityRetriever(Module):
    """Retrieve entities from a knowledge base, based on the embedding vector.

    This module is useful to implement vector-only (retrieval augmented generation) RAG
    systems, for KAG (knowledge augmented generation) systems see the
    `KnowledgeRetriever` module.

    If you give multiple entity models to this module, the LM will select the most
    suitable one to perform the search. Having multiple entity models to search
    for is an easy way to enhance the performance of you RAG system by having
    multiple indexes (one per entity model type).

    ```python
    import synalinks
    import asyncio
    from typing import Literal

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class Answer(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The answer to the user query",
        )

    class Document(synalinks.Entity):
        label: Literal["Document"]
        filename: str = synalinks.Field(
            description="The document's filename",
        )
        text: str = synalinks.Field(
            description="The document's text",
        )

    class Chunk(synalinks.Entity):
        label: Literal["Chunk"]
        text: str = synalinks.Field(
            description="The chunk's text",
        )

    class IsPartOf(synalinks.Relation):
        subj: Chunk
        label: Literal["IsPartOf"]
        obj: Document

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large"
    )

    knowledge_base = synalinks.KnowledgeBase(
        uri="neo4j://localhost:7687",
        entity_models=[Country, City],
        relation_models=[IsCapitalOf, IsCityOf],
        embedding_model=embedding_model,
        metric="cosine",
    )

    async def main():
        inputs = synalinks.Input(data_model=Query)
        x = await synalinks.EntityRetriever(
            entity_models=[Chunk],
            language_model=language_model,
            knowledge_base=knowledge_base,
        )(inputs)
        outputs = await synalinks.Generator(
            data_model=Answer,
            language_model=language_model,
        )(x)

        program = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="rag_program",
            description="A naive RAG program",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to use.
        language_model (LanguageModel): The language model to use.
        entity_models (list): The list of entities models to search for
            being a list of `Entity` data models.
        k (int): Maximum number of similar entities to return
            (Defaults to 10).
        threshold (float): Minimum similarity score for results.
            Entities with similarity below this threshold are excluded.
            Should be between 0.0 and 1.0 (Defaults to 0.5).
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        examples (list): The default list of examples, the examples
            are a list of tuples containing input/output JSON pairs.
        instructions (str): The default instructions being a string containing
            instructions for the language model.
        seed_instructions (list): Optional. A list of instructions to use as seed for the
            optimization. If not provided, use the default instructions as seed.
        temperature (float): Optional. The temperature for the LM call.
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False) (see `Generator`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False) (see `Generator`).
        return_inputs (bool): Optional. Whether or not to concatenate the inputs to
            the outputs (Default to True).
        return_query (bool): Optional. Whether or not to concatenate the search query to
            the outputs (Default to True).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        knowledge_base=None,
        language_model=None,
        entity_models=None,
        k=10,
        threshold=0.5,
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        temperature=0.0,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_inputs=True,
        return_query=True,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.knowledge_base = knowledge_base
        self.language_model = language_model
        self.k = k
        self.threshold = threshold
        self.prompt_template = prompt_template
        self.examples = examples
        if entity_models:
            node_labels = [
                entity_model.get_schema().get("title") for entity_model in entity_models
            ]
        else:
            node_labels = []
        if not instructions:
            instructions = default_entity_retriever_instructions(node_labels)
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_query = return_query
        self.schema = SimilaritySearch.get_schema()
        self.schema = dynamic_enum(
            schema=self.schema,
            prop_to_update="entity_label",
            labels=node_labels,
            description="The entity label to search for",
        )
        self.query_generator = Generator(
            schema=self.schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name=self.name + "_query_generator",
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        similarity_search_query = await self.query_generator(
            inputs,
            training=training,
        )
        if self.return_inputs:
            if self.return_query:
                return await ops.concat(
                    inputs,
                    await ops.concat(
                        similarity_search_query,
                        await ops.similarity_search(
                            similarity_search_query,
                            knowledge_base=self.knowledge_base,
                            k=self.k,
                            threshold=self.threshold,
                            name=self.name + "_similarity_search",
                        ),
                        name=self.name + "_similarity_search_with_query_and_inputs",
                    ),
                )
            else:
                return await ops.concat(
                    inputs,
                    await ops.similarity_search(
                        similarity_search_query,
                        knowledge_base=self.knowledge_base,
                        k=self.k,
                        threshold=self.threshold,
                        name=self.name + "_similarity_search",
                    ),
                    name=self.name + "_similarity_search_with_inputs",
                )
        else:
            if self.return_query:
                return await ops.concat(
                    similarity_search_query,
                    await ops.similarity_search(
                        similarity_search_query,
                        knowledge_base=self.knowledge_base,
                        k=self.k,
                        threshold=self.threshold,
                        name=self.name + "_similarity_search",
                    ),
                    name=self.name + "_similarity_search_with_query",
                )
            else:
                return await ops.similarity_search(
                    similarity_search_query,
                    knowledge_base=self.knowledge_base,
                    k=self.k,
                    threshold=self.threshold,
                    name=self.name + "_similarity_search",
                )

    def get_config(self):
        config = {
            "k": self.question,
            "threshold": self.labels,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs": self.return_inputs,
            "return_query": self.return_query,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        knowledge_base_config = {
            "knowledge_base": serialization_lib.serialize_synalinks_object(
                self.knowledge_base,
            )
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.teacher_language_model,
            )
        }
        entity_models_config = {
            "entity_models": [
                (
                    serialization_lib.serialize_synalinks_object(
                        entity_model.to_symbolic_data_model(
                            name=self.name + "_entity_model" + ("_{i}" if i > 0 else "")
                        )
                    )
                    if not is_symbolic_data_model(entity_model)
                    else serialization_lib.serialize_synalinks_object(entity_model)
                )
                for i, entity_model in enumerate(self.entity_models)
            ]
        }
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
            **entity_models_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base"),
        )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        entity_models_config = config.pop("entity_models")
        entity_models = [
            serialization_lib.deserialize_synalinks_object(entity_model)
            for entity_model in entity_models_config
        ]
        return cls(
            knowledge_base=knowledge_base,
            entity_models=entity_models,
            language_model=teacher_language_model,
            **config,
        )
