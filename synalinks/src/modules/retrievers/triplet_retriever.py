# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import TripletSearch
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum
from synalinks.src.modules import Module
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought


def default_triplet_retriever_instructions(entity_labels, relation_labels):
    """The default instructions for the triplet retriever"""
    return f"""
Your task is to retrive triplets among the following entity labels: {entity_labels} and relation labels: {relation_labels}.
Think about the triplet you are looking for, which relation label do you need, then the subject and object label.
The similarity search parameters should be a short natural language string describing the entities to match.
Remember to replace the similarity search with `?` for the entity you are looking for.
""".strip()


@synalinks_export(
    [
        "synalinks.modules.TripletRetriever",
        "synalinks.TripletRetriever",
    ]
)
class TripletRetriever(Module):
    """Retrieve triplets using a hybrid neuro-symbolic approach.

    Unlike the Text2Cypher approach, this retriever is 100%
    guaranteed to generate a valid Cypher query **every time**.

    It doesn't need to have the graph schema in the prompt, thus
    helping the language models by avoiding prompt confusion, because
    the nodes and relation labels are enforced dynamically
    using **constrained structured output** (similar to the `Decision` module).

    It works by using the language model to infer the subject, object and
    relation labels to search for, along with a similarity search
    field for the object and subject triplets.

    These parameters are then used to *programmatically create a valid Cypher query*.

    This approach not only ensures the syntactical correctness of the
    Cypher query but also protects the graph database from Cypher
    injections that could arise from an adversarial prompt injection.

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

    class Country(synalinks.Entity):
        label: Literal["Country"]
        name: str = synalinks.Field(
            description="The country's name",
        )

    class City(synalinks.Entity):
        label: Literal["City"]
        name: str = synalinks.Field(
            description="The city's name",
        )

    class IsCapitalOf(synalinks.Relation):
        subj: City
        label: Literal["IsCapitalOf"]
        obj: Country

    class IsCityOf(synalinks.Relation):
        subj: City
        label: Literal["IsCityOf"]
        obj: Country

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
        x = await synalinks.KnowledgeRetriever(
            entity_models=[Country, City],
            relation_models=[IsCityOf, IsCapitalOf]
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
            name="kag_program",
            description="A simple KAG program",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to use.
        language_model (LanguageModel): The language model to use.
        entity_models (list): The list of entities models to search for
            being a list of `Entity` data models.
        relation_models (list): The list of relations models to seach for.
            being a list of `Relation` data models.
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
        relation_models=None,
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

        if relation_models:
            relation_labels = [
                relation_model.get_schema().get("title")
                for relation_model in relation_models
            ]
        else:
            relation_labels = []
        if not instructions:
            instructions = default_triplet_retriever_instructions(
                node_labels, relation_labels
            )
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_query = return_query

        self.schema = TripletSearch.get_schema()

        self.schema = dynamic_enum(
            schema=self.schema,
            prop_to_update="subject_label",
            labels=node_labels,
            description="The subject label to match",
        )

        self.schema = dynamic_enum(
            schema=self.schema,
            prop_to_update="relation_label",
            labels=relation_labels,
            description="The relation label to match",
        )

        self.schema = dynamic_enum(
            schema=self.schema,
            prop_to_update="object_label",
            labels=node_labels,
            description="The object label to match",
        )

        self.query_generator = ChainOfThought(
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
            name="query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        triplet_search_query = await self.query_generator(
            inputs,
            training=training,
        )
        if self.return_inputs:
            if self.return_query:
                return await ops.logical_and(
                    inputs,
                    await ops.logical_and(
                        triplet_search_query,
                        await ops.triplet_search(
                            triplet_search_query,
                            knowledge_base=self.knowledge_base,
                            k=self.k,
                            threshold=self.threshold,
                            name="similarity_search_" + self.name,
                        ),
                        name="similarity_search_with_query_and_inputs_" + self.name,
                    ),
                )
            else:
                return await ops.logical_and(
                    inputs,
                    await ops.triplet_search(
                        triplet_search_query,
                        knowledge_base=self.knowledge_base,
                        k=self.k,
                        threshold=self.threshold,
                        name="similarity_search_" + self.name,
                    ),
                    name="similarity_search_with_inputs_" + self.name,
                )
        else:
            if self.return_query:
                return await ops.logical_and(
                    triplet_search_query,
                    await ops.triplet_search(
                        triplet_search_query,
                        knowledge_base=self.knowledge_base,
                        k=self.k,
                        threshold=self.threshold,
                        name="similarity_search_" + self.name,
                    ),
                    name="similarity_search_with_query_" + self.name,
                )
            else:
                return await ops.triplet_search(
                    triplet_search_query,
                    knowledge_base=self.knowledge_base,
                    k=self.k,
                    threshold=self.threshold,
                    name="similarity_search_" + self.name,
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
                self.language_model,
            )
        }
        entity_models_config = {
            "entity_models": [
                (
                    serialization_lib.serialize_synalinks_object(
                        entity_model.to_symbolic_data_model(
                            name="entity_model" + (f"_{i}_" if i > 0 else "_") + self.name
                        )
                    )
                    if not is_symbolic_data_model(entity_model)
                    else serialization_lib.serialize_synalinks_object(entity_model)
                )
                for i, entity_model in enumerate(self.entity_models)
            ]
        }
        relation_models_config = {
            "relation_models": [
                (
                    serialization_lib.serialize_synalinks_object(
                        relation_model.to_symbolic_data_model(
                            name="relation_model" + (f"_{i}_" if i > 0 else "_") + self.name
                        )
                    )
                    if not is_symbolic_data_model(relation_model)
                    else serialization_lib.serialize_synalinks_object(relation_model)
                )
                for i, relation_model in enumerate(self.relation_models)
            ]
        }
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
            **entity_models_config,
            **relation_models_config,
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
        relation_models_config = config.pop("relation_models")
        relation_models = [
            serialization_lib.deserialize_synalinks_object(relation_model)
            for relation_model in relation_models_config
        ]
        return cls(
            knowledge_base=knowledge_base,
            entity_models=entity_models,
            relation_models=relation_models,
            language_model=teacher_language_model,
            **config,
        )
