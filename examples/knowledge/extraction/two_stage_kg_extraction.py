import synalinks
import asyncio
from typing import List, Union

from knowledge_graph_schema import City, Country, Place, Event
from knowledge_graph_schema import IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn
from knowledge_dataset import Document, load_data


class KnowledgeEntities(synalinks.Entities):
    entities: List[Union[City, Country, Place, Event]] = synalinks.Field(
        description=(
            "A comprehensive list containing various entities such as cities,"
            " countries, places, and events."
        ),
    )


class KnowledgeRelations(synalinks.Relations):
    relations: List[Union[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn]] = (
        synalinks.Field(
            description=(
                "A comprehensive list of relations including IsCapitalOf, IsLocatedIn,"
                " IsCityOf, and TookPlaceIn, which describe interactions and associations"
                " between entities."
            ),
        )
    )


async def main():
    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large",
    )

    knowledge_base = synalinks.KnowledgeBase(
        uri="memgraph://localhost:7687",
        entity_models=[City, Country, Place, Event],
        relation_models=[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn],
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=True,
    )

    inputs = synalinks.Input(data_model=Document)
    entities = await synalinks.Generator(
        data_model=KnowledgeEntities,
        language_model=language_model,
    )(inputs)

    # inputs_with_entities = inputs OR entities (See Control Flow tutorial)
    inputs_with_entities = inputs | entities
    relations = await synalinks.Generator(
        data_model=KnowledgeRelations,
        language_model=language_model,
    )(inputs_with_entities)

    # knowledge_graph = entities OR relations
    knowledge_graph = entities | relations

    embedded_knowledge_graph = await synalinks.Embedding(
        embedding_model=embedding_model,
        in_mask=["name"],
    )(knowledge_graph)

    updated_knowledge_graph = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded_knowledge_graph)

    outputs = updated_knowledge_graph

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="two_stage_extraction",
        description="A two stage KG extraction pipeline",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/knowledge/extraction",
        show_trainable=True,
    )

    dataset = load_data()

    print("Starting KG extraction...")
    await program.predict(dataset, batch_size=1)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
