import synalinks
import asyncio
from typing import List, Union

from dotenv import load_dotenv

load_dotenv()

from knowledge_graph_schema import City, Country, Place, Event
from knowledge_graph_schema import IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn
from knowledge_dataset import Document, load_data


class KnowledgeEntitiesAndRelations(synalinks.KnowledgeGraph):
    entities: List[Union[City, Country, Place, Event]] = synalinks.Field(
        description=(
            "A comprehensive list containing various entities such as cities,"
            " countries, places, and events."
        ),
    )
    relations: List[Union[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn]] = (
        synalinks.Field(
            description=(
                "A comprehensive list containing various entities such as cities,"
                " countries, places, and events."
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
    knowledge_graph = await synalinks.Generator(
        data_model=KnowledgeEntitiesAndRelations,
        language_model=language_model,
    )(inputs)

    embedded_knowledge_graph = await synalinks.Embedding(
        embedding_model=embedding_model,
        in_mask=["name"],
    )(knowledge_graph)

    outputs = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded_knowledge_graph)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="one_stage_extraction",
        description="A one stage KG extraction pipeline",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/knowledge/extraction",
        show_trainable=True,
    )

    dataset = load_data()

    await program.predict(dataset, batch_size=1)


if __name__ == "__main__":
    asyncio.run(main())
