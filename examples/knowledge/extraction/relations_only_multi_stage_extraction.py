import synalinks
import asyncio
from typing import List

from knowledge_graph_schema import City, Country, Place, Event
from knowledge_graph_schema import IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn
from knowledge_dataset import Document, load_data

FOLDER = "examples/knowledge/extraction"

synalinks.clear_session()


class IsCapitalOfRelations(synalinks.Relations):
    relations: List[IsCapitalOf] = synalinks.Field(
        description="A list of relations specifically describing capital-city relationships between city and country entities.",
    )


class IsCityOfRelations(synalinks.Relations):
    relations: List[IsCityOf] = synalinks.Field(
        description="A list of relations specifically describing the association of cities as part of countries.",
    )


class IsLocatedInRelations(synalinks.Relations):
    relations: List[IsLocatedIn] = synalinks.Field(
        description="A list of relations specifically describing the geographical containment of places within cities or countries.",
    )


class TookPlaceInRelations(synalinks.Relations):
    relations: List[TookPlaceIn] = synalinks.Field(
        description="A list of relations specifically describing the occurrence of events within cities or countries.",
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

    is_capital_of_relations = await synalinks.Generator(
        data_model=IsCapitalOfRelations,
        language_model=language_model,
    )(inputs)
    is_located_in_relations = await synalinks.Generator(
        data_model=IsLocatedInRelations,
        language_model=language_model,
    )(inputs)
    is_city_of_relations = await synalinks.Generator(
        data_model=IsCityOfRelations,
        language_model=language_model,
    )(inputs)
    took_place_in_relations = await synalinks.Generator(
        data_model=TookPlaceInRelations,
        language_model=language_model,
    )(inputs)

    relations = await synalinks.Or()(
        [
            is_capital_of_relations,
            is_located_in_relations,
            is_city_of_relations,
            took_place_in_relations,
        ]
    )
    relations = relations.factorize()

    embedded_relations = await synalinks.Embedding(
        embedding_model=embedding_model,
        in_mask=["name"],
    )(relations)

    updated_relations = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded_relations)

    outputs = updated_relations

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="relations_only_multi_stage_extraction",
        description="A multi stage KG extraction pipeline that only extract the relations",
    )

    synalinks.utils.plot_program(
        program,
        to_folder=FOLDER,
        show_trainable=True,
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
