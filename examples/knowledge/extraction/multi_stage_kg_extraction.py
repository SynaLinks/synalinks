import synalinks
import asyncio
from typing import List

from knowledge_graph_schema import City, Country, Place, Event
from knowledge_graph_schema import IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn
from knowledge_dataset import Document, load_data


class Cities(synalinks.Entities):
    entities: List[City] = synalinks.Field(
        description="A list exclusively containing city entities, such as 'Tokyo' or 'London'.",
    )


class Countries(synalinks.Entities):
    entities: List[Country] = synalinks.Field(
        description="A list exclusively containing country entities, such as 'Japan' or 'United Kingdom'.",
    )


class Places(synalinks.Entities):
    entities: List[Place] = synalinks.Field(
        description="A list exclusively containing place entities, which could be landmarks or points of interest, such as 'Mount Fuji' or 'Big Ben'.",
    )


class Events(synalinks.Entities):
    entities: List[Event] = synalinks.Field(
        description="A list exclusively containing event entities, such as 'Olympic Games' or 'Coachella Festival'.",
    )


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
    cities = await synalinks.Generator(
        data_model=Cities,
        language_model=language_model,
    )(inputs)
    countries = await synalinks.Generator(
        data_model=Countries,
        language_model=language_model,
    )(inputs)
    places = await synalinks.Generator(
        data_model=Places,
        language_model=language_model,
    )(inputs)
    events = await synalinks.Generator(
        data_model=Events,
        language_model=language_model,
    )(inputs)

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

    entities = await synalinks.Or()([cities, countries, places, events])

    entities = entities.factorize()

    relations = await synalinks.Or()(
        [
            is_capital_of_relations,
            is_located_in_relations,
            is_city_of_relations,
            took_place_in_relations,
        ]
    )
    relations = relations.factorize()

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
        name="multi_stage_extraction",
        description="A multi stage KG extraction pipeline",
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
