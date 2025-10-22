import synalinks
import asyncio

from knowledge_graph_schema import City, Country, Place, Event
from knowledge_graph_schema import IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn


class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The answer to the user query",
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
        wipe_on_start=False,
    )

    inputs = synalinks.Input(data_model=Query)
    query_result = await synalinks.TripletRetriever(
        entity_models=[City, Country, Place, Event],
        relation_models=[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn],
        knowledge_base=knowledge_base,
        language_model=language_model,
        return_inputs=True,
        return_query=True,
    )(inputs)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
        instructions=[
            "Your task is to answer in natural language to the query based on the results of the search",
            "If the result of the search is not relevant, just say that you don't know",
        ],
        return_inputs=True,
    )(query_result)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_kag",
        description="A simple KAG program",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/knowledge/retrieval",
        show_trainable=True,
        show_schemas=True,
    )

    result = await program(Query(query="What is the French capital?"))

    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
