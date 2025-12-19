import synalinks
import asyncio

synalinks.enable_logging()


class Query(synalinks.DataModel):
    query: str


class Answer(synalinks.DataModel):
    answer: str


class AnswerWithThinking(synalinks.DataModel):
    thinking: str
    answer: str


# Logical Or

# When a two data models are provided, the logical or perform a concatenation
# of the two data models. However when given a `None`, it ignore it to give
# you the one that isn't None.

# This behavior can be summarized in the following truth table:

# Truth Table:

# | `x1`   | `x2`   | Logical Or (`|`) |
# | ------ | ------ | ---------------- |
# | `x1`   | `x2`   | `x1 + x2`        |
# | `x1`   | `None` | `x1`             |
# | `None` | `x2`   | `x2`             |
# | `None` | `None` | `None`           |

answer = Answer(answer="Toulouse") | None

print(answer.prettify_json())
# {
#   "answer": "Toulouse"
# }

answer = None | AnswerWithThinking(
    thinking=(
        "LAAS CNRS (Laboratoire d'Analyse et d'Architecture des Systèmes) is located in "
        "Toulouse and is renowned for its research in robotics."
        " Toulouse is also widely recognized as a central hub for aeronautics and"
        " space in Europe. It houses the headquarters of Airbus and several "
        "important aerospace research centers."
    ),
    answer="Toulouse",
)

print(answer.prettify_json())
# {
#   "thinking": "LAAS CNRS (Laboratoire d'Analyse et d'Architecture des
# Syst\u00e8mes) is located in Toulouse and is renowned for its research
# in robotics. Toulouse is also widely recognized as a central hub for
# aeronautics and space in Europe. It houses the headquarters of Airbus
# and several important aerospace research centers.",
#   "answer": "Toulouse"
# }

# Why is that useful ? Let's explain it with an example,
# imagine you want an adaptative system that is able to
# answer shortly, or take more time to "think" before answering
# depending on the question difficulty.
#
# Example:


async def main():
    language_model = synalinks.LanguageModel(model="ollama/mistral")

    inputs = synalinks.Input(data_model=Query)
    answer_without_thinking, answer_with_thinking = await synalinks.Branch(
        question="Evaluate the difficulty of the query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
        language_model=language_model,
        # We can optionally return the decision,
        # in Synalinks there is no black-box component!
        # Every LM inference, can be returned
        # for evaluation or explainability
        return_decision=False,
    )(inputs)

    # The outputs is the answer without thinking OR the answer with thinking
    outputs = answer_without_thinking | answer_with_thinking

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="adaptative_qa",
        description="A program that take the time to think if the query is difficult to answer",
    )

    answer = await program(
        Query(query="What is French city of robotics and aeronautics?")
    )

    print(answer.prettify_json())


# {
#   "thinking": "The answer to the given query involves finding a city in
# France that is known for robotics and aeronautics. While there might be
# several cities that have significant presence in these fields, Toulouse
# is one of the most renowned due to the presence of well-established
# institutions like EADS (European Aeronautic Defence and Space Company),
# IRIT (Institut de Recherche en Informatique pour le Traitement Automatique des Images)
# and LAAS CNRS (Laboratoire d'Analyse et d'Architecture des Syst\u00e8mes).",
#   "answer": "Toulouse"
# }

if __name__ == "__main__":
    asyncio.run(main())
