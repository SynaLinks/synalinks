import synalinks
import asyncio

from dotenv import load_dotenv

load_dotenv()

synalinks.enable_logging()


class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )


class NumericalFinalAnswer(synalinks.DataModel):
    final_answer: float = synalinks.Field(
        description="The correct final numerical answer",
    )


@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
            parentheses, and spaces.
    """
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {
            "result": None,
            "log": (
                "Error: invalid characters in expression. "
                "The expression can only contain numbers, operators (+, -, *, /),"
                " parentheses, and spaces NOT letters."
            ),
        }
    try:
        # Evaluate the mathematical expression safely
        result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
        return {
            "result": result,
            "log": "Successfully executed",
        }
    except Exception as e:
        return {
            "result": None,
            "log": f"Error: {e}",
        }


async def main():
    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    tools = [
        synalinks.Tool(calculate),
    ]

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=NumericalFinalAnswer,
        tools=tools,
        language_model=language_model,
        max_iterations=5,
        return_inputs_with_trajectory=True,
        autonomous=True,
    )(inputs)
    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_agent",
        description="A math agent",
    )

    input_query = Query(query="How much is 152648 + 485?")
    response = await agent(input_query)

    print(response.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
