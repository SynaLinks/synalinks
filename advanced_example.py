import synalinks
import asyncio

async def main():
    # Define our data models
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query about a mathematical calculation",
        )

    class ExtractionResult(synalinks.DataModel):
        expression: str = synalinks.Field(
            description="The mathematical expression extracted from the query",
        )
        explanation: str = synalinks.Field(
            description="Explanation of how the expression was extracted",
        )

    class CalculationResult(synalinks.DataModel):
        expression: str = synalinks.Field(
            description="The mathematical expression that was calculated",
        )
        result: float = synalinks.Field(
            description="The result of the calculation",
        )
        log: str = synalinks.Field(
            description="Log messages from the calculation",
        )

    class FinalResponse(synalinks.DataModel):
        original_query: str = synalinks.Field(
            description="The original user query",
        )
        answer: float = synalinks.Field(
            description="The final numerical answer",
        )
        explanation: str = synalinks.Field(
            description="A clear explanation of the calculation and result",
        )

    # Define our calculation function to be used by the Action module
    async def calculate(expression: str):
        """Calculate the result of a mathematical expression.

        Args:
            expression (str): The mathematical expression to calculate, such as
                '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
                parentheses, and spaces.
        """
        if not all(char in "0123456789+-*/^(). " for char in expression):
            return {
                "result": None,
                "log": "Error: invalid characters in expression",
            }
        try:
            # Replace ^ with ** for exponentiation
            expression = expression.replace("^", "**")
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

    # Initialize the language model
    language_model = synalinks.LanguageModel(
        model="ollama_chat/llama3",
    )

    # Create a sequential program with multiple steps
    program = synalinks.Sequential(
        [
            # First step: Input module
            synalinks.Input(
                data_model=Query,
            ),
            
            # Second step: Extract the mathematical expression
            synalinks.Generator(
                data_model=ExtractionResult,
                language_model=language_model,
                hints=[
                    "Extract only the mathematical expression from the query.",
                    "Make sure to convert words like 'square root' to the appropriate mathematical notation.",
                    "For 'square root', use the format 'sqrt(x)' or 'x^0.5'.",
                ],
            ),
            
            # Third step: Calculate the result using the Action module
            synalinks.Action(
                fn=calculate,
                language_model=language_model,
            ),
            
            # Fourth step: Generate a final response
            synalinks.Generator(
                data_model=FinalResponse,
                language_model=language_model,
            ),
        ],
        name="math_calculator",
        description="A program that extracts and calculates mathematical expressions.",
    )
    
    # Print a summary of the program
    program.summary()
    
    # Test with several examples
    examples = [
        "What is 25 plus 17?",
        "If I have 10 apples and give 3 to my friend, how many do I have left?",
        "What is the square root of 144?",
        "Calculate 5 raised to the power of 3",
    ]
    
    for i, query_text in enumerate(examples):
        print(f"\nExample {i+1}: {query_text}")
        result = await program(
            Query(query=query_text),
        )
        
        # Access the result data using get()
        original_query = result.get("original_query")
        answer = result.get("answer")
        explanation = result.get("explanation")
        
        print(f"Original Query: {original_query}")
        print(f"Answer: {answer}")
        print(f"Explanation: {explanation}")

if __name__ == "__main__":
    asyncio.run(main()) 