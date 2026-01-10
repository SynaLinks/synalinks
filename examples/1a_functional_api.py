"""
# Lesson 1a: The Functional API

Welcome to your first Synalinks lesson! In this tutorial, you will learn
how to build AI applications using the Functional API - the most intuitive
and recommended approach for creating programs.

## What is Synalinks?

Synalinks is a framework for building AI applications powered by Large Language
Models (LLMs). Think of it like building with LEGO blocks - you connect
different pieces (called "modules") together to create something useful.

## Core Concepts

### 1. Programs and Modules

A **Program** in Synalinks is like a recipe - it defines the steps your AI
application will follow. Each step is performed by a **Module**, which is
a reusable building block.

```mermaid
graph LR
    Input --> Module1[Module 1] --> Module2[Module 2] --> Output
```

### 2. Data Models

Data flows through your program in structured formats called **DataModels**.
Think of them as blueprints that define what information looks like:

```python
class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user's question")
```

### 3. The Functional API

The Functional API lets you build programs by:
1. Creating an Input placeholder
2. Passing it through modules (like connecting pipes)
3. Wrapping everything in a Program

```python
# Step 1: Define where data enters
inputs = synalinks.Input(data_model=Query)

# Step 2: Pass through a module (Generator uses an LLM to create output)
outputs = await synalinks.Generator(
    data_model=Answer,
    language_model=language_model,
)(inputs)

# Step 3: Create the program
program = synalinks.Program(inputs=inputs, outputs=outputs)
```

## What You'll Build

In this lesson, you'll create a "Chain of Thought" program - an AI that
shows its reasoning step-by-step before giving an answer. This technique
helps LLMs produce more accurate responses.

## Running the Example

```bash
# Make sure you have set up your API key in .env file
uv run python examples/1a_functional_api.py
```

## Program Visualization

![chain_of_thought](../assets/examples/chain_of_thought.png)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Your Data Models
# =============================================================================
# Data Models are like forms that define what information your program
# expects as input and what it will produce as output.
#
# Each field has:
#   - A name (e.g., "query")
#   - A type (e.g., str for text)
#   - A description (helps the LLM understand what to fill in)


class Query(synalinks.DataModel):
    """The input to our program - a user's question."""

    query: str = synalinks.Field(
        description="The user query",
    )


class AnswerWithThinking(synalinks.DataModel):
    """The output from our program - reasoning + final answer.

    By asking the LLM to show its thinking, we get better answers.
    This is called "Chain of Thought" prompting.
    """

    thinking: str = synalinks.Field(
        description="Your step by step thinking",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )


# =============================================================================
# STEP 2: Build and Run the Program
# =============================================================================


async def main():
    # Load environment variables (like your API key) from .env file
    load_dotenv()

    # Enable observability for tracing (view traces at http://localhost:5000)
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="lesson_1a_functional_api",
    )

    # -------------------------------------------------------------------------
    # 2.1: Configure the Language Model
    # -------------------------------------------------------------------------
    # The LanguageModel is the AI brain that will process our requests.
    # We're using OpenAI's GPT-4.1 model here.
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # -------------------------------------------------------------------------
    # 2.2: Build the Program with the Functional API
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Building a Chain-of-Thought Program")
    print("=" * 60)

    # Create the input placeholder
    # This tells Synalinks what kind of data will enter the program
    inputs = synalinks.Input(data_model=Query)

    # Create a Generator module and connect it to the input
    # The Generator uses the LLM to transform input into output
    outputs = await synalinks.Generator(
        data_model=AnswerWithThinking,  # What to produce
        language_model=language_model,  # Which AI to use
    )(inputs)  # <-- This connects the module to our input

    # Wrap everything into a Program
    # Now we have a reusable AI application!
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="chain_of_thought",
        description="Useful to answer in a step by step manner.",
    )

    # Generate a visualization of our program (optional but helpful!)
    synalinks.utils.plot_program(
        program,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    # -------------------------------------------------------------------------
    # 2.3: Run the Program
    # -------------------------------------------------------------------------
    print("\nRunning the program...")
    print("-" * 60)

    # Call the program like a function, passing in our query
    result = await program(
        Query(query="What are the key aspects of human cognition?"),
    )

    # Display the result in a nicely formatted way
    print("\nResult:")
    print(result.prettify_json())

    # -------------------------------------------------------------------------
    # 2.4: Understanding the Output
    # -------------------------------------------------------------------------
    # The result is a JsonDataModel containing:
    #   - thinking: The LLM's step-by-step reasoning
    #   - answer: The final answer
    #
    # JsonDataModel supports direct field access - no need for get_json()!
    print("\n" + "=" * 60)
    print("Accessing individual fields:")
    print("=" * 60)
    print(f"\nThinking: {result['thinking'][:100]}...")
    print(f"\nAnswer: {result['answer'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
