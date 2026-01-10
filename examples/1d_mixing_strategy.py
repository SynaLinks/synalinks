"""
# Lesson 1d: The Mixing Strategy (Recommended)

You've learned three ways to build programs:
- **Functional API** (1a): Flexible graph building
- **Subclassing** (1b): Full control but more boilerplate
- **Sequential** (1c): Simplest for linear pipelines

Now, let's explore the **recommended approach**: mixing subclassing with the
Functional API. This gives you the best of both worlds!

## Why Use the Mixing Strategy?

| Approach | Encapsulation | Boilerplate | Flexibility |
|----------|---------------|-------------|-------------|
| Functional API | Low | Low | High |
| Subclassing | High | High | High |
| Sequential | Medium | Very Low | Low |
| **Mixing** | **High** | **Low** | **High** |

The mixing strategy provides:
1. **Encapsulation**: Your program is a reusable class
2. **No boilerplate**: No need for `call()`, `get_config()`, or `from_config()`
3. **Flexibility**: Full power of the Functional API inside

## The Pattern

```python
# Step 1: Define a reusable module using the mixing strategy
class MyModule(synalinks.Program):

    def __init__(self, language_model=None, ...):
        super().__init__()  # Initialize without inputs/outputs
        self.language_model = language_model  # Store config

    async def build(self, inputs):
        # Use Functional API to create the graph
        # `inputs` is a SymbolicDataModel (from the outer program)
        outputs = await synalinks.Generator(...)(inputs)

        # Re-initialize with inputs and outputs
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
        )

# Step 2: Use the module inside a functional program
my_module = MyModule(language_model=lm)

inputs = synalinks.Input(data_model=Query)  # Creates symbolic input
outputs = await my_module(inputs)           # Triggers build() with symbolic input

program = synalinks.Program(inputs=inputs, outputs=outputs)
```

## How It Works

1. **Define your class**: Implement `__init__()` and `build()` only
2. **Create an instance**: Store configuration (language models, settings)
3. **Use in Functional API**: Call the module with a symbolic `Input`
4. **build() is triggered**: Receives symbolic data, creates the graph

The key insight: **the mixing strategy creates reusable modules** that you
compose using the Functional API. The `build()` method receives symbolic
inputs when called during graph construction.

## Running the Example

```bash
uv run python examples/1d_mixing_strategy.py
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


class Query(synalinks.DataModel):
    """The input to our program - a user's question."""

    query: str = synalinks.Field(
        description="The user query",
    )


class AnswerWithThinking(synalinks.DataModel):
    """The output from our program - reasoning + final answer."""

    thinking: str = synalinks.Field(
        description="Your step by step thinking",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )


# =============================================================================
# STEP 2: Define Your Program Using the Mixing Strategy
# =============================================================================


class ChainOfThought(synalinks.Program):
    """A program that answers questions with step-by-step reasoning.

    This uses the MIXING STRATEGY: subclassing + Functional API.
    Notice how we DON'T implement call(), get_config(), or from_config()!
    """

    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=True,
    ):
        # Step 1: Initialize the base Program (without inputs/outputs yet)
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )

        # Step 2: Store configuration for later use in build()
        # These are NOT modules yet - just configuration!
        self.language_model = language_model

    async def build(self, inputs: synalinks.SymbolicDataModel) -> None:
        """Build the program graph using the Functional API.

        This method is called AUTOMATICALLY when the program is first used.
        You don't need to call it yourself!

        Args:
            inputs (SymbolicDataModel): A SymbolicDataModel representing
                    the input data model.
        """
        # Step 3: Use Functional API to create the computation graph
        # This is exactly like the Functional API (Lesson 1a)!
        outputs = await synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=self.language_model,
        )(inputs)

        # Step 4: Re-initialize as a Functional program
        # This tells Synalinks the complete graph structure
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )


# =============================================================================
# STEP 3: A More Complex Example - Multi-Step Reasoning
# =============================================================================


class Critique(synalinks.DataModel):
    """A critique of an answer."""

    issues: str = synalinks.Field(
        description="Any issues or problems with the answer",
    )
    is_correct: bool = synalinks.Field(
        description="Whether the answer appears correct",
    )


class RefinedAnswer(synalinks.DataModel):
    """A refined answer after self-critique."""

    original_answer: str = synalinks.Field(
        description="The original answer",
    )
    refinement: str = synalinks.Field(
        description="Any refinements or corrections",
    )
    final_answer: str = synalinks.Field(
        description="The final, refined answer",
    )


class SelfCritiquingReasoner(synalinks.Program):
    """A more complex program: reason, critique, then refine.

    This demonstrates building a multi-step pipeline with the mixing strategy.

    Flow: Query -> Think+Answer -> Critique -> Refine
    """

    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.language_model = language_model

    async def build(self, inputs: synalinks.SymbolicDataModel) -> None:
        """Build a multi-step reasoning pipeline."""

        # Step 1: Generate initial answer with thinking
        initial_answer = await synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=self.language_model,
            name="initial_reasoner",
        )(inputs)

        # Step 2: Critique the answer
        # Combine query + answer for context
        critique_input = inputs + initial_answer
        critique = await synalinks.Generator(
            data_model=Critique,
            language_model=self.language_model,
            name="self_critic",
        )(critique_input)

        # Step 3: Refine based on critique
        refine_input = critique_input + critique
        refined = await synalinks.Generator(
            data_model=RefinedAnswer,
            language_model=self.language_model,
            name="refiner",
        )(refine_input)

        # Initialize as Functional program
        super().__init__(
            inputs=inputs,
            outputs=refined,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )


# =============================================================================
# STEP 4: Run and Test
# =============================================================================


async def main():
    load_dotenv()

    # Enable observability for tracing (view traces at http://localhost:5000)
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="lesson_1d_mixing_strategy",
    )

    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # -------------------------------------------------------------------------
    # 4.1: Simple Example - ChainOfThought
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Simple Chain-of-Thought (Mixing Strategy)")
    print("=" * 60)

    # Create the mixed program as a reusable module
    chain_of_thought = ChainOfThought(
        language_model=language_model,
        name="chain_of_thought",
        description="Answers questions with step-by-step reasoning",
    )

    # Use it inside a functional program - THIS is the key!
    # The symbolic Input triggers the build() with symbolic data
    inputs = synalinks.Input(data_model=Query)
    outputs = await chain_of_thought(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="my_application",
        description="Application using Chain-of-Thought",
    )

    print(f"Program built: {program.built}")
    print(f"Inner module built: {chain_of_thought.built}")

    # Now call the program with actual data
    print("\nCalling program...")
    result = await program(
        Query(query="What is 15% of 80?"),
    )

    print("\nResult:")
    print(f"  Thinking: {result['thinking'][:80]}...")
    print(f"  Answer: {result['answer']}")

    # Generate visualization
    synalinks.utils.plot_program(
        program,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    # -------------------------------------------------------------------------
    # 4.2: Complex Example - Self-Critiquing Reasoner
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Self-Critiquing Reasoner (Multi-Step)")
    print("=" * 60)

    # Create the mixed program as a reusable module
    self_critic = SelfCritiquingReasoner(
        language_model=language_model,
        name="self_critiquing_reasoner",
        description="Reasons, critiques itself, then refines the answer",
    )

    # Encapsulate in a functional program
    inputs = synalinks.Input(data_model=Query)
    outputs = await self_critic(inputs)

    complex_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="self_critique_application",
    )

    result = await complex_program(
        Query(query="Is a tomato a fruit or a vegetable?"),
    )

    print("\nResult:")
    print(f"  Original: {result['original_answer'][:60]}...")
    print(f"  Refinement: {result['refinement'][:60]}...")
    print(f"  Final: {result['final_answer']}")

    # Generate visualization
    synalinks.utils.plot_program(
        complex_program,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    # -------------------------------------------------------------------------
    # 4.3: Key Advantages Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Advantages of the Mixing Strategy")
    print("=" * 60)

    print("""
1. ENCAPSULATION
   - Logic is neatly packaged in a class
   - Easy to share, import, and reuse

2. MINIMAL BOILERPLATE
   - Only implement __init__() and build()
   - No need for call(), get_config(), or from_config()

3. FLEXIBILITY
   - Full power of Functional API inside build()
   - Can create complex graphs with branches, parallel paths, etc.

4. COMPOSABILITY
   - Mixed programs work as modules in larger programs
   - Build complex applications by combining simple pieces

5. AUTOMATIC SERIALIZATION
   - The Functional API handles saving/loading automatically
   - No manual serialization code needed

This pattern is recommended for most production applications!
""")


if __name__ == "__main__":
    asyncio.run(main())
