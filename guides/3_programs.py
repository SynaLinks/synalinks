"""
# Guide 3: Programs

Programs are the main abstraction for building LM-based applications in
Synalinks. This guide covers the four ways to create programs.

## Program Creation Strategies

### 1. Functional API (Recommended)

Chain module calls from Input to outputs:

```python
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
program = synalinks.Program(inputs=inputs, outputs=outputs, name="my_program")
```

### 2. Subclassing API

Override the `call()` method for custom logic:

```python
class MyProgram(synalinks.Program):
    async def call(self, inputs, training=False):
        return await self.generator(inputs, training=training)
```

### 3. Sequential API

Stack modules in sequence:

```python
program = synalinks.Sequential([module1, module2], input_data_model=Query)
```

### 4. Mixing Strategy

Combine subclassing with Functional API for reusable components.

## Running Programs

```python
# Single inference
result = await program(Query(query="What is 2+2?"))

# Batch inference
results = await program.predict([query1, query2, query3])
```

## Running the Example

```bash
uv run python guides/3_programs.py
```
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User question."""

    query: str = synalinks.Field(description="User question")


class ThinkingOutput(synalinks.DataModel):
    """Intermediate thinking output."""

    thinking: str = synalinks.Field(description="Step by step thinking")


class Answer(synalinks.DataModel):
    """Final answer."""

    answer: str = synalinks.Field(description="The final answer")


class AnswerWithThinking(synalinks.DataModel):
    """Answer with reasoning."""

    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# STEP 2: Subclassing API Example
# =============================================================================


class QAProgram(synalinks.Program):
    """A QA program using subclassing."""

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model
        self.generator = synalinks.Generator(
            data_model=Answer,
            language_model=language_model,
        )

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        return await self.generator(inputs, training=training)


# =============================================================================
# STEP 3: Mixing Strategy Example
# =============================================================================


class ChainOfThought(synalinks.Program):
    """Reusable chain-of-thought component using mixing strategy."""

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model

    async def build(self, inputs: synalinks.SymbolicDataModel) -> None:
        """Build the computation graph."""
        outputs = await synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=self.language_model,
        )(inputs)

        # Reinitialize with the built graph
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
        )


# =============================================================================
# STEP 4: Demonstrate All Strategies
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_3_programs",
    )

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # 4.1: Functional API
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Strategy 1: Functional API")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
    )(inputs)

    functional_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="functional_qa",
    )

    result = await functional_program(Query(query="What is 2+2?"))
    print(f"\nFunctional API Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # 4.2: Subclassing API
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 2: Subclassing API")
    print("=" * 60)

    subclass_program = QAProgram(language_model=lm, name="subclass_qa")
    result = await subclass_program(Query(query="What is 3+3?"))
    print(f"\nSubclassing API Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # 4.3: Sequential API
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 3: Sequential API")
    print("=" * 60)

    sequential_program = synalinks.Sequential(
        [
            synalinks.Generator(
                data_model=ThinkingOutput,
                language_model=lm,
            ),
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
            ),
        ],
        input_data_model=Query,
        name="sequential_qa",
    )

    result = await sequential_program(Query(query="What is 4+4?"))
    print(f"\nSequential API Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # 4.4: Mixing Strategy
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 4: Mixing Strategy")
    print("=" * 60)

    # Create reusable component
    cot = ChainOfThought(language_model=lm, name="cot_component")

    # Use it inside a functional program
    inputs = synalinks.Input(data_model=Query)
    outputs = await cot(inputs)

    mixing_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="mixing_qa",
    )

    result = await mixing_program(Query(query="What is 5+5?"))
    print(f"\nMixing Strategy Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # 4.5: Program Features
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Program Features")
    print("=" * 60)

    # Print summary
    print("\nProgram Summary:")
    functional_program.summary()

    # Save program
    functional_program.save("functional_qa.json")
    print("\nSaved program to functional_qa.json")

    # Load program
    loaded = synalinks.Program.load("functional_qa.json")
    print(f"Loaded program: {loaded.name}")

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. FUNCTIONAL API: Most flexible, recommended for most cases
2. SUBCLASSING: When you need custom logic in call()
3. SEQUENTIAL: For simple linear pipelines
4. MIXING STRATEGY: For reusable components
5. SAVE/LOAD: Programs are serializable to JSON
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
