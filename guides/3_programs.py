"""
# Programs

A **Program** in Synalinks is the fundamental unit of deployment and training.
Just as a function encapsulates logic in traditional programming, a Program
encapsulates the entire computation graph of your Language Model application,
from input to output, including all intermediate transformations.

## Why Programs Matter

In traditional LLM development, you write procedural code that calls APIs:

```mermaid
graph LR
    subgraph Traditional Approach
        A[Function] --> B[API Call 1]
        B --> C[Parse]
        C --> D[API Call 2]
        D --> E[Return]
    end
```

This approach has limitations: no training, no serialization, no visualization.

Synalinks Programs provide a **declarative computation graph**:

```mermaid
graph LR
    subgraph Synalinks Program
        A[Input DataModel] --> B[Module 1]
        B --> C[Module 2]
        C --> D[Output DataModel]
    end
    E[Training] -.-> B
    E -.-> C
    F[Save/Load] -.-> B
    F -.-> C
```

Programs provide:

1. **Trainability**: Optimize instructions and examples over time
2. **Serialization**: Save and load trained state
3. **Visualization**: Understand your computation graph
4. **Composability**: Nest programs within programs

## The Four Program Creation Strategies

Synalinks offers four distinct strategies for creating programs, each suited
to different use cases:

```mermaid
graph TD
    A[Program Creation] --> B[Functional API]
    A --> C[Subclassing API]
    A --> D[Sequential API]
    A --> E[Mixing Strategy]
    B --> F["Most Flexible<br>(Recommended)"]
    C --> G["Custom Logic<br>in call()"]
    D --> H["Simple Linear<br>Pipelines"]
    E --> I["Reusable<br>Components"]
```

### Strategy 1: The Functional API (Recommended)

The **Functional API** is the most powerful and flexible approach. You build
a computation graph by chaining module calls, starting from an `Input` node:

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    \"\"\"User question.\"\"\"
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    \"\"\"Answer with reasoning.\"\"\"
    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # Step 1: Define the entry point
    inputs = synalinks.Input(data_model=Query)

    # Step 2: Chain module calls (this builds the graph)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    # Step 3: Wrap in a Program
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
    )

    # Step 4: Use the program
    result = await program(Query(query="What is 2+2?"))
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

The Functional API excels at:

- **Parallel branches**: Multiple modules can process the same input
- **Complex routing**: Decisions and branches based on content
- **Merging**: Combining outputs from multiple paths

### Strategy 2: The Subclassing API

The **Subclassing API** gives you complete control over the execution logic.
You inherit from `synalinks.Program` and override the `call()` method:

```python
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The final answer")

class QAProgram(synalinks.Program):
    \"\"\"A custom QA program using subclassing.\"\"\"

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model
        # Create modules in __init__
        self.generator = synalinks.Generator(
            data_model=Answer,
            language_model=language_model,
        )

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        # Custom logic here
        return await self.generator(inputs, training=training)
```

Use the Subclassing API when you need:

- Custom logic that doesn't fit the functional paradigm
- State management beyond trainable variables
- Integration with external systems during execution

### Strategy 3: The Sequential API

The **Sequential API** is the simplest approach for linear pipelines where
each module feeds directly into the next:

```mermaid
graph LR
    A[Input] --> B[Module 1]
    B --> C[Module 2]
    C --> D[Module 3]
    D --> E[Output]
```

```python
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

class Thinking(synalinks.DataModel):
    thinking: str = synalinks.Field(description="Step by step thinking")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The final answer")

lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

# Simple linear pipeline using .add() method
program = synalinks.Sequential(
    name="sequential_qa",
    description="A sequential question-answering pipeline",
)
program.add(synalinks.Input(data_model=Query))
program.add(synalinks.Generator(data_model=Thinking, language_model=lm))
program.add(synalinks.Generator(data_model=Answer, language_model=lm))
```

The Sequential API is ideal for:

- Simple, linear processing pipelines
- Quick prototyping
- When each step naturally flows to the next

### Strategy 4: The Mixing Strategy

The **Mixing Strategy** combines subclassing with the Functional API to create
**reusable components** that can be used inside other programs:

```mermaid
graph TD
    subgraph Reusable Component
        A[build] --> B[Create Functional Graph]
        B --> C[Reinitialize as Program]
    end
    subgraph Main Program
        D[Input] --> E[Component]
        E --> F[More Processing]
        F --> G[Output]
    end
```

```python
import synalinks

class ChainOfThought(synalinks.Program):
    \"\"\"Reusable chain-of-thought component.\"\"\"

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model

    async def build(self, inputs: synalinks.SymbolicDataModel) -> None:
        \"\"\"Build the computation graph when first called.\"\"\"
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
```

The Mixing Strategy is powerful for:

- Creating library components
- Encapsulating complex sub-graphs
- Building a toolkit of reusable patterns

## Program Features

### Saving and Loading

Programs serialize their entire state to JSON:

```python
# Save a program
program.save("my_program.json")

# Load a program
loaded = synalinks.Program.load("my_program.json")
```

This includes all trainable variables (optimized instructions and examples).

### Program Summary

Inspect your program's structure:

```python
program.summary()
```

Output:
```
Program: qa_program
===============================
| Module          | Trainable |
|-----------------|-----------|
| Input           | No        |
| Generator       | Yes       |
===============================
Total parameters: 2
Trainable parameters: 2
```

### Batch Inference

Process multiple inputs efficiently:

```python
results = await program.predict([query1, query2, query3])
```

## Key Takeaways

- **Functional API**: The recommended approach for most use cases. Build
  computation graphs by chaining module calls from `Input` to outputs. Supports
  parallel branches, decisions, and complex routing.

- **Subclassing API**: Use when you need custom logic in the `call()` method.
  Gives you complete control but loses some declarative benefits.

- **Sequential API**: Perfect for simple linear pipelines where modules feed
  directly into each other. Minimal boilerplate.

- **Mixing Strategy**: Create reusable components that can be embedded in
  other programs. Best for building a library of patterns.

- **Serialization**: All programs can be saved to JSON and loaded back,
  preserving trained state and configuration.

- **Program.summary()**: Use this to inspect your program's structure and
  identify trainable modules.

## API References

- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [Sequential](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Sequential%20class/)
- [Input](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Input%20module/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Data Models
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
# Subclassing Example
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
# Mixing Strategy Example
# =============================================================================


class ChainOfThought(synalinks.Program):
    """Reusable chain-of-thought component."""

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model

    async def build(self, inputs: synalinks.SymbolicDataModel) -> None:
        outputs = await synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=self.language_model,
        )(inputs)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
        )


# =============================================================================
# Main Demonstration
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
    # Functional API
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
    # Subclassing API
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 2: Subclassing API")
    print("=" * 60)

    subclass_program = QAProgram(language_model=lm, name="subclass_qa")
    result = await subclass_program(Query(query="What is 3+3?"))
    print(f"\nSubclassing API Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # Sequential API
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 3: Sequential API")
    print("=" * 60)

    sequential_program = synalinks.Sequential(
        name="sequential_qa",
        description="A sequential question-answering pipeline",
    )
    sequential_program.add(synalinks.Input(data_model=Query))
    sequential_program.add(
        synalinks.Generator(data_model=ThinkingOutput, language_model=lm)
    )
    sequential_program.add(synalinks.Generator(data_model=Answer, language_model=lm))

    result = await sequential_program(Query(query="What is 4+4?"))
    print(f"\nSequential API Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # Mixing Strategy
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 4: Mixing Strategy")
    print("=" * 60)

    cot = ChainOfThought(language_model=lm, name="cot_component")

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
    # Program Features
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Program Features")
    print("=" * 60)

    print("\nProgram Summary:")
    functional_program.summary()

    functional_program.save("functional_qa.json")
    print("\nSaved program to functional_qa.json")

    loaded = synalinks.Program.load("functional_qa.json")
    print(f"Loaded program: {loaded.name}")

    if os.path.exists("functional_qa.json"):
        os.remove("functional_qa.json")


if __name__ == "__main__":
    asyncio.run(main())
