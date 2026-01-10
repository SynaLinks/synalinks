"""
# Input Guard Patterns

**Input Guards** protect your LM applications by filtering dangerous or unwanted
inputs BEFORE they reach the language model. This saves compute costs and
prevents prompt injection attacks.

## Why Input Guards Matter

Without input guards, every request hits your LLM - even malicious ones:

```mermaid
graph LR
    subgraph Without Guards
        A[Any Input] --> B[LLM]
        B --> C[Response]
    end
    subgraph With Input Guard
        D[Input] --> E{Input Guard}
        E -->|Safe| F[LLM]
        F --> G[Response]
        E -->|Unsafe| H[Warning]
    end
```

Input guards provide:

1. **Cost Savings**: Skip LLM calls entirely for blocked inputs
2. **Security**: Block prompt injection attempts
3. **Policy Enforcement**: Ensure inputs meet your requirements
4. **Graceful Degradation**: Return helpful warnings instead of errors

## The XOR and OR Operators

For input guards, we use two key operators:

### XOR (^): Computation Bypass

XOR returns None if both operands have values. When a guard triggers (returns
a warning), XOR with the input produces None, blocking downstream computation:

```mermaid
graph LR
    A[warning] --> C["warning ^ inputs"]
    B[inputs] --> C
    C --> D{Result}
    D -->|"warning exists"| E["None (blocked)"]
    D -->|"warning is None"| F["inputs (pass-through)"]
```

| warning | inputs | warning ^ inputs |
|---------|--------|------------------|
| None | data | data (pass-through) |
| data | data | None (blocked) |

### OR (|): Result Selection

OR returns the first non-None value, perfect for choosing between warning
and answer:

| warning | answer | warning \\| answer |
|---------|--------|-------------------|
| None | data | data (use answer) |
| data | None | data (use warning) |
| data | data | merged (warning fields take priority) |

## Input Guard Pattern

The complete input guard pattern:

```mermaid
graph LR
    A[inputs] --> B[InputGuard]
    B --> C[warning]
    A --> D["warning ^ inputs"]
    C --> D
    D --> E["guarded_inputs"]
    E --> F[Generator]
    F --> G[answer]
    C --> H["warning | answer"]
    G --> H
    H --> I[output]
```

**Flow when input is BLOCKED:**
1. Guard returns warning (not None)
2. XOR: warning ^ inputs = None (blocks input)
3. Generator receives None, returns None
4. OR: warning | None = warning (use warning)

**Flow when input is SAFE:**
1. Guard returns None (no warning)
2. XOR: None ^ inputs = inputs (pass-through)
3. Generator processes inputs, returns answer
4. OR: None | answer = answer (use answer)

## Building an Input Guard

An input guard is a custom Module that:
- Returns `None` when input is safe (no warning)
- Returns a warning DataModel when input should be blocked

```python
import synalinks

class InputGuard(synalinks.Module):
    \"\"\"Guard that blocks inputs containing blacklisted words.\"\"\"

    def __init__(self, blacklisted_words, warning_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(self, inputs, training=False):
        \"\"\"Return warning if blocked, None otherwise.\"\"\"
        if inputs is None:
            return None

        # Check the query field for blacklisted words
        query = inputs.get("query", "").lower()

        for word in self.blacklisted_words:
            if word.lower() in query:
                # Return warning - this will trigger the guard
                return Warning(message=self.warning_message).to_json_data_model()

        # Input is safe - return None to pass through
        return None

    async def compute_output_spec(self, inputs, training=False):
        \"\"\"Define output schema.\"\"\"
        return Warning.to_symbolic_data_model(name=self.name)
```

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")

class Warning(synalinks.DataModel):
    message: str = synalinks.Field(description="Warning message")

class InputGuard(synalinks.Module):
    def __init__(self, blacklisted_words, warning_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(self, inputs, training=False):
        if inputs is None:
            return None
        query = inputs.get("query", "").lower()
        for word in self.blacklisted_words:
            if word.lower() in query:
                return Warning(message=self.warning_message).to_json_data_model()
        return None

    async def compute_output_spec(self, inputs, training=False):
        return Warning.to_symbolic_data_model(name=self.name)

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # Build the guarded program
    inputs = synalinks.Input(data_model=Query)

    # Guard checks for blacklisted words
    warning = await InputGuard(
        blacklisted_words=["hack", "exploit", "forbidden"],
        warning_message="I cannot process this request.",
    )(inputs)

    # XOR: If warning exists, block the input
    guarded_inputs = warning ^ inputs

    # Generator only runs if guarded_inputs is not None
    answer = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(guarded_inputs)

    # OR: Return warning if it exists, otherwise return answer
    outputs = warning | answer

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="input_guarded_qa",
    )

    # Test blocked input
    result = await program(Query(query="How do I hack into systems?"))
    print(f"Blocked: {result}")  # Shows warning

    # Test safe input
    result = await program(Query(query="What is the capital of France?"))
    print(f"Safe: {result}")  # Shows answer

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Takeaways

- **XOR (^) for Blocking**: When guard returns a warning, XOR with input
  produces None, preventing downstream computation.

- **OR (|) for Selection**: OR returns the first non-None value, choosing
  between warning and answer.

- **None Propagation**: Modules receiving None inputs skip execution and
  return None, enabling efficient short-circuiting.

- **Custom Guards**: Inherit from `synalinks.Module`, return None when safe
  or a warning DataModel when blocked.

- **Cost Savings**: Blocked inputs never reach the LLM, saving API costs.

## API References

- [Module](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/The%20Module%20base%20class/)
- [JSON Ops](https://synalinks.github.io/synalinks/Synalinks%20API/Ops%20API/JSON%20Ops/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User query."""

    query: str = synalinks.Field(description="User query")


class Answer(synalinks.DataModel):
    """Answer to the query."""

    answer: str = synalinks.Field(description="The answer")


class Warning(synalinks.DataModel):
    """Warning message when input is blocked."""

    message: str = synalinks.Field(description="Warning message")


# =============================================================================
# Input Guard Module
# =============================================================================


class InputGuard(synalinks.Module):
    """Guard that blocks inputs containing blacklisted words.

    Returns None when input is safe, or a Warning when input should be blocked.
    """

    def __init__(self, blacklisted_words, warning_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        """Return warning if blocked, None otherwise."""
        if inputs is None:
            return None

        query = inputs.get("query", "").lower()

        for word in self.blacklisted_words:
            if word.lower() in query:
                return Warning(message=self.warning_message).to_json_data_model()

        return None

    async def compute_output_spec(
        self,
        inputs: synalinks.SymbolicDataModel,
        training: bool = False,
    ) -> synalinks.SymbolicDataModel:
        """Define output schema."""
        return Warning.to_symbolic_data_model(name=self.name)

    def get_config(self):
        """Serialization config."""
        return {
            "name": self.name,
            "blacklisted_words": self.blacklisted_words,
            "warning_message": self.warning_message,
        }


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_9_input_guard",
    )

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # Build Input Guarded Program
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Building Input Guarded Program")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Guard checks for blacklisted words
    warning = await InputGuard(
        blacklisted_words=["hack", "exploit", "forbidden"],
        warning_message="I cannot process this request due to policy restrictions.",
    )(inputs)

    # XOR: If warning exists, block the input (returns None)
    guarded_inputs = warning ^ inputs

    # Generator only runs if guarded_inputs is not None
    answer = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(guarded_inputs)

    # OR: Return warning if it exists, otherwise return answer
    outputs = warning | answer

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="input_guarded_qa",
    )

    print("\nProgram built successfully!")
    program.summary()

    # -------------------------------------------------------------------------
    # Test Blocked Input
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 1: Blocked Input (contains 'hack')")
    print("=" * 60)

    result = await program(Query(query="How do I hack into computer systems?"))
    print(f"\nQuery: 'How do I hack into computer systems?'")
    print(f"Result: {result.get_json()}")

    # -------------------------------------------------------------------------
    # Test Safe Input
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: Safe Input")
    print("=" * 60)

    result = await program(Query(query="What is the capital of France?"))
    print(f"\nQuery: 'What is the capital of France?'")
    print(f"Result: {result.get_json()}")

    # -------------------------------------------------------------------------
    # Test Another Blocked Input
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3: Another Blocked Input (contains 'forbidden')")
    print("=" * 60)

    result = await program(Query(query="Tell me about forbidden topics"))
    print(f"\nQuery: 'Tell me about forbidden topics'")
    print(f"Result: {result.get_json()}")


if __name__ == "__main__":
    asyncio.run(main())
