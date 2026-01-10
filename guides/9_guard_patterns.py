"""
# Guide 9: Input/Output Guard Patterns

Guard patterns in Synalinks use XOR (^) and OR (|) operators to control
data flow and bypass computation when guards trigger.

## Key Concepts

### XOR (^) for Computation Bypass

When a guard returns a warning, XOR makes the other input None:
- `warning ^ inputs` → if warning exists, inputs becomes None
- This bypasses downstream modules (they receive None)

### OR (|) for Result Merging

OR returns the first non-None value or merges if both exist:
- `warning | answer` → returns warning if it exists, otherwise answer

## Guard Flow Patterns

### Input Guard Pattern

```mermaid
graph LR
    A[inputs] --> B[InputGuard]
    B --> C[warning]
    A --> D["warning ^ inputs"]
    C --> D
    D --> E[Generator]
    E --> F[answer]
    C --> G["warning | answer"]
    F --> G
    G --> H[output]
```

### Output Guard Pattern

```mermaid
graph LR
    A[inputs] --> B[Generator]
    B --> C[answer]
    C --> D[OutputGuard]
    D --> E{warning?}
    C --> F["(answer ^ warning) | warning"]
    E --> F
    F --> G[output]
```

## Custom Guard Module

Create a guard by inheriting from `synalinks.Module`:

```python
class ConversationalInputGuard(synalinks.Module):
    def __init__(self, blacklisted_words, warning_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(self, inputs, training=False):
        if not synalinks.is_chat_messages(inputs):
            raise ValueError("Input guard works only for ChatMessages")

        content = inputs["messages"][-1]["content"].lower()

        if any(bw.lower() in content for bw in self.blacklisted_words):
            return synalinks.ChatMessage(
                role=synalinks.ChatRole.ASSISTANT,
                content=self.warning_message,
            ).to_json_data_model()
        return None

    def compute_output_spec(self, inputs):
        return synalinks.ChatMessage.to_symbolic_data_model(name=self.name)
```

## Input Guard Program

```python
async def build_input_guarded_program(language_model):
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    # Check input for blacklisted words
    warning_msg = await ConversationalInputGuard(
        blacklisted_words=["forbidden", "blocked", "hack"],
        warning_message="I'm unable to comply with your request.",
    )(inputs)

    # XOR: if warning exists, inputs becomes None (bypassing generator)
    guarded_inputs = warning_msg ^ inputs

    # Generator only runs if guarded_inputs is not None
    answer = await synalinks.Generator(
        language_model=language_model,
    )(guarded_inputs)

    # OR: return warning if it exists, otherwise return answer
    outputs = warning_msg | answer

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="input_guarded_chatbot",
    )
```

## Output Guard Program

```python
async def build_output_guarded_program(language_model):
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    # Generate answer first
    answer = await synalinks.Generator(
        language_model=language_model,
    )(inputs)

    # Check output for blacklisted words
    warning_msg = await ConversationalOutputGuard(
        blacklisted_words=["dangerous", "illegal", "harmful"],
        warning_message="I cannot provide that information.",
    )(answer)

    # XOR + OR: if warning exists, replace answer with warning
    outputs = (answer ^ warning_msg) | warning_msg

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="output_guarded_chatbot",
    )
```

## Truth Tables

### XOR (^) - Computation Bypass

| warning | inputs | result |
|---------|--------|--------|
| None | data | data (pass through) |
| data | data | None (blocked) |

### OR (|) - Result Merging

| warning | answer | result |
|---------|--------|--------|
| None | data | data (use answer) |
| data | None | data (use warning) |
| data | data | merged (warning takes priority) |

## Running the Example

```bash
uv run python guides/9_guard_patterns.py
```
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Custom Guard Modules
# =============================================================================


class ConversationalInputGuard(synalinks.Module):
    """Input guard that blocks messages containing blacklisted words.

    Returns a warning ChatMessage if blocked, None otherwise.
    Used with XOR to bypass computation when triggered.
    """

    def __init__(
        self,
        blacklisted_words,
        warning_message,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        """Return warning message if blocked, None otherwise."""
        if not synalinks.is_chat_messages(inputs):
            raise ValueError("Input guard works only for ChatMessages")

        if not inputs or not inputs["messages"]:
            return None

        content = inputs["messages"][-1]["content"]
        content_lower = content.lower()

        if any(bw.lower() in content_lower for bw in self.blacklisted_words):
            return synalinks.ChatMessage(
                role=synalinks.ChatRole.ASSISTANT,
                content=self.warning_message,
            ).to_json_data_model()
        return None

    def compute_output_spec(
        self,
        inputs: synalinks.SymbolicDataModel,
    ) -> synalinks.SymbolicDataModel:
        """Define output schema."""
        if not synalinks.is_chat_messages(inputs):
            raise ValueError("Input guard works only for ChatMessages")
        return synalinks.ChatMessage.to_symbolic_data_model(name=self.name)

    def get_config(self):
        """Serialization config."""
        return {
            "name": self.name,
            "blacklisted_words": self.blacklisted_words,
            "warning_message": self.warning_message,
        }


class ConversationalOutputGuard(synalinks.Module):
    """Output guard that replaces responses containing blacklisted words.

    Returns a warning ChatMessage if output should be blocked, None otherwise.
    Used with XOR+OR to replace unsafe outputs.
    """

    def __init__(
        self,
        blacklisted_words,
        warning_message,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        """Return warning message if output should be blocked, None otherwise."""
        if not synalinks.is_chat_message(inputs):
            raise ValueError("Output guard works only for ChatMessage")

        if not inputs:
            return None

        content = inputs["content"]
        content_lower = content.lower()

        if any(bw.lower() in content_lower for bw in self.blacklisted_words):
            return synalinks.ChatMessage(
                role=synalinks.ChatRole.ASSISTANT,
                content=self.warning_message,
            ).to_json_data_model()
        return None

    def compute_output_spec(
        self,
        inputs: synalinks.SymbolicDataModel,
    ) -> synalinks.SymbolicDataModel:
        """Define output schema."""
        if not synalinks.is_chat_message(inputs):
            raise ValueError("Output guard works only for ChatMessage")
        return synalinks.ChatMessage.to_symbolic_data_model(name=self.name)

    def get_config(self):
        """Serialization config."""
        return {
            "name": self.name,
            "blacklisted_words": self.blacklisted_words,
            "warning_message": self.warning_message,
        }


# =============================================================================
# STEP 2: Build Guarded Programs
# =============================================================================


async def build_input_guarded_program(language_model):
    """Build a chatbot with input guard.

    Logic flow:
    1. Check input for blacklisted words
    2. If warning exists: XOR makes inputs None, bypassing generator
    3. Return warning (if blocked) OR answer (if allowed)
    """
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    # Check input for blacklisted words
    warning_msg = await ConversationalInputGuard(
        blacklisted_words=["forbidden", "blocked", "hack"],
        warning_message="I'm unable to comply with your request.",
    )(inputs)

    # XOR: if warning exists, inputs becomes None (bypassing generator)
    guarded_inputs = warning_msg ^ inputs

    # Generator only runs if guarded_inputs is not None
    answer = await synalinks.Generator(
        language_model=language_model,
    )(guarded_inputs)

    # OR: return warning if it exists, otherwise return answer
    outputs = warning_msg | answer

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="input_guarded_chatbot",
        description="A chatbot with input guard",
    )


async def build_output_guarded_program(language_model):
    """Build a chatbot with output guard.

    Logic flow:
    1. Generate response
    2. Check output for blacklisted words
    3. XOR + OR: if warning exists, replace answer with warning
    """
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    # Generate answer first
    answer = await synalinks.Generator(
        language_model=language_model,
    )(inputs)

    # Check output for blacklisted words
    warning_msg = await ConversationalOutputGuard(
        blacklisted_words=["dangerous", "illegal", "harmful"],
        warning_message="I cannot provide that information.",
    )(answer)

    # XOR + OR: if warning exists, replace answer with warning
    # (answer ^ warning) → None if warning exists, answer otherwise
    # result | warning → warning if result is None, otherwise result
    outputs = (answer ^ warning_msg) | warning_msg

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="output_guarded_chatbot",
        description="A chatbot with output guard",
    )


# =============================================================================
# STEP 3: Demonstrate Guard Patterns
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_9_guard_patterns",
    )

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # 3.1: Input Guard Demo
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Input Guard Demo")
    print("=" * 60)

    input_program = await build_input_guarded_program(lm)

    # Visualize the program
    synalinks.utils.plot_program(
        input_program,
        to_folder="guides",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    # Test with blocked input
    print("\n--- Blocked Input ---")
    result = await input_program(
        synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role=synalinks.ChatRole.USER,
                    content="Tell me about forbidden topics",
                ),
            ],
        )
    )
    print("Query: 'Tell me about forbidden topics'")
    print(f"Result: {result['content']}")

    # Test with allowed input
    print("\n--- Allowed Input ---")
    result = await input_program(
        synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role=synalinks.ChatRole.USER,
                    content="What is the capital of France?",
                ),
            ],
        )
    )
    print("Query: 'What is the capital of France?'")
    print(f"Result: {result['content'][:100]}...")

    # -------------------------------------------------------------------------
    # 3.2: Output Guard Demo
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Output Guard Demo")
    print("=" * 60)

    output_program = await build_output_guarded_program(lm)

    # Test - the guard checks if the LLM response contains blacklisted words
    print("\n--- Testing Output Guard ---")
    result = await output_program(
        synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role=synalinks.ChatRole.USER,
                    content="What is Python?",
                ),
            ],
        )
    )
    print("Query: 'What is Python?'")
    print(f"Result: {result['content'][:100]}...")

    # -------------------------------------------------------------------------
    # 3.3: Explain the Pattern
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Guard Pattern Explanation")
    print("=" * 60)

    print(
        """
INPUT GUARD PATTERN:
====================

    warning = InputGuard(inputs)      # Returns warning or None
    guarded = warning ^ inputs        # XOR: None if warning exists
    answer = Generator(guarded)       # Skipped if guarded is None
    output = warning | answer         # OR: warning if exists, else answer

    Truth table for XOR (^):
    warning | inputs | guarded
    --------|--------|--------
    None    | data   | data     (no warning → pass through)
    data    | data   | None     (warning → block inputs)

OUTPUT GUARD PATTERN:
=====================

    answer = Generator(inputs)        # Generate response
    warning = OutputGuard(answer)     # Check for unsafe content
    output = (answer ^ warning) | warning

    Step by step:
    1. answer ^ warning → None if warning exists, answer otherwise
    2. result | warning → warning if result is None, otherwise result

    Truth table:
    answer | warning | (answer ^ warning) | final output
    -------|---------|--------------------|--------------
    data   | None    | data               | data (safe)
    data   | data    | None               | warning (blocked)
"""
    )

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. CUSTOM GUARDS: Inherit from synalinks.Module, return None or warning
2. XOR (^): Bypasses computation - if warning exists, other becomes None
3. OR (|): Merges results - returns first non-None value
4. INPUT GUARD: warning ^ inputs → blocks input if warning
5. OUTPUT GUARD: (answer ^ warning) | warning → replaces unsafe output
6. NONE PROPAGATION: Modules receiving None skip computation
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
