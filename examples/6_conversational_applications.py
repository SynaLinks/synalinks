"""
# Conversational Applications

Synalinks is designed to handle conversational applications as well as
query-based systems. In the case of a conversational applications, the
input data model is a list of chat messages, and the output an individual
chat message. The `Program` is in that case responsible of handling a
**single conversation turn**.

```mermaid
sequenceDiagram
    participant User
    participant Program
    participant LLM

    User->>Program: ChatMessages [msg1, msg2, ...]
    Program->>LLM: Full conversation context
    LLM-->>Program: New response
    Program-->>User: ChatMessage (assistant)
    Note over User,Program: Add response to history
    User->>Program: ChatMessages [..., new_msg]
```

```python
inputs = synalinks.Input(data_model=synalinks.ChatMessages)
outputs = await synalinks.Generator(
    language_model=language_model,
    streaming=False,
)(inputs)

program = synalinks.Program(
    inputs=inputs,
    outputs=outputs,
    name="simple_chatbot",
)
```

By default, if no data_model/schema is provided to the `Generator` it will
output a `ChatMessage` like output. If the data model is `None`, then you
can enable streaming.

To use the chatbot, pass a `ChatMessages` object with the conversation history:

```python
input_messages = synalinks.ChatMessages(
    messages=[
        synalinks.ChatMessage(
            role="user",
            content="Hello! What is the capital of France?",
        )
    ]
)
response = await program(input_messages)
```

**Note:** Streaming is disabled during training and should only be used in
the **last** `Generator` of your pipeline.

### Key Takeaways

- **Conversational Flow Management**: Synalinks effectively manages
    conversational applications by handling inputs as a list of chat messages
    and generating individual chat messages as outputs.
- **Streaming and Real-Time Interaction**: Synalinks supports streaming for
    real-time interactions. However, streaming is disabled during training
    and should be used only in the final `Generator`.
- **Simple Setup**: Just use `ChatMessages` as input data model and the
    `Generator` will handle the conversation context automatically.

## Program Visualization

![simple_chatbot](../assets/examples/simple_chatbot.png)

## API References

- [ChatMessages (Base DataModels)](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Input](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Input%20module/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="conversational_chatbot",
    )

    # Initialize the language model
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # ==========================================================================
    # Simple Chatbot Example
    # ==========================================================================
    print("Simple Chatbot Example")

    inputs = synalinks.Input(data_model=synalinks.ChatMessages)
    outputs = await synalinks.Generator(
        language_model=language_model,
        streaming=False,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_chatbot",
        description="A simple conversation application",
    )

    # Plot this program to understand it
    synalinks.utils.plot_program(
        program,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    # ==========================================================================
    # Running the chatbot
    # ==========================================================================
    print("Running the chatbot...")

    # Create a conversation with a user message
    input_messages = synalinks.ChatMessages(
        messages=[
            synalinks.ChatMessage(
                role="user",
                content="Hello! What is the capital of France?",
            )
        ]
    )

    # Get the response
    response = await program(input_messages)
    print(response.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
