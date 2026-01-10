"""
# Implementing Custom Modules Via Subclassing

This tutorial is for more advanced users, it will cover how to
create custom modules/programs via subclassing.

In this tutorial, we will cover the following themes:

- The `Module` class
- The `add_variable()` method
- Trainable and non-trainable variables
- The `compute_output_spec()` and `build()` method
- The training argument in `call()`
- Making sure your module/program can be serialized

---

One of the main abstraction of Synalinks is the `Module` class.
A `Module` encapsulate both a state (the module's variables) and
a transformation from inputs to outputs (the `call()` method).

For this tutorial, we are going to make a simple neuro-symbolic component
called `BacktrackingOfThought`. This component is an adaptation of the
famous backtracking algorithm, used a lot in symbolic planning/reasoning,
combined with chain of thought, nowadays most used technique to enhance
the LMs predicitons.

The principle is straitforward, the component will have to "think" then
we will critique at runtime the thinking and aggregate it to
the current chain of thinking only if it is above the given threshold.
This mechanism will allow the system to discard bad thinking to resume
at the previsous step. Additionally we will add a stop condition.

```mermaid
graph TD
    A[Input] --> B[Think]
    B --> C[Critique]
    C --> D{Score >= Threshold?}
    D -->|Yes| E[Add to Chain]
    D -->|No| B
    E --> F{Stop Condition?}
    F -->|No| B
    F -->|Yes| G[Output]
```

This algorithm a simplified version of the popular `TreeOfThought` that
instead of being a tree strucutre, is only a sequential chain of thinking.

## Module Structure

A custom module inherits from `synalinks.Module` and implements key methods:

```python
class MyCustomModule(synalinks.Module):
    def __init__(self, language_model=None, ...):
        super().__init__(name=name, description=description, trainable=trainable)
        # Initialize your generators and components
        self.generator = synalinks.Generator(...)

    async def call(self, inputs, training=False):
        # Define the forward pass logic
        return await self.generator(inputs, training=training)

    async def compute_output_spec(self, inputs, training=False):
        # Define how to compute output specification
        return await self.generator(inputs)

    def get_config(self):
        # Return configuration for serialization
        return {"language_model": synalinks.saving.serialize_synalinks_object(...)}

    @classmethod
    def from_config(cls, config):
        # Reconstruct the module from config
        return cls(...)
```

### The `__init__()` function

When implementing modules that use a `Generator`, you want to externalize
the generator's parameters (`prompt_template`, `instructions`, `examples`,
`use_inputs_schema`, `use_outputs_schema`) to give maximum flexibility to
your module when possible.

### How to know when using a `Variable`?

As a rule of thumb, the variables should be anything that evolve over time
during inference/training. These variables could be updated by the module
itself, or by the optimizer if you have an optimizer designed for that.

### The `call()` function

The `call()` function is the core of the `Module` class. It defines the
computation performed at every call of the module.

### The `compute_output_spec()` function

The `compute_output_spec()` function is responsible for defining the output
data model of the module/program. Its inputs is always a `SymbolicDataModel`,
a placeholder that only contains a JSON schema that serve as data specification.

### Serialization and Deserialization

To ensure that your module can be saved and loaded correctly, you need to
implement serialization and deserialization methods (`get_config()` and
`from_config()`).

### Key Takeaways

- **Module Class**: Encapsulates both state (variables) and transformation
    logic (`call()` method).
- **Initialization and Variables**: Externalize generator parameters for
    flexibility. Use `add_variables` for state management.
- **Call Function**: Defines the core computation of the module.
- **Output Specification**: `compute_output_spec()` defines the output data model.
- **Serialization**: Implement `get_config()` and `from_config()` for saving
    and loading.

## Program Visualization

![backtracking_of_thought](../assets/examples/backtracking_of_thought.png)

## API References

- [Module (Base Class)](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Base%20Module%20class/)
- [ChainOfThought](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Test%20Time%20Compute%20Modules/ChainOfThought%20module/)
- [SelfCritique](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Test%20Time%20Compute%20Modules/SelfCritique%20module/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Program Saving API](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/Program%20Saving%20API/Program%20saving%20and%20loading/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


# Define the data models
class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The correct answer",
    )


# Custom module implementation
class BacktrackingOfThought(synalinks.Module):
    """A Backtracking of Thought algorithm.

    This component combines the backtracking algorithm with chain of thought
    to enhance LMs predictions through iterative self-critique.
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model=None,
        backtracking_threshold=0.5,
        stop_threshold=0.9,
        max_iterations=5,
        return_inputs=False,
        prompt_template=None,
        examples=None,
        instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema
        self.language_model = language_model
        self.backtracking_threshold = backtracking_threshold
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations
        self.return_inputs = return_inputs
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema

        self.thinking = []
        for i in range(self.max_iterations):
            self.thinking.append(
                synalinks.ChainOfThought(
                    schema=self.schema,
                    language_model=self.language_model,
                    prompt_template=self.prompt_template,
                    examples=self.examples,
                    return_inputs=False,
                    instructions=self.instructions,
                    use_inputs_schema=self.use_inputs_schema,
                    use_outputs_schema=self.use_outputs_schema,
                    name=f"thinking_generator_{i}_" + self.name,
                )
            )
        self.critique = []
        for i in range(self.max_iterations):
            self.critique.append(
                synalinks.SelfCritique(
                    language_model=self.language_model,
                    prompt_template=self.prompt_template,
                    examples=self.examples,
                    return_inputs=True,
                    instructions=self.instructions,
                    use_inputs_schema=self.use_inputs_schema,
                    use_outputs_schema=self.use_outputs_schema,
                    name=f"critique_generator_{i}" + self.name,
                )
            )
        # This is going to be the final generator
        self.generator = synalinks.Generator(
            schema=self.schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            return_inputs=self.return_inputs,
            instructions=self.instructions,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            name="generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            # This is to allow logical flows
            # (e.g. don't run the module if no inputs provided)
            return None
        for i in range(self.max_iterations):
            thinking = await self.thinking[i](
                inputs,
                training=training,
            )
            critique = await self.critique[i](
                thinking,
                training=training,
            )
            reward = critique.get("reward")
            if reward > self.backtracking_threshold:
                inputs = await synalinks.ops.concat(
                    inputs,
                    critique,
                    name=f"_inputs_with_thinking_{i}_" + self.name,
                )
                if reward > self.stop_threshold:
                    break
        return await self.generator(
            inputs,
            training=training,
        )

    async def compute_output_spec(self, inputs, training=False):
        for i in range(self.max_iterations):
            inputs = await self.thinking[i](inputs)
            inputs = await self.critique[i](inputs)
        return await self.generator(inputs)

    def get_config(self):
        config = {
            "schema": self.schema,
            "backtracking_threshold": self.backtracking_threshold,
            "stop_threshold": self.stop_threshold,
            "max_iterations": self.max_iterations,
            "return_inputs": self.return_inputs,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": synalinks.saving.serialize_synalinks_object(
                self.language_model,
            )
        }
        return {**language_model_config, **config}

    @classmethod
    def from_config(cls, config):
        language_model = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            language_model=language_model,
            **config,
        )


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="custom_backtracking_module",
    )

    # Initialize the language model
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # ==========================================================================
    # Create a program using the custom module
    # ==========================================================================
    print("Creating a program with BacktrackingOfThought module...")

    inputs = synalinks.Input(data_model=Query)
    outputs = await BacktrackingOfThought(
        language_model=language_model,
        data_model=Answer,
        return_inputs=True,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="backtracking_of_thought",
        description="A Backtracking of Thought algorithm",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    # ==========================================================================
    # Run the program
    # ==========================================================================
    print("Running the program...")

    result = await program(
        Query(
            query=(
                "How can we develop a scalable, fault-tolerant, and secure quantum"
                " computing system that can solve problems intractable for classical"
                " computers, and what are the practical implications for cryptography"
                " and data security?"
            )
        )
    )

    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
