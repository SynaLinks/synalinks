
<div align="center">
<img height=200 src="https://github.com/SynaLinks/synalinks/blob/main/img/synalinks_logo_square.png?raw=true">
</div>

<div align="center">


<p align="center">
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>Documentation</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks" target="_blank"><strong>Code Examples</strong></a>
</p>

<b>Synalinks:</b> <em>A production-first LM framework built with decade old Deep Learning best practices</em>

</div>

<div align="center">

![Beta](https://img.shields.io/badge/Release-Beta-blue.svg)
![Coverage Badge](https://raw.githubusercontent.com/SynaLinks/synalinks/refs/heads/main/coverage-badge.svg)
[![Pypi Downloads](https://img.shields.io/pypi/dm/synalinks)](https://pypistats.org/packages/synalinks)
[![Discord](https://img.shields.io/discord/1118241178723291219)](https://discord.gg/82nt97uXcM)
[![Python package](https://github.com/SynaLinks/Synalinks/actions/workflows/tests.yml/badge.svg)](https://github.com/SynaLinks/SynaLinks/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/license/apache-2-0)

</div>

## What is Synalinks?

Synalinks is an open-source framework that makes it easy to create, evaluate, train, and deploy industry-standard Language Models (LMs) applications. Synalinks follows the principle of *progressive disclosure of complexity*: meaning that simple workflows should be quick and easy, while arbitrarily advanced ones should be possible via a clear path that builds upon what you've already learned.

Synalinks is an *adaptation of Keras 3* focused on neuro-symbolic systems and in-context reinforcement learning, an ensemble of techniques that enhance the LMs predictions and accuracy without changing the weights of the model. The goal of Synalinks is to facilitate the rapid setup of simple applications while providing the flexibility for researchers and advanced users to develop sophisticated systems.

## Who is Synalinks for?

Synalinks is designed for a diverse range of users, from professionals and AI researchers to students, independent developers, and hobbyists. It is suitable for anyone who wants to learn about AI by building/composing blocks or build solid foundations for enterprise-grade products. While a background in Machine Learning and Deep Learning can be advantageous — as Synalinks leverages design patterns from Keras, one of the most user-friendly and popular Deep Learning frameworks — it is not a prerequisite. Synalinks is designed to be accessible to anyone with programming skills in Python, making it a versatile and inclusive platform for AI development.

## Why use Synalinks?

Developping a successful LM application in a profesional context, beyond stateless chatbots, is difficult and typically include:

- **Building optimized prompts with examples/instructions at each step**: Synalinks uses advanced In-Context Reinforcement Learning techniques to optimize each prompt.
- **Pipelines that change over time**: Easily edit your pipelines, re-run your training, and you're good to go.
- **Ensuring the correctness of the LMs output**: Synalinks combines constrained structured output with In-Context RL to ensure both format and content correctness.
- **Async Optimization**: Synalinks automatically optimizes your pipelines by detecting parallel processes.
- **Assessing the performance of your application**: Synalinks provides built-in metrics and rewards to evaluate your workflows.
- **Configuring Language & Embedding Models**: Seamlessly integrate multiple LM providers like Ollama, OpenAI, Anthropic, Mistral or Groq.
- **Documenting your ML workflows**: Plot your workflows, training history, and evaluations; document everything.
- **Versioning the prompts/pipelines**: Each program is serializable into JSON so you can version it with git.
- **Deploying REST APIs**: Compatible out-of-the-box with FastAPI so your Data Scientists and Web Developers can stop tearing each other apart.

Synalinks can help you simplify these tasks by leveraging decade old practices in Deep Learning frameworks. We provide a comprehensive suite of tools and features designed to streamline the development process, making it easier to create, evaluate, train, document and deploy robust neuro-symbolic LMs applications.

## Install

```shell
uv pip install synalinks
```

Start your project with

```shell
uv run synalinks init
```

## Programming your application: 4 ways

### Using the `Functional` API

You start from `Input`, you chain modules calls to specify the program's structure, and finally, you create your program from inputs and outputs:

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking process",
    )
    answer: float = synalinks.Field(
        description="The correct numerical answer",
    )

async def main():

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    x0 = synalinks.Input(data_model=Query)
    x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(x0)

    program = synalinks.Program(
        inputs=x0,
        outputs=x1,
        name="chain_of_thought",
        description="Useful to answer in a step by step manner.",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Subclassing the `Program` class

In that case, you should define your modules in `__init__()` and implement the program's structure in `call()`.

**Note:** you can optionaly have a `training` argument (boolean), which you can use to specify a different behavior in training and inference.

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking process",
    )
    answer: float = synalinks.Field(
        description="The correct numerical answer",
    )

class ChainOfThought(synalinks.Program):
    """Useful to answer in a step by step manner.
    
    The first line of the docstring is provided as description
    for the program if not provided in the `super().__init__()`.
    In a similar way the name is automatically infered based on
    the class name if not provided.
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
        self.answer = synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=language_model,
            name=self.name+"_generator",
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        x = await self.answer(inputs, training=training)
        return x

    def get_config(self):
        config = {
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = \
        {
            "language_model": synalinks.saving.serialize_synalinks_object(
                self.language_model
            )
        }
        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=language_model, **config)

async def main():

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    program = ChainOfThought(
        language_model=language_model,
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Mixing the subclassing and the `Functional` API

This way of programming is recommended to encapsulate your application while providing an easy to use setup.
It is the recommended way for most users as it avoid making your program/agents from scratch.
In that case, you should implement only the `__init__()` and `build()` methods.

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking process",
    )
    answer: float = synalinks.Field(
        description="The correct numerical answer",
    )

async def main():

    class ChainOfThought(synalinks.Program):
        """Useful to answer in a step by step manner."""

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
        
        async def build(self, inputs):
            outputs = await synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=self.language_model,
            )(inputs)

            # Create your program using the functional API
            super().__init__(
                inputs=inputs,
                outputs=outputs,
                name=self.name,
                description=self.description,
                trainable=self.trainable,
            )

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    program = ChainOfThought(
        language_model=language_model,
    )

if __name__ == "__main__":
    asyncio.run(main())
```

This allows you to not have to implement the `call()` and serialization methods
(`get_config()` and `from_config()`). The program will be built for any inputs the first time called.

### Using the `Sequential` API

In addition, `Sequential` is a special case of program where the program
is purely a stack of single-input, single-output modules.

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking process",
    )
    answer: float = synalinks.Field(
        description="The correct numerical answer",
    )

async def main():

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    program = synalinks.Sequential(
        [
            synalinks.Input(
                data_model=Query,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
        name="chain_of_thought",
        description="Useful to answer in a step by step manner.",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Getting a summary of your program

To print a tabular summary of your program:

```python
program.summary()
```

Or a plot (Useful to document your system):

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

![chain_of_thought](./docs/assets/chain_of_thought.png)

## Running your program

To run your program use the following:

```python
result = await program(
    Query(query="What is the French city of aerospace?"),
)
```

## Training your program

```python
async def main():

    # ... your program definition

    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot()
    )

    batch_size=32
    epochs=10

    history = await program.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
    )

    synalinks.utils.plot_history(history)

if __name__ == "__main__":
    asyncio.run(main())
```
<div align="center">

![training_history](./docs/assets/training_history.png)

</div>

### Learn more

You can learn more by reading our [documentation](https://synalinks.github.io/synalinks/). If you have questions, the [FAQ](https://synalinks.github.io/synalinks/FAQ/) might help you.

### Contributions

Contributions are welcome, either for the implementation of additional modules, metrics, or optimizers.
For more information, or help for implementing your ideas (or ones from a paper), please join our discord.

Beware that every additional metric/module/optimizer should be approved by the core team, we want to keep the library minimal and clean as possible to avoid an uncontrolled growth leading to bad software practices like in most current leading LM frameworks.

### Community

Join our community to learn more about neuro-symbolic systems and the future of AI. We welcome the participation of people from very different backgrounds or education levels.

### Credit

Synalinks would not be possible without the great work of the following open-source projects:

- [Keras](https://keras.io/) for the graph-based computation backbone, API and overall code, design and philosophy.
- [DSPy](https://dspy.ai/) for the modules/optimizers inspiration.
- [Pydantic](https://docs.pydantic.dev/latest/) for the backend data layer.
- [LiteLLM](https://docs.litellm.ai/docs/) for the LMs integrations.