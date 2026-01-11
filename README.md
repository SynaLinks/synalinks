<div align="center">
<img height=200 src="https://github.com/SynaLinks/synalinks/blob/main/img/synalinks.png?raw=true">
</div>

<div align="center">

<b>From idea to production in just few lines</b>

<em>The first neuro-symbolic Language Model (LM) framework leveraging the simplicity of Keras and the rigor of Deep Learning best practices.</em>

<b>Build RAGs, autonomous agents, multi-agents systems, self-evolving systems and more in just few lines</b>

[Deutsch](https://zdoc.app/de/SynaLinks/synalinks) | 
[English](https://zdoc.app/en/SynaLinks/synalinks) | 
[Español](https://zdoc.app/es/SynaLinks/synalinks) | 
[Français](https://zdoc.app/fr/SynaLinks/synalinks) | 
[日本語](https://zdoc.app/ja/SynaLinks/synalinks) | 
[한국어](https://zdoc.app/ko/SynaLinks/synalinks) | 
[Português](https://zdoc.app/pt/SynaLinks/synalinks) | 
[Русский](https://zdoc.app/ru/SynaLinks/synalinks) | 
[中文](https://zdoc.app/zh/SynaLinks/synalinks)

<p align="center">
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>Documentation</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>Code Examples</strong></a>
</p>

</div>

<div align="center">

⭐ If you find Synalinks useful, please star the repo! Help us reach more AI/ML engineers and grow the community. ⭐

![Beta](https://img.shields.io/badge/Release-Beta-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Coverage Badge](https://raw.githubusercontent.com/SynaLinks/synalinks/refs/heads/main/coverage-badge.svg)
[![Downloads](https://static.pepy.tech/badge/synalinks)](https://pepy.tech/project/synalinks)
[![Discord](https://img.shields.io/discord/1118241178723291219)](https://discord.gg/82nt97uXcM)
[![Python package](https://github.com/SynaLinks/Synalinks/actions/workflows/tests.yml/badge.svg)](https://github.com/SynaLinks/SynaLinks/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/license/apache-2-0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SynaLinks/synalinks)

</div>

<div align="center">

Too busy to read the documentation? Give the [llms.txt](https://synalinks.github.io/synalinks/llms.txt) or [llms-full.txt](https://synalinks.github.io/synalinks/llms-full.txt) to you favorite LMs or AI coding tools. Or better, use [Synalinks Claude Skills](https://github.com/SynaLinks/synalinks-skills) with Claude Code to use Synalinks right away!

</div>

## What Is Synalinks?

Synalinks is an open-source neuro-symbolic framework that makes it simple to create, train, evaluate, and deploy advanced LM-based applications, including RAGs, autonomous agents, and self-evolving reasoning systems.

Think Keras for Language Models applications, a clean, declarative API where:

- 🧩 You **compose** `Module`s like you would with deep learning `Layer`s.
- ⚙️ You **train & optimize** with in-context reinforcement learning.
- 🌐 You **deploy** as REST APIs or MCP servers.

### Key Principles

- **Progressive complexity**: Start simple and grow advanced naturally.
- **Neuro-symbolic learning**: Combine logic, structure, and language models.
- **In-context optimization**: Improve model reasoning without retraining weights.

## Who Is It For?

<div align="center">

| Role                      | Why Synalinks Helps                                      |
| ------------------------- | -------------------------------------------------------- |
| 🧑‍💻 **Developers**      | Build complex LM apps without boilerplate.               |
| 🧠 **Researchers**        | Prototype neuro-symbolic and RL-in-context systems fast. |
| 🏢 **Data Scientists**    | Integrate LM workflows with APIs & databases.            |
| 🎓 **Students/Hobbyists** | Learn AI composition in a clean, intuitive framework.    |

</div>

## Why Synalinks?

Building robust LM apps is hard. Synalinks simplifies it with:

- **Prompt/Anything optimization** per module via In-Context RL
- **Versionable**, JSON-serializable pipelines
- **Constrained structured outputs** (JSON) for correctness
- **Automatic async & parallel execution** by default
- **Metrics, rewards & evaluations** built-in
- **Native integrations**: OpenAI, Ollama, Anthropic, Mistral, Azure, Groq, Gemini, XAI
- **Embeddable fast knowledge base support**: based on DuckDB
- **API-ready**: Deploy with FastAPI or FastMCP
- **KerasTuner compatibility** for hyperparameter search
- **Built-In MLFlow callbacks and hooks** for observability

<div align="center">

| Framework | MCP | Logical Flow | Robust Branching | Parallel Function Calling | Hyperparameter Tuning | Ease of Use |
| --- | --- | --- | --- | --- | --- | --- |
| Synalinks | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | 😀 |
| DSPy      | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No | 😢 |
| AdalFlow  | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No | 😢 |
| TextGrad  | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No  | 😭 |
| Trace     | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No  | 😭 |

</div>

## Installation

```shell
uv pip install synalinks
```

## Example

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class NumericalAnswer(synalinks.DataModel):
    answer: float = synalinks.Field(
        description="The final numerical answer",
    )

language_model = synalinks.LanguageModel(
    model="gemini/gemini-2.5-pro",
)

@synalinks.saving.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
            parentheses, and spaces.
    """
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {
            "result": None,
            "log": "Error: invalid characters in expression",
        }
    try:
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

async def main():
    inputs = synalinks.Input(data_model=Query)

    outputs = await synalinks.FunctionCallingAgent(
        data_model=NumericalAnswer,
        tools=[
            synalinks.Tool(calculate),
        ],
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_agent",
        description="A math agent",
    )

```

## Data Model Operators

Synalinks provides Python operators for combining and manipulating data models, enabling sophisticated control flow:

<div align="center">

| Operator | Name | Description | Use Case |
| :---: | --- | --- | --- |
| `+` | Concatenation | Combines fields from both data models. Raises exception if either is `None`. | Merging outputs from parallel branches |
| `&` | Logical And | Safe concatenation that returns `None` if either input is `None`. | Combining with potentially null branch outputs |
| `\|` | Logical Or | Returns the non-`None` data model. If both are non-`None`, merges them. | Gathering outputs from conditional branches |
| `^` | Logical Xor | Returns data if exactly one input is non-`None`, otherwise `None`. | Exclusive branch selection |
| `~` | Logical Not | Returns `None` if input is non-`None`, or a empty data model if `None`. | Inverting branch conditions |
| `in` | Contains | Checks if a string key exists in the schema properties, or if another data model's schema is contained. Returns `True` or `False`. | Conditional field checking, schema validation |

</div>

```python
# Parallel branches with concatenation
x1 = await generator1(inputs)
x2 = await generator2(inputs)
combined = x1 & x2  # Merge both outputs

# Conditional branches with logical or
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
result = easy | hard  # Get whichever branch was selected
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

## Running your program

To run your program use the following:

```python
result = await program(
    Query(
        query=(
            "A bookstore receives a shipment of 135 new books."
            "They place the books evenly onto 9 shelves."
            "Later, they decide to move 3 books from each shelf to a display table"
            " at the front of the store. "
            "How many books are left on the shelves after the books are moved?"
        )
    ),
)
```

## Training your program/agent

```python
async def main():

    # ... your program definition

    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

    program.compile(
        reward=synalinks.rewards.ExactMatch(
            in_mask=["answer"],
        ),
        optimizer=synalinks.optimizers.OMEGA(
            language_model=language_model,
            embedding_model=embedding_model,
        ),
    )

    batch_size=1
    epochs=10

    history = await program.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Saving & Loading

To save the entire architecture and variables (the program's state) into a JSON file, do:

```python
program.save("my_program.json")
```

In order to load it, do:

```python
loaded_program = synalinks.Program.load("my_program.json")
```

To save only the state your program (the variables) into JSON:

```python
program.save_variables("my_program.variables.json")
```

To load its variables (needs a program with the same architecture), do:

```python
program.load_variables("my_program.variables.json")
```

## Logging

To enable logging, use the following at the beginning of your script:

```python
synalinks.enable_logging()
```

## Observability

Synalinks provides built-in observability through MLflow for tracing and monitoring your programs.

> **Important**: Call `enable_observability()` **before** creating any modules.

```python
import synalinks

# Enable observability first
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # Optional: MLflow server URI
    experiment_name="my_experiment"         # Optional: defaults to "synalinks_traces"
)

# Then create your modules - they will be automatically traced
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

For training metrics and artifacts, use the `Monitor` callback:

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

See the [Observability documentation](https://synalinks.github.io/synalinks/Observability/MLflow/) for Docker setup and advanced configuration.

### Learn more

You can learn more by reading our [documentation](https://synalinks.github.io/synalinks/). If you have questions, the [FAQ](https://synalinks.github.io/synalinks/FAQ/) might help you.

### Contributions

Contributions are welcome, either for the implementation of additional modules, metrics, or optimizers.
For more information, or help for implementing your ideas (or ones from a paper), please join our discord.

Beware that every additional metric/module/optimizer should be approved by the core team, we want to keep the library minimal and clean as possible to avoid an uncontrolled growth leading to bad software practices like in most current leading LM frameworks.

If you have specific feedbacks or features request we invite you to open an [issue](https://github.com/SynaLinks/synalinks/issues).

### Contributors

Your contributions, feedback, and support are what make this project thrive.

From small bug fixes to major features, thank you for believing in open collaboration and the future of neuro-symbolic AI.

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### Community

Join our community to learn more about neuro-symbolic systems and the future of AI. We welcome the participation of people from very different backgrounds or education levels.

### Citing our work

This work have been done under the supervision of François Chollet, the author of Keras. If this work is useful for your research please use the following bibtex entry:

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### Credit

Synalinks would not be possible without the great work of the following open-source projects:

- [Keras](https://keras.io/) for the graph-based computation backbone, API and overall code, design and philosophy.
- [DSPy](https://dspy.ai/) for the modules/optimizers inspiration.
- [Pydantic](https://docs.pydantic.dev/latest/) for the backend data layer.
- [LiteLLM](https://docs.litellm.ai/docs/) for the LMs integrations.
- [DuckDB](https://duckdb.org/) for the fast embeddable knowledge base.
