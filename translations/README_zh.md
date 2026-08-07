<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/synalinks-dark.svg">
  <img height=200 alt="Synalinks" src="../img/synalinks-light.svg">
</picture>
</div>

<div align="center">

<b>从想法到生产，只需寥寥数行代码</b>

<em>首个神经符号语言模型（LM）框架，兼具 Keras 的简洁与深度学习最佳实践的严谨。</em>

<b>只需几行代码即可构建 [RAG](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/)、[工具调用智能体](https://synalinks.github.io/synalinks/guides/Agents/)、多智能体系统、[递归智能体](https://synalinks.github.io/synalinks/guides/Recursive%20Language%20Model%20Agent/)等应用</b>

[Deutsch](README_de.md) | 
[English](../README.md) | 
[Español](README_es.md) | 
[Français](README_fr.md) | 
[Italiano](README_it.md) | 
[日本語](README_ja.md) | 
[한국어](README_ko.md) | 
[Português](README_pt.md) | 
[Русский](README_ru.md) | 
[中文](README_zh.md)

<p align="center">
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>文档</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>常见问题</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>代码示例</strong></a> .
  <a href="https://github.com/SynaLinks/synalinks/tree/main/guides" target="_blank"><strong>指南</strong></a>
</p>

</div>

<div align="center">

如果你觉得 Synalinks 有用，请给仓库点个 Star！帮助我们触达更多 AI/ML 工程师，共同壮大社区。

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

想让你自己的编码智能体（Claude Code、Cursor、Copilot 等）使用 Synalinks？把 GitHub 上 [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills) 中的 Synalinks 专用技能添加到你的智能体中：这些技能会教它框架的约定，并为它提供所需的上下文，让它可以立即开始构建 Synalinks 程序。

</div>

## Synalinks 是什么？

Synalinks 是一个开源的神经符号框架，可以轻松地创建、训练、评估和部署先进的基于 LM 的应用，包括 RAG、自主智能体和自我进化的推理系统。

可以把它理解为面向语言模型应用的 Keras，一套简洁的声明式 API，在这里：

- 你可以像组合深度学习 `Layer` 一样**组合** [`Module`](https://synalinks.github.io/synalinks/guides/Modules/)。
- 你可以通过上下文内强化学习进行**[训练与优化](https://synalinks.github.io/synalinks/guides/Training/)**。
- 你可以将其**部署**为 [REST API](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) 或 [MCP 服务器](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)。

### 核心原则

- **渐进式复杂度**：[从简单起步，自然地走向进阶](https://synalinks.github.io/synalinks/guides/Getting%20Started/)。
- **神经符号学习**：将[逻辑、结构](https://synalinks.github.io/synalinks/guides/Data%20Models/)与[语言模型](https://synalinks.github.io/synalinks/guides/Getting%20Started/)相结合。
- **上下文内优化**：[无需重新训练权重即可提升模型推理能力](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/)。

## 适合哪些人？

<div align="center">

| 角色                      | Synalinks 能带来什么                                         |
| ------------------------- | ----------------------------------------------------------- |
| **AI 开发者**      | 无需样板代码即可构建复杂的生产级 LM 应用。 |
| **AI 研究者**     | 快速原型化神经符号与上下文内 RL 系统。    |
| **数据科学家**    | 将 LM 工作流与 API 和数据库集成。               |
| **学生/爱好者** | 在一个简洁、直观的框架中学习 AI 组合式开发。       |

</div>

## 为什么选择 Synalinks？

如今已有许多框架；以下是 Synalinks 的与众不同之处：

- **内嵌、免容器的沙箱**：智能体在一个[安全、隔离的运行时](https://synalinks.github.io/synalinks/guides/Agents/)中运行不可信代码和工具，**无需 Docker 或外部沙箱服务**。整个技术栈是纯 Python 且可嵌入的，非常适合脚本编写、研究、无服务器/云端部署（S3、Lambda、notebook 等），甚至可以用来打造 CLI 工具！
- **内嵌数据库支持**：基于内嵌图数据库构建[基于图的 RAG 和智能体记忆](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/)，支持**受约束的知识图谱抽取**和**自动语义去重**，无需运行独立的图数据库服务器。此外，还提供快速的内嵌 **SQL 知识库**，用于存储关系型数据并构建向量/SQL RAG。
- **用上下文内 RL 优化你的提示词（以及其他一切）**：使用熟悉的 `.compile()` / `.fit()` / `.evaluate()` / `.predict()` API，为每个模块[训练和优化](https://synalinks.github.io/synalinks/guides/Training/)提示词、少样本示例以及[任何可训练变量](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/)，**完全不触碰模型权重**。
- **轻松切换模型**：通过 `synalinks.set_default_language_model(...)` 一次性设置默认值，或传入字符串标识符，即可借助 [LiteLLM](https://docs.litellm.ai/docs/) 在 Ollama、vLLM、OpenAI、Azure、Anthropic、Mistral、Groq、Gemini、xAI、Cohere、DeepSeek、Together AI、OpenRouter、AWS Bedrock 和 Doubleword 之间自由切换，还包括[多目标模型选择](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/)，帮你在成本/质量之间挑选最佳模型。
- **一条命令搭建脚手架，自带编码智能体也没问题**：用 `synalinks init` 引导创建一个生产就绪的项目（内置脚本、智能体和训练模板），再装上官方的 [Synalinks 技能](https://github.com/SynaLinks/synalinks-skills)，让 Claude Code、Cursor、Copilot 等从一开始就写出地道的 Synalinks 代码。

此外，还有你对一个生产级框架所期待的一切：

- **新功能**：现在所有智能体都支持 [Agent Skills](https://agentskills.io/home)、`AGENTS.md` 和子智能体。
- **[受约束的结构化输出](https://synalinks.github.io/synalinks/guides/Data%20Models/)**（JSON），确保正确性
- **兼容 Chat Completions 的消息 API**：消息与 OpenAI Chat Completions 格式逐键对应，并借助 litellm 扩展了 `reasoning_content` 和 `thinking_blocks`，使模型提供方的推理内容能在多轮往返中得以保留。同时支持**[多模态输入](https://synalinks.github.io/synalinks/guides/Multimodal%20Inputs/)**（图像和音频作为标准内容部分）
- **可版本化**、可 JSON 序列化的[流水线](https://synalinks.github.io/synalinks/guides/Programs/)
- 默认**自动[异步与并行执行](https://synalinks.github.io/synalinks/guides/Programs/)**
- 内置**[指标](https://synalinks.github.io/synalinks/guides/Metrics/)、[奖励](https://synalinks.github.io/synalinks/guides/Rewards/)与[数据集](https://synalinks.github.io/synalinks/guides/Datasets/)**
- **API 就绪**：使用 [FastAPI](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) 或 [FastMCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/) 部署
- **[兼容 KerasTuner](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/)**，支持超参数搜索
- **内置[回调](https://synalinks.github.io/synalinks/guides/Callbacks/)与钩子**，支持[可观测性](https://synalinks.github.io/synalinks/guides/Observability/)（包括 MLflow 的 `Monitor` 回调）

# 环境要求

- Python 3.12 或更高版本
- Windows 用户需使用 WSL2

## 使用 `uv` 3 秒快速上手（推荐）

如果你还不了解 `uv`，请在[这里](https://docs.astral.sh/uv/getting-started/installation/)安装。

按照指引，3 秒内启动一个新的 synalinks 项目：

```shell
uvx synalinks init
```

---

你也可以在新项目中这样安装该库：

```shell
uv add synalinks
```

要把你的编码智能体变成一名 AI 工程师，请在项目根目录执行：

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

## 示例

Synalinks 智能体现在可以读取你项目的 [`AGENTS.md`](https://agents.md)
约定并使用 [Agent Skills](https://agentskills.io/home)。下面的示例将官方
[Synalinks 技能](https://github.com/SynaLinks/synalinks-skills)接入一个
[`DeepAgent`](https://synalinks.github.io/synalinks/guides/Agents/)（一个
沙箱化的编码智能体），并要求它为某个任务设计输入/输出数据模型，
将结果写入 `workspace/` 文件夹。

首先搭建工作区。使用 [`skills`](https://skills.sh) CLI 安装官方 Synalinks
技能，并添加一个 `AGENTS.md`。技能会安装到工作目录下，
因此沙箱化的智能体可以按需读取其正文：

```shell
mkdir -p workspace && cd workspace
# 将 `synalinks` 技能安装到 ./.agents/skills/ 并写入 skills-lock.json。
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

这会得到如下目录结构，其中 `.agents/skills` 是技能的*根目录*（每个技能一个
子文件夹，各自包含一个 `SKILL.md`）：

```text
workspace/
├── AGENTS.md                     # 作为智能体的约定被注入
├── skills-lock.json              # 将技能固定到源仓库 + 内容哈希
└── .agents/
    └── skills/                   # 技能根目录
        └── synalinks/
            └── SKILL.md          # 名称 + 描述会被展示；正文按需读取
```

`main.py`：

```python
import synalinks
import asyncio

# 一次性设置默认值，各模块会自动使用它。
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")


# 智能体的结构化最终答案。
class Deliverable(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="What was created and where",
    )
    files: list[str] = synalinks.Field(
        description="Paths of the files written into the workspace",
    )


async def main():
    # DeepAgent 通过 ChatMessages 进行对话（它是一个编码智能体）。
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    agent = synalinks.DeepAgent(
        data_model=Deliverable,
        # 沙箱会以该目录为种子进行初始化（对宿主机安全：智能体的
        # 写入只落在沙箱副本中，绝不会写到你的磁盘上）。其中的 `AGENTS.md`
        # 会被注入，使智能体遵循你的约定。
        workdir="workspace",
        # 技能根目录（由 `skills add` 安装）。会以
        # `<available_skills>` 的形式列给智能体；它会从沙箱中按需读取
        # 每个 `SKILL.md`，这正是技能要放在 `workdir` 下的原因。
        skills=["workspace/.agents/skills"],
    )
    outputs = await agent(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="datamodel_designer",
        description="Designs Synalinks data models for a given task",
    )

    task = (
        "Define the input and output Synalinks DataModels for a support-ticket "
        "triage task: the input is a raw customer message; the output is the "
        "predicted category, a priority, and a short suggested reply. Write them "
        "to `models.py` using idiomatic Synalinks; consult the skills first."
    )
    result = await program(
        synalinks.ChatMessages(
            messages=[synalinks.ChatMessage(role="user", content=task)],
        )
    )
    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
```

## 数据模型运算符

Synalinks 提供了用于组合和操作数据模型的 Python 运算符，从而实现复杂精细的控制流。这些运算符所支持的路由、扇出与合并模式，请参见[控制流指南](https://synalinks.github.io/synalinks/guides/Control%20Flow/)：

<div align="center">

| 运算符 | 名称 | 描述 | 使用场景 |
| :---: | --- | --- | --- |
| `+` | 拼接 | 合并两个数据模型的字段。若任一方为 `None` 则抛出异常。 | 合并并行分支的输出 |
| `&` | 逻辑与 | 安全的拼接，若任一输入为 `None` 则返回 `None`。 | 与可能为空的分支输出进行组合 |
| `\|` | 逻辑或 | 返回非 `None` 的数据模型。若两者均非 `None`，则将其合并。 | 收集条件分支的输出 |
| `^` | 逻辑异或 | 当且仅当恰好一个输入非 `None` 时返回数据，否则返回 `None`。 | 互斥的分支选择 |
| `~` | 逻辑非 | 若输入非 `None` 则返回 `None`；若为 `None` 则返回空数据模型。 | 反转分支条件 |
| `in` | 包含 | 检查某个字符串键是否存在于 schema 属性中，或另一个数据模型的 schema 是否被包含。返回 `True` 或 `False`。 | 条件式字段检查、schema 校验 |

</div>

```python
# 使用拼接的并行分支
x1 = await generator1(inputs)
x2 = await generator2(inputs)
# combined = x1 *and* x2
combined = x1 & x2  # 合并两个输出（若键冲突则添加 _{i} 后缀）
# [...]
# 使用逻辑或的条件分支
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
# result = easy *or* hard
result = easy | hard  # 获取被选中的那个分支
```

## 获取程序的摘要

打印程序的表格化摘要：

```python
program.summary()
```

或者生成一张图（便于为你的系统撰写文档）：

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

<div align="center">
<img src="../docs/assets/examples/datamodel_designer.png" alt="数据模型设计器程序" width="600">

<em>使用 plot_program 可视化的数据模型设计器程序：Input → DeepAgent。可训练模块以绿色标出。</em>
</div>

## 运行你的程序

使用以下方式运行你的程序：

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

## 训练你的程序/智能体

```python
# 设置默认的语言/嵌入模型后，你就可以使用字符串标识符
# （Keras 风格）来配置你的流水线/训练。
# 如果需要细粒度控制，你仍然可以直接实例化这些类。
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")
synalinks.set_default_embedding_model("gemini/text-embedding-004")


async def main():

    # ... 你的程序定义

    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer="omega",
    )

    batch_size = 1
    epochs = 10

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

## 保存与加载

将整个架构和变量（程序的状态）保存为一个 JSON 文件：

```python
program.save("my_program.json")
```

加载它：

```python
loaded_program = synalinks.Program.load("my_program.json")
```

仅将程序的状态（变量）保存为 JSON：

```python
program.save_variables("my_program.variables.json")
```

加载其变量（需要一个具有相同架构的程序）：

```python
program.load_variables("my_program.variables.json")
```

## 日志

要启用日志，请在脚本开头使用：

```python
synalinks.enable_logging()
```

## 可观测性

Synalinks 通过 MLflow 提供内置的可观测性，用于追踪和监控你的程序。

> **重要**：请在创建任何模块**之前**调用 `enable_observability()`。

```python
import synalinks

# 先启用可观测性
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # 可选：MLflow 服务器 URI
    experiment_name="my_experiment",  # 可选：默认为 "synalinks_traces"
)

# 然后再创建你的模块，它们会被自动追踪
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

如需记录训练指标和产物，请使用 `Monitor` 回调：

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

高级配置请参见[可观测性指南](https://synalinks.github.io/synalinks/guides/Observability/)。

### 了解更多

你可以通过阅读我们的[文档](https://synalinks.github.io/synalinks/)了解更多。如果有疑问，[常见问题](https://synalinks.github.io/synalinks/FAQ/)或许能帮到你。

### 参与贡献

欢迎贡献，无论是实现额外的模块、指标还是优化器。
如需更多信息，或希望获得实现你的想法（或论文中的想法）的帮助，请加入我们的 Discord。

请注意，每个新增的指标/模块/优化器都需经过核心团队的批准。我们希望让这个库尽可能保持精简和整洁，避免像当前大多数主流 LM 框架那样，因失控的膨胀而导致糟糕的软件实践。

如果你有具体的反馈或功能需求，欢迎提交 [issue](https://github.com/SynaLinks/synalinks/issues)。

### 贡献者

正是你们的贡献、反馈和支持，让这个项目蓬勃发展。

从小的 bug 修复到重大功能，感谢你们对开放协作以及神经符号 AI 未来的信念。

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### 社区

加入我们的社区，了解更多关于神经符号系统和 AI 未来的内容。我们欢迎来自不同背景、不同教育水平的人们参与。

### 引用我们的工作

本工作是在 Keras 作者 François Chollet 的指导下完成的。如果这项工作对你的研究有帮助，请使用以下 bibtex 条目：

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### 致谢

没有以下开源项目的杰出工作，Synalinks 就不可能存在：

- [Keras](https://keras.io/)：基于图的计算骨架、API 以及整体的代码、设计与理念。
- [DSPy](https://dspy.ai/)：模块/优化器方面的灵感来源。
- [Pydantic](https://docs.pydantic.dev/latest/)：后端数据层。
- [LiteLLM](https://docs.litellm.ai/docs/)：语言模型集成。
- [DuckDB](https://duckdb.org/)、[Ladybug](https://ladybugdb.com/)、[LanceDB](https://www.lancedb.com/)：出色的内嵌数据库。
- [MirageAI](https://www.strukto.ai/mirage)：出色的沙箱！
