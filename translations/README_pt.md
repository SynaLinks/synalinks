<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/synalinks-dark.svg">
  <img height=200 alt="Synalinks" src="../img/synalinks-light.svg">
</picture>
</div>

<div align="center">

<b>Da ideia à produção em apenas algumas linhas</b>

<em>O primeiro framework neuro-simbólico de Modelos de Linguagem (LM) que combina a simplicidade do Keras com o rigor das melhores práticas de Deep Learning.</em>

<b>Construa [RAGs](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/), [agentes que usam ferramentas](https://synalinks.github.io/synalinks/guides/Agents/), sistemas multiagentes, [agentes recursivos](https://synalinks.github.io/synalinks/guides/Recursive%20Language%20Model%20Agent/) e muito mais em apenas algumas linhas</b>

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
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>Documentação</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>Exemplos de Código</strong></a> .
  <a href="https://github.com/SynaLinks/synalinks/tree/main/guides" target="_blank"><strong>Guias</strong></a>
</p>

</div>

<div align="center">

Se você achar o Synalinks útil, deixe uma estrela no repositório! Ajude-nos a alcançar mais engenheiros de IA/ML e a fazer a comunidade crescer.

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

Quer usar o Synalinks com o seu próprio agente de programação (Claude Code, Cursor, Copilot, etc.)? Adicione as skills específicas do Synalinks a partir do repositório [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills) no GitHub ao seu agente; elas ensinam as convenções do framework e fornecem o contexto necessário para que ele construa programas Synalinks desde o primeiro momento.

</div>

## O que é o Synalinks?

O Synalinks é um framework neuro-simbólico de código aberto que torna simples criar, treinar, avaliar e implantar aplicações avançadas baseadas em LMs, incluindo RAGs, agentes autônomos e sistemas de raciocínio autoevolutivos.

Pense em um Keras para aplicações com Modelos de Linguagem: uma API limpa e declarativa em que:

- Você **compõe** [`Module`s](https://synalinks.github.io/synalinks/guides/Modules/) como faria com `Layer`s de deep learning.
- Você **[treina e otimiza](https://synalinks.github.io/synalinks/guides/Training/)** com aprendizado por reforço in-context.
- Você **implanta** como [APIs REST](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) ou [servidores MCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/).

### Princípios Fundamentais

- **Complexidade progressiva**: [Comece simples e evolua naturalmente para o avançado](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Aprendizado neuro-simbólico**: Combine [lógica, estrutura](https://synalinks.github.io/synalinks/guides/Data%20Models/) e [modelos de linguagem](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Otimização in-context**: [Melhore o raciocínio do modelo sem retreinar os pesos](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/).

## Para quem é?

<div align="center">

| Perfil                    | Por que o Synalinks Ajuda                                   |
| ------------------------- | ----------------------------------------------------------- |
| **Desenvolvedores de IA**      | Construa aplicações LM complexas de nível de produção sem boilerplate. |
| **Pesquisadores de IA**     | Prototipe rapidamente sistemas neuro-simbólicos e de RL in-context.    |
| **Cientistas de Dados**    | Integre fluxos de trabalho com LMs a APIs e bancos de dados.               |
| **Estudantes/Entusiastas** | Aprenda composição de IA em um framework limpo e intuitivo.       |

</div>

## Por que Synalinks?

Existem muitos frameworks hoje em dia; eis o que o Synalinks faz de diferente:

- **Sandbox embutida, sem contêineres**: os agentes executam código e ferramentas não confiáveis em um [runtime seguro e isolado](https://synalinks.github.io/synalinks/guides/Agents/) que **não precisa de Docker nem de serviço externo de sandbox**. Toda a stack é Python puro e embutível, o que a torna ótima para scripts, pesquisa, implantações serverless/na nuvem (S3, Lambda, notebooks, etc.) ou até para criar harnesses de CLI!
- **Suporte a bancos de dados embutidos**: construa [RAGs baseados em grafos e memórias agênticas](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/) com **extração restrita de Grafos de Conhecimento** e **deduplicação semântica automática**, sobre um banco de dados de grafos embutido, sem nenhum servidor de grafos separado para gerenciar. Além disso, uma **base de conhecimento SQL** embutida e rápida está disponível para armazenar dados relacionais e construir RAGs vetoriais/SQL.
- **RL in-context para otimizar seus prompts (e qualquer outra coisa)**: [treine e otimize](https://synalinks.github.io/synalinks/guides/Training/) prompts, exemplos few-shot e [qualquer variável treinável](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/) por módulo **sem tocar nos pesos do modelo**, usando a API familiar `.compile()` / `.fit()` / `.evaluate()` / `.predict()`.
- **Troca de modelo sem esforço**: defina um padrão uma única vez com `synalinks.set_default_language_model(...)` ou passe um identificador em string, e alterne entre Ollama, vLLM, OpenAI, Azure, Anthropic, Mistral, Groq, Gemini, xAI, Cohere, DeepSeek, Together AI, OpenRouter, AWS Bedrock e Doubleword via [LiteLLM](https://docs.litellm.ai/docs/), incluindo [seleção multiobjetivo de modelos](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/) para escolher o melhor modelo em termos de custo/qualidade.
- **Estruture o projeto em um comando, traga seu próprio agente de programação**: inicialize um projeto pronto para produção com `synalinks init` (templates completos para scripts, agentes e treinamento) e, em seguida, adicione as [skills oficiais do Synalinks](https://github.com/SynaLinks/synalinks-skills) para que Claude Code, Cursor, Copilot e companhia escrevam código Synalinks idiomático desde o início.

Além de tudo o que você esperaria de um framework de nível de produção:

- **NOVO**: Agora todos os agentes suportam [Agent Skills](https://agentskills.io/home), `AGENTS.md` e subagentes.
- **[Saídas estruturadas restritas](https://synalinks.github.io/synalinks/guides/Data%20Models/)** (JSON) para garantir correção
- **API de mensagens compatível com Chat Completions**: as mensagens espelham o formato Chat Completions da OpenAI chave por chave + estendidas via litellm com `reasoning_content` e `thinking_blocks`, para que o raciocínio do provedor sobreviva a idas e vindas multi-turno. Também lida com **[entradas multimodais](https://synalinks.github.io/synalinks/guides/Multimodal%20Inputs/)** (imagens e áudio como partes de conteúdo padrão)
- **[Pipelines](https://synalinks.github.io/synalinks/guides/Programs/)** versionáveis e serializáveis em JSON
- **[Execução assíncrona e paralela](https://synalinks.github.io/synalinks/guides/Programs/) automática** por padrão
- **[Métricas](https://synalinks.github.io/synalinks/guides/Metrics/), [recompensas](https://synalinks.github.io/synalinks/guides/Rewards/) e [datasets](https://synalinks.github.io/synalinks/guides/Datasets/)** integrados
- **Pronto para APIs**: Implante com [FastAPI](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) ou [FastMCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)
- **[Compatibilidade com o KerasTuner](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/)** para busca de hiperparâmetros
- **[Callbacks](https://synalinks.github.io/synalinks/guides/Callbacks/) e hooks integrados** para [observabilidade](https://synalinks.github.io/synalinks/guides/Observability/) (incluindo um callback `Monitor` para MLflow)

# Requisitos

- Python 3.12 ou superior
- WSL2 para usuários de Windows

## Início rápido em 3s com `uv` (recomendado)

Se você não conhece o `uv`, instale-o [aqui](https://docs.astral.sh/uv/getting-started/installation/).

Siga as instruções para iniciar um novo projeto synalinks em 3s:

```shell
uvx synalinks init
```

---

Você também pode instalar a biblioteca em um novo projeto com:

```shell
uv add synalinks
```

Para transformar seu agente de programação em um engenheiro de IA, execute isto na raiz do seu projeto:

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

## Exemplo

Os agentes Synalinks agora conseguem ler as convenções do
[`AGENTS.md`](https://agents.md) do seu projeto e usar
[Agent Skills](https://agentskills.io/home). O exemplo abaixo conecta as
[skills oficiais do Synalinks](https://github.com/SynaLinks/synalinks-skills)
a um [`DeepAgent`](https://synalinks.github.io/synalinks/guides/Agents/), um
agente de programação em sandbox, e pede que ele projete os data models de
entrada/saída de uma tarefa, escrevendo-os em uma pasta `workspace/`.

Primeiro, prepare o workspace. Instale a skill oficial do Synalinks com a
CLI [`skills`](https://skills.sh) e adicione um `AGENTS.md`. A skill fica sob
o diretório de trabalho, de modo que o agente em sandbox pode ler seu conteúdo
sob demanda:

```shell
mkdir -p workspace && cd workspace
# Instala a skill `synalinks` em ./.agents/skills/ e escreve skills-lock.json.
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

Isso produz a estrutura abaixo; `.agents/skills` é a *raiz* das skills (uma
subpasta por skill, cada uma contendo um `SKILL.md`):

```text
workspace/
├── AGENTS.md                     # injetado como as convenções do agente
├── skills-lock.json              # fixa a skill a um repositório de origem + hash de conteúdo
└── .agents/
    └── skills/                   # a raiz das skills
        └── synalinks/
            └── SKILL.md          # nome + descrição expostos; corpo lido sob demanda
```

`main.py`:

```python
import synalinks
import asyncio

# Defina o padrão uma única vez; os módulos o utilizam automaticamente.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")


# A resposta final estruturada do agente.
class Deliverable(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="What was created and where",
    )
    files: list[str] = synalinks.Field(
        description="Paths of the files written into the workspace",
    )


async def main():
    # Um DeepAgent conversa em ChatMessages (ele é um agente de programação).
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    agent = synalinks.DeepAgent(
        data_model=Deliverable,
        # A sandbox é inicializada a partir deste diretório (seguro para o host:
        # as escritas do agente vão para a cópia na sandbox, nunca para o seu
        # disco). Seu `AGENTS.md` é injetado para que o agente siga suas convenções.
        workdir="workspace",
        # A raiz das skills (instalada por `skills add`). Listada para o agente como
        # `<available_skills>`; ele lê cada `SKILL.md` sob demanda a partir da
        # sandbox; é por isso que as skills ficam sob `workdir`.
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

## Operadores de Data Models

O Synalinks fornece operadores Python para combinar e manipular data models, permitindo fluxos de controle sofisticados. Consulte o [guia de Fluxo de Controle](https://synalinks.github.io/synalinks/guides/Control%20Flow/) para os padrões de roteamento, fan-out e mesclagem que esses operadores possibilitam:

<div align="center">

| Operador | Nome | Descrição | Caso de Uso |
| :---: | --- | --- | --- |
| `+` | Concatenação | Combina os campos de ambos os data models. Lança uma exceção se um deles for `None`. | Mesclar saídas de branches paralelas |
| `&` | E Lógico | Concatenação segura que retorna `None` se qualquer entrada for `None`. | Combinar com saídas de branches potencialmente nulas |
| `\|` | Ou Lógico | Retorna o data model não-`None`. Se ambos forem não-`None`, mescla-os. | Reunir saídas de branches condicionais |
| `^` | Xor Lógico | Retorna os dados se exatamente uma entrada for não-`None`; caso contrário, `None`. | Seleção exclusiva de branch |
| `~` | Não Lógico | Retorna `None` se a entrada for não-`None`, ou um data model vazio se for `None`. | Inverter condições de branch |
| `in` | Contém | Verifica se uma chave string existe nas propriedades do schema, ou se o schema de outro data model está contido. Retorna `True` ou `False`. | Verificação condicional de campos, validação de schema |

</div>

```python
# Branches paralelas com concatenação
x1 = await generator1(inputs)
x2 = await generator2(inputs)
# combined = x1 *e* x2
combined = (
    x1 & x2
)  # Mescla as duas saídas (adiciona o sufixo _{i} em caso de colisão de chaves)
# [...]
# Branches condicionais com ou lógico
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
# result = easy *ou* hard
result = easy | hard  # Obtém a branch que foi selecionada
```

## Obtendo um resumo do seu programa

Para imprimir um resumo tabular do seu programa:

```python
program.summary()
```

Ou um gráfico (útil para documentar seu sistema):

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

<div align="center">
<img src="../docs/assets/examples/datamodel_designer.png" alt="Programa Data Model Designer" width="600">

<em>O programa de design de data models visualizado com plot_program: Input → DeepAgent. Os módulos treináveis são marcados em verde.</em>
</div>

## Executando seu programa

Para executar seu programa, use o seguinte:

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

## Treinando seu programa/agente

```python
# Definir os modelos de linguagem/embedding padrão permite usar
# o identificador em string (ao estilo Keras) para configurar seu pipeline/treinamento.
# Você ainda pode instanciar as classes se quiser controle refinado.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")
synalinks.set_default_embedding_model("gemini/text-embedding-004")


async def main():

    # ... a definição do seu programa

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

## Salvando e Carregando

Para salvar toda a arquitetura e as variáveis (o estado do programa) em um arquivo JSON, faça:

```python
program.save("my_program.json")
```

Para carregá-lo, faça:

```python
loaded_program = synalinks.Program.load("my_program.json")
```

Para salvar apenas o estado do seu programa (as variáveis) em JSON:

```python
program.save_variables("my_program.variables.json")
```

Para carregar suas variáveis (requer um programa com a mesma arquitetura), faça:

```python
program.load_variables("my_program.variables.json")
```

## Logging

Para habilitar o logging, use o seguinte no início do seu script:

```python
synalinks.enable_logging()
```

## Observabilidade

O Synalinks oferece observabilidade integrada por meio do MLflow para rastrear e monitorar seus programas.

> **Importante**: Chame `enable_observability()` **antes** de criar qualquer módulo.

```python
import synalinks

# Habilite a observabilidade primeiro
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # Opcional: URI do servidor MLflow
    experiment_name="my_experiment",  # Opcional: o padrão é "synalinks_traces"
)

# Depois crie seus módulos - eles serão rastreados automaticamente
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

Para métricas de treinamento e artefatos, use o callback `Monitor`:

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

Consulte o [guia de Observabilidade](https://synalinks.github.io/synalinks/guides/Observability/) para configurações avançadas.

### Saiba mais

Você pode saber mais lendo nossa [documentação](https://synalinks.github.io/synalinks/). Se tiver dúvidas, o [FAQ](https://synalinks.github.io/synalinks/FAQ/) pode ajudar.

### Contribuições

Contribuições são bem-vindas, seja para a implementação de módulos, métricas ou otimizadores adicionais.
Para mais informações, ou ajuda para implementar suas ideias (ou ideias de um artigo), junte-se ao nosso discord.

Tenha em mente que toda métrica/módulo/otimizador adicional deve ser aprovado pela equipe principal; queremos manter a biblioteca o mais minimalista e limpa possível para evitar um crescimento descontrolado que leve a más práticas de software, como acontece na maioria dos frameworks de LM dominantes atualmente.

Se você tiver feedbacks específicos ou pedidos de funcionalidades, convidamos você a abrir uma [issue](https://github.com/SynaLinks/synalinks/issues).

### Contribuidores

Suas contribuições, seu feedback e seu apoio são o que faz este projeto prosperar.

De pequenas correções de bugs a grandes funcionalidades, obrigado por acreditar na colaboração aberta e no futuro da IA neuro-simbólica.

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### Comunidade

Junte-se à nossa comunidade para saber mais sobre sistemas neuro-simbólicos e o futuro da IA. Damos as boas-vindas à participação de pessoas com formações e níveis de educação muito diversos.

### Citando nosso trabalho

Este trabalho foi realizado sob a supervisão de François Chollet, o autor do Keras. Se este trabalho for útil para a sua pesquisa, use a seguinte entrada bibtex:

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### Créditos

O Synalinks não seria possível sem o excelente trabalho dos seguintes projetos de código aberto:

- [Keras](https://keras.io/) pela espinha dorsal de computação baseada em grafos, pela API e pelo código, design e filosofia em geral.
- [DSPy](https://dspy.ai/) pela inspiração dos módulos/otimizadores.
- [Pydantic](https://docs.pydantic.dev/latest/) pela camada de dados do backend.
- [LiteLLM](https://docs.litellm.ai/docs/) pelas integrações com LMs.
- [DuckDB](https://duckdb.org/), [Ladybug](https://ladybugdb.com/), [LanceDB](https://www.lancedb.com/) por seus incríveis bancos de dados embutidos.
- [MirageAI](https://www.strukto.ai/mirage) por sua incrível sandbox!
