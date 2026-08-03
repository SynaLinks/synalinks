<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/synalinks-dark.svg">
  <img height=200 alt="Synalinks" src="../img/synalinks-light.svg">
</picture>
</div>

<div align="center">

<b>아이디어에서 프로덕션까지, 단 몇 줄이면 충분합니다</b>

<em>Keras의 단순함과 딥러닝 모범 사례의 엄밀함을 결합한 최초의 뉴로심볼릭 언어 모델(LM) 프레임워크.</em>

<b>[RAG](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/), [툴 사용 에이전트](https://synalinks.github.io/synalinks/guides/Agents/), 멀티 에이전트 시스템, [재귀 에이전트](https://synalinks.github.io/synalinks/guides/Recursive%20Language%20Model%20Agent/) 등을 단 몇 줄로 구축하세요</b>

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
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>문서</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>코드 예제</strong></a> .
  <a href="https://github.com/SynaLinks/synalinks/tree/main/guides" target="_blank"><strong>가이드</strong></a>
</p>

</div>

<div align="center">

Synalinks가 유용하다고 생각하시면 리포지토리에 스타를 눌러주세요! 더 많은 AI/ML 엔지니어에게 다가가고 커뮤니티를 성장시키는 데 큰 힘이 됩니다.

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

Synalinks를 여러분의 코딩 에이전트(Claude Code, Cursor, Copilot 등)와 함께 사용하고 싶으신가요? GitHub의 [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills)에 있는 Synalinks 전용 스킬을 에이전트에 추가하세요. 프레임워크의 컨벤션을 학습시키고, 곧바로 Synalinks 프로그램을 작성하는 데 필요한 컨텍스트를 제공합니다.

</div>

## Synalinks란 무엇인가요?

Synalinks는 RAG, 자율 에이전트, 자기 진화형 추론 시스템을 포함한 고급 LM 기반 애플리케이션을 손쉽게 만들고, 학습시키고, 평가하고, 배포할 수 있게 해주는 오픈소스 뉴로심볼릭 프레임워크입니다.

언어 모델 애플리케이션을 위한 Keras라고 생각하시면 됩니다. 깔끔하고 선언적인 API로:

- 딥러닝의 `Layer`처럼 [`Module`](https://synalinks.github.io/synalinks/guides/Modules/)을 **조합**합니다.
- 인컨텍스트 강화학습으로 **[학습하고 최적화](https://synalinks.github.io/synalinks/guides/Training/)**합니다.
- [REST API](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) 또는 [MCP 서버](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)로 **배포**합니다.

### 핵심 원칙

- **점진적 복잡성**: [단순하게 시작해서 자연스럽게 고급으로 확장합니다](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **뉴로심볼릭 학습**: [로직과 구조](https://synalinks.github.io/synalinks/guides/Data%20Models/), 그리고 [언어 모델](https://synalinks.github.io/synalinks/guides/Getting%20Started/)을 결합합니다.
- **인컨텍스트 최적화**: [가중치 재학습 없이 모델의 추론 능력을 개선합니다](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/).

## 누구를 위한 것인가요?

<div align="center">

| 역할                      | Synalinks가 도움이 되는 이유                                 |
| ------------------------- | ----------------------------------------------------------- |
| **AI 개발자**      | 보일러플레이트 없이 복잡한 프로덕션급 LM 앱을 구축합니다. |
| **AI 연구자**     | 뉴로심볼릭 및 인컨텍스트 RL 시스템을 빠르게 프로토타이핑합니다.    |
| **데이터 과학자**    | LM 워크플로를 API 및 데이터베이스와 통합합니다.               |
| **학생/취미 개발자** | 깔끔하고 직관적인 프레임워크로 AI 조합 방식을 배웁니다.       |

</div>

## 왜 Synalinks인가요?

오늘날 많은 프레임워크가 존재합니다. Synalinks가 다르게 하는 점은 다음과 같습니다:

- **컨테이너가 필요 없는 내장 샌드박스** : 에이전트가 신뢰할 수 없는 코드와 툴을 [안전하고 격리된 런타임](https://synalinks.github.io/synalinks/guides/Agents/)에서 실행하며, **Docker나 외부 샌드박스 서비스가 전혀 필요 없습니다**. 전체 스택이 순수 Python이고 임베딩 가능하므로 스크립팅, 연구, 서버리스/클라우드 배포(S3, Lambda, 노트북 등)는 물론 CLI 하네스 제작에도 안성맞춤입니다!
- **내장 데이터베이스 지원** : 내장 그래프 데이터베이스 위에서 **제약 기반 지식 그래프 추출**과 **자동 시맨틱 중복 제거**를 활용해 [그래프 기반 RAG와 에이전트 메모리](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/)를 구축하세요. 별도의 그래프 서버를 운영할 필요가 없습니다. 또한 관계형 데이터를 저장하고 벡터/SQL RAG를 구축할 수 있는 빠른 내장 **SQL 지식 베이스**도 제공됩니다.
- **프롬프트(그리고 그 밖의 무엇이든)를 최적화하는 인컨텍스트 RL** : 익숙한 `.compile()` / `.fit()` / `.evaluate()` / `.predict()` API를 사용해 **모델 가중치를 건드리지 않고** 프롬프트, few-shot 예제, 그리고 모듈별 [학습 가능한 변수](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/)를 [학습하고 최적화](https://synalinks.github.io/synalinks/guides/Training/)합니다.
- **손쉬운 모델 전환** : `synalinks.set_default_language_model(...)`로 기본값을 한 번만 설정하거나 문자열 식별자를 전달하면, [LiteLLM](https://docs.litellm.ai/docs/)을 통해 Ollama, vLLM, OpenAI, Azure, Anthropic, Mistral, Groq, Gemini, xAI, Cohere, DeepSeek, Together AI, OpenRouter, AWS Bedrock, Doubleword 사이를 자유롭게 전환할 수 있습니다. 비용/품질 기준으로 최적의 모델을 고르는 [다목적 모델 선택](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/)도 포함됩니다.
- **명령 한 번으로 스캐폴딩, 코딩 에이전트는 원하는 것으로** : `synalinks init`으로 프로덕션 준비가 된 프로젝트를 부트스트랩하고(스크립트, 에이전트, 학습을 위한 템플릿이 기본 제공), 공식 [Synalinks 스킬](https://github.com/SynaLinks/synalinks-skills)을 추가하면 Claude Code, Cursor, Copilot 등이 처음부터 관용적인 Synalinks 코드를 작성합니다.

여기에 프로덕션급 프레임워크에서 기대할 수 있는 모든 것이 더해집니다:

- **신규**: 이제 모든 에이전트가 [Agent Skills](https://agentskills.io/home), `AGENTS.md`, 서브 에이전트를 지원합니다.
- 정확성을 위한 **[제약 기반 구조화 출력](https://synalinks.github.io/synalinks/guides/Data%20Models/)**(JSON)
- **Chat Completions 호환 메시지 API**: 메시지가 OpenAI Chat Completions 포맷을 키 단위로 그대로 따르며, litellm 확장인 `reasoning_content`와 `thinking_blocks`를 더해 프로바이더의 추론 내용이 멀티턴 왕복에서도 유지됩니다. 또한 **[멀티모달 입력](https://synalinks.github.io/synalinks/guides/Multimodal%20Inputs/)**(표준 콘텐츠 파트로서의 이미지 및 오디오)도 처리합니다
- **버전 관리 가능**하고 JSON으로 직렬화되는 [파이프라인](https://synalinks.github.io/synalinks/guides/Programs/)
- 기본으로 제공되는 **자동 [비동기 및 병렬 실행](https://synalinks.github.io/synalinks/guides/Programs/)**
- **[메트릭](https://synalinks.github.io/synalinks/guides/Metrics/), [리워드](https://synalinks.github.io/synalinks/guides/Rewards/), [데이터셋](https://synalinks.github.io/synalinks/guides/Datasets/)** 내장
- **API 배포 준비 완료**: [FastAPI](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) 또는 [FastMCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)로 배포
- 하이퍼파라미터 탐색을 위한 **[KerasTuner 호환성](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/)**
- [관측성](https://synalinks.github.io/synalinks/guides/Observability/)을 위한 **내장 [콜백](https://synalinks.github.io/synalinks/guides/Callbacks/)과 훅**(MLflow `Monitor` 콜백 포함)

# 요구 사항

- Python 3.12 이상
- Windows 사용자의 경우 WSL2

## `uv`로 3초 만에 시작하기 (권장)

`uv`를 모르신다면 [여기](https://docs.astral.sh/uv/getting-started/installation/)에서 설치하세요.

안내에 따라 3초 만에 새 synalinks 프로젝트를 시작해 보세요:

```shell
uvx synalinks init
```

---

다음 명령으로 새 프로젝트에 라이브러리를 설치할 수도 있습니다:

```shell
uv add synalinks
```

여러분의 코딩 에이전트를 AI 엔지니어로 변신시키려면, 프로젝트 루트에서 다음을 실행하세요:

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

## 예제

Synalinks 에이전트는 이제 프로젝트의 [`AGENTS.md`](https://agents.md)
컨벤션을 읽고 [Agent Skills](https://agentskills.io/home)를 사용할 수 있습니다. 아래 예제는
공식 [Synalinks 스킬](https://github.com/SynaLinks/synalinks-skills)을
샌드박스 코딩 에이전트인 [`DeepAgent`](https://synalinks.github.io/synalinks/guides/Agents/)에
연결하고, 특정 작업에 대한 입력/출력 데이터 모델을 설계하여
`workspace/` 폴더에 작성하도록 요청합니다.

먼저 워크스페이스를 설정합니다. [`skills`](https://skills.sh) CLI로 공식
Synalinks 스킬을 설치하고 `AGENTS.md`를 추가하세요. 스킬은 작업 디렉터리 아래에
설치되므로, 샌드박스 에이전트가 필요할 때 스킬 본문을 읽을 수 있습니다:

```shell
mkdir -p workspace && cd workspace
# `synalinks` 스킬을 ./.agents/skills/ 에 설치하고 skills-lock.json을 작성합니다.
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

이렇게 하면 아래와 같은 레이아웃이 만들어집니다. `.agents/skills`가 스킬 *루트*입니다
(스킬마다 하위 폴더가 하나씩 있고, 각각 `SKILL.md`를 담고 있습니다):

```text
workspace/
├── AGENTS.md                     # 에이전트의 컨벤션으로 주입됨
├── skills-lock.json              # 스킬을 소스 리포지토리 + 콘텐츠 해시에 고정
└── .agents/
    └── skills/                   # 스킬 루트
        └── synalinks/
            └── SKILL.md          # 이름 + 설명이 노출됨; 본문은 필요할 때 읽음
```

`main.py`:

```python
import synalinks
import asyncio

# 기본값을 한 번만 설정하세요. 모듈이 자동으로 이를 사용합니다.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")


# 에이전트의 구조화된 최종 답변.
class Deliverable(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="What was created and where",
    )
    files: list[str] = synalinks.Field(
        description="Paths of the files written into the workspace",
    )


async def main():
    # DeepAgent는 ChatMessages로 대화합니다 (코딩 에이전트이기 때문입니다).
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    agent = synalinks.DeepAgent(
        data_model=Deliverable,
        # 샌드박스는 이 디렉터리를 시드로 초기화됩니다 (호스트 안전: 에이전트의
        # 쓰기 작업은 샌드박스 사본에만 반영되고 여러분의 디스크에는 절대 닿지
        # 않습니다). 이 디렉터리의 `AGENTS.md`가 주입되어 에이전트가 여러분의
        # 컨벤션을 따르게 됩니다.
        workdir="workspace",
        # 스킬 루트 (`skills add`로 설치됨). 에이전트에게
        # `<available_skills>`로 나열되며, 각 `SKILL.md`는 필요할 때 샌드박스에서
        # 읽습니다. 스킬이 `workdir` 아래에 있어야 하는 이유입니다.
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

## 데이터 모델 연산자

Synalinks는 데이터 모델을 결합하고 조작하기 위한 Python 연산자를 제공하여 정교한 제어 흐름을 가능하게 합니다. 이 연산자들로 구현할 수 있는 라우팅, 팬아웃, 병합 패턴은 [제어 흐름 가이드](https://synalinks.github.io/synalinks/guides/Control%20Flow/)를 참고하세요:

<div align="center">

| 연산자 | 이름 | 설명 | 사용 사례 |
| :---: | --- | --- | --- |
| `+` | 연결(Concatenation) | 두 데이터 모델의 필드를 결합합니다. 둘 중 하나라도 `None`이면 예외를 발생시킵니다. | 병렬 브랜치의 출력 병합 |
| `&` | 논리 And | 둘 중 하나라도 `None`이면 `None`을 반환하는 안전한 연결입니다. | `None`일 수 있는 브랜치 출력과의 결합 |
| `\|` | 논리 Or | `None`이 아닌 데이터 모델을 반환합니다. 둘 다 `None`이 아니면 병합합니다. | 조건부 브랜치의 출력 수집 |
| `^` | 논리 Xor | 정확히 하나의 입력만 `None`이 아니면 그 데이터를 반환하고, 그렇지 않으면 `None`을 반환합니다. | 배타적 브랜치 선택 |
| `~` | 논리 Not | 입력이 `None`이 아니면 `None`을 반환하고, `None`이면 빈 데이터 모델을 반환합니다. | 브랜치 조건 반전 |
| `in` | 포함(Contains) | 문자열 키가 스키마 속성에 존재하는지, 또는 다른 데이터 모델의 스키마가 포함되는지 확인합니다. `True` 또는 `False`를 반환합니다. | 조건부 필드 확인, 스키마 검증 |

</div>

```python
# 연결을 활용한 병렬 브랜치
x1 = await generator1(inputs)
x2 = await generator2(inputs)
# combined = x1 *and* x2
combined = x1 & x2  # 두 출력을 병합 (키 충돌 시 _{i} 접미사 추가)
# [...]
# 논리 or를 활용한 조건부 브랜치
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
# result = easy *or* hard
result = easy | hard  # 선택된 브랜치의 결과를 가져옴
```

## 프로그램 요약 확인하기

프로그램의 표 형식 요약을 출력하려면:

```python
program.summary()
```

또는 플롯으로 확인할 수 있습니다 (시스템 문서화에 유용합니다):

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

<div align="center">
<img src="../docs/assets/examples/datamodel_designer.png" alt="데이터 모델 디자이너 프로그램" width="600">

<em>plot_program으로 시각화한 데이터 모델 디자이너 프로그램: Input → DeepAgent. 학습 가능한 모듈은 녹색으로 표시됩니다.</em>
</div>

## 프로그램 실행하기

프로그램을 실행하려면 다음과 같이 하세요:

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

## 프로그램/에이전트 학습시키기

```python
# 기본 언어/임베딩 모델을 설정하면 (Keras 스타일의) 문자열 식별자로
# 파이프라인/학습을 구성할 수 있습니다.
# 세밀한 제어가 필요하면 여전히 클래스를 직접 인스턴스화할 수 있습니다.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")
synalinks.set_default_embedding_model("gemini/text-embedding-004")


async def main():

    # ... 여러분의 프로그램 정의

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

## 저장 및 불러오기

전체 아키텍처와 변수(프로그램의 상태)를 JSON 파일로 저장하려면:

```python
program.save("my_program.json")
```

불러오려면 다음과 같이 하세요:

```python
loaded_program = synalinks.Program.load("my_program.json")
```

프로그램의 상태(변수)만 JSON으로 저장하려면:

```python
program.save_variables("my_program.variables.json")
```

변수를 불러오려면 (동일한 아키텍처의 프로그램이 필요합니다):

```python
program.load_variables("my_program.variables.json")
```

## 로깅

로깅을 활성화하려면 스크립트 시작 부분에서 다음을 사용하세요:

```python
synalinks.enable_logging()
```

## 관측성

Synalinks는 MLflow를 통한 내장 관측성을 제공하여 프로그램을 추적하고 모니터링할 수 있습니다.

> **중요**: 모듈을 생성하기 **전에** `enable_observability()`를 호출하세요.

```python
import synalinks

# 먼저 관측성을 활성화합니다
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # 선택 사항: MLflow 서버 URI
    experiment_name="my_experiment",  # 선택 사항: 기본값은 "synalinks_traces"
)

# 그런 다음 모듈을 생성하세요 - 자동으로 추적됩니다
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

학습 메트릭과 아티팩트를 위해서는 `Monitor` 콜백을 사용하세요:

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

고급 설정은 [관측성 가이드](https://synalinks.github.io/synalinks/guides/Observability/)를 참고하세요.

### 더 알아보기

[문서](https://synalinks.github.io/synalinks/)를 읽으면 더 많은 것을 배울 수 있습니다. 궁금한 점이 있다면 [FAQ](https://synalinks.github.io/synalinks/FAQ/)가 도움이 될 수 있습니다.

### 기여

추가 모듈, 메트릭, 옵티마이저 구현 등 어떤 형태의 기여도 환영합니다.
자세한 정보가 필요하거나 여러분의 아이디어(또는 논문의 아이디어)를 구현하는 데 도움이 필요하면 디스코드에 참여해 주세요.

다만 추가되는 모든 메트릭/모듈/옵티마이저는 코어 팀의 승인을 받아야 한다는 점을 유의해 주세요. 현재 주요 LM 프레임워크 대부분에서 볼 수 있는 것처럼 통제되지 않은 확장이 나쁜 소프트웨어 관행으로 이어지는 것을 피하기 위해, 우리는 라이브러리를 최대한 미니멀하고 깔끔하게 유지하고자 합니다.

구체적인 피드백이나 기능 요청이 있다면 [이슈](https://github.com/SynaLinks/synalinks/issues)를 열어 주시기 바랍니다.

### 기여자

여러분의 기여, 피드백, 지원이 이 프로젝트를 성장시키는 원동력입니다.

작은 버그 수정부터 주요 기능까지, 오픈 협업과 뉴로심볼릭 AI의 미래를 믿어 주셔서 감사합니다.

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### 커뮤니티

뉴로심볼릭 시스템과 AI의 미래에 대해 더 알아보려면 커뮤니티에 참여하세요. 배경이나 교육 수준에 관계없이 다양한 분들의 참여를 환영합니다.

### 인용하기

이 작업은 Keras의 저자인 François Chollet의 지도 아래 이루어졌습니다. 이 작업이 여러분의 연구에 유용하다면 다음 bibtex 항목을 사용해 주세요:

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### 크레딧

Synalinks는 다음 오픈소스 프로젝트들의 훌륭한 작업이 없었다면 불가능했을 것입니다:

- [Keras](https://keras.io/): 그래프 기반 연산 백본, API, 전반적인 코드, 설계 및 철학.
- [DSPy](https://dspy.ai/): 모듈/옵티마이저에 대한 영감.
- [Pydantic](https://docs.pydantic.dev/latest/): 백엔드 데이터 레이어.
- [LiteLLM](https://docs.litellm.ai/docs/): LM 통합.
- [DuckDB](https://duckdb.org/), [Ladybug](https://ladybugdb.com/), [LanceDB](https://www.lancedb.com/): 놀라운 내장 데이터베이스.
- [MirageAI](https://www.strukto.ai/mirage): 놀라운 샌드박스!
