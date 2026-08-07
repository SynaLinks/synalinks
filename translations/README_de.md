<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/synalinks-dark.svg">
  <img height=200 alt="Synalinks" src="../img/synalinks-light.svg">
</picture>
</div>

<div align="center">

<b>Von der Idee zur Produktion in nur wenigen Zeilen</b>

<em>Das erste neuro-symbolische Language-Model-(LM-)Framework, das die Einfachheit von Keras mit der Strenge der Best Practices des Deep Learning verbindet.</em>

<b>Erstelle [RAGs](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/), [Agenten mit Tool-Nutzung](https://synalinks.github.io/synalinks/guides/Agents/), Multi-Agenten-Systeme, [rekursive Agenten](https://synalinks.github.io/synalinks/guides/Recursive%20Language%20Model%20Agent/) und mehr in nur wenigen Zeilen</b>

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
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>Dokumentation</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>Codebeispiele</strong></a> .
  <a href="https://github.com/SynaLinks/synalinks/tree/main/guides" target="_blank"><strong>Guides</strong></a>
</p>

</div>

<div align="center">

Wenn du Synalinks nützlich findest, gib dem Repo bitte einen Stern! Hilf uns, mehr AI/ML-Engineers zu erreichen und die Community wachsen zu lassen.

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

Du möchtest Synalinks mit deinem eigenen Coding-Agenten (Claude Code, Cursor, Copilot usw.) verwenden? Füge deinem Agenten die Synalinks-spezifischen Skills aus [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills) auf GitHub hinzu; sie bringen ihm die Konventionen des Frameworks bei und geben ihm den nötigen Kontext, um sofort Synalinks-Programme zu erstellen.

</div>

## Was ist Synalinks?

Synalinks ist ein neuro-symbolisches Open-Source-Framework, mit dem sich fortgeschrittene LM-basierte Anwendungen, darunter RAGs, autonome Agenten und selbst-evolvierende Reasoning-Systeme, einfach erstellen, trainieren, evaluieren und deployen lassen.

Stell es dir wie Keras für Language-Model-Anwendungen vor: eine saubere, deklarative API, mit der du

- [`Module`](https://synalinks.github.io/synalinks/guides/Modules/) **komponierst**, so wie du es mit Deep-Learning-`Layer`n tun würdest.
- Mit In-Context Reinforcement Learning **[trainierst und optimierst](https://synalinks.github.io/synalinks/guides/Training/)**.
- Als [REST-APIs](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) oder [MCP-Server](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/) **deployst**.

### Zentrale Prinzipien

- **Progressive Komplexität**: [Einfach anfangen und ganz natürlich zu fortgeschrittenen Systemen wachsen](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Neuro-symbolisches Lernen**: Kombiniere [Logik, Struktur](https://synalinks.github.io/synalinks/guides/Data%20Models/) und [Sprachmodelle](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **In-Context-Optimierung**: [Verbessere das Reasoning des Modells ohne erneutes Training der Gewichte](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/).

## Für wen ist es gedacht?

<div align="center">

| Rolle                     | Warum Synalinks hilft                                       |
| ------------------------- | ----------------------------------------------------------- |
| **AI-Entwickler**      | Erstelle komplexe, produktionsreife LM-Apps ohne Boilerplate. |
| **AI-Forscher**     | Prototypisiere neuro-symbolische und RL-in-Context-Systeme in kürzester Zeit.    |
| **Data Scientists**    | Integriere LM-Workflows mit APIs und Datenbanken.               |
| **Studierende/Hobbyisten** | Lerne AI-Komposition in einem sauberen, intuitiven Framework.       |

</div>

## Warum Synalinks?

Es gibt heute viele Frameworks; hier ist, was Synalinks anders macht:

- **Eingebettete, containerfreie Sandbox**: Agenten führen nicht vertrauenswürdigen Code und Tools in einer [sicheren, isolierten Laufzeitumgebung](https://synalinks.github.io/synalinks/guides/Agents/) aus, die **weder Docker noch einen externen Sandbox-Dienst benötigt**. Der gesamte Stack ist reines Python und einbettbar und damit ideal für Scripting, Forschung, Serverless-/Cloud-Deployments (S3, Lambda, Notebooks usw.) oder sogar für den Bau von CLI-Harnessen!
- **Unterstützung für eingebettete Datenbanken**: Baue [graphbasiertes RAG und agentische Gedächtnisse](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/) mit **constrained Knowledge-Graph-Extraktion** und **automatischer semantischer Deduplizierung** auf Basis einer eingebetteten Graphdatenbank, ohne separaten Graph-Server. Zusätzlich steht eine schnelle eingebettete **SQL-Wissensbasis** zur Verfügung, um relationale Daten zu speichern und Vektor-/SQL-RAGs zu bauen.
- **In-Context RL zur Optimierung deiner Prompts (und alles Weiteren)**: [Trainiere und optimiere](https://synalinks.github.io/synalinks/guides/Training/) Prompts, Few-Shot-Beispiele und [jede trainierbare Variable](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/) pro Modul, **ohne die Modellgewichte anzufassen**, und zwar mit der vertrauten `.compile()`- / `.fit()`- / `.evaluate()`- / `.predict()`-API.
- **Müheloser Modellwechsel**: Setze einmal einen Default mit `synalinks.set_default_language_model(...)` oder übergib einen String-Identifier und wechsle über [LiteLLM](https://docs.litellm.ai/docs/) zwischen Ollama, vLLM, OpenAI, Azure, Anthropic, Mistral, Groq, Gemini, xAI, Cohere, DeepSeek, Together AI, OpenRouter, AWS Bedrock und Doubleword, inklusive [multikriterieller Modellauswahl](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/), um das beste Modell für Kosten/Qualität zu finden.
- **Scaffolding mit einem Befehl, bring deinen eigenen Coding-Agenten mit**: Setze mit `synalinks init` ein produktionsreifes Projekt auf (Batteries-included-Vorlagen für Skripte, Agenten und Training) und füge dann die offiziellen [Synalinks-Skills](https://github.com/SynaLinks/synalinks-skills) hinzu, damit Claude Code, Cursor, Copilot und Co. von Anfang an idiomatischen Synalinks-Code schreiben.

Dazu alles, was man von einem produktionsreifen Framework erwartet:

- **NEU**: Alle Agenten unterstützen jetzt [Agent Skills](https://agentskills.io/home), `AGENTS.md` und Sub-Agenten.
- **[Constrained Structured Outputs](https://synalinks.github.io/synalinks/guides/Data%20Models/)** (JSON) für Korrektheit
- **Chat-Completions-kompatible Message-API**: Messages spiegeln das OpenAI-Chat-Completions-Format Schlüssel für Schlüssel wider, litellm-erweitert um `reasoning_content` und `thinking_blocks`, sodass das Provider-Reasoning Multi-Turn-Roundtrips übersteht. Ebenfalls unterstützt: **[multimodale Eingaben](https://synalinks.github.io/synalinks/guides/Multimodal%20Inputs/)** (Bilder und Audio als Standard-Content-Parts)
- **Versionierbare**, JSON-serialisierbare [Pipelines](https://synalinks.github.io/synalinks/guides/Programs/)
- **Automatische [asynchrone und parallele Ausführung](https://synalinks.github.io/synalinks/guides/Programs/)** per Default
- **[Metriken](https://synalinks.github.io/synalinks/guides/Metrics/), [Rewards](https://synalinks.github.io/synalinks/guides/Rewards/) und [Datasets](https://synalinks.github.io/synalinks/guides/Datasets/)** direkt eingebaut
- **API-ready**: Deployment mit [FastAPI](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) oder [FastMCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)
- **[KerasTuner-Kompatibilität](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/)** für die Hyperparameter-Suche
- **Eingebaute [Callbacks](https://synalinks.github.io/synalinks/guides/Callbacks/) und Hooks** für [Observability](https://synalinks.github.io/synalinks/guides/Observability/) (inklusive eines MLflow-`Monitor`-Callbacks)

# Voraussetzungen

- Python 3.12 oder neuer
- WSL2 für Windows-Nutzer

## Quickstart in 3 Sekunden mit `uv` (empfohlen)

Falls du `uv` noch nicht kennst: Installiere es [hier](https://docs.astral.sh/uv/getting-started/installation/).

Folge den Anweisungen, um in 3 Sekunden ein neues Synalinks-Projekt zu starten:

```shell
uvx synalinks init
```

---

Du kannst die Bibliothek auch in einem neuen Projekt installieren mit:

```shell
uv add synalinks
```

Um deinen Coding-Agenten in einen AI-Engineer zu verwandeln, führe im Wurzelverzeichnis deines Projekts Folgendes aus:

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

## Beispiel

Synalinks-Agenten können jetzt die [`AGENTS.md`](https://agents.md)-Konventionen
deines Projekts lesen und [Agent Skills](https://agentskills.io/home) nutzen. Das
Beispiel unten bindet die offiziellen [Synalinks-Skills](https://github.com/SynaLinks/synalinks-skills)
in einen [`DeepAgent`](https://synalinks.github.io/synalinks/guides/Agents/) ein, einen
sandboxed Coding-Agenten, und bittet ihn, die Eingabe-/Ausgabe-Datenmodelle für eine
Aufgabe zu entwerfen und sie in einen `workspace/`-Ordner zu schreiben.

Richte zuerst den Workspace ein. Installiere den offiziellen Synalinks-Skill mit dem
[`skills`](https://skills.sh)-CLI und füge eine `AGENTS.md` hinzu. Der Skill landet unter
dem Workdir, sodass der sandboxed Agent seinen Inhalt bei Bedarf lesen kann:

```shell
mkdir -p workspace && cd workspace
# Installiert den Skill `synalinks` in ./.agents/skills/ und schreibt skills-lock.json.
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

Das ergibt die folgende Struktur; `.agents/skills` ist der Skills-*Root* (ein
Unterordner pro Skill, jeder mit einer `SKILL.md`):

```text
workspace/
├── AGENTS.md                     # wird als Konventionen des Agenten injiziert
├── skills-lock.json              # pinnt den Skill auf ein Quell-Repo + Content-Hash
└── .agents/
    └── skills/                   # der Skills-Root
        └── synalinks/
            └── SKILL.md          # Name + Beschreibung sichtbar; Inhalt wird bei Bedarf gelesen
```

`main.py`:

```python
import synalinks
import asyncio

# Den Default einmal setzen; Module übernehmen ihn automatisch.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")


# Die strukturierte finale Antwort des Agenten.
class Deliverable(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="What was created and where",
    )
    files: list[str] = synalinks.Field(
        description="Paths of the files written into the workspace",
    )


async def main():
    # Ein DeepAgent kommuniziert über ChatMessages (er ist ein Coding-Agent).
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    agent = synalinks.DeepAgent(
        data_model=Deliverable,
        # Die Sandbox wird aus diesem Verzeichnis initialisiert (host-sicher:
        # die Schreibzugriffe des Agenten landen in der Sandbox-Kopie, nie auf
        # deiner Festplatte). Seine `AGENTS.md` wird injiziert, damit der Agent
        # deinen Konventionen folgt.
        workdir="workspace",
        # Der Skills-Root (installiert per `skills add`). Wird dem Agenten als
        # `<available_skills>` aufgelistet; er liest jede `SKILL.md` bei Bedarf
        # aus der Sandbox; deshalb liegen die Skills unter `workdir`.
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

## Datenmodell-Operatoren

Synalinks stellt Python-Operatoren zum Kombinieren und Manipulieren von Datenmodellen bereit und ermöglicht so ausgefeilte Kontrollflüsse. Siehe den [Control-Flow-Guide](https://synalinks.github.io/synalinks/guides/Control%20Flow/) für die Routing-, Fan-out- und Merge-Muster, die diese Operatoren ermöglichen:

<div align="center">

| Operator | Name | Beschreibung | Anwendungsfall |
| :---: | --- | --- | --- |
| `+` | Konkatenation | Kombiniert die Felder beider Datenmodelle. Wirft eine Exception, wenn eines `None` ist. | Zusammenführen der Ausgaben paralleler Zweige |
| `&` | Logisches Und | Sichere Konkatenation, die `None` zurückgibt, wenn eine der Eingaben `None` ist. | Kombinieren mit potenziell null-wertigen Zweigausgaben |
| `\|` | Logisches Oder | Gibt das Nicht-`None`-Datenmodell zurück. Sind beide nicht `None`, werden sie zusammengeführt. | Einsammeln der Ausgaben bedingter Zweige |
| `^` | Logisches Xor | Gibt die Daten zurück, wenn genau eine Eingabe nicht `None` ist, andernfalls `None`. | Exklusive Zweigauswahl |
| `~` | Logisches Nicht | Gibt `None` zurück, wenn die Eingabe nicht `None` ist, oder ein leeres Datenmodell bei `None`. | Invertieren von Zweigbedingungen |
| `in` | Enthält | Prüft, ob ein String-Schlüssel in den Schema-Properties existiert oder ob das Schema eines anderen Datenmodells enthalten ist. Gibt `True` oder `False` zurück. | Bedingte Feldprüfung, Schema-Validierung |

</div>

```python
# Parallele Zweige mit Konkatenation
x1 = await generator1(inputs)
x2 = await generator2(inputs)
# combined = x1 *und* x2
combined = x1 & x2  # Beide Ausgaben zusammenführen (Suffix _{i} bei Schlüsselkollision)
# [...]
# Bedingte Zweige mit logischem Oder
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
# result = easy *oder* hard
result = easy | hard  # Den jeweils ausgewählten Zweig erhalten
```

## Eine Zusammenfassung deines Programms erhalten

Um eine tabellarische Zusammenfassung deines Programms auszugeben:

```python
program.summary()
```

Oder als Plot (nützlich, um dein System zu dokumentieren):

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

<div align="center">
<img src="../docs/assets/examples/datamodel_designer.png" alt="Programm des Datenmodell-Designers" width="600">

<em>Das Datenmodell-Designer-Programm, visualisiert mit plot_program: Input → DeepAgent. Trainierbare Module sind grün markiert.</em>
</div>

## Dein Programm ausführen

Um dein Programm auszuführen, verwende Folgendes:

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

## Dein Programm/deinen Agenten trainieren

```python
# Das Setzen der Default-Sprach-/Embedding-Modelle erlaubt dir,
# den String-Identifier (Keras-artig) zur Konfiguration deiner Pipeline/deines Trainings zu nutzen.
# Du kannst die Klassen weiterhin instanziieren, wenn du feingranulare Kontrolle möchtest.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")
synalinks.set_default_embedding_model("gemini/text-embedding-004")


async def main():

    # ... deine Programmdefinition

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

## Speichern & Laden

Um die gesamte Architektur samt Variablen (den Zustand des Programms) in eine JSON-Datei zu speichern:

```python
program.save("my_program.json")
```

Um es zu laden:

```python
loaded_program = synalinks.Program.load("my_program.json")
```

Um nur den Zustand deines Programms (die Variablen) als JSON zu speichern:

```python
program.save_variables("my_program.variables.json")
```

Um seine Variablen zu laden (erfordert ein Programm mit derselben Architektur):

```python
program.load_variables("my_program.variables.json")
```

## Logging

Um Logging zu aktivieren, verwende Folgendes am Anfang deines Skripts:

```python
synalinks.enable_logging()
```

## Observability

Synalinks bietet eingebaute Observability über MLflow zum Tracen und Monitoren deiner Programme.

> **Wichtig**: Rufe `enable_observability()` auf, **bevor** du irgendwelche Module erstellst.

```python
import synalinks

# Zuerst Observability aktivieren
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # Optional: URI des MLflow-Servers
    experiment_name="my_experiment",  # Optional: Standard ist "synalinks_traces"
)

# Dann deine Module erstellen; sie werden automatisch getract
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

Für Trainingsmetriken und Artefakte verwende den `Monitor`-Callback:

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

Siehe den [Observability-Guide](https://synalinks.github.io/synalinks/guides/Observability/) für erweiterte Konfiguration.

### Mehr erfahren

Mehr erfährst du in unserer [Dokumentation](https://synalinks.github.io/synalinks/). Bei Fragen hilft dir vielleicht die [FAQ](https://synalinks.github.io/synalinks/FAQ/) weiter.

### Beiträge

Beiträge sind willkommen, sei es für die Implementierung zusätzlicher Module, Metriken oder Optimizer.
Für weitere Informationen oder Hilfe bei der Umsetzung deiner Ideen (oder solcher aus einem Paper) tritt bitte unserem Discord bei.

Beachte, dass jede zusätzliche Metrik bzw. jedes zusätzliche Modul und jeder Optimizer vom Core-Team genehmigt werden muss. Wir wollen die Bibliothek so minimal und sauber wie möglich halten, um ein unkontrolliertes Wachstum zu vermeiden, das, wie bei den meisten führenden LM-Frameworks, zu schlechten Software-Praktiken führt.

Wenn du konkretes Feedback oder Feature-Wünsche hast, laden wir dich ein, ein [Issue](https://github.com/SynaLinks/synalinks/issues) zu eröffnen.

### Mitwirkende

Eure Beiträge, euer Feedback und eure Unterstützung sind es, die dieses Projekt gedeihen lassen.

Von kleinen Bugfixes bis zu großen Features: Danke, dass ihr an offene Zusammenarbeit und die Zukunft der neuro-symbolischen AI glaubt.

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### Community

Tritt unserer Community bei, um mehr über neuro-symbolische Systeme und die Zukunft der AI zu erfahren. Wir freuen uns über die Teilnahme von Menschen mit ganz unterschiedlichen Hintergründen und Bildungswegen.

### Unsere Arbeit zitieren

Diese Arbeit entstand unter der Betreuung von François Chollet, dem Autor von Keras. Wenn diese Arbeit für deine Forschung nützlich ist, verwende bitte den folgenden BibTeX-Eintrag:

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### Danksagung

Synalinks wäre ohne die großartige Arbeit der folgenden Open-Source-Projekte nicht möglich:

- [Keras](https://keras.io/) für das graphbasierte Berechnungs-Backbone, die API sowie Code, Design und Philosophie insgesamt.
- [DSPy](https://dspy.ai/) für die Inspiration zu Modulen/Optimizern.
- [Pydantic](https://docs.pydantic.dev/latest/) für die Backend-Datenschicht.
- [LiteLLM](https://docs.litellm.ai/docs/) für die LM-Integrationen.
- [DuckDB](https://duckdb.org/), [Ladybug](https://ladybugdb.com/), [LanceDB](https://www.lancedb.com/) für ihre großartigen eingebetteten Datenbanken.
- [MirageAI](https://www.strukto.ai/mirage) für ihre großartige Sandbox!
