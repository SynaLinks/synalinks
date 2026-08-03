<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/synalinks-dark.svg">
  <img height=200 alt="Synalinks" src="../img/synalinks-light.svg">
</picture>
</div>

<div align="center">

<b>Dall'idea alla produzione in poche righe</b>

<em>Il primo framework neuro-simbolico per Language Model (LM) che unisce la semplicità di Keras al rigore delle best practice del Deep Learning.</em>

<b>Costruisci [RAG](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/), [agenti che usano strumenti](https://synalinks.github.io/synalinks/guides/Agents/), sistemi multi-agente, [agenti ricorsivi](https://synalinks.github.io/synalinks/guides/Recursive%20Language%20Model%20Agent/) e altro ancora in poche righe</b>

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
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>Documentazione</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>Esempi di codice</strong></a> .
  <a href="https://github.com/SynaLinks/synalinks/tree/main/guides" target="_blank"><strong>Guide</strong></a>
</p>

</div>

<div align="center">

Se trovi utile Synalinks, metti una stella al repo! Aiutaci a raggiungere più ingegneri AI/ML e a far crescere la community.

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

Vuoi usare Synalinks con il tuo coding agent (Claude Code, Cursor, Copilot, ecc.)? Aggiungi al tuo agente le skill specifiche per Synalinks da [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills) su GitHub; gli insegnano le convenzioni del framework e gli forniscono il contesto necessario per costruire programmi Synalinks fin da subito.

</div>

## Che cos'è Synalinks?

Synalinks è un framework neuro-simbolico open source che rende semplice creare, addestrare, valutare e distribuire applicazioni avanzate basate su LM, tra cui RAG, agenti autonomi e sistemi di ragionamento auto-evolutivi.

Pensa a un Keras per le applicazioni basate su Language Model: un'API pulita e dichiarativa in cui:

- **Componi** i [`Module`](https://synalinks.github.io/synalinks/guides/Modules/) come faresti con i `Layer` del deep learning.
- **[Addestri e ottimizzi](https://synalinks.github.io/synalinks/guides/Training/)** con il reinforcement learning in-context.
- **Fai il deploy** come [API REST](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) o [server MCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/).

### Principi chiave

- **Complessità progressiva**: [inizia in modo semplice e cresci verso l'avanzato in modo naturale](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Apprendimento neuro-simbolico**: combina [logica, struttura](https://synalinks.github.io/synalinks/guides/Data%20Models/) e [language model](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Ottimizzazione in-context**: [migliora il ragionamento del modello senza riaddestrare i pesi](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/).

## A chi è rivolto?

<div align="center">

| Ruolo                      | Perché Synalinks è utile                                         |
| ------------------------- | ----------------------------------------------------------- |
| **Sviluppatori AI**      | Costruire applicazioni LM complesse e pronte per la produzione senza boilerplate. |
| **Ricercatori AI**     | Prototipare rapidamente sistemi neuro-simbolici e di RL in-context.    |
| **Data Scientist**    | Integrare workflow LM con API e database.               |
| **Studenti/Appassionati** | Imparare la composizione di sistemi AI con un framework pulito e intuitivo.       |

</div>

## Perché Synalinks?

Oggi esistono molti framework; ecco cosa fa Synalinks di diverso:

- **Sandbox integrata, senza container** : gli agenti eseguono codice e strumenti non fidati in un [runtime sicuro e isolato](https://synalinks.github.io/synalinks/guides/Agents/) che **non richiede Docker né servizi di sandbox esterni**. L'intero stack è puro Python e integrabile, quindi è perfetto per lo scripting, la ricerca, i deployment serverless/cloud (S3, Lambda, notebook, ecc.) o persino per creare harness CLI!
- **Supporto per database embedded** : costruisci [RAG basati su grafi e memorie agentiche](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/) con **estrazione vincolata di Knowledge Graph** e **deduplicazione semantica automatica**, sopra un database a grafo embedded, senza alcun server di grafi separato da gestire. In più, è disponibile una veloce **knowledge base SQL** embedded per archiviare dati relazionali e costruire RAG vettoriali/SQL.
- **RL in-context per ottimizzare i tuoi prompt (e qualsiasi altra cosa)** : [addestra e ottimizza](https://synalinks.github.io/synalinks/guides/Training/) prompt, esempi few-shot e [qualsiasi variabile addestrabile](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/) per ciascun modulo **senza toccare i pesi del modello**, usando la familiare API `.compile()` / `.fit()` / `.evaluate()` / `.predict()`.
- **Cambio di modello senza sforzo** : imposta un default una sola volta con `synalinks.set_default_language_model(...)` oppure passa un identificatore stringa, e passa da Ollama, vLLM, OpenAI, Azure, Anthropic, Mistral, Groq, Gemini, xAI, Cohere, DeepSeek, Together AI, OpenRouter, AWS Bedrock e Doubleword tramite [LiteLLM](https://docs.litellm.ai/docs/), inclusa la [selezione multi-obiettivo del modello](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/) per scegliere il modello migliore in termini di costo/qualità.
- **Scaffolding in un solo comando, porta il tuo coding agent** : avvia un progetto pronto per la produzione con `synalinks init` (template completi per script, agenti e addestramento), poi aggiungi le [skill Synalinks](https://github.com/SynaLinks/synalinks-skills) ufficiali così che Claude Code, Cursor, Copilot e compagnia scrivano codice Synalinks idiomatico fin dall'inizio.

Più tutto ciò che ti aspetteresti da un framework di livello produzione:

- **NOVITÀ**: ora tutti gli agenti supportano [Agent Skills](https://agentskills.io/home), `AGENTS.md` e sub-agenti.
- **[Output strutturati vincolati](https://synalinks.github.io/synalinks/guides/Data%20Models/)** (JSON) per garantire la correttezza
- **API dei messaggi compatibile con le Chat Completions**: i messaggi rispecchiano il formato OpenAI Chat Completions chiave per chiave + le estensioni litellm con `reasoning_content` e `thinking_blocks`, così il reasoning del provider sopravvive ai round-trip multi-turno. Gestisce inoltre **[input multimodali](https://synalinks.github.io/synalinks/guides/Multimodal%20Inputs/)** (immagini e audio come parti di contenuto standard)
- **[Pipeline](https://synalinks.github.io/synalinks/guides/Programs/) versionabili** e serializzabili in JSON
- **[Esecuzione asincrona e parallela](https://synalinks.github.io/synalinks/guides/Programs/) automatica** di default
- **[Metriche](https://synalinks.github.io/synalinks/guides/Metrics/), [reward](https://synalinks.github.io/synalinks/guides/Rewards/) e [dataset](https://synalinks.github.io/synalinks/guides/Datasets/)** integrati
- **Pronto per le API**: deploy con [FastAPI](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) o [FastMCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)
- **[Compatibilità con KerasTuner](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/)** per la ricerca degli iperparametri
- **[Callback](https://synalinks.github.io/synalinks/guides/Callbacks/) e hook integrati** per l'[osservabilità](https://synalinks.github.io/synalinks/guides/Observability/) (incluso un callback MLflow `Monitor`)

# Requisiti

- Python 3.12 o superiore
- WSL2 per gli utenti Windows

## Quickstart in 3 secondi con `uv` (consigliato)

Se non conosci `uv`, installalo da [qui](https://docs.astral.sh/uv/getting-started/installation/).

Segui le istruzioni per avviare un nuovo progetto synalinks in 3 secondi:

```shell
uvx synalinks init
```

---

Puoi anche installare la libreria in un nuovo progetto con:

```shell
uv add synalinks
```

Per trasformare il tuo coding agent in un ingegnere AI, esegui questo nella root del tuo progetto:

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

## Esempio

Gli agenti Synalinks possono ora leggere le convenzioni del tuo progetto in [`AGENTS.md`](https://agents.md)
e usare le [Agent Skills](https://agentskills.io/home). L'esempio
qui sotto collega le [skill Synalinks](https://github.com/SynaLinks/synalinks-skills) ufficiali
a un [`DeepAgent`](https://synalinks.github.io/synalinks/guides/Agents/), un
coding agent in sandbox, e gli chiede di progettare i data model di input/output per un
task, scrivendoli in una cartella `workspace/`.

Per prima cosa prepara il workspace. Installa la skill Synalinks ufficiale con la
CLI [`skills`](https://skills.sh) e aggiungi un `AGENTS.md`. La skill viene installata sotto
la workdir, così l'agente in sandbox può leggerne il contenuto su richiesta:

```shell
mkdir -p workspace && cd workspace
# Installa la skill `synalinks` in ./.agents/skills/ e scrive skills-lock.json.
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

Questo produce il layout qui sotto; `.agents/skills` è la *root* delle skill (una
sotto-cartella per skill, ognuna con il proprio `SKILL.md`):

```text
workspace/
├── AGENTS.md                     # iniettato come convenzioni dell'agente
├── skills-lock.json              # blocca la skill su un repo sorgente + hash del contenuto
└── .agents/
    └── skills/                   # la root delle skill
        └── synalinks/
            └── SKILL.md          # nome + descrizione esposti; corpo letto su richiesta
```

`main.py`:

```python
import synalinks
import asyncio

# Imposta il default una sola volta; i moduli lo rilevano automaticamente.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")


# La risposta finale strutturata dell'agente.
class Deliverable(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="What was created and where",
    )
    files: list[str] = synalinks.Field(
        description="Paths of the files written into the workspace",
    )


async def main():
    # Un DeepAgent conversa in ChatMessages (è un coding agent).
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    agent = synalinks.DeepAgent(
        data_model=Deliverable,
        # La sandbox viene inizializzata da questa directory (sicura per l'host:
        # le scritture dell'agente finiscono nella copia in sandbox, mai sul tuo disco).
        # Il suo `AGENTS.md` viene iniettato così l'agente segue le tue convenzioni.
        workdir="workspace",
        # La root delle skill (installata da `skills add`). Elencata all'agente come
        # `<available_skills>`; legge ogni `SKILL.md` su richiesta dalla
        # sandbox; ecco perché le skill risiedono sotto `workdir`.
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

## Operatori sui Data Model

Synalinks fornisce operatori Python per combinare e manipolare i data model, abilitando un controllo di flusso sofisticato. Consulta la [guida al Control Flow](https://synalinks.github.io/synalinks/guides/Control%20Flow/) per i pattern di routing, fan-out e merge che questi operatori rendono possibili:

<div align="center">

| Operatore | Nome | Descrizione | Caso d'uso |
| :---: | --- | --- | --- |
| `+` | Concatenazione | Combina i campi di entrambi i data model. Solleva un'eccezione se uno dei due è `None`. | Unire gli output di rami paralleli |
| `&` | And logico | Concatenazione sicura che restituisce `None` se uno degli input è `None`. | Combinare output di rami potenzialmente nulli |
| `\|` | Or logico | Restituisce il data model non-`None`. Se entrambi sono non-`None`, li unisce. | Raccogliere gli output di rami condizionali |
| `^` | Xor logico | Restituisce i dati se esattamente un input è non-`None`, altrimenti `None`. | Selezione esclusiva di un ramo |
| `~` | Not logico | Restituisce `None` se l'input è non-`None`, oppure un data model vuoto se è `None`. | Invertire le condizioni dei rami |
| `in` | Contains | Verifica se una chiave stringa esiste nelle proprietà dello schema, o se lo schema di un altro data model è contenuto. Restituisce `True` o `False`. | Controllo condizionale dei campi, validazione dello schema |

</div>

```python
# Rami paralleli con concatenazione
x1 = await generator1(inputs)
x2 = await generator2(inputs)
# combined = x1 *and* x2
combined = (
    x1 & x2
)  # Unisce entrambi gli output (aggiunge il suffisso _{i} in caso di collisione di chiavi)
# [...]
# Rami condizionali con l'or logico
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
# result = easy *or* hard
result = easy | hard  # Restituisce il ramo che è stato selezionato
```

## Ottenere un riepilogo del tuo programma

Per stampare un riepilogo tabellare del tuo programma:

```python
program.summary()
```

Oppure un grafico (utile per documentare il tuo sistema):

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

<div align="center">
<img src="../docs/assets/examples/datamodel_designer.png" alt="Programma Data Model Designer" width="600">

<em>Il programma data model designer visualizzato con plot_program: Input → DeepAgent. I moduli addestrabili sono evidenziati in verde.</em>
</div>

## Eseguire il tuo programma

Per eseguire il tuo programma usa quanto segue:

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

## Addestrare il tuo programma/agente

```python
# Impostare i language/embedding model di default ti permette
# di usare l'identificatore stringa (in stile Keras) per configurare la pipeline/l'addestramento.
# Puoi comunque istanziare le classi se vuoi un controllo più fine.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")
synalinks.set_default_embedding_model("gemini/text-embedding-004")


async def main():

    # ... la definizione del tuo programma

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

## Salvataggio e caricamento

Per salvare l'intera architettura e le variabili (lo stato del programma) in un file JSON, esegui:

```python
program.save("my_program.json")
```

Per caricarlo, esegui:

```python
loaded_program = synalinks.Program.load("my_program.json")
```

Per salvare solo lo stato del tuo programma (le variabili) in JSON:

```python
program.save_variables("my_program.variables.json")
```

Per caricare le sue variabili (richiede un programma con la stessa architettura), esegui:

```python
program.load_variables("my_program.variables.json")
```

## Logging

Per abilitare il logging, usa quanto segue all'inizio del tuo script:

```python
synalinks.enable_logging()
```

## Osservabilità

Synalinks fornisce osservabilità integrata tramite MLflow per il tracing e il monitoraggio dei tuoi programmi.

> **Importante**: chiama `enable_observability()` **prima** di creare qualsiasi modulo.

```python
import synalinks

# Abilita prima l'osservabilità
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # Opzionale: URI del server MLflow
    experiment_name="my_experiment",  # Opzionale: il default è "synalinks_traces"
)

# Poi crea i tuoi moduli: verranno tracciati automaticamente
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

Per le metriche di addestramento e gli artefatti, usa il callback `Monitor`:

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

Consulta la [guida all'Osservabilità](https://synalinks.github.io/synalinks/guides/Observability/) per la configurazione avanzata.

### Per saperne di più

Puoi saperne di più leggendo la nostra [documentazione](https://synalinks.github.io/synalinks/). Se hai domande, le [FAQ](https://synalinks.github.io/synalinks/FAQ/) potrebbero aiutarti.

### Contributi

I contributi sono benvenuti, sia per l'implementazione di moduli, metriche o optimizer aggiuntivi.
Per maggiori informazioni, o per un aiuto nell'implementare le tue idee (o quelle di un paper), unisciti al nostro discord.

Tieni presente che ogni metrica/modulo/optimizer aggiuntivo deve essere approvato dal core team: vogliamo mantenere la libreria il più minimale e pulita possibile per evitare una crescita incontrollata che porti a cattive pratiche software come nella maggior parte degli attuali framework LM più diffusi.

Se hai feedback specifici o richieste di funzionalità ti invitiamo ad aprire una [issue](https://github.com/SynaLinks/synalinks/issues).

### Contributor

I tuoi contributi, i tuoi feedback e il tuo supporto sono ciò che fa prosperare questo progetto.

Dalle piccole correzioni di bug alle funzionalità più importanti, grazie per credere nella collaborazione aperta e nel futuro dell'AI neuro-simbolica.

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### Community

Unisciti alla nostra community per saperne di più sui sistemi neuro-simbolici e sul futuro dell'AI. Accogliamo con piacere la partecipazione di persone con background o livelli di istruzione molto diversi.

### Citare il nostro lavoro

Questo lavoro è stato svolto sotto la supervisione di François Chollet, l'autore di Keras. Se questo lavoro è utile per la tua ricerca, usa la seguente voce bibtex:

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### Riconoscimenti

Synalinks non sarebbe possibile senza l'eccellente lavoro dei seguenti progetti open source:

- [Keras](https://keras.io/) per la struttura di calcolo basata su grafi, l'API e in generale il codice, il design e la filosofia.
- [DSPy](https://dspy.ai/) per l'ispirazione su moduli/optimizer.
- [Pydantic](https://docs.pydantic.dev/latest/) per il layer dati del backend.
- [LiteLLM](https://docs.litellm.ai/docs/) per le integrazioni con gli LM.
- [DuckDB](https://duckdb.org/), [Ladybug](https://ladybugdb.com/), [LanceDB](https://www.lancedb.com/) per i loro straordinari database embedded.
- [MirageAI](https://www.strukto.ai/mirage) per la loro straordinaria sandbox!
