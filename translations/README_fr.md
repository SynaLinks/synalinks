<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/synalinks-dark.svg">
  <img height=200 alt="Synalinks" src="../img/synalinks-light.svg">
</picture>
</div>

<div align="center">

<b>De l'idée à la production en quelques lignes seulement</b>

<em>Le premier framework neuro-symbolique pour modèles de langage (LM) alliant la simplicité de Keras et la rigueur des bonnes pratiques du Deep Learning.</em>

<b>Construisez des [RAGs](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/), des [agents utilisant des outils](https://synalinks.github.io/synalinks/guides/Agents/), des systèmes multi-agents, des [agents récursifs](https://synalinks.github.io/synalinks/guides/Recursive%20Language%20Model%20Agent/) et bien plus, en quelques lignes seulement</b>

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
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>Documentation</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>Exemples de code</strong></a> .
  <a href="https://github.com/SynaLinks/synalinks/tree/main/guides" target="_blank"><strong>Guides</strong></a>
</p>

</div>

<div align="center">

Si vous trouvez Synalinks utile, mettez une étoile au dépôt ! Aidez-nous à toucher davantage d'ingénieurs IA/ML et à faire grandir la communauté.

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

Vous voulez utiliser Synalinks avec votre propre agent de codage (Claude Code, Cursor, Copilot, etc.) ? Ajoutez à votre agent les skills spécifiques à Synalinks depuis [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills) sur GitHub ; elles lui enseignent les conventions du framework et lui donnent le contexte nécessaire pour construire des programmes Synalinks dès le départ.

</div>

## Qu'est-ce que Synalinks ?

Synalinks est un framework neuro-symbolique open source qui simplifie la création, l'entraînement, l'évaluation et le déploiement d'applications avancées basées sur des LM, notamment des RAGs, des agents autonomes et des systèmes de raisonnement auto-évolutifs.

Pensez à Keras pour les applications à base de modèles de langage : une API claire et déclarative où :

- Vous **composez** des [`Module`s](https://synalinks.github.io/synalinks/guides/Modules/) comme vous le feriez avec des `Layer`s de deep learning.
- Vous **[entraînez et optimisez](https://synalinks.github.io/synalinks/guides/Training/)** par apprentissage par renforcement en contexte.
- Vous **déployez** sous forme d'[API REST](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) ou de [serveurs MCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/).

### Principes clés

- **Complexité progressive** : [Commencez simplement et montez naturellement en puissance](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Apprentissage neuro-symbolique** : Combinez [logique, structure](https://synalinks.github.io/synalinks/guides/Data%20Models/) et [modèles de langage](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Optimisation en contexte** : [Améliorez le raisonnement du modèle sans réentraîner les poids](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/).

## À qui s'adresse-t-il ?

<div align="center">

| Rôle                      | Pourquoi Synalinks vous aide                                |
| ------------------------- | ----------------------------------------------------------- |
| **Développeurs IA**      | Construisez des applications LM complexes de qualité production, sans code passe-partout. |
| **Chercheurs en IA**     | Prototypez rapidement des systèmes neuro-symboliques et de RL en contexte. |
| **Data Scientists**    | Intégrez des workflows LM avec des API et des bases de données. |
| **Étudiants/Amateurs** | Apprenez la composition d'IA dans un framework clair et intuitif. |

</div>

## Pourquoi Synalinks ?

De nombreux frameworks existent aujourd'hui ; voici ce que Synalinks fait différemment :

- **Sandbox embarquée, sans conteneur** : les agents exécutent du code et des outils non fiables dans un [environnement d'exécution sûr et isolé](https://synalinks.github.io/synalinks/guides/Agents/) qui ne nécessite **ni Docker ni service de sandbox externe**. Toute la pile est en pur Python et embarquable, ce qui la rend idéale pour le scripting, la recherche, le déploiement serverless/cloud (S3, Lambda, notebooks, etc.) ou même pour créer des harnais CLI !
- **Support de bases de données embarquées** : construisez des [RAG à base de graphes et des mémoires agentiques](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/) avec **extraction contrainte de graphes de connaissances** et **déduplication sémantique automatique**, au-dessus d'une base de données graphe embarquée, sans aucun serveur de graphes séparé à faire tourner. De plus, une **base de connaissances SQL** embarquée et rapide est disponible pour stocker des données relationnelles et construire des RAG vectoriels/SQL.
- **RL en contexte pour optimiser vos prompts (et tout le reste)** : [entraînez et optimisez](https://synalinks.github.io/synalinks/guides/Training/) les prompts, les exemples few-shot et [toute variable entraînable](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/) par module **sans toucher aux poids du modèle**, grâce à l'API familière `.compile()` / `.fit()` / `.evaluate()` / `.predict()`.
- **Changement de modèle sans effort** : définissez une valeur par défaut une seule fois avec `synalinks.set_default_language_model(...)` ou passez un identifiant sous forme de chaîne, et basculez entre Ollama, vLLM, OpenAI, Azure, Anthropic, Mistral, Groq, Gemini, xAI, Cohere, DeepSeek, Together AI, OpenRouter, AWS Bedrock et Doubleword via [LiteLLM](https://docs.litellm.ai/docs/), y compris la [sélection multi-objectifs de modèles](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/) pour choisir le meilleur modèle selon le rapport coût/qualité.
- **Un projet en une commande, avec l'agent de codage de votre choix** : initialisez un projet prêt pour la production avec `synalinks init` (des templates tout inclus pour les scripts, les agents et l'entraînement), puis ajoutez les [skills Synalinks](https://github.com/SynaLinks/synalinks-skills) officielles pour que Claude Code, Cursor, Copilot et consorts écrivent du code Synalinks idiomatique dès le départ.

Plus tout ce que vous attendez d'un framework de qualité production :

- **NOUVEAU** : Tous les agents supportent désormais les [Agent Skills](https://agentskills.io/home), `AGENTS.md` et les sous-agents.
- **[Sorties structurées contraintes](https://synalinks.github.io/synalinks/guides/Data%20Models/)** (JSON) pour garantir la validité
- **API de messages compatible Chat Completions** : les messages reflètent le format Chat Completions d'OpenAI clé pour clé + les extensions litellm `reasoning_content` et `thinking_blocks`, afin que le raisonnement du fournisseur survive aux allers-retours multi-tours. Gère aussi les **[entrées multimodales](https://synalinks.github.io/synalinks/guides/Multimodal%20Inputs/)** (images et audio comme parties de contenu standard)
- **[Pipelines](https://synalinks.github.io/synalinks/guides/Programs/) versionnables**, sérialisables en JSON
- **[Exécution asynchrone et parallèle](https://synalinks.github.io/synalinks/guides/Programs/) automatique** par défaut
- **[Métriques](https://synalinks.github.io/synalinks/guides/Metrics/), [récompenses](https://synalinks.github.io/synalinks/guides/Rewards/) et [jeux de données](https://synalinks.github.io/synalinks/guides/Datasets/)** intégrés
- **Prêt pour les API** : déployez avec [FastAPI](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) ou [FastMCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)
- **[Compatibilité KerasTuner](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/)** pour la recherche d'hyperparamètres
- **[Callbacks](https://synalinks.github.io/synalinks/guides/Callbacks/) et hooks intégrés** pour l'[observabilité](https://synalinks.github.io/synalinks/guides/Observability/) (y compris un callback MLflow `Monitor`)

# Prérequis

- Python 3.12 ou plus
- WSL2 pour les utilisateurs Windows

## Démarrage en 3 s avec `uv` (recommandé)

Si vous ne connaissez pas `uv`, installez-le [ici](https://docs.astral.sh/uv/getting-started/installation/).

Suivez les instructions pour démarrer un nouveau projet synalinks en 3 s :

```shell
uvx synalinks init
```

---

Vous pouvez aussi installer la bibliothèque dans un nouveau projet avec :

```shell
uv add synalinks
```

Pour transformer votre agent de codage en ingénieur IA, exécutez ceci à la racine de votre projet :

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

## Exemple

Les agents Synalinks peuvent désormais lire les conventions [`AGENTS.md`](https://agents.md)
de votre projet et utiliser les [Agent Skills](https://agentskills.io/home). L'exemple
ci-dessous branche les [skills Synalinks](https://github.com/SynaLinks/synalinks-skills)
officielles sur un [`DeepAgent`](https://synalinks.github.io/synalinks/guides/Agents/), un
agent de codage sandboxé, et lui demande de concevoir les modèles de données d'entrée/sortie
pour une tâche donnée, en les écrivant dans un dossier `workspace/`.

Commencez par préparer le workspace. Installez la skill Synalinks officielle avec la
CLI [`skills`](https://skills.sh) et ajoutez un `AGENTS.md`. La skill s'installe sous
le répertoire de travail, si bien que l'agent sandboxé peut en lire le contenu à la demande :

```shell
mkdir -p workspace && cd workspace
# Installe la skill `synalinks` dans ./.agents/skills/ et écrit skills-lock.json.
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

On obtient l'arborescence ci-dessous ; `.agents/skills` est la *racine* des skills (un
sous-dossier par skill, chacun contenant un `SKILL.md`) :

```text
workspace/
├── AGENTS.md                     # injecté comme conventions de l'agent
├── skills-lock.json              # fige la skill sur un dépôt source + un hash de contenu
└── .agents/
    └── skills/                   # la racine des skills
        └── synalinks/
            └── SKILL.md          # nom + description exposés ; contenu lu à la demande
```

`main.py` :

```python
import synalinks
import asyncio

# Définissez la valeur par défaut une seule fois ; les modules la récupèrent automatiquement.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")


# La réponse finale structurée de l'agent.
class Deliverable(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="What was created and where",
    )
    files: list[str] = synalinks.Field(
        description="Paths of the files written into the workspace",
    )


async def main():
    # Un DeepAgent converse en ChatMessages (c'est un agent de codage).
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    agent = synalinks.DeepAgent(
        data_model=Deliverable,
        # La sandbox est initialisée à partir de ce répertoire (sans risque pour l'hôte :
        # les écritures de l'agent vont dans la copie sandbox, jamais sur votre disque).
        # Son `AGENTS.md` est injecté pour que l'agent suive vos conventions.
        workdir="workspace",
        # La racine des skills (installée par `skills add`). Présentée à l'agent sous
        # forme de `<available_skills>` ; il lit chaque `SKILL.md` à la demande depuis
        # la sandbox ; c'est pourquoi les skills se trouvent sous `workdir`.
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

## Opérateurs de modèles de données

Synalinks fournit des opérateurs Python pour combiner et manipuler les modèles de données, permettant un flux de contrôle sophistiqué. Consultez le [guide Control Flow](https://synalinks.github.io/synalinks/guides/Control%20Flow/) pour les patterns de routage, de fan-out et de fusion que ces opérateurs rendent possibles :

<div align="center">

| Opérateur | Nom | Description | Cas d'usage |
| :---: | --- | --- | --- |
| `+` | Concaténation | Combine les champs des deux modèles de données. Lève une exception si l'un des deux est `None`. | Fusionner les sorties de branches parallèles |
| `&` | Et logique | Concaténation sûre qui renvoie `None` si l'une des entrées est `None`. | Combiner avec des sorties de branches potentiellement nulles |
| `\|` | Ou logique | Renvoie le modèle de données non-`None`. Si les deux sont non-`None`, les fusionne. | Rassembler les sorties de branches conditionnelles |
| `^` | Ou exclusif logique | Renvoie les données si exactement une entrée est non-`None`, sinon `None`. | Sélection exclusive de branche |
| `~` | Non logique | Renvoie `None` si l'entrée est non-`None`, ou un modèle de données vide si `None`. | Inverser des conditions de branche |
| `in` | Contient | Vérifie si une clé de type chaîne existe dans les propriétés du schéma, ou si le schéma d'un autre modèle de données est contenu. Renvoie `True` ou `False`. | Vérification conditionnelle de champs, validation de schéma |

</div>

```python
# Branches parallèles avec concaténation
x1 = await generator1(inputs)
x2 = await generator2(inputs)
# combined = x1 *et* x2
combined = (
    x1 & x2
)  # Fusionne les deux sorties (ajoute un suffixe _{i} en cas de collision de clés)
# [...]
# Branches conditionnelles avec le ou logique
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
# result = easy *ou* hard
result = easy | hard  # Récupère la branche qui a été sélectionnée
```

## Obtenir un résumé de votre programme

Pour afficher un résumé tabulaire de votre programme :

```python
program.summary()
```

Ou un diagramme (utile pour documenter votre système) :

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

<div align="center">
<img src="../docs/assets/examples/datamodel_designer.png" alt="Programme de conception de modèles de données" width="600">

<em>Le programme de conception de modèles de données visualisé avec plot_program : Input → DeepAgent. Les modules entraînables sont indiqués en vert.</em>
</div>

## Exécuter votre programme

Pour exécuter votre programme, utilisez ce qui suit :

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

## Entraîner votre programme/agent

```python
# Définir les modèles de langage/d'embedding par défaut vous permet
# d'utiliser l'identifiant sous forme de chaîne (à la Keras) pour configurer votre pipeline/entraînement.
# Vous pouvez toujours instancier les classes si vous voulez un contrôle fin.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")
synalinks.set_default_embedding_model("gemini/text-embedding-004")


async def main():

    # ... la définition de votre programme

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

## Sauvegarde et chargement

Pour sauvegarder l'architecture complète et les variables (l'état du programme) dans un fichier JSON :

```python
program.save("my_program.json")
```

Pour le charger :

```python
loaded_program = synalinks.Program.load("my_program.json")
```

Pour sauvegarder uniquement l'état de votre programme (les variables) en JSON :

```python
program.save_variables("my_program.variables.json")
```

Pour charger ses variables (nécessite un programme avec la même architecture) :

```python
program.load_variables("my_program.variables.json")
```

## Journalisation

Pour activer la journalisation, utilisez ce qui suit au début de votre script :

```python
synalinks.enable_logging()
```

## Observabilité

Synalinks fournit une observabilité intégrée via MLflow pour tracer et surveiller vos programmes.

> **Important** : Appelez `enable_observability()` **avant** de créer le moindre module.

```python
import synalinks

# Activez d'abord l'observabilité
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # Optionnel : URI du serveur MLflow
    experiment_name="my_experiment",  # Optionnel : "synalinks_traces" par défaut
)

# Puis créez vos modules - ils seront automatiquement tracés
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

Pour les métriques d'entraînement et les artefacts, utilisez le callback `Monitor` :

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

Consultez le [guide Observabilité](https://synalinks.github.io/synalinks/guides/Observability/) pour la configuration avancée.

### En savoir plus

Vous pouvez en apprendre davantage en lisant notre [documentation](https://synalinks.github.io/synalinks/). Si vous avez des questions, la [FAQ](https://synalinks.github.io/synalinks/FAQ/) pourra vous aider.

### Contributions

Les contributions sont les bienvenues, que ce soit pour l'implémentation de modules, de métriques ou d'optimiseurs supplémentaires.
Pour plus d'informations, ou pour obtenir de l'aide dans l'implémentation de vos idées (ou de celles issues d'un article de recherche), rejoignez notre Discord.

Gardez à l'esprit que chaque métrique/module/optimiseur supplémentaire doit être approuvé par l'équipe cœur : nous voulons garder la bibliothèque aussi minimale et propre que possible pour éviter une croissance incontrôlée menant à de mauvaises pratiques logicielles, comme dans la plupart des frameworks LM dominants actuels.

Si vous avez des retours spécifiques ou des demandes de fonctionnalités, nous vous invitons à ouvrir une [issue](https://github.com/SynaLinks/synalinks/issues).

### Contributeurs

Ce sont vos contributions, vos retours et votre soutien qui font vivre ce projet.

Des petites corrections de bugs aux fonctionnalités majeures, merci de croire en la collaboration ouverte et en l'avenir de l'IA neuro-symbolique.

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### Communauté

Rejoignez notre communauté pour en apprendre davantage sur les systèmes neuro-symboliques et l'avenir de l'IA. Nous accueillons volontiers des personnes de tous horizons et de tous niveaux de formation.

### Citer notre travail

Ce travail a été réalisé sous la supervision de François Chollet, l'auteur de Keras. Si ce travail est utile à vos recherches, merci d'utiliser l'entrée bibtex suivante :

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### Remerciements

Synalinks ne serait pas possible sans le formidable travail des projets open source suivants :

- [Keras](https://keras.io/) pour le socle de calcul à base de graphes, l'API et, de manière générale, le code, la conception et la philosophie.
- [DSPy](https://dspy.ai/) pour l'inspiration des modules/optimiseurs.
- [Pydantic](https://docs.pydantic.dev/latest/) pour la couche de données du backend.
- [LiteLLM](https://docs.litellm.ai/docs/) pour les intégrations de LM.
- [DuckDB](https://duckdb.org/), [Ladybug](https://ladybugdb.com/), [LanceDB](https://www.lancedb.com/) pour leurs formidables bases de données embarquées.
- [MirageAI](https://www.strukto.ai/mirage) pour leur formidable sandbox !
