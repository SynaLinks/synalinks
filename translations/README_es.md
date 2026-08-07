<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/synalinks-dark.svg">
  <img height=200 alt="Synalinks" src="../img/synalinks-light.svg">
</picture>
</div>

<div align="center">

<b>De la idea a producción en solo unas pocas líneas</b>

<em>El primer framework neuro-simbólico para Modelos de Lenguaje (LM) que aprovecha la simplicidad de Keras y el rigor de las mejores prácticas del Deep Learning.</em>

<b>Construye [RAGs](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/), [agentes con herramientas](https://synalinks.github.io/synalinks/guides/Agents/), sistemas multiagente, [agentes recursivos](https://synalinks.github.io/synalinks/guides/Recursive%20Language%20Model%20Agent/) y mucho más en solo unas pocas líneas</b>

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
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>Documentación</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>Ejemplos de código</strong></a> .
  <a href="https://github.com/SynaLinks/synalinks/tree/main/guides" target="_blank"><strong>Guías</strong></a>
</p>

</div>

<div align="center">

Si Synalinks te resulta útil, ¡dale una estrella al repositorio! Ayúdanos a llegar a más ingenieros de AI/ML y a hacer crecer la comunidad.

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

¿Quieres usar Synalinks con tu propio agente de programación (Claude Code, Cursor, Copilot, etc.)? Añade a tu agente las skills específicas de Synalinks desde [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills) en GitHub: le enseñan las convenciones del framework y le dan el contexto necesario para construir programas Synalinks desde el primer momento.

</div>

## ¿Qué es Synalinks?

Synalinks es un framework neuro-simbólico de código abierto que simplifica la creación, el entrenamiento, la evaluación y el despliegue de aplicaciones avanzadas basadas en LM, incluyendo RAGs, agentes autónomos y sistemas de razonamiento auto-evolutivos.

Piensa en un Keras para aplicaciones de Modelos de Lenguaje: una API limpia y declarativa donde:

- **Compones** [`Module`s](https://synalinks.github.io/synalinks/guides/Modules/) igual que lo harías con las `Layer`s de deep learning.
- **[Entrenas y optimizas](https://synalinks.github.io/synalinks/guides/Training/)** con aprendizaje por refuerzo en contexto.
- **Despliegas** como [APIs REST](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) o [servidores MCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/).

### Principios clave

- **Complejidad progresiva**: [Empieza de forma sencilla y avanza de manera natural](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Aprendizaje neuro-simbólico**: Combina [lógica, estructura](https://synalinks.github.io/synalinks/guides/Data%20Models/) y [modelos de lenguaje](https://synalinks.github.io/synalinks/guides/Getting%20Started/).
- **Optimización en contexto**: [Mejora el razonamiento del modelo sin reentrenar los pesos](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/).

## ¿Para quién es?

<div align="center">

| Rol                      | Por qué Synalinks ayuda                                         |
| ------------------------- | ----------------------------------------------------------- |
| **Desarrolladores de IA**      | Construye aplicaciones LM complejas de nivel de producción sin código repetitivo. |
| **Investigadores de IA**     | Prototipa sistemas neuro-simbólicos y de RL en contexto rápidamente.    |
| **Científicos de datos**    | Integra flujos de trabajo con LM junto a APIs y bases de datos.               |
| **Estudiantes/Aficionados** | Aprende composición de IA con un framework limpio e intuitivo.       |

</div>

## ¿Por qué Synalinks?

Hoy existen muchos frameworks; esto es lo que Synalinks hace de manera diferente:

- **Sandbox embebido, sin contenedores**: los agentes ejecutan código y herramientas no confiables en un [entorno de ejecución seguro y aislado](https://synalinks.github.io/synalinks/guides/Agents/) que **no necesita Docker ni ningún servicio de sandbox externo**. Toda la pila es Python puro y embebible, por lo que resulta ideal para scripting, investigación, despliegue serverless/en la nube (S3, Lambda, notebooks, etc.) ¡o incluso para crear harnesses de CLI!
- **Soporte de bases de datos embebidas**: construye [RAG basado en grafos y memorias agénticas](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/) con **extracción restringida de Grafos de Conocimiento** y **deduplicación semántica automática**, sobre una base de datos de grafos embebida, sin necesidad de ejecutar un servidor de grafos aparte. Además, hay disponible una **base de conocimiento SQL** embebida y rápida para almacenar datos relacionales y construir RAGs vectoriales/SQL.
- **RL en contexto para optimizar tus prompts (y cualquier otra cosa)**: [entrena y optimiza](https://synalinks.github.io/synalinks/guides/Training/) prompts, ejemplos few-shot y [cualquier variable entrenable](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/) por módulo **sin tocar los pesos del modelo**, usando la conocida API `.compile()` / `.fit()` / `.evaluate()` / `.predict()`.
- **Cambio de modelo sin esfuerzo**: define un valor por defecto una sola vez con `synalinks.set_default_language_model(...)` o pasa un identificador de cadena, y alterna entre Ollama, vLLM, OpenAI, Azure, Anthropic, Mistral, Groq, Gemini, xAI, Cohere, DeepSeek, Together AI, OpenRouter, AWS Bedrock y Doubleword mediante [LiteLLM](https://docs.litellm.ai/docs/), incluyendo [selección multiobjetivo de modelos](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/) para elegir el mejor modelo según coste/calidad.
- **Andamiaje en un solo comando, trae tu propio agente de programación**: arranca un proyecto listo para producción con `synalinks init` (plantillas con todo incluido para scripts, agentes y entrenamiento), y luego añade las [skills oficiales de Synalinks](https://github.com/SynaLinks/synalinks-skills) para que Claude Code, Cursor, Copilot y compañía escriban código Synalinks idiomático desde el principio.

Además de todo lo que esperarías de un framework de nivel de producción:

- **NUEVO**: Ahora todos los agentes soportan [Agent Skills](https://agentskills.io/home), `AGENTS.md` y subagentes.
- **[Salidas estructuradas restringidas](https://synalinks.github.io/synalinks/guides/Data%20Models/)** (JSON) para garantizar la corrección
- **API de mensajes compatible con Chat Completions**: los mensajes replican clave por clave el formato Chat Completions de OpenAI + extensiones de litellm con `reasoning_content` y `thinking_blocks`, de modo que el razonamiento del proveedor sobrevive a los intercambios multi-turno. También maneja **[entradas multimodales](https://synalinks.github.io/synalinks/guides/Multimodal%20Inputs/)** (imágenes y audio como partes de contenido estándar)
- **[Pipelines](https://synalinks.github.io/synalinks/guides/Programs/) versionables** y serializables en JSON
- **[Ejecución asíncrona y paralela](https://synalinks.github.io/synalinks/guides/Programs/) automática** por defecto
- **[Métricas](https://synalinks.github.io/synalinks/guides/Metrics/), [recompensas](https://synalinks.github.io/synalinks/guides/Rewards/) y [datasets](https://synalinks.github.io/synalinks/guides/Datasets/)** integrados
- **Listo para APIs**: Despliega con [FastAPI](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) o [FastMCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)
- **[Compatibilidad con KerasTuner](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/)** para la búsqueda de hiperparámetros
- **[Callbacks](https://synalinks.github.io/synalinks/guides/Callbacks/) y hooks integrados** para [observabilidad](https://synalinks.github.io/synalinks/guides/Observability/) (incluyendo un callback `Monitor` de MLflow)

# Requisitos

- Python 3.12 o superior
- WSL2 para usuarios de Windows

## Inicio rápido en 3 segundos con `uv` (recomendado)

Si no conoces `uv`, instálalo [aquí](https://docs.astral.sh/uv/getting-started/installation/).

Sigue las instrucciones para iniciar un nuevo proyecto synalinks en 3 segundos:

```shell
uvx synalinks init
```

---

También puedes instalar la biblioteca en un proyecto nuevo con:

```shell
uv add synalinks
```

Para transformar tu agente de programación en un ingeniero de IA, ejecuta esto en la raíz de tu proyecto:

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

## Ejemplo

Los agentes de Synalinks ahora pueden leer las convenciones de tu proyecto en
[`AGENTS.md`](https://agents.md) y usar [Agent Skills](https://agentskills.io/home).
El siguiente ejemplo conecta las [skills oficiales de Synalinks](https://github.com/SynaLinks/synalinks-skills)
a un [`DeepAgent`](https://synalinks.github.io/synalinks/guides/Agents/), un
agente de programación en sandbox, y le pide que diseñe los modelos de datos de
entrada/salida para una tarea, escribiéndolos en una carpeta `workspace/`.

Primero prepara el workspace. Instala la skill oficial de Synalinks con la
CLI [`skills`](https://skills.sh) y añade un `AGENTS.md`. La skill se instala bajo
el directorio de trabajo, de modo que el agente en sandbox puede leer su contenido bajo demanda:

```shell
mkdir -p workspace && cd workspace
# Instala la skill `synalinks` en ./.agents/skills/ y escribe skills-lock.json.
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

Esto produce la estructura siguiente; `.agents/skills` es la *raíz* de skills (una
subcarpeta por skill, cada una con un `SKILL.md`):

```text
workspace/
├── AGENTS.md                     # inyectado como las convenciones del agente
├── skills-lock.json              # fija la skill a un repo de origen + hash de contenido
└── .agents/
    └── skills/                   # la raíz de skills
        └── synalinks/
            └── SKILL.md          # se exponen nombre + descripción; el cuerpo se lee bajo demanda
```

`main.py`:

```python
import synalinks
import asyncio

# Define el valor por defecto una vez; los módulos lo detectan automáticamente.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")


# La respuesta final estructurada del agente.
class Deliverable(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="What was created and where",
    )
    files: list[str] = synalinks.Field(
        description="Paths of the files written into the workspace",
    )


async def main():
    # Un DeepAgent conversa mediante ChatMessages (es un agente de programación).
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    agent = synalinks.DeepAgent(
        data_model=Deliverable,
        # El sandbox se inicializa desde este directorio (seguro para el host: las
        # escrituras del agente van a la copia del sandbox, nunca a tu disco). Su `AGENTS.md` se
        # inyecta para que el agente siga tus convenciones.
        workdir="workspace",
        # La raíz de skills (instalada por `skills add`). Se lista al agente como
        # `<available_skills>`; lee cada `SKILL.md` bajo demanda desde el
        # sandbox; por eso las skills viven bajo `workdir`.
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

## Operadores de modelos de datos

Synalinks proporciona operadores de Python para combinar y manipular modelos de datos, habilitando un control de flujo sofisticado. Consulta la [guía de Control de Flujo](https://synalinks.github.io/synalinks/guides/Control%20Flow/) para conocer los patrones de enrutamiento, fan-out y fusión que estos operadores hacen posibles:

<div align="center">

| Operador | Nombre | Descripción | Caso de uso |
| :---: | --- | --- | --- |
| `+` | Concatenación | Combina los campos de ambos modelos de datos. Lanza una excepción si alguno es `None`. | Fusionar salidas de ramas paralelas |
| `&` | And lógico | Concatenación segura que devuelve `None` si alguna de las entradas es `None`. | Combinar con salidas de ramas potencialmente nulas |
| `\|` | Or lógico | Devuelve el modelo de datos que no es `None`. Si ambos son no-`None`, los fusiona. | Recoger salidas de ramas condicionales |
| `^` | Xor lógico | Devuelve los datos si exactamente una entrada es no-`None`; en caso contrario, `None`. | Selección exclusiva de ramas |
| `~` | Not lógico | Devuelve `None` si la entrada es no-`None`, o un modelo de datos vacío si es `None`. | Invertir condiciones de ramas |
| `in` | Contains | Comprueba si una clave de tipo cadena existe en las propiedades del esquema, o si el esquema de otro modelo de datos está contenido. Devuelve `True` o `False`. | Comprobación condicional de campos, validación de esquemas |

</div>

```python
# Ramas paralelas con concatenación
x1 = await generator1(inputs)
x2 = await generator2(inputs)
# combined = x1 *and* x2
combined = (
    x1 & x2
)  # Fusiona ambas salidas (añade el sufijo _{i} si hay colisión de claves)
# [...]
# Ramas condicionales con or lógico
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
# result = easy *or* hard
result = easy | hard  # Obtiene la rama que haya sido seleccionada
```

## Obtener un resumen de tu programa

Para imprimir un resumen tabular de tu programa:

```python
program.summary()
```

O un gráfico (útil para documentar tu sistema):

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

<div align="center">
<img src="../docs/assets/examples/datamodel_designer.png" alt="Programa de diseño de modelos de datos" width="600">

<em>El programa de diseño de modelos de datos visualizado con plot_program: Input → DeepAgent. Los módulos entrenables se marcan en verde.</em>
</div>

## Ejecutar tu programa

Para ejecutar tu programa, usa lo siguiente:

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

## Entrenar tu programa/agente

```python
# Definir los modelos de lenguaje/embedding por defecto te permite
# usar el identificador de cadena (al estilo Keras) para configurar tu pipeline/entrenamiento.
# Aún puedes instanciar las clases si quieres un control más fino.
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")
synalinks.set_default_embedding_model("gemini/text-embedding-004")


async def main():

    # ... la definición de tu programa

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

## Guardar y cargar

Para guardar la arquitectura completa y las variables (el estado del programa) en un archivo JSON, haz:

```python
program.save("my_program.json")
```

Para cargarlo, haz:

```python
loaded_program = synalinks.Program.load("my_program.json")
```

Para guardar solo el estado de tu programa (las variables) en JSON:

```python
program.save_variables("my_program.variables.json")
```

Para cargar sus variables (requiere un programa con la misma arquitectura), haz:

```python
program.load_variables("my_program.variables.json")
```

## Logging

Para habilitar el logging, usa lo siguiente al principio de tu script:

```python
synalinks.enable_logging()
```

## Observabilidad

Synalinks proporciona observabilidad integrada a través de MLflow para trazar y monitorizar tus programas.

> **Importante**: Llama a `enable_observability()` **antes** de crear cualquier módulo.

```python
import synalinks

# Habilita la observabilidad primero
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # Opcional: URI del servidor MLflow
    experiment_name="my_experiment",  # Opcional: por defecto es "synalinks_traces"
)

# Después crea tus módulos - se trazarán automáticamente
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

Para métricas de entrenamiento y artefactos, usa el callback `Monitor`:

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

Consulta la [guía de Observabilidad](https://synalinks.github.io/synalinks/guides/Observability/) para la configuración avanzada.

### Aprende más

Puedes aprender más leyendo nuestra [documentación](https://synalinks.github.io/synalinks/). Si tienes preguntas, el [FAQ](https://synalinks.github.io/synalinks/FAQ/) puede ayudarte.

### Contribuciones

Las contribuciones son bienvenidas, ya sea para la implementación de módulos, métricas u optimizadores adicionales.
Para más información, o ayuda para implementar tus ideas (o las de algún artículo), únete a nuestro Discord.

Ten en cuenta que cada métrica/módulo/optimizador adicional debe ser aprobado por el equipo central; queremos mantener la biblioteca lo más mínima y limpia posible para evitar un crecimiento descontrolado que conduzca a malas prácticas de software, como ocurre en la mayoría de los frameworks de LM líderes actuales.

Si tienes comentarios específicos o solicitudes de funcionalidades, te invitamos a abrir un [issue](https://github.com/SynaLinks/synalinks/issues).

### Contribuidores

Tus contribuciones, comentarios y apoyo son lo que hace prosperar este proyecto.

Desde pequeñas correcciones de bugs hasta grandes funcionalidades, gracias por creer en la colaboración abierta y en el futuro de la IA neuro-simbólica.

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### Comunidad

Únete a nuestra comunidad para aprender más sobre los sistemas neuro-simbólicos y el futuro de la IA. Damos la bienvenida a la participación de personas con trayectorias y niveles de formación muy diversos.

### Citar nuestro trabajo

Este trabajo se ha realizado bajo la supervisión de François Chollet, el autor de Keras. Si este trabajo es útil para tu investigación, utiliza la siguiente entrada de bibtex:

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### Créditos

Synalinks no sería posible sin el gran trabajo de los siguientes proyectos de código abierto:

- [Keras](https://keras.io/) por la columna vertebral de computación basada en grafos, la API y, en general, el código, el diseño y la filosofía.
- [DSPy](https://dspy.ai/) por la inspiración en módulos/optimizadores.
- [Pydantic](https://docs.pydantic.dev/latest/) por la capa de datos del backend.
- [LiteLLM](https://docs.litellm.ai/docs/) por las integraciones de LMs.
- [DuckDB](https://duckdb.org/), [Ladybug](https://ladybugdb.com/), [LanceDB](https://www.lancedb.com/) por sus increíbles bases de datos embebidas.
- [MirageAI](https://www.strukto.ai/mirage) ¡por su increíble sandbox!
