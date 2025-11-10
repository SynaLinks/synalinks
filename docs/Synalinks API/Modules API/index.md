# Modules API

Modules are the basic building blocks of programs in Synalinks. A `Module` consists of data model-in & data model-out computation function (the module's `call()` method) and some state (held in `Variable`).

A module instance is a callable, much like a function:

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )

async def main():
    language_model = LanguageModel(
        model="ollama/deepseek-r1"
    )

    generator = synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )

    inputs = Query(query="What is the capital of France?")
    outputs = await generator(inputs)


if __name__ == "__main__":
    asyncio.run(main())
```

## Modules API overview

- [Base Module class](Base Module class.md)

---

### Core Modules

- [Input module](Core Modules/Input module.md)
- [Identity module](Core Modules/Identity module.md)
- [Not module](Core Modules/Not module.md)
- [Generator module](Core Modules/Generator module.md)
- [Decision module](Core Modules/Decision module.md)
- [Action module](Core Modules/Action module.md)
- [Branch module](Core Modules/Branch module.md)

---

### Merging Modules

- [Concat module](Merging Modules/Concat module.md)
- [And module](Merging Modules/And module.md)
- [Or module](Merging Modules/Or module.md)
- [Xor module](Merging Modules/Xor module.md)

---

### Test Time Compute Modules

- [ChainOfThought module](Test Time Compute Modules/ChainOfThought module.md)
- [SelfCritique module](Test Time Compute Modules/SelfCritique module.md)

---

### Knowledge Modules

- [Embedding module](Knowledge Modules/Embedding module.md)
- [UpdateKnowledge module](Knowledge Modules/UpdateKnowledge module.md)

---

### Retrievers Modules

- [EntityRetriever module](Retrievers Modules/EntityRetriever module.md)
- [TripletRetriever module](Retrievers Modules/TripletRetriever module.md)

---

### Synthesis Modules

- [PythonSynthesis module](Synthesis Modules/PythonSynthesis module.md)
- [SequentialPlanSynthesis module](Synthesis Modules/SequentialPlanSynthesis module.md)

---

### Agents Modules

- [FunctionCallingAgent module](Agents Modules/FunctionCallingAgent module.md)