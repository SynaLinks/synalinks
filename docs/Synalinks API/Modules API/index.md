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
- [Tool module](Core Modules/Tool module.md)
- [Lambda module](Core Modules/Lambda module.md)

---

### Agents Modules

- [FunctionCallingAgent module](Agents Modules/FunctionCallingAgent module.md)
- [SQLAgent module](Agents Modules/SQLAgent module.md)
- [CypherAgent module](Agents Modules/CypherAgent module.md)
- [VectorRAGAgent module](Agents Modules/VectorRAGAgent module.md)
- [DeepAgent module](Agents Modules/DeepAgent module.md)
- [RLMAgent module](Agents Modules/RLMAgent module.md)

---

### Masking Modules

- [InMask module](Masking Modules/InMask module.md)
- [OutMask module](Masking Modules/OutMask module.md)

---

### Merging Modules

- [Concat module](Merging Modules/Concat module.md)
- [And module](Merging Modules/And module.md)
- [Or module](Merging Modules/Or module.md)
- [Xor module](Merging Modules/Xor module.md)

---

### Retrievers Modules

- [SimilaritySearch module](Retrievers Modules/SimilaritySearch module.md)
- [FullTextSearch module](Retrievers Modules/FullTextSearch module.md)
- [RegexSearch module](Retrievers Modules/RegexSearch module.md)
- [HybridFTSSearch module](Retrievers Modules/HybridFTSSearch module.md)
- [HybridRegexSearch module](Retrievers Modules/HybridRegexSearch module.md)
- [EntitySimilaritySearch module](Retrievers Modules/EntitySimilaritySearch module.md)
- [EntityFullTextSearch module](Retrievers Modules/EntityFullTextSearch module.md)
- [EntityRegexSearch module](Retrievers Modules/EntityRegexSearch module.md)
- [EntityHybridFTSSearch module](Retrievers Modules/EntityHybridFTSSearch module.md)
- [EntityHybridRegexSearch module](Retrievers Modules/EntityHybridRegexSearch module.md)
- [RelationSimilaritySearch module](Retrievers Modules/RelationSimilaritySearch module.md)
- [RelationFullTextSearch module](Retrievers Modules/RelationFullTextSearch module.md)
- [RelationRegexSearch module](Retrievers Modules/RelationRegexSearch module.md)
- [RelationHybridFTSSearch module](Retrievers Modules/RelationHybridFTSSearch module.md)
- [RelationHybridRegexSearch module](Retrievers Modules/RelationHybridRegexSearch module.md)
- [PathSimilaritySearch module](Retrievers Modules/PathSimilaritySearch module.md)
- [PathFullTextSearch module](Retrievers Modules/PathFullTextSearch module.md)
- [PathRegexSearch module](Retrievers Modules/PathRegexSearch module.md)
- [PathHybridFTSSearch module](Retrievers Modules/PathHybridFTSSearch module.md)
- [PathHybridRegexSearch module](Retrievers Modules/PathHybridRegexSearch module.md)

---

### Rerankers Modules

- [RRFReranker module](Rerankers Modules/RRFReranker module.md)

---

### Test Time Compute Modules

- [ChainOfThought module](Test Time Compute Modules/ChainOfThought module.md)
- [SelfCritique module](Test Time Compute Modules/SelfCritique module.md)

---

### Knowledge Modules

- [EmbedKnowledge module](Knowledge Modules/EmbedKnowledge module.md)
- [UpdateKnowledge module](Knowledge Modules/UpdateKnowledge module.md)
- [RetrieveKnowledge module](Knowledge Modules/RetrieveKnowledge module.md)

---

### Synthesis Modules

- [PythonSynthesis module](Synthesis Modules/PythonSynthesis module.md)
- [SequentialPlanSynthesis module](Synthesis Modules/SequentialPlanSynthesis module.md)