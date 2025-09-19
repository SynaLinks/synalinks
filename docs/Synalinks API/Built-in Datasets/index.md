# Built-in Datasets

The `synalinks.datasets` module provide a few datasets that can be used for debugging, evaluation or to create code examples.

These datasets are leaked in nowadays LMs training data, which is a big concern in todays ML community, so they won't give you much information about the reasoning abilities of the underlying models. But they are still useful as baseline to compare neuro-symbolic methods or when using small language models.

---

- [GSM8K dataset](GSM8K.md): A dataset of 8.5K high quality linguistically diverse grade school math word problems. Useful to evaluate reasoning capabilities.

- [HotpotQA](HotpotQA.md): A dataset of 113k wikipedia-based question/answer pairs that need multiple documents to answer. This dataset is useful to evaluate Agentic RAGs or KnowledgeGraph RAGs with multi-hop.

- [ARC-AGI](ARC-AGI.md): A challenging dataset containing tasks based on core knowledge principles to evaluate reasoning and program synthesis systems. 