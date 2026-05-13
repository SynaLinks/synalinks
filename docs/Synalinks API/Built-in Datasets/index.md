# Built-in Datasets

The `synalinks.datasets` module provide a few datasets that can be used for debugging, evaluation or to create code examples.

These datasets are leaked in nowadays LMs training data, which is a big concern in todays ML community, so they won't give you much information about the reasoning abilities of the underlying models. But they are still useful as baseline to compare neuro-symbolic methods or when using small language models.

Every dataset module exposes the same four entrypoints:
`get_input_data_model()`, `get_output_data_model()`, `iterable_dataset(...)`,
and `load_data(...)` returning `(x_train, y_train), (x_test, y_test)`.

---

## Reasoning & math

- [GSM8K](GSM8K.md): 8.5K linguistically diverse grade-school math word problems. Useful to evaluate multi-step arithmetic reasoning.
- [DROP](DROP.md): Discrete Reasoning Over Paragraphs — passage + question pairs whose answers require arithmetic, counting, or sorting over the passage.
- [BBH](BBH.md): BIG-Bench Hard — challenging subset of BIG-Bench tasks (boolean expressions by default).
- [LogiQA](LogiQA.md): Multiple-choice logical-reasoning questions adapted from civil-service exams.
- [ARC-AGI](ARC-AGI.md): Tasks based on core knowledge principles to evaluate reasoning and program synthesis systems.
- [ARC-Challenge](ARC-Challenge.md): Grade-school science questions hard for retrieval and word-cooccurrence baselines.

## Question answering & retrieval

- [HotpotQA](HotpotQA.md): 113k Wikipedia-based question/answer pairs that need multiple documents to answer. Useful to evaluate Agentic RAGs with multi-hop retrieval.
- [SQuAD](SQuAD.md): Stanford Question Answering Dataset v1.1 — short-span extractive QA over Wikipedia passages.
- [BoolQ](BoolQ.md): Naturally occurring yes/no questions paired with a Wikipedia passage.
- [TruthfulQA](TruthfulQA.md): MC1 multiple-choice questions designed to test whether a model avoids common human misconceptions.

## Multiple-choice knowledge & commonsense

- [MMLU](MMLU.md): Massive Multitask Language Understanding — 57 subjects from elementary to professional level, four-way multiple choice.
- [HellaSwag](HellaSwag.md): Commonsense sentence completion — pick the most plausible continuation among four candidates.
- [WinoGrande](WinoGrande.md): Winograd-style pronoun-disambiguation sentences with a `_` blank and two numbered options.

## Bias, safety & instruction following

- [BBQ](BBQ.md): Bias Benchmark for QA — questions probing social biases across demographic categories (age, gender, race, religion, …).
- [IFEval](IFEval.md): Instruction-Following Eval — verifiable instructions (formatting, length, language, etc.) graded by rules or an LM judge.

## Reading comprehension & code

- [LAMBADA](LAMBADA.md): Final-word prediction over narrative passages — measures broad-context language modeling.
- [HumanEval](HumanEval.md): 164 Python programming problems with unit tests. Useful to evaluate code-generation systems.
