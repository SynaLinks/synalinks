# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Datasets

Up to this point we have hand-built every input the program sees —
typing `Query(question="...")` in a script, or pasting examples
into a `numpy` array. That works for a tutorial. It does not work
when your training set is ten thousand rows, your data lives on
disk or on the Hugging Face Hub, and your validation split is
another five thousand rows you do not want to load into memory all
at once.

Synalinks' answer is the `Dataset` class — a small but principled
streaming interface that hands batches of `(x, y)` pairs to the
trainer one at a time, in exactly the shape `program.fit(...)`
expects. By the end of this guide you will have:

- loaded a public Hugging Face dataset into a Synalinks program
  without any ad-hoc parsing,
- understood how **Jinja2 templates** convert raw rows into your
  `DataModel` types,
- chosen between **streaming** and **materialized** loading and
  known when each one is right,
- learned the convenience helpers (`load_split`,
  `split_train_test`) and the catalog of **built-in datasets**
  Synalinks ships out of the box.

## The Picture: What the Trainer Actually Wants

Before we look at `HuggingFaceDataset`, it helps to understand
what `program.fit(...)` consumes. The trainer takes a Python
generator that yields one **batch** at a time. A batch is either:

- a one-tuple `(x,)` for inference-only use, or
- a two-tuple `(x, y)` for training.

Both `x` and `y` are **NumPy object arrays** whose elements are
`DataModel` instances. In [Guide 14](https://synalinks.github.io/synalinks/guides/Training/) (Training) we will build such
arrays by hand — e.g.
`np.array([Question(...), Question(...)], dtype="object")` — to
keep the training example self-contained. A `Dataset` produces
arrays of exactly the same shape, just one batch at a time and
from a real source instead of a Python literal.

```mermaid
flowchart LR
    SRC["raw rows<br/>(HF / CSV / API / ...)"] --> T["Jinja2 template renders<br/>each row to JSON"]
    T --> V["Pydantic validates<br/>JSON → DataModel"]
    V --> B["Buffered into batches"]
    B --> P["program.fit(x=dataset())"]
```

A `Dataset` subclass plugs in at the leftmost box (where do the
rows come from?). The rest — templating, validation, batching — is
inherited from the `synalinks.Dataset` base class and is the same
across every source.

## Loading a Hugging Face Dataset

The Hugging Face Hub is the largest public catalog of text
datasets — millions of rows on tens of thousands of tasks, all
accessible through one library. Synalinks wraps it in
`synalinks.HuggingFaceDataset`. You give the wrapper a path on the
Hub, two Jinja2 templates that explain how to convert one HF row
into your input/output `DataModel`s, and a batch size; it yields
fit-ready batches.

Here is the minimal recipe, against the `gsm8k` math-word-problem
dataset:

```python
import synalinks


class MathQuestion(synalinks.DataModel):
    question: str = synalinks.Field(description="The math word problem")


class NumericalAnswer(synalinks.DataModel):
    answer: float = synalinks.Field(description="The numerical answer")


ds = synalinks.HuggingFaceDataset(
    path="gsm8k",
    name="main",
    split="train",
    input_data_model=MathQuestion,
    input_template='{"question": {{ question | tojson }}}',
    output_data_model=NumericalAnswer,
    output_template=(
        '{"answer": {{ answer.split("####")[-1].strip().replace(",", "")'
        " | float }}}"
    ),
    batch_size=8,
)

# `ds()` returns a fresh generator each time the trainer asks for one.
# program.fit(x=ds(), ...)
```

Walking through the arguments:

- **`path`** is the dataset's repo name on the Hub (the
  first positional argument of `datasets.load_dataset`).
  `"gsm8k"` here; for a community dataset you would use the
  full `"owner/name"` form, as in `"dair-ai/emotion"`
  ([Guide 17](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/)).
- **`name`** is the **configuration name** when a dataset
  ships several variants. `gsm8k` has a configuration called
  `"main"` and one called `"socratic"`; we pick `"main"`.
- **`split`** is the slice of the dataset you want —
  typically `"train"`, `"validation"`, or `"test"`. Passing
  `None` iterates every split in order.
- **`input_data_model`** + **`input_template`** describe the
  `x` side of each batch.
- **`output_data_model`** + **`output_template`** describe
  the `y` side. Omit both for an inputs-only dataset (useful
  at inference time).
- **`batch_size`** is the number of examples accumulated
  before yielding one batch. Default `1` — bump it up to give
  the trainer larger batches and the optimizer better
  statistics.

The two `template` arguments are doing the heavy lifting. Let's
look at them more closely.

## Templates: From Raw Row to DataModel

A Hugging Face row arrives as a plain Python dict whose keys are
whatever the dataset chose to name them. For `gsm8k` those keys
are `question` (a string) and `answer` (a string in the awkward
form `"<chain of thought>\\n#### 42"`). Your `DataModel` does not
care about that shape — it just wants typed Python fields. The
two templates convert from one to the other.

**Jinja2** is the standard Python templating language; the
double-curly-brace syntax `{{ ... }}` evaluates an expression
against the row's keys and substitutes its value. Each template
should render to **valid JSON** that matches the corresponding
`DataModel` schema, because under the hood Synalinks runs
`DataModel.model_validate_json(rendered_string)`.

Two filters you will use over and over:

- **`| tojson`** — the Jinja2 filter that quotes and escapes a
  Python value into a JSON literal. Always use it around any
  string field. Skipping `tojson` is the templating equivalent
  of forgetting to parameterize a SQL query ([Guide 6](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/)) — quotes,
  backslashes, and Unicode in the source row will quietly break
  your output.
- **`| float`** (and `| int`, etc.) — coerce the value to a
  number before it lands in JSON, so Pydantic can validate a
  numeric field.

A complete gsm8k input template:

```jinja
{"question": {{ question | tojson }}}
```

The output template is more interesting because `gsm8k` encodes
its answer as text:

```python
# raw answer field, e.g.:
# "John has 5 apples. He gives 2 away.\\n#### 3"
```

We split on `"####"`, take the last piece, strip whitespace,
remove thousands-separator commas, and coerce to `float`:

```jinja
{"answer": {{ answer.split("####")[-1].strip().replace(",", "") | float }}}
```

The whole rendered string then validates cleanly against
`NumericalAnswer`. No bespoke parser, no try/except, no
post-processing in your training code.

## Streaming vs Materialized

`HuggingFaceDataset` accepts a `streaming=` flag (default
`True`). The two modes have very different trade-offs:

- **Streaming** (`streaming=True`). Rows are downloaded on
  demand from the Hub. The generator naturally terminates when
  the source is exhausted. **Required** when the dataset does
  not fit on disk (e.g. `c4`, `RedPajama`). Length is unknown
  ahead of time, so `len(ds)` raises — unless you also pass
  `limit=N`, in which case the size is capped and known.
- **Materialized** (`streaming=False`). The entire split is
  downloaded once, then iterated locally. Use it for small
  benchmark datasets where you want fast random access, reliable
  `len`, and reproducibility across runs.

For a 24-row evaluation slice, materialized is usually fine. For
a 1 M-row pretraining shard, streaming is the only option that
won't fill your disk. When in doubt, start with streaming plus
`limit=` and only switch to materialized if you measure a real
benefit.

## Three Convenience Helpers

Three pieces of API turn the common streaming-to-arrays patterns
into one-liners. The first lives on every `Dataset`, the other
two are module-level functions.

### `ds.materialize()` — stream → in-memory arrays

For evaluation or a small experiment, you usually want the
*whole* dataset sitting in memory as one NumPy object array, not
a stream of batches. The `Dataset.materialize()` method does
exactly that: iterate to exhaustion, concatenate every batch,
return a single `(x, y)` pair.

```python
ds = synalinks.HuggingFaceDataset(
    path="gsm8k",
    name="main",
    split="test",
    streaming=False,
    input_data_model=MathQuestion,
    input_template='{"question": {{ question | tojson }}}',
    output_data_model=NumericalAnswer,
    output_template=(
        '{"answer": {{ answer.split("####")[-1].strip().replace(",", "")'
        " | float }}}"
    ),
    limit=200,
)

x, y = ds.materialize()
# x and y are now NumPy object arrays you can hand straight to
# program.evaluate(x=x, y=y).
```

`materialize()` works for any `Dataset` subclass — `HuggingFaceDataset`,
your own CSV loader, anything — because it is defined on the base
class. Use it for small benchmark datasets that fit in memory;
for huge sources, iterate via `ds()` instead so rows stream on
demand.

### `synalinks.datasets.load_split` — one HF split → one `(x, y)`

When the source is Hugging Face, the construct-and-materialize
pattern above is so common that Synalinks ships a one-line
convenience around it:

```python
x, y = synalinks.datasets.load_split(
    path="gsm8k",
    name="main",
    split="test",
    input_data_model=MathQuestion,
    input_template='{"question": {{ question | tojson }}}',
    output_data_model=NumericalAnswer,
    output_template=(
        '{"answer": {{ answer.split("####")[-1].strip().replace(",", "")'
        " | float }}}"
    ),
    limit=200,
)
```

This is exactly equivalent to constructing the dataset with
`streaming=False` and calling `materialize()` on it; under the
hood, that is precisely what `load_split` does.

### `synalinks.datasets.split_train_test` — head/tail split

Some benchmark datasets ship a *single* labeled split (HumanEval,
IFEval, BBH, BBQ, TruthfulQA, ...). When you need a train/eval
cut from one such split, a deterministic head/tail slice is the
standard recipe — the same convention Keras uses with
`validation_split=`:

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.split_train_test(
    x, y, validation_split=0.2,
)
```

There is no shuffling here; the trade-off is *reproducibility*
across runs in exchange for the risk that the head/tail order is
biased. If the source dataset is already shuffled (most HF
benchmark splits are), head/tail is fine; if not, shuffle `x`
and `y` together with a fixed seed before calling this helper.

## Built-in Datasets: a Pre-built Catalog

For the standard LM-evaluation benchmarks, Synalinks ships
ready-made loaders under `synalinks.datasets.*` so you do not
have to write the templates yourself. Each one wraps
`HuggingFaceDataset` and exposes a `load_data()` function plus
`get_input_data_model()` / `get_output_data_model()` helpers.

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

print(x_train.shape, y_train.shape)
# (7473,) (7473,)   — NumPy object arrays of DataModels
```

The catalog at the time of writing — most of these are the
canonical reasoning/QA benchmarks you will see in the LM
literature:

- `synalinks.datasets.gsm8k` — grade-school math word problems
- `synalinks.datasets.hotpotqa` — multi-hop question answering
- `synalinks.datasets.squad` — reading-comprehension QA
- `synalinks.datasets.mmlu` — multitask multiple-choice
- `synalinks.datasets.bbh` — Big-Bench-Hard
- `synalinks.datasets.hellaswag` — commonsense completion
- `synalinks.datasets.humaneval` — code-generation
- `synalinks.datasets.ifeval` — instruction-following
- `synalinks.datasets.truthfulqa`, `synalinks.datasets.bbq`,
  `synalinks.datasets.arc_challenge`,
  `synalinks.datasets.arcagi`, `synalinks.datasets.boolq`,
  `synalinks.datasets.drop`, `synalinks.datasets.lambada`,
  `synalinks.datasets.logiqa`, `synalinks.datasets.winogrande`

If your task is on this list, prefer the built-in loader. If it
is not, write a `HuggingFaceDataset` directly — the same
machinery, just with templates you choose.

## Other Knobs Worth Knowing

A few `HuggingFaceDataset` arguments you may need later:

- **`limit=N`** — cap how many *raw* rows are consumed across
  all splits. Useful for smoke tests; also makes `len(ds)`
  available on streaming datasets.
- **`repeat=K`** — emit each raw example `K` times in a row.
  Setting `repeat == batch_size` produces "group of K
  rollouts of the same prompt" batches, which is the layout
  GRPO-style RL training expects.
- **`revision=...`** — pin to a specific dataset commit or
  branch. Important for reproducibility on a moving Hub.
- **`**kwargs`** are forwarded straight to
  `datasets.load_dataset`, so anything that library accepts
  (`data_files`, `token`, `trust_remote_code`, ...) works
  here too.

## Custom Sources: Subclassing `Dataset`

When the data does not live on the Hub — your own SQLite
database, a CSV file, an internal API — subclass
`synalinks.Dataset` and implement one method, `_iter_rows()`,
which yields raw row dicts. The base class handles templates,
validation, batching, and the `repeat` / `limit` knobs. A
sketch:

```python
class CsvDataset(synalinks.Dataset):
    def __init__(self, csv_path, **kwargs):
        super().__init__(**kwargs)
        self.csv_path = csv_path

    def _iter_rows(self):
        import csv
        with open(self.csv_path) as f:
            for row in csv.DictReader(f):
                yield row    # dict keyed by column name

ds = CsvDataset(
    csv_path="data.csv",
    input_data_model=Question,
    input_template='{"question": {{ question | tojson }}}',
    output_data_model=Answer,
    output_template='{"answer": {{ answer | tojson }}}',
    batch_size=8,
)
```

`HuggingFaceDataset` is itself just such a subclass — its
`_iter_rows()` is a tiny wrapper around the HF library.

## Failure Modes Worth Watching For

- **Template rendering errors.** Synalinks uses
  `jinja2.StrictUndefined`, so a typo in a template variable
  raises a clean `UndefinedError` rather than producing
  silently-wrong JSON. Read the error; the missing variable
  name is in it.
- **Schema validation errors.** If your template renders JSON
  that does not match the declared `DataModel`, Pydantic raises
  on the first bad row. Print one rendered string before
  starting a long run to confirm shapes match.
- **`tojson` omissions.** A field that contains quotes,
  backslashes, or newlines will tear your output JSON apart if
  you skip `tojson`. The error usually surfaces as a
  `JSONDecodeError`. Always wrap string fields in `| tojson`.
- **Forgetting that `ds()` returns a fresh generator.** Pass
  `x=ds()` to `fit()`, not `x=ds`. A `Dataset` instance is the
  configuration; calling it produces the iterator the trainer
  consumes.

## Take-Home Summary

- **A `Dataset` feeds batches of `(x, y)` `DataModel` arrays
  to the trainer.** The shape is identical to the
  hand-built NumPy arrays you have used so far; the dataset
  just produces them lazily.
- **`HuggingFaceDataset`** is the standard wrapper for the
  Hugging Face Hub. You give it a path, two Jinja2 templates,
  and a batch size; it yields fit-ready batches.
- **Templates render raw rows to JSON that validates against
  your `DataModel`.** Use `| tojson` for safe string escaping
  and `| float` / `| int` for type coercion.
- **Streaming vs materialized:** streaming downloads on demand
  and is required for huge datasets; materialized loads the
  whole split once and gives reliable `len`. When in doubt,
  start streaming with `limit=`.
- **Three convenience helpers** wrap the common patterns:
  `Dataset.materialize()` (any source → in-memory arrays),
  `synalinks.datasets.load_split` (one HF split in one call),
  and `synalinks.datasets.split_train_test` (deterministic
  head/tail split).
- **The built-in catalog (`synalinks.datasets.gsm8k`, ...)**
  ships pre-templated loaders for the standard LM benchmarks.
  Prefer it when your task is on the list.
- Pass **`x=ds()`** (calling the dataset to get a fresh
  generator), not `x=ds` (the configuration object).

## API References

- [synalinks.Dataset](https://synalinks.github.io/synalinks/Synalinks%20API/Datasets/Dataset/)
- [synalinks.HuggingFaceDataset](https://synalinks.github.io/synalinks/Synalinks%20API/Datasets/HuggingFaceDataset/)
- [Built-in datasets](https://synalinks.github.io/synalinks/Synalinks%20API/Built-in%20Datasets/)
- [Hugging Face `datasets` library](https://huggingface.co/docs/datasets/)
- [Jinja2 template language](https://jinja.palletsprojects.com/en/stable/templates/)
"""

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Data Models — match the shape of `gsm8k` rows
# =============================================================================


class MathQuestion(synalinks.DataModel):
    """A grade-school math word problem."""

    question: str = synalinks.Field(description="The math word problem")


class NumericalAnswer(synalinks.DataModel):
    """A single numerical answer."""

    answer: float = synalinks.Field(description="The numerical answer")


# =============================================================================
# Templates — convert one raw gsm8k row into JSON matching the data models
# =============================================================================

# Input is simple: just quote/escape the question string.
INPUT_TEMPLATE = '{"question": {{ question | tojson }}}'

# Output needs work: gsm8k stores answers as
# "<reasoning>\n#### <numeric>", possibly with thousands-commas.
# Split, strip, drop the commas, coerce to float.
OUTPUT_TEMPLATE = (
    '{"answer": {{ answer.split("####")[-1].strip().replace(",", "") | float }}}'
)


# =============================================================================
# Main Demonstration
# =============================================================================


def main():
    load_dotenv()

    # -------------------------------------------------------------------------
    # 1) Direct use of `HuggingFaceDataset`. Pull 16 training rows in
    #    a streaming fashion (cheap; no full download), batch them into
    #    groups of 4, and inspect the first batch.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("1) HuggingFaceDataset: stream 16 rows of gsm8k")
    print("=" * 60)

    ds = synalinks.HuggingFaceDataset(
        path="gsm8k",
        name="main",
        split="train",
        streaming=True,
        input_data_model=MathQuestion,
        input_template=INPUT_TEMPLATE,
        output_data_model=NumericalAnswer,
        output_template=OUTPUT_TEMPLATE,
        batch_size=4,
        limit=16,
    )

    # Calling the dataset returns a fresh generator. Iterate one batch.
    first_batch_x, first_batch_y = next(iter(ds))
    print(f"\n  batch shape: x={first_batch_x.shape}, y={first_batch_y.shape}")
    print(f"  first question: {first_batch_x[0].question[:80]}...")
    print(f"  first answer:   {first_batch_y[0].answer}")

    # -------------------------------------------------------------------------
    # 2) Convenience helper `load_split`. Same setup as (1) but with
    #    `streaming=False` and one call instead of construct + materialize.
    #    Hands back NumPy object arrays directly.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2) synalinks.datasets.load_split: one HF split → (x, y) arrays")
    print("=" * 60)

    x, y = synalinks.datasets.load_split(
        path="gsm8k",
        name="main",
        split="train",
        input_data_model=MathQuestion,
        input_template=INPUT_TEMPLATE,
        output_data_model=NumericalAnswer,
        output_template=OUTPUT_TEMPLATE,
        limit=20,
    )
    print(f"\n  loaded: x={x.shape}, y={y.shape}")

    # -------------------------------------------------------------------------
    # 3) `split_train_test` — deterministic head/tail slice. Useful for
    #    sources that ship a single labeled split, or (as here) for
    #    carving a quick validation slice out of an HF train split.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3) synalinks.datasets.split_train_test: head/tail validation slice")
    print("=" * 60)

    (x_train, y_train), (x_val, y_val) = synalinks.datasets.split_train_test(
        x,
        y,
        validation_split=0.2,
    )
    print(f"\n  train: x={x_train.shape}, y={y_train.shape}")
    print(f"  val:   x={x_val.shape},  y={y_val.shape}")

    # -------------------------------------------------------------------------
    # 4) Built-in loader. `synalinks.datasets.gsm8k.load_data()` does the
    #    same template setup behind the scenes and hands back two
    #    materialized (x, y) pairs — one for train, one for test.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4) Built-in loader: synalinks.datasets.gsm8k.load_data()")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
    print(f"\n  train: x={x_train.shape}, y={y_train.shape}")
    print(f"  test:  x={x_test.shape},  y={y_test.shape}")
    print(f"  first y_train type: {type(y_train[0]).__name__}")


if __name__ == "__main__":
    main()
