"""Where your data comes from — currently empty on purpose.

`train.py` loads the built-in GSM8K benchmark directly:

    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

When you move to *your own* task, put the data loading here and have `train.py`
import it instead, e.g.:

    from prepare_data import load_data
    (x_train, y_train), (x_test, y_test) = load_data()

`x` and `y` are lists of `synalinks.DataModel` instances (the program's input and
target). There are three ways to produce them — pick one and uncomment it.


# =============================================================================
# 1. Use a built-in benchmark dataset
# =============================================================================
# Synalinks ships ready-to-train benchmarks under `synalinks.datasets.*`, each
# exposing `load_data()` -> `(x_train, y_train), (x_test, y_test)`:
#
#   gsm8k, mmlu, hotpotqa, humaneval, squad, drop, boolq, arc_challenge, bbh,
#   hellaswag, ifeval, logiqa, lambada, truthfulqa, winogrande, bbq, arcagi
#
#   import synalinks
#
#   def load_data():
#       return synalinks.datasets.mmlu.load_data()
#
# Each module also exposes `get_input_data_model()` / `get_output_data_model()`
# so your program can target the exact schema the dataset produces.


# =============================================================================
# 2. Load your own files (CSV / JSON / JSONL / Parquet / HuggingFace)
# =============================================================================
# A `Dataset` streams raw rows, renders each through a Jinja2 *template* into
# JSON, and validates it against a `DataModel`. The classes share one contract:
# `CSVDataset`, `JSONDataset`, `JSONLDataset`, `ParquetDataset`,
# `HuggingFaceDataset`, `DuckDBDataset`, `LanceDBDataset`, `MarkdownDataset`.
#
#   import synalinks
#
#   class Question(synalinks.DataModel):
#       question: str = synalinks.Field(description="The question to answer")
#
#   class Answer(synalinks.DataModel):
#       answer: str = synalinks.Field(description="The expected answer")
#
#   def load_data():
#       # The template keys are the source column / field names. `tojson`
#       # quotes+escapes strings; use `float` / `int` to coerce numbers.
#       ds = synalinks.CSVDataset(
#           path="data/qa.csv",                       # columns: question, answer
#           input_data_model=Question,
#           input_template='{"question": {{ question | tojson }}}',
#           output_data_model=Answer,
#           output_template='{"answer": {{ answer | tojson }}}',
#           batch_size=None,                          # None -> the whole split at once
#       )
#       x, y = next(iter(ds))                         # materialize to (x, y) lists
#       # Source has a single labeled split? Carve out a test set deterministically:
#       return synalinks.datasets.split_train_test(x, y, validation_split=0.2)
#
# For RL-style streaming (batched, repeated rollouts) pass the dataset object
# straight to fit: keep `batch_size=...`/`repeat=...` and call `program.fit(x=ds)`.


# =============================================================================
# 3. Build the (x, y) lists yourself
# =============================================================================
# No file at all — just construct DataModel instances in code. Simplest path for
# small / synthetic / programmatically-generated data.
#
#   import synalinks
#
#   def load_data():
#       x = [Question(question="2 + 2 = ?"), Question(question="3 * 3 = ?")]
#       y = [Answer(answer="4"), Answer(answer="9")]
#       split = int(len(x) * 0.8)
#       return (x[:split], y[:split]), (x[split:], y[split:])
#
# For full control over streaming/rendering you can subclass `synalinks.Dataset`
# and implement its row iterator — see `synalinks/src/datasets/` for examples.
"""
