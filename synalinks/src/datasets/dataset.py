# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import jinja2
import numpy as np
import orjson

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_data_model import JsonDataModel
from synalinks.src.backend.pydantic.common import ChatMessage
from synalinks.src.backend.pydantic.common import ChatMessages


@synalinks_export(["synalinks.Dataset", "synalinks.datasets.Dataset"])
class Dataset:
    """Base class for streaming datasets compatible with synalinks trainers.

    `Trainer.fit/evaluate/predict(x=...)` accepts a Python generator that
    yields `(inputs,)` or `(inputs, targets)` tuples — one tuple per
    batch. See `synalinks/src/trainers/data_adapters/generator_data_adapter.py`
    and the dispatch in `synalinks/src/trainers/data_adapters/__init__.py`.

    Subclasses implement ``_iter_rows()`` as a generator yielding raw row
    dicts (one per source example). The base class' ``__iter__`` then
    renders each row through the Jinja2 templates, validates the shape,
    and yields batched ``(x, y)`` numpy object arrays — including the
    ``repeat`` expansion. Calling the dataset returns a fresh generator
    suitable for synalinks:

    ```python
    program.evaluate(x=my_dataset())
    ```

    The shape of the per-row input/target objects is controlled by either
    a Python ``DataModel`` class (``input_data_model`` / ``output_data_model``)
    OR a raw JSON Schema (``input_schema`` / ``output_schema``). The two
    are mutually exclusive on each side. With a class, rows are validated
    via ``cls.model_validate_json(rendered)``. With a schema, rows are
    wrapped as ``JsonDataModel(schema=..., json=json.loads(rendered))`` —
    the schema flows directly into the LM as a structured-output
    constraint, so any JSON Schema feature (enum, const, oneOf, ...) is
    supported.

    Args:
        input_data_model (DataModel): Python class describing batch
            inputs. Defaults to ``synalinks.ChatMessages`` when neither
            this nor ``input_schema`` is provided.
        input_schema (dict | str): Raw JSON Schema for batch inputs. May
            be given as a dict or as a JSON-encoded string. Mutually
            exclusive with ``input_data_model``.
        input_template (str): Jinja2 template string used to render raw
            rows into the input shape. Required.
        output_data_model (DataModel): Python class describing batch
            targets. Defaults to ``synalinks.ChatMessage`` when
            ``output_template`` is given but neither this nor
            ``output_schema`` is. Must be omitted when ``output_template``
            is omitted.
        output_schema (dict | str): Raw JSON Schema for batch targets.
            Mutually exclusive with ``output_data_model``. Must be omitted
            when ``output_template`` is omitted.
        output_template (str): Jinja2 template string used to render raw
            rows into the target shape. Optional — when omitted, the
            dataset is inputs-only and yields single-element ``(x,)``
            batches (no targets). Rewards that need ``y_true`` will see
            it missing.
        batch_size (int): Number of examples per yielded batch. ``None``
            accumulates everything into a single trailing batch.
        limit (int): Optional. Maximum number of *raw* (pre-repeat)
            examples to iterate over. ``None`` (default) means no cap.
            Useful for capping huge or streaming sources for quick
            experiments / smoke tests.
        repeat (int): Number of consecutive copies to emit per raw
            example. Defaults to 1 (no expansion). Setting
            ``repeat == batch_size`` makes every batch a group of N
            rollouts of the same prompt — the expected layout for
            GRPO-style RL where reward statistics are computed across
            rollouts of one input.
        **kwargs: Provider-specific fields forwarded by subclasses (e.g.
            HF dataset name, split, revision, API key, file path, ...).
    """

    def __init__(
        self,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        batch_size=None,
        limit=None,
        repeat=1,
        **kwargs,
    ):
        if input_template is None:
            raise ValueError("`input_template` is required (Jinja2 template).")
        if input_data_model is not None and input_schema is not None:
            raise ValueError(
                "Pass either `input_data_model` or `input_schema`, not both."
            )
        if output_data_model is not None and output_schema is not None:
            raise ValueError(
                "Pass either `output_data_model` or `output_schema`, not both."
            )
        if output_template is None and (
            output_data_model is not None or output_schema is not None
        ):
            raise ValueError(
                "`output_data_model` / `output_schema` require `output_template` "
                "(omit all three for an inputs-only dataset)."
            )
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"`repeat` must be a positive int; got {repeat!r}.")
        # Default to ChatMessages / ChatMessage when neither a data_model
        # nor a schema is given.
        if input_data_model is None and input_schema is None:
            input_data_model = ChatMessages
        if (
            output_template is not None
            and output_data_model is None
            and output_schema is None
        ):
            output_data_model = ChatMessage
        self.input_data_model = input_data_model
        self.input_schema = _coerce_schema(input_schema)
        self.input_template = input_template
        self.output_data_model = output_data_model
        self.output_schema = _coerce_schema(output_schema)
        self.output_template = output_template
        self.batch_size = batch_size
        self.limit = limit
        self.repeat = repeat

        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        self._input_tmpl = env.from_string(input_template)
        self._output_tmpl = (
            env.from_string(output_template) if output_template is not None else None
        )

    def _make_input(self, rendered):
        if self.input_schema is not None:
            return JsonDataModel(schema=self.input_schema, json=orjson.loads(rendered))
        return self.input_data_model.model_validate_json(rendered)

    def _make_target(self, rendered):
        if self.output_schema is not None:
            return JsonDataModel(schema=self.output_schema, json=orjson.loads(rendered))
        return self.output_data_model.model_validate_json(rendered)

    def _iter_rows(self):
        """Yield raw row dicts from the underlying source.

        Subclasses must implement this. Each yielded dict is passed as
        kwargs to the Jinja2 input/output templates, so its keys must be
        valid Python identifiers matching the template variables.
        """
        raise NotImplementedError

    def __iter__(self):
        """Render rows through the templates and yield batches.

        Yields ``(x, y)`` when an ``output_template`` is configured, or
        single-element ``(x,)`` batches when it isn't. Honors ``limit``
        (caps raw rows), ``repeat`` (each raw example is emitted
        ``repeat`` times in a row), and ``batch_size`` (size of the
        yielded numpy object arrays; ``None`` accumulates everything into
        a single trailing batch). The trailing partial batch is always
        flushed at the end.
        """
        inputs_only = self._output_tmpl is None
        bs = self.batch_size
        x_buf, y_buf = [], []
        seen = 0
        for row in self._iter_rows():
            if self.limit is not None and seen >= self.limit:
                break
            seen += 1
            x = self._make_input(self._input_tmpl.render(**row))
            y = (
                None
                if inputs_only
                else self._make_target(self._output_tmpl.render(**row))
            )
            for _ in range(self.repeat):
                x_buf.append(x)
                if not inputs_only:
                    y_buf.append(y)
                if bs is not None and len(x_buf) >= bs:
                    yield _batch(x_buf, y_buf, inputs_only)
                    x_buf, y_buf = [], []
        if x_buf:
            yield _batch(x_buf, y_buf, inputs_only)

    def _total_batches(self, num_rows):
        """Number of batches given ``num_rows`` raw (pre-repeat) examples.

        ``batch_size=None`` collapses to a single batch.
        """
        n = num_rows * self.repeat
        if self.batch_size is None:
            return 1 if n > 0 else 0
        return (n + self.batch_size - 1) // self.batch_size

    def __call__(self):
        """Return a fresh generator over the dataset's batches."""
        return iter(self)

    def __len__(self):
        """Number of batches, if known. Override when the size is finite."""
        raise NotImplementedError

    def materialize(self):
        """Iterate the dataset to exhaustion and concatenate every batch.

        Returns a single ``(x,)`` or ``(x, y)`` pair — numpy object
        arrays of ``DataModel`` instances — suitable for
        ``program.evaluate(x=x, y=y)``, ``program.fit(x=x, y=y)``,
        or for slicing into train/test splits with
        ``split_train_test``.

        This is the streaming-to-arrays bridge: any ``Dataset``
        subclass — ``HuggingFaceDataset``, a custom CSV loader,
        anything else — can be force-evaluated into in-memory NumPy
        object arrays with a single method call. Use it for small
        benchmark datasets that fit comfortably in memory; for huge
        sources, iterate via ``ds()`` instead so rows stream on
        demand.

        Returns:
            ``(x,)`` if the dataset is inputs-only (no
            ``output_template`` configured), otherwise ``(x, y)``.
        """
        inputs_only = self._output_tmpl is None
        x_buf, y_buf = [], []
        for batch in self:
            x_buf.extend(batch[0])
            if not inputs_only:
                y_buf.extend(batch[1])
        x = np.array(x_buf, dtype="object")
        if inputs_only:
            return (x,)
        return (x, np.array(y_buf, dtype="object"))


@synalinks_export(["synalinks.datasets.split_train_test"])
def split_train_test(x, y, validation_split=0.2):
    """Deterministic head/tail split — for sources that ship a single
    labeled split (HumanEval, IFEval, BBH, TruthfulQA, BBQ, ...).

    Args:
        x: Input numpy object array.
        y: Target numpy object array.
        validation_split (float): Fraction of the data that goes to
            the test set. Defaults to ``0.2`` (Keras convention).

    Returns:
        ``(x_train, y_train), (x_test, y_test)``.
    """
    n = len(x)
    cut = int(n * (1.0 - validation_split))
    return (x[:cut], y[:cut]), (x[cut:], y[cut:])


def _batch(x_buf, y_buf, inputs_only):
    x = np.array(x_buf, dtype="object")
    if inputs_only:
        return (x,)
    return (x, np.array(y_buf, dtype="object"))


def _coerce_schema(schema):
    """Accept a JSON Schema as either a dict or a JSON-encoded string."""
    if schema is None:
        return None
    if isinstance(schema, str):
        return orjson.loads(schema)
    if isinstance(schema, dict):
        return schema
    raise TypeError(
        f"`schema` must be a dict or JSON string; got {type(schema).__name__}."
    )
