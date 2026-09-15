# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import warnings

from synalinks.src.api_export import synalinks_export
from synalinks.src.callbacks.callback import Callback
from synalinks.src.metrics.em_metrics import _collect_embedding_models
from synalinks.src.metrics.lm_metrics import _collect_language_models
from synalinks.src.utils import io_utils


@synalinks_export("synalinks.callbacks.BudgetStopping")
class BudgetStopping(Callback):
    """Stop training, evaluation or prediction once a budget has been reached.

    Running an LM program costs real money: every batch triggers inference
    calls, reward computations and optimizer calls, each billed by the
    provider. This callback tracks what the current `program.fit()`,
    `program.evaluate()` or `program.predict()` run has spent so far, in
    dollars and/or in tokens, and stops it as soon as one of the budgets
    is reached. The check runs after every batch (training, validation,
    evaluation and prediction), so a run stops mid-epoch instead of
    finishing the epoch it was in when the budget ran out. Inside `fit()`,
    validation spend counts toward the same budget and a budget hit during
    validation ends both the validation pass and the training.

    Spending is measured as the increase, since the run began, of the
    all-time `cumulated_cost` and `cumulated_tokens` counters of every
    `LanguageModel` and `EmbeddingModel` reachable from the program
    (including their `fallback` chains). Calls made from every phase
    count: inference, reward and optimizer.

    The dollar cost comes from the provider's response (via LiteLLM's
    `response_cost`). Local providers such as Ollama or vLLM don't report
    a cost, so `max_cost` never triggers with them; use `max_tokens`
    instead. The callback warns once when it detects that situation.

    Example:

    ```python
    callback = synalinks.callbacks.BudgetStopping(max_cost=5.0)
    # This callback will stop the training as soon as the run has spent
    # more than 5 dollars across all LM and EM calls.
    program = synalinks.programs.Sequential(
        [synalinks.modules.Generator(data_model=Answer)])
    program.compile(
        synalinks.optimizers.RandomFewShot(),
        reward=synalinks.rewards.ExactMatch())
    history = await program.fit(
        ..., epochs=10, batch_size=1, callbacks=[callback], verbose=0)
    # The same callback caps a standalone evaluation or prediction.
    results = await program.evaluate(..., callbacks=[callback])
    ```

    Args:
        max_cost (float): Maximum amount, in dollars, the run is
            allowed to spend before stopping. `None` disables the cost budget.
            Defaults to `None`.
        max_tokens (int): Maximum number of tokens (prompt + completion,
            summed over LM and EM calls) the run is allowed to
            consume before stopping. `None` disables the token budget.
            Defaults to `None`.
        verbose (int): Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
            displays a message when the callback stops the run.
            Defaults to `0`.
    """

    def __init__(
        self,
        max_cost=None,
        max_tokens=None,
        verbose=0,
    ):
        super().__init__()
        if max_cost is None and max_tokens is None:
            raise ValueError(
                "BudgetStopping requires at least one budget: "
                "pass `max_cost` (in dollars) and/or `max_tokens`."
            )
        if max_cost is not None and max_cost <= 0:
            raise ValueError(f"`max_cost` must be strictly positive, got {max_cost}.")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"`max_tokens` must be strictly positive, got {max_tokens}.")
        self.max_cost = max_cost
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._cost_baseline = 0.0
        self._tokens_baseline = 0
        self._cost_warned = False
        self._epoch = 0
        self._in_training = False
        self._batch_kind = "batch"
        self.stopped_epoch = None
        self.stopped_batch = None
        self.stop_reason = None

    def _models(self):
        return _collect_language_models(self.program) + _collect_embedding_models(
            self.program
        )

    def _read_cost(self):
        return sum(getattr(m, "cumulated_cost", 0.0) for m in self._models())

    def _read_tokens(self):
        return sum(getattr(m, "cumulated_tokens", 0) for m in self._models())

    @property
    def spent_cost(self):
        """Dollars spent since the start of the current run."""
        return self._read_cost() - self._cost_baseline

    @property
    def spent_tokens(self):
        """Tokens consumed since the start of the current run."""
        return self._read_tokens() - self._tokens_baseline

    def _start_run(self):
        # Allow instances to be re-used
        self._cost_baseline = self._read_cost()
        self._tokens_baseline = self._read_tokens()
        self._cost_warned = False
        self._epoch = 0
        self.stopped_epoch = None
        self.stopped_batch = None
        self.stop_reason = None

    def on_train_begin(self, logs=None):
        self._in_training = True
        self._batch_kind = "batch"
        self._start_run()

    def on_test_begin(self, logs=None):
        # `fit()` runs validation through the same callbacks: keep its budget.
        if not self._in_training:
            self._batch_kind = "evaluation batch"
            self._start_run()

    def on_predict_begin(self, logs=None):
        if not self._in_training:
            self._batch_kind = "prediction batch"
            self._start_run()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        self._check_budget(batch)

    def on_test_batch_end(self, batch, logs=None):
        self._check_budget(batch)

    def on_predict_batch_end(self, batch, logs=None):
        self._check_budget(batch)

    def on_epoch_end(self, epoch, logs=None):
        self._epoch = epoch
        if (
            self.max_cost is not None
            and not self._cost_warned
            and self.spent_cost == 0.0
            and self.spent_tokens > 0
        ):
            self._cost_warned = True
            warnings.warn(
                "BudgetStopping has a `max_cost` budget but the provider did not "
                "report any cost for the calls made so far (local providers "
                "such as Ollama or vLLM never do). The cost budget will never "
                "trigger; consider using `max_tokens` instead.",
                stacklevel=2,
            )
        self._check_budget(None)

    def _check_budget(self, batch):
        if self.stop_reason is not None:
            return
        reason = None
        if self.max_cost is not None and self.spent_cost >= self.max_cost:
            reason = f"cost budget reached (${self.spent_cost:.4f} >= ${self.max_cost})"
        elif self.max_tokens is not None and self.spent_tokens >= self.max_tokens:
            reason = f"token budget reached ({self.spent_tokens} >= {self.max_tokens})"
        if reason is None:
            return
        self.stop_reason = reason
        self.stopped_epoch = self._epoch
        self.stopped_batch = batch
        self.program.stop_training = True
        self.program.stop_evaluating = True
        self.program.stop_predicting = True

    def _report(self):
        if self.stop_reason is None or self.verbose == 0:
            return
        where = f"Epoch {self.stopped_epoch + 1}" if self._in_training else ""
        if self.stopped_batch is not None:
            where += ", " if where else ""
            where += f"{self._batch_kind} {self.stopped_batch + 1}"
        io_utils.print_msg(f"{where.capitalize()}: budget stopping, {self.stop_reason}")

    def on_train_end(self, logs=None):
        self._report()
        self._in_training = False

    def on_test_end(self, logs=None):
        if not self._in_training:
            self._report()

    def on_predict_end(self, logs=None):
        if not self._in_training:
            self._report()
