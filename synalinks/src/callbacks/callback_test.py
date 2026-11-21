# Modified from: keras/src/callbacks/callback_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.callbacks import Callback
from synalinks.src.optimizers.random_few_shot import RandomFewShot
from synalinks.src.programs import Program
from synalinks.src.rewards import ExactMatch


class CallbackTest(testing.TestCase):
    def test_model_state_is_current_on_epoch_end(self):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        class Iterations(DataModel):
            count: int = 0

        class TestProgram(Program):
            def __init__(self):
                super().__init__()
                self.iterations = self.add_variable(
                    data_model=Iterations,
                )

            def call(self, inputs):
                self.iterations.json["count"] += 1
                return inputs

        class CBK(Callback):
            def on_batch_end(self, batch, logs):
                assert self.iterations.json["count"] == batch + 1

        model = TestProgram()
        model.compile(optimizer=RandomFewShot(), reward=ExactMatch())
        x = [Query(query="test 1"), Query(query="test 2")]
        y = [Answer(answer="answer 1"), Answer(answer="answer 2")]
        model.fit(x, y, callbacks=[CBK()], batch_size=2)


async def _failing_fit(*callbacks):
    import json
    from unittest.mock import patch

    import numpy as np

    from synalinks.src.modules import Generator
    from synalinks.src.modules import Input
    from synalinks.src.modules.language_models import LanguageModel
    from synalinks.src.testing.test_utils import AnswerWithRationale
    from synalinks.src.testing.test_utils import Query

    class Boom(Callback):
        def on_epoch_end(self, epoch, logs=None):
            raise RuntimeError("boom")

    x0 = Input(data_model=Query)
    x1 = await Generator(
        data_model=AnswerWithRationale,
        language_model=LanguageModel(model="ollama/mistral"),
    )(x0)
    program = Program(inputs=x0, outputs=x1, name="p")
    program.compile(reward=ExactMatch(in_mask=["answer"]), optimizer=RandomFewShot())
    x = np.array([Query(query="q")] * 2, dtype="object")
    y = np.array([AnswerWithRationale(rationale="r", answer="a")] * 2, dtype="object")
    answer = json.dumps({"rationale": "r", "answer": "a"})
    with patch("litellm.acompletion") as completion:
        completion.return_value = {"choices": [{"message": {"content": answer}}]}
        await program.fit(
            x=x, y=y, epochs=1, batch_size=2, callbacks=[*callbacks, Boom()]
        )


class TrainErrorTest(testing.TestCase):
    async def test_failed_fit_calls_on_train_error_not_on_train_end(self):
        seen = []

        class Recorder(Callback):
            def on_train_end(self, logs=None):
                seen.append("end")

            def on_train_error(self, error):
                seen.append(error)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await _failing_fit(Recorder())
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], RuntimeError)

    async def test_failed_fit_resets_budget_stopping(self):
        from synalinks.src.callbacks.budget_stopping import BudgetStopping

        budget = BudgetStopping(max_tokens=10**9)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            await _failing_fit(budget)
        self.assertFalse(budget._in_training)

    async def test_failed_fit_ends_the_mlflow_run_as_failed(self):
        import os
        import tempfile

        import mlflow

        from synalinks.src.callbacks.monitor import Monitor

        uri = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "mlflow.db")
        monitor = Monitor(
            tracking_uri=uri,
            experiment_name="e",
            log_program_plot=False,
            log_program_model=False,
        )
        with self.assertRaisesRegex(RuntimeError, "boom"):
            await _failing_fit(monitor)
        self.assertIsNone(mlflow.active_run())
        self.assertFalse(monitor._in_training)
        runs = mlflow.search_runs(experiment_names=["e"], output_format="list")
        self.assertEqual([r.info.status for r in runs], ["FAILED"])
