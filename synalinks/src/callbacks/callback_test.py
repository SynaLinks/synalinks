# Modified from: keras/src/callbacks/callback_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

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
