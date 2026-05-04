# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import AsyncMock
from unittest.mock import patch

from mlflow.entities import SpanType

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.hooks.monitor import Monitor
from synalinks.src.modules.embedding_models import EmbeddingModel
from synalinks.src.modules.language_models import LanguageModel


def _new_monitor(module):
    """Build a Monitor without running its real `__init__` (avoids MLflow setup)."""
    monitor = Monitor.__new__(Monitor)
    monitor.set_module(module)
    return monitor


class MonitorSpanTypeTest(testing.TestCase):
    def test_language_model_maps_to_chat_model(self):
        monitor = _new_monitor(LanguageModel(model="ollama/mistral"))
        self.assertEqual(monitor._get_span_type(), SpanType.CHAT_MODEL)

    def test_embedding_model_maps_to_embedding(self):
        monitor = _new_monitor(EmbeddingModel(model="ollama/all-minilm"))
        self.assertEqual(monitor._get_span_type(), SpanType.EMBEDDING)

    def test_orchestrator_modules_map_to_chain(self):
        """`Generator` / `ChainOfThought` / `SelfCritique` are
        orchestrators that wrap an LM call — the actual `CHAT_MODEL`
        span is emitted by the inner `LanguageModel`, so these wrappers
        belong to `SpanType.CHAIN` (MLflow convention)."""
        for class_name in ("Generator", "ChainOfThought", "SelfCritique"):
            fake_module = type(class_name, (), {"name": "x", "description": ""})()
            monitor = Monitor.__new__(Monitor)
            monitor.set_module(fake_module)
            self.assertEqual(monitor._get_span_type(), SpanType.CHAIN)


class MonitorSerializeDataTest(testing.TestCase):
    def test_serialize_raw_list_of_strings_preserves_payload(self):
        """`EmbeddingModel` inputs are `list[str]` — flattening would
        keep the strings but `.get_json()` would crash. The non-DataModel
        path must pass the value through unchanged."""
        monitor = _new_monitor(EmbeddingModel(model="ollama/all-minilm"))

        serialized, is_symbolic = monitor._serialize_data((["a", "b"],))

        self.assertFalse(is_symbolic)
        self.assertEqual(serialized, [(["a", "b"],)])

    def test_serialize_plain_dict_preserves_keys(self):
        """`tree.flatten` on a dict drops keys, which destroys the LM/EM
        output payload. Plain dicts must round-trip with structure intact."""
        monitor = _new_monitor(EmbeddingModel(model="ollama/all-minilm"))

        payload = {"embeddings": [[0.1, 0.2]]}
        serialized, is_symbolic = monitor._serialize_data(payload)

        self.assertFalse(is_symbolic)
        self.assertEqual(serialized, [payload])

    def test_serialize_data_model_uses_get_json(self):
        """The standard module IO path: a DataModel goes through `get_json`."""
        monitor = _new_monitor(LanguageModel(model="ollama/mistral"))

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        serialized, is_symbolic = monitor._serialize_data(messages)

        self.assertFalse(is_symbolic)
        self.assertEqual(serialized, [messages.get_json()])


class MonitorEndToEndTest(testing.TestCase):
    """Exercise the full hook pipeline on LM/EM with a real `Monitor`
    instance, mocking out `litellm` and MLflow's tracing surface so we
    don't actually start a tracking server."""

    def _make_monitor(self):
        monitor = Monitor.__new__(Monitor)
        monitor.tracking_uri = None
        monitor.experiment_name = "test"
        monitor.call_start_times = {}
        monitor._spans = {}
        import logging

        monitor.logger = logging.getLogger("monitor_test")
        monitor._setup_done = True  # skip mlflow.set_experiment
        return monitor

    @patch("synalinks.src.hooks.monitor.mlflow")
    @patch("litellm.aembedding")
    async def test_embedding_model_call_traces_without_crashing(
        self, mock_embedding, mock_mlflow
    ):
        mock_embedding.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        fake_span = AsyncMock()
        fake_span.set_attributes = lambda *a, **kw: None
        fake_span.set_inputs = lambda *a, **kw: None
        fake_span.set_outputs = lambda *a, **kw: None
        fake_span.set_status = lambda *a, **kw: None
        fake_span.end = lambda: None
        mock_mlflow.start_span_no_context.return_value = fake_span

        monitor = self._make_monitor()
        em = EmbeddingModel(model="ollama/all-minilm", hooks=[monitor])

        result = await em(["hello world"])

        self.assertEqual(result, {"embeddings": [[0.1, 0.2]]})
        # Begin + end → span created exactly once.
        self.assertEqual(mock_mlflow.start_span_no_context.call_count, 1)
        span_type = mock_mlflow.start_span_no_context.call_args.kwargs["span_type"]
        self.assertEqual(span_type, SpanType.EMBEDDING)

    @patch("synalinks.src.hooks.monitor.mlflow")
    @patch("litellm.acompletion")
    async def test_language_model_call_traces_without_crashing(
        self, mock_completion, mock_mlflow
    ):
        mock_completion.return_value = {"choices": [{"message": {"content": "Hi there"}}]}
        fake_span = AsyncMock()
        fake_span.set_attributes = lambda *a, **kw: None
        fake_span.set_inputs = lambda *a, **kw: None
        fake_span.set_outputs = lambda *a, **kw: None
        fake_span.set_status = lambda *a, **kw: None
        fake_span.end = lambda: None
        mock_mlflow.start_span_no_context.return_value = fake_span

        monitor = self._make_monitor()
        lm = LanguageModel(model="ollama/mistral", hooks=[monitor])
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        result = await lm(messages)

        self.assertEqual(
            result,
            ChatMessage(role=ChatRole.ASSISTANT, content="Hi there").get_json(),
        )
        self.assertEqual(mock_mlflow.start_span_no_context.call_count, 1)
        span_type = mock_mlflow.start_span_no_context.call_args.kwargs["span_type"]
        self.assertEqual(span_type, SpanType.CHAT_MODEL)
