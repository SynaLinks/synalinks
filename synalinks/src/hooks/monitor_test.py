# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import AsyncMock
from unittest.mock import patch

from mlflow.entities import SpanType

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import EmbeddingRequest
from synalinks.src.hooks.monitor import MLFLOW_TRACE_SESSION_KEY
from synalinks.src.hooks.monitor import MLFLOW_TRACE_USER_KEY
from synalinks.src.hooks.monitor import Monitor
from synalinks.src.hooks.monitor import current_trace_context
from synalinks.src.hooks.monitor import root_trace_ids_since
from synalinks.src.hooks.monitor import root_trace_mark
from synalinks.src.hooks.monitor import trace_context
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
        orchestrators that wrap an LM call: the actual `CHAT_MODEL`
        span is emitted by the inner `LanguageModel`, so these wrappers
        belong to `SpanType.CHAIN` (MLflow convention)."""
        for class_name in ("Generator", "ChainOfThought", "SelfCritique"):
            fake_module = type(class_name, (), {"name": "x", "description": ""})()
            monitor = Monitor.__new__(Monitor)
            monitor.set_module(fake_module)
            self.assertEqual(monitor._get_span_type(), SpanType.CHAIN)


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

        result = await em(EmbeddingRequest(texts=["hello world"]))

        self.assertEqual(result.get_json(), {"embeddings": [[0.1, 0.2]]})
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
            result.get_json(),
            ChatMessage(role=ChatRole.ASSISTANT, content="Hi there").get_json(),
        )
        self.assertEqual(mock_mlflow.start_span_no_context.call_count, 1)
        span_type = mock_mlflow.start_span_no_context.call_args.kwargs["span_type"]
        self.assertEqual(span_type, SpanType.CHAT_MODEL)

    @patch("synalinks.src.hooks.monitor.mlflow")
    @patch("litellm.acompletion")
    async def test_monitor_forwards_trace_context_to_span(
        self, mock_completion, mock_mlflow
    ):
        mock_completion.return_value = {"choices": [{"message": {"content": "Hi"}}]}
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

        await lm(messages)
        kwargs = mock_mlflow.start_span_no_context.call_args.kwargs
        self.assertIsNone(kwargs["metadata"])
        self.assertIsNone(kwargs["tags"])

        with trace_context(
            user_id="user-123", session_id="session-123", tags={"env": "test"}
        ):
            await lm(messages)
        kwargs = mock_mlflow.start_span_no_context.call_args.kwargs
        self.assertEqual(
            kwargs["metadata"],
            {
                MLFLOW_TRACE_USER_KEY: "user-123",
                MLFLOW_TRACE_SESSION_KEY: "session-123",
            },
        )
        self.assertEqual(kwargs["tags"], {"env": "test"})


class TraceContextTest(testing.TestCase):
    def test_no_context_by_default(self):
        self.assertIsNone(current_trace_context())

    def test_user_and_session_map_to_mlflow_reserved_keys(self):
        with trace_context(user_id="user-123", session_id="session-123"):
            ctx = current_trace_context()
        self.assertEqual(
            ctx["metadata"],
            {
                MLFLOW_TRACE_USER_KEY: "user-123",
                MLFLOW_TRACE_SESSION_KEY: "session-123",
            },
        )
        self.assertEqual(ctx["tags"], {})
        self.assertIsNone(current_trace_context())

    def test_nested_contexts_merge_innermost_wins(self):
        with trace_context(user_id="user-1", tags={"env": "prod"}):
            with trace_context(session_id="s-2", metadata={"turn": 3}):
                ctx = current_trace_context()
                self.assertEqual(
                    ctx["metadata"],
                    {
                        MLFLOW_TRACE_USER_KEY: "user-1",
                        MLFLOW_TRACE_SESSION_KEY: "s-2",
                        "turn": "3",
                    },
                )
                self.assertEqual(ctx["tags"], {"env": "prod"})
            self.assertEqual(
                current_trace_context()["metadata"], {MLFLOW_TRACE_USER_KEY: "user-1"}
            )
        self.assertIsNone(current_trace_context())

    async def test_concurrent_tasks_see_their_own_context(self):
        import asyncio

        async def handle(user):
            with trace_context(user_id=user):
                await asyncio.sleep(0)
                return current_trace_context()["metadata"][MLFLOW_TRACE_USER_KEY]

        users = await asyncio.gather(handle("a"), handle("b"))
        self.assertEqual(list(users), ["a", "b"])


class RootTraceRegistryTest(testing.TestCase):
    @patch("synalinks.src.hooks.monitor.mlflow")
    @patch("litellm.acompletion")
    async def test_root_traces_recorded_in_call_order(self, mock_completion, mock_mlflow):
        mock_completion.return_value = {"choices": [{"message": {"content": "Hi"}}]}
        spans = []

        def make_span(**kwargs):
            span = AsyncMock()
            span.trace_id = f"tr-{len(spans)}"
            span.set_attributes = lambda *a, **kw: None
            span.set_inputs = lambda *a, **kw: None
            span.set_outputs = lambda *a, **kw: None
            span.set_status = lambda *a, **kw: None
            span.end = lambda: None
            spans.append(span)
            return span

        mock_mlflow.start_span_no_context.side_effect = make_span
        monitor = MonitorEndToEndTest._make_monitor(self)
        lm = LanguageModel(model="ollama/mistral", hooks=[monitor])
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        before = root_trace_mark()
        await lm(messages)
        mark = root_trace_mark()
        import asyncio

        await asyncio.gather(lm(messages), lm(messages), lm(messages))

        self.assertEqual(root_trace_ids_since(mark), ["tr-1", "tr-2", "tr-3"])
        self.assertEqual(len(root_trace_ids_since(before)), 4)
        self.assertEqual(root_trace_ids_since(root_trace_mark()), [])
