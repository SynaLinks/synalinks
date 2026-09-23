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
        self.assertEqual(
            kwargs["tags"],
            {"synalinks.program": "language_model", "synalinks.phase": "inference"},
        )

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
        self.assertEqual(
            kwargs["tags"],
            {
                "env": "test",
                "synalinks.program": "language_model",
                "synalinks.phase": "inference",
            },
        )


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


class _RecordingSpan:
    """Fake live span that records what the hook sets on it."""

    def __init__(self):
        self.attributes = {}
        self.inputs = None
        self.outputs = None
        self.status = None
        self.trace_id = "tr-1"

    def set_attributes(self, attributes):
        self.attributes.update(attributes)

    def set_inputs(self, inputs):
        self.inputs = inputs

    def set_outputs(self, outputs):
        self.outputs = outputs

    def set_status(self, status):
        self.status = status

    def add_event(self, event):
        pass

    def end(self):
        pass


class MonitorStandardAttributesTest(testing.TestCase):
    def _make_monitor(self):
        monitor = Monitor.__new__(Monitor)
        monitor.tracking_uri = None
        monitor.experiment_name = "test"
        monitor.call_start_times = {}
        monitor._setup_done = True
        return monitor

    @patch("synalinks.src.hooks.monitor.mlflow")
    @patch("litellm.acompletion")
    async def test_language_model_span_carries_usage_cost_model_and_chat(
        self, mock_completion, mock_mlflow
    ):
        from litellm.types.utils import Choices
        from litellm.types.utils import Message
        from litellm.types.utils import ModelResponse
        from litellm.types.utils import Usage

        response = ModelResponse(
            choices=[Choices(message=Message(content="Hi there"), finish_reason="stop")],
            usage=Usage(prompt_tokens=12, completion_tokens=3, total_tokens=15),
        )
        response._hidden_params = {"response_cost": 0.0021}
        mock_completion.return_value = response
        span = _RecordingSpan()
        mock_mlflow.start_span_no_context.return_value = span

        lm = LanguageModel(model="openai/gpt-4o-mini", hooks=[self._make_monitor()])
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        await lm(messages)

        self.assertEqual(span.attributes["mlflow.llm.model"], "openai/gpt-4o-mini")
        self.assertEqual(span.attributes["mlflow.llm.provider"], "openai")
        self.assertEqual(span.attributes["mlflow.message.format"], "openai")
        self.assertEqual(
            span.attributes["mlflow.chat.tokenUsage"],
            {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        )
        self.assertEqual(span.attributes["mlflow.llm.cost"], {"total_cost": 0.0021})
        self.assertEqual(span.inputs["messages"], [{"role": "user", "content": "Hello"}])
        self.assertIn("data", span.inputs)
        choice = span.outputs["choices"][0]
        self.assertEqual(choice["message"]["content"], "Hi there")
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["finish_reason"], "stop")
        tags = mock_mlflow.start_span_no_context.call_args.kwargs["tags"]
        self.assertEqual(tags["synalinks.phase"], "inference")

    @patch("synalinks.src.hooks.monitor.mlflow")
    @patch("litellm.acompletion")
    async def test_usage_without_cost_sets_no_cost_attribute(
        self, mock_completion, mock_mlflow
    ):
        from litellm.types.utils import Choices
        from litellm.types.utils import Message
        from litellm.types.utils import ModelResponse
        from litellm.types.utils import Usage

        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="ok"))],
            usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        span = _RecordingSpan()
        mock_mlflow.start_span_no_context.return_value = span
        lm = LanguageModel(model="ollama/mistral", hooks=[self._make_monitor()])
        await lm(ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="x")]))
        self.assertIn("mlflow.chat.tokenUsage", span.attributes)
        self.assertNotIn("mlflow.llm.cost", span.attributes)

    @patch("synalinks.src.hooks.monitor.mlflow")
    @patch("litellm.acompletion")
    async def test_collecting_another_monitor_keeps_live_spans(
        self, mock_completion, mock_mlflow
    ):
        # Every monitor shares the span registry. A stale monitor (say, from an
        # earlier program) garbage-collected while another's call is in flight
        # must end only its own spans, not wipe the live call's.
        import gc

        from litellm.types.utils import Choices
        from litellm.types.utils import Message
        from litellm.types.utils import ModelResponse
        from litellm.types.utils import Usage

        from synalinks.src.hooks import monitor as monitor_module

        stale = self._make_monitor()
        stale_span = _RecordingSpan()
        stale_span.ended = False
        stale_span.end = lambda *args, **kwargs: setattr(stale_span, "ended", True)
        stale.call_start_times["stale-call"] = 0.0
        monitor_module._GLOBAL_SPANS_REGISTRY["stale-call"] = stale_span

        async def completion(*args, **kwargs):
            nonlocal stale
            stale = None
            gc.collect()  # the stale monitor is collected mid-call
            return ModelResponse(
                choices=[Choices(message=Message(content="ok"))],
                usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
            )

        mock_completion.side_effect = completion
        span = _RecordingSpan()
        mock_mlflow.start_span_no_context.return_value = span
        lm = LanguageModel(model="ollama/mistral", hooks=[self._make_monitor()])
        await lm(ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="x")]))
        self.assertIn("mlflow.chat.tokenUsage", span.attributes)
        # ...while the stale monitor's own unfinished span was ended.
        self.assertTrue(stale_span.ended)
        self.assertNotIn("stale-call", monitor_module._GLOBAL_SPANS_REGISTRY)

    @patch("synalinks.src.hooks.monitor.mlflow")
    async def test_module_with_registered_prompt_links_it_on_span(self, mock_mlflow):
        from synalinks.src.backend import DataModel
        from synalinks.src.modules import Module

        class Query(DataModel):
            query: str

        class Echo(Module):
            async def call(self, inputs, training=False):
                return inputs

        from mlflow.entities.model_registry import PromptVersion

        span = _RecordingSpan()
        mock_mlflow.start_span_no_context.return_value = span
        module = Echo(name="echo", hooks=[self._make_monitor()])
        module._mlflow_prompt_version = PromptVersion(
            name="prog.echo", version=3, template="hello {{ inputs }}"
        )
        with patch("mlflow.tracing.trace_manager.InMemoryTraceManager") as manager:
            await module(Query(query="q"))
        self.assertEqual(
            span.attributes["mlflow.linkedPrompts"],
            '[{"name": "prog.echo", "version": "3"}]',
        )
        manager.get_instance.return_value.register_prompt.assert_called_once()
        self.assertNotIn("mlflow.llm.model", span.attributes)
        self.assertNotIn("mlflow.chat.tokenUsage", span.attributes)

        # A version restored by `Program.load()` is a plain record: the span
        # attribute is still set, the trace-level registration skipped.
        span = _RecordingSpan()
        mock_mlflow.start_span_no_context.return_value = span
        module._mlflow_prompt_version = type(
            "PV", (), {"name": "prog.echo", "version": 4, "uri": "prompts:/prog.echo/4"}
        )()
        with patch("mlflow.tracing.trace_manager.InMemoryTraceManager") as manager:
            await module(Query(query="q"))
        self.assertEqual(
            span.attributes["mlflow.linkedPrompts"],
            '[{"name": "prog.echo", "version": "4"}]',
        )
        manager.get_instance.return_value.register_prompt.assert_not_called()
