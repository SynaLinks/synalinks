# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import collections
import contextvars
import itertools
import json
import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.backend.common.op_scope import current_op_scope
from synalinks.src.backend.config import mlflow_experiment_name
from synalinks.src.backend.config import mlflow_tracking_uri
from synalinks.src.hooks.hook import Hook
from synalinks.src.utils.async_utils import run_maybe_nested

try:
    import mlflow
    from mlflow.entities import SpanType

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    SpanType = None

# Standard MLflow span attribute keys (`mlflow.tracing.constant`), read by
# the UI for chat rendering and by the trace-level token/cost roll-ups.
_ATTR_CHAT_USAGE = "mlflow.chat.tokenUsage"
_ATTR_LLM_COST = "mlflow.llm.cost"
_ATTR_LLM_MODEL = "mlflow.llm.model"
_ATTR_LLM_PROVIDER = "mlflow.llm.provider"
_ATTR_MESSAGE_FORMAT = "mlflow.message.format"
_ATTR_LINKED_PROMPTS = "mlflow.linkedPrompts"
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _chat_messages_of(inputs):
    """OpenAI-format message list of a `LanguageModel` call's inputs, or `None`."""
    if not inputs:
        return None
    first = inputs[0]
    if hasattr(first, "get_json"):
        data = first.get_json()
    elif isinstance(first, dict):
        data = first
    else:
        return None
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return None
    return [
        {k: v for k, v in m.items() if v is not None} if isinstance(m, dict) else m
        for m in messages
    ]


def _chat_tools_of(kwargs):
    """Wire tool declarations of a `LanguageModel` call, or an empty list."""
    tools = []
    for tool in kwargs.get("tools") or []:
        try:
            from synalinks.src.modules.language_models.language_model import _tool_to_wire

            tools.append(_tool_to_wire(tool))
        except Exception:
            continue
    tools.extend(kwargs.get("tool_schemas") or [])
    return tools


def _chat_output_of(serialized_outputs, usage):
    """OpenAI-style `choices` for a `LanguageModel` span's outputs."""
    if not serialized_outputs:
        return None
    payload = serialized_outputs[0]
    if isinstance(payload, dict) and "role" in payload:
        message = {k: v for k, v in payload.items() if v is not None}
    else:
        message = {"role": "assistant", "content": json.dumps(payload)}
    choice = {"index": 0, "message": message}
    if usage and usage.get("finish_reason"):
        choice["finish_reason"] = usage["finish_reason"]
    return [choice]


# Module-level logger. Keeping the logger off the `Monitor` instance is
# load-bearing: the hook is attached to a Module and reachable from every
# `TrackedDict` via the synalinks `Tracker`. A `logging.Logger` reaches the
# global logging tree (handlers carry a `_thread.RLock`), so a `self.logger`
# attribute would make any `copy.deepcopy` of a tracked schema/dict fail
# with `cannot pickle '_thread.lock' object`.
_LOGGER = logging.getLogger(__name__)

# Global registry to track spans across hook instances for parent-child relationships
_GLOBAL_SPANS_REGISTRY: Dict[str, Any] = {}

# Root (non-symbolic) traces in call order: (sequence number, trace id).
_ROOT_TRACES = collections.deque(maxlen=100_000)
_ROOT_TRACE_SEQ = itertools.count()


def root_trace_mark():
    """Returns a mark to pass to `root_trace_ids_since()` later."""
    return next(_ROOT_TRACE_SEQ)


def root_trace_ids_since(mark):
    """Returns the ids of the root traces (top-level, non-symbolic module
    calls) started since `mark`, in call order.

    The order is the order in which the calls began, which is the order of
    the inputs when a batch is run through `asyncio.gather` (the trainer's
    `predict_on_batch`). `callbacks.Monitor` uses this to attach each
    sample's reward to its trace.
    """
    return [tid for seq, tid in sorted(_ROOT_TRACES) if seq > mark]


# ContextVar (not a global) so concurrent async requests each see their own value
_TRACE_CONTEXT = contextvars.ContextVar("synalinks_trace_context", default=None)

# Reserved MLflow metadata keys, used by the UI to group traces per user/session
MLFLOW_TRACE_USER_KEY = "mlflow.trace.user"
MLFLOW_TRACE_SESSION_KEY = "mlflow.trace.session"


@synalinks_export(["synalinks.hooks.trace_context", "synalinks.trace_context"])
class trace_context:
    """Attach a user, a session and free-form metadata to the MLflow traces
    created inside the block.

    MLflow groups traces by user and by chat session (a multi-turn
    conversation) through two reserved metadata keys, `mlflow.trace.user`
    and `mlflow.trace.session`, which lets you inspect what happened at each
    turn of a conversation in the MLflow UI. The `Monitor` hook creates its
    spans outside of MLflow's fluent context (nested module calls are linked
    explicitly), so `mlflow.update_current_trace()` cannot see them. This
    context manager is the Synalinks way to set that metadata: every trace
    started inside the block carries the user, session, tags and metadata
    given here.

    The state lives in a `contextvars.ContextVar`, so it is copied into the
    concurrent tasks spawned inside the block and it is safe to use per
    request in an async server: concurrent requests each see their own value.
    Nested blocks merge with the enclosing one, the innermost value winning.

    Args:
        user_id (str): Optional. The user the traces belong to.
        session_id (str): Optional. The chat session (conversation) the
            traces belong to.
        metadata (dict): Optional. Extra trace metadata (immutable once the
            trace is logged).
        tags (dict): Optional. Extra trace tags (editable afterwards in the
            MLflow UI).

    Example:

    ```python
    import synalinks

    synalinks.enable_observability()

    # ... build your program ...

    with synalinks.trace_context(user_id="user-123", session_id="session-123"):
        result = await program(inputs)
    ```

    In a FastAPI/FastMCP server, wrap each request handler so that the user
    and session ids coming from the client end up on the traces:

    ```python
    @app.post("/chat")
    async def chat(request: ChatRequest):
        with synalinks.trace_context(
            user_id=request.user_id, session_id=request.session_id
        ):
            return await program(request.messages)
    ```
    """

    def __init__(self, user_id=None, session_id=None, metadata=None, tags=None):
        self.metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
        if user_id is not None:
            self.metadata[MLFLOW_TRACE_USER_KEY] = str(user_id)
        if session_id is not None:
            self.metadata[MLFLOW_TRACE_SESSION_KEY] = str(session_id)
        self.tags = {str(k): str(v) for k, v in (tags or {}).items()}
        self._token = None

    def __enter__(self):
        outer = _TRACE_CONTEXT.get() or {"metadata": {}, "tags": {}}
        self._token = _TRACE_CONTEXT.set(
            {
                "metadata": {**outer["metadata"], **self.metadata},
                "tags": {**outer["tags"], **self.tags},
            }
        )
        return self

    def __exit__(self, *exc_info):
        _TRACE_CONTEXT.reset(self._token)
        return False


def current_trace_context():
    """Returns the `{"metadata": ..., "tags": ...}` set by the innermost
    active `trace_context`, or `None` when no trace context is active."""
    return _TRACE_CONTEXT.get()


@synalinks_export("synalinks.callbacks.monitor.Span")
class Span(DataModel):
    """Data model representing a span for module call tracing."""

    event: Literal["call_begin", "call_end"]
    is_symbolic: bool
    call_id: str
    parent_call_id: Optional[str]
    module: str
    module_name: str
    module_description: str
    timestamp: float
    inputs: Optional[List[Dict[str, Any]]] = None
    outputs: Optional[List[Dict[str, Any]]] = None
    duration: Optional[float] = None
    exception: Optional[str] = None
    success: Optional[bool] = None
    cost: Optional[float] = None


@synalinks_export("synalinks.hooks.Monitor")
class Monitor(Hook):
    """Monitor hook for tracing module calls using MLflow.

    This hook creates MLflow spans for each module call, enabling distributed
    tracing and observability of your synalinks programs.

    You can enable monitoring for every module by using
    `synalinks.enable_observability()` at the beginning of your scripts.

    Args:
        tracking_uri (str): MLflow tracking server URI. If None, uses the
            value from `synalinks.enable_observability()` or the default
            (local ./mlruns directory or MLFLOW_TRACKING_URI env var).
        experiment_name (str): Name of the MLflow experiment for tracing.
            If None, uses the value from `synalinks.enable_observability()`
            or defaults to "synalinks_traces".

    Example:

    ```python
    import synalinks

    # Basic usage - uses local MLflow storage
    synalinks.enable_observability()

    # With custom MLflow tracking server
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="my_traces"
    )

    # Or create a Monitor hook directly with custom settings
    monitor = synalinks.hooks.Monitor(
        tracking_uri="http://localhost:5000",
        experiment_name="my_experiment"
    )
    ```
    """

    def __init__(
        self,
        tracking_uri=None,
        experiment_name=None,
    ):
        super().__init__()
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow is required for the Monitor hook. "
                "Install it with: pip install mlflow"
            )

        # Use provided values or fall back to global config
        self.tracking_uri = tracking_uri or mlflow_tracking_uri()
        self.experiment_name = experiment_name or mlflow_experiment_name()
        self.call_start_times = {}
        self._setup_done = False

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        if self._setup_done:
            return

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)
        self._setup_done = True

    def _get_span_type(self):
        """Determine the MLflow span type based on the module class."""
        if SpanType is None:
            return None

        module_class = self.module.__class__.__name__

        # Map module types to MLflow span types
        if module_class == "LanguageModel":
            return SpanType.CHAT_MODEL
        elif module_class == "EmbeddingModel":
            return SpanType.EMBEDDING
        elif module_class == "DecisionModel":
            # Same interface as a `LanguageModel` (chat messages in, structured
            # output out), so it is traced as a chat model too.
            return SpanType.CHAT_MODEL
        elif module_class in ("FunctionCallingAgent",):
            return SpanType.AGENT
        elif module_class in ("EmbedKnowledge", "RetrieveKnowledge", "UpdateKnowledge"):
            return SpanType.RETRIEVER
        elif module_class in ("Tool",):
            return SpanType.TOOL
        else:
            return SpanType.CHAIN

    async def _begin_span_async(
        self,
        call_id,
        parent_call_id,
        serialized_inputs,
        serialized_kwargs,
        is_symbolic,
        span_name,
        span_type,
        metadata=None,
        tags=None,
        root_seq=None,
        chat_messages=None,
        chat_tools=None,
    ):
        """Async implementation of span creation."""
        global _GLOBAL_SPANS_REGISTRY

        # Look up parent span from global registry for proper trace hierarchy
        parent_span_obj = None
        if parent_call_id and parent_call_id in _GLOBAL_SPANS_REGISTRY:
            parent_span_obj = _GLOBAL_SPANS_REGISTRY[parent_call_id]

        # Use start_span_no_context for manual lifecycle management
        # This properly supports parent-child relationships
        span = await asyncio.to_thread(
            mlflow.start_span_no_context,
            name=span_name,
            span_type=span_type,
            parent_span=parent_span_obj,
            metadata=metadata or None,
            tags=tags or None,
        )

        # Store only in the module-level registry. Storing the span on the
        # hook instance would expose mlflow's internal `_thread.lock` state
        # through the tracker graph (see comment on `_LOGGER` above).
        _GLOBAL_SPANS_REGISTRY[call_id] = span
        if root_seq is not None:
            _ROOT_TRACES.append((root_seq, span.trace_id))

        attributes = {
            "synalinks.call_id": call_id,
            "synalinks.parent_call_id": parent_call_id or "",
            "synalinks.module": str(self.module.__class__.__name__),
            "synalinks.module_name": self.module.name or "",
            "synalinks.module_description": self.module.description or "",
            "synalinks.is_symbolic": is_symbolic,
        }
        if span_type == SpanType.CHAT_MODEL:
            model = getattr(self.module, "model", None) or ""
            attributes[_ATTR_LLM_MODEL] = model
            attributes[_ATTR_LLM_PROVIDER] = model.split("/")[0] if model else ""
            attributes[_ATTR_MESSAGE_FORMAT] = "openai"
        prompt_version = getattr(self.module, "_mlflow_prompt_version", None)
        if prompt_version is not None:
            attributes[_ATTR_LINKED_PROMPTS] = json.dumps(
                [{"name": prompt_version.name, "version": str(prompt_version.version)}]
            )
            try:
                from mlflow.entities.model_registry import PromptVersion
                from mlflow.tracing.trace_manager import InMemoryTraceManager

                # A version restored by `Program.load()` is a plain record; only
                # a registry object can be attached to the trace itself.
                if isinstance(prompt_version, PromptVersion):
                    InMemoryTraceManager.get_instance().register_prompt(
                        trace_id=span.trace_id, prompt=prompt_version
                    )
            except Exception as e:
                _LOGGER.debug(f"Failed to link prompt to trace: {e}")
        span.set_attributes(attributes)
        if chat_tools:
            try:
                from mlflow.tracing.utils import set_span_chat_tools

                set_span_chat_tools(span, chat_tools)
            except Exception as e:
                _LOGGER.debug(f"Failed to set chat tools on span: {e}")

        inputs_dict = {"data": serialized_inputs}
        if chat_messages is not None:
            inputs_dict = {"messages": chat_messages, **inputs_dict}
        if serialized_kwargs:
            inputs_dict["kwargs"] = serialized_kwargs
        span.set_inputs(inputs_dict)

        _LOGGER.debug(f"Started span for call {call_id}: {span_name}")

    def on_call_begin(
        self,
        call_id,
        parent_call_id=None,
        inputs=None,
        kwargs=None,
    ):
        """Called when a module call begins."""
        self._setup_mlflow()
        self.call_start_times[call_id] = time.time()

        # `hasattr(d, "get_schema")` filters out `LanguageModel` streaming
        # outputs (a `StreamingIterator` has no materialized payload to log).
        is_symbolic = any_symbolic_data_models(inputs)
        leaves = [
            d for d in tree.flatten(inputs) if d is not None and hasattr(d, "get_schema")
        ]
        serialized_inputs = (
            [d.get_schema() for d in leaves]
            if is_symbolic
            else [d.get_json() for d in leaves]
        )

        # Serialize kwargs if present (for modules that use keyword arguments).
        # Like the inputs: schemas when the call is symbolic, JSON data
        # otherwise. So the target `schema` of a model call is only logged
        # when building the program, not on every real call.
        serialized_kwargs = {}
        if kwargs:
            # Filter out non-serializable kwargs like 'training'
            for key, value in kwargs.items():
                if key == "training":
                    serialized_kwargs[key] = value
                elif key == "schema":
                    if is_symbolic:
                        serialized_kwargs[key] = value
                elif hasattr(value, "get_json"):
                    serialized_kwargs[key] = (
                        value.get_schema() if is_symbolic else value.get_json()
                    )
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serialized_kwargs[key] = value

        span_name = f"{self.module.__class__.__name__}"
        if self.module.name:
            span_name = f"{span_name}:{self.module.name}"

        # Get the appropriate span type for this module
        span_type = self._get_span_type()

        # Read the ContextVar in the caller's context, not inside the coroutine
        trace_ctx = current_trace_context() or {}
        tags = dict(trace_ctx.get("tags") or {})
        # Sequence taken synchronously here so it follows the call order, not
        # the completion order of the threaded span creation.
        root_seq = None
        if parent_call_id is None and not is_symbolic:
            root_seq = next(_ROOT_TRACE_SEQ)
            # Trace-level tags of the root span: which program ran, in which
            # phase (inference, reward or optimizer), so traces can be filtered.
            tags.setdefault("synalinks.program", self.module.name or "")
            tags.setdefault("synalinks.phase", current_op_scope() or "inference")

        chat_messages = None
        chat_tools = None
        # A symbolic call has no messages yet, only their schema (in the inputs).
        if span_type == SpanType.CHAT_MODEL and not is_symbolic:
            chat_messages = _chat_messages_of(inputs)
            chat_tools = _chat_tools_of(kwargs or {})

        run_maybe_nested(
            self._begin_span_async(
                call_id=call_id,
                parent_call_id=parent_call_id,
                serialized_inputs=serialized_inputs,
                serialized_kwargs=serialized_kwargs,
                is_symbolic=is_symbolic,
                span_name=span_name,
                span_type=span_type,
                metadata=trace_ctx.get("metadata"),
                tags=tags or None,
                root_seq=root_seq,
                chat_messages=chat_messages,
                chat_tools=chat_tools,
            )
        )

    async def _end_span_async(
        self,
        call_id,
        span,
        serialized_outputs,
        duration,
        cost,
        exception,
        usage=None,
    ):
        """Async implementation of span ending."""
        attributes = {
            "synalinks.duration": duration,
            "synalinks.success": exception is None,
            "synalinks.cost": cost or 0.0,
        }
        if usage and "input_tokens" in usage:
            attributes[_ATTR_CHAT_USAGE] = {
                key: usage[key]
                for key in _USAGE_KEYS
                if usage.get(key) or key in _USAGE_KEYS[:3]
            }
            if usage.get("cost") is not None:
                attributes[_ATTR_LLM_COST] = {"total_cost": usage["cost"]}
        if usage and usage.get("cache_hit"):
            attributes["synalinks.cache_hit"] = True
        if usage and usage.get("model"):
            attributes["synalinks.response_model"] = usage["model"]
        if usage and usage.get("fallback"):
            attributes["synalinks.fallback"] = True
        # A call that failed every attempt returns `None` instead of raising:
        # report it as the failure it is.
        if exception is None and usage and usage.get("error"):
            exception = RuntimeError(usage["error"])
            attributes["synalinks.success"] = False
        span.set_attributes(attributes)

        if exception:
            span.set_attributes({"synalinks.exception": str(exception)})
            # Add exception event for better visibility in MLflow UI
            span.add_event(
                mlflow.entities.SpanEvent(
                    name="exception",
                    attributes={
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )
            )
            span.set_status("ERROR")
        else:
            span.set_status("OK")

        outputs_dict = {"data": serialized_outputs}
        if usage is not None and exception is None:
            choices = _chat_output_of(serialized_outputs, usage)
            if choices is not None:
                outputs_dict = {"choices": choices, **outputs_dict}
        if usage and usage.get("answers") is not None:
            # A decision model's raw answers, with probabilities and confidence.
            outputs_dict["answers"] = usage["answers"]
        span.set_outputs(outputs_dict)

        await asyncio.to_thread(span.end)

        success = exception is None
        _LOGGER.debug(
            f"Ended span for call {call_id}, duration={duration:.3f}s, success={success}"
        )

    def on_call_end(
        self,
        call_id,
        parent_call_id=None,
        outputs=None,
        exception=None,
    ):
        """Called when a module call ends."""
        global _GLOBAL_SPANS_REGISTRY

        end_time = time.time()
        start_time = self.call_start_times.pop(call_id, end_time)
        duration = end_time - start_time

        span = _GLOBAL_SPANS_REGISTRY.pop(call_id, None)

        if span is None:
            _LOGGER.warning(f"No span found for call_id {call_id}")
            return

        leaves = [
            d for d in tree.flatten(outputs) if d is not None and hasattr(d, "get_schema")
        ]
        serialized_outputs = (
            [d.get_schema() for d in leaves]
            if any_symbolic_data_models(outputs)
            else [d.get_json() for d in leaves]
        )

        cost = None
        if self.module._get_call_context():
            cost = self.module._get_call_context().cost

        usage = None
        module_class = self.module.__class__.__name__
        if module_class == "LanguageModel":
            from synalinks.src.modules.language_models.language_model import (
                current_call_usage,
            )

            # Read in the caller's task: the ContextVar was set by `call()`.
            usage = current_call_usage() or {}
        elif module_class == "DecisionModel":
            from synalinks.src.modules.decision_models.decision_model import (
                current_call_usage,
            )

            usage = current_call_usage() or {}

        run_maybe_nested(
            self._end_span_async(
                call_id=call_id,
                span=span,
                serialized_outputs=serialized_outputs,
                duration=duration,
                cost=cost,
                exception=exception,
                usage=usage,
            )
        )

    def __del__(self):
        """End the spans this monitor opened and never closed.

        Only its own: the registry is shared by every monitor in the process
        (a child span finds its parent there), so clearing it all would end
        the in-flight spans of any other live monitor whenever this one
        happens to be garbage-collected. The calls this monitor opened and
        has not ended yet are exactly the keys of ``call_start_times``.
        """
        for call_id in list(getattr(self, "call_start_times", {})):
            span = _GLOBAL_SPANS_REGISTRY.pop(call_id, None)
            if span is None:
                continue
            try:
                span.end()
            except Exception:
                pass
