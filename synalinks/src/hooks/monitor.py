# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
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

# Global registry to track spans across hook instances for parent-child relationships
_GLOBAL_SPANS_REGISTRY: Dict[str, Any] = {}


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
        self._spans = {}
        self.logger = logging.getLogger(__name__)
        self._setup_done = False

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        if self._setup_done:
            return

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)
        self._setup_done = True

    def _serialize_data(self, data):
        """Serialize data models to JSON-compatible format."""
        flatten_data = tree.flatten(data)
        is_symbolic = any_symbolic_data_models(data)

        if is_symbolic:
            serialized = [dm.get_schema() for dm in flatten_data if dm is not None]
        else:
            serialized = [dm.get_json() for dm in flatten_data if dm is not None]

        return serialized, is_symbolic

    def _get_span_type(self):
        """Determine the MLflow span type based on the module class."""
        if SpanType is None:
            return None

        module_class = self.module.__class__.__name__

        # Map module types to MLflow span types
        if module_class in ("Generator", "ChainOfThought", "SelfCritique"):
            return SpanType.LLM
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
        )

        # Store in both local and global registry
        self._spans[call_id] = span
        _GLOBAL_SPANS_REGISTRY[call_id] = span

        span.set_attributes(
            {
                "synalinks.call_id": call_id,
                "synalinks.parent_call_id": parent_call_id or "",
                "synalinks.module": str(self.module.__class__.__name__),
                "synalinks.module_name": self.module.name or "",
                "synalinks.module_description": self.module.description or "",
                "synalinks.is_symbolic": is_symbolic,
            }
        )

        # Set inputs as a dictionary (MLflow handles serialization)
        inputs_dict = {"data": serialized_inputs}
        if serialized_kwargs:
            inputs_dict["kwargs"] = serialized_kwargs
        span.set_inputs(inputs_dict)

        self.logger.debug(f"Started span for call {call_id}: {span_name}")

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

        serialized_inputs, is_symbolic = self._serialize_data(inputs)

        # Serialize kwargs if present (for modules that use keyword arguments)
        serialized_kwargs = {}
        if kwargs:
            # Filter out non-serializable kwargs like 'training'
            for key, value in kwargs.items():
                if key == "training":
                    serialized_kwargs[key] = value
                elif hasattr(value, "get_json"):
                    serialized_kwargs[key] = value.get_json()
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serialized_kwargs[key] = value

        span_name = f"{self.module.__class__.__name__}"
        if self.module.name:
            span_name = f"{span_name}:{self.module.name}"

        # Get the appropriate span type for this module
        span_type = self._get_span_type()

        run_maybe_nested(
            self._begin_span_async(
                call_id=call_id,
                parent_call_id=parent_call_id,
                serialized_inputs=serialized_inputs,
                serialized_kwargs=serialized_kwargs,
                is_symbolic=is_symbolic,
                span_name=span_name,
                span_type=span_type,
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
    ):
        """Async implementation of span ending."""
        span.set_attributes(
            {
                "synalinks.duration": duration,
                "synalinks.success": exception is None,
                "synalinks.cost": cost or 0.0,
            }
        )

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

        # Set outputs as a dictionary (MLflow handles serialization)
        span.set_outputs({"data": serialized_outputs})

        await asyncio.to_thread(span.end)

        success = exception is None
        self.logger.debug(
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

        span = self._spans.pop(call_id, None)
        # Also remove from global registry
        _GLOBAL_SPANS_REGISTRY.pop(call_id, None)

        if span is None:
            self.logger.warning(f"No span found for call_id {call_id}")
            return

        serialized_outputs, _ = self._serialize_data(outputs)

        cost = None
        if self.module._get_call_context():
            cost = self.module._get_call_context().cost

        run_maybe_nested(
            self._end_span_async(
                call_id=call_id,
                span=span,
                serialized_outputs=serialized_outputs,
                duration=duration,
                cost=cost,
                exception=exception,
            )
        )

    def __del__(self):
        """Cleanup any open spans."""
        if hasattr(self, "_spans"):
            for call_id, span in list(self._spans.items()):
                try:
                    span.end()
                except Exception:
                    pass
            self._spans.clear()
