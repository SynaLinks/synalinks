# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import logging
import time
from typing import Dict, Optional, Any, List, Literal

import requests

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.backend import api_base
from synalinks.src.backend import api_key
from synalinks.src.hooks.hook import Hook
from synalinks.src.backend import DataModel


@synalinks_export("synalinks.callbacks.monitor.Span")
class Span(DataModel):
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
    """Monitor hook for sending module call spans to a remote endpoint in realtime.

    This hook sends span data immediately to a specified endpoint for realtime monitoring.
    Spans are sent asynchronously using asyncio to avoid blocking module execution.

    You can enable monitoring for every modules by using `synalinks.enable_observability()`
    at the beginning of your scripts:

    Example:

    ```python
    import synalinks

    synalinks.enable_observability()
    ```

    Args:
        timeout: Request timeout in seconds (default: 5).
        headers: Optional additional headers to include in requests
    """

    def __init__(
        self,
        timeout=5,
        headers=None,
    ):
        super().__init__()
        self.endpoint = api_base()
        self.timeout = timeout
        if api_key() is not None and not headers:
            headers = {"Authorization": api_key()}
        self.headers = headers or {}
        self.call_start_times = {}
        self._pending_tasks = []
        self.logger = logging.getLogger(__name__)

    async def _post_span(self, span):
        """POST span data to the endpoint asynchronously."""
        url = f"{self.endpoint}/traces"

        try:
            loop = asyncio.get_event_loop()
            # Run requests in executor to make it non-blocking
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    url,
                    json=span.get_json(),
                    headers=self.headers,
                    timeout=self.timeout,
                )
            )
            response.raise_for_status()
            self.logger.debug(
                f"Span sent successfully: {span.event} for call {span.call_id}"
            )
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout sending span to {url}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send span to {url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error sending span: {e}")

    def _send_span_async(self, span):
        """Send span asynchronously without blocking."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create task and store reference to prevent garbage collection
        task = loop.create_task(self._post_span(span))
        self._pending_tasks.append(task)

        # Clean up completed tasks
        self._pending_tasks = [t for t in self._pending_tasks if not t.done()]

    def on_call_begin(
        self,
        call_id,
        parent_call_id=None,
        inputs=None,
    ):
        """Called when a module call begins."""
        self.call_start_times[call_id] = time.time()
        
        flatten_inputs = tree.flatten(inputs)
        is_symbolic = False
        if any_symbolic_data_models(inputs):
            is_symbolic = True
            inputs = [dm.get_schema() for dm in flatten_inputs if dm is not None]
        else:
            inputs = [dm.get_json() for dm in flatten_inputs if dm is not None]

        span = Span(
            event="call_begin",
            is_symbolic=is_symbolic,
            call_id=call_id,
            parent_call_id=parent_call_id,
            module=str(self.module.__class__.__name__),
            module_name=self.module.name,
            module_description=self.module.description,
            timestamp=self.call_start_times[call_id],
            success=True,
            inputs=inputs,
        )

        self._send_span_async(span)

    def on_call_end(
        self,
        call_id,
        parent_call_id=None,
        outputs=None,
        exception=None,
    ):
        """Called when a module call ends."""
        end_time = time.time()
        start_time = self.call_start_times.pop(call_id, end_time)
        duration = end_time - start_time
        
        flatten_outputs = tree.flatten(outputs)
        is_symbolic = False
        if any_symbolic_data_models(outputs):
            is_symbolic = True
            outputs = [dm.get_schema() for dm in flatten_outputs if dm is not None]
        else:
            outputs = [dm.get_json() for dm in flatten_outputs if dm is not None]

        span = Span(
            event="call_end",
            is_symbolic=is_symbolic,
            call_id=call_id,
            parent_call_id=parent_call_id,
            module=str(self.module.__class__.__name__),
            module_name=self.module.name,
            module_description=self.module.description,
            timestamp=end_time,
            duration=duration,
            outputs=outputs,
            exception=str(exception) if exception else None,
            success=exception is None,
            cost=self.module._get_call_context().cost,
        )
        
        self._send_span_async(span)

    async def _cleanup(self):
        """Wait for pending tasks."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)

    def __del__(self):
        """Cleanup pending spans."""
        if hasattr(self, "_pending_tasks") and self._pending_tasks:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    loop.create_task(self._cleanup())
                else:
                    # If loop is not running, run cleanup
                    loop.run_until_complete(self._cleanup())
            except Exception:
                pass