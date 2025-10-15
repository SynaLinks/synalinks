# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import logging
import time

import requests

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.backend import api_base
from synalinks.src.backend import api_key
from synalinks.src.hooks.hook import Hook


@synalinks_export("synalinks.hooks.Monitor")
class Monitor(Hook):
    """Monitor hook for sending module call traces to a remote endpoint in realtime.

    This hook sends trace data immediately to a specified endpoint for realtime monitoring.
    Traces are sent asynchronously using asyncio to avoid blocking module execution.

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

    async def _post_trace(self, data: dict):
        """POST trace data to the endpoint asynchronously."""
        url = f"{self.endpoint}/trace"

        try:
            loop = asyncio.get_event_loop()
            # Run requests in executor to make it non-blocking
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    url,
                    json=data,
                    headers=self.headers,
                    timeout=self.timeout,
                )
            )
            response.raise_for_status()
            self.logger.debug(
                f"Trace sent successfully: {data.get('event')} for call {data.get('call_id')}"
            )
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout sending trace to {url}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send trace to {url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error sending trace: {e}")

    def _send_trace_async(self, trace_data: dict):
        """Send trace asynchronously without blocking."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create task and store reference to prevent garbage collection
        task = loop.create_task(self._post_trace(trace_data))
        self._pending_tasks.append(task)

        # Clean up completed tasks
        self._pending_tasks = [t for t in self._pending_tasks if not t.done()]

    def _extract_data_models_info(self, data):
        """Extract data model information from inputs/outputs."""
        if not data:
            return None

        flatten_data = tree.flatten(data)

        if any_symbolic_data_models(data):
            schemas = [dm.get_schema() for dm in flatten_data if dm is not None]
            return {
                "type": "symbolic",
                "schemas": schemas,
            }
        else:
            jsons = [dm.get_json() for dm in flatten_data if dm is not None]
            return {
                "type": "data",
                "data": jsons,
            }

    def on_call_begin(
        self,
        call_id,
        inputs=None,
    ):
        """Called when a module call begins."""
        self.call_start_times[call_id] = time.time()

        trace_data = {
            "event": "call_begin",
            "call_id": call_id,
            "module_name": self.module.name,
            "module_description": self.module.description,
            "timestamp": self.call_start_times[call_id],
            "inputs": self._extract_data_models_info(inputs),
        }

        self._send_trace_async(trace_data)

    def on_call_end(
        self,
        call_id,
        outputs=None,
        exception=None,
    ):
        """Called when a module call ends."""
        end_time = time.time()
        start_time = self.call_start_times.pop(call_id, end_time)
        duration = end_time - start_time

        trace_data = {
            "event": "call_end",
            "call_id": call_id,
            "module_name": self.module.name,
            "module_description": self.module.description,
            "timestamp": end_time,
            "duration": duration,
            "outputs": self._extract_data_models_info(outputs),
            "exception": str(exception) if exception else None,
            "success": exception is None,
        }

        self._send_trace_async(trace_data)

    async def _cleanup(self):
        """Wait for pending tasks."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)

    def __del__(self):
        """Cleanup pending traces."""
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