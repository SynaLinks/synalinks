# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import logging
import time
import uuid

import requests

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import api_base
from synalinks.src.backend import api_key
from synalinks.src.callbacks.callback import Callback


@synalinks_export("synalinks.callbacks.Monitor")
class Monitor(Callback):
    """Monitor callback for sending training/evaluation/prediction logs to a remote endpoint in realtime.

    This callback sends trace data immediately to a specified endpoint for realtime monitoring
    of training progress, and evaluation metrics.
    Traces are sent asynchronously using asyncio to avoid blocking program execution.

    Args:
        timeout: Request timeout in seconds (default: 5)
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
        self._run_id = None
        self._epoch_start_times = {}
        self._batch_start_times = {}

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
                f"Trace sent successfully: {data.get('event')} for {data.get('phase')}"
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

    def _create_trace(self, event: str, phase: str, logs: dict = None, **kwargs):
        """Create a trace data dictionary."""
        trace_data = {
            "event": event,
            "phase": phase,
            "timestamp": time.time(),
            "run_id": self._run_id,
            "program_name": self.program.name if self.program else None,
            "logs": logs or {},
            **kwargs,
        }
        return trace_data

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self._run_id = f"train_{int(time.time() * 1000)}"
        trace_data = self._create_trace(
            event="train_begin", phase="train", logs=logs, params=self.params
        )
        self._send_trace_async(trace_data)

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        trace_data = self._create_trace(event="train_end", phase="train", logs=logs)
        self._send_trace_async(trace_data)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch."""
        if not self.send_epoch_events:
            return

        self._epoch_start_times[epoch] = time.time()
        trace_data = self._create_trace(
            event="epoch_begin", phase="train", logs=logs, epoch=epoch
        )
        self._send_trace_async(trace_data)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        if not self.send_epoch_events:
            return

        start_time = self._epoch_start_times.pop(epoch, time.time())
        duration = time.time() - start_time

        trace_data = self._create_trace(
            event="epoch_end", phase="train", logs=logs, epoch=epoch, duration=duration
        )
        self._send_trace_async(trace_data)

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch."""
        if not self.send_batch_events:
            return

        self._batch_start_times[batch] = time.time()
        trace_data = self._create_trace(
            event="batch_begin", phase="train", logs=logs, batch=batch
        )
        self._send_trace_async(trace_data)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch."""
        if not self.send_batch_events:
            return

        start_time = self._batch_start_times.pop(batch, time.time())
        duration = time.time() - start_time

        trace_data = self._create_trace(
            event="batch_end", phase="train", logs=logs, batch=batch, duration=duration
        )
        self._send_trace_async(trace_data)

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation."""
        if self._run_id is None:
            self._run_id = f"test_{int(time.time() * 1000)}"

        trace_data = self._create_trace(
            event="test_begin", phase="test", logs=logs, params=self.params
        )
        self._send_trace_async(trace_data)

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation."""
        trace_data = self._create_trace(event="test_end", phase="test", logs=logs)
        self._send_trace_async(trace_data)

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a test batch."""
        if not self.send_batch_events:
            return

        self._batch_start_times[batch] = time.time()
        trace_data = self._create_trace(
            event="batch_begin", phase="test", logs=logs, batch=batch
        )
        self._send_trace_async(trace_data)

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a test batch."""
        if not self.send_batch_events:
            return

        start_time = self._batch_start_times.pop(batch, time.time())
        duration = time.time() - start_time

        trace_data = self._create_trace(
            event="batch_end", phase="test", logs=logs, batch=batch, duration=duration
        )
        self._send_trace_async(trace_data)

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction."""
        self._run_id = str(uuid.uuid4())
        trace_data = self._create_trace(
            event="predict_begin", phase="predict", logs=logs, params=self.params
        )
        self._send_trace_async(trace_data)

    def on_predict_end(self, logs=None):
        """Called at the end of prediction."""
        trace_data = self._create_trace(event="predict_end", phase="predict", logs=logs)
        self._send_trace_async(trace_data)

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a prediction batch."""
        if not self.send_batch_events:
            return

        self._batch_start_times[batch] = time.time()
        trace_data = self._create_trace(
            event="batch_begin", phase="predict", logs=logs, batch=batch
        )
        self._send_trace_async(trace_data)

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a prediction batch."""
        if not self.send_batch_events:
            return

        start_time = self._batch_start_times.pop(batch, time.time())
        duration = time.time() - start_time

        trace_data = self._create_trace(
            event="batch_end", phase="predict", logs=logs, batch=batch, duration=duration
        )
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