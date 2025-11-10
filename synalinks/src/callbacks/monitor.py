# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import logging
import uuid
from typing import Dict, Literal

import requests

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import api_base
from synalinks.src.backend import api_key
from synalinks.src.callbacks.callback import Callback
from synalinks.src.backend import DataModel


@synalinks_export("synalinks.callbacks.monitor.LogEntry")
class LogEntry(DataModel):
    experiment_id: str
    program_name: str
    program_description: str
    event: Literal["train_begin", "train_end", "batch_begin", "batch_end", "epoch_begin", "epoch_end", "predict_begin", "predict_end"]
    phase: Literal["train", "test", "predict"]
    logs: Dict[str, float]


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
        send_batch_events=False,
        send_predict_events=False,
        send_epoch_events=True,
    ):
        super().__init__()
        self.endpoint = api_base()
        self.timeout = timeout
        if api_key() is not None and not headers:
            headers = {"Authorization": api_key()}
        self.headers = headers or {}
        self._pending_tasks = []
        self.logger = logging.getLogger(__name__)
        self.send_batch_events = send_batch_events
        self.send_predict_events = send_predict_events
        self.send_epoch_events = send_epoch_events
        self._experiment_id = None

    async def _post_trace(self, data: dict):
        """POST trace data to the endpoint asynchronously."""
        url = f"{self.endpoint}/logs"

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

    def _send_trace_async(self, log_data: dict):
        """Send trace asynchronously without blocking."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create task and store reference to prevent garbage collection
        task = loop.create_task(self._post_trace(log_data))
        self._pending_tasks.append(task)

        # Clean up completed tasks
        self._pending_tasks = [t for t in self._pending_tasks if not t.done()]

    def _create_log(self, event: str, phase: str, logs: dict = None):
        """Create a LogEntry and return its JSON representation."""
        log_entry = LogEntry(
            event=event,
            phase=phase,
            experiment_id=self._experiment_id,
            program_name=self.program.name,
            program_description=self.program.description,
            logs=logs or {},
        )
        return log_entry.get_json()

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self._experiment_id = str(uuid.uuid4())
        log_data = self._create_log(event="train_begin", phase="train", logs=logs)
        self._send_trace_async(log_data)

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        log_data = self._create_log(event="train_end", phase="train", logs=logs)
        self._send_trace_async(log_data)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch."""
        if not self.send_epoch_events:
            return

        log_data = self._create_log(event="epoch_begin", phase="train", logs=logs)
        self._send_trace_async(log_data)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        if not self.send_epoch_events:
            return

        log_data = self._create_log(event="epoch_end", phase="train", logs=logs)
        self._send_trace_async(log_data)

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch."""
        if not self.send_batch_events:
            return

        log_data = self._create_log(event="batch_begin", phase="train", logs=logs)
        self._send_trace_async(log_data)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch."""
        if not self.send_batch_events:
            return

        log_data = self._create_log(event="batch_end", phase="train", logs=logs)
        self._send_trace_async(log_data)

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation."""
        if self._experiment_id is None:
            self._experiment_id = f"test_{uuid.uuid4()}"

        log_data = self._create_log(event="train_begin", phase="test", logs=logs)
        self._send_trace_async(log_data)

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation."""
        log_data = self._create_log(event="train_end", phase="test", logs=logs)
        self._send_trace_async(log_data)

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a test batch."""
        if not self.send_batch_events:
            return

        log_data = self._create_log(event="batch_begin", phase="test", logs=logs)
        self._send_trace_async(log_data)

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a test batch."""
        if not self.send_batch_events:
            return

        log_data = self._create_log(event="batch_end", phase="test", logs=logs)
        self._send_trace_async(log_data)

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction."""
        if not self.send_predict_events:
            return

        self._experiment_id = f"predict_{uuid.uuid4()}"
        log_data = self._create_log(event="predict_begin", phase="predict", logs=logs)
        self._send_trace_async(log_data)

    def on_predict_end(self, logs=None):
        """Called at the end of prediction."""
        if not self.send_predict_events:
            return

        log_data = self._create_log(event="predict_end", phase="predict", logs=logs)
        self._send_trace_async(log_data)

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a prediction batch."""
        if not self.send_batch_events:
            return

        log_data = self._create_log(event="batch_begin", phase="predict", logs=logs)
        self._send_trace_async(log_data)

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a prediction batch."""
        if not self.send_batch_events:
            return

        log_data = self._create_log(event="batch_end", phase="predict", logs=logs)
        self._send_trace_async(log_data)

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