# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import socket
import getpass
import uuid
import logging
import platform
from functools import lru_cache
import requests
import mixpanel

from synalinks.src.api_export import synalinks_export
from synalinks.src.hooks.hook import Hook
from synalinks.src.version import version
from synalinks.src.utils.async_utils import run_maybe_nested


_SYNALINKS_NAMESPACE = uuid.UUID("15f23928-68be-40c1-b69e-bc8b1fa71686")


_TELEMETRY = mixpanel.Mixpanel(
    "fbb605804abbb93efd661f5898c43cb5",
    consumer=mixpanel.Consumer(
        api_host="api-eu.mixpanel.com",
        retry_limit=2,
    ),
)

def device_id():
    """Generate a stable anonymous ID from hostname + username."""
    hostname = socket.gethostname()
    user = getpass.getuser()
    raw = f"{hostname}::{user}"
    return str(uuid.uuid5(_SYNALINKS_NAMESPACE, raw))


@lru_cache(maxsize=1)
def get_public_geo():
    resp = requests.get("https://ipapi.co/json/", timeout=3)
    resp.raise_for_status()
    data = resp.json()

    return {
        "city": data.get("city", "Unknown"),
        "region": data.get("region", "Unknown"),
        "country_code": data.get("country_code", "UN").upper(),
    }


class Telemetry(Hook):
    """
    Synalinks Telemetry Hook
    
    To disable the telemetry just use `synalinks.disable_telemetry()` at the beginning of your scripts
    
    ```python
    import synalinks
    
    synalinks.disable_telemetry()
    ```
    """

    async def _send(self, event, extra=None):
        try:
            geo = get_public_geo()

            props = {
                "module_class": self.module.__class__.__name__,
                "module_name": getattr(self.module, "name", "Unknown"),
                "module_description": getattr(self.module, "description", ""),
                "$os": platform.system(),
                "$city": geo["city"],
                "$region": geo["region"],
                "mp_country_code": geo["country_code"],
                "version": version(),
            }

            if extra:
                props.update(extra)

        
            _TELEMETRY.track(
                distinct_id=device_id(),
                event_name=event,
                properties=props,
            )

            # Identify this device/person once for Mixpanel People
            _TELEMETRY.people_set_once(device_id(), props)

            # Flush immediately
            if hasattr(_TELEMETRY, "flush"):
                _TELEMETRY.flush()

        except Exception as e:
            pass

    def on_call_begin(self, call_id, parent_call_id=None, inputs=None):
        run_maybe_nested(self._send("call_begin_"+self.module.name))

    def on_call_end(self, call_id, parent_call_id=None, outputs=None, exception=None):
        if exception:
            run_maybe_nested(self._send("exception_"+self.module.name, {"exception": str(exception)}))
        else:
            run_maybe_nested(self._send("call_end_"+self.module.name))
