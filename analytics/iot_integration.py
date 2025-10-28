"""IoT device integration stubs.

This module provides an interface to trigger IoT devices or actuators in
response to security events. In a production environment, this could
send HTTP requests, activate relays, or interact with proprietary
protocols. For this offline demo, the `IoTController` simply prints
messages or writes to a log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import datetime


class IoTController:
    """Simple IoT integration interface.

    Parameters
    ----------
    log_dir : str or Path, optional
        Directory where IoT trigger events will be logged. If not provided,
        events are only printed to stdout.
    """

    def __init__(self, log_dir: Optional[str | Path] = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def trigger(self, event: dict) -> None:
        """Trigger an IoT device based on an event.

        Parameters
        ----------
        event : dict
            Event dictionary containing at least a `type` key and any
            additional metadata needed by the device.
        """
        timestamp = datetime.datetime.utcnow().isoformat()
        msg = f"[IoT] {timestamp}: Triggering action for event: {event}"
        print(msg)
        if self.log_dir:
            log_path = self.log_dir / "iot_triggers.jsonl"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": timestamp, "event": event}) + "\n")