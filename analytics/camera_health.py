"""Camera health monitoring.

This module tracks the liveness of camera feeds. If no frame has been
received for a configurable period, it flags the feed as down. This
information can be used by higher layers (e.g. rule engine or
dashboard) to alert operators to hardware issues.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple


class CameraHealthMonitor:
    """Monitors the health of multiple cameras.

    Parameters
    ----------
    timeout : float
        Number of seconds after which a camera is considered down if
        no frame has been received.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self.last_seen: Dict[str, float] = {}
        self.status: Dict[str, bool] = {}

    def register(self, camera_id: str) -> None:
        """Initialise monitoring state for a camera."""
        now = time.time()
        self.last_seen.setdefault(camera_id, now)
        self.status.setdefault(camera_id, True)

    def update(self, camera_id: str, timestamp: Optional[float] = None) -> None:
        """Record that a frame has been received from a camera."""
        self.last_seen[camera_id] = timestamp or time.time()

    def is_down(self, camera_id: str) -> bool:
        """Return True if the camera has not sent a frame within timeout."""
        last = self.last_seen.get(camera_id)
        if last is None:
            return True
        return (time.time() - last) > self.timeout

    def evaluate(self, camera_id: str) -> Tuple[bool, bool]:
        """Return (healthy, changed) for the given camera."""
        healthy = not self.is_down(camera_id)
        prev = self.status.get(camera_id)
        changed = prev is not None and prev != healthy
        self.status[camera_id] = healthy
        return healthy, changed

    def force_down(self, camera_id: str) -> Tuple[bool, bool]:
        """Mark a camera as down and return (healthy, changed)."""
        prev = self.status.get(camera_id, True)
        self.status[camera_id] = False
        return False, bool(prev)

    def get_status(self, camera_id: str) -> bool:
        """Return True if the camera is currently healthy."""
        return self.status.get(camera_id, False)

    def get_down_cameras(self) -> list[str]:
        """Return a list of camera IDs that are currently down."""
        now = time.time()
        result = []
        for cid, last in self.last_seen.items():
            healthy = (now - last) <= self.timeout
            self.status[cid] = healthy
            if not healthy:
                result.append(cid)
        return result
