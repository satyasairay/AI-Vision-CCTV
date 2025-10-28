"""Camera health monitoring.

This module tracks the liveness of camera feeds. If no frame has been
received for a configurable period, it flags the feed as down. This
information can be used by higher layers (e.g. rule engine or
dashboard) to alert operators to hardware issues.
"""

from __future__ import annotations

import time
from typing import Dict, Optional


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

    def update(self, camera_id: str) -> None:
        """Record that a frame has been received from a camera."""
        self.last_seen[camera_id] = time.time()

    def is_down(self, camera_id: str) -> bool:
        """Return True if the camera has not sent a frame within timeout."""
        last = self.last_seen.get(camera_id)
        if last is None:
            # No record yet, treat as down until first frame
            return True
        return (time.time() - last) > self.timeout

    def get_down_cameras(self) -> list[str]:
        """Return a list of camera IDs that are currently down."""
        now = time.time()
        return [cid for cid, t_last in self.last_seen.items() if (now - t_last) > self.timeout]