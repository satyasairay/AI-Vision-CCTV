"""Duplicate license plate detection.

This module defines a `DuplicatePlateDetector` class that tracks
observations of license plates over time. If the same plate appears
again within a specified time window, an event can be triggered.
Duplicate detection is useful for identifying surveillance or
reconnaissance behavior (e.g., the same vehicle circling a facility).
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import time


class DuplicatePlateDetector:
    """Detect duplicate license plates within a given time window."""

    def __init__(self, window_seconds: float = 300.0) -> None:
        """
        Parameters
        ----------
        window_seconds : float, default 300.0
            Time window in seconds during which repeated appearances of
            the same plate will be considered duplicates.
        """
        self.window = window_seconds
        # Map plate string to last seen timestamp
        self.last_seen: Dict[str, float] = {}

    def update(self, plate: str) -> bool:
        """Record a plate observation and return True if it's a duplicate.

        Parameters
        ----------
        plate : str
            The license plate text.

        Returns
        -------
        is_duplicate : bool
            ``True`` if the plate has been seen within the time window,
            otherwise ``False``.
        """
        if not plate:
            return False
        now = time.time()
        duplicate = False
        if plate in self.last_seen:
            if now - self.last_seen[plate] <= self.window:
                duplicate = True
        self.last_seen[plate] = now
        return duplicate