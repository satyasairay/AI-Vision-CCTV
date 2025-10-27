"""Loitering and dwell time monitoring.

This module provides a `DwellTimeMonitor` class that tracks how long
objects remain within a defined zone. When an object's dwell time
exceeds a threshold, a loitering event can be triggered. The monitor
updates its internal timers on each frame and should be called for
every detected object.

Define a zone as a rectangle in pixel coordinates. The module offers a
helper function to determine if a point lies within the zone. To use,
initialize with a dwell threshold and call `update(track_id, centroid,
timestamp)` per frame.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional


class DwellTimeMonitor:
    """Monitor how long objects remain within a zone."""

    def __init__(self, zone: Tuple[int, int, int, int], threshold: float) -> None:
        """
        Parameters
        ----------
        zone : tuple
            The rectangular zone defined as (x1, y1, x2, y2) in pixels.
        threshold : float
            Dwell time threshold in seconds; exceeding this will trigger an event.
        """
        self.zone = zone
        self.threshold = threshold
        # Map track ID to (entry_time, last_seen_time)
        self.timers: Dict[int, Tuple[float, float]] = {}

    @staticmethod
    def _inside_zone(point: Tuple[float, float], zone: Tuple[int, int, int, int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2

    def update(self, track_id: int, centroid: Tuple[float, float], timestamp: float) -> Optional[float]:
        """Update dwell time for a track and return dwell duration if over threshold.

        Parameters
        ----------
        track_id : int
            Identifier of the tracked object.
        centroid : tuple
            The centroid coordinates (x, y) in pixels.
        timestamp : float
            Current time in seconds.

        Returns
        -------
        dwell_time : float or None
            The time spent in the zone if it exceeds the threshold; otherwise ``None``.
        """
        if not self._inside_zone(centroid, self.zone):
            # If outside the zone, reset the timer for this track
            if track_id in self.timers:
                del self.timers[track_id]
            return None
        # Inside zone
        if track_id not in self.timers:
            # Record entry time
            self.timers[track_id] = (timestamp, timestamp)
            return None
        entry_time, _ = self.timers[track_id]
        self.timers[track_id] = (entry_time, timestamp)
        dwell_time = timestamp - entry_time
        if dwell_time >= self.threshold:
            return dwell_time
        return None