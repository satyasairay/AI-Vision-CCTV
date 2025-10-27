"""Speed estimation and violation detection.

This module defines a `SpeedEstimator` class that computes the approximate
speed of tracked objects based on bounding‑box centroids across
consecutive frames. A per‑pixel scaling factor and frame time interval
are used to estimate real‑world speed. When an object's speed exceeds
a configurable threshold, a speed violation event can be generated.

The estimator maintains state across frames for each track ID. To use
it, call `update(track_id, bbox, timestamp)` on every frame. The
returned speed is expressed in arbitrary units; calibrate the
`pixels_per_meter` setting to convert to km/h or mph.

Example::

    est = SpeedEstimator(pixels_per_meter=8.0)
    speed = est.update(track_id, (x1, y1, x2, y2), current_time)
    if speed and speed > speed_limit:
        # trigger over‑speed event

"""

from __future__ import annotations

from typing import Tuple, Dict, Optional

import math


class SpeedEstimator:
    """Estimate the speed of tracked objects from bounding boxes.

    Parameters
    ----------
    pixels_per_meter : float, optional
        Conversion factor from pixel distance to meters. Adjust this
        parameter based on the camera's calibration. Defaults to 1.0
        (i.e., speed in pixels per second).
    """

    def __init__(self, pixels_per_meter: float = 1.0) -> None:
        self.pixels_per_meter = pixels_per_meter
        # history maps track ID to (centroid_x, centroid_y, timestamp)
        self.history: Dict[int, Tuple[float, float, float]] = {}

    @staticmethod
    def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, track_id: int, bbox: Tuple[int, int, int, int], timestamp: float) -> Optional[float]:
        """Update the estimator with a new observation and return speed.

        Parameters
        ----------
        track_id : int
            Identifier of the tracked object.
        bbox : tuple
            Bounding box (x1, y1, x2, y2) of the object in pixels.
        timestamp : float
            Time of the current observation in seconds.

        Returns
        -------
        speed : float or None
            Estimated speed in meters per second. Returns ``None`` if
            this is the first observation of the track.
        """
        cx, cy = self._centroid(bbox)
        if track_id not in self.history:
            self.history[track_id] = (cx, cy, timestamp)
            return None
        prev_cx, prev_cy, prev_ts = self.history[track_id]
        dt = timestamp - prev_ts
        if dt <= 0:
            return None
        # distance in pixels
        dx = cx - prev_cx
        dy = cy - prev_cy
        dist_pixels = math.hypot(dx, dy)
        # convert to meters
        dist_meters = dist_pixels / self.pixels_per_meter
        speed_m_per_s = dist_meters / dt
        # update history
        self.history[track_id] = (cx, cy, timestamp)
        return speed_m_per_s