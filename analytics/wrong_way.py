"""Wrong‑way detection.

This module implements a simple `WrongWayDetector` to flag objects
moving against an expected travel direction. It computes the angle
between an object's movement vector and the expected direction vector.
If the cosine of the angle is negative (i.e., angle > 90 degrees),
the object is considered moving the wrong way. You can adjust the
threshold for how strict the detection should be.

Use the detector by calling `update(track_id, bbox, timestamp)`. It
returns a boolean indicating a wrong‑way movement when sufficient
history is available.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional

import math


class WrongWayDetector:
    """Detect wrong‑way movement relative to an expected direction."""

    def __init__(self, expected_direction: Tuple[float, float] = (1.0, 0.0), threshold: float = 0.0) -> None:
        """
        Parameters
        ----------
        expected_direction : tuple, default (1.0, 0.0)
            A 2D vector representing the allowed travel direction. It will
            be normalized internally.
        threshold : float, default 0.0
            Cosine threshold for wrong‑way detection. Values below 0
            indicate a vector pointing more than 90° away from the expected
            direction. Increase threshold toward 1.0 for stricter checks.
        """
        ex, ey = expected_direction
        norm = math.hypot(ex, ey)
        self.expected = (ex / norm if norm else 1.0, ey / norm if norm else 0.0)
        self.threshold = threshold
        # history maps track ID to (centroid_x, centroid_y)
        self.history: Dict[int, Tuple[float, float]] = {}

    @staticmethod
    def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, track_id: int, bbox: Tuple[int, int, int, int]) -> Optional[bool]:
        """Update movement direction and check for wrong‑way.

        Parameters
        ----------
        track_id : int
            Identifier of the tracked object.
        bbox : tuple
            Bounding box (x1, y1, x2, y2) in pixels.

        Returns
        -------
        wrong_way : bool or None
            ``True`` if the object is detected moving the wrong way, ``False``
            if moving correctly, or ``None`` if insufficient history.
        """
        cx, cy = self._centroid(bbox)
        if track_id not in self.history:
            self.history[track_id] = (cx, cy)
            return None
        prev_cx, prev_cy = self.history[track_id]
        dx, dy = cx - prev_cx, cy - prev_cy
        self.history[track_id] = (cx, cy)
        # Normalise movement vector
        norm = math.hypot(dx, dy)
        if norm == 0:
            return None
        mv = (dx / norm, dy / norm)
        # Dot product with expected direction gives cosine of angle
        dot = mv[0] * self.expected[0] + mv[1] * self.expected[1]
        # Compare to threshold; wrong way if below threshold
        if dot < self.threshold:
            return True
        return False