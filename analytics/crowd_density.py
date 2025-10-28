"""Crowd density monitoring.

This module defines a `CrowdDensityMonitor` that counts the number of
persons detected within a specified area of the frame. When the count
exceeds a configurable threshold, a crowding event can be generated.
"""

from __future__ import annotations

from typing import Tuple, List


class CrowdDensityMonitor:
    """Monitor crowd density within a rectangular zone."""

    def __init__(self, zone: Tuple[int, int, int, int], threshold: int) -> None:
        """
        Parameters
        ----------
        zone : tuple
            The rectangular area (x1, y1, x2, y2) in which to count persons.
        threshold : int
            Maximum allowed number of persons before triggering a crowd event.
        """
        self.zone = zone
        self.threshold = threshold

    @staticmethod
    def _inside_zone(bbox: Tuple[int, int, int, int], zone: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        zx1, zy1, zx2, zy2 = zone
        # Use centroid of bbox for membership test
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    def count(self, person_bboxes: List[Tuple[int, int, int, int]]) -> bool:
        """Count persons inside the zone and return True if crowd detected.

        Parameters
        ----------
        person_bboxes : list of tuples
            Bounding boxes of detected persons.

        Returns
        -------
        crowd : bool
            ``True`` if the number of persons inside the zone exceeds the
            threshold.
        """
        count = 0
        for bbox in person_bboxes:
            if self._inside_zone(bbox, self.zone):
                count += 1
                if count > self.threshold:
                    return True
        return False