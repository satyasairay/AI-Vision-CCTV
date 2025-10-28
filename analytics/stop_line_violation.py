"""Stop‑line and red‑light violation detection.

This module implements a simple stop‑line violation detector for vehicles. The
detector monitors object bounding boxes relative to a predefined line (or
rectangular zone) and triggers a violation event when a vehicle crosses
the line while the `red_light` flag is active. The detector maintains state
per track to avoid repeated events for the same crossing.

Usage
-----
```
from road_security.analytics.stop_line_violation import StopLineViolationDetector

# Define a stop line as (x1, y1, x2, y2) in pixel coordinates. The line is
# assumed horizontal and spans from (x1, y) to (x2, y).
detector = StopLineViolationDetector(line=(100, 400, 500, 400))

crossed = detector.update(track_id, bbox, red_light=True)
if crossed:
    # handle violation
    ...
```

The detector does not interpret traffic light states. The user must supply
the current `red_light` state (True for red, False otherwise). In a real
deployment, this could be wired to a physical signal or integrated with
a traffic‑signal controller API.
"""

from __future__ import annotations

from typing import Dict, Tuple


class StopLineViolationDetector:
    """Detects stop‑line / red‑light violations for tracked vehicles.

    Parameters
    ----------
    line : tuple[int, int, int, int]
        Coordinates of the stop line in the form (x1, y1, x2, y2). The
        detector considers the y‑coordinate of the line for crossing logic.

    Attributes
    ----------
    line_y : int
        The y‑coordinate of the horizontal stop line.
    triggered : Dict[int, bool]
        Keeps track of whether each track has already triggered a violation
        to avoid repeated events.
    """

    def __init__(self, line: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = line
        # We only need the y coordinate; assume horizontal line
        self.line_y: int = int((y1 + y2) / 2)
        self.triggered: Dict[int, bool] = {}

    def update(self, track_id: int, bbox: Tuple[int, int, int, int], red_light: bool = True) -> bool:
        """Update the detector with a new bounding box.

        Parameters
        ----------
        track_id : int
            Identifier of the tracked object.
        bbox : tuple[int, int, int, int]
            Bounding box coordinates (x1, y1, x2, y2) in pixel space.
        red_light : bool, optional
            Whether the red light is currently active. If False, crossings
            are ignored.

        Returns
        -------
        bool
            True if a stop‑line violation is detected for this update.
        """
        if not red_light:
            return False
        if self.triggered.get(track_id):
            return False
        x1, y1, x2, y2 = bbox
        # Consider the bottom of the bounding box
        bottom_y = y2
        # A crossing occurs when the bottom of the bbox goes below the line
        if bottom_y > self.line_y:
            self.triggered[track_id] = True
            return True
        return False