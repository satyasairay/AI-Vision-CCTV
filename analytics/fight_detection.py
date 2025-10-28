"""Fight or violence detection heuristics.

This module provides a very simple and heuristic‑based fight/violence
detection mechanism. Detecting aggressive behaviour from CCTV footage is
an active area of research and typically requires sophisticated
action‑recognition models (e.g. 3D CNNs or transformers trained on
surveillance datasets). In this offline environment we cannot deploy
state‑of‑the‑art models, so instead we implement a basic heuristic:

* Track changes in the area of person bounding boxes over time.
* If the area fluctuates rapidly beyond a threshold within a short
  time window, we consider this indicative of aggressive movement.

This approach is extremely naive and intended only as a placeholder. In
a production system it should be replaced with a properly trained
violence/action classifier. The API is designed to facilitate such a
replacement.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple


class ViolenceDetector:
    """Detects potential fight or violence events based on bounding box changes.

    Parameters
    ----------
    window : int, optional
        Number of recent frames to consider for computing area changes. A
        shorter window reacts more quickly to changes but may produce
        more false positives.
    change_threshold : float, optional
        Relative change in area required to flag an event. The default of
        1.5 means the area must increase or decrease by 150% within
        the window to trigger a warning.

    Notes
    -----
    The detector keeps a deque of recent areas for each track. On each
    update it computes the ratio of max to min area within the deque. If
    this ratio exceeds `change_threshold` it returns True once. Further
    updates for the same track do not produce events until the area has
    stabilised again.
    """

    def __init__(self, window: int = 5, change_threshold: float = 1.5) -> None:
        self.window = window
        self.change_threshold = change_threshold
        # Track history of areas per track
        self.histories: Dict[int, deque] = {}
        # State to avoid repeated events
        self.triggered: Dict[int, bool] = {}

    def update(self, track_id: int, bbox: Tuple[int, int, int, int]) -> bool:
        """Update bounding box history and evaluate for violence.

        Parameters
        ----------
        track_id : int
            Identifier of the tracked person.
        bbox : tuple[int, int, int, int]
            Bounding box (x1, y1, x2, y2) for the person.

        Returns
        -------
        bool
            True if a potential fight/violence event is detected.
        """
        x1, y1, x2, y2 = bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        hist = self.histories.setdefault(track_id, deque(maxlen=self.window))
        hist.append(area)
        # If we don't have enough history yet, no detection
        if len(hist) < self.window:
            return False
        # Compute ratio of max to min area in the window
        max_area = max(hist)
        min_area = min(hist)
        # Avoid division by zero
        ratio = max_area / max(1.0, min_area)
        if ratio > self.change_threshold and not self.triggered.get(track_id, False):
            self.triggered[track_id] = True
            return True
        # Reset trigger if ratio goes back to near 1
        if ratio < 1.1:
            self.triggered[track_id] = False
        return False