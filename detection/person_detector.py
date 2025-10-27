"""Person detection and mask classification module.

This detector identifies persons in a frame and can optionally classify
whether they are wearing a face mask. The implementation should load a
pretrained model (such as a YOLO model or a dedicated face mask classifier)
from local weights. The `detect` method returns bounding boxes, class
labels, and confidence scores.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class PersonDetector:
    """Detect persons and mask status in an image frame."""

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """Detect persons and classify mask usage.

        Parameters
        ----------
        frame : ndarray
            The input image in BGR format.

        Returns
        -------
        detections : list of tuples
            Each tuple is `(x1, y1, x2, y2, label, score)` representing
            the bounding box coordinates, label (e.g., "mask" or "no_mask"),
            and confidence score. Currently returns an empty list.
        """
        # Placeholder implementation – replace with actual model inference.
        return []
