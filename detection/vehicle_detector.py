"""Vehicle detection module.

This module defines a `VehicleDetector` class that uses a pre‑trained object
detection model to identify vehicles in a frame. For an initial
implementation, you can load weights from a YOLO‑based model stored locally
and perform inference via PyTorch. The `detect` method returns a list of
bounding boxes and confidence scores for detected vehicles.

Note: To keep this repository fully local, download model weights in the
`scripts/setup.sh` script and store them under `models/` or a similar
directory. See the README for setup instructions.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import cv2


class VehicleDetector:
    """Detect vehicles in an image frame."""

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        model_path : str, optional
            Path to the pretrained model weights. Can be `None` if a
            default model is shipped with the package.
        device : str, default "cpu"
            Device on which to run the model (e.g., "cpu" or "cuda").
        """
        # In a full implementation, load the model here. For now, we
        # implement a simple background‑subtraction based detector that
        # identifies moving objects. This avoids downloading heavy
        # pretrained weights and works for demonstration purposes.
        self.model_path = model_path
        self.device = device
        # Initialize a background subtractor. Parameters can be tuned via config.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect vehicles in a frame.

        Parameters
        ----------
        frame : ndarray
            The input image in BGR format.

        Returns
        -------
        detections : list of tuples
            Each tuple is `(x1, y1, x2, y2, score)` representing the
            bounding box coordinates and confidence score for a detected
            vehicle. At this stage, this method returns an empty list.
        """
        # Apply background subtraction to get foreground mask
        fg_mask = self.bg_subtractor.apply(frame)
        # Threshold the mask to binarize
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_DILATE, kernel, iterations=2)
        # Find contours of moving objects
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Tuple[int, int, int, int, float]] = []
        for cnt in contours:
            # Ignore small areas
            area = cv2.contourArea(cnt)
            if area < 500:  # tune threshold based on expected object size
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x + w, y + h
            detections.append((x1, y1, x2, y2, 1.0))
        return detections
