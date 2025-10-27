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
    """Detect persons and (optionally) mask status in an image frame.

    This implementation uses OpenCV's HOG descriptor with a pretrained SVM
    classifier for pedestrian detection. It does not perform mask
    classification; all detections are labelled as "person". When a more
    advanced model is available (e.g., YOLO with mask/no-mask classes), it
    can be loaded via the `model_path` argument.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        # Initialize HOG descriptor for person detection. If more advanced
        # models are provided via `model_path`, they can be loaded here.
        try:
            import cv2

            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception:
            self.hog = None

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
            the bounding box coordinates, label (e.g., "person"),
            and confidence score.
        """
        detections: List[Tuple[int, int, int, int, str, float]] = []
        if self.hog is None:
            return detections
        try:
            import cv2

            # The HOG detector expects grayscale or color images with people
            # upright; detectMultiScale returns bounding boxes and weights
            rects, weights = self.hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.05,
            )
            for (x, y, w, h), weight in zip(rects, weights):
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                label = "person"
                score = float(weight)
                detections.append((x1, y1, x2, y2, label, score))
        except Exception:
            # In case OpenCV is not installed, return no detections
            pass
        return detections
