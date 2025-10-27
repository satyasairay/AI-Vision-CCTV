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
        # store parameters and provide a stub implementation.
        self.model_path = model_path
        self.device = device

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
        # Placeholder implementation – replace with actual model inference.
        # When implementing, convert frame to RGB, preprocess for the model,
        # run inference, then postprocess to produce bounding boxes.
        return []
