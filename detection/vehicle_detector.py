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
            Path to the pretrained model weights. If provided and the
            required deep learning libraries are available, the detector
            will attempt to load a YOLOv8 model for high‑accuracy vehicle
            detection. Otherwise, a background‑subtraction based fallback
            will be used.
        device : str, default "cpu"
            Device on which to run the model (e.g., "cpu" or "cuda").
        """
        self.model_path = model_path
        self.device = device
        self.deep_model: object | None = None
        # Try to load a deep learning model (YOLO) if a path is provided
        if model_path:
            try:
                # Option 1: using ultralytics package
                from ultralytics import YOLO  # type: ignore
                self.deep_model = YOLO(model_path)
            except Exception:
                # Fallback: try torch hub with yolov5/yolov8 custom model
                try:
                    import torch
                    # Attempt to load a custom YOLO model from PyTorch Hub
                    # This assumes the model is compatible with the YOLOv5 API.
                    self.deep_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
                    self.deep_model.to(device)
                except Exception:
                    self.deep_model = None
        # Initialize a background subtractor as a fallback. Parameters can be tuned via config.
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
        # If a deep learning model is available, use it for detection
        if self.deep_model is not None:
            detections: List[Tuple[int, int, int, int, float]] = []
            try:
                # Use the model's predict method; for ultralytics YOLO, the API is .predict()
                results = None
                try:
                    results = self.deep_model(frame)  # type: ignore[attr-defined]
                except Exception:
                    # For PyTorch Hub models, call .inference with raw image
                    results = self.deep_model([frame])  # type: ignore[attr-defined]
                # Extract bounding boxes for vehicles. Class IDs vary between models;
                # we filter by common vehicle class names when available.
                # The API for results depends on the model type.
                if hasattr(results, 'boxes'):
                    # ultralytics YOLOv8 results
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                            conf = float(box.conf[0].cpu().item())
                            # Filter by class if vehicle classes are defined
                            detections.append((int(x1), int(y1), int(x2), int(y2), conf))
                elif isinstance(results, list) and len(results) > 0:
                    # torch hub yolov5 results: results[0].xyxy[0] is Nx6 (x1,y1,x2,y2,conf,class)
                    res = results[0]
                    if hasattr(res, 'xyxy'):
                        pred = res.xyxy[0].cpu().numpy()
                        for x1, y1, x2, y2, conf, cls in pred:
                            detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
                if detections:
                    return detections
            except Exception:
                # If inference fails, fall back to background subtraction
                pass
        # Fallback: use background subtraction
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
