"""Mask classification module.

This module defines a `MaskClassifier` class responsible for determining
whether a detected person's face is wearing a mask. In production, this
would load a lightweight convolutional neural network (e.g., a MobileNet
variant) trained on face images with and without masks. For environments
where no model is available, the classifier returns an "unknown" label.

Example usage::

    from road_security.detection.mask_classifier import MaskClassifier
    classifier = MaskClassifier(model_path="models/mask_classifier.pt")
    label, confidence = classifier.classify(face_image)

When a valid model path is provided, the classifier will load the
model and use it to predict whether the face is masked or not. If the
model cannot be loaded, `classify` returns ("unknown", 0.0).
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


class MaskClassifier:
    """Classify whether a face image shows a person wearing a mask.

    Parameters
    ----------
    model_path : str, optional
        Path to a PyTorch model file containing the weights for the
        mask/no-mask classifier. If not provided or loading fails,
        classification will return "unknown".
    device : str, optional
        Device on which to run inference (e.g., "cpu" or "cuda").
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self.model: Optional[object] = None
        # Attempt to load a PyTorch model if a path is provided
        if model_path:
            try:
                import torch
                # Example: model architecture should be defined elsewhere. For now, we load
                # a scripted/serialized model directly.
                self.model = torch.jit.load(model_path, map_location=device)
                self.model.eval()
            except Exception:
                # If loading fails, leave model as None
                self.model = None

    def classify(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Classify a face image as masked or not.

        Parameters
        ----------
        face_image : ndarray
            The cropped face region in BGR format.

        Returns
        -------
        label : str
            One of "mask", "no_mask", or "unknown".
        confidence : float
            Confidence of the prediction in [0, 1]. Returns 0.0 if
            classification cannot be performed.
        """
        # If no model is loaded, return unknown
        if self.model is None:
            return "unknown", 0.0
        try:
            import cv2
            import torch
            # Preprocess the image: convert BGR to RGB, resize to 224x224,
            # normalize to [0,1], and convert to tensor with shape (1, 3, H, W)
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (224, 224)).astype("float32") / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor)
                # Assume the model outputs logits with two elements: [no_mask_logit, mask_logit]
                probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
            # Determine label and confidence
            no_mask_prob, mask_prob = float(probs[0]), float(probs[1])
            if mask_prob > no_mask_prob:
                return "mask", mask_prob
            else:
                return "no_mask", no_mask_prob
        except Exception:
            # Fallback to unknown on any failure
            return "unknown", 0.0