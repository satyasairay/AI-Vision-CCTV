"""Periocular/eye region recognition module.

In situations where individuals wear masks, full face recognition is
impractical. This module defines `EyeRecognition` for performing
biometric recognition using only the eye region (periocular area). It
serves as a placeholder for integrating a lightweight neural network
trained on periocular images. Use the `identify` method to compare
extracted eye regions against a gallery of known identities.
"""

from __future__ import annotations

from typing import Tuple, Optional, List

import numpy as np


class EyeRecognition:
    """Identify individuals using periocular information."""

    def __init__(self, model_path: Optional[str] = None, database: Optional[dict] = None) -> None:
        """
        Parameters
        ----------
        model_path : str, optional
            Path to the periocular recognition model weights. If `None`,
            a default model may be loaded during setup.
        database : dict, optional
            Mapping from identity names to feature vectors representing
            known individuals. The database can be populated from
            enrollment images.
        """
        self.model_path = model_path
        self.database: dict[str, np.ndarray] = database or {}

    def identify(self, eye_image: np.ndarray) -> Tuple[str, float]:
        """Identify the person from an eye region image.

        Parameters
        ----------
        eye_image : ndarray
            The cropped eye region from the original frame.

        Returns
        -------
        identity : str
            The best matching identity name, or an empty string if none match.
        confidence : float
            Confidence score of the match (0–1). Returns 0.0 for now.
        """
        # If no database is provided, cannot identify
        if not self.database:
            return "", 0.0
        # Extract a simple feature vector from the eye region
        try:
            import cv2

            # Convert to grayscale and resize to a fixed size
            gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            feature = resized.flatten().astype("float32")
            # Normalize the feature vector
            norm = np.linalg.norm(feature)
            feature = feature / norm if norm > 0 else feature
        except Exception:
            return "", 0.0

        # Compare to each entry in the database
        best_identity = ""
        best_score = 0.0
        for identity, db_feature in self.database.items():
            # Ensure database feature is normalized
            db_feature_norm = db_feature / (np.linalg.norm(db_feature) or 1.0)
            # Compute cosine similarity (dot product) between features
            score = float(np.dot(feature, db_feature_norm))
            if score > best_score:
                best_identity = identity
                best_score = score
        # Threshold can be adjusted based on desired strictness; if below 0.5, return unknown
        if best_score < 0.5:
            return "", 0.0
        return best_identity, best_score
