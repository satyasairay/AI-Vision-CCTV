"""Cross‑camera re‑identification utilities.

This module provides a minimal implementation of appearance‑based
re‑identification (ReID) for vehicles or persons across multiple
cameras. In a comprehensive system, ReID models learn embeddings that
are invariant to viewpoint, lighting and camera calibration. Here we
approximate this by computing simple colour histograms, which are then
compared using a histogram intersection metric. This approach is very
limited but demonstrates how a feature extractor and matcher could be
structured.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ReIdentifier:
    """Maintains a gallery of features and performs matching."""

    def __init__(self, gallery: Optional[Dict[str, np.ndarray]] = None) -> None:
        # Gallery maps identity names to feature vectors
        self.gallery: Dict[str, np.ndarray] = gallery or {}

    @staticmethod
    def extract_feature(img: np.ndarray) -> np.ndarray:
        """Compute a simple colour histogram feature vector.

        Parameters
        ----------
        img : numpy.ndarray
            Input image region in BGR format.

        Returns
        -------
        numpy.ndarray
            Flattened histogram feature (normalized).
        """
        # Resize to a fixed size for consistency
        resized = cv2.resize(img, (64, 64))
        # Compute histograms for each channel
        hist_b = cv2.calcHist([resized], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([resized], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([resized], [2], None, [16], [0, 256])
        feature = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        # Normalize
        feature = feature / (np.sum(feature) + 1e-8)
        return feature

    @staticmethod
    def similarity(f1: np.ndarray, f2: np.ndarray) -> float:
        """Compute histogram intersection similarity between two features."""
        return float(np.sum(np.minimum(f1, f2)))

    def match(self, img: np.ndarray, threshold: float = 0.5) -> Optional[str]:
        """Match an image against the gallery.

        Parameters
        ----------
        img : numpy.ndarray
            Image to match.
        threshold : float
            Minimum similarity required to return a match. Values in
            [0,1], where 1 is a perfect match.

        Returns
        -------
        Optional[str]
            Identity name if matched above threshold, else None.
        """
        feature = self.extract_feature(img)
        best_name = None
        best_score = 0.0
        for name, gallery_feature in self.gallery.items():
            score = self.similarity(feature, gallery_feature)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name is not None and best_score >= threshold:
            return best_name
        return None