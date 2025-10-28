"""Privacy and compliance utilities.

This module provides helpers for applying privacy safeguards to video
frames. Key features include:

* **Blurring/Redaction**: For persons or vehicles not on a watchlist,
  blur their bounding boxes before storage or display. This ensures
  non‑target individuals remain anonymous.
* **No‑record zones**: Areas (polygons or rectangles) where recording
  and analytics should be disabled. Bounding boxes fully inside these
  zones are omitted from further processing.

The implementation here uses a simple mosaic blur to obscure faces or
plates. More sophisticated methods (e.g. homomorphic encryption or
in‑camera redaction) could be integrated via this interface.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple


class PrivacyManager:
    """Handles blurring and opt‑out zones."""

    def __init__(self, no_record_zones: List[Tuple[int, int, int, int]] | None = None) -> None:
        # Zones are axis‑aligned rectangles defined as (x1,y1,x2,y2)
        self.no_record_zones = no_record_zones or []

    @staticmethod
    def _apply_mosaic(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, kernel_size: int = 15) -> None:
        """Apply a mosaic blur to the region [x1:x2, y1:y2] in place."""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        h, w = roi.shape[:2]
        k = max(1, min(h, w) // kernel_size)
        # Downsample and upsample to create mosaic effect
        small = cv2.resize(roi, (w // k, h // k), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = blurred

    def in_no_record_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if a bounding box is fully contained in a no‑record zone."""
        x1, y1, x2, y2 = bbox
        for zx1, zy1, zx2, zy2 in self.no_record_zones:
            if x1 >= zx1 and y1 >= zy1 and x2 <= zx2 and y2 <= zy2:
                return True
        return False

    def redact_detections(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, str, float]], watchlist: List[str]) -> None:
        """Blur detections not on the watchlist or inside opt‑out zones.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame on which to apply blurring.
        detections : list of tuples
            Each detection is (x1, y1, x2, y2, label, confidence). For
            vehicles the label can be 'vehicle', for persons it may be
            'person', 'mask', 'no_mask'.
        watchlist : list of str
            A list of identifiers (e.g. licence plates or identities) that
            should not be blurred. If the detection's identifier is not
            present, its bounding box will be blurred.

        Note: the identifier extraction (e.g. plate text) must be
        performed separately by the caller. The function expects the
        caller to map each detection to an identifier.
        """
        # Example implementation: expects detections have an extra field at
        # index 5 with the identifier, which may be None.
        for det in detections:
            x1, y1, x2, y2, label, conf = det[:6]
            identifier = det[6] if len(det) > 6 else None
            if identifier and identifier in watchlist:
                continue
            # If inside no‑record zone or not on watchlist, blur
            if self.in_no_record_zone((x1, y1, x2, y2)) or (identifier is None):
                self._apply_mosaic(frame, int(x1), int(y1), int(x2), int(y2))