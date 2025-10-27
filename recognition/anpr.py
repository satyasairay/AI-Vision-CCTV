"""Automatic Number Plate Recognition (ANPR).

This module defines the `ANPR` class responsible for extracting and
recognizing license plates from vehicle images. It uses an OCR engine
such as EasyOCR or PaddleOCR, which should be installed locally via
the setup script. The `recognize` method returns the recognized text
and a confidence score.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


class ANPR:
    """Recognize license plates from vehicle images using an OCR engine."""

    def __init__(self, ocr_engine: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        ocr_engine : str, optional
            The OCR engine to use (currently only "easyocr" is supported). If
            `None`, EasyOCR will be used by default.
        """
        self.ocr_engine = ocr_engine or "easyocr"
        self.reader: Optional[object] = None
        if self.ocr_engine == "easyocr":
            try:
                import easyocr  # type: ignore
                # Initialize the reader for English characters. GPU usage is disabled for portability.
                self.reader = easyocr.Reader(["en"], gpu=False)
            except Exception:
                # If EasyOCR is unavailable, leave reader as None; recognition will return empty strings
                self.reader = None

    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """Recognize text from a cropped license plate image.

        Parameters
        ----------
        plate_image : ndarray
            The cropped region containing the license plate.

        Returns
        -------
        text : str
            The recognized license plate characters.
        confidence : float
            Confidence score of the recognition (0–1). Returns 0.0 for now.
        """
        # If OCR engine isn't initialized, return empty result
        if self.reader is None:
            return "", 0.0
        try:
            # Convert image to grayscale for better OCR performance
            import cv2
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            results = self.reader.readtext(gray)
            # results is a list of tuples: (bbox, text, confidence)
            if not results:
                return "", 0.0
            # Choose the result with the highest confidence
            best = max(results, key=lambda x: x[2])
            text = best[1].strip()
            confidence = float(best[2])
            return text, confidence
        except Exception:
            return "", 0.0
