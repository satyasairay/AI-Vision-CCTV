"""Weather and lighting adaptation utilities.

Detection algorithms often perform poorly under varying lighting or
weather conditions (e.g., low light, rain, fog). The goal of this
module is to provide a simple preprocessing hook that can be inserted
into the pipeline. It analyses frame brightness and applies basic
image adjustments such as histogram equalization or gamma correction
to improve contrast.

Users can extend this class or replace it entirely with a more
sophisticated implementation that integrates with hardware sensors or
external weather APIs. For example, you could adjust exposure based on
ambient light sensors or disable certain analytics during heavy rain.
"""

from __future__ import annotations

import cv2
import numpy as np


class WeatherAdapter:
    """Simple brightness‑based frame preprocessor.

    Parameters
    ----------
    brightness_threshold : float, optional
        A value between 0 and 255. If the mean intensity of the frame
        falls below this threshold, the frame will be gamma corrected.
    gamma : float, optional
        Gamma value to apply when the brightness is below the threshold.
    """

    def __init__(self, brightness_threshold: float = 80.0, gamma: float = 1.5) -> None:
        self.brightness_threshold = brightness_threshold
        self.gamma = gamma
        # Precompute gamma lookup table
        invGamma = 1.0 / max(1e-3, gamma)
        self.table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(
            "uint8"
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the frame based on brightness.

        If the frame is dark (mean pixel intensity below threshold), apply
        gamma correction to brighten it. Otherwise return the frame
        unchanged.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image in BGR format.

        Returns
        -------
        numpy.ndarray
            The preprocessed image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        if mean_intensity < self.brightness_threshold:
            # apply gamma correction
            return cv2.LUT(frame, self.table)
        return frame