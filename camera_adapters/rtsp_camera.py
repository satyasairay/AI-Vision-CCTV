"""RTSP camera adapter.

This module provides a class for connecting to RTSP streams using OpenCV. The
adapter reads frames from the network stream and yields them for further
processing in the detection pipeline. If the stream disconnects or fails,
reconnection logic can be implemented here.
"""

from __future__ import annotations

import cv2
from typing import Optional


class RTSPCamera:
    """Adapter for RTSP video streams.

    Attributes
    ----------
    url : str
        The RTSP URL to connect to.
    capture : Optional[cv2.VideoCapture]
        The underlying OpenCV video capture object.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the RTSP stream for reading."""
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.url)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.url}")

    def read(self):  # -> Tuple[bool, Any]
        """Read a single frame from the RTSP stream.

        Returns
        -------
        ret : bool
            Whether a frame was successfully read.
        frame : ndarray
            The image frame in BGR format.
        """
        if self.capture is None:
            raise RuntimeError("RTSPCamera: stream not opened. Call open() first.")
        return self.capture.read()

    def release(self) -> None:
        """Release the RTSP stream."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
