"""Local file camera adapter.

This adapter wraps OpenCV's `VideoCapture` to read frames from a local video
file. It's useful for development and testing when live camera streams are
unavailable. The interface mirrors that of the RTSP adapter.
"""

from __future__ import annotations

import cv2
from typing import Optional


class LocalFileCamera:
    """Adapter for local video files."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the video file for reading."""
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.filepath)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.filepath}")

    def read(self):  # -> Tuple[bool, Any]
        """Read a single frame from the file."""
        if self.capture is None:
            raise RuntimeError("LocalFileCamera: file not opened. Call open() first.")
        return self.capture.read()

    def release(self) -> None:
        """Release the video file."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
