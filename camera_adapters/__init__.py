"""Camera adapters package.

This package contains modules that provide a uniform interface for capturing
frames from different camera sources (e.g. RTSP streams, local video files,
USB webcams). Each adapter exposes a `CameraAdapter` class with methods to
open the stream, read frames, and release resources.
"""

from .rtsp_camera import RTSPCamera
from .local_file import LocalFileCamera

__all__ = [
    "RTSPCamera",
    "LocalFileCamera",
]
