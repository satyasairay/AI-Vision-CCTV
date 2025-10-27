"""Detection package.

This package contains modules for detecting objects of interest in video
frames. It currently provides vehicle and person detection using
pretrained deep learning models (e.g., YOLO). Each detector exposes
a `detect` method that accepts an image and returns bounding boxes and
class labels.
"""

from .vehicle_detector import VehicleDetector
from .person_detector import PersonDetector

__all__ = [
    "VehicleDetector",
    "PersonDetector",
]
