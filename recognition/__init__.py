"""Recognition package.

This package provides functionality for recognizing textual and biometric
information from detected objects. Modules include automatic number
plate recognition (ANPR) for reading license plates and eye region
recognition for identifying masked individuals.
"""

from .anpr import ANPR
from .eye_recognition import EyeRecognition

__all__ = ["ANPR", "EyeRecognition"]
