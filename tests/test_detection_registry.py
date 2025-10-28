from __future__ import annotations

from detection.registry import build_detector


def test_vehicle_detector_registry_default() -> None:
    detector = build_detector("vehicle", "auto", model_path=None, device="cpu")
    detections = detector.detect  # type: ignore[attr-defined]
    assert callable(detections)


def test_person_detector_registry_default() -> None:
    detector = build_detector("person", "hog", model_path=None, device="cpu", mask_model_path=None)
    assert hasattr(detector, "detect")
