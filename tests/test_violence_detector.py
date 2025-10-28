from __future__ import annotations

import numpy as np

from analytics.fight_detection import ViolenceDetector


def test_violence_detector_processes_frames() -> None:
    detector = ViolenceDetector(window=4, threshold=0.99, stride=1, device="cpu")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = (50, 50, 150, 150)
    result = False
    for idx in range(4):
        frame[:, :, 0] = (idx * 10) % 255
        result = detector.update(frame, bbox, track_id=1)
    assert result in {True, False}
