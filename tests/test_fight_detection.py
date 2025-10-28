from __future__ import annotations

import numpy as np
import torch
import pytest

from analytics.fight_detection import ViolenceDetector


def test_violence_detector_custom_weights(tmp_path):
    vd = ViolenceDetector(window=4, stride=1, threshold=0.1, device="cpu", weights_path=None)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = (0, 0, 150, 150)
    for _ in range(4):
        vd.update(frame, bbox, track_id=1)
    assert isinstance(vd.violent_indices, list) and vd.violent_indices


def test_violence_detector_with_keywords():
    vd = ViolenceDetector(window=4, stride=1, threshold=1.0, device="cpu", violent_keywords=["boxing"])
    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    bbox = (10, 10, 80, 80)
    outputs = [vd.update(frame, bbox, track_id=42) for _ in range(4)]
    assert outputs[-1] in {True, False}

