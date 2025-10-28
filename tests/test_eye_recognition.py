from __future__ import annotations

import numpy as np
import pytest

from recognition.eye_recognition import EyeRecognition


@pytest.mark.parametrize("color", [64, 192])
def test_eye_embedding_not_none(color: int) -> None:
    recogniser = EyeRecognition(device="cpu", threshold=0.1)
    sample = np.full((80, 160, 3), fill_value=color, dtype=np.uint8)
    embedding = recogniser.embed(sample)
    assert embedding is not None
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)


def test_eye_recognition_identifies_enrolled_identity() -> None:
    recogniser = EyeRecognition(device="cpu", threshold=0.1)
    alice_img = np.full((80, 160, 3), fill_value=220, dtype=np.uint8)
    bob_img = np.full((80, 160, 3), fill_value=40, dtype=np.uint8)
    recogniser.enroll("Alice", recogniser.embed(alice_img))
    recogniser.enroll("Bob", recogniser.embed(bob_img))

    identity, score = recogniser.identify(alice_img)
    assert identity == "Alice"
    assert score > 0.1
