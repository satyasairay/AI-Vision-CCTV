from __future__ import annotations

import numpy as np

from tracking.deep_sort import DeepSortTracker


def test_tracker_consistent_ids() -> None:
    tracker = DeepSortTracker(max_age=5, n_init=1, iou_threshold=0.1)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    detections = [(10, 10, 60, 60, 0.9)]
    tracks = tracker.update(detections, frame=frame)
    assert len(tracks) == 1
    track_id = tracks[0][0]

    moved_detections = [(20, 12, 70, 62, 0.88)]
    tracks2 = tracker.update(moved_detections, frame=frame)
    assert len(tracks2) == 1
    assert tracks2[0][0] == track_id
