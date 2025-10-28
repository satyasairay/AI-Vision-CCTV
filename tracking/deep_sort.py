"""Kalman-filter backed multi-object tracker inspired by Deep SORT.

This tracker keeps the dependency footprint small while providing many of
the robustness characteristics of Deep SORT: constant-velocity motion
model, Hungarian-based data association, and lightweight appearance
descriptors. It is suitable for single-machine deployments where
third-party tracker packages or GPU-heavy re-identification models are
undesirable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _tlbr_to_xyah(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Convert [x1, y1, x2, y2] box to centre x/y, aspect ratio, and height."""
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    aspect = w / h
    return np.array([cx, cy, aspect, h], dtype=float)


def _xyah_to_tlbr(state: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert state vector (x, y, a, h, vx, vy, va, vh) to tlbr box."""
    cx, cy, aspect, h = state[:4]
    w = max(1.0, aspect * h)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute intersection over union between two TLBR boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _cosine_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Return cosine distance in [0, 2]."""
    if a is None or b is None:
        return 1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


class _KalmanFilter:
    """Constant-velocity Kalman filter for bounding boxes."""

    ndim = 4

    def __init__(self) -> None:
        dt = 1.0
        self.motion_mat = np.eye(2 * self.ndim, dtype=float)
        for i in range(self.ndim):
            self.motion_mat[i, self.ndim + i] = dt
        self.update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=float)

        self.std_pos = np.array([1.0, 1.0, 0.1, 1.0], dtype=float)
        self.std_vel = np.array([10.0, 10.0, 1.0, 10.0], dtype=float)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8, dtype=float)
        mean[:4] = measurement
        covariance = np.diag(np.concatenate((self.std_pos ** 2, self.std_vel ** 2)))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        motion_cov = np.diag(np.concatenate((self.std_pos ** 2, self.std_vel ** 2)))
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        innovation_cov = np.diag(self.std_pos ** 2)
        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T + innovation_cov
        return projected_mean, projected_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ self.update_mat @ covariance
        return new_mean, new_covariance


@dataclass
class _Track:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    feature: Optional[np.ndarray] = None
    confirmed: bool = False
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def to_tlbr(self) -> Tuple[int, int, int, int]:
        return _xyah_to_tlbr(self.mean)


class DeepSortTracker:
    """Kalman/Hungarian tracker with basic appearance modelling."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        iou_threshold: float = 0.3,
        appearance_weight: float = 0.1,
    ) -> None:
        self.max_age = max_age
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight

        self._kf = _KalmanFilter()
        self._tracks: List[_Track] = []
        self._next_track_id = 1

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._next_track_id = 1

    # ------------------------------------------------------------------
    def update(
        self,
        detections: Sequence[Tuple[int, int, int, int, float]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int, int, int, int]]:
        """Update tracker with a list of detections for the current frame.

        Parameters
        ----------
        detections:
            Iterable of `(x1, y1, x2, y2, score)` bounding boxes.
        frame:
            Optional BGR frame used to extract lightweight colour histograms.
        """
        detections = list(detections)

        # Step 1: predict existing tracks forward.
        for track in self._tracks:
            track.mean, track.covariance = self._kf.predict(track.mean, track.covariance)
            track.age += 1
            track.time_since_update += 1

        # Step 2: compute appearance features if frame available.
        features: List[Optional[np.ndarray]] = [
            self._extract_feature(frame, det[:4]) for det in detections
        ]

        # Step 3: data association.
        matches, unmatched_tracks, unmatched_detections = self._associate(
            detections, features
        )

        # Step 4: update matched tracks.
        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            measurement = _tlbr_to_xyah(detections[det_idx][:4])
            track.mean, track.covariance = self._kf.update(track.mean, track.covariance, measurement)
            track.hits += 1
            track.time_since_update = 0
            track.feature = features[det_idx]
            track.history.append(track.to_tlbr())
            if not track.confirmed and track.hits >= self.n_init:
                track.confirmed = True

        # Step 5: age unmatched tracks and remove stale ones.
        survivors: List[_Track] = []
        for idx, track in enumerate(self._tracks):
            if idx in unmatched_tracks and track.time_since_update > self.max_age:
                continue
            survivors.append(track)
        self._tracks = survivors

        # Step 6: initialise new tracks.
        for det_idx in unmatched_detections:
            bbox = detections[det_idx][:4]
            measurement = _tlbr_to_xyah(bbox)
            mean, covariance = self._kf.initiate(measurement)
            track = _Track(
                track_id=self._next_track_id,
                mean=mean,
                covariance=covariance,
                hits=1,
                age=1,
                time_since_update=0,
                feature=features[det_idx],
                confirmed=self.n_init <= 1,
                history=[bbox],
            )
            self._tracks.append(track)
            self._next_track_id += 1

        # Step 7: return confirmed tracks.
        return [
            (track.track_id, *track.to_tlbr())
            for track in self._tracks
            if track.confirmed and track.time_since_update <= self.max_age
        ]

    # ------------------------------------------------------------------
    def _associate(
        self,
        detections: Sequence[Tuple[int, int, int, int, float]],
        features: Sequence[Optional[np.ndarray]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self._tracks:
            return [], [], list(range(len(detections)))

        cost_matrix = np.zeros((len(self._tracks), len(detections)), dtype=float)
        for t_idx, track in enumerate(self._tracks):
            track_bbox = track.to_tlbr()
            for d_idx, det in enumerate(detections):
                det_bbox = det[:4]
                iou = _iou(track_bbox, det_bbox)
                if iou <= 0:
                    cost = 1.0
                else:
                    appearance_cost = _cosine_distance(track.feature, features[d_idx])
                    cost = (1.0 - iou) + self.appearance_weight * appearance_cost
                cost_matrix[t_idx, d_idx] = cost

        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))

        for t_idx, d_idx in zip(row_idx, col_idx):
            if cost_matrix[t_idx, d_idx] >= (1.0 - self.iou_threshold + self.appearance_weight):
                continue
            matches.append((t_idx, d_idx))
            unmatched_tracks.discard(t_idx)
            unmatched_detections.discard(d_idx)

        return matches, sorted(unmatched_tracks), sorted(unmatched_detections)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_feature(
        frame: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        if frame is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        patch = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
