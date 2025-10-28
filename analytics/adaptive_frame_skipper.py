"""Adaptive frame skipping to conserve resources.

In static scenes with little or no activity, processing every frame can be
wasteful. This module implements a simple adaptive frame skipper that
skips a number of frames after a configurable number of consecutive
idle frames (frames with no detections). When activity resumes, the
skipper resets and processes frames normally.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveFrameSkipper:
    """Adaptive frame skipper state."""
    idle_count: int = 0
    skip_until: int = 0
    idle_threshold: int = 30  # number of idle frames before skipping begins
    skip_frames: int = 10  # number of frames to skip after idle threshold

    def should_skip(self, frame_index: int) -> bool:
        """Return True if the current frame should be skipped."""
        return frame_index < self.skip_until

    def update(self, frame_index: int, has_activity: bool) -> None:
        """Update the state based on whether activity was detected."""
        if has_activity:
            self.idle_count = 0
            self.skip_until = frame_index  # ensure no skipping
            return
        # No activity detected
        self.idle_count += 1
        if self.idle_count >= self.idle_threshold:
            # Start skipping frames
            self.skip_until = frame_index + self.skip_frames