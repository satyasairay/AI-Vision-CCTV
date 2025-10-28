"""Utility to generate a placeholder sample video for demos/tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "sample_data"
    out_dir.mkdir(exist_ok=True)
    video_path = out_dir / "sample_video.mp4"

    width, height = 640, 480
    fps = 10
    frame_count = 120

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {video_path}")

    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Frame {idx:03d}",
            (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        x_offset = (idx * 5) % width
        cv2.rectangle(frame, (x_offset, 150), (x_offset + 120, 300), (255, 0, 0), -1)
        writer.write(frame)

    writer.release()
    print(f"Sample video written to {video_path}")


if __name__ == "__main__":
    main()
