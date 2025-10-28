"""Streamlit-based calibration wizard for zones and stop lines."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import yaml
from streamlit_drawable_canvas import st_canvas


def _load_image(image_bytes: bytes) -> np.ndarray:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported image format.")
    return image


def _load_from_path(path: str) -> np.ndarray:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)
    if path_obj.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(path_obj))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to grab frame from video {path_obj}")
        return frame
    image = cv2.imread(str(path_obj))
    if image is None:
        raise RuntimeError(f"Failed to read image {path_obj}")
    return image


def _rect_to_coords(obj: dict) -> Tuple[int, int, int, int]:
    x = int(obj.get("left", 0))
    y = int(obj.get("top", 0))
    w = int(obj.get("width", 0))
    h = int(obj.get("height", 0))
    return (x, y, x + w, y + h)


def _line_to_coords(obj: dict) -> Tuple[int, int, int, int]:
    x1 = int(obj.get("x1", 0))
    y1 = int(obj.get("y1", 0))
    x2 = int(obj.get("x2", 0))
    y2 = int(obj.get("y2", 0))
    return (x1, y1, x2, y2)


def main() -> None:
    st.set_page_config(page_title="AI-Vision-CCTV Calibration", layout="wide")
    st.title("Camera Calibration Wizard")

    st.sidebar.header("Configuration")
    config_path = st.sidebar.text_input("Config YAML", value="configs/default.yaml")
    sample_frame_path = st.sidebar.text_input("Sample frame/video path", "")
    uploaded_file = st.sidebar.file_uploader("Or upload a frame", type=["jpg", "png", "jpeg"])

    bg_image: np.ndarray | None = None
    if uploaded_file is not None:
        bg_image = _load_image(uploaded_file.getvalue())
    elif sample_frame_path:
        try:
            bg_image = _load_from_path(sample_frame_path)
        except Exception as exc:
            st.sidebar.error(f"Failed to load frame: {exc}")

    if bg_image is None:
        st.info("Provide a sample frame (upload or path) to begin calibration.")
        return

    display_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    st.write("Draw rectangles (dwell, crowd, privacy) and optional lines (stop-line).")
    st.caption(
        "Order suggestion: 1) dwell zone, 2) crowd zone, 3+) privacy zones. Use line tool for stop-line."
    )

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=display_image,
        height=display_image.shape[0],
        width=display_image.shape[1],
        drawing_mode="transform",
        key="calibration_canvas",
        display_toolbar=True,
    )

    objects = canvas_result.json_data["objects"] if canvas_result.json_data else []
    rects: List[Tuple[int, int, int, int]] = []
    lines: List[Tuple[int, int, int, int]] = []
    for obj in objects:
        obj_type = obj.get("type")
        if obj_type == "rect":
            rects.append(_rect_to_coords(obj))
        elif obj_type == "line":
            lines.append(_line_to_coords(obj))

    st.subheader("Detected Shapes")
    st.write(
        {
            "rectangles": rects,
            "lines": lines,
        }
    )

    if st.button("Apply to config") and config_path:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            st.error(f"Config path {cfg_path} not found.")
        else:
            with cfg_path.open("r") as f:
                config = yaml.safe_load(f) or {}

            analytics = config.setdefault("analytics", {})

            if rects:
                analytics.setdefault("dwell_time", {})["zone"] = list(rects[0])
            if len(rects) >= 2:
                analytics.setdefault("crowd_density", {})["zone"] = list(rects[1])
            if len(rects) > 2:
                analytics.setdefault("privacy", {}).setdefault("no_record_zones", [])
                analytics["privacy"]["no_record_zones"] = [list(r) for r in rects[2:]]
            if lines:
                analytics.setdefault("stop_line", {})["line"] = list(lines[0])

            config["analytics"] = analytics
            with cfg_path.open("w") as f:
                yaml.safe_dump(config, f)
            st.success(f"Calibration values written to {cfg_path}")


if __name__ == "__main__":
    main()
