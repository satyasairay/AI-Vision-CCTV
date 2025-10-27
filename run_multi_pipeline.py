"""Run the road security pipeline for multiple cameras concurrently.

This module spawns a thread per camera defined in the configuration and
processes each stream independently. It reuses the same components from
`run_pipeline.py` but allows for concurrent execution. Events are logged
through a shared `EventLogger`, and the latest processed frame for each
camera is saved to disk for dashboard streaming.

Usage::

    python -m road_security.run_multi_pipeline --config configs/default.yaml

The configuration file can specify either a single camera under `camera` or
a list of cameras under `cameras`. Each camera entry should contain at
minimum a `source` (RTSP URL or file path). Other settings fall back to
global defaults in the config.
"""

from __future__ import annotations

import argparse
import threading
import yaml
from pathlib import Path
from typing import Dict, Any, List

import cv2

from .camera_adapters import RTSPCamera, LocalFileCamera
from .detection import VehicleDetector, PersonDetector
from .tracking import DeepSortTracker
from .recognition import ANPR, EyeRecognition
from .rules import RuleEngine
from .storage import EventLogger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_camera(source: str):
    if source.startswith("rtsp://"):
        return RTSPCamera(source)
    return LocalFileCamera(source)


class CameraPipeline(threading.Thread):
    """Threaded pipeline processing for a single camera."""

    def __init__(self, cam_id: str, camera_cfg: Dict[str, Any], global_cfg: dict, logger: EventLogger, rules_cfg: List[Dict[str, Any]]) -> None:
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.camera_cfg = camera_cfg
        self.global_cfg = global_cfg
        self.logger = logger
        self.rules_cfg = rules_cfg
        self.stop_event = threading.Event()

    def run(self) -> None:
        # Initialize camera
        source = self.camera_cfg.get("source")
        camera = create_camera(source)
        camera.open()

        # Build detectors and other components using either camera-specific or global settings
        detection_cfg = self.global_cfg.get("detection", {})
        vehicle_detector = VehicleDetector(
            detection_cfg.get("vehicle_model_path"), detection_cfg.get("device", "cpu")
        )
        person_detector = PersonDetector(
            detection_cfg.get("person_model_path"), detection_cfg.get("device", "cpu"), mask_model_path=detection_cfg.get("mask_model_path")
        )
        tracking_cfg = self.global_cfg.get("tracking", {})
        tracker = DeepSortTracker(tracking_cfg.get("max_age", 30), tracking_cfg.get("distance_threshold", 50.0))

        recog_cfg = self.global_cfg.get("recognition", {})
        anpr = ANPR(recog_cfg.get("anpr_ocr_engine"))
        eye_recognition = EyeRecognition(recog_cfg.get("eye_model_path"), database=recog_cfg.get("eye_database"))

        # Rule engine with custom handlers
        rule_engine = RuleEngine(self.rules_cfg)

        def license_plate_watchlist(context: dict) -> bool:
            plate = context.get("license_plate")
            watchlist: list[str] = context.get("watchlist", [])
            return plate and plate in watchlist

        def person_identity_watchlist(context: dict) -> bool:
            identity = context.get("person_identity")
            watchlist: list[str] = context.get("person_watchlist", [])
            return identity and identity in watchlist

        rule_engine.register_handler("license_plate_watchlist", license_plate_watchlist)
        rule_engine.register_handler("person_identity_watchlist", person_identity_watchlist)

        # Save the last processed frame to disk for dashboard streaming
        output_dir = Path(self.global_cfg.get("storage", {}).get("log_dir", "logs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        latest_frame_path = output_dir / f"latest_frame_{self.cam_id}.jpg"

        while not self.stop_event.is_set():
            ret, frame = camera.read()
            if not ret:
                break
            # Vehicle detection and tracking
            vehicle_detections = vehicle_detector.detect(frame)
            tracked_objects = tracker.update([(x1, y1, x2, y2, score) for (x1, y1, x2, y2, score) in vehicle_detections])
            # Process each tracked vehicle for ANPR
            for obj in tracked_objects:
                t_id, x1, y1, x2, y2 = obj
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                plate_img = frame[y1:y2, x1:x2]
                plate_text, plate_conf = anpr.recognize(plate_img)
                # Gather watchlist
                plate_watchlist: List[str] = []
                for rule in self.rules_cfg:
                    if rule.get("type") == "license_plate_watchlist":
                        wl = rule.get("watchlist", [])
                        if isinstance(wl, list):
                            plate_watchlist.extend(wl)
                context = {
                    "camera_id": self.cam_id,
                    "track_id": t_id,
                    "license_plate": plate_text,
                    "license_plate_confidence": plate_conf,
                    "watchlist": plate_watchlist,
                }
                events = rule_engine.evaluate(context)
                for event in events:
                    self.logger.log(event, image=plate_img)

            # Person detection and recognition
            person_detections = person_detector.detect(frame)
            for (px1, py1, px2, py2, label, score) in person_detections:
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(frame.shape[1], px2), min(frame.shape[0], py2)
                person_img = frame[py1:py2, px1:px2]
                # Extract eye region (top third)
                eye_region = person_img[0: max(1, (py2 - py1) // 3), :]
                person_id, person_conf = eye_recognition.identify(eye_region)
                # Gather person watchlist
                person_watchlist: List[str] = []
                for rule in self.rules_cfg:
                    if rule.get("type") == "person_identity_watchlist":
                        wl = rule.get("watchlist", [])
                        if isinstance(wl, list):
                            person_watchlist.extend(wl)
                context_p = {
                    "camera_id": self.cam_id,
                    "person_identity": person_id,
                    "person_identity_confidence": person_conf,
                    "person_watchlist": person_watchlist,
                }
                events_p = rule_engine.evaluate(context_p)
                for event in events_p:
                    self.logger.log(event, image=eye_region)

            # Save the latest processed frame for the dashboard to display
            try:
                cv2.imwrite(str(latest_frame_path), frame)
            except Exception:
                pass
            # Sleep briefly to allow other threads to run
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-camera road security pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    global_cfg = config
    # Determine camera configurations: either a single camera or a list of cameras
    cam_list: List[Dict[str, Any]] = []
    if "cameras" in config:
        cam_list = config["cameras"]
    elif "camera" in config:
        cam_list = [config["camera"]]
    else:
        raise ValueError("No camera configuration found in config file.")

    # Shared event logger
    storage_cfg = config.get("storage", {})
    logger = EventLogger(storage_cfg.get("log_dir", "logs"))

    rules_cfg = config.get("routing_rules", [])

    # Start a thread per camera
    threads: List[CameraPipeline] = []
    for idx, cam_cfg in enumerate(cam_list):
        cam_id = cam_cfg.get("id", f"cam{idx}")
        thread = CameraPipeline(cam_id, cam_cfg, global_cfg, logger, rules_cfg)
        thread.start()
        threads.append(thread)

    # Join threads (this blocks until all threads finish)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        # Signal threads to stop on Ctrl+C
        for t in threads:
            t.stop_event.set()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()