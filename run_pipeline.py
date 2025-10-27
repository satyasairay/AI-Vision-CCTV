"""Entry point for running the road security pipeline.

This script stitches together the components from the various packages to
process a video stream, perform detection/tracking/recognition, evaluate
rules, log events, and optionally display results. It reads configuration
parameters from a YAML file.

Usage
-----
```bash
python -m road_security.run_pipeline --config configs/default.yaml
```
"""

from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import cv2

from .camera_adapters import RTSPCamera, LocalFileCamera
from .detection import VehicleDetector, PersonDetector
from .tracking import DeepSortTracker
from .recognition import ANPR, EyeRecognition
from .rules import RuleEngine
from .storage import EventLogger
from .analytics.speed_estimator import SpeedEstimator
from .analytics.dwell_time import DwellTimeMonitor


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_camera(config: dict):
    source = config["camera"]["source"]
    if source.startswith("rtsp://"):
        return RTSPCamera(source)
    return LocalFileCamera(source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the road security pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Initialize components from configuration
    camera = create_camera(config)
    detection_cfg = config.get("detection", {})
    vehicle_detector = VehicleDetector(detection_cfg.get("vehicle_model_path"), detection_cfg.get("device", "cpu"))
    # Person detector can optionally use a mask classifier model to distinguish
    # masked vs. unmasked faces. The path is read from the detection config.
    person_detector = PersonDetector(
        detection_cfg.get("person_model_path"),
        detection_cfg.get("device", "cpu"),
        mask_model_path=detection_cfg.get("mask_model_path"),
    )

    tracking_cfg = config.get("tracking", {})
    # distance_threshold controls how close detections must be to associate with an existing track
    max_age = tracking_cfg.get("max_age", 30)
    distance_threshold = tracking_cfg.get("distance_threshold", 50.0)
    tracker = DeepSortTracker(max_age, distance_threshold)

    recognition_cfg = config.get("recognition", {})
    anpr = ANPR(recognition_cfg.get("anpr_ocr_engine"))
    # Pass optional eye database to the recognizer; if none provided, the recognizer will act as a stub
    eye_database = recognition_cfg.get("eye_database")
    eye_recognition = EyeRecognition(recognition_cfg.get("eye_model_path"), database=eye_database)

    storage_cfg = config.get("storage", {})
    logger = EventLogger(storage_cfg.get("log_dir", "logs"))

    rules_cfg = config.get("routing_rules", [])
    rule_engine = RuleEngine(rules_cfg)

    # Custom rule handler for watchlist plates. It expects the context to
    # include a `license_plate` string and a `watchlist` list of strings.
    def license_plate_watchlist(context: dict) -> bool:
        plate = context.get("license_plate")
        if not plate:
            return False
        watchlist: list[str] = context.get("watchlist", [])
        return plate in watchlist

    rule_engine.register_handler("license_plate_watchlist", license_plate_watchlist)

    # Custom rule handler for person identity watchlist. It expects the
    # context to include a `person_identity` string and a `person_watchlist`
    # list of identity names. Only triggers if identity is non-empty and
    # present in the watchlist.
    def person_identity_watchlist(context: dict) -> bool:
        identity = context.get("person_identity")
        if not identity:
            return False
        watchlist: list[str] = context.get("person_watchlist", [])
        return identity in watchlist

    rule_engine.register_handler("person_identity_watchlist", person_identity_watchlist)

    # Analytics: speed estimation and dwell time
    analytics_cfg = config.get("analytics", {})
    speed_cfg = analytics_cfg.get("speed", {})
    speed_enabled = speed_cfg.get("enable", False)
    speed_estimator: SpeedEstimator | None = None
    if speed_enabled:
        ppm = speed_cfg.get("pixels_per_meter", 1.0)
        speed_estimator = SpeedEstimator(pixels_per_meter=ppm)
        speed_limit = speed_cfg.get("speed_limit", 10.0)

        # Handler for over-speed rule
        def over_speed_handler(context: dict) -> bool:
            spd = context.get("speed")
            return spd is not None and spd > speed_limit

        rule_engine.register_handler("over_speed", over_speed_handler)

    dwell_cfg = analytics_cfg.get("dwell_time", {})
    dwell_enabled = dwell_cfg.get("enable", False)
    dwell_monitor: DwellTimeMonitor | None = None
    if dwell_enabled:
        zone = dwell_cfg.get("zone", [0, 0, 0, 0])
        threshold = dwell_cfg.get("threshold", 10.0)
        dwell_monitor = DwellTimeMonitor(tuple(zone), threshold)

        def loitering_handler(context: dict) -> bool:
            # Trigger if dwell_time is provided
            return context.get("dwell_time") is not None

        rule_engine.register_handler("loitering", loitering_handler)

    # Open camera
    camera.open()
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # Vehicle detection
            vehicle_detections = vehicle_detector.detect(frame)
            tracked_objects = tracker.update([(x1, y1, x2, y2, score) for (x1, y1, x2, y2, score) in vehicle_detections])

            # For each tracked vehicle, perform ANPR
            for obj in tracked_objects:
                track_id, x1, y1, x2, y2 = obj
                # Ensure bounding box indices are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                plate_img = frame[y1:y2, x1:x2]
                plate_text, plate_conf = anpr.recognize(plate_img)
                # Extract the watchlist from config for the current rule
                watchlist: list[str] = []
                for rule in rules_cfg:
                    if rule.get("type") == "license_plate_watchlist":
                        wl = rule.get("watchlist", [])
                        if isinstance(wl, list):
                            watchlist.extend(wl)
                context = {
                    "track_id": track_id,
                    "license_plate": plate_text,
                    "license_plate_confidence": plate_conf,
                    "watchlist": watchlist,
                }
                events = rule_engine.evaluate(context)
                for event in events:
                    # Log the event with the cropped plate image
                    logger.log(event, image=plate_img)

                # Speed estimation for each tracked vehicle if enabled
                if speed_enabled and speed_estimator is not None:
                    import time
                    now = time.time()
                    speed = speed_estimator.update(track_id, (x1, y1, x2, y2), now)
                    if speed is not None:
                        ctx_speed = {
                            "track_id": track_id,
                            "speed": speed,
                        }
                        events_speed = rule_engine.evaluate(ctx_speed)
                        for event in events_speed:
                            logger.log(event)

                # Dwell time monitoring for vehicles if enabled
                if dwell_enabled and dwell_monitor is not None:
                    import time
                    now = time.time()
                    # compute centroid of vehicle bbox
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    dwell_time = dwell_monitor.update(track_id, (cx, cy), now)
                    if dwell_time is not None:
                        ctx_dwell = {
                            "track_id": track_id,
                            "dwell_time": dwell_time,
                        }
                        events_dwell = rule_engine.evaluate(ctx_dwell)
                        for event in events_dwell:
                            logger.log(event)

            # Person detection (optional)
            person_detections = person_detector.detect(frame)
            for (px1, py1, px2, py2, label, score) in person_detections:
                # Crop the person bounding box
                px1 = max(0, px1)
                py1 = max(0, py1)
                px2 = min(frame.shape[1], px2)
                py2 = min(frame.shape[0], py2)
                person_img = frame[py1:py2, px1:px2]
                # Extract eye region (top portion of the bounding box)
                eye_region = person_img[0: max(1, (py2 - py1) // 3), :]
                person_id, person_conf = eye_recognition.identify(eye_region)
                # Flatten person watchlist from config
                person_watchlist: list[str] = []
                for rule in rules_cfg:
                    if rule.get("type") == "person_identity_watchlist":
                        wl = rule.get("watchlist", [])
                        if isinstance(wl, list):
                            person_watchlist.extend(wl)
                context_p = {
                    "person_identity": person_id,
                    "person_identity_confidence": person_conf,
                    "person_watchlist": person_watchlist,
                }
                events_p = rule_engine.evaluate(context_p)
                for event in events_p:
                    # Log the event with the cropped person eye image
                    logger.log(event, image=eye_region)

                # Dwell time monitoring for vehicles is handled in the vehicle loop

            # Optional: display frame for visual confirmation (press q to quit)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
