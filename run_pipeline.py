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
from .analytics.wrong_way import WrongWayDetector
from .analytics.duplicate_plate import DuplicatePlateDetector
from .analytics.crowd_density import CrowdDensityMonitor
from .analytics.stop_line_violation import StopLineViolationDetector
from .analytics.fight_detection import ViolenceDetector
from .analytics.weather_adapter import WeatherAdapter
from .analytics.adaptive_frame_skipper import AdaptiveFrameSkipper
from .analytics.privacy import PrivacyManager
from .analytics.camera_health import CameraHealthMonitor
from .analytics.event_stats import event_counts_by_type  # example usage
from .analytics.model_switcher import ModelSwitcher
from .analytics.iot_integration import IoTController
from .storage.database_logger import DatabaseEventLogger
from .storage.audit_logger import AuditLogger


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
    anpr = ANPR(recognition_cfg.get("anpr_ocr_engine"), languages=recognition_cfg.get("anpr_languages"))
    # Pass optional eye database to the recognizer; if none provided, the recognizer will act as a stub
    eye_database = recognition_cfg.get("eye_database")
    eye_recognition = EyeRecognition(recognition_cfg.get("eye_model_path"), database=eye_database)

    storage_cfg = config.get("storage", {})
    # Choose storage backend: JSONL (default) or SQLite
    backend = config.get("storage_backend", "jsonl").lower()
    if backend == "sqlite":
        db_cfg = config.get("database", {})
        db_path = db_cfg.get("db_path", "events.db")
        image_dir = db_cfg.get("image_dir", "logs/images")
        logger = DatabaseEventLogger(db_path=db_path, image_dir=image_dir)
    else:
        logger = EventLogger(storage_cfg.get("log_dir", "logs"))

    # IoT controller for external triggers (no‑op if no log_dir specified)
    iot = IoTController(log_dir=storage_cfg.get("log_dir", "logs"))

    # Audit logger to record administrative actions (currently unused here)
    audit_logger = AuditLogger(storage_cfg.get("log_dir", "logs"))

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

    # Wrong-way detection
    wrong_cfg = analytics_cfg.get("wrong_way", {})
    wrong_enabled = wrong_cfg.get("enable", False)
    wrong_detector: WrongWayDetector | None = None
    if wrong_enabled:
        ed = wrong_cfg.get("expected_direction", [1.0, 0.0])
        thr = wrong_cfg.get("threshold", 0.0)
        wrong_detector = WrongWayDetector(tuple(ed), thr)

        def wrong_way_handler(context: dict) -> bool:
            return context.get("wrong_way") is True

        rule_engine.register_handler("wrong_way", wrong_way_handler)

    # Duplicate plate detection
    dup_cfg = analytics_cfg.get("duplicate_plate", {})
    dup_enabled = dup_cfg.get("enable", False)
    dup_detector: DuplicatePlateDetector | None = None
    if dup_enabled:
        window = dup_cfg.get("window", 300.0)
        dup_detector = DuplicatePlateDetector(window_seconds=window)

        def duplicate_plate_handler(context: dict) -> bool:
            return context.get("duplicate_plate") is True

        rule_engine.register_handler("duplicate_plate", duplicate_plate_handler)

    # Crowd density monitoring
    crowd_cfg = analytics_cfg.get("crowd_density", {})
    crowd_enabled = crowd_cfg.get("enable", False)
    crowd_monitor: CrowdDensityMonitor | None = None
    if crowd_enabled:
        zone = crowd_cfg.get("zone", [0, 0, 0, 0])
        threshold = crowd_cfg.get("threshold", 10)
        crowd_monitor = CrowdDensityMonitor(tuple(zone), threshold)

        def crowd_density_handler(context: dict) -> bool:
            return context.get("crowd") is True

        rule_engine.register_handler("crowd_density", crowd_density_handler)

    # Stop‑line violation detection
    stop_cfg = analytics_cfg.get("stop_line", {})
    stop_enabled = stop_cfg.get("enable", False)
    stop_detector: StopLineViolationDetector | None = None
    if stop_enabled:
        line = stop_cfg.get("line", [0, 0, 0, 0])
        red_flag = stop_cfg.get("red_light", True)
        stop_detector = StopLineViolationDetector(tuple(line))

        def stop_line_handler(context: dict) -> bool:
            return context.get("stop_line_violation") is True

        rule_engine.register_handler("stop_line_violation", stop_line_handler)

    # Violence detection
    violence_cfg = analytics_cfg.get("violence", {})
    violence_enabled = violence_cfg.get("enable", False)
    violence_detector: ViolenceDetector | None = None
    if violence_enabled:
        window = violence_cfg.get("window", 5)
        thresh = violence_cfg.get("change_threshold", 1.5)
        violence_detector = ViolenceDetector(window=window, change_threshold=thresh)

        def violence_handler(context: dict) -> bool:
            return context.get("violence") is True

        rule_engine.register_handler("violence", violence_handler)

    # Weather adaptation
    weather_cfg = analytics_cfg.get("weather", {})
    weather_enabled = weather_cfg.get("enable", False)
    weather_adapter: WeatherAdapter | None = None
    if weather_enabled:
        brightness = weather_cfg.get("brightness_threshold", 80.0)
        gamma = weather_cfg.get("gamma", 1.5)
        weather_adapter = WeatherAdapter(brightness_threshold=brightness, gamma=gamma)

    # Adaptive frame skipping
    skip_cfg = analytics_cfg.get("adaptive_skip", {})
    skip_enabled = skip_cfg.get("enable", False)
    skipper: AdaptiveFrameSkipper | None = None
    frame_index = 0
    if skip_enabled:
        idle_thr = skip_cfg.get("idle_threshold", 30)
        skip_frames = skip_cfg.get("skip_frames", 10)
        skipper = AdaptiveFrameSkipper(idle_threshold=idle_thr, skip_frames=skip_frames)

    # Privacy manager
    priv_cfg = analytics_cfg.get("privacy", {})
    privacy_enabled = priv_cfg.get("enable", False)
    privacy_manager: PrivacyManager | None = None
    if privacy_enabled:
        zones = priv_cfg.get("no_record_zones", []) or []
        privacy_manager = PrivacyManager(no_record_zones=[tuple(z) for z in zones])
        blur_non_watchlist = priv_cfg.get("blur_non_watchlist", True)


    # Open camera
    camera.open()
    try:
        while True:
            # Increment frame index and apply adaptive skipping
            frame_index += 1
            if skip_enabled and skipper is not None and skipper.should_skip(frame_index):
                # Skip processing this frame but still read from camera
                ret, _ = camera.read()
                if not ret:
                    break
                continue

            ret, frame = camera.read()
            if not ret:
                break

            # Apply weather adaptation preprocessing if enabled
            if weather_enabled and weather_adapter is not None:
                frame = weather_adapter.preprocess(frame)

            # Collect detection info for privacy redaction
            vehicle_det_info: list = []

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
                    # Trigger connected IoT devices (e.g., alarms) based on the event
                    iot.trigger(event)

                # Stop‑line violation detection
                if stop_enabled and stop_detector is not None:
                    # Note: stop_cfg is available from outer scope; red_light may change externally
                    violation = stop_detector.update(track_id, (x1, y1, x2, y2), red_light=stop_cfg.get("red_light", True))
                    if violation:
                        ctx_stop = {"track_id": track_id, "stop_line_violation": True}
                        events_stop = rule_engine.evaluate(ctx_stop)
                        for event in events_stop:
                            logger.log(event)
                            iot.trigger(event)

                # Record detection info for privacy manager: include identifier (plate_text)
                vehicle_det_info.append((x1, y1, x2, y2, "vehicle", 1.0, plate_text))

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
                            iot.trigger(event)

                # Wrong-way detection
                if wrong_enabled and wrong_detector is not None:
                    wrong_way_flag = wrong_detector.update(track_id, (x1, y1, x2, y2))
                    if wrong_way_flag:
                        ctx_wrong = {"track_id": track_id, "wrong_way": True}
                        events_wrong = rule_engine.evaluate(ctx_wrong)
                        for event in events_wrong:
                            logger.log(event)
                            iot.trigger(event)

                # Duplicate plate detection
                if dup_enabled and dup_detector is not None:
                    if plate_text:
                        is_dup = dup_detector.update(plate_text)
                        if is_dup:
                            ctx_dup = {
                                "license_plate": plate_text,
                                "duplicate_plate": True,
                            }
                            events_dup = rule_engine.evaluate(ctx_dup)
                            for event in events_dup:
                                logger.log(event)
                                iot.trigger(event)

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
                            iot.trigger(event)

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
                    iot.trigger(event)

                # Violence detection for this person
                if violence_enabled and violence_detector is not None:
                    if violence_detector.update(track_id=0, bbox=(px1, py1, px2, py2)):
                        ctx_vio = {"violence": True}
                        events_vio = rule_engine.evaluate(ctx_vio)
                        for event in events_vio:
                            logger.log(event)
                            iot.trigger(event)

                # Append person detection info for privacy manager (include identity)
                vehicle_det_info.append((px1, py1, px2, py2, label, score, person_id if person_id else None))

                # Dwell time monitoring for vehicles is handled in the vehicle loop

            # Crowd density check across all person detections
            if crowd_enabled and crowd_monitor is not None and person_detections:
                # Build list of bounding boxes (x1,y1,x2,y2) from person detections
                boxes = [(px1, py1, px2, py2) for (px1, py1, px2, py2, _, _) in person_detections]
                is_crowd = crowd_monitor.count(boxes)
                if is_crowd:
                    events_crowd = rule_engine.evaluate({"crowd": True})
                    for event in events_crowd:
                        logger.log(event)
                        iot.trigger(event)

            # Apply privacy redaction if enabled
            if privacy_enabled and privacy_manager is not None:
                # Build watchlist of identifiers to preserve from routing rules
                watch_plates: list[str] = []
                for rule in rules_cfg:
                    if rule.get("type") == "license_plate_watchlist":
                        wl = rule.get("watchlist", [])
                        if isinstance(wl, list):
                            watch_plates.extend(wl)
                privacy_manager.redact_detections(frame, vehicle_det_info, watchlist=watch_plates)

            # Update adaptive skipper with current activity status
            if skip_enabled and skipper is not None:
                has_activity = bool(vehicle_detections) or bool(person_detections)
                skipper.update(frame_index, has_activity)

            # Optional: display frame for visual confirmation (press q to quit)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
