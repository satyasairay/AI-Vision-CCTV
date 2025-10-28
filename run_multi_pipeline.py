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
import warnings

import cv2
import time

try:
    from .camera_adapters import RTSPCamera, LocalFileCamera
    from .detection import build_detector
    from .tracking import DeepSortTracker
    from .tracking.appearance import ColorHistogramExtractor, TorchReIDExtractor
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
    from .analytics.iot_integration import IoTController
    from .storage.database_logger import DatabaseEventLogger
    from .storage.audit_logger import AuditLogger
    from .monitoring import MetricsExporter
except ImportError:
    from camera_adapters import RTSPCamera, LocalFileCamera
    from detection import build_detector
    from tracking import DeepSortTracker
    from tracking.appearance import ColorHistogramExtractor, TorchReIDExtractor
    from recognition import ANPR, EyeRecognition
    from rules import RuleEngine
    from storage import EventLogger
    from analytics.speed_estimator import SpeedEstimator
    from analytics.dwell_time import DwellTimeMonitor
    from analytics.wrong_way import WrongWayDetector
    from analytics.duplicate_plate import DuplicatePlateDetector
    from analytics.crowd_density import CrowdDensityMonitor
    from analytics.stop_line_violation import StopLineViolationDetector
    from analytics.fight_detection import ViolenceDetector
    from analytics.weather_adapter import WeatherAdapter
    from analytics.adaptive_frame_skipper import AdaptiveFrameSkipper
    from analytics.privacy import PrivacyManager
    from analytics.camera_health import CameraHealthMonitor
    from analytics.iot_integration import IoTController
    from storage.database_logger import DatabaseEventLogger
    from storage.audit_logger import AuditLogger
    from monitoring import MetricsExporter


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_camera(source: str):
    if source.startswith("rtsp://"):
        return RTSPCamera(source)
    return LocalFileCamera(source)


class CameraPipeline(threading.Thread):
    """Threaded pipeline processing for a single camera."""

    def __init__(
        self,
        cam_id: str,
        camera_cfg: Dict[str, Any],
        global_cfg: dict,
        logger: EventLogger,
        rules_cfg: List[Dict[str, Any]],
        display_enabled: bool = True,
        metrics: MetricsExporter | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.camera_cfg = camera_cfg
        self.global_cfg = global_cfg
        self.logger = logger
        self.rules_cfg = rules_cfg
        self.stop_event = threading.Event()
        self.display_enabled = display_enabled
        self.metrics = metrics

        # Analytics configuration per camera
        analytics_cfg = global_cfg.get("analytics", {})
        # Speed estimation
        speed_cfg = analytics_cfg.get("speed", {})
        self.speed_enabled = speed_cfg.get("enable", False)
        self.speed_limit = speed_cfg.get("speed_limit", 10.0)
        self.speed_estimator: SpeedEstimator | None = None
        if self.speed_enabled:
            ppm = speed_cfg.get("pixels_per_meter", 1.0)
            self.speed_estimator = SpeedEstimator(pixels_per_meter=ppm)
        # Dwell time monitoring
        dwell_cfg = analytics_cfg.get("dwell_time", {})
        self.dwell_enabled = dwell_cfg.get("enable", False)
        self.dwell_monitor: DwellTimeMonitor | None = None
        if self.dwell_enabled:
            zone = dwell_cfg.get("zone", [0, 0, 0, 0])
            threshold = dwell_cfg.get("threshold", 10.0)
            self.dwell_monitor = DwellTimeMonitor(tuple(zone), threshold)

        # Wrong-way detection
        wrong_cfg = analytics_cfg.get("wrong_way", {})
        self.wrong_enabled = wrong_cfg.get("enable", False)
        self.wrong_detector: WrongWayDetector | None = None
        if self.wrong_enabled:
            ed = wrong_cfg.get("expected_direction", [1.0, 0.0])
            thr = wrong_cfg.get("threshold", 0.0)
            self.wrong_detector = WrongWayDetector(tuple(ed), thr)
        # Duplicate plate detection
        dup_cfg = analytics_cfg.get("duplicate_plate", {})
        self.dup_enabled = dup_cfg.get("enable", False)
        self.dup_detector: DuplicatePlateDetector | None = None
        if self.dup_enabled:
            window = dup_cfg.get("window", 300.0)
            self.dup_detector = DuplicatePlateDetector(window_seconds=window)
        # Crowd density monitoring
        crowd_cfg = analytics_cfg.get("crowd_density", {})
        self.crowd_enabled = crowd_cfg.get("enable", False)
        self.crowd_monitor: CrowdDensityMonitor | None = None
        if self.crowd_enabled:
            cz = crowd_cfg.get("zone", [0, 0, 0, 0])
            thr_c = crowd_cfg.get("threshold", 10)
            self.crowd_monitor = CrowdDensityMonitor(tuple(cz), thr_c)

        health_cfg = analytics_cfg.get("camera_health", {})
        self.health_enabled = health_cfg.get("enable", False)
        self.health_timeout = health_cfg.get("timeout", 10.0)
        self.health_id = health_cfg.get("camera_id", self.cam_id)

        # Stop-line violation detection per camera
        stop_cfg = analytics_cfg.get("stop_line", {})
        self.stop_enabled = stop_cfg.get("enable", False)
        self.stop_detector: StopLineViolationDetector | None = None
        self.stop_red_light = stop_cfg.get("red_light", True)
        if self.stop_enabled:
            line = stop_cfg.get("line", [0, 0, 0, 0])
            self.stop_detector = StopLineViolationDetector(tuple(line))

        # Violence detection per camera
        vio_cfg = analytics_cfg.get("violence", {})
        self.violence_enabled = vio_cfg.get("enable", False)
        self.violence_detector: ViolenceDetector | None = None
        if self.violence_enabled:
            window = vio_cfg.get("window", 16)
            thr = vio_cfg.get("threshold", 0.55)
            stride = vio_cfg.get("stride", 2)
            keywords = vio_cfg.get("keywords")
            detector_device = vio_cfg.get("device", self.global_cfg.get("detection", {}).get("device", "cpu"))
            self.violence_detector = ViolenceDetector(
                window=window,
                threshold=thr,
                stride=stride,
                device=detector_device,
                violent_keywords=keywords,
                weights_path=vio_cfg.get("weights_path"),
            )

        # Weather adaptation
        weather_cfg = analytics_cfg.get("weather", {})
        self.weather_enabled = weather_cfg.get("enable", False)
        self.weather_adapter: WeatherAdapter | None = None
        if self.weather_enabled:
            bthr = weather_cfg.get("brightness_threshold", 80.0)
            gamma = weather_cfg.get("gamma", 1.5)
            self.weather_adapter = WeatherAdapter(brightness_threshold=bthr, gamma=gamma)

        # Adaptive frame skipping
        skip_cfg = analytics_cfg.get("adaptive_skip", {})
        self.skip_enabled = skip_cfg.get("enable", False)
        self.skipper: AdaptiveFrameSkipper | None = None
        self.frame_index: int = 0
        if self.skip_enabled:
            idle_thr = skip_cfg.get("idle_threshold", 30)
            skip_frames = skip_cfg.get("skip_frames", 10)
            self.skipper = AdaptiveFrameSkipper(idle_threshold=idle_thr, skip_frames=skip_frames)

        # Privacy manager
        priv_cfg = analytics_cfg.get("privacy", {})
        self.privacy_enabled = priv_cfg.get("enable", False)
        self.privacy_manager: PrivacyManager | None = None
        if self.privacy_enabled:
            zones = priv_cfg.get("no_record_zones", []) or []
            self.privacy_manager = PrivacyManager(no_record_zones=[tuple(z) for z in zones])
            self.blur_non_watchlist = priv_cfg.get("blur_non_watchlist", True)

        # IoT controller (shared logger passed as argument but we might need IoT)
        self.iot = IoTController(log_dir=global_cfg.get("storage", {}).get("log_dir", "logs"))

    def run(self) -> None:
        # Initialize camera
        source = self.camera_cfg.get("source")
        camera = create_camera(source)
        camera.open()
        health_monitor: CameraHealthMonitor | None = None
        if self.health_enabled:
            health_monitor = CameraHealthMonitor(timeout=self.health_timeout)
            health_monitor.register(self.health_id)

        # Build detectors and other components using either camera-specific or global settings
        detection_cfg = self.global_cfg.get("detection", {})
        detector_device = detection_cfg.get("device", "cpu")
        vehicle_backend = detection_cfg.get("vehicle_backend", "auto")
        person_backend = detection_cfg.get("person_backend", "auto")
        vehicle_detector = build_detector(
            "vehicle",
            vehicle_backend,
            model_path=detection_cfg.get("vehicle_model_path"),
            device=detector_device,
        )
        person_detector = build_detector(
            "person",
            person_backend,
            model_path=detection_cfg.get("person_model_path"),
            device=detector_device,
            mask_model_path=detection_cfg.get("mask_model_path"),
        )
        tracking_cfg = self.global_cfg.get("tracking", {})
        appearance_cfg = tracking_cfg.get("appearance", {})
        appearance_mode = (appearance_cfg.get("mode") or "color").lower()
        extractor = ColorHistogramExtractor()
        if appearance_mode == "reid":
            try:
                extractor = TorchReIDExtractor(
                    model_path=appearance_cfg.get("model_path"),
                    device=appearance_cfg.get("device", detector_device),
                    embedding_dim=appearance_cfg.get("embedding_dim", 512),
                )
            except Exception as exc:
                warnings.warn(f"[{self.cam_id}] ReID extractor fallback to color histograms ({exc}).", stacklevel=2)
                extractor = ColorHistogramExtractor()
        tracker = DeepSortTracker(
            max_age=tracking_cfg.get("max_age", 30),
            n_init=tracking_cfg.get("n_init", 3),
            iou_threshold=tracking_cfg.get("iou_threshold", 0.3),
            appearance_weight=tracking_cfg.get("appearance_weight", 0.1),
            feature_extractor=extractor,
        )

        recog_cfg = self.global_cfg.get("recognition", {})
        anpr = ANPR(recog_cfg.get("anpr_ocr_engine"), languages=recog_cfg.get("anpr_languages"))
        eye_database = recog_cfg.get("eye_database")
        if not eye_database:
            eye_database = recog_cfg.get("eye_database_path")
        eye_recognition = EyeRecognition(
            recog_cfg.get("eye_model_path"),
            database=eye_database,
            device=recog_cfg.get("eye_device", detector_device),
            threshold=recog_cfg.get("eye_threshold", 0.65),
        )

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

        # Register analytics rule handlers
        if self.speed_enabled and self.speed_estimator is not None:
            def over_speed_handler(context: dict) -> bool:
                spd = context.get("speed")
                return spd is not None and spd > self.speed_limit
            rule_engine.register_handler("over_speed", over_speed_handler)

        if self.dwell_enabled and self.dwell_monitor is not None:
            def loitering_handler(context: dict) -> bool:
                return context.get("dwell_time") is not None
            rule_engine.register_handler("loitering", loitering_handler)

        if self.wrong_enabled and self.wrong_detector is not None:
            def wrong_way_handler(context: dict) -> bool:
                return context.get("wrong_way") is True
            rule_engine.register_handler("wrong_way", wrong_way_handler)

        if self.dup_enabled and self.dup_detector is not None:
            def duplicate_plate_handler(context: dict) -> bool:
                return context.get("duplicate_plate") is True
            rule_engine.register_handler("duplicate_plate", duplicate_plate_handler)

        if self.crowd_enabled and self.crowd_monitor is not None:
            def crowd_density_handler(context: dict) -> bool:
                return context.get("crowd") is True
            rule_engine.register_handler("crowd_density", crowd_density_handler)

        # Stop‑line violation handler
        if self.stop_enabled and self.stop_detector is not None:
            def stop_line_handler(context: dict) -> bool:
                return context.get("stop_line_violation") is True
            rule_engine.register_handler("stop_line_violation", stop_line_handler)

        # Violence handler
        if self.violence_enabled and self.violence_detector is not None:
            def violence_handler(context: dict) -> bool:
                return context.get("violence") is True
            rule_engine.register_handler("violence", violence_handler)

        def emit_camera_health_event(_: bool) -> None:
            return

        if self.health_enabled and health_monitor is not None:
            def camera_down_handler(context: dict) -> bool:
                return context.get("camera_down") is True

            def camera_recovered_handler(context: dict) -> bool:
                return context.get("camera_recovered") is True

            rule_engine.register_handler("camera_down", camera_down_handler)
            rule_engine.register_handler("camera_recovered", camera_recovered_handler)

            def emit_camera_health_event(is_healthy: bool) -> None:
                context_health = {"camera_id": self.health_id}
                if is_healthy:
                    context_health["camera_recovered"] = True
                else:
                    context_health["camera_down"] = True
                events_health = rule_engine.evaluate(context_health)
                for event in events_health:
                    self.logger.log(event)
                    self.iot.trigger(event)

        def process_health_status(healthy: bool, changed: bool) -> None:
            if not (self.health_enabled and health_monitor is not None):
                return
            if self.metrics is not None:
                self.metrics.record_camera_health(self.health_id, healthy)
            if changed:
                emit_camera_health_event(healthy)

        # Save the last processed frame to disk for dashboard streaming
        output_dir = Path(self.global_cfg.get("storage", {}).get("log_dir", "logs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        latest_frame_path = output_dir / f"latest_frame_{self.cam_id}.jpg"

        while not self.stop_event.is_set():
            frame_start = time.perf_counter()
            # Increment frame index for adaptive skipping
            self.frame_index += 1
            if self.health_enabled and health_monitor is not None:
                healthy, changed = health_monitor.evaluate(self.health_id)
                process_health_status(healthy, changed)
            if self.skip_enabled and self.skipper is not None and self.skipper.should_skip(self.frame_index):
                # Skip frame processing but still read from camera
                ret, _ = camera.read()
                if self.health_enabled and health_monitor is not None and ret:
                    health_monitor.update(self.health_id)
                    healthy, changed = health_monitor.evaluate(self.health_id)
                    process_health_status(healthy, changed)
                if not ret:
                    if self.health_enabled and health_monitor is not None:
                        healthy, changed = health_monitor.force_down(self.health_id)
                        process_health_status(healthy, changed)
                    break
                continue
            ret, frame = camera.read()
            if not ret:
                if self.health_enabled and health_monitor is not None:
                    healthy, changed = health_monitor.force_down(self.health_id)
                    process_health_status(healthy, changed)
                break
            if self.health_enabled and health_monitor is not None:
                health_monitor.update(self.health_id)
                healthy, changed = health_monitor.evaluate(self.health_id)
                process_health_status(healthy, changed)
            # Weather adaptation
            if self.weather_enabled and self.weather_adapter is not None:
                frame = self.weather_adapter.preprocess(frame)

            # Collect detection info for privacy
            det_info: list = []

            # Vehicle detection and tracking
            vehicle_detections = vehicle_detector.detect(frame)
            tracked_objects = tracker.update(
                [(x1, y1, x2, y2, score) for (x1, y1, x2, y2, score) in vehicle_detections],
                frame=frame,
            )
            active_track_count = len(tracked_objects)
            for obj in tracked_objects:
                t_id, x1, y1, x2, y2 = obj
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                plate_img = frame[y1:y2, x1:x2]
                plate_text, plate_conf = anpr.recognize(plate_img)
                # Assemble context and watchlist
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
                    self.iot.trigger(event)
                # Speed estimation
                if self.speed_enabled and self.speed_estimator is not None:
                    now = time.time()
                    spd = self.speed_estimator.update(t_id, (x1, y1, x2, y2), now)
                    if spd is not None:
                        ctx_speed = {"track_id": t_id, "speed": spd}
                        events_spd = rule_engine.evaluate(ctx_speed)
                        for event in events_spd:
                            self.logger.log(event)
                            self.iot.trigger(event)
                # Dwell time monitoring
                if self.dwell_enabled and self.dwell_monitor is not None:
                    now = time.time()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    dtm = self.dwell_monitor.update(t_id, (cx, cy), now)
                    if dtm is not None:
                        ctx_dwell = {"track_id": t_id, "dwell_time": dtm}
                        events_dt = rule_engine.evaluate(ctx_dwell)
                        for event in events_dt:
                            self.logger.log(event)
                            self.iot.trigger(event)
                # Wrong‑way detection
                if self.wrong_enabled and self.wrong_detector is not None:
                    wrong = self.wrong_detector.update(t_id, (x1, y1, x2, y2))
                    if wrong:
                        events_wrong = rule_engine.evaluate({"wrong_way": True})
                        for event in events_wrong:
                            self.logger.log(event)
                            self.iot.trigger(event)
                # Duplicate plate detection
                if self.dup_enabled and self.dup_detector is not None:
                    if plate_text:
                        dup = self.dup_detector.update(plate_text)
                        if dup:
                            events_dup = rule_engine.evaluate({"duplicate_plate": True, "license_plate": plate_text})
                            for event in events_dup:
                                self.logger.log(event)
                                self.iot.trigger(event)
                # Stop‑line violation detection
                if self.stop_enabled and self.stop_detector is not None:
                    if self.stop_detector.update(t_id, (x1, y1, x2, y2), red_light=self.stop_red_light):
                        events_stop = rule_engine.evaluate({"stop_line_violation": True})
                        for event in events_stop:
                            self.logger.log(event)
                            self.iot.trigger(event)
                # Append detection info for privacy
                det_info.append((x1, y1, x2, y2, "vehicle", 1.0, plate_text))

            # Person detection
            person_detections = person_detector.detect(frame)
            for idx, (px1, py1, px2, py2, label, score) in enumerate(person_detections):
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(frame.shape[1], px2), min(frame.shape[0], py2)
                person_img = frame[py1:py2, px1:px2]
                eye_region = person_img[0: max(1, (py2 - py1) // 3), :]
                person_id, person_conf = eye_recognition.identify(eye_region)
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
                    self.iot.trigger(event)
                # Violence detection
                if self.violence_enabled and self.violence_detector is not None:
                    center_x = (px1 + px2) // 2
                    center_y = (py1 + py2) // 2
                    track_hash = hash((center_x // 10, center_y // 10, idx % 4))
                    if self.violence_detector.update(frame, (px1, py1, px2, py2), track_hash):
                        events_vio = rule_engine.evaluate({"violence": True})
                        for event in events_vio:
                            self.logger.log(event)
                            self.iot.trigger(event)
                # Append detection info for privacy
                det_info.append((px1, py1, px2, py2, label, score, person_id if person_id else None))

            # Crowd density
            if self.crowd_enabled and self.crowd_monitor is not None and person_detections:
                boxes = [(px1, py1, px2, py2) for (px1, py1, px2, py2, _, _) in person_detections]
                crowd = self.crowd_monitor.count(boxes)
                if crowd:
                    events_crowd = rule_engine.evaluate({"crowd": True})
                    for event in events_crowd:
                        self.logger.log(event)
                        self.iot.trigger(event)

            # Apply privacy redaction
            if self.privacy_enabled and self.privacy_manager is not None:
                # Build watchlist of plates for redaction
                wlist: list[str] = []
                for rule in self.rules_cfg:
                    if rule.get("type") == "license_plate_watchlist":
                        wl = rule.get("watchlist", [])
                        if isinstance(wl, list):
                            wlist.extend(wl)
                self.privacy_manager.redact_detections(frame, det_info, watchlist=wlist)

            # Update adaptive skipper
            if self.skip_enabled and self.skipper is not None:
                has_activity = bool(vehicle_detections) or bool(person_detections)
                self.skipper.update(self.frame_index, has_activity)

            if self.metrics is not None:
                self.metrics.record_frame(
                    self.cam_id,
                    time.perf_counter() - frame_start,
                    len(vehicle_detections),
                    len(person_detections),
                    active_track_count,
                )

            # Save latest frame
            try:
                cv2.imwrite(str(latest_frame_path), frame)
            except Exception:
                pass
            if self.display_enabled and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-camera road security pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV UI windows (useful on headless environments).",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Expose Prometheus metrics on this port (overrides config).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    global_cfg = config
    display_enabled = not args.no_display
    monitoring_cfg = config.get("monitoring", {})
    metrics_port = args.metrics_port if args.metrics_port is not None else monitoring_cfg.get("metrics_port")
    metrics_enabled = monitoring_cfg.get("enable_metrics", False) or metrics_port is not None
    metrics_exporter: MetricsExporter | None = None
    if metrics_enabled:
        metrics_exporter = MetricsExporter(port=metrics_port or 9095)
    # Determine camera configurations: either a single camera or a list of cameras
    cam_list: List[Dict[str, Any]] = []
    if "cameras" in config:
        cam_list = config["cameras"]
    elif "camera" in config:
        cam_list = [config["camera"]]
    else:
        raise ValueError("No camera configuration found in config file.")

    # Shared event logger: choose between JSONL and SQLite
    storage_cfg = config.get("storage", {})
    backend = config.get("storage_backend", "jsonl").lower()
    if backend == "sqlite":
        db_cfg = config.get("database", {})
        db_path = db_cfg.get("db_path", "events.db")
        image_dir = db_cfg.get("image_dir", "logs/images")
        logger = DatabaseEventLogger(db_path=db_path, image_dir=image_dir)
    else:
        logger = EventLogger(storage_cfg.get("log_dir", "logs"))

    rules_cfg = config.get("routing_rules", [])

    # Start a thread per camera
    threads: List[CameraPipeline] = []
    for idx, cam_cfg in enumerate(cam_list):
        cam_id = cam_cfg.get("id", f"cam{idx}")
        thread = CameraPipeline(
            cam_id,
            cam_cfg,
            global_cfg,
            logger,
            rules_cfg,
            display_enabled=display_enabled,
            metrics=metrics_exporter,
        )
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
