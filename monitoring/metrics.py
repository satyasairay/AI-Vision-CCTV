"""Prometheus metrics exporter utilities for pipeline observability."""

from __future__ import annotations

import threading
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

_server_lock = threading.Lock()
_server_started_ports: set[int] = set()


class MetricsExporter:
    """Expose processing metrics via Prometheus."""

    def __init__(self, port: int = 9095) -> None:
        self.port = port
        with _server_lock:
            if port not in _server_started_ports:
                start_http_server(port)
                _server_started_ports.add(port)

        self.frame_latency = Histogram(
            "aivision_frame_latency_seconds",
            "Per-frame processing latency",
            ["camera"],
        )
        self.vehicle_detections = Counter(
            "aivision_vehicle_detections_total",
            "Total vehicle detections emitted by detectors",
            ["camera"],
        )
        self.person_detections = Counter(
            "aivision_person_detections_total",
            "Total person detections emitted by detectors",
            ["camera"],
        )
        self.active_tracks = Gauge(
            "aivision_active_tracks",
            "Number of active vehicle tracks",
            ["camera"],
        )
        self.camera_health = Gauge(
            "aivision_camera_health_status",
            "Camera health status (1=healthy, 0=down)",
            ["camera"],
        )
        self.errors = Counter(
            "aivision_pipeline_errors_total",
            "Count of runtime errors by type",
            ["camera", "category"],
        )

    def record_frame(
        self,
        camera_id: str,
        latency_s: float,
        vehicle_count: int,
        person_count: int,
        active_tracks: int,
    ) -> None:
        self.frame_latency.labels(camera_id).observe(max(latency_s, 0.0))
        if vehicle_count:
            self.vehicle_detections.labels(camera_id).inc(vehicle_count)
        if person_count:
            self.person_detections.labels(camera_id).inc(person_count)
        self.active_tracks.labels(camera_id).set(max(active_tracks, 0))

    def record_error(self, camera_id: str, category: str) -> None:
        self.errors.labels(camera_id, category).inc()

    def record_camera_health(self, camera_id: str, healthy: bool) -> None:
        self.camera_health.labels(camera_id).set(1 if healthy else 0)
