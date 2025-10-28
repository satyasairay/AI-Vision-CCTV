from __future__ import annotations

import time

from analytics.camera_health import CameraHealthMonitor


def test_camera_health_status_transitions() -> None:
    monitor = CameraHealthMonitor(timeout=0.2)
    monitor.register("cam0")

    healthy, changed = monitor.evaluate("cam0")
    assert healthy is True
    assert changed is False

    monitor.last_seen["cam0"] -= 1.0  # simulate inactivity
    healthy, changed = monitor.evaluate("cam0")
    assert healthy is False
    assert changed is True

    monitor.update("cam0", timestamp=time.time())
    healthy, changed = monitor.evaluate("cam0")
    assert healthy is True
    assert changed is True

    healthy, changed = monitor.force_down("cam0")
    assert healthy is False
    assert changed is True
