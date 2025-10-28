"""Event statistics utilities.

This module provides helper functions for computing basic statistics from
event logs. These statistics can be displayed in the dashboard as
charts or used for reporting. Statistics include event counts per
type, per hour/day, and mean/median speeds, dwell times, etc.

Event logs are expected to be stored in JSONL format (one JSON object
per line) as produced by the `EventLogger` and `AuditLogger` classes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_events(log_path: str | Path) -> List[Dict[str, Any]]:
    """Load events from a JSONL log file."""
    events: List[Dict[str, Any]] = []
    path = Path(log_path)
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def event_counts_by_type(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count how many events of each type occurred."""
    counts: Dict[str, int] = defaultdict(int)
    for evt in events:
        evt_type = evt.get("type")
        if evt_type:
            counts[evt_type] += 1
    return dict(counts)


def hourly_counts(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count events per hour (UTC)."""
    counts: Dict[str, int] = defaultdict(int)
    for evt in events:
        ts = evt.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        hour_key = dt.strftime("%Y-%m-%d %H:00")
        counts[hour_key] += 1
    return dict(counts)