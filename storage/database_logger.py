"""SQLite event logger.

This logger persists events in a SQLite database. It complements the
JSONL file logger by enabling structured queries on historical events.
When initialized, it creates a table if it doesn't exist. Each log
operation inserts a row and optionally writes the associated image to
the filesystem.

Usage
-----
```
from road_security.storage.database_logger import DatabaseEventLogger
logger = DatabaseEventLogger(db_path="events.db", image_dir="logs/images")
logger.log({"type": "over_speed", "track_id": 1, "speed": 12.3}, image)
```
"""

from __future__ import annotations

import sqlite3
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import cv2


class DatabaseEventLogger:
    """Persist events and metadata to a SQLite database."""

    def __init__(self, db_path: str, image_dir: Optional[str | Path] = None) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
        # Directory for storing images associated with events
        self.image_dir = Path(image_dir) if image_dir else None
        if self.image_dir:
            self.image_dir.mkdir(parents=True, exist_ok=True)

    def _create_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT,
                track_id INTEGER,
                data TEXT,
                image_path TEXT
            )
            """
        )
        self.conn.commit()

    def log(self, event: Dict[str, Any], image: Optional[Any] = None) -> None:
        timestamp = datetime.datetime.utcnow().isoformat()
        event_type = event.get("type")
        track_id = event.get("track_id")
        data_json = json.dumps(event)
        image_path_str: Optional[str] = None
        if image is not None and self.image_dir is not None:
            image_filename = f"{timestamp}.jpg"
            image_path = self.image_dir / image_filename
            cv2.imwrite(str(image_path), image)
            image_path_str = str(image_path)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO events (timestamp, type, track_id, data, image_path) VALUES (?, ?, ?, ?, ?)",
            (timestamp, event_type, track_id, data_json, image_path_str),
        )
        self.conn.commit()