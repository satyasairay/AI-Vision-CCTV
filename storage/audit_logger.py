"""Audit logging for user actions and configuration changes.

To meet compliance requirements, it is often necessary to record every
administrative action performed through the system: adding/removing
watchlist entries, enrolling identities, modifying rules, etc. The
`AuditLogger` writes these events to a JSONL file with timestamps and
details.

This logger is separate from the main `EventLogger`, which records
runtime detection events. Keeping audit logs distinct helps with
organizational separation of duties and simplifies retention policies.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class AuditLogger:
    """Records administrative actions and user events to a log file."""

    def __init__(self, log_dir: Optional[str | Path] = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "audit.jsonl"

    def log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """Append an audit record to the log.

        Parameters
        ----------
        action_type : str
            The high‑level category of the action (e.g. `add_watchlist_plate`).
        details : dict
            Arbitrary additional information about the action (e.g. user
            identity, parameters changed).
        """
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "action_type": action_type,
            "details": details,
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")