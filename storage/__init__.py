"""Storage package.

This package provides utilities for logging events, saving frames, and
persisting metadata. Different backends (e.g., local files, SQLite,
PostgreSQL) can be implemented here. The default implementation uses
simple file-based logging.
"""

from .event_logger import EventLogger

__all__ = ["EventLogger"]
