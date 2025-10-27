"""Tracking package.

This package provides tracking algorithms for assigning consistent
identifiers to objects detected in video frames. The primary implementation
is based on the Deep SORT algorithm, which combines motion and appearance
information to track objects across frames.
"""

from .deep_sort import DeepSortTracker

__all__ = ["DeepSortTracker"]
