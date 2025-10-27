"""Rules package.

This package contains the rule engine responsible for deciding when
security events should be logged or alerts generated based on detections
and recognitions. Rules can be configured to watch for specific
conditions, such as unauthorized vehicles entering restricted zones or
masked individuals after certain hours.
"""

from .rule_engine import RuleEngine

__all__ = ["RuleEngine"]
