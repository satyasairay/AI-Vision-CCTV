"""Security rule engine.

The `RuleEngine` class evaluates conditions on detection, tracking, and
recognition outputs to determine whether a security event should be
generated. Rules can be defined via configuration files and loaded at
runtime. This engine is intended to be extendable with custom rules.
"""

from __future__ import annotations

from typing import List, Dict, Any, Callable


class RuleEngine:
    """Evaluate security rules and trigger events."""

    def __init__(self, rules: List[Dict[str, Any]] | None = None) -> None:
        """
        Parameters
        ----------
        rules : list of dict, optional
            A list of rule definitions. Each rule can specify conditions
            on detections, tracked objects, or recognitions. For example,
            a rule could trigger when a vehicle's license plate matches
            a watchlist entry.
        """
        self.rules: List[Dict[str, Any]] = rules or []
        self.custom_handlers: Dict[str, Callable[[Dict[str, Any]], bool]] = {}

    def register_handler(self, name: str, handler: Callable[[Dict[str, Any]], bool]) -> None:
        """Register a custom rule handler.

        Custom handlers can implement complex logic. When a rule's
        `type` matches `name`, the handler will be invoked with the
        context dictionary. The handler should return `True` if the
        event is triggered.
        """
        self.custom_handlers[name] = handler

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all rules against the current context.

        Parameters
        ----------
        context : dict
            Dictionary containing detection/tracking/recognition outputs
            for the current frame or object.

        Returns
        -------
        events : list of dict
            A list of triggered events. Each event dictionary contains
            the rule that was triggered and any additional metadata.
        """
        triggered_events = []
        for rule in self.rules:
            rule_type = rule.get("type")
            handler = self.custom_handlers.get(rule_type)
            if handler is not None:
                if handler(context):
                    triggered_events.append({"rule": rule, "context": context})
        return triggered_events
