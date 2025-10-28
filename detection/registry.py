"""Detection backend registry utility.

This module centralises detector discovery so that pipelines can select a
backend by name (configured per deployment) without hard-coding specific
classes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Tuple, Type

DetectorFactory = Callable[..., Any]

_REGISTRY: Dict[str, Dict[str, DetectorFactory]] = defaultdict(dict)


def register_detector(task: str, name: str) -> Callable[[DetectorFactory], DetectorFactory]:
    """Decorator to register a detector factory under ``task/name``.

    Parameters
    ----------
    task:
        Logical detector grouping, e.g. ``"vehicle"`` or ``"person"``.
    name:
        Backend identifier selected via configuration.
    """

    def decorator(factory: DetectorFactory) -> DetectorFactory:
        key = name.lower()
        if key in _REGISTRY[task]:
            raise ValueError(f"Detector already registered for task '{task}' with name '{name}'")
        _REGISTRY[task][key] = factory
        return factory

    return decorator


def build_detector(task: str, name: str, **kwargs: Any) -> Any:
    """Instantiate a detector for ``task`` using backend ``name``.

    Raises
    ------
    KeyError
        If no detector is registered under the given task/name.
    """

    backends = _REGISTRY.get(task)
    if not backends:
        raise KeyError(f"No detectors registered for task '{task}'")
    backend = backends.get(name.lower())
    if backend is None:
        available = ", ".join(sorted(backends))
        raise KeyError(f"Detector '{name}' not registered for task '{task}'. Available: {available}")
    return backend(**kwargs)


def available_detectors(task: str) -> Iterable[str]:
    """Return registered backend names for a task."""
    return _REGISTRY.get(task, {}).keys()
