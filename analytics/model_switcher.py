"""Resource‑aware model selection.

Some deployments may have access to GPUs, while others may only have
CPUs. Heavier neural network models like YOLOv8 perform best on GPUs but
are slower or impractical on CPUs. This helper chooses between a heavy
model and a lightweight fallback based on the available device.

The selection can be influenced by configuration. If the heavy model
path is not provided, the light model is used regardless of device.
"""

from __future__ import annotations

from typing import Optional
try:
    import torch  # type: ignore
except Exception:
    # Provide a stub if torch is unavailable
    class _DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _DummyTorch:
        cuda = _DummyCuda()

    torch = _DummyTorch()  # type: ignore


class ModelSwitcher:
    """Selects detection model paths based on hardware and configuration."""

    @staticmethod
    def select_model(device: str, heavy_model_path: Optional[str], light_model_path: Optional[str]) -> Optional[str]:
        """Return the model path to use.

        Parameters
        ----------
        device : str
            Requested device ("cpu" or "cuda"). If "auto", GPU will be
            used if available.
        heavy_model_path : str or None
            Path to a heavy detector (e.g., YOLOv8). If None, no heavy
            model is available.
        light_model_path : str or None
            Path to a lightweight detector or None if not available.

        Returns
        -------
        str or None
            The path to the selected model or None if no model is found.
        """
        use_gpu = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        if use_gpu and heavy_model_path:
            return heavy_model_path
        return light_model_path