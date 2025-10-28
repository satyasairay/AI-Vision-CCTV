"""Violence / aggressive behaviour detection using video classification models.

The detector wraps a lightweight 3D CNN (torchvision ``r3d_18`` pretrained
on Kinetics-400) and evaluates short clips cropped around tracked persons.
While not a substitute for a domain-specific model, it markedly improves on
the area-change heuristic and provides a clear extension point for bespoke
weights when available.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18

DEFAULT_VIOLENT_KEYWORDS = {
    "fight",
    "punch",
    "kick",
    "boxing",
    "wrestling",
    "martial",
    "slap",
    "fencing",
    "karate",
    "taekwondo",
    "kung fu",
}


class ViolenceDetector:
    """Detect violent/aggressive actions from short person-centric clips.

    Parameters
    ----------
    window : int, optional
        Number of frames per clip. Model expects powers of two; 16 works well.
    threshold : float, optional
        Probability threshold above which an event is flagged.
    device : str, optional
        Torch device for inference (``"cpu"`` or ``"cuda"``).
    violent_keywords : Iterable[str], optional
        Keywords matched (case-insensitive substring) against the model's
        Kinetics-400 class names to decide which classes count as violence.
    stride : int, optional
        Process every ``stride``th frame to reduce compute while maintaining
        responsiveness.
    """

    def __init__(
        self,
        window: int = 16,
        threshold: float = 0.55,
        device: Optional[str] = None,
        violent_keywords: Optional[Iterable[str]] = None,
        stride: int = 2,
        weights_path: Optional[str] = None,
    ) -> None:
        self.window = window
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.stride = max(1, stride)

        weights_enum = R3D_18_Weights.KINETICS400_V1
        self.model = r3d_18(weights=weights_enum).to(self.device)
        if weights_path:
            try:
                state = torch.load(weights_path, map_location=self.device)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    warnings.warn(
                        f"Violence model loaded with missing={missing}, unexpected={unexpected}",
                        stacklevel=2,
                    )
            except FileNotFoundError:
                warnings.warn(f"Custom violence weights not found at {weights_path}; using default Kinetics model.", stacklevel=2)
            except Exception as exc:
                warnings.warn(f"Failed to load custom violence weights ({exc}); using default.", stacklevel=2)
        self.model.eval()

        meta = weights_enum.meta or {}
        mean = meta.get("mean", (0.43216, 0.394666, 0.37645))
        std = meta.get("std", (0.22803, 0.22145, 0.216989))
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=torch.float32).view(3, 1, 1)
        keywords = {kw.lower() for kw in (violent_keywords or DEFAULT_VIOLENT_KEYWORDS)}
        categories: Sequence[str] = meta.get("categories", [])
        if not categories:
            raise RuntimeError("Violence detection weights missing class categories.")
        self.violent_indices = [
            idx for idx, label in enumerate(categories) if any(kw in label.lower() for kw in keywords)
        ]
        if not self.violent_indices:
            raise RuntimeError("No violent classes matched for violence detection model.")

        self.buffers: Dict[int, Deque[torch.Tensor]] = {}
        self._frame_counters: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: int,
    ) -> bool:
        """Append a cropped frame for ``track_id`` and evaluate when ready."""
        if frame is None:
            return False
        crop = self._crop_frame(frame, bbox)
        if crop is None:
            return False
        counter = self._frame_counters.setdefault(track_id, 0)
        counter += 1
        self._frame_counters[track_id] = counter
        if counter % self.stride != 0:
            return False

        tensor = self._to_tensor(crop)
        buffer = self.buffers.setdefault(track_id, deque(maxlen=self.window))
        buffer.append(tensor)
        if len(buffer) < self.window:
            return False

        clip = torch.stack(list(buffer), dim=1).unsqueeze(0)  # (1, C, T, H, W)
        with torch.no_grad():
            logits = self.model(clip.to(self.device))
            probs = F.softmax(logits, dim=1)[0]
            violence_score = float(probs[self.violent_indices].sum().item())
        return violence_score >= self.threshold

    # ------------------------------------------------------------------
    @staticmethod
    def _crop_frame(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all buffers, e.g. when switching cameras."""
        self.buffers.clear()
        self._frame_counters.clear()
