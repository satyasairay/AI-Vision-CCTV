"""Appearance embedding extractors for multi-object tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class FeatureExtractor(Protocol):
    """Protocol for feature extractors used by the tracker."""

    def __call__(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        ...


@dataclass
class ColorHistogramExtractor:
    """Fallback colour histogram embedding."""

    bins: Tuple[int, int, int] = (8, 8, 8)

    def __call__(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        patch = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)


class TorchReIDExtractor:
    """ResNet-based embedding extractor for appearance features."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: int = 512,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        self.embedding_dim = embedding_dim
        if model_path:
            path = Path(model_path)
            if path.exists():
                state = torch.load(path, map_location=self.device)
                missing, unexpected = backbone.load_state_dict(state, strict=False)
                if missing or unexpected:
                    warnings.warn(
                        f"ReID weights loaded with missing={missing} unexpected={unexpected}",
                        stacklevel=2,
                    )
            else:
                warnings.warn(f"ReID weights not found at {path}, using ImageNet initialization.", stacklevel=2)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.model = backbone.to(self.device).eval()
        self.transforms = weights.transforms()

    def __call__(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensor = self.transforms(patch_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        emb = embedding.squeeze(0)
        emb = emb / (emb.norm(p=2) + 1e-8)
        return emb.cpu().numpy().astype(np.float32)
