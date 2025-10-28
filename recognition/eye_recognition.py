"""Periocular recognition using pretrained CNN embeddings.

The module extracts embeddings from eye crops using a MobileNetV2 backbone
pretrained on ImageNet, providing a stronger baseline than the former
grayscale heuristic. Features are L2-normalised so cosine similarity can be
used for matching against an enrolled gallery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class EyeRecognition:
    """Identify individuals using periocular embeddings."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        database: Optional[Dict[str, Iterable[float]]] = None,
        device: Optional[str] = None,
        threshold: float = 0.65,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold

        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        base_model = mobilenet_v2(weights=weights)
        if model_path:
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                state = torch.load(model_path_obj, map_location=self.device)
                base_model.load_state_dict(state, strict=False)
            else:
                warnings.warn(f"Eye recognition weights not found at {model_path_obj}, using ImageNet backbone.", stacklevel=2)
        # Drop classifier head -> feature extractor returns (B, 1280, 1, 1)
        self.feature_extractor = nn.Sequential(*(list(base_model.children())[:-1]))
        self.feature_extractor.to(self.device).eval()

        meta = weights.meta or {}
        mean = meta.get("mean", (0.485, 0.456, 0.406))
        std = meta.get("std", (0.229, 0.224, 0.225))
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=torch.float32).view(3, 1, 1)

        self.database: Dict[str, np.ndarray] = {}
        if isinstance(database, str):
            db_path = Path(database)
            if db_path.exists():
                loaded = np.load(db_path, allow_pickle=True)
                for key in loaded.files:
                    self.database[key] = self._normalise(np.asarray(loaded[key], dtype=np.float32))
        elif database:
            for identity, vector in database.items():
                arr = np.asarray(vector, dtype=np.float32)
                self.database[identity] = self._normalise(arr)

    # ------------------------------------------------------------------
    def identify(self, eye_image: np.ndarray) -> Tuple[str, float]:
        if not self.database:
            return "", 0.0
        embedding = self.embed(eye_image)
        if embedding is None:
            return "", 0.0
        best_id = ""
        best_score = 0.0
        for identity, enrolled_vec in self.database.items():
            score = float(np.dot(embedding, enrolled_vec))
            if score > best_score:
                best_id = identity
                best_score = score
        if best_score < self.threshold:
            return "", 0.0
        return best_id, best_score

    # ------------------------------------------------------------------
    def embed(self, eye_image: np.ndarray) -> Optional[np.ndarray]:
        if eye_image is None or eye_image.size == 0:
            return None
        img = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(tensor)
        embedding = features.view(-1)
        embedding = F.normalize(embedding, dim=0)
        return embedding.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    def enroll(self, identity: str, embedding: np.ndarray) -> None:
        self.database[identity] = self._normalise(np.asarray(embedding, dtype=np.float32))

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return vec
        return vec / norm
