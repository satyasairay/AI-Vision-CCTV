#!/usr/bin/env python
"""Enroll a new identity for periocular (eye-region) recognition.

Embeddings are generated with the same MobileNetV2 backbone used in
runtime inference to ensure consistent similarity scores. Embeddings can
be stored directly inside the YAML configuration or persisted to an
external `.npz` file referenced from the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from recognition.eye_recognition import EyeRecognition


def _save_npz(path: Path, database: Dict[str, np.ndarray]) -> None:
    arrays = {key: np.asarray(vec, dtype=np.float32) for key, vec in database.items()}
    np.savez(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a new identity for periocular recognition.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--identity", type=str, required=True, help="Name of the identity to enroll")
    parser.add_argument("--image", type=str, required=True, help="Path to the cropped eye-region image")
    parser.add_argument(
        "--database-path",
        type=str,
        help="Override output .npz path for embeddings (also updates config eye_database_path)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    recog_cfg = config.get("recognition", {})
    existing_db = recog_cfg.get("eye_database", {}) or {}
    db_path = args.database_path or recog_cfg.get("eye_database_path")
    device = recog_cfg.get("eye_device", "cpu")
    threshold = recog_cfg.get("eye_threshold", 0.65)

    recogniser = EyeRecognition(
        model_path=recog_cfg.get("eye_model_path"),
        database=db_path or existing_db,
        device=device,
        threshold=threshold,
    )

    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    embedding = recogniser.embed(img)
    if embedding is None:
        raise RuntimeError("Failed to compute embedding from provided image.")
    recogniser.enroll(args.identity, embedding)

    if db_path:
        out_path = Path(db_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_npz(out_path, recogniser.database)
        recog_cfg["eye_database_path"] = str(out_path)
        recog_cfg["eye_database"] = {}
        print(f"Saved embeddings to {out_path}")
    else:
        recog_cfg["eye_database"] = {
            identity: vec.tolist() for identity, vec in recogniser.database.items()
        }
        recog_cfg["eye_database_path"] = None
        print(f"Stored embeddings in config for identities: {list(recogniser.database.keys())}")

    config["recognition"] = recog_cfg
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)
    print(f"Enrolled identity '{args.identity}' using image {args.image}")


if __name__ == "__main__":
    main()
