#!/usr/bin/env python
"""Enroll a new identity for eye recognition.

This utility script adds a new person's periocular feature vector to the
`eye_database` field in the YAML configuration. The feature is computed
using the same simple extraction method as in `EyeRecognition.identify`: the
cropped eye image is converted to grayscale, resized to 64×64, flattened,
and normalized. The resulting list of floats is stored in the config.

Usage::

    python -m road_security.scripts.enroll_identity --config configs/default.yaml \
        --identity "Alice" --image /path/to/alice_eye.jpg

After running this script, restart the pipeline to load the updated
database.
"""

from __future__ import annotations

import argparse
import yaml
from pathlib import Path
from typing import List

import numpy as np


def extract_feature(image_path: str) -> List[float]:
    """Extract a normalized feature vector from an eye image.

    Parameters
    ----------
    image_path : str
        Path to the eye region image file.

    Returns
    -------
    feature : list of float
        The normalized feature vector as a Python list.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    vec = resized.flatten().astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a new identity for eye recognition.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--identity", type=str, required=True, help="Name of the identity to enroll")
    parser.add_argument("--image", type=str, required=True, help="Path to the eye region image")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Extract feature vector from the provided image
    feature = extract_feature(args.image)

    # Load and update the configuration
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    recog_cfg = config.get("recognition", {})
    eye_db = recog_cfg.get("eye_database", {}) or {}
    # Assign the feature vector to the identity
    eye_db[args.identity] = feature
    recog_cfg["eye_database"] = eye_db
    config["recognition"] = recog_cfg

    with config_path.open("w") as f:
        yaml.safe_dump(config, f)
    print(f"Enrolled identity '{args.identity}' with feature vector of length {len(feature)} into {config_path}")


if __name__ == "__main__":
    main()