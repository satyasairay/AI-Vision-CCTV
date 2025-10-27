#!/usr/bin/env bash
set -e

# Demo script for the road security platform.
#
# This script runs a minimal pipeline using a sample video. It assumes
# dependencies have been installed via `setup.sh` and model weights
# are available. Adjust the `--config` argument to point to your
# configuration file if needed.

CONFIG_FILE="configs/default.yaml"

echo "[Demo] Activating virtual environment"
source ".venv/bin/activate"

echo "[Demo] Running demo pipeline"
python3 -m road_security.run_pipeline --config "$CONFIG_FILE"
