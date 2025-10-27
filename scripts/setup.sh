#!/usr/bin/env bash
set -e

# Setup script for the road security platform.
#
# This script creates a Python virtual environment, installs required
# dependencies, and downloads model weights. Run this once after
# cloning the repository. You can re-run it to update dependencies.

VENV_DIR=".venv"

echo "[Setup] Creating virtual environment in ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

echo "[Setup] Activating virtual environment and installing dependencies"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# Install core dependencies. Add your required versions here. Avoid external
# downloads during runtime; ensure model files are available locally.
pip install opencv-python-headless numpy streamlit
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install easyocr

echo "[Setup] Downloading/preparing model weights"
# TODO: Download pre-trained model weights into the models/ directory.
# For example: wget -O models/vehicle_detector.pt <local URL or offline storage>

echo "[Setup] Done. To activate the environment, run 'source ${VENV_DIR}/bin/activate'"
