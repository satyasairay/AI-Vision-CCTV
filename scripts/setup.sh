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

echo "[Setup] Preparing model weights"

WEIGHTS_DIR="models"
mkdir -p "${WEIGHTS_DIR}"

VEHICLE_MODEL_URL=${VEHICLE_MODEL_URL:-}
VEHICLE_MODEL_SHA256=${VEHICLE_MODEL_SHA256:-}
PERSON_MODEL_URL=${PERSON_MODEL_URL:-}
PERSON_MODEL_SHA256=${PERSON_MODEL_SHA256:-}
MASK_MODEL_URL=${MASK_MODEL_URL:-}
MASK_MODEL_SHA256=${MASK_MODEL_SHA256:-}
VIOLENCE_MODEL_URL=${VIOLENCE_MODEL_URL:-}
VIOLENCE_MODEL_SHA256=${VIOLENCE_MODEL_SHA256:-}
REID_MODEL_URL=${REID_MODEL_URL:-}
REID_MODEL_SHA256=${REID_MODEL_SHA256:-}

download_weight() {
  local label="$1"
  local url="$2"
  local dest="$3"
  local checksum="$4"

  if [[ -z "${url}" ]]; then
    echo "[Setup] Skipping ${label} (no URL provided; set ${label^^}_URL)"
    return 0
  fi

  if [[ -f "${dest}" ]]; then
    echo "[Setup] ${label} already present at ${dest}"
  else
    echo "[Setup] Downloading ${label} from ${url}"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location "${url}" --output "${dest}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${dest}" "${url}"
    else
      echo "[Setup] Neither curl nor wget is available. Please install one of them to fetch ${label}."
      return 1
    fi
  fi

  if [[ -n "${checksum}" ]]; then
    echo "[Setup] Verifying checksum for ${label}"
    local computed
    if command -v sha256sum >/dev/null 2>&1; then
      computed=$(sha256sum "${dest}" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
      computed=$(shasum -a 256 "${dest}" | awk '{print $1}')
    else
      echo "[Setup] sha256sum/shasum not available. Skipping checksum verification for ${label}."
      return 0
    fi
    if [[ "${computed}" != "${checksum}" ]]; then
      echo "[Setup] Checksum mismatch for ${label} (expected ${checksum}, got ${computed})"
      return 1
    fi
  fi

  echo "[Setup] ${label} ready at ${dest}"
}

download_weight "vehicle_model" "${VEHICLE_MODEL_URL}" "${WEIGHTS_DIR}/vehicle_detector.pt" "${VEHICLE_MODEL_SHA256}"
download_weight "person_model" "${PERSON_MODEL_URL}" "${WEIGHTS_DIR}/person_detector.pt" "${PERSON_MODEL_SHA256}"
download_weight "mask_model" "${MASK_MODEL_URL}" "${WEIGHTS_DIR}/mask_classifier.pt" "${MASK_MODEL_SHA256}"
download_weight "violence_model" "${VIOLENCE_MODEL_URL}" "${WEIGHTS_DIR}/violence_detector.pth" "${VIOLENCE_MODEL_SHA256}"
download_weight "reid_model" "${REID_MODEL_URL}" "${WEIGHTS_DIR}/reid_resnet18.pth" "${REID_MODEL_SHA256}"

echo "[Setup] Done. To activate the environment, run 'source ${VENV_DIR}/bin/activate'"
