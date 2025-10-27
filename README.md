[# Road Security Platform

## Overview

This project implements a **modular road‑security platform** designed to run entirely on local infrastructure. It integrates CCTV video streams for the following tasks:

* **Moving‑vehicle detection & tracking** – detect moving objects in each frame using either a lightweight background‑subtraction approach **or**, when local weights are provided, a **YOLO‑based detector** for higher accuracy. Vehicles are tracked across frames using a centroid‑based tracker.
* **License‑plate recognition (ANPR)** – extract and recognize license plates from detected vehicles.
* **Masked‑person & periocular/eye recognition** – detect people using a HOG‑based detector (or an optional YOLO model), classify mask usage with a lightweight CNN (if weights are available), and identify individuals using only the eye region. Identities are stored in a local `eye_database` and can be enrolled via a provided script.
* **Event logging & dashboard visualization** – log detection events with metadata and display them through a local dashboard.

The system is designed to be **fully local** with no external cloud dependencies. All computation happens on devices you control, and models are loaded from local files. Configuration allows for adapting to different camera specifications (resolution, FPS, GPU availability, NIR support).

For this demo environment, detection and tracking are designed to run locally without external downloads. When no model weights are supplied, the system uses OpenCV's background subtraction for vehicles and a HOG‑based pedestrian detector for persons. When local YOLO weights are specified in `configs/default.yaml`, the system automatically loads them for higher‑accuracy detections. Similarly, if you provide a `mask_classifier.pt`, the person detector will classify mask usage. All models run on your local CPU or GPU.

Person detection in the demo uses OpenCV's built‑in HOG‑based pedestrian detector. If you provide YOLO weights, the system will switch to a deep detector automatically. Mask classification is performed via an optional lightweight classifier; if no weights are provided, the label defaults to ``unknown``. The periocular recognition module extracts a normalized feature vector from the eye region and compares it against a local `eye_database`. Use the enrollment script (see below) to add new identities.

## Repository Layout

The code is organized into modular packages:

| Directory          | Description                                                                                   |
|--------------------|-----------------------------------------------------------------------------------------------|
| `camera_adapters/` | Interfaces for different camera sources (RTSP streams, local files, webcams).                 |
| `detection/`       | Modules for detecting vehicles and persons in frames.                                         |
| `tracking/`        | Algorithms for assigning consistent IDs to detected objects over time (e.g., Deep SORT).      |
| `recognition/`     | Automatic number plate recognition and periocular recognition for masked individuals.         |
| `rules/`           | Logic for defining security rules and triggering events.                                      |
| `storage/`         | Components for persisting logs, images, and metadata.                                         |
| `dashboard/`       | A local Streamlit/FastAPI application for visualizing detection events and camera feeds.      |
| `configs/`         | YAML/JSON files containing configurable parameters.                                           |
| `scripts/`         | Setup scripts (e.g., `setup.sh`, `run_demo.sh`) for installing dependencies and running demos. |
| `tests/`           | Unit tests covering the core functionality.                                                   |
| `sample_data/`     | Sample CCTV video and test images for development and demonstration.                          |

## Getting Started

1. Clone this repository and navigate into it.
2. Review and modify the configuration files under `configs/` to match your cameras and system capabilities.
3. Run the setup script: `./scripts/setup.sh`. This installs Python dependencies and prepares model weights.
4. Launch the demo pipeline with `./scripts/run_demo.sh`, which uses a sample video from `sample_data/`.
5. Open the dashboard (e.g., via Streamlit) to observe real‑time detections, recognition results, and event logs.

6. To process multiple cameras concurrently and stream processed frames to the dashboard, run the multi‑camera pipeline:

   ```bash
   python -m road_security.run_multi_pipeline --config configs/default.yaml
   ```

   Ensure your configuration defines either a single `camera` entry or a list of `cameras` under the `cameras` key. Each entry should specify at least a `source` URL or file path.

### Using the Dashboard

The dashboard is implemented with Streamlit and provides several controls:

* **License plate watchlist** – Use the sidebar to view and edit the list of plates that trigger alerts. Adding a plate updates the configuration; restart the pipeline for changes to take effect.
* **Person identity watchlist** – Manage a list of known identities for periocular recognition. Enroll new identities using `scripts/enroll_identity.py` and add their names via the dashboard. Matches trigger alerts when recognized.
* **Event logs** – The main page displays a table of logged events (timestamp, rule type, license plate or identity, confidence, track ID) loaded from the log file in `logs/events.jsonl`. Click **Refresh Logs** to reload the latest events.
* **Live feeds** – When the multi‑camera pipeline is running, processed frames are saved to `logs/latest_frame_cam*.jpg`. The dashboard detects these files and displays the latest frame for each camera. Use **Refresh Feeds** to update the images.

## Ethical Considerations

While this platform is designed for legitimate security purposes, please be mindful of privacy and legal regulations. Ensure that deployment complies with local laws governing video surveillance, data retention, and personal identification. Avoid misuse, particularly in ways that could infringe on individual rights or freedoms.

## Enrolling New Identities

To recognize specific individuals using periocular features, first capture a clear image of the person's eye region and run the enrollment script:

```bash
python -m road_security.scripts.enroll_identity --config configs/default.yaml \
    --identity "Alice" --image path/to/alice_eye.jpg
```

This computes a normalized feature vector from the image and adds it to the `eye_database` in your configuration. After enrolling, restart the pipeline to load the updated database, then add the identity name to the person watchlist via the dashboard to trigger alerts when recognized.
