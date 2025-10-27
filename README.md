[# Road Security Platform

## Overview

This project implements a **modular road‑security platform** designed to run entirely on local infrastructure. It integrates CCTV video streams for the following tasks:

* **Moving‑vehicle detection & tracking** – detect moving objects in each frame using a lightweight background‑subtraction approach and assign consistent IDs across frames using a simple centroid‑based tracker. This avoids heavy model downloads and runs entirely locally.
* **License‑plate recognition (ANPR)** – extract and recognize license plates from detected vehicles.
* **Masked‑person & periocular/eye recognition** – detect people, classify mask usage, and identify individuals using only the eye region when masks are present.
* **Event logging & dashboard visualization** – log detection events with metadata and display them through a local dashboard.

The system is designed to be **fully local** with no external cloud dependencies. All computation happens on devices you control, and models are loaded from local files. Configuration allows for adapting to different camera specifications (resolution, FPS, GPU availability, NIR support).

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

## Ethical Considerations

While this platform is designed for legitimate security purposes, please be mindful of privacy and legal regulations. Ensure that deployment complies with local laws governing video surveillance, data retention, and personal identification. Avoid misuse, particularly in ways that could infringe on individual rights or freedoms.
