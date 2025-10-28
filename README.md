## Overview

This project implements a **modular road‑security platform** designed to run entirely on local infrastructure. It integrates CCTV video streams for the following tasks:

* **Moving-vehicle detection & tracking** – detect moving objects in each frame using either a lightweight background-subtraction approach **or**, when local weights are provided, a **YOLO-based detector** for higher accuracy. Vehicles are tracked across frames with a Kalman/Hungarian multi-object tracker that combines motion and colour appearance cues.
* **License‑plate recognition (ANPR)** – extract and recognize license plates from detected vehicles.
* **Masked-person & periocular/eye recognition** – detect people using a HOG-based detector (or an optional YOLO model), classify mask usage with a lightweight CNN (if weights are available), and identify individuals using only the eye region. MobileNet-based embeddings keep identities in a local `eye_database` or external `.npz`, and can be enrolled via the provided script.
* **Event logging, metrics & dashboard visualization** – log detection events with metadata, expose Prometheus metrics, and display insights through a local dashboard and calibration wizard.

The system is designed to be **fully local** with no external cloud dependencies. All computation happens on devices you control, and models are loaded from local files. Configuration allows for adapting to different camera specifications (resolution, FPS, GPU availability, NIR support) and enabling or disabling advanced analytics. A wide range of security features are implemented via modular components (see **Features** below). You can turn each analytic on or off in `configs/default.yaml`.

For this demo environment, detection and tracking are designed to run locally without external downloads. When no model weights are supplied, the system uses OpenCV's background subtraction for vehicles and a HOG‑based pedestrian detector for persons. When local YOLO weights are specified in `configs/default.yaml`, the system automatically loads them for higher‑accuracy detections. Similarly, if you provide a `mask_classifier.pt`, the person detector will classify mask usage. All models run on your local CPU or GPU.

Person detection in the demo uses OpenCV's built‑in HOG‑based pedestrian detector. If you provide YOLO weights, the system will switch to a deep detector automatically. Mask classification is performed via an optional lightweight classifier; if no weights are provided, the label defaults to ``unknown``. The periocular recognition module extracts a normalized feature vector from the eye region and compares it against a local `eye_database`. Use the enrollment script (see below) to add new identities.

## Features

The platform supports a variety of surveillance features, each implemented in its own module. Most analytics can be enabled or disabled via the `analytics` section of the YAML configuration. Here is a summary of the available features and where to find their implementations:

| Feature | Description | Implementation |
|---------|-------------|---------------|
| **Vehicle & Person Detection** | Uses background subtraction or YOLO for vehicles (`detection/vehicle_detector.py`), HOG or YOLO for persons (`detection/person_detector.py`). Automatically switches based on model availability and hardware via `ModelSwitcher` (`analytics/model_switcher.py`). | `detection/vehicle_detector.py`, `detection/person_detector.py` |
| **Tracking** | Associates detections across frames using a Kalman/Hungarian tracker with colour appearance cues (`tracking/deep_sort.py`). | `tracking/deep_sort.py` |
| **License Plate Recognition (ANPR)** | Recognizes number plates via EasyOCR with support for multiple languages (`recognition/anpr.py`). | `recognition/anpr.py` |
| **Eye/Periocular Recognition** | Identifies masked persons with MobileNet-based eye embeddings; enroll identities with `scripts/enroll_identity.py`. | `recognition/eye_recognition.py`, `scripts/enroll_identity.py` |
| **Speed Estimation & Over‑Speed Alerts** | Calculates object speed from centroid trajectories and triggers events above a configurable limit. | `analytics/speed_estimator.py` |
| **Wrong‑Way Detection** | Flags vehicles moving against an expected direction vector. | `analytics/wrong_way.py` |
| **Loitering/Dwell Time** | Monitors how long objects remain in a zone; alerts if above threshold. | `analytics/dwell_time.py` |
| **Duplicate Plate Detection** | Detects repeated license plates within a time window. | `analytics/duplicate_plate.py` |
| **Crowd Density** | Counts persons in a region and flags overcrowding. | `analytics/crowd_density.py` |
| **Stop‑Line/Red‑Light Violations** | Triggers when vehicles cross a defined line while a red‑light flag is set. | `analytics/stop_line_violation.py` |
| **Violence/Fight Detection** | 3D CNN (R3D-18) classification over short clips cropped around persons; tune thresholds per camera. | `analytics/fight_detection.py` |
| **Weather & Lighting Adaptation** | Applies gamma correction to dark frames. | `analytics/weather_adapter.py` |
| **Adaptive Frame Skipping** | Skips frames during idle periods to save resources. | `analytics/adaptive_frame_skipper.py` |
| **Privacy Redaction & No‑Record Zones** | Blurs detections not on watchlists and excludes designated zones. | `analytics/privacy.py` |
| **IoT Integration** | Sends events to external devices or logs for automation. | `analytics/iot_integration.py` |
| **Audit Logging** | Records administrative actions (e.g., watchlist changes) for compliance. | `storage/audit_logger.py` |
| **SQLite Event Storage** | Persists events in a structured database instead of JSON lines. | `storage/database_logger.py` |
| **Camera Health Monitoring** | Checks if camera feeds are alive. | `analytics/camera_health.py` |
| **Cross-Camera Re-Identification** | Prototype colour-histogram matching across cameras. | `recognition/reid.py` |
| **Prometheus Metrics** | Exposes frame latency, detection counts, and track statistics for observability. | `monitoring/metrics.py` |
| **Calibration Wizard** | Streamlit UI to draw dwell zones, stop lines, and privacy regions. | `tools/calibration_app.py` |

Each analytic has a corresponding handler in the rule engine. To activate a feature, set its `enable` flag to `true` in the configuration and adjust its parameters (e.g., zones, thresholds). See **Configuration Guide** below for details.

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
| `tools/`           | Auxiliary tooling such as the Streamlit calibration wizard.                                   |
| `tests/`           | Unit tests covering the core functionality.                                                   |
| `sample_data/`     | Sample CCTV video and test images for development and demonstration.                          |

## Getting Started

### Installation

1. **Clone the repository** and change into its directory:

   ```bash
   git clone https://github.com/your‑org/road_security.git
   cd road_security
   ```

2. **Create a virtual environment** and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   The main dependencies are OpenCV, EasyOCR, Streamlit, and PyYAML. Optional dependencies such as PyTorch are only required for heavy models.

3. **Prepare model weights** (optional):

   * Place YOLO weights for vehicles/persons under `models/` and update `vehicle_model_path` and `person_model_path` in `configs/default.yaml`.
   * Place a mask classifier weight file at `models/mask_classifier.pt` if you wish to classify masks.
   * If you choose SQLite storage, create the database path specified in the config; it will be created automatically on first run.

4. **Review and customize the configuration** in `configs/default.yaml`. Pay particular attention to:

   * **Camera** section – specify your RTSP URL or local video file.
   * **Detection & Tracking** – point to your model weights and adjust `device` (cpu/cuda/auto).
   * **Analytics** – enable or disable features like speed estimation, loitering, stop‑line detection, violence detection, privacy redaction, etc. Configure zones and thresholds as needed.
   * **Routing Rules** – define watchlists and additional rules (e.g., over_speed) to trigger alerts.
   * **Storage Backend** – choose between `jsonl` and `sqlite` for event persistence.

### Running the Pipeline Locally

* **Single Camera**:

  ```bash
  python -m road_security.run_pipeline --config configs/default.yaml
  ```

  This command reads frames from the configured camera, runs detection/tracking/recognition, evaluates rules, logs events, applies privacy redaction, and optionally displays the processed frames in a window (press `q` to quit).

* **Multiple Cameras**:

  ```bash
  python -m road_security.run_multi_pipeline --config configs/default.yaml
  ```

  Define a `cameras` list in your YAML file with multiple `source` entries. A separate thread processes each feed concurrently. Processed frames are saved to `logs/latest_frame_<cam_id>.jpg` for dashboard streaming.

### Using the Dashboard

Run the dashboard via Streamlit (inside your virtual environment):

```bash
streamlit run road_security/dashboard/app.py
```

Navigate to `http://localhost:8501` in your browser. The dashboard includes:

* **Watchlist Management** – Add plates or identities via the sidebar. Changes are persisted to the YAML config and logged in `logs/audit.jsonl`.
* **Live Feeds** – Displays the latest processed frame per camera. Click *Refresh Feeds* to reload.
* **Event Analytics & Logs** – Shows a bar chart of event counts by type and a table of recent events. Click *Refresh Logs* to reload.
* **Custom Rule Builder** – Create or update routing rules via a form. For watchlists, supply comma‑separated values; for numeric thresholds (e.g. over_speed), enter a number. After updating, restart the pipeline to apply changes.

### Deploying the Dashboard to Vercel

The heavy lifting (detection, tracking, recognition) is computationally intensive and not suitable for serverless platforms like Vercel. A recommended deployment strategy is:

1. **Run the pipeline on edge devices or a dedicated server** near your cameras. This machine performs detection and analytics and writes processed frames and event logs to a shared volume or cloud storage.
2. **Deploy the dashboard as a lightweight front‑end** on Vercel (or any static hosting) to visualize results. You can convert the Streamlit app into a static React/Vue page or use an SSR framework (Next.js) that fetches processed images and event data from your server via an API.
3. **Expose an API endpoint** on your analytics server to serve event logs and latest frames (e.g., via FastAPI). Configure CORS so the Vercel‑hosted dashboard can access it securely.

While it is technically possible to run Streamlit on Vercel using their Python runtime, you will quickly hit execution limits when performing real‑time video processing. Keep compute on your hardware and use Vercel only for visualisation.

## Configuration Guide

The YAML file `configs/default.yaml` contains all tunable settings. Key sections include:

* **camera / cameras** – Specify one or more sources (RTSP URL or file path). Per‑camera overrides can be defined under each entry.
* **detection** – Paths to vehicle and person detectors, inference device (cpu/cuda/auto), and optional mask classifier. The system automatically selects heavy or light models based on GPU availability.
* **tracking** – `max_age` controls how long a track is kept without detections; `distance_threshold` controls association sensitivity.
* **recognition** – `anpr_ocr_engine` (currently `easyocr`), `anpr_languages` list for multi‑language plate recognition, path to the eye recognition model, and the `eye_database` mapping identities to feature vectors.
* **storage_backend** – `jsonl` (default) writes events to `logs/events.jsonl` and associated images to JPEG files. `sqlite` writes to a SQLite database (`database.db`) with optional image directory. Both backends can be used simultaneously if desired.
* **analytics** – Nested settings for each analytic. Set `enable: true` to activate an analytic and configure its parameters (e.g., zones, thresholds, direction vectors, idle thresholds). See the inline comments in the file for examples.
* **routing_rules** – A list of rule objects. Each rule has a `type` field (matching a handler) and additional parameters (e.g., `watchlist`, `threshold`). Rules are evaluated in sequence for each detection context. Custom rules can be added via the dashboard.

## Logging & Storage

Detection events are logged with timestamps and contextual data (e.g., license plate text, speed, track ID) either to a JSONL file (`logs/events.jsonl`) or a SQLite database (`events.db`). Associated images are saved alongside if provided. Administrative actions (like adding watchlist entries or rules) are recorded in `logs/audit.jsonl` via the `AuditLogger`.

Event logs can be parsed with helper functions in `analytics/event_stats.py`. For example, use `event_counts_by_type()` to generate bar charts of event frequencies.

## Extending the Platform

The modular design makes it straightforward to add new analytics. To implement a new feature:

1. Create a new module under `analytics/` that encapsulates the computation.
2. Add a configuration section in `configs/default.yaml` with an `enable` flag and parameters.
3. Register a handler in `rules/rule_engine.py` (or within the pipeline) and add any necessary context fields.
4. Update the pipeline(s) to instantiate and call your analytic when enabled.
5. Expose visualisations or controls via the dashboard if relevant.

## Deployment Tips

* **Hardware acceleration:** Use a GPU for deep‑learning models whenever possible. The `ModelSwitcher` picks a heavy model when a GPU is available. On CPU‑only systems, rely on background subtraction and HOG detectors for acceptable performance.
* **Scalability:** For more than a handful of cameras, consider running the pipelines inside containers orchestrated by Kubernetes or Docker Compose. Use a message queue (e.g. RabbitMQ) and a database (e.g. TimescaleDB) for scalable event storage and processing.
* **Security:** Secure your RTSP streams and dashboard behind authentication. Use HTTPS/TLS and restrict access via firewall rules.

---

This project aims to serve as a flexible starting point for on‑premises video security analytics. Contributions and customizations are welcome. Please ensure any deployment complies with privacy regulations and ethical guidelines.

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
