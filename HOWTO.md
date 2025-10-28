# AI-Vision-CCTV HOWTO

This guide walks you through every major feature in the project, the current implementation status, and how to try each capability locally. Follow it end-to-end if you’re onboarding or need a refresher.

---

## 1. Environment & Setup

- **Create virtual environment**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate
  ```
- **Install dependencies**
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  ```
- **Optional GUI support**  
  The default `requirements.txt` installs `opencv-python-headless`. To view frames in a window, replace it with `opencv-python`:
  ```powershell
  .\.venv\Scripts\python.exe -m pip uninstall opencv-python-headless
  .\.venv\Scripts\python.exe -m pip install opencv-python
  ```
- **Generate demo video (placeholder)**
  ```powershell
  .\.venv\Scripts\python.exe scripts\generate_sample_video.py
  ```
  Creates `sample_data/sample_video.mp4` for quick trials.

---

## 2. Core Pipelines

### 2.1 Single-Camera Pipeline (`run_pipeline.py`)
- **What it does**: Orchestrates camera input, detection, tracking, recognition, analytics, rule evaluation, logging, and optional display.
- **Run**
  ```powershell
  .\.venv\Scripts\python.exe run_pipeline.py --config configs\default.yaml
  ```
- **Headless mode** (avoids `cv2.imshow`)
  ```powershell
  .\.venv\Scripts\python.exe run_pipeline.py --config configs\default.yaml --no-display
  ```
- **Limit run length** (handy for demos)
  ```powershell
  ... --max-frames 200
  ```
- **Where to look**: Logs and captured data go to the directory specified under `storage.log_dir` in the config (default `logs/`).

### 2.2 Multi-Camera Pipeline (`run_multi_pipeline.py`)
- **What it does**: Spawns a thread per camera configuration; shares loggers, analytics, and IoT outputs.
- **Run**
  ```powershell
  .\.venv\Scripts\python.exe run_multi_pipeline.py --config configs\default.yaml
  ```
- **Headless flag available**
  ```powershell
  ... --no-display
  ```
- **Where to configure cameras**: Either a single `camera:` block or a `cameras:` list in the YAML file. Each entry supports `source`, `frame_width/height`, etc.

---

## 3. Feature Catalog

| Feature | Module | Status | How to Enable / Use |
|---------|--------|--------|---------------------|
| **Camera Input (RTSP & Files)** | `camera_adapters/rtsp_camera.py`, `camera_adapters/local_file.py` | Implemented | Set `camera.source` (or each `cameras[n].source`) to an RTSP URL (`rtsp://...`) or file path. |
| **Vehicle Detection** | `detection/vehicle_detector.py` | Implemented w/ YOLO fallback | Provide model path in `detection.vehicle_model_path`. Without a model, falls back to background subtraction. |
| **Person Detection & Mask Classification** | `detection/person_detector.py`, `detection/mask_classifier.py` | Detection implemented, mask clf placeholder | Toggle by setting `detection.person_model_path` and optional `mask_model_path` for masked/no-mask classification. |
| **Tracking (Deep SORT)** | `tracking/deep_sort.py` | Simplified placeholder | Uses centroid matching. Works for demos but not production-grade. |
| **ANPR (License Plate OCR)** | `recognition/anpr.py` | Implemented (EasyOCR dependency) | Requires EasyOCR model download (handled via `pip install easyocr`). Called automatically after vehicle detection. |
| **Eye Recognition** | `recognition/eye_recognition.py` | Placeholder | Needs trained embeddings. Populate `recognition.eye_database` in config and provide `eye_model_path`. |
| **Re-Identification** | `recognition/reid.py` | Placeholder | Hooks available; not plugged into pipeline yet. |
| **Rule Engine** | `rules/rule_engine.py` | Implemented | Configure rules under `routing_rules` in YAML. Custom handlers registered in pipeline. |
| **Event Logging (JSONL / SQLite)** | `storage/event_logger.py`, `storage/database_logger.py` | Implemented | Switch via `storage_backend` (`jsonl` or `sqlite`). Configure paths in `storage` and `database` sections. |
| **Audit Logging** | `storage/audit_logger.py` | Implemented | Used by dashboard actions (e.g., watchlist updates). Stores under `logs/` by default. |
| **Analytics: Speed Estimation** | `analytics/speed_estimator.py` | Implemented | Enable via `analytics.speed.enable`, set calibration parameters. |
| **Analytics: Dwell Time** | `analytics/dwell_time.py` | Implemented | Enable with zone + threshold in config. |
| **Analytics: Wrong-Way Detection** | `analytics/wrong_way.py` | Implemented | Configure expected direction vector and threshold. |
| **Analytics: Duplicate Plate** | `analytics/duplicate_plate.py` | Implemented | Requires plate recognition; set `analytics.duplicate_plate.enable`. |
| **Analytics: Crowd Density** | `analytics/crowd_density.py` | Implemented | Enable plus zone & person threshold. |
| **Analytics: Stop-Line Violation** | `analytics/stop_line_violation.py` | Implemented | Define stop line coordinates and `red_light` in config. |
| **Analytics: Violence Detection** | `analytics/fight_detection.py` | Placeholder heuristic | Uses bounding box motion heuristic; replace with ML model for production. |
| **Analytics: Weather Adapter** | `analytics/weather_adapter.py` | Implemented | Adjusts brightness/gamma before processing. Enable in `analytics.weather`. |
| **Analytics: Adaptive Frame Skipper** | `analytics/adaptive_frame_skipper.py` | Implemented | Skips frames during inactivity; configure thresholds. |
| **Analytics: Privacy Manager** | `analytics/privacy.py` | Implemented | Define `no_record_zones` and `blur_non_watchlist` to redact frames. |
| **Analytics: Camera Health Monitor** | `analytics/camera_health.py` | Placeholder | Checks frame timestamps; requires integration with pipeline (partially wired). |
| **Analytics: Model Switcher** | `analytics/model_switcher.py` | Placeholder | Infrastructure exists; no dynamic switching logic yet. |
| **Analytics: IoT Integration** | `analytics/iot_integration.py` | Implemented stub | Writes to console; extend to control relays/alarms. |
| **Analytics: Event Stats** | `analytics/event_stats.py` | Utility function | Example to tally events; not wired into pipeline output. |
| **Dashboard (Streamlit)** | `dashboard/app.py` | Implemented prototype | Displays watchlists, events, allows adding plates. Run `streamlit run dashboard/app.py`. |
| **Scripts** | `scripts/setup.sh`, `scripts/run_demo.sh`, `scripts/enroll_identity.py`, `scripts/generate_sample_video.py` | Setup + placeholder tooling | `enroll_identity.py` helps build the eye-recognition database (requires manual data). |
| **Tests** | `tests/` | Placeholder | Currently empty; ready for future unit tests. |

---

## 4. Configuration Tips (`configs/default.yaml`)

- **Camera block**
  ```yaml
  camera:
    source: "sample_data/sample_video.mp4"
    frame_width: null
    frame_height: null
    target_fps: 15
  ```
  Override per camera when using the multi-pipeline.

- **Detection & Recognition**
  ```yaml
  detection:
    vehicle_model_path: "models/vehicle_detector.pt"
    person_model_path: "models/person_detector.pt"
    mask_model_path: "models/mask_classifier.pt"  # optional
  recognition:
    anpr_ocr_engine: "easyocr"
    anpr_languages: ["en"]
    eye_model_path: "models/eye_recognition.pt"
    eye_database: {}
  ```

- **Analytics toggles**  
  Each feature has an `enable` flag plus parameters (zones, thresholds, etc.). Turn features on gradually to measure performance impact.

- **Routing rules**  
  ```yaml
  routing_rules:
    - type: "license_plate_watchlist"
      watchlist: ["ABC123"]
    - type: "person_identity_watchlist"
      watchlist: ["John Doe"]
  ```
  Extend with custom handlers via `RuleEngine.register_handler`.

---

## 5. Dashboard Walkthrough

1. Activate environment and run Streamlit:
   ```powershell
   .\.venv\Scripts\activate
   streamlit run dashboard/app.py
   ```
2. UI highlights:
   - **Sidebar Watchlist**: see and add new plates (writes back to config and audit log).
   - **Events Table**: reads `logs/events.jsonl`.
   - **Stats / Frames**: extend dashboard to pull latest frame images saved by pipeline threads.

---

## 6. Extending & Customizing

- **Swap detection models**: Drop your weights into `models/` and update paths in config.
- **Add new rules**: Implement custom function, register via `rule_engine.register_handler`, and reference the handler name in YAML.
- **Integrate real IoT**: Replace stubs in `analytics/iot_integration.py` with GPIO/REST/MQTT hooks.
- **Upgrade tracking**: Replace `DeepSortTracker` placeholder with a full Deep SORT or ByteTrack implementation.
- **Testing hooks**: Create tests under `tests/` that import modules directly (no special scaffolding required).

---

## 7. Known Placeholders / TODOs

- `tracking/deep_sort.py`: simplified centroid matching instead of true Deep SORT.
- `recognition/eye_recognition.py`: requires real embeddings and inference model.
- `recognition/reid.py`: scaffolded, not plugged in.
- `analytics/fight_detection.py`: heuristic-only; replace with real model.
- `analytics/model_switcher.py`: logic for dynamic model swapping not implemented.
- `analytics/camera_health.py`: needs integration with actual heartbeat/alarm outputs.
- `tests/`: placeholder package; add unit/integration tests as components mature.
- `scripts/setup.sh`: TODO for downloading model weights.

---

## 8. Quick Trouble-Shooting

- **Relative import errors**: Always activate venv and run modules from project root. Use `python run_pipeline.py ...` (absolute imports) or rename package and run via `python -m`.
- **`cv2.imshow` crashes**: Install full `opencv-python` or run with `--no-display`.
- **Missing sample video**: Re-run `scripts/generate_sample_video.py` or point config to your footage.
- **Slow inference**: Without GPU, set YOLO-based detectors to more efficient models or disable analytics you don’t need.

---

You’re ready to experiment. Start the single-camera pipeline, turn on the features you care about in `configs/default.yaml`, and watch the logs (or dashboard) to validate functionality. For deeper development, tackle the placeholders listed above. Happy hacking!
