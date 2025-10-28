# Phase 1 Implementation Plan

This plan details the execution path for Phase 1 of the AI-Vision-CCTV roadmap. It breaks the three Phase 1 themes into actionable workstreams, each with milestones, tasks, estimated effort, dependencies, and success criteria.

> **Status (Oct 2025):** Phase 1 delivered. Tracker upgraded with Kalman/Hungarian association, detector registry introduced, violence detection now 3D CNN-based, periocular recognition uses MobileNet embeddings with enrollment tooling, calibration wizard shipped, Prometheus metrics exposed, and a pytest suite covers core modules.

---

## Overview

| Workstream | Goal | Estimated Effort | Core Contributors |
|------------|------|------------------|-------------------|
| WS1 | Tracking & Detection Reliability | 3–4 weeks | Computer vision engineer, Python developer |
| WS2 | Analytics Hardening | 4–5 weeks | CV/ML engineer, data scientist |
| WS3 | Observability & Validation | 3 weeks | MLOps engineer, QA |

Total Phase 1 duration: ~6–8 weeks (parallelizable with 3 engineers).

---

## WS1 – Tracking & Detection Reliability

### Milestone 1: Full Deep SORT/ByteTrack Integration (Week 1–2)
- **Tasks**
  1. Benchmark available trackers (Deep SORT, ByteTrack) against demo footage.
  2. Integrate chosen library or custom implementation into `tracking/deep_sort.py`.
  3. Introduce embedding extractor (lightweight CNN, e.g., MobileNet) for appearance features.
  4. Add configuration toggles for tracker selection and thresholds.
  5. Update `run_pipeline.py` and `run_multi_pipeline.py` to surface new options.
- **Dependencies**: Access to sample videos with multi-object scenes; compute resources for embedding extraction.
- **Success Criteria**
  - Track IDs remain stable across occlusions in test footage.
  - Unit tests covering track creation, update, deletion.
  - Config documentation updated (`HOWTO.md`, `configs/default.yaml` comments).

### Milestone 2: Detector Abstraction Layer (Week 2–3)
- **Tasks**
  1. Design detection registry interface (`detection/base.py`).
  2. Implement adapters for YOLOv8 (Ultralytics), YOLO-NAS, and OpenVINO (CPU optimized).
  3. Update configs to reference detector names and per-model parameters.
  4. Implement graceful fallback logic (load failure → next best model).
- **Dependencies**: Model weights stored in `models/`; OpenVINO runtime (if used).
- **Success Criteria**
  - Pipeline can switch detectors via config only.
  - Model load failures captured with actionable errors.
  - Tests mocking each adapter confirm correct invocation.

---

## WS2 – Analytics Hardening

### Milestone 3: Violence Detection Upgrade (Week 3–5)
- **Tasks**
  1. Curate training dataset (CCTV violence vs. benign incidents).
  2. Train action recognition model (e.g., SlowFast, TSM, Video Swin).
  3. Implement inference service/class in `analytics/fight_detection.py`.
  4. Add batching/windowing to minimize latency.
  5. Evaluate precision/recall; set threshold defaults.
- **Dependencies**: Labelled dataset, GPU resources for training.
- **Success Criteria**
  - Recognition accuracy metrics documented (target >80% recall on benchmark).
  - Inference runs within budget (<100ms CPU/GPU depending on hardware).
  - Integration test showing alert trigger with sample clip.

### Milestone 4: Eye Recognition Revamp (Week 4–6)
- **Tasks**
  1. Gather/clean biometric dataset; define enrollment process.
  2. Implement embedding model (e.g., ArcFace-based) with liveness checks.
  3. Extend `recognition/eye_recognition.py` to load embeddings from secure storage.
  4. Provide `scripts/enroll_identity.py` upgrade to capture new embeddings.
  5. Add evaluation suite to measure FAR/FRR at chosen thresholds.
- **Dependencies**: Consent-cleared biometric data; security policy for storing embeddings.
- **Success Criteria**
  - FAR/FRR metrics within agreed thresholds (e.g., FAR <1%).
  - Enrollment script usable by operators with clear instructions.
  - Logs capture enrollment and usage events for audit.

### Milestone 5: Calibration Toolkit (Week 5–6)
- **Tasks**
  1. Prototype interactive calibration via Streamlit or standalone script (OpenCV imshow).
  2. Allow drawing zones, stop-lines, and selecting reference distances.
  3. Persist calibration outputs to YAML/JSON consumed by configs.
  4. Automate validation—generate overlay preview for operator confirmation.
- **Dependencies**: UI decision (temporary Streamlit vs. new tool), sample frames.
- **Success Criteria**
  - Operators can complete calibration end-to-end without code edits.
  - Generated configs validate against schema and load in pipeline.
  - Documentation/tutorial included in HOWTO.

---

## WS3 – Observability & Validation

### Milestone 6: Automated Testing Framework (Week 1–3)
- **Tasks**
  1. Set up pytest with fixtures for synthetic frames/detections.
  2. Add unit tests for detection outputs, tracker state transitions, analytics triggers.
  3. Implement regression tests with reference outputs (golden JSON/metrics).
  4. Configure coverage thresholds and reporting.
- **Dependencies**: Synthetic frame generator utilities, baseline outputs.
- **Success Criteria**
  - `pytest` passes locally and in CI.
  - Coverage baseline achieved (target >60% critical modules).
  - Clear contribution guide for writing new tests.

### Milestone 7: Metrics & Monitoring (Week 2–4)
- **Tasks**
  1. Instrument pipeline with Prometheus metrics (frame rate, inference latency, queue sizes).
  2. Expose `/metrics` endpoint (for Python processes) or push metrics into Prometheus gateway.
  3. Provide Grafana dashboard JSON with default panels.
  4. Add health checks (camera heartbeat, event logging status).
- **Dependencies**: Prometheus client libraries, staging environment to validate dashboards.
- **Success Criteria**
  - Metrics accessible in dev/staging.
  - Alerts configured for key thresholds (e.g., FPS drop, camera timeout).
  - Grafana dashboard committed and documented.

### Milestone 8: CI/CD Enhancements (Week 3)
- **Tasks**
  1. Implement GitHub Actions/GitLab CI pipeline running linting, tests, coverage.
  2. Add artifact generation (wheel/docker image) for nightly builds.
  3. Enforce branch protection requiring green CI.
- **Dependencies**: CI permissions, docker registry (if artifact upload required).
- **Success Criteria**
  - CI pipeline green across branches.
  - Artifacts accessible for manual QA.
  - Contributors acknowledge process in README/HOWTO.

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model performance gaps due to limited data | Delays in WS2 milestones | Reserve time for data acquisition; consider synthetic augmentation. |
| Tracker integration incompatibilities | Pipeline instability | Pilot integration behind config flag; maintain fallback to existing tracker until proven. |
| Metric overhead impacting FPS | Reduced throughput | Sample metrics at configurable intervals; benchmark overhead. |
| CI flakiness on GPU-dependent tests | Slower iteration | Use CPU-friendly mocks; run GPU tests nightly instead of per-PR. |

---

## Deliverables Checklist

- [ ] Upgraded tracking module with tests and docs.
- [ ] Detector registry with at least two backends.
- [ ] Violence detection model + evaluation report.
- [ ] Eye recognition pipeline + enrollment tooling.
- [ ] Calibration UI/tool with operator guide.
- [ ] Pytest suite and coverage reports.
- [ ] Prometheus metrics integration + Grafana dashboard.
- [ ] CI pipeline enforcing lint/tests.

Completion of the checklist marks Phase 1 readiness and unlocks Phase 2 (UI & governance enhancements). Adjust timelines based on team allocation, but maintain milestone order to reduce integration risk.
