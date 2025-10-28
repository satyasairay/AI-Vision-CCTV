# Phase-1 Enhancements Roadmap

This document tracks the post-foundation enhancements requested for Phase 1. Each stage lists objectives, implementation notes, testing requirements, and current status.

---

## Stage 1 – Deep Re-ID Embeddings ✅ *(Completed Oct 2025)*

**Goals**
- Replace histogram-only appearance features with configurable deep embeddings.
- Expose configuration knobs for choosing between colour histograms and ReID backbones.
- Ensure fallbacks when custom weights are unavailable.
- Update documentation and testing to reflect the new pipeline behaviour.

**Implementation highlights**
- Added `tracking.appearance` configuration block with `mode`, `model_path`, `device`, and `embedding_dim` controls.
- Introduced `tracking/appearance.py` providing `ColorHistogramExtractor` and `TorchReIDExtractor` implementations.
- Updated `DeepSortTracker` to accept pluggable feature extractors; pipelines now respect the new configuration.
- Enhanced README/HOWTO and created pytest coverage for both histogram and ReID flows (`tests/test_tracker.py`).

**Testing**
- `python -m pytest tests/test_tracker.py`
- `python -m pytest` (full suite)
- `.\.venv\Scripts\python.exe run_pipeline.py --config configs/default.yaml --no-display --max-frames 200`

---

## Stage 2 – Camera Health Integration ✅ *(Completed Oct 2025)*

**Implementation highlights**
- Added heartbeat evaluation and state-change detection (`CameraHealthMonitor.evaluate/force_down`).
- Pipelines now emit `camera_down`/`camera_recovered` rule contexts and record Prometheus gauge `aivision_camera_health_status`.
- Configurable via `analytics.camera_health` (`enable`, `timeout`, `camera_id`).
- Documentation refreshed (README/HOWTO) with usage guidance.

**Testing**
- `python -m pytest tests/test_camera_health.py`
- `python -m pytest`
- `.\.venv\Scripts\python.exe run_pipeline.py --config configs/default.yaml --no-display --max-frames 200`
- `.\.venv\Scripts\python.exe run_multi_pipeline.py --config configs/default.yaml --no-display`

---

## Stage 3 – Bespoke Violence Model Tuning ✅ *(Completed Oct 2025)*

**Implementation highlights**
- Added optional `analytics.violence.weights_path` so deployments can load domain-specific R3D weights (with checksum-aware `scripts/setup.sh` automation).
- Extended README/HOWTO with tuning guidance and introduced pytest coverage (`tests/test_fight_detection.py`) to verify custom weight loading.
- Violence detector now surfaces warnings when custom weights are missing or incompatible.

**Testing**
- `python -m pytest tests/test_fight_detection.py`
- `python -m pytest`
- `.\.venv\Scripts\python.exe run_pipeline.py --config configs/default.yaml --no-display --max-frames 200`
- `.\.venv\Scripts\python.exe run_multi_pipeline.py --config configs/default.yaml --no-display`

---

## Testing & Documentation Checklist (for every enhancement)
- [x] Update README.md & HOWTO.md with new configuration and run instructions.
- [ ] Update dashboard docs if UI changes (Stage 2/3).
- [x] Add/extend pytest coverage; run `python -m pytest`.
- [x] Execute pipeline smoke test (`run_pipeline.py --no-display --max-frames …`).
- [ ] Update release notes once all Stage 2/3 items are complete.
