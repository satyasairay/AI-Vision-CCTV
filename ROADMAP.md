# AI-Vision-CCTV Roadmap

This roadmap outlines a three-phase plan to mature the AI-Vision-CCTV platform. Each phase balances capability growth with operational discipline, aiming to meet production-grade expectations for fraud, abuse, and physical security monitoring.

---

## Phase 1 – Solid Foundations (0–2 months)

> **Status (Oct 2025):** Delivered — tracker upgraded with Kalman/Hungarian association, detector registry introduced, violence detection moved to 3D CNN, periocular embeddings refreshed, metrics & calibration tooling shipped, and pytest coverage established.

### 1. Tracking & Detection Reliability
- **Upgrade tracker**: Replace the centroid placeholder with a true Deep SORT or ByteTrack implementation (Kalman filter, appearance embeddings, IOU gating).
- **Detector abstraction**: Introduce a pluggable detection registry so each camera can select YOLOv8/YOLO-NAS/OpenVINO models without code changes.

### 2. Analytics Hardening
- **Real violence detection model**: Swap heuristics for a trained action-recognition model (e.g., SlowFast or transformer-based) curated for security scenarios.
- **Eye recognition revamp**: Integrate an actual biometric embedding pipeline with dataset enrollment tooling.
- **Calibration toolkit**: Build a wizard to define pixels-per-meter, zones, and stop-lines directly from sample frames.

### 3. Observability & Validation
- **Automated testing**: Establish pytest suites with synthetic frames covering detection, tracking, and analytics outputs.
- **Metrics & health**: Expose Prometheus metrics (frame rate, queue depth, model latency) and integrate with Grafana dashboards.
- **CI gating**: Add lint/test workflows to prevent regressions.

---

## Phase 2 – Operator Experience & Governance (2–5 months)

### 1. Production Dashboard
- **Web UI overhaul**: Move beyond the Streamlit prototype to a real-time web dashboard (e.g., React + WebSocket/WebRTC) featuring live video panes, bounding-box overlays, alert feed, and acknowledgment workflows.
- **Configuration management**: Allow controlled edits to watchlists, zones, model versions, and analytics toggles with role-based access and audit trails.
- **Privacy controls**: Expose opt-out zones, anonymization options, and retention policies as first-class UI elements.

### 2. Event Distribution & Workflow
- **Event bus integration**: Publish rule hits to Kafka/MQTT for downstream alerting and SOAR platforms.
- **Notification channels**: Integrate email/SMS/Slack alerting with escalation policies.
- **Case management hooks**: Provide APIs to link events to fraud/abuse investigation tooling.

### 3. Edge & Infrastructure Readiness
- **Dockerized deployments**: Deliver container images optimized for CPU/GPU and NVIDIA Jetson targets, with model caching and health probes.
- **Configuration as code**: Store deployment configs in Git-backed repositories; enforce reviews for sensitive changes.
- **Secrets & compliance**: Integrate secret management (e.g., HashiCorp Vault/Azure Key Vault) and capture privacy impact assessments.

---

## Phase 3 – Intelligence & Scale (5–12 months)

### 1. Adaptive Intelligence
- **Model switcher automation**: Enable real-time swapping of detectors per environment (day/night, weather) using the existing scaffold.
- **Self-calibration**: Employ SLAM/perspective estimation to auto-calibrate camera geometry.
- **Anomaly detection**: Layer unsupervised detection (e.g., autoencoders) to flag unseen behaviors beyond rule-based triggers.

### 2. Enterprise-Grade Operations
- **Multi-tenancy**: Support segregated tenants with per-tenant configs, storage, and access controls.
- **Data lake & analytics**: Stream events and metadata into a lakehouse for longitudinal fraud/abuse analytics.
- **SOC/SIEM integration**: Normalize events to STIX/TAXII or vendor schema and push to SIEM/SOC platforms.

### 3. Assurance & Governance
- **Red-team scenarios**: Run adversarial simulations (masking, spoofing, laser interference) to validate defenses.
- **Continuous validation**: Implement nightly regression jobs on curated video corpora with accuracy drift tracking.
- **Regulatory alignment**: Map features to regional privacy/surveillance laws; provide audit-ready reporting.

---

## UI Guidance – Dos & Don’ts (Always Applicable)

### Do
- **Architect for decoupling**: Keep the analytics pipeline and UI loosely coupled via APIs or message buses so UI lag never stalls detection.
- **Secure media delivery**: Serve video streams over authenticated, encrypted channels; watermark or redact sensitive content.
- **Provide operator workflow support**: Include alert triage, acknowledgment, escalation, and annotation within the UI.
- **Emphasize auditability**: Log every UI-driven change—watchlist edits, privacy overrides, retention updates—with operator identity and timestamp.
- **Design for accessibility**: Ensure interfaces meet accessibility standards (WCAG) to serve diverse SOC teams.

### Don’t
- **Embed UI display directly in the pipeline loop**: Rendering frames with `cv2.imshow` inside the main loop ties up processing threads and breaks headless deployments; push frames to dedicated UI services instead.
- **Stream raw feeds without controls**: Unsecured or unredacted streaming can breach privacy laws; always combine with encryption, masking, and RBAC.
- **Bypass configuration governance**: Avoid ad-hoc scripts or manual JSON edits that skip approvals; rely on managed configuration flows with review and rollback.
- **Ignore operator context**: Overloading screens with raw detections without prioritization or context leads to alert fatigue; invest in summarization, filtering, and relevance scoring.
- **Neglect watchdogs**: UIs without health indicators or failover messaging leave operators blind during outages; integrate heartbeat indicators and fallback messaging.

---

## Effort Snapshot

| Phase | Duration (est.) | Highlights | Primary Skill Sets |
|-------|-----------------|-----------|--------------------|
| Phase 1 | 0–2 months | Tracking upgrade, analytics hardening, test/monitor stack | CV engineering, Python, MLOps |
| Phase 2 | 2–5 months | Full UI build, event bus, edge deploys, governance | Full-stack, DevOps, SecOps |
| Phase 3 | 5–12 months | Adaptive intelligence, enterprise scale, assurance programs | Applied ML, Data engineering, Security architecture |

This roadmap sets the course from demo-ready pipelines to a resilient, scalable surveillance intelligence platform. Adjust timelines as team size and compliance needs evolve, but keep the phase boundaries in sight to maintain momentum and architectural clarity.
