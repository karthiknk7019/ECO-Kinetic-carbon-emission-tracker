# Project Handoff & Technical Status Report
**Project:** Eco-Kinetic Traffic Carbon Digital Twin  
**Hackathon:** DSU Code Clash — Track 05: Carbon Footprint Intelligence Platform  
**Architecture Paradigm:** Lightweight Edge CCTV Retrofit • CPU-First Deployment • Zero-PII Telemetry Uplink • Deterministic Carbon Engine + Schema-Constrained LLM Dispatcher  

---

## 1. Executive Summary & System Architecture

This project retrofits existing municipal CCTV traffic camera fleets into a real-time carbon intelligence system without requiring new physical hardware or video streaming over the WAN. 

```
                                      +---------------------------------------------+
                                      |             EDGE JUNCTION BOX               |
                                      |                                             |
[ Existing CCTV Fleet ]               |  +---------------------------------------+  |
       │                              |  | 1. Perception & Tracking (CPU)        |  |
       ▼ (RTSP / Local Video)         |  |    - Fine-tuned YOLO (IDD-5 classes)   |  |
+---------------------+               |  |    - ByteTrack tracking               |  |
| 1080p/720p 30 FPS   | ─────────────►|  +---------------------------------------+  |
+---------------------+               |                       │                     |
                                      |                       ▼                     |
                                      |  +---------------------------------------+  |
                                      |  | 2. Kinematics State Machine           |  |
                                      |  |    - Length-calibrated speed proxy    |  |
                                      |  |    - EMA smoothing (alpha=0.3)        |  |
                                      |  |    - States: IDLE, ACCEL, CRUISE,     |  |
                                      |  |      DECEL (gated at >= 15 frames)    |  |
                                      |  +---------------------------------------+  |
                                      |                       │                     |
                                      |                       ▼                     |
                                      |  +---------------------------------------+  |
                                      |  | 3. Deterministic Carbon Twin Engine   |  |
                                      |  |    - CPCB/BS-VI factor lookup         |  |
                                      |  |    - Cumulative grams CO2 / vehicle   |  |
                                      |  |    - 60s Corridor Aggregation         |  |
                                      |  |    - 4x4 Spatial Choke Grid Ranking   |  |
                                      |  +---------------------------------------+  |
                                      +-----------------------┬---------------------+
                                                              │
                                    Telemetry-Only Uplink (JSON / Parquet)
                                    [NO VIDEO EGRESS / NO PII]
                                                              │
                                                              ▼
                                      +---------------------------------------------+
                                      |       ICCC / MUNICIPAL OPERATIONS           |
                                      |                                             |
                                      |  +---------------------------------------+  |
                                      |  | 4. GenAI Municipal Dispatcher         |  |
                                      |  |    - Schema-constrained advisory      |  |
                                      |  |    - Signal timing & corridor policy  |  |
                                      |  |    - Deterministic CO2 savings inject |  |
                                      |  |    - Automatic rules fallback         |  |
                                      |  +---------------------------------------+  |
                                      |                       │                     |
                                      |                       ▼                     |
                                      |  +---------------------------------------+  |
                                      |  | 5. Streamlit Command Dashboard        |  |
                                      |  |    - Real-time KPIs & Choke Matrix    |  |
                                      |  |    - Synced overlay video player      |  |
                                      |  |    - What-If counterfactual simulator |  |
                                      |  +---------------------------------------+  |
                                      +---------------------------------------------+
```

---

## 2. What Is DONE (Completed Milestones & Assets)

### ✅ M0 — Scaffolding, Environment & Edge CPU Constraints
- **Zero-CUDA Edge Thesis:** Confirmed CPU-only environment (`torch==2.14.1+cpu`, `torchvision==0.29.1+cpu`), with `--extra-index-url https://download.pytorch.org/whl/cpu` pinned in `requirements.txt`. Zero `nvidia-*` wheels installed.
- **Config Management:** Fully parameterised YAML configuration files:
  - `config/app.yaml`: Active classes (`car`, `motorcycle`, `auto_rickshaw`, `bus`, `truck`), `frame_skip=2`, `imgsz=640`, confidence, kinematic thresholds, and camera settings.
  - `config/emission_factors.yaml`: CPCB/EPA BS-VI empirical emission matrix (`idle_g_per_min`, `cruise_g_per_km`, `accel_multiplier`, `mean_length_m`) with citations.
  - `config/camera_calibration.yaml`: Calibration zones and homography placeholders.
- **Video Dataset Audition:** Audited candidate clips (`assets_workspace/candidates/`) for Indian intersection dynamics, auto-rickshaw density, and stationary CCTV perspective. Selected junction video saved at `assets_workspace/junction.mp4`.

### ✅ M-T — Fine-Tuning & 5-Class Indian Road Taxonomy
- **Dataset Preparation:** Converted India Driving Dataset (IDD-FGVD) annotations into YOLO format using `scripts/convert_to_yolo.py`.
- **Model Training & Distribution:** Fine-tuned `yolo26n` on IDD to produce a custom 5-class detector:
  - Model path: `models/finetuned/best.pt` (and lightweight base weights `yolo26n.pt`).
  - Active classes: `car`, `motorcycle`, `auto_rickshaw`, `bus`, `truck`.
  - ONNX Exports: `models/finetuned/best.onnx` (FP32, 43.29 FPS) and `models/finetuned/best_int8.onnx` (INT8 quantized, 33.12 FPS).
- **A/B Evaluation & Gate Validation:** `scripts/ab_eval.py` executed to verify auto-rickshaw detection retention while safeguarding light vehicle recall. Auto-rickshaw detection contact sheet generated (`ab_autos.png`).
- **Model Governance:** Detailed technical documentation committed in `docs/model_card.md` and `docs/data_provenance.md`.

### ✅ M1 — Perception & Kinematics Pipeline
- **Detector & Tracker:** `perception/detector.py` and `perception/tracker.py` integrate Ultralytics YOLO with ByteTrack (`bytetrack.yaml`), dynamically mapping class IDs (`model.names ∩ active_classes`).
- **Kinematic State Engine:** `perception/kinematics.py`:
  - Speed proxy: Centroid displacement normalised by bounding box width, converted to km/h via class-specific mean vehicle lengths.
  - Exponential moving average (EMA $\alpha=0.3$) for noise rejection.
  - 4-state state machine: `IDLE` ($v < 2\text{ km/h}$ for $> 3\text{ s}$ after motion), `ACCEL`, `CRUISE` ($15\text{--}50\text{ km/h}$), `DECEL`.
  - Gated at track age $\ge 15\text{ frames}$ to prevent tracking churn noise.
- **Pipeline Execution Script:** `scripts/run_pipeline.py` processes video and outputs `telemetry.parquet` and `telemetry_meta.json`.
- **Synthetic Telemetry Generator:** `scripts/make_synthetic_telemetry.py` creates robust deterministic synthetic telemetry datasets (`artifacts/telemetry_synthetic.parquet`) for testing without heavy video dependencies.

### ✅ M2 — Twin Carbon Engine & Corridor Aggregations
- **Emission Engine:** `twin/emission_engine.py` implements mode-bin emission accumulation (g CO2) based strictly on kinematic state and class parameters.
- **Corridor Analytics:** `twin/corridor.py`:
  - 60-second sliding window corridor roll-ups (total kg CO2, idle share %, heavy stop events).
  - 4x4 spatial grid decomposition with cell ID ranking to highlight thermal choke zones.
  - Append-only event extractor (`events.jsonl`).
  - Strict payload generation function `to_dispatcher_payload(...)` ready for Teammate D / Dispatcher consumption.

### ✅ M3 — Visual Overlay Video
- **Renderer Script:** `scripts/render_overlay.py` burns state-coloured bounding boxes (Green=CRUISE, Amber=ACCEL, Red=IDLE, Blue=DECEL), track IDs, class tags, and cumulative grams CO2 directly into video.
- **Artifact:** Generated and committed `artifacts/overlay.mp4`.

### ✅ M4 & M5 — Streamlit Command Dashboard & What-If Twin
- **Dashboard Application:** `dashboard/app.py`:
  - 5-KPI metric cards: *Vehicles Tracked*, *Cumulative CO2*, *Idle Share %*, *Stop Events*, *Top Choke Zone*.
  - Synced playback: Time window scrub slider coupled with video playback and kinematic activity bar charts.
  - 4x4 spatial choke-point ranking table.
  - Directive feed UI with LLM and rules-fallback visual badges.
  - Methodology sidebar with full factor transparency and assumptions.
- **Counterfactual Policy Simulator (`twin/scenarios.py`):**
  - Instantaneous vectorized re-computation (< 1 second response).
  - Individual sliders for fleet electrification (EV share 0–100% for car, motorcycle, auto_rickshaw, bus, truck).
  - Signal policy toggles: Adaptive green extension (0–15s) and Heavy Vehicle priority corridor.
  - Outputs baseline vs. scenario CO2, avoided CO2, percentage abatement, and class-by-class audit table.

### ✅ M7 (Partial) — Edge Benchmarking & Deployment Architecture
- **Benchmarking Suite:** `scripts/benchmark.py` outputs `artifacts/perf_report.json`:
  - Model size: 5.19 MB
  - CPU FPS: 28.47 FPS (PyTorch CPU) / 43.29 FPS (ONNX FP32)
  - Memory: 1.75 MB peak RAM footprint
  - Telemetry bandwidth: ~30.9 KB/min (ultra-low bandwidth telemetry uplink)
- **Deployment Documentation:** `docs/deployment.md` covering ICCC integration, ATCS tie-ins, edge hardware BOM, and privacy architecture.

### ✅ Test Suite Health
- **27/27 Golden Path Tests Passing** (`.venv/bin/pytest -q`):
  - `tests/test_placeholder.py`: Config schemas & keys verification.
  - `tests/test_probe.py`: Video codec, FPS, and dimension checks.
  - `tests/test_convert_to_yolo.py`: IDD dataset bounding box conversion tests.
  - `tests/test_kinematics.py`: Synthetic vehicle stopped/accelerating state transition tests.
  - `tests/test_twin.py`: Emission accumulation (±5% golden hand calculations), corridor aggregator, and dispatcher payload schema tests.
  - `tests/test_dashboard_and_benchmark.py`: Benchmark JSON schema and dashboard loading verification.

---

### ✅ M6 — GenAI Dispatcher Module & Strict Carbon Injection
- **Pydantic Contracts & Mocks:** `dispatcher/schema.py` implements frozen contracts (`ActionType`, `Action`, `Directive`, `TopChokeZone`, `CorridorPayload`) with UUID4 hex `directive_id`, UTC ISO timestamp, validation bounds, exported `docs/contracts/directive.schema.json`, and 10 diverse mock directives in `artifacts/directives_mock.jsonl`.
- **Deterministic Rules Engine:** `dispatcher/rules_fallback.py` triggers on corridor congestion or heavy vehicle density; uses single-source-of-truth carbon math directly via `twin/scenarios.py::compute_action_savings`.
- **Schema-Constrained LLM Client:** `dispatcher/llm_dispatcher.py` implements `LLMProposal` with **zero numeric emissions fields** (structurally preventing LLM hallucinated savings), 8s timeout, 1 retry, post-validation Python CO2 injection, append-only `artifacts/directives.jsonl`, and `--offline` flag.
- **Guardrail & Robustness Tests:** Tested across 20 payloads, proving LLM carbon injections are ignored, corrupt schemas fall back gracefully, and no-key execution is 100% schema-valid.

### ✅ End-to-End Real CCTV Video Telemetry & Live Aggregation
- **Perception on Real Video:** `scripts/run_pipeline.py` executed on `assets_workspace/junction.mp4` (96.5s duration, 2412 frames at 25 FPS) producing `artifacts/telemetry.parquet` (2344 rows, 124 tracks).
- **Realistic Class Distribution:** Verified all 5 classes (`motorcycle: 1165`, `car: 780`, `truck: 346`, `auto_rickshaw: 46`, `bus: 7`).
- **Corridor Events & Directives:** Emitted `artifacts/events.jsonl` (22 events). Tuned `rules_fallback` to real traffic distribution — Window 1 activates heavy vehicle restriction and green wave saving 1.2467g CO2, exactly matching `apply_scenario`.
- **Pipeline Flexibility:** Added `--mock` CLI flag to `scripts/run_pipeline.py` while keeping real CCTV inference as the default path.

### ✅ M7 — Complete Documentation, Architecture & Deployment Topology
- **Comprehensive README:** Full rewrite of `README.md` featuring Mermaid architecture diagram, CLI quickstart, Edge $\rightarrow$ ICCC deployment topology, CPCB/ARAI BS-VI equations, mode-bin formulas, strict privacy statement (zero PII, no ANPR, edge-only RAM inference), and known limitations.
- **Hygiene & Artifacts:** `.env` gitignored, `.env.example` committed, large `.mp4` binaries excluded from git tracking, realistic dual-mode fallback directives pre-populated in `artifacts/directives.jsonl`.
- **Test Suite Health:** **54/54 Golden Path Tests Passing** (`.venv/bin/pytest -q` in ~28s).

### ✅ Modern Next.js Control Center: MunicipalDispatchFeed.tsx
- **Directive Cards Stream:** Real-time chronological incoming advisory directive feed with timestamp, junction identifier, and dynamic urgency indicators (`CRITICAL INTERVENTION`, `HIGH PRIORITY`, `MODERATE`, `ROUTINE`).
- **High-Contrast Mode Badges:** Pill badges showing `Mode: Gemini 1.5 Flash` (purple/indigo with sparkle icon) or `Mode: Rules-Engine Fallback` (amber with shield alert icon).
- **Confidence & Rationale Bounds:** Visual progress gauge for confidence (0–100%) and concise municipal rationale box strictly bounded to $\le 300\text{ characters}$ (schema compliant).
- **Concrete Action Chips:** Clear visual badges for:
  - `extend_green` (e.g., `+10s on Lane N`)
  - `green_wave_priority` (e.g., `Corridor E-W Synchronized`)
  - `heavy_vehicle_restriction` (e.g., `Restrict Heavy Vehicles >10t`)
  - `no_action` (`No Action: Monitoring Baseline`)
- **Deterministic CO2 Savings Badge:** Prominent badge highlighting avoided grams CO2 (`⚡ 1,246.7 g CO2 Avoided`) with an interactive tooltip explaining: *"Computed via CPCB mode-bin engine, never LLM hallucination"*.
- **Interactive Operator Dispatch & ATCS/SCATS Transmission:** One-click "Approve & Push to ATCS / SCATS" with simulated transmission ping animation (`animate-ping` beacon) updating status to "Dispatched to Signal Controller", and "Dismiss / Reject" or "Recall Phase".
- **Multi-Dimensional Filters & Search:** Live search across junction IDs, directive IDs, actions, and rationale keywords, plus dual-row status (`Pending`, `Dispatched`, `Dismissed`) and mode (`Gemini 1.5 Flash`, `Rules Fallback`) filter tabs with badge counts.
- **Production Validation:** Zero ESLint warnings/errors and clean static build via `npm run build` in `dashboard-web/`.

### ✅ Traffic Image Intelligence (Single-Frame Photograph Analysis)
- **Perception Backend (`perception/image_inference.py`):** Reusable CPU-first single-image analyzer (`TrafficImageAnalyzer`) with cached YOLO weights, rigorous format validation (`.jpg`, `.jpeg`, `.png`, `.webp`), byte size limits (20 MB), and dynamic class resolution.
- **Accurate Model Taxonomy:** Automatically intersects configured `active_classes` with loaded weights; transparently reports missing classes (e.g. stock COCO `auto_rickshaw` limitation) without hallucinating detections.
- **Strict Guardrails:** Guaranteed absence of fabricated speeds, accelerations, motion states, or emissions for still images.
- **Streamlit Integration (`dashboard/app.py`):** Sidebar navigation mode offering image upload, confidence slider, process button, original vs annotated preview, KPI cards, class breakdown chart, detection table, and JPG/CSV/JSON exports.
- **Test Suite Health:** **72/72 tests passing** (`pytest -q` in ~33s) with dedicated coverage in `tests/test_image_intelligence.py`.

---

## 3. What HAS TO BE DONE (Pending Tasks & Open Gaps)

### 🟢 Priority 1: Milestone M8 — Demo Video Recording & Git Release Freeze
- **Demo Backup Video:** Record a 2–3 minute video recording (`artifacts/demo_backup.mp4`) demonstrating:
  - Dashboard overview and live metric cards.
  - Video overlay playback with dynamic state tracking.
  - What-If counterfactual scenario slider adjustments showing live carbon abatement.
  - Real-time municipal dispatcher directive stream with both LLM and fallback badges.
- **Git Tag:** Create tag `demo-v1` at freeze hour.

---

## 4. Key Constraints & Guardrails to Pass to the Next LLM

When handing off this project to another LLM, enforce the following non-negotiable rules:
1. **CPU-First Dev & Deploy:** Never install CUDA torch packages or `nvidia-*` libraries in this repository. All PyTorch calls must pass `device="cpu"`.
2. **Deterministic Carbon Calculation:** The LLM dispatcher must NEVER calculate carbon figures or fuel savings. All mathematics must be calculated in Python using `config/emission_factors.yaml`.
3. **Strict Privacy by Design:** Never introduce license plate recognition (ANPR), facial recognition, or video upload APIs. Only numerical telemetry (bounding box coordinates, class, speed, state) leaves the processing boundary.
4. **Zero Unapproved Dependencies:** Do not add arbitrary new packages to `requirements.txt` without necessity. Standard packages (`pydantic`, `google-generativeai` or `litellm`, `pandas`, `numpy`, `streamlit`) are already present in `.venv`.
5. **Always Run Tests with Virtualenv Python:** Use `.venv/bin/pytest -q` or ensure virtualenv is activated; system Python does not have project dependencies installed.

---

## 5. File & Directory Reference Map

| Path | Status | Purpose |
|---|---|---|
| `config/app.yaml` | Complete | Master application configuration, class map, thresholds |
| `config/emission_factors.yaml` | Complete | CPCB/EPA BS-VI empirical vehicle emission parameters |
| `config/camera_calibration.yaml` | Complete | Camera zone polygons and homography schema |
| `models/finetuned/best.pt` | Complete | Custom IDD-trained 5-class YOLO vehicle detector |
| `models/finetuned/best.onnx` | Complete | FP32 ONNX export for lightweight edge inference |
| `models/finetuned/best_int8.onnx` | Complete | INT8 quantized ONNX model |
| `perception/detector.py` | Complete | NMS-free YOLO inference with active class filtering |
| `perception/tracker.py` | Complete | ByteTrack tracker wrapper |
| `perception/kinematics.py` | Complete | Speed proxy, EMA smoothing, 4-state kinematic classifier |
| `perception/image_inference.py` | Complete | Reusable single-image vehicle detector & export utilities |
| `twin/emission_engine.py` | Complete | Mode-bin CO2 emission accumulator |
| `twin/corridor.py` | Complete | 60s window rollups, 4x4 choke points, dispatcher payload |
| `twin/scenarios.py` | Complete | Counterfactual policy engine (EV mix, green extension) |
| `dispatcher/schema.py` | Complete | Pydantic contracts for actions, directives, payloads |
| `dispatcher/rules_fallback.py` | Complete | Deterministic rule-based dispatcher fallback |
| `dispatcher/llm_dispatcher.py` | Complete | Schema-constrained LLM client + Python CO2 inject |
| `dashboard/app.py` | Complete | Streamlit command UI with Video Twin & Image Intelligence modes |
| `tests/test_image_intelligence.py` | Complete | Unit & integration tests for single-image perception backend |
| `dashboard-web/src/components/MunicipalDispatchFeed.tsx` | Complete | GenAI advisory feed & ATCS/SCATS dispatch control center |
| `scripts/run_pipeline.py` | Complete | Video $\rightarrow$ tracking $\rightarrow$ kinematics $\rightarrow$ `telemetry.parquet` |
| `scripts/render_overlay.py` | Complete | Video annotator with state colors and CO2 values |
| `scripts/benchmark.py` | Complete | Measures CPU FPS, memory, telemetry bandwidth |
| `docs/deployment.md` | Complete | One-pager edge deployment & ICCC integration doc |
| `docs/model_card.md` | Complete | Technical card for fine-tuned IDD vehicle detector |
| `docs/contracts/directive.schema.json` | Complete | Exported Pydantic v2 JSON Schema for frontend |
| `README.md` | Complete | Comprehensive project readme with architecture & run steps |
| `artifacts/perf_report.json` | Complete | Measured benchmark numbers |
| `artifacts/overlay.mp4` | Complete | Sample state-overlay video |
| `artifacts/telemetry.parquet` | Complete | 96.5s real CCTV perception telemetry ledger |
| `artifacts/events.jsonl` | Complete | Extracted corridor tracking and stop events |
| `artifacts/directives_mock.jsonl` | Complete | 10 diverse mock directives for frontend contract |
| `artifacts/directives.jsonl` | Complete | Append-only municipal directives log with fallback entries |
| `artifacts/demo_backup.mp4` | **Pending** | 2-3 minute backup video recording |
| `dashboard-web/public/junction.mp4` | Complete | Active raw CCTV video stream (auto-synced) |
| `dashboard-web/public/overlay.mp4` | Complete | H.264 fast-start AI overlay video feed (auto-synced from render_overlay.py) |
| `scripts/swap_video.py` | Complete | CLI utility to swap CCTV video, transcode H.264, and refresh web assets |
| `dashboard-web/src/app/api/video/upload/route.ts` | Complete | REST API route for instant multipart video upload & persistence |

### 📹 Video Swapping Workflows (At Will)

#### Method 1: Directly in the Web UI
1. Click the **"↑ Swap Video"** button located in the CCTV player top bar.
2. Select any `.mp4`, `.webm`, `.mkv`, or `.mov` video file from your local machine.
3. The video instantly loads via browser `URL.createObjectURL` and begins playing with real-time HUD overlays.
4. It is automatically persisted to `dashboard-web/public/junction.mp4` and `data/videos/junction.mp4` via the `/api/video/upload` endpoint so it remains active across reloads.
5. Click **"Reset"** at any time to return to the default junction feed.

#### Method 2: Via Terminal CLI
Run the automated swap utility:
```bash
python scripts/swap_video.py /path/to/any_video.mp4
```
*(Automatically normalizes codec to 720p H.264 `+faststart`, updates both public web feeds, and re-renders `artifacts/overlay.mp4`).*


