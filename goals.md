Hackathon: DSU Code Clash · Track 05: Carbon Footprint Intelligence PlatformSprint: 36 effective hours. Demo freeze at h34 — hard stop.Deployment thesis: Software-only retrofit on the government's existing CCTVenforcement camera fleet. Lightweight enough for per-junction edge boxes or sharedICCC compute. Telemetry-only uplink (no video egress, no PII).Agent working mode: Execute milestones in order. After each milestone: run itsverification command, commit (conventional message), report status. If any criterionfails twice, STOP and report. If a milestone exceeds 1.5× its timebox, STOP andreport. Never silently deviate.
1. Mission

Ingest intersection video, track vehicles (yolo26n + ByteTrack), classify kinematicstate (IDLE / ACCEL / CRUISE / DECEL), convert state + class to CO2 via aconfig-driven emission matrix, run counterfactual what-ifs, and issue municipaldirectives via a schema-constrained LLM dispatcher (all carbon math deterministicPython — never the LLM). Everything must run CPU-first and produce a measurableperformance report supporting the "existing camera fleet" deployment claim. Vehicle taxonomy is 5 classes: auto_rickshaw, bus, car, motorcycle, truck (auto_rickshaw added via fine-tuning on Indian road data, performed on an external workstation). The repo consumes best.pt via a config-only flip, gated by an A/B evaluation.
2. Non-negotiable guardrails

    NEVER let the LLM compute emission numbers. LLM selects actions + rationale only.
    ENVIRONMENT SPLIT: The repo/dev/deploy environment is CPU-ONLY — torch installed from https://download.pytorch.org/whl/cpu, torch.cuda.is_available() == False, zero nvidia-* wheels, requirements.txt line 1 is '--extra-index-url https://download.pytorch.org/whl/cpu' with torch pinned to a +cpu local version so fresh clones never pull CUDA wheels. EXCEPTION: the model-training workstation (RTX 4060) may install CUDA torch wheels (--index-url .../whl/cu124) in a SEPARATE venv OUTSIDE the repo. No system CUDA Toolkit install anywhere. perception/ inference calls pass device=cfg['device'] explicitly.
    Privacy by architecture: no license-plate reading, no face/PII processing anywherein the codebase or docs. Design docs state video never leaves the processing node.
    Currency: INR. Fuel price, equivalences — config-driven, never hardcoded.
    All API keys via .env (provide .env.example). Never committed.
    Tests never call paid APIs — mock the LLM client.
    No new dependencies without user approval. Pin versions. Seed randomness.
    After M8 (freeze): bug fixes only, with user approval. No refactors, no "quickimprovements."
    Commit after every milestone. Never delete files without asking.
    ALWAYS update information in goals.md, PROJECT_HANDOFF.md, README.md, and relevant documentation whenever milestones, features, or components are added or updated.

3. Definition of Done

    pytest -q passes (golden-path tests only — this is a 36h sprint).
    streamlit run dashboard/app.py runs the full demo from committed artifacts.
    artifacts/perf_report.json exists and meets M7 gates (or gates miss aredocumented with measured numbers — measured-bad beats unmeasured).
    README contains: architecture diagram, run instructions, deployment topology,privacy statement, limitations (incl. auto-rickshaw caveat).
    artifacts/demo_backup.mp4 exists (2–3 min full user flow).

4. Repo structure

eco-kinetic-twin/├── config/│   ├── emission_factors.yaml   # per-class factors + source + year (CPCB/EPA-style)│   ├── camera_calibration.yaml # zone polygons + homography placeholder (schema only)│   └── app.yaml                # thresholds, class map {2,3,5,7}, frame_skip=2,│                               # imgsz=640, fuel_price_inr, LLM settings├── models/finetuned/ (data.yaml; *.pt gitignored)├── perception/ (detector.py, kinematics.py, pipeline.py)├── twin/       (emission_engine.py, corridor.py, scenarios.py)├── dispatcher/ (schema.py, llm_dispatcher.py, rules_fallback.py)├── dashboard/app.py├── scripts/│   ├── run_pipeline.py│   ├── export_model.py         # ONNX export│   ├── benchmark.py            # writes artifacts/perf_report.json│   ├── prepare_dataset.py│   ├── train.py│   └── ab_eval.py├── tests/├── data/videos/                # gitignored — Indian intersection clip, license noted├── artifacts/  (overlay.mp4, telemetry.parquet, events.jsonl, directives.jsonl,│                perf_report.json, demo_backup.mp4)├── docs/deployment.md          # one-pager: topology, cost, privacy, ICCC integration├── docs/model_card.md          # (authored on the training workstation, committed after review)├── requirements.txt, .env.example, README.md
5. Emission model spec (implement exactly; defaults cited CPCB/EPA-style urban

   BS-VI figures — citation strings live in the YAML)Per tracked vehicle per frame (dt from video timestamps):

    speed proxy: v_bls = |Δcentroid| / bbox_width_px; v_kmh = v_bls *MEAN_LENGTH_M[class] * 3.6; EMA alpha=0.3.
    state machine (3s sliding window):  IDLE: v < 2 km/h for > 3s (only after once exceeding 5 km/h)  ACCEL: dv/dt above threshold, or v 5->15 km/h within 5s  CRUISE: 15–50 km/h steady; DECEL: falling beyond threshold
    rates (g/s):  CRUISE/DECEL: cruise_g_per_km * v_kmh / 3600  IDLE:         idle_g_per_min / 60  ACCEL:        cruise rate * accel_multiplier
    Ledger entry only if track age >= 15 frames.Defaults (config/emission_factors.yaml):  class        idle_g_per_min  cruise_g_per_km  accel_mult  mean_length_m  car          22              150              1.6         4.5  motorcycle   8               80               1.4         2.0  auto_rickshaw 12             95               1.4         3.0  bus          45              1000             1.8         12.0  truck        55              900              2.0         10.0 (auto_rickshaw row implemented by track C in config/emission_factors.yaml with source TODO). State: auto_rickshaw counts as LIGHT; heavy = bus + truck only. Perception spec amendment: class IDs are NEVER hardcoded — resolve at runtime as model.names ∩ active_classes. With stock COCO weights, auto_rickshaw is simply absent → behavior unchanged. Known limitation (README + config comment): COCO has no auto-rickshaw class;3-wheelers are detected as car and accounted as such. Future work: lightweight3-wheeler classifier; PUC cross-referencing of anomalous idlers.Corridor aggregates per 60s window: kg CO2, idle share %, heavy-vehicle stopevents/h, per-zone idle-minutes -> ranked choke points.Event log: append-only artifacts/events.jsonl.

6. LLM dispatcher contract

Trigger: every 30 twin-seconds OR on event (top choke zone idle share > 40%, or anyheavy vehicle cumulative idle > 90s).Input payload (built in Python): intersection_id, timestamp, queue_length_per_lane,idle_ratio, heavy_vehicle_density, co2_g_per_min, top_choke_zone.Output (Pydantic; out-of-bounds -> 1 retry -> rules_fallback):{  "timestamp": str, "intersection_id": str,  "actions": [{"type": "extend_green"|"green_wave_priority"|               "heavy_vehicle_restriction"|"no_action",               "lane": "N"|"S"|"E"|"W"|null, "seconds": 0–15, "target_class": str|null}],  "estimated_co2_saved_g": float,   # computed in Python, injected post-validation  "confidence": 0–1, "rationale": "<=300 chars", "mode": "llm"|"rules_fallback"}Append to artifacts/directives.jsonl; render in dashboard feed with a compliancesummary paragraph.
7. Milestones (36h)
M0 — Env & assets (0–2h)

    venv + pinned requirements (ultralytics, opencv-python, numpy, pandas, pydantic,streamlit, plotly, python-dotenv, pyarrow).
    yolo26n.pt weights; Indian intersection clip (Pexels preferred; license noted inREADME) at data/videos/junction.mp4.
    Verify: python -m perception.detector --smoke on 100 frames, exit 0.

M-T — Fine-tune track (parallel, external workstation, owned by track A)

    fine-tune yolo26n on IDD fine-grained Indian-road data (5-class taxonomy). GATE: A/B eval on junction.mp4 — ship fine-tuned ONLY IF auto_rickshaw detected plausibly AND car+motorcycle detections drop <10% vs stock COCO. Gate fails → keep stock yolo26n + README limitation note.
    config/app.yaml gains: active_classes: [car, motorcycle, auto_rickshaw, bus, truck] and device: "cpu"; model stays "yolo26n.pt" until the gate passes, then flips to "models/finetuned/best.pt".

M1 — Perception + kinematics (2–8h)

    model.track(tracker="bytetrack.yaml"), verbose=False,frame_skip=2 with dt from timestamps. Class IDs are NEVER hardcoded — resolve at runtime as model.names ∩ active_classes. With stock COCO weights, auto_rickshaw is simply absent → behavior unchanged.
    Speed proxy + EMA + 4-state machine per §5. Min track age 15 frames.
    Tests: synthetic stopped vehicle + accelerating truck -> expected state sequences.
    Verify: pytest tests/test_kinematics.py -q; 60s pipeline run producestelemetry.parquet with columns [video_time_s, track_id, cls, cx, cy, speed_kmh,state, bbox_w_px], no NaNs, IDs manually spot-checked as stable.

M2 — Emission engine (8–12h)

    Per-frame accumulation per §5; per-vehicle cumulative g; corridor aggregates;choke ranking; events.jsonl.
    Verify: pytest tests/test_emission.py -q — hand-computed 10-vehicle golden testwithin ±5%.

M3 — Pipeline + overlay (12–16h)

    scripts/run_pipeline.py -> artifacts/overlay.mp4: boxes colored by state(CRUISE=green, ACCEL=amber, IDLE=red, DECEL=blue), track IDs, cumulative grams.
    Verify: input processed >= 1.5x realtime on dev CPU at frame_skip=2; overlaylegible at 720p; artifacts written.

M4 — Dashboard (16–21h)

    Streamlit: st.cache_resource telemetry; synced playback — time slider scrubsoverlay.mp4 and charts together; metric cards (cumulative kg CO2, idle share %,heavy stop events/h, vehicles tracked); CO2/s line chart; choke-point table.
    Verify: loads < 5s; slider seek to t=0 and t=end without crash.

M5 — What-if shadow twin (21–25h)

    Controls: EV share per class (0–100%), green-extension toggle, heavy-vehiclerestriction toggle. Synthetic mix becomes ~50% motorcycle / 25% car / 15% auto_rickshaw / 5% bus / 5% truck. Vectorized NumPy recompute from telemetry, < 1s response.
    Show delta + annualized extrapolation + equivalences (INR fuel saved, trees).
    Verify: EV share 100% -> tailpipe CO2 ~ 0 (test asserts this).

M6 — GenAI dispatcher (25–29h) [COMPLETE]

    Full §6 contract + rules_fallback + dashboard directive feed + compliance summary. Dispatcher target_class enum gains "auto_rickshaw".
    Verify: 20 mock payloads -> 100% schema-valid after <= 1 retry; API disabled ->mode="rules_fallback" visible in UI.
    Status: COMPLETE. Pydantic v2 schemas in dispatcher/schema.py, deterministic rules fallback in dispatcher/rules_fallback.py, LLM dispatcher in dispatcher/llm_dispatcher.py with strict Python carbon injection. 47/47 tests passing.

M7 — Perf, deployment story, docs (29–33h) [COMPLETE]

    scripts/export_model.py -> ONNX; scripts/benchmark.py ->artifacts/perf_report.json: {model_size_mb, avg_fps_cpu, peak_ram_mb,telemetry_bytes_per_min}. Timebox INT8 attempt to 30 min.
    docs/deployment.md one-pager: topology diagram (junction edge box -> JSONtelemetry uplink -> ICCC ops center), per-junction cost estimate, privacystatement, ICCC/ATCS integration note, limitations.
    README: architecture, run instructions, methodology, factor citations, privacy,limitations. Dashboard methodology panel renders emission_factors.yaml + sources
        confidence band.
    Verify: gates in §3 met or measured-and-documented; pytest -q green.
    Status: COMPLETE. Full README with Mermaid diagram, quickstart, CPCB/ARAI equations & citations, strict privacy policy. Telemetry verified on real video. All tests green.

M8 — Demo freeze (33–34h, HARD STOP)

    Record artifacts/demo_backup.mp4 (2–3 min full flow). Tag demo-v1. After this:bug fixes only.

34–36h — Buffer. Dry runs only.
8. Fallbacks (apply only with user approval)

    INT8 fails -> FP32 ONNX + measured FPS (always report real numbers).
    CPU too slow -> imgsz 480, frame_skip 3; keep dt-based kinematics.
    Track ID churn severe -> min track age 30 frames; note in README.
    LLM down/quota exhausted -> rules_fallback, UI shows mode.
    Streamlit playback issues -> offline synced playback is the primary path anyway.

9. Out of scope — do NOT build

Multi-camera fusion, RTSP ingestion (mention as thin adapter in docs only),homography execution (config schema placeholder only), model fine-tuning, ANPR/plate reading/any PII, GPS matching, signal hardware control, auth, Docker, mobile.

10. Video Swapping & Playback Specification
- Raw CCTV sources reside in `data/videos/junction.mp4` and `dashboard-web/public/junction.mp4`.
- Re-running `scripts/render_overlay.py` automatically generates the H.264 web-compatible `overlay.mp4` with `+faststart` and syncs it to `dashboard-web/public/overlay.mp4`.
- `CctvPlayer.tsx` dynamically detects `video.duration` upon load and supports switching between `AI CCTV Twin`, `Raw CCTV`, and `Vector Twin`.

11. Image Intelligence Specification & Status [COMPLETE]
- **Perception Backend**: `perception/image_inference.py` (`TrafficImageAnalyzer`, `validate_and_decode_image`).
- **CPU-First Architecture**: Strictly runs on CPU using cached YOLO weights (`get_cached_yolo_model`).
- **Format Support & Validation**: Supports `.jpg`, `.jpeg`, `.png`, `.webp` (up to 20 MB) with dimension and corruption checks.
- **Dynamic Taxonomy**: Runtime resolution of `model.names ∩ active_classes`. Transparently warns when `auto_rickshaw` is missing in stock COCO weights.
- **Streamlit Integration**: Dual-mode navigation in `dashboard/app.py` preserving continuous video operations while adding single-frame inspection, KPI cards, class chart, detection table, and JPG/CSV/JSON exports.
- **Privacy & Physical Boundaries**: 100% local processing; zero PII; zero external cloud transmission; strictly guarantees zero fabricated speed, acceleration, or emission values from still photographs.
- **Status**: 72/72 tests passing (`pytest -q`). Full regression and unit coverage in `tests/test_image_intelligence.py`.

