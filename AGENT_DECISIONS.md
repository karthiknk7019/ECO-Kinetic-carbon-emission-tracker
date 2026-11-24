# Agent Decisions Log

Log of architectural choices, deviations, and judgment calls during development.

- Installed CPU-only PyTorch (`torch==2.14.1+cpu` and `torchvision==0.29.1+cpu`) from the PyTorch CPU index to maintain the lightweight CCTV edge deployment thesis and avoid heavy CUDA runtime dependencies.
- Verified `yolo26n.pt` downloaded and executed NMS-free inference directly with `ultralytics==8.4.177` without model substitution.
- Implemented `test_placeholder.py` to validate existence, parsing, and required schema keys for `app.yaml`, `emission_factors.yaml`, and `camera_calibration.yaml`.
- Verified CPU-only environment: uninstalled `nvidia-ml-py` so `pip list | grep -i nvidia` is empty and `torch.cuda.is_available()` is False. Pinned `--extra-index-url https://download.pytorch.org/whl/cpu` on line 1 of requirements.txt with `torch==2.14.1+cpu` and `torchvision==0.29.1+cpu`.
- Delivered M6 (GenAI Dispatcher Module): Implemented `dispatcher/schema.py`, `dispatcher/rules_fallback.py`, and `dispatcher/llm_dispatcher.py`. Enforces strict Pydantic contracts with auto_rickshaw target class support, deterministic Python avoided CO2 calculations, 1-retry schema recovery, and automatic rules fallback for offline/air-gapped edge deployments.
- Implemented `tests/test_dispatcher.py` covering 20 mock payloads gate (100% schema validity), bounds enforcement, offline fallback, and append-only JSONL persistence.
- Updated `config/emission_factors.yaml` with official ARAI / CPCB source apportionment citations for auto-rickshaws.
- Authored comprehensive `README.md` documenting end-to-end architecture (Mermaid), installation, methodology math, CPCB factor citations, zero-PII privacy guarantee, and deployment topology.
- Recreated the project virtualenv with the pinned CPU-only dependency set; uninstalled the transitive `nvidia-ml-py` wheel to preserve the zero-CUDA edge guardrail (`torch.cuda.is_available() == False`).
- Redesigned the Streamlit operations console: added `.streamlit/config.toml`, custom KPI/hero/status components, Plotly state donut and 4×4 choke heatmap, quick-action row, synced time-window jump controls, interactive directive dispatch/dismiss cards with filters/archive, scenario reset/export controls, and image-mode preset/publish actions.
- Fixed a Streamlit widget-state bug in the What-If reset by moving state mutation into an `on_click` callback so it executes before widget instantiation.
- Verified all 73 tests pass after the UI refresh and visually validated both console modes in headless Chrome.

- Trimmed the Image Intelligence sidebar to remove duplicated controls (presets, clear, publish, downloads) that already exist in the main panel, leaving a compact engine status, taxonomy expander, active-image summary, and unique Re-run/Reload actions.
- Validated the real upload path through browser file inputs on the bundled CCTV presets; removed the out-of-domain Delhi aerial image from the preset gallery after it produced under-detection and car/auto misclassification, while retaining the file for limitation testing.
- Recorded the final M8 demo (`artifacts/demo_backup.mp4`, 120.5s) from the finished UI and tagged the release `demo-v1`.
- Validated the full pipeline on the 45s stable-camera junction clip (60 tracks, 25 processed FPS, 61.9 g CO2) and stress-tested four unrelated CCTV stills; cross-camera performance collapsed to near-all-`car` detections with missed buses/trucks, confirming the weights are viewpoint-specific and documenting the need for per-camera fine-tuning.
- Fixed the dashboard corridor CO2 KPI to sum per-track final cumulative grams (was reporting only the single highest track: 9.7 g vs. the correct 61.9 g on the 45s clip).
- Bug-fix and declutter pass on the Streamlit console: replaced post-widget `focus_time` mutations with `on_click` callbacks (Start/Midpoint/End no longer crash), persisted advisory confirmations across reruns, cleared stale image results on validation failure, hashed image bytes for result caching (API-key edits no longer trigger inference), removed the no-op What-If jump button, removed the misleading annualised-extrapolation card, and dropped the duplicate Original Photograph tab.
- Second declutter pass on the video console: removed the duplicate ACTIVE VIEWPORT sidebar card (the workspace radio and hero already state the mode) and the technical video metadata strip (filename/size/codec). Let the choke-point table auto-size instead of reserving blank rows, and relabelled the tracked-vehicle foot from "analysed clip" to "telemetry span" so the 117s ledger is not confused with the 45s playback file.
