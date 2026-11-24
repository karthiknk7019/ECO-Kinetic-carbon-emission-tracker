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

