"""Test suite for Track D deliverables: overlay renderer, dashboard app, and benchmark."""

from pathlib import Path
import json
import shutil
import cv2
import pandas as pd
import pyarrow.parquet as pq
import pytest
from streamlit.testing.v1 import AppTest

# Ensure repo root is on sys.path
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_overlay import render_overlay
from scripts.benchmark import run_benchmark


def test_telemetry_fixture_exists_and_valid():
    """Verify fixture exists and matches telemetry contract."""
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "telemetry_fixture.parquet"
    assert fixture_path.exists(), "telemetry_fixture.parquet must exist"
    table = pq.read_table(fixture_path)
    df = table.to_pandas()
    assert len(df) > 100
    assert df["cls"].nunique() == 5
    assert set(["IDLE", "ACCEL", "CRUISE", "DECEL", "UNINIT"]).issubset(set(df["state"].unique()))


def test_render_overlay_on_fixture(tmp_path: Path):
    """Test render_overlay on synthetic canvas using fixture data."""
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "telemetry_fixture.parquet"
    output_video = tmp_path / "test_overlay.mp4"

    out = render_overlay(
        video_path=tmp_path / "non_existent.mp4",
        telemetry_path=fixture_path,
        output_path=output_video,
        max_seconds=2.0,
    )
    assert output_video.exists()
    assert output_video.stat().st_size > 1000

    cap = cv2.VideoCapture(str(output_video))
    assert cap.isOpened()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n_frames > 20
    cap.release()


def test_dashboard_app_renders_safely():
    """Verify Streamlit dashboard runs without exceptions on fixture data."""
    at = AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"))
    at.run(timeout=10)
    assert not at.exception, f"Streamlit app raised an exception: {at.exception}"
    assert len(at.title) > 0
    assert "Eco-Kinetic" in at.title[0].value


def test_dashboard_image_intelligence_mode_renders_safely():
    """Verify Image Intelligence mode in Streamlit dashboard renders cleanly."""
    at = AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"))
    at.run(timeout=10)
    assert not at.exception
    # Select Image Intelligence mode in sidebar
    radios = at.sidebar.radio
    assert len(radios) > 0
    radios[0].set_value("Image Intelligence")
    at.run(timeout=10)
    assert not at.exception
    assert len(at.title) > 0
    assert "Image Intelligence" in at.title[0].value


def test_dashboard_app_resilient_to_missing_artifacts(tmp_path: Path):
    """Verify Streamlit dashboard runs cleanly even if artifacts dir is missing."""
    artifacts_dir = REPO_ROOT / "artifacts"
    backup_dir = REPO_ROOT / "artifacts_test_backup_tmp"

    if artifacts_dir.exists():
        artifacts_dir.rename(backup_dir)

    try:
        at = AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"))
        at.run(timeout=10)
        assert not at.exception, f"Streamlit failed with missing artifacts: {at.exception}"
    finally:
        if backup_dir.exists():
            backup_dir.rename(artifacts_dir)


def test_benchmark_runner(tmp_path: Path):
    """Verify benchmark runner generates valid perf_report.json."""
    report_file = tmp_path / "perf_report.json"
    rep = run_benchmark(output_path=str(report_file), n_frames=5)
    assert report_file.exists()
    assert "model_size_mb" in rep
    assert "avg_fps_cpu" in rep
    assert "peak_ram_mb" in rep
    assert "telemetry_bytes_per_min" in rep
