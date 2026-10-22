"""Unit and contract tests for kinematics state machine and perception pipeline.

Runs without weights or video files.
"""

from collections import deque
import json
from pathlib import Path
import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from perception.kinematics import (
    DEFAULT_MEAN_LENGTH_M,
    classify_state,
    smooth_speed,
    update_track,
)
from scripts.run_pipeline import run_pipeline


def test_ema_sequence():
    """EMA hand-computed sequence test."""
    alpha = 0.3
    # Step 1: prev=0, raw=10 -> 0.3*10 + 0.7*0 = 3.0
    v1 = smooth_speed(0.0, 10.0, alpha)
    assert pytest.approx(3.0, abs=1e-5) == v1

    # Step 2: prev=3.0, raw=20 -> 0.3*20 + 0.7*3 = 6.0 + 2.1 = 8.1
    v2 = smooth_speed(v1, 20.0, alpha)
    assert pytest.approx(8.1, abs=1e-5) == v2

    # Step 3: prev=8.1, raw=20 -> 0.3*20 + 0.7*8.1 = 6.0 + 5.67 = 11.67
    v3 = smooth_speed(v2, 20.0, alpha)
    assert pytest.approx(11.67, abs=1e-5) == v3

    # Step 4: prev=11.67, raw=0 -> 0.3*0 + 0.7*11.67 = 8.169
    v4 = smooth_speed(v3, 0.0, alpha)
    assert pytest.approx(8.169, abs=1e-5) == v4


def test_first_two_observations_uninit():
    """First 2 observations → UNINIT."""
    cfg = {"cls": "car", "img_w": 640, "ema_alpha": 0.3, "state_thresholds": {"idle_v_kmh": 2.0}}
    dt_s = 0.1

    # Observation 1
    s1 = update_track(None, cx_norm=0.5, cy_norm=0.5, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
    assert s1.state == "UNINIT"
    assert s1.track_age_frames == 1
    assert s1.speed_kmh == 0.0

    # Observation 2
    s2 = update_track(s1, cx_norm=0.55, cy_norm=0.5, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
    assert s2.state == "UNINIT"
    assert s2.track_age_frames == 2


def test_stopped_after_motion():
    """Stopped-after-motion: 5s at 40 km/h then 5s at 0 → CRUISE → IDLE."""
    dt_s = 0.1
    cfg = {
        "cls": "car",
        "img_w": 640,
        "ema_alpha": 0.3,
        "mean_length_m": 4.5,
        "state_thresholds": {"idle_v_kmh": 2.0, "idle_min_seconds": 3},
    }

    # Desired raw speed ~40 km/h with physical velocity:
    # raw_v = (|delta_cx * 640| / (bbox_w * dt_s)) * 4.5 * 3.6 = 40
    delta_cx_40 = (40.0 * 50.0 * dt_s) / (4.5 * 3.6 * 640.0)

    state = None
    cx = 0.1

    # 1. 5s at ~40 km/h (50 steps at dt_s=0.1)
    cruise_states = []
    for _ in range(50):
        state = update_track(state, cx_norm=cx, cy_norm=0.5, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
        cx += delta_cx_40
        cruise_states.append(state.state)

    # After warm-up (age >= 30 frames / 3 seconds), state should be CRUISE
    assert cruise_states[-1] == "CRUISE"
    assert state.speed_kmh > 35.0
    assert state.hist_max_kmh > 35.0

    # 2. 5s at 0 km/h (50 steps at dt_s=0.1 with zero displacement)
    stop_states = []
    for _ in range(50):
        state = update_track(state, cx_norm=cx, cy_norm=0.5, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
        stop_states.append(state.state)

    # After vehicle stops for 5s (> 3s window), state must become IDLE
    assert stop_states[-1] == "IDLE"
    assert state.speed_kmh < 0.1


def test_accelerating_truck():
    """Accelerating truck: 0→45 km/h over 10s → ACCEL then CRUISE."""
    dt_s = 0.1
    cfg = {
        "cls": "truck",
        "img_w": 640,
        "ema_alpha": 0.3,
        "mean_length_m": 10.0,
        "state_thresholds": {"idle_v_kmh": 2.0, "idle_min_seconds": 3},
    }

    # Linear acceleration from 0 to 45 km/h over 100 steps (10 seconds)
    # delta = v_now - window[0]. Over 30 steps (3s), speed increases by ~13.5 km/h (> 5 km/h)
    state = None
    cx = 0.1

    states_during_accel = []
    for step in range(100):
        target_v = 45.0 * (step / 100.0)
        # raw_v = (delta_px / (bbox_w * dt_s)) * 10.0 * 3.6
        delta_cx = (target_v * 80.0 * dt_s) / (10.0 * 3.6 * 640.0)
        state = update_track(state, cx_norm=cx, cy_norm=0.5, bbox_w_px=80.0, dt_s=dt_s, cfg=cfg)
        cx += delta_cx
        states_during_accel.append(state.state)

    # In the middle/late acceleration phase (after window initialization), state must be ACCEL
    accel_slice = states_during_accel[40:95]
    assert "ACCEL" in accel_slice
    assert states_during_accel[-1] == "ACCEL"

    # Now maintain 45 km/h for 5s (50 steps)
    target_v = 45.0
    delta_cx_cruise = (target_v * 80.0 * dt_s) / (10.0 * 3.6 * 640.0)
    cruise_states = []
    for _ in range(50):
        state = update_track(state, cx_norm=cx, cy_norm=0.5, bbox_w_px=80.0, dt_s=dt_s, cfg=cfg)
        cx += delta_cx_cruise
        cruise_states.append(state.state)

    # After holding steady speed for > 3s, delta drops below 5 km/h -> CRUISE
    assert cruise_states[-1] == "CRUISE"


def test_never_exceeded_5kmh_not_idle():
    """Never-exceeded-5-kmh track does NOT become IDLE."""
    dt_s = 0.1
    cfg = {
        "cls": "car",
        "img_w": 640,
        "ema_alpha": 0.3,
        "mean_length_m": 4.5,
        "state_thresholds": {"idle_v_kmh": 2.0, "idle_min_seconds": 3},
    }

    # Crawl at 2.5 km/h (< 5.0 km/h) for 5s
    delta_cx_crawl = (2.5 * 50.0 * dt_s) / (4.5 * 3.6 * 640.0)
    state = None
    cx = 0.1

    for _ in range(50):
        state = update_track(state, cx_norm=cx, cy_norm=0.5, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
        cx += delta_cx_crawl

    assert state.hist_max_kmh <= 5.0

    # Stop at 0 km/h for 5s
    for _ in range(50):
        state = update_track(state, cx_norm=cx, cy_norm=0.5, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)

    # hist_max_kmh never exceeded 5.0 km/h, so condition hist_max_kmh > 5 is False -> NOT IDLE!
    assert state.state != "IDLE"
    assert state.state == "CRUISE"


def test_bbox_width_under_5_keeps_ema():
    """If bbox_w_px < 5: keep previous EMA."""
    cfg = {"cls": "car", "img_w": 640, "ema_alpha": 0.3}
    dt_s = 0.1

    s1 = update_track(None, cx_norm=0.5, cy_norm=0.5, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
    s2 = update_track(s1, cx_norm=0.55, cy_norm=0.5, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
    ema_before = s2.speed_kmh
    assert ema_before > 0.0

    # Large displacement but tiny bbox_w_px (< 5)
    s3 = update_track(s2, cx_norm=0.90, cy_norm=0.5, bbox_w_px=3.0, dt_s=dt_s, cfg=cfg)
    # Must preserve previous EMA
    assert s3.speed_kmh == ema_before


def test_2d_euclidean_motion():
    """Verify 2D Euclidean motion displacement calculation."""
    cfg = {
        "cls": "car",
        "img_w": 1000,
        "img_h": 1000,
        "use_2d_motion": True,
        "use_dt_in_speed": True,
        "ema_alpha": 0.3,
    }
    dt_s = 1.0
    # Move 0.03 along x (30px) and 0.04 along y (40px) -> 30-40-50 right triangle -> hypot = 50px
    s1 = update_track(None, cx_norm=0.10, cy_norm=0.10, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
    s2 = update_track(s1, cx_norm=0.13, cy_norm=0.14, bbox_w_px=50.0, dt_s=dt_s, cfg=cfg)
    # raw_v = (50px / (50px * 1.0s)) * 4.5m * 3.6 = 16.2 km/h
    # v_now = 0.3 * 16.2 + 0.7 * 0.0 = 4.86 km/h
    assert pytest.approx(4.86, abs=0.01) == s2.speed_kmh


def test_pipeline_with_injected_detector(tmp_path: Path):
    """Pipeline test with INJECTED fake detector_fn producing known boxes.

    Asserts:
      - telemetry schema matches exact column names and dtypes.
      - ~constant speed_kmh for constant pixel velocity.
      - telemetry_meta.json and perception_stats.json created with required keys.
    """
    # 1. Create a dummy synthetic video of 40 frames at 30 fps
    video_path = tmp_path / "test_junction.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(40):
        out.write(dummy_frame)
    out.release()

    # 2. Injected fake detector producing constant pixel velocity
    frame_counter = {"idx": 0}

    def fake_detector(frame: np.ndarray):
        idx = frame_counter["idx"]
        frame_counter["idx"] += 1
        # Constant horizontal movement of 0.01 normalized width per frame
        cx = 0.10 + (idx * 0.01)
        return [
            {
                "track_id": 42,
                "cls": "car",
                "cx_norm": cx,
                "cy_norm": 0.5,
                "bbox_w_px": 64.0,
                "conf": 0.95,
            }
        ]

    fake_detector.class_map = {0: "car", 1: "motorcycle", 2: "auto_rickshaw", 3: "bus", 4: "truck"}

    # 3. Create app config
    app_cfg_path = tmp_path / "app.yaml"
    app_cfg = {
        "model": "dummy",
        "device": "cpu",
        "active_classes": ["car", "motorcycle", "auto_rickshaw", "bus", "truck"],
        "imgsz": 640,
        "frame_skip": 2,
        "confidence": 0.30,
        "min_track_age_frames": 5,
        "ema_alpha": 0.3,
        "video_source": str(video_path),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "state_thresholds": {
            "idle_v_kmh": 2.0,
            "idle_min_seconds": 3,
            "cruise_low_kmh": 15,
            "cruise_high_kmh": 50,
        },
    }
    with open(app_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(app_cfg, f)

    # 4. Run pipeline
    artifacts_dir = tmp_path / "artifacts"
    stats = run_pipeline(
        input_source=video_path,
        output_dir=artifacts_dir,
        config_path=app_cfg_path,
        detector_fn=fake_detector,
    )

    assert stats is not None

    # 5. Assert Telemetry Parquet exists and matches exact schema
    parquet_path = artifacts_dir / "telemetry.parquet"
    assert parquet_path.exists(), "telemetry.parquet must exist"

    table = pq.read_table(parquet_path)
    expected_schema = pa.schema([
        ("video_time_s", pa.float64()),
        ("track_id", pa.int32()),
        ("cls", pa.string()),
        ("cx_norm", pa.float32()),
        ("cy_norm", pa.float32()),
        ("bbox_w_px", pa.float32()),
        ("bbox_h_px", pa.float32()),
        ("speed_kmh", pa.float32()),
        ("state", pa.string()),
        ("track_age_frames", pa.int32()),
    ])

    assert table.schema == expected_schema, f"Schema mismatch: {table.schema} vs {expected_schema}"

    df = table.to_pandas()
    assert len(df) > 0, "Telemetry dataframe should not be empty"
    assert df.isna().sum().sum() == 0, "Telemetry must not contain NaNs"

    # Constant pixel velocity check:
    # After EMA warm-up (track_age_frames >= 12), speed should be ~constant
    warm_tracks = df[df["track_age_frames"] >= 12]
    speeds = warm_tracks["speed_kmh"].values
    assert len(speeds) >= 5
    speed_std = float(np.std(speeds))
    assert speed_std < 0.25, f"Speed std dev {speed_std} is too high for constant pixel velocity"

    # 6. Assert telemetry_meta.json
    meta_path = artifacts_dir / "telemetry_meta.json"
    assert meta_path.exists(), "telemetry_meta.json must exist"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for required_key in ["video_fps", "frame_skip", "dt_s", "img_w", "img_h", "min_track_age_frames", "class_map"]:
        assert required_key in meta, f"Missing key '{required_key}' in telemetry_meta.json"

    # 7. Assert perception_stats.json
    stats_path = artifacts_dir / "perception_stats.json"
    assert stats_path.exists(), "perception_stats.json must exist"
    with open(stats_path, "r", encoding="utf-8") as f:
        pstats = json.load(f)
    for required_stat in [
        "n_unique_tracks",
        "track_length_p50",
        "track_length_p90",
        "churn_pct",
        "class_histogram",
        "state_distribution",
        "mean_speed_kmh_per_class",
    ]:
        assert required_stat in pstats, f"Missing '{required_stat}' in perception_stats.json"
    assert pstats["n_unique_tracks"] == 1
    assert pstats["class_histogram"]["car"] == len(df)
