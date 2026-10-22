"""Analytic golden tests and contract verifications for twin/ modules."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import time
import pytest
import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twin.emission_engine import compute_track_emissions, frame_rate_g
from twin.corridor import aggregate, compute_zone_rank, extract_events, to_dispatcher_payload
from twin.scenarios import apply_scenario
from scripts.make_synthetic_telemetry import generate_synthetic_telemetry


@pytest.fixture
def emission_factors() -> dict:
    """Fixture providing standard emission factors."""
    config_path = Path(__file__).parent.parent / "config" / "emission_factors.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_car_cruise_60s_golden(emission_factors):
    """1 car CRUISE constant 36 km/h for 60s -> 90.0 g +-0.5."""
    # 60s with dt=0.08s -> 750 frames
    n_frames = 750
    dt_s = 0.08
    times = [round(i * dt_s, 4) for i in range(n_frames)]
    df = pd.DataFrame({
        "video_time_s": times,
        "track_id": [1] * n_frames,
        "cls": ["car"] * n_frames,
        "cx_norm": [0.5] * n_frames,
        "cy_norm": [0.5] * n_frames,
        "bbox_w_px": [90.0] * n_frames,
        "speed_kmh": [36.0] * n_frames,
        "state": ["CRUISE"] * n_frames,
        "track_age_frames": list(range(15, 15 + n_frames)),
    })

    emitted = compute_track_emissions(df, emission_factors, min_age=15)
    total_g = float(emitted["co2_cum_g"].iloc[-1])
    assert pytest.approx(90.0, abs=0.5) == total_g


def test_bus_idle_60s_golden(emission_factors):
    """1 bus IDLE 60s -> 45.0 g."""
    n_frames = 750
    dt_s = 0.08
    times = [round(i * dt_s, 4) for i in range(n_frames)]
    df = pd.DataFrame({
        "video_time_s": times,
        "track_id": [2] * n_frames,
        "cls": ["bus"] * n_frames,
        "cx_norm": [0.5] * n_frames,
        "cy_norm": [0.5] * n_frames,
        "bbox_w_px": [140.0] * n_frames,
        "speed_kmh": [0.0] * n_frames,
        "state": ["IDLE"] * n_frames,
        "track_age_frames": list(range(15, 15 + n_frames)),
    })

    emitted = compute_track_emissions(df, emission_factors, min_age=15)
    total_g = float(emitted["co2_cum_g"].iloc[-1])
    assert pytest.approx(45.0, abs=0.5) == total_g


def test_auto_rickshaw_idle_60s_golden(emission_factors):
    """1 auto_rickshaw IDLE 60s -> 12.0 g."""
    n_frames = 750
    dt_s = 0.08
    times = [round(i * dt_s, 4) for i in range(n_frames)]
    df = pd.DataFrame({
        "video_time_s": times,
        "track_id": [5] * n_frames,
        "cls": ["auto_rickshaw"] * n_frames,
        "cx_norm": [0.5] * n_frames,
        "cy_norm": [0.5] * n_frames,
        "bbox_w_px": [70.0] * n_frames,
        "speed_kmh": [0.0] * n_frames,
        "state": ["IDLE"] * n_frames,
        "track_age_frames": list(range(15, 15 + n_frames)),
    })

    emitted = compute_track_emissions(df, emission_factors, min_age=15)
    total_g = float(emitted["co2_cum_g"].iloc[-1])
    assert pytest.approx(12.0, abs=0.5) == total_g


def test_car_accel_60s_golden(emission_factors):
    """1 car ACCEL at 36 km/h 60s -> 144.0 g."""
    n_frames = 750
    dt_s = 0.08
    times = [round(i * dt_s, 4) for i in range(n_frames)]
    df = pd.DataFrame({
        "video_time_s": times,
        "track_id": [3] * n_frames,
        "cls": ["car"] * n_frames,
        "cx_norm": [0.5] * n_frames,
        "cy_norm": [0.5] * n_frames,
        "bbox_w_px": [90.0] * n_frames,
        "speed_kmh": [36.0] * n_frames,
        "state": ["ACCEL"] * n_frames,
        "track_age_frames": list(range(15, 15 + n_frames)),
    })

    emitted = compute_track_emissions(df, emission_factors, min_age=15)
    total_g = float(emitted["co2_cum_g"].iloc[-1])
    assert pytest.approx(144.0, abs=0.5) == total_g


def test_age_gate(emission_factors):
    """Age gate: identical track with age 10 -> 0 g."""
    n_frames = 10
    dt_s = 0.08
    times = [round(i * dt_s, 4) for i in range(n_frames)]
    df = pd.DataFrame({
        "video_time_s": times,
        "track_id": [4] * n_frames,
        "cls": ["car"] * n_frames,
        "cx_norm": [0.5] * n_frames,
        "cy_norm": [0.5] * n_frames,
        "bbox_w_px": [90.0] * n_frames,
        "speed_kmh": [36.0] * n_frames,
        "state": ["CRUISE"] * n_frames,
        "track_age_frames": list(range(1, 11)),
    })

    emitted = compute_track_emissions(df, emission_factors, min_age=15)
    assert np.all(emitted["co2_rate_gps"] == 0.0)
    assert float(emitted["co2_cum_g"].iloc[-1]) == 0.0


def test_ev_share_full_car(emission_factors):
    """EV share car=1.0 -> car contributes 0."""
    df = generate_synthetic_telemetry(duration_s=120.0, n_vehicles=30, seed=42)
    res = apply_scenario(
        df,
        emission_factors,
        ev_share={"car": 1.0},
        green_extension_s=0,
        heavy_restriction=False,
    )

    car_stats = res["per_class_breakdown"]["car"]
    assert car_stats["baseline_co2_g"] > 0
    assert car_stats["scenario_co2_g"] == 0.0
    assert car_stats["avoided_g"] == car_stats["baseline_co2_g"]


def test_green_extension_strictly_increasing(emission_factors):
    """green_extension savings strictly increasing X=5 < X=10 < X=15, all > 0,
    and savings <= idle_rate*idle_s + (mult-1)*cruise*accel_s summed (upper bound).
    """
    df = generate_synthetic_telemetry(duration_s=600.0, n_vehicles=120, seed=42)

    res_5 = apply_scenario(df, emission_factors, ev_share={}, green_extension_s=5, heavy_restriction=False)
    res_10 = apply_scenario(df, emission_factors, ev_share={}, green_extension_s=10, heavy_restriction=False)
    res_15 = apply_scenario(df, emission_factors, ev_share={}, green_extension_s=15, heavy_restriction=False)

    s5 = res_5["avoided_g"]
    s10 = res_10["avoided_g"]
    s15 = res_15["avoided_g"]

    assert s5 > 0
    assert s10 > s5
    assert s15 > s10

    # Theoretical upper bound with cut_fraction = 1.0: sum of idle_rate*idle_s + (mult-1)*cruise*accel_s
    from twin.scenarios import _extract_stop_and_accel_events
    from twin.emission_engine import get_class_factors
    events = _extract_stop_and_accel_events(df, min_age=15, dt_s=0.08, min_idle_s=3.0)
    theoretical_upper_bound = 0.0
    for ev in events:
        c = ev["cls"]
        cfg = get_class_factors(emission_factors, c)
        idle_rate = float(cfg.get("idle_g_per_min", 0.0)) / 60.0
        cruise_rate = (float(cfg.get("cruise_g_per_km", 0.0)) * 36.0) / 3600.0
        accel_mult = float(cfg.get("accel_multiplier", 1.0))
        theoretical_upper_bound += (idle_rate * ev["idle_s"] + (accel_mult - 1.0) * cruise_rate * ev["accel_s"])

    assert s15 <= theoretical_upper_bound
    assert s15 <= res_15["baseline_co2_g"]


def test_generator_determinism():
    """Generator determinism: two runs same seed -> identical file hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "run1.parquet"
        p2 = Path(tmpdir) / "run2.parquet"

        df1 = generate_synthetic_telemetry(duration_s=120.0, n_vehicles=20, seed=42)
        df1.to_parquet(p1, engine="pyarrow", index=False)

        df2 = generate_synthetic_telemetry(duration_s=120.0, n_vehicles=20, seed=42)
        df2.to_parquet(p2, engine="pyarrow", index=False)

        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()

        assert h1 == h2


def test_dispatcher_payload_exact_schema(emission_factors):
    """Verify exact schema of to_dispatcher_payload."""
    df = generate_synthetic_telemetry(duration_s=120.0, n_vehicles=20, seed=42)
    emitted = compute_track_emissions(df, emission_factors, min_age=15)
    aggs = aggregate(emitted, window_s=60.0)

    assert len(aggs) >= 1
    agg_row = aggs[0]
    payload = to_dispatcher_payload(agg_row, intersection_id="junction_01")

    expected_keys = {
        "intersection_id",
        "timestamp",
        "window_s",
        "co2_g_total",
        "co2_g_per_min",
        "idle_ratio",
        "heavy_vehicle_density",
        "n_counted_vehicles",
        "n_stop_events",
        "top_choke_zone",
    }

    assert set(payload.keys()) == expected_keys
    assert isinstance(payload["intersection_id"], str)
    assert isinstance(payload["timestamp"], str)
    assert isinstance(payload["window_s"], float)
    assert isinstance(payload["co2_g_total"], float)
    assert isinstance(payload["co2_g_per_min"], float)
    assert isinstance(payload["idle_ratio"], float)
    assert isinstance(payload["heavy_vehicle_density"], float)
    assert isinstance(payload["n_counted_vehicles"], int)
    assert isinstance(payload["n_stop_events"], int)

    top_zone = payload["top_choke_zone"]
    if top_zone is not None:
        assert set(top_zone.keys()) == {"id", "idle_vehicle_minutes"}
        assert top_zone["id"].startswith("z_r")


def test_scenario_execution_time(emission_factors):
    """apply_scenario on 10-min telemetry runs in < 1s."""
    df = generate_synthetic_telemetry(duration_s=600.0, n_vehicles=120, seed=42)

    t0 = time.perf_counter()
    res = apply_scenario(
        df,
        emission_factors,
        ev_share={"car": 0.25, "bus": 0.5},
        green_extension_s=10,
        heavy_restriction=True,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"apply_scenario took {elapsed:.3f}s (target < 1.0s)"
    assert res["avoided_g"] > 0


def test_emission_engine_rejects_static_image_detections(emission_factors):
    """Confirm emission engine refuses to calculate emissions on static image detections."""
    static_detections_df = pd.DataFrame([
        {"detection_index": 1, "class_name": "car", "confidence": 0.85, "bbox_w_px": 80.0},
        {"detection_index": 2, "class_name": "truck", "confidence": 0.75, "bbox_w_px": 120.0},
    ])

    with pytest.raises(ValueError, match="Cannot calculate dynamic emissions"):
        compute_track_emissions(static_detections_df, emission_factors)

