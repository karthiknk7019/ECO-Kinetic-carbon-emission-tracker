"""Emission Engine for Eco-Kinetic Traffic Carbon Digital Twin.

Pure Python + NumPy + pandas engine for frame-level emission rate calculation,
track-level emission accumulation, and ledger gating based on kinematic state and track age.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def get_class_factors(factors: Dict[str, Any], cls: str) -> Dict[str, Any]:
    """Extract emission factors dictionary for a vehicle class."""
    if "classes" in factors and cls in factors["classes"]:
        return factors["classes"][cls]
    if cls in factors:
        return factors[cls]
    return factors


def frame_rate_g(
    cls: str,
    state: str,
    speed_kmh: float,
    factors: Dict[str, Any],
) -> float:
    """Calculate instantaneous CO2 emission rate in grams per second (g/s).

    Formulas per analyzed frame:
        IDLE:          idle_g_per_min / 60
        ACCEL:         cruise_g_per_km * speed_kmh / 3600 * accel_multiplier
        CRUISE, DECEL: cruise_g_per_km * speed_kmh / 3600
        else:          0
    """
    if state not in ("IDLE", "ACCEL", "CRUISE", "DECEL"):
        return 0.0

    cls_cfg = get_class_factors(factors, cls)
    idle_g_per_min = float(cls_cfg.get("idle_g_per_min", 0.0))
    cruise_g_per_km = float(cls_cfg.get("cruise_g_per_km", 0.0))
    accel_multiplier = float(cls_cfg.get("accel_multiplier", 1.0))

    if state == "IDLE":
        return idle_g_per_min / 60.0

    cruise_rate = (cruise_g_per_km * speed_kmh) / 3600.0

    if state == "ACCEL":
        return cruise_rate * accel_multiplier

    # CRUISE or DECEL
    return cruise_rate


def compute_track_emissions(
    telemetry_df: pd.DataFrame,
    factors: Dict[str, Any],
    min_age: int = 15,
) -> pd.DataFrame:
    """Calculate per-frame and cumulative CO2 emissions per track.

    Applies ledger gate: rate = 0 unless state in (IDLE, ACCEL, CRUISE, DECEL)
    and track_age_frames >= min_age.

    Adds columns:
        co2_rate_gps: float (g/s)
        co2_cum_g: float (cumulative grams per track_id)

    Returns a new DataFrame.
    """
    df = telemetry_df.copy()

    if df.empty:
        df["co2_rate_gps"] = np.array([], dtype=float)
        df["co2_cum_g"] = np.array([], dtype=float)
        return df

    required_cols = {"track_id", "video_time_s", "cls", "state", "speed_kmh", "track_age_frames"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Cannot calculate dynamic emissions: missing required tracking/kinematic columns: {sorted(list(missing_cols))}. "
            "Single-frame image detections lack temporal history and cannot be fed into the emission engine."
        )

    # Calculate dt_s per track from video_time_s deltas
    # Sort temporarily by track_id and video_time_s to ensure proper diffs
    orig_index = df.index
    df.sort_values(by=["track_id", "video_time_s"], inplace=True)

    dt_series = df.groupby("track_id", sort=False)["video_time_s"].diff()
    # For first frame of each track, backfill or default to 0.08
    dt_series = dt_series.bfill().fillna(0.08)
    # Clip any non-positive or extreme dt anomalies
    dt_series = dt_series.clip(lower=0.001, upper=5.0)

    # Pre-extract factor lookup tables for fast vectorized computation
    classes = df["cls"].unique()
    idle_rates = {}
    cruise_coeffs = {}
    accel_mults = {}

    for c in classes:
        cfg = get_class_factors(factors, str(c))
        idle_rates[c] = float(cfg.get("idle_g_per_min", 0.0)) / 60.0
        cruise_coeffs[c] = float(cfg.get("cruise_g_per_km", 0.0)) / 3600.0
        accel_mults[c] = float(cfg.get("accel_multiplier", 1.0))

    state_arr = df["state"].values
    cls_arr = df["cls"].values
    speed_arr = df["speed_kmh"].values
    age_arr = df["track_age_frames"].values

    # Gate condition: state in (IDLE, ACCEL, CRUISE, DECEL) and age >= min_age
    is_valid_state = np.isin(state_arr, ["IDLE", "ACCEL", "CRUISE", "DECEL"])
    is_gated = (age_arr >= min_age) & is_valid_state

    rates = np.zeros(len(df), dtype=float)

    for c in classes:
        c_mask = (cls_arr == c) & is_gated
        if not np.any(c_mask):
            continue

        c_idle_mask = c_mask & (state_arr == "IDLE")
        rates[c_idle_mask] = idle_rates[c]

        c_cruise_mask = c_mask & np.isin(state_arr, ["CRUISE", "DECEL"])
        rates[c_cruise_mask] = cruise_coeffs[c] * speed_arr[c_cruise_mask]

        c_accel_mask = c_mask & (state_arr == "ACCEL")
        rates[c_accel_mask] = cruise_coeffs[c] * speed_arr[c_accel_mask] * accel_mults[c]

    df["co2_rate_gps"] = rates
    df["_frame_dt_s"] = dt_series

    # Frame emissions = rate * dt
    frame_emissions = df["co2_rate_gps"].values * df["_frame_dt_s"].values
    df["_frame_emissions_g"] = frame_emissions

    # Cumulative emissions per track
    df["co2_cum_g"] = df.groupby("track_id", sort=False)["_frame_emissions_g"].cumsum()

    # Drop temporary calculation columns and restore original order
    df.drop(columns=["_frame_dt_s", "_frame_emissions_g"], inplace=True)
    df = df.reindex(orig_index)

    return df
