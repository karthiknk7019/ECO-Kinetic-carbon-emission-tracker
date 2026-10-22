import json
from pathlib import Path
import sys
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_telemetry_series import build_telemetry_series


def test_build_telemetry_series_from_artifacts():
    repo_root = Path(__file__).resolve().parent.parent
    parquet_path = repo_root / "artifacts" / "telemetry.parquet"
    factors_path = repo_root / "config" / "emission_factors.yaml"

    assert parquet_path.exists(), "artifacts/telemetry.parquet must exist"
    assert factors_path.exists(), "config/emission_factors.yaml must exist"

    result = build_telemetry_series(parquet_path, factors_path, interval_sec=2.0)

    # 1. Structure checks
    assert "points" in result
    assert "metadata" in result
    assert "vehiclesByTime" in result

    # 2. Metadata checks
    meta = result["metadata"]
    assert meta["maxTimeSec"] == 96.5
    assert meta["totalTracks"] == 236
    assert meta["intervalSec"] == 2.0
    assert meta["totalPoints"] >= 48

    # 3. Points checks
    points = result["points"]
    assert len(points) >= 48

    # First point
    p0 = points[0]
    assert p0["timestampSec"] == 0
    assert "emissionRate_gs" in p0
    assert "cumulativeCO2_kg" in p0
    assert "idleSharePct" in p0
    assert "trackedCount" in p0
    assert "byClassEmission" in p0
    assert "byStateCount" in p0

    # Class keys
    expected_classes = {"car", "motorcycle", "auto_rickshaw", "bus", "truck"}
    assert set(p0["byClassEmission"].keys()) == expected_classes

    # State keys
    expected_states = {"IDLE", "ACCEL", "CRUISE", "DECEL"}
    assert set(p0["byStateCount"].keys()) == expected_states

    # Cumulative monotonically increasing
    cums = [p["cumulativeCO2_kg"] for p in points]
    assert cums[-1] >= cums[0]


def test_directives_jsonl_validity():
    repo_root = Path(__file__).resolve().parent.parent
    directives_path = repo_root / "artifacts" / "directives.jsonl"

    assert directives_path.exists(), "artifacts/directives.jsonl must exist"

    with open(directives_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 4, "Must have at least 4 active directives in ledger"

    for line in lines:
        entry = json.loads(line)
        assert "directive_id" in entry or "id" in entry
        assert "timestamp" in entry
        assert "actions" in entry
        assert "rationale" in entry
        assert entry["mode"] in ("llm", "rules_fallback")
