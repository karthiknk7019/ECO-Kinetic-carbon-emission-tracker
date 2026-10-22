"""Unit and integration tests for Traffic Image Intelligence backend."""

import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.image_inference import (
    DEFAULT_MAX_IMAGE_SIZE_BYTES,
    ImageValidationError,
    TrafficImageAnalyzer,
    draw_detection_annotations,
    encode_image_to_bytes,
    export_results_to_csv,
    export_results_to_json,
    validate_and_decode_image,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TRAFFIC_IMAGE = REPO_ROOT / "assets_workspace" / "frame_cand1.jpg"


@pytest.fixture
def analyzer():
    """Fixture providing initialized TrafficImageAnalyzer with default config."""
    return TrafficImageAnalyzer()


def test_valid_image_numpy_input(analyzer):
    """Test inference on synthetic 3-channel numpy array."""
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(dummy)

    assert result["image_dimensions"] == {"width": 640, "height": 480, "channels": 3}
    assert result["total_detections"] == 0
    assert isinstance(result["per_class_counts"], dict)
    assert result["annotated_image"].shape == (480, 640, 3)
    assert result["original_image"].shape == (480, 640, 3)


def test_valid_image_bytes_input(analyzer):
    """Test inference on valid encoded image bytes."""
    dummy = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy)
    result = analyzer.analyze(buf.tobytes())

    assert result["image_dimensions"] == {"width": 100, "height": 100, "channels": 3}
    assert result["total_detections"] == 0


def test_unsupported_file_format():
    """Verify rejection of unsupported file extensions."""
    with pytest.raises(ImageValidationError, match="Unsupported file format"):
        validate_and_decode_image("test_sample.txt")


def test_corrupted_image_bytes():
    """Verify rejection of malformed/corrupted byte buffer."""
    with pytest.raises(ImageValidationError, match="Failed to decode image bytes"):
        validate_and_decode_image(b"not_an_image_random_bytes_123456789")


def test_empty_image_buffer():
    """Verify rejection of empty 0-byte buffer."""
    with pytest.raises(ImageValidationError, match="empty"):
        validate_and_decode_image(b"")


def test_excessively_large_image():
    """Verify rejection of images exceeding maximum byte size limit."""
    with pytest.raises(ImageValidationError, match="exceeds maximum allowed limit"):
        validate_and_decode_image(b"x" * 1000, max_size_bytes=500)


def test_tiny_image_dimensions():
    """Verify rejection of images smaller than detection minimum."""
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ImageValidationError, match="too small"):
        validate_and_decode_image(tiny)


def test_cpu_device_configuration(analyzer):
    """Verify inference runs strictly on CPU."""
    assert analyzer.device == "cpu"
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    res = analyzer.analyze(dummy)
    assert res["device"] == "cpu"


def test_class_name_mapping_and_limitations(analyzer):
    """Verify class mapping intersects model names with active classes accurately."""
    assert isinstance(analyzer.supported_classes, list)
    assert isinstance(analyzer.unsupported_classes, list)

    # For stock yolo26n.pt, auto_rickshaw is not present in COCO
    if analyzer.model_name == "yolo26n.pt":
        assert "auto_rickshaw" in analyzer.unsupported_classes
        assert analyzer.limitation_notes is not None
        assert "auto_rickshaw" in analyzer.limitation_notes


def test_absence_of_fabricated_motion_and_emissions(analyzer):
    """Strict guardrail test: ensure single-image inference NEVER produces motion/emissions."""
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res = analyzer.analyze(dummy)

    forbidden_keys = {
        "speed_kmh", "speed", "velocity", "acceleration",
        "trajectory", "state", "co2_g", "co2_cum_g",
        "emissions", "fuel_saved", "choke_score"
    }

    # Root result check
    assert not forbidden_keys.intersection(res.keys()), "Found forbidden kinematic/emission keys in root result"

    # Per-detection record check
    for d in res["detection_records"]:
        assert not forbidden_keys.intersection(d.keys()), "Found forbidden kinematic/emission keys in detection record"


def test_csv_export_format(analyzer):
    """Verify CSV export contains correct headers and records."""
    dummy_result = {
        "detection_records": [
            {
                "detection_index": 1,
                "class_name": "car",
                "confidence": 0.8542,
                "bbox": [10.0, 20.0, 110.0, 120.0],
                "bbox_w_px": 100.0,
                "bbox_h_px": 100.0,
                "bbox_area_px": 10000.0,
            }
        ]
    }
    csv_str = export_results_to_csv(dummy_result)
    lines = csv_str.strip().splitlines()

    assert len(lines) == 2
    assert "detection_index,class_name,confidence" in lines[0]
    assert "1,car,0.8542" in lines[1]


def test_json_export_format(analyzer):
    """Verify JSON export serializes cleanly without numpy arrays."""
    dummy_result = {
        "model_identifier": "yolo26n.pt",
        "total_detections": 1,
        "annotated_image": np.zeros((10, 10, 3), dtype=np.uint8),
        "original_image": np.zeros((10, 10, 3), dtype=np.uint8),
        "detection_records": [{"detection_index": 1, "class_name": "car"}],
    }
    json_str = export_results_to_json(dummy_result)
    parsed = json.loads(json_str)

    assert "annotated_image" not in parsed
    assert "original_image" not in parsed
    assert parsed["model_identifier"] == "yolo26n.pt"
    assert parsed["total_detections"] == 1


def test_image_encoding_to_bytes():
    """Verify encoding RGB numpy image to JPEG/PNG bytes."""
    rgb_img = np.full((120, 120, 3), 200, dtype=np.uint8)
    png_bytes = encode_image_to_bytes(rgb_img, format="png")
    jpg_bytes = encode_image_to_bytes(rgb_img, format="jpg")

    assert len(png_bytes) > 0
    assert len(jpg_bytes) > 0
    # PNG magic header: \x89PNG
    assert png_bytes[:4] == b"\x89PNG"
    # JPEG magic header: \xff\xd8
    assert jpg_bytes[:2] == b"\xff\xd8"


@pytest.mark.skipif(not SAMPLE_TRAFFIC_IMAGE.exists(), reason="Sample traffic image not found")
def test_real_traffic_image_inference(analyzer):
    """End-to-end test on real traffic photograph."""
    res_low_conf = analyzer.analyze(SAMPLE_TRAFFIC_IMAGE, conf=0.25)
    res_high_conf = analyzer.analyze(SAMPLE_TRAFFIC_IMAGE, conf=0.60)

    # Valid structure
    assert res_low_conf["total_detections"] >= 1
    assert res_low_conf["image_dimensions"] == {"width": 1920, "height": 1080, "channels": 3}
    assert res_low_conf["processing_latency_ms"] > 0

    # Higher confidence threshold filters detections monotonically
    assert res_high_conf["total_detections"] <= res_low_conf["total_detections"]

    # Bounding boxes within image dimensions
    for det in res_low_conf["detection_records"]:
        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 <= 1920
        assert 0 <= y1 <= 1080
        assert 0 <= x2 <= 1920
        assert 0 <= y2 <= 1080
        assert det["bbox_w_px"] == pytest.approx(abs(x2 - x1), abs=0.2)
        assert det["bbox_h_px"] == pytest.approx(abs(y2 - y1), abs=0.2)
        assert det["bbox_area_px"] == pytest.approx(det["bbox_w_px"] * det["bbox_h_px"], abs=0.2)
        assert 0.0 <= det["confidence"] <= 1.0

    # Counts consistency
    active_sum = sum(res_low_conf["per_class_counts"].values())
    assert active_sum == res_low_conf["total_detections"]
