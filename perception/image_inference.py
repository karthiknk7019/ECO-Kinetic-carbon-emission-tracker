"""Reusable single-image traffic perception module using YOLO.

Analyzes traffic intersection photographs locally on CPU:
- Validates file formats, sizes, and dimensions.
- Caches model weights to avoid redundant reloads.
- Resolves class names dynamically from loaded weights.
- Reports class coverage and limitations accurately (e.g., auto-rickshaw in COCO weights).
- Never fabricates speeds, trajectories, motion states, or emissions.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO
import yaml

from perception.detector import get_cached_yolo_model, get_default_config_path

# Standard supported image formats
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Color palette in BGR for visual annotations (crisp, modern contrast)
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "car": (255, 140, 0),         # Deep sky blue in RGB -> BGR is (0, 140, 255)
    "motorcycle": (0, 215, 255),  # Gold / Amber
    "auto_rickshaw": (80, 208, 120),  # Mint / Emerald green
    "bus": (204, 50, 153),        # Purple
    "truck": (50, 80, 240),       # Coral / Crimson
}
DEFAULT_BOX_COLOR_BGR = (180, 180, 180)


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation (format, dimensions, or corruption)."""
    pass


def validate_and_decode_image(
    input_data: Union[str, Path, bytes, np.ndarray],
    max_size_bytes: int = DEFAULT_MAX_IMAGE_SIZE_BYTES,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Validate and decode image input into a standard 3-channel BGR numpy array.

    Args:
        input_data: File path, raw bytes, or existing numpy array.
        max_size_bytes: Maximum allowed byte size for files/buffers.

    Returns:
        Tuple of (bgr_array, (height, width, channels)).

    Raises:
        ImageValidationError: If format, size, decoding, or dimensions are invalid.
    """
    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ImageValidationError(
                f"Unsupported file format '{path.suffix}'. Supported formats are: {allowed}"
            )
        if not path.exists():
            raise ImageValidationError(f"Image file does not exist: {path.name}")
        file_size = path.stat().st_size
        if file_size <= 0:
            raise ImageValidationError("Image file is empty (0 bytes).")
        if file_size > max_size_bytes:
            mb = file_size / (1024 * 1024)
            limit_mb = max_size_bytes / (1024 * 1024)
            raise ImageValidationError(
                f"File size ({mb:.1f} MB) exceeds maximum allowed limit ({limit_mb:.1f} MB)."
            )
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ImageValidationError(
                f"Failed to decode image from '{path.name}'. File may be corrupted or malformed."
            )

    elif isinstance(input_data, bytes):
        if len(input_data) <= 0:
            raise ImageValidationError("Uploaded image byte buffer is empty (0 bytes).")
        if len(input_data) > max_size_bytes:
            mb = len(input_data) / (1024 * 1024)
            limit_mb = max_size_bytes / (1024 * 1024)
            raise ImageValidationError(
                f"Uploaded image size ({mb:.1f} MB) exceeds maximum allowed limit ({limit_mb:.1f} MB)."
            )
        nparr = np.frombuffer(input_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageValidationError(
                "Failed to decode image bytes. File may be corrupted, truncated, or in an unsupported format."
            )

    elif isinstance(input_data, np.ndarray):
        if input_data.size == 0:
            raise ImageValidationError("Provided image array is empty.")
        if input_data.ndim == 2:
            img = cv2.cvtColor(input_data, cv2.COLOR_GRAY2BGR)
        elif input_data.ndim == 3:
            channels = input_data.shape[2]
            if channels == 4:
                img = cv2.cvtColor(input_data, cv2.COLOR_BGRA2BGR)
            elif channels == 3:
                img = input_data.copy()
            else:
                raise ImageValidationError(
                    f"Unsupported image channel count ({channels}). Expected 1 (grayscale), 3 (BGR/RGB), or 4 (BGRA)."
                )
        else:
            raise ImageValidationError(f"Invalid image array dimension ({input_data.ndim}D). Expected 2D or 3D.")
    else:
        raise ImageValidationError(f"Unsupported input type: {type(input_data).__name__}")

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ImageValidationError(f"Invalid image dimensions: {w}x{h} px. Dimensions must be positive.")
    if h < 16 or w < 16:
        raise ImageValidationError(f"Image dimensions ({w}x{h} px) are too small for vehicle detection (min 16x16 px).")

    return img, (h, w, 3)


def draw_detection_annotations(
    bgr_image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """Render high-contrast, scaled bounding boxes and label badges onto a copy of the image.

    Args:
        bgr_image: Source BGR image numpy array.
        detections: List of detection records containing 'class_name', 'confidence', 'bbox'.

    Returns:
        Annotated image in RGB format (ready for display or saving).
    """
    annotated = bgr_image.copy()
    h, w = annotated.shape[:2]

    # Dynamically scale line thickness and font size based on image resolution
    base_dim = max(w, h)
    line_thickness = max(1, int(round(base_dim / 500.0)))
    font_scale = max(0.4, base_dim / 1500.0)
    font_thickness = max(1, int(round(font_scale * 1.5)))

    for det in detections:
        cls_name = det.get("class_name", "vehicle")
        conf = det.get("confidence", 0.0)
        idx = det.get("detection_index", "")
        bbox = det.get("bbox", [0, 0, 0, 0])

        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
        # Ensure within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = CLASS_COLORS_BGR.get(cls_name, DEFAULT_BOX_COLOR_BGR)

        # 1. Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # 2. Label badge
        label_text = f"#{idx} {cls_name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        badge_y1 = max(0, y1 - text_h - baseline - 4)
        badge_y2 = y1
        badge_x2 = min(w - 1, x1 + text_w + 6)

        # If badge would clip above image top, draw inside the bounding box
        if y1 - text_h - baseline - 4 < 0:
            badge_y1 = y1
            badge_y2 = y1 + text_h + baseline + 6

        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

        # White or black text based on badge background contrast
        text_color = (0, 0, 0) if cls_name in ("motorcycle", "auto_rickshaw") else (255, 255, 255)
        text_y = badge_y2 - baseline - 2
        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    # Convert to standard RGB for presentation
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


class TrafficImageAnalyzer:
    """Reusable single-image traffic vehicle analyzer with CPU-enforced YOLO inference."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        cfg: Optional[Dict[str, Any]] = None,
        model: Optional[YOLO] = None,
    ) -> None:
        if cfg is not None:
            self.config = dict(cfg)
        else:
            c_path = Path(config_path) if config_path else get_default_config_path()
            if not c_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {c_path}")
            with open(c_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        self.model_name: str = str(self.config.get("model", "yolo26n.pt"))
        self.device: str = str(self.config.get("device", "cpu"))
        self.imgsz: int = int(self.config.get("imgsz", 640))
        self.default_confidence: float = float(self.config.get("confidence", 0.30))
        self.active_classes: List[str] = list(
            self.config.get("active_classes", ["car", "motorcycle", "auto_rickshaw", "bus", "truck"])
        )

        # Load or reuse cached model instance (CPU-only)
        if model is not None:
            self.model = model
        else:
            self.model = get_cached_yolo_model(self.model_name)

        # Dynamic class name mapping: model.names ∩ active_classes
        # Never assume class IDs are identical across models.
        active_set = set(self.active_classes)
        self.class_map: Dict[int, str] = {
            cid: name for cid, name in self.model.names.items() if name in active_set
        }
        self.active_class_ids: List[int] = list(self.class_map.keys())
        self.supported_classes: List[str] = sorted(list(self.class_map.values()))
        self.unsupported_classes: List[str] = sorted([c for c in self.active_classes if c not in set(self.supported_classes)])

        # Accurately track model limitations
        if "auto_rickshaw" in self.unsupported_classes:
            self.limitation_notes = (
                f"Loaded model '{self.model_name}' uses stock COCO classes and does not have a dedicated "
                f"'auto_rickshaw' class in its weights. Three-wheelers may be classified as 'car' or omitted. "
                f"Fine-tuned weights (IDD-5) are required for dedicated 3-wheeler recognition."
            )
        else:
            self.limitation_notes = None

    def analyze(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run single-image vehicle perception on a traffic photograph.

        Args:
            image_input: File path, byte buffer, or numpy image.
            conf: Optional confidence threshold override (default: configured value).
            imgsz: Optional input inference dimension override (default: configured value).

        Returns:
            Structured dictionary containing detections, metrics, counts, and annotated RGB image.
        """
        # 1. Input validation & decoding
        bgr_image, (h, w, c) = validate_and_decode_image(image_input)

        target_conf = float(conf) if conf is not None else self.default_confidence
        target_imgsz = int(imgsz) if imgsz is not None else self.imgsz

        # 2. Run model inference explicitly on CPU
        t0 = time.perf_counter()
        results = self.model(
            bgr_image,
            conf=target_conf,
            imgsz=target_imgsz,
            classes=self.active_class_ids,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Parse detections
        detection_records: List[Dict[str, Any]] = []
        per_class_counts: Dict[str, int] = {cls_name: 0 for cls_name in self.active_classes}

        if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
                cls_name = self.class_map.get(cls_id, self.model.names.get(cls_id, f"class_{cls_id}"))

                # Only include detections belonging to active classes
                if cls_name not in per_class_counts:
                    continue

                conf_val = float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])

                # Extract coordinates and clamp to image dimensions
                if hasattr(box, "xyxy") and len(box.xyxy) > 0:
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]
                    x1 = float(max(0.0, min(float(w), float(xyxy[0]))))
                    y1 = float(max(0.0, min(float(h), float(xyxy[1]))))
                    x2 = float(max(0.0, min(float(w), float(xyxy[2]))))
                    y2 = float(max(0.0, min(float(h), float(xyxy[3]))))
                else:
                    continue

                bbox_w = float(round(abs(x2 - x1), 1))
                bbox_h = float(round(abs(y2 - y1), 1))
                bbox_area = float(round(bbox_w * bbox_h, 1))

                record = {
                    "detection_index": len(detection_records) + 1,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_w_px": bbox_w,
                    "bbox_h_px": bbox_h,
                    "bbox_area_px": bbox_area,
                }
                detection_records.append(record)
                per_class_counts[cls_name] += 1

        # 4. Compute aggregate metrics
        total_detections = len(detection_records)
        mean_conf = (
            float(np.mean([d["confidence"] for d in detection_records]))
            if detection_records
            else 0.0
        )
        classes_detected_count = sum(1 for cnt in per_class_counts.values() if cnt > 0)

        # 5. Generate annotated image in original coordinates
        annotated_rgb = draw_detection_annotations(bgr_image, detection_records)
        original_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        return {
            "image_dimensions": {"width": w, "height": h, "channels": c},
            "model_identifier": self.model_name,
            "device": self.device,
            "confidence_threshold": target_conf,
            "processing_latency_ms": round(latency_ms, 2),
            "total_detections": total_detections,
            "per_class_counts": per_class_counts,
            "mean_confidence": round(mean_conf, 4),
            "classes_detected_count": classes_detected_count,
            "supported_classes": self.supported_classes,
            "unsupported_classes": self.unsupported_classes,
            "limitation_notes": self.limitation_notes,
            "detection_records": detection_records,
            "annotated_image": annotated_rgb,
            "original_image": original_rgb,
        }


def export_results_to_json(result: Dict[str, Any]) -> str:
    """Serialize structured detection result to a JSON string, excluding image arrays."""
    clean_dict = {
        k: v for k, v in result.items()
        if k not in ("annotated_image", "original_image")
    }
    return json.dumps(clean_dict, indent=2)


def export_results_to_csv(result: Dict[str, Any]) -> str:
    """Serialize detection records into a CSV string."""
    records = result.get("detection_records", [])
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "detection_index",
        "class_name",
        "confidence",
        "x1_px",
        "y1_px",
        "x2_px",
        "y2_px",
        "bbox_w_px",
        "bbox_h_px",
        "bbox_area_px",
    ])

    for r in records:
        bbox = r.get("bbox", [0, 0, 0, 0])
        writer.writerow([
            r.get("detection_index", ""),
            r.get("class_name", ""),
            r.get("confidence", ""),
            bbox[0] if len(bbox) > 0 else "",
            bbox[1] if len(bbox) > 1 else "",
            bbox[2] if len(bbox) > 2 else "",
            bbox[3] if len(bbox) > 3 else "",
            r.get("bbox_w_px", ""),
            r.get("bbox_h_px", ""),
            r.get("bbox_area_px", ""),
        ])

    return output.getvalue()


def encode_image_to_bytes(image_rgb: np.ndarray, format: str = "png") -> bytes:
    """Encode an RGB numpy image into JPEG or PNG bytes for user download."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ext = f".{format.lower().lstrip('.')}"
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    success, buffer = cv2.imencode(ext, bgr)
    if not success:
        raise RuntimeError("Failed to encode image to buffer.")
    return buffer.tobytes()
