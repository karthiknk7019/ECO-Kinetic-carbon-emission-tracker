"""Vehicle perception module using YOLO."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
from ultralytics import YOLO
import yaml

# COCO vehicle class mappings
VEHICLE_COCO_IDS: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_default_config_path() -> Path:
    return get_repo_root() / "config" / "app.yaml"


def get_sample_image_path() -> Path:
    """Find bundled Ultralytics sample asset (e.g. bus.jpg)."""
    try:
        from ultralytics.utils import ASSETS

        bus_path = Path(ASSETS) / "bus.jpg"
        if bus_path.exists():
            return bus_path
    except Exception:
        pass

    try:
        import ultralytics

        candidate = Path(ultralytics.__file__).resolve().parent / "assets" / "bus.jpg"
        if candidate.exists():
            return candidate
    except Exception:
        pass

    return Path("bus.jpg")


_MODEL_CACHE: Dict[str, YOLO] = {}


def get_cached_yolo_model(model_name_or_path: Union[str, Path]) -> YOLO:
    """Retrieve or cache a YOLO model instance to prevent redundant weight reloads."""
    key = str(model_name_or_path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """Clear cached YOLO model instances."""
    _MODEL_CACHE.clear()


class Detector:
    VEHICLE_COCO_IDS: Dict[int, str] = VEHICLE_COCO_IDS

    def __init__(self, config_path: Optional[Union[str, Path]] = None, model: Optional[YOLO] = None) -> None:
        self.config_path = Path(config_path) if config_path else get_default_config_path()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.model_name: str = self.config.get("model", "yolo26n.pt")
        self.device: str = str(self.config.get("device", "cpu"))
        self.imgsz: int = int(self.config.get("imgsz", 640))
        self.confidence: float = float(self.config.get("confidence", 0.30))

        # Initialize or reuse YOLO model
        if model is not None:
            self.model = model
        else:
            self.model = get_cached_yolo_model(self.model_name)

    def detect(
        self,
        source: Any,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ) -> Any:
        """Run object detection on the provided source."""
        target_conf = conf if conf is not None else self.confidence
        target_imgsz = imgsz if imgsz is not None else self.imgsz
        target_device = device if device is not None else self.device
        return self.model(
            source,
            conf=target_conf,
            imgsz=target_imgsz,
            classes=classes,
            device=target_device,
            verbose=False,
        )


def run_smoke_test(config_path: Optional[Union[str, Path]] = None) -> None:
    repo_root = get_repo_root()
    detector = Detector(config_path=config_path)

    # Stage 1: Bundled sample image inference
    sample_img = get_sample_image_path()
    print(f"--- Stage 1: Running inference on sample image: {sample_img} ---")
    results = detector.detect(str(sample_img))
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names
        print(f"Stage 1: Detected {len(boxes)} objects:")
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = names.get(cls_id, f"class_{cls_id}")
            print(f"  - Class: {cls_name} (ID: {cls_id}), Confidence: {conf:.3f}")
    else:
        print("Stage 1: No detections found on sample image.")

    # Stage 2: Video inference (conditional)
    video_source_rel = detector.config.get("video_source", "data/videos/junction.mp4")
    video_path = Path(video_source_rel)
    if not video_path.is_absolute():
        video_path = repo_root / video_path

    print("\n--- Stage 2: Video inference ---")
    if not video_path.exists():
        print("STAGE 2 SKIPPED: data/videos/junction.mp4 not present (human teammate will add).")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video file at {video_path}")
        return

    frame_skip = int(detector.config.get("frame_skip", 2))
    frame_idx = 0
    sampled_count = 0
    vehicle_counts: List[int] = []

    while cap.isOpened() and sampled_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            res = detector.detect(
                frame,
                classes=list(VEHICLE_COCO_IDS.keys()),
            )
            count = len(res[0].boxes) if res and res[0].boxes is not None else 0
            vehicle_counts.append(count)
            print(f"Frame {frame_idx} (sample {sampled_count + 1}): {count} vehicles detected")
            sampled_count += 1
        frame_idx += 1

    cap.release()

    mean_vehicles = (sum(vehicle_counts) / len(vehicle_counts)) if vehicle_counts else 0.0
    print(f"Stage 2 Summary: Mean vehicles/frame = {mean_vehicles:.2f} across {sampled_count} sampled frames")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perception detector module")
    parser.add_argument("--smoke", action="store_true", help="Run perception smoke test")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    if args.smoke:
        run_smoke_test(config_path=args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
