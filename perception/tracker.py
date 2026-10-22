"""Ultralytics ByteTrack integration module with runtime class resolution."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ultralytics import YOLO
import yaml

from perception.detector import get_cached_yolo_model


def get_default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "app.yaml"


class Tracker:
    """ByteTrack vehicle tracker wrapping fine-tuned YOLO model."""

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        model: Optional[YOLO] = None,
    ) -> None:
        if cfg is None:
            c_path = Path(config_path) if config_path else get_default_config_path()
            with open(c_path, "r", encoding="utf-8") as f:
                self.cfg: Dict[str, Any] = yaml.safe_load(f)
        else:
            self.cfg = dict(cfg)

        self.model_path = str(self.cfg.get("model", "models/finetuned/best.pt"))
        self.device = str(self.cfg.get("device", "cpu"))
        self.confidence = float(self.cfg.get("confidence", 0.30))
        self.imgsz = int(self.cfg.get("imgsz", 640))
        self.tracker_config = str(self.cfg.get("tracker", "bytetrack.yaml"))

        # Load or reuse YOLO model
        if model is not None:
            self.model = model
        else:
            self.model = get_cached_yolo_model(self.model_path)

        # Critical design rule:
        # Class IDs are NEVER hardcoded. Resolve at startup:
        # class_map = {cid: name for cid, name in model.names.items() if name in cfg["active_classes"]}
        active_classes = set(self.cfg.get("active_classes", []))
        self.class_map: Dict[int, str] = {
            cid: name for cid, name in self.model.names.items() if name in active_classes
        }
        # Print class_map at startup
        print(f"[Tracker] class_map: {self.class_map}")
        self.active_classes_ids: List[int] = list(self.class_map.keys())

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Track objects in frame with ByteTrack.

        Returns list of detection dictionaries:
          {
            "track_id": int,
            "cls": str,
            "cx_norm": float,
            "cy_norm": float,
            "bbox_w_px": float,
            "conf": float,
          }
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            classes=self.active_classes_ids,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return detections

        # Handle boxes.id None
        track_ids = getattr(boxes, "id", None)
        if track_ids is None:
            return detections

        h, w = frame.shape[:2]

        for i, box in enumerate(boxes):
            if box.id is None:
                continue

            track_id = int(box.id[0].item() if hasattr(box.id[0], "item") else box.id[0])
            cls_id = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
            cls_name = self.class_map.get(cls_id)
            if cls_name is None:
                continue

            conf = float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])

            # Safely extract bounding box geometry
            if hasattr(box, "xywh") and len(box.xywh) > 0:
                xywh = box.xywh[0]
                cx_px = float(xywh[0].item() if hasattr(xywh[0], "item") else xywh[0])
                cy_px = float(xywh[1].item() if hasattr(xywh[1], "item") else xywh[1])
                bbox_w_px = float(xywh[2].item() if hasattr(xywh[2], "item") else xywh[2])
                bbox_h_px = float(xywh[3].item() if hasattr(xywh[3], "item") else xywh[3])
            elif hasattr(box, "xyxy") and len(box.xyxy) > 0:
                xyxy = box.xyxy[0]
                x1 = float(xyxy[0].item() if hasattr(xyxy[0], "item") else xyxy[0])
                y1 = float(xyxy[1].item() if hasattr(xyxy[1], "item") else xyxy[1])
                x2 = float(xyxy[2].item() if hasattr(xyxy[2], "item") else xyxy[2])
                y2 = float(xyxy[3].item() if hasattr(xyxy[3], "item") else xyxy[3])
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                bbox_w_px = abs(x2 - x1)
                bbox_h_px = abs(y2 - y1)
            else:
                continue

            cx_norm = float(np.clip(cx_px / max(1.0, float(w)), 0.0, 1.0))
            cy_norm = float(np.clip(cy_px / max(1.0, float(h)), 0.0, 1.0))

            detections.append({
                "track_id": track_id,
                "cls": cls_name,
                "cx_norm": cx_norm,
                "cy_norm": cy_norm,
                "bbox_w_px": bbox_w_px,
                "bbox_h_px": bbox_h_px,
                "conf": conf,
            })

        return detections
