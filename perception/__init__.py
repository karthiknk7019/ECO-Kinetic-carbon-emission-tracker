"""Perception package for vehicle detection, tracking, and kinematics."""

from perception.kinematics import (
    DEFAULT_MEAN_LENGTH_M,
    TrackState,
    classify_state,
    load_mean_lengths,
    smooth_speed,
    update_track,
)

__all__ = [
    "DEFAULT_MEAN_LENGTH_M",
    "TrackState",
    "classify_state",
    "load_mean_lengths",
    "smooth_speed",
    "update_track",
    "Tracker",
    "Detector",
    "TrafficImageAnalyzer",
    "get_cached_yolo_model",
]


def __getattr__(name: str):
    if name == "Tracker":
        from perception.tracker import Tracker

        return Tracker
    if name == "Detector":
        from perception.detector import Detector

        return Detector
    if name == "get_cached_yolo_model":
        from perception.detector import get_cached_yolo_model

        return get_cached_yolo_model
    if name == "TrafficImageAnalyzer":
        from perception.image_inference import TrafficImageAnalyzer

        return TrafficImageAnalyzer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
