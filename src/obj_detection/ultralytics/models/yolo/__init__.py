# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import detect, segment

from .model import YOLO, YOLOWorld

__all__ = "segment", "detect", "YOLO", "YOLOWorld"
