"""object_detector.py — YOLOv8 object detection wrapper."""
from typing import Dict, List
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class ObjectDetector:
    def __init__(self, conf: float = 0.40):
        self.conf = float(conf)
        if YOLO is None:
            raise ImportError("ultralytics (YOLO) is required for ObjectDetector")
        self._model = YOLO("yolov8n")

    def detect(self, image_path: str, conf: float | None = None) -> Dict[str, object]:
        conf = float(conf) if conf is not None else self.conf
        results = self._model.predict(source=str(image_path), conf=conf, verbose=False)
        res = results[0]
        objects = []
        names = getattr(self._model, "names", {}) or {}
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                confv = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
                label = names.get(cls_id, str(cls_id))
                objects.append({"label": label, "confidence": confv, "bbox": [x1, y1, x2, y2]})
        return {"num_objects": len(objects), "objects": objects}

    def _attach_track_ids(self, frames: List[Dict]) -> List[Dict]:
        return frames
