"""Face detection wrapper with MediaPipe and Haar cascade fallbacks."""
from typing import Dict, List
import cv2
import numpy as np
import os
from pathlib import Path

try:
    import mediapipe as mp
except Exception:
    mp = None


class FaceDetector:
    """Adaptive face detector with MediaPipe and OpenCV backends."""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = float(min_confidence)
        self._backend = None

        try:
            if mp is not None:
                sol = getattr(mp, "solutions", None)
                fd_module = getattr(sol, "face_detection", None) if sol is not None else None
                if fd_module is not None:
                    self._backend = "solutions"
                    self._mp_fd = fd_module
                    self._model = self._mp_fd.FaceDetection(min_detection_confidence=self.min_confidence)
                    return
        except Exception:
            pass

        model_candidates = [
            Path(__file__).parent / "blaze_face_short_range.tflite",
            Path(__file__).parent / "mini_milestone_3" / "blaze_face_short_range.tflite",
        ]
        try:
            from mediapipe.tasks.python.vision import (
                FaceDetector as MpFaceDetector,
                FaceDetectorOptions,
                RunningMode,
            )
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except Exception:
            MpFaceDetector = None
            FaceDetectorOptions = None
            RunningMode = None
            BaseOptions = None

        model_path = None
        for c in model_candidates:
            if c.exists():
                model_path = str(c)
                break

        if MpFaceDetector is not None and model_path is not None:
            self._backend = "tasks"
            opts = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                min_detection_confidence=self.min_confidence,
            )
            try:
                self._model = MpFaceDetector.create_from_options(opts)
            except Exception:
                self._model = None

        if self._backend is None or (self._backend == "tasks" and self._model is None):
            self._backend = "haar"
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(haar_path):
                raise RuntimeError("OpenCV Haar cascade not found")
            self._haar = cv2.CascadeClassifier(haar_path)

    def detect(self, image_path: str) -> Dict[str, object]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        h, w = img.shape[:2]

        faces = []
        if self._backend == "solutions":
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self._model.process(rgb)
            if res.detections:
                for d in res.detections:
                    score = float(d.score[0]) if d.score else 0.0
                    if score < self.min_confidence:
                        continue
                    bboxR = d.location_data.relative_bounding_box
                    x = int(bboxR.xmin * w)
                    y = int(bboxR.ymin * h)
                    bw = int(bboxR.width * w)
                    bh = int(bboxR.height * h)
                    x2 = x + bw
                    y2 = y + bh
                    faces.append({"bbox": [x, y, x2, y2], "confidence": score})

        elif self._backend == "tasks":
            try:
                from mediapipe.tasks.python.vision import Image as MpImage
                mp_img = MpImage.create_from_file(str(image_path))
                result = self._model.detect(mp_img)
                for d in result.detections:
                    box = d.bounding_box
                    x1 = int(box.origin_x)
                    y1 = int(box.origin_y)
                    x2 = int(box.origin_x + box.width)
                    y2 = int(box.origin_y + box.height)
                    score = float(d.categories[0].score) if d.categories else 0.0
                    if score < self.min_confidence:
                        continue
                    faces.append({"bbox": [x1, y1, x2, y2], "confidence": score})
            except Exception:
                rects = self._haar.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5)
                for (x, y, w2, h2) in rects:
                    faces.append({"bbox": [int(x), int(y), int(x + w2), int(y + h2)], "confidence": 1.0})

        else:  # haar
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = self._haar.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(40, 40),
            )
            for (x, y, w2, h2) in rects:
                faces.append({"bbox": [int(x), int(y), int(x + w2), int(y + h2)], "confidence": 1.0})

        return {"num_faces": len(faces), "faces": faces}
