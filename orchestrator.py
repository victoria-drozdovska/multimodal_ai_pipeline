"""Orchestration layer for the multimodal pipeline."""

from __future__ import annotations

import base64
import io
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image, ImageDraw, ImageFont



_face_model = None
_face_model_conf = None
_yolo_model = None
_whisper_model = None


def _get_face_model(min_confidence: float = 0.7):
    global _face_model, _face_model_conf
    if _face_model is None or _face_model_conf != float(min_confidence):
        from face_detector import FaceDetector
        _face_model = FaceDetector(min_confidence=float(min_confidence))
        _face_model_conf = float(min_confidence)
    return _face_model


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from object_detector import ObjectDetector
        _yolo_model = ObjectDetector(conf=0.40)
    return _yolo_model


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from speech_recognizer import SpeechRecognizer
        _whisper_model = SpeechRecognizer(model_size="base")
    return _whisper_model


def _segment_text_at_time(segments: list[dict], t_sec: float) -> str:
    """Return transcript segment text active at timestamp t_sec."""
    for seg in segments:
        if seg.get("start_sec", 0.0) <= t_sec <= seg.get("end_sec", 0.0):
            return seg.get("text", "").strip()
    return ""


def _parse_frame_time_sec(frame_path: str) -> float | None:
    m = re.search(r"_t([0-9]+(?:\.[0-9]+)?)s", Path(frame_path).name)
    return float(m.group(1)) if m else None


def _annotate_frame_to_b64(
    frame_path: str,
    faces: list[dict],
    objects: list[dict],
    subtitle: str = "",
    max_width: int = 960,
) -> str:
    """
    Draw face/object boxes + a bottom-aligned subtitle on the frame and
    return a base64 JPEG suitable for direct <img src="data:..."> embedding.
    """
    img = Image.open(frame_path).convert("RGB")
    if img.width > max_width:
        new_h = int(img.height * (max_width / img.width))
        img = img.resize((max_width, new_h), Image.LANCZOS)

    sx = img.width / Image.open(frame_path).width
    sy = img.height / Image.open(frame_path).height

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for obj in objects:
        x1, y1, x2, y2 = [int(v * (sx if i % 2 == 0 else sy))
                          for i, v in enumerate(obj["bbox"])]
        label = obj["label"]
        conf = obj["confidence"]
        tid = obj.get("track_id")
        draw.rectangle([x1, y1, x2, y2], outline=(255, 110, 40), width=3)
        cap = f"{label} {conf:.2f}" + (f" #{tid}" if tid is not None else "")
        text_bg_y = max(0, y1 - 18)
        try:
            tw = draw.textlength(cap, font=font)
        except Exception:
            tw = len(cap) * 8
        draw.rectangle([x1, text_bg_y, x1 + tw + 6, text_bg_y + 18], fill=(255, 110, 40))
        draw.text((x1 + 3, text_bg_y + 1), cap, fill=(255, 255, 255), font=font)

    for face in faces:
        x1, y1, x2, y2 = [int(v * (sx if i % 2 == 0 else sy))
                          for i, v in enumerate(face["bbox"])]
        conf = face.get("confidence", 0.0)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 130), width=3)
        cap = f"face {conf:.2f}"
        text_bg_y = min(img.height - 18, y2 + 2)
        try:
            tw = draw.textlength(cap, font=font)
        except Exception:
            tw = len(cap) * 8
        draw.rectangle([x1, text_bg_y, x1 + tw + 6, text_bg_y + 18], fill=(0, 180, 100))
        draw.text((x1 + 3, text_bg_y + 1), cap, fill=(255, 255, 255), font=font)

    if subtitle:
        import textwrap
        wrapped = textwrap.wrap(subtitle, width=70)
        line_h = 18
        bar_h = 12 + line_h * len(wrapped)
        draw.rectangle([0, img.height - bar_h, img.width, img.height], fill=(0, 0, 0))
        for i, ln in enumerate(wrapped):
            draw.text((10, img.height - bar_h + 6 + i * line_h),
                      ln, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _pick_preview_frames(per_frame: list[dict], n: int = 6) -> list[int]:
    """
    Pick up to N representative frame indices, preferring frames that have
    BOTH faces and objects, then any with detections, then evenly-spaced.
    """
    if not per_frame:
        return []

    rich = [i for i, fr in enumerate(per_frame)
            if fr["num_faces"] > 0 and fr["num_objects"] > 0]
    some = [i for i, fr in enumerate(per_frame)
            if fr["num_faces"] > 0 or fr["num_objects"] > 0]

    chosen: list[int] = []
    for source in (rich, some):
        for i in source:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= n:
                return chosen[:n]

    if len(chosen) < n:
        step = max(1, len(per_frame) // n)
        for i in range(0, len(per_frame), step):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= n:
                break
    return chosen[:n]


def orchestrate_video(
    video_path: str,
    enable_faces: bool = True,
    enable_objects: bool = True,
    enable_speech: bool = True,
    face_conf: float = 0.7,
    yolo_conf: float = 0.4,
    sample_fps: float = 1.0,
    max_preview_frames: int = 6,
    cleanup: bool = True,
) -> dict:
    """Run the full multimodal pipeline on a single video file."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    started = datetime.now()
    workdir = Path(tempfile.mkdtemp(prefix="orchestrator_"))
    frames_dir = workdir / "frames"
    frames_dir.mkdir()
    audio_wav = workdir / "audio.wav"

    try:
        # Stage 1: extract frames and audio
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total / fps if fps else 0

        step = max(1, int(round(fps / max(0.1, sample_fps))))
        frame_paths: list[str] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                ts = idx / fps
                out = frames_dir / f"frame_{len(frame_paths):05d}_t{ts:.2f}s.jpg"
                cv2.imwrite(str(out), frame)
                frame_paths.append(str(out))
            idx += 1
        cap.release()

        # Audio
        ffmpeg_res = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-ac", "1", "-ar", "16000", "-vn", str(audio_wav)],
            capture_output=True,
        )
        has_audio = ffmpeg_res.returncode == 0 and audio_wav.exists() and audio_wav.stat().st_size > 0

        # Stage 2: run vision models
        per_frame: list[dict] = []
        face_det = _get_face_model(face_conf) if enable_faces else None
        obj_det = _get_yolo_model() if enable_objects else None

        for fp in frame_paths:
            faces_out = face_det.detect(fp) if face_det else {"num_faces": 0, "faces": []}
            objs_out = obj_det.detect(fp, conf=yolo_conf) if obj_det else {"num_objects": 0, "objects": []}
            per_frame.append({
                "frame": Path(fp).name,
                "frame_path": fp,
                "timestamp_sec": _parse_frame_time_sec(fp),
                "num_faces": faces_out.get("num_faces", 0),
                "faces": faces_out.get("faces", []),
                "num_objects": objs_out.get("num_objects", 0),
                "objects": objs_out.get("objects", []),
            })

        # Apply IoU tracking after the per-frame pass.
        if obj_det and per_frame:
            tracked = obj_det._attach_track_ids(
                [{"objects": fr["objects"]} for fr in per_frame]
            )
            for fr, tr in zip(per_frame, tracked):
                fr["objects"] = tr["objects"]

        # Stage 3: speech recognition
        transcription: dict
        if enable_speech and has_audio:
            transcription = _get_whisper_model().transcribe(str(audio_wav))
        else:
            transcription = {
                "model": "skipped" if not enable_speech else "no_audio_track",
                "full_text": "",
                "language": "N/A",
                "word_count": 0,
                "segment_count": 0,
                "segments": [],
            }

        # Stage 4: build a unified timeline
        segments = transcription.get("segments", [])
        timeline = []
        for fr in per_frame:
            t = fr["timestamp_sec"] or 0.0
            timeline.append({
                "t_sec": round(t, 2),
                "frame": fr["frame"],
                "num_faces": fr["num_faces"],
                "num_objects": fr["num_objects"],
                "object_labels": [o["label"] for o in fr["objects"]],
                "subtitle": _segment_text_at_time(segments, t),
            })

        # Stage 5: pick preview frames and annotate them as base64
        preview_indices = _pick_preview_frames(per_frame, n=max_preview_frames)
        preview_frames = []
        for i in preview_indices:
            fr = per_frame[i]
            subtitle = _segment_text_at_time(segments, fr["timestamp_sec"] or 0.0)
            data_url = _annotate_frame_to_b64(
                fr["frame_path"], fr["faces"], fr["objects"], subtitle
            )
            preview_frames.append({
                "frame": fr["frame"],
                "t_sec": round(fr["timestamp_sec"] or 0.0, 2),
                "num_faces": fr["num_faces"],
                "num_objects": fr["num_objects"],
                "object_labels": [o["label"] for o in fr["objects"]],
                "subtitle": subtitle,
                "image_data_url": data_url,
            })

        # Stage 6: summary
        all_objs = [o["label"] for fr in per_frame for o in fr["objects"]]
        track_ids = {o.get("track_id") for fr in per_frame for o in fr["objects"]
                     if o.get("track_id") is not None}

        elapsed = (datetime.now() - started).total_seconds()

        # Strip the absolute frame_path before returning.
        for fr in per_frame:
            fr.pop("frame_path", None)

        return {
            "status": "success",
            "generated_at": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed, 2),
            "video_metadata": {
                "filename": video_path.name,
                "width": w_v,
                "height": h_v,
                "fps": round(fps, 3),
                "total_frames": total,
                "duration_sec": round(duration_sec, 2),
                "has_audio": has_audio,
            },
            "models_used": {
                "face_detection": "MediaPipe FaceDetection" if enable_faces else "disabled",
                "face_conf": round(face_conf, 2) if enable_faces else None,
                "object_detection": f"YOLOv8n (conf>={yolo_conf})" if enable_objects else "disabled",
                "speech_recognition": "OpenAI Whisper (base)" if enable_speech and has_audio else (
                    "disabled" if not enable_speech else "skipped — no audio"),
            },
            "summary": {
                "frames_analyzed": len(per_frame),
                "total_faces_detected": sum(f["num_faces"] for f in per_frame),
                "frames_with_faces": sum(1 for f in per_frame if f["num_faces"] > 0),
                "total_objects_detected": sum(f["num_objects"] for f in per_frame),
                "frames_with_objects": sum(1 for f in per_frame if f["num_objects"] > 0),
                "unique_object_classes": sorted(set(all_objs)),
                "object_class_counts": dict(Counter(all_objs).most_common()),
                "unique_object_tracks": len(track_ids),
                "transcript_word_count": transcription.get("word_count", 0),
                "transcript_language": transcription.get("language", "N/A"),
            },
            "transcription": transcription,
            "timeline": timeline,
            "preview_frames": preview_frames,
            "per_frame_results": per_frame,
        }

    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)


# CLI entry point
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <video_file> [output.json]")
        sys.exit(1)

    out_path = sys.argv[2] if len(sys.argv) > 2 else "orchestrator_result.json"
    res = orchestrate_video(sys.argv[1])

    # Don't dump base64 image blobs into the saved file — they bloat it.
    light = {**res, "preview_frames": [
        {**pf, "image_data_url": "<base64 omitted in CLI dump>"} for pf in res["preview_frames"]
    ]}
    with open(out_path, "w") as f:
        json.dump(light, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(json.dumps(res["summary"], indent=2))
