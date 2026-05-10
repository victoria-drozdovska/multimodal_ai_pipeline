"""
app.py
------
Flask REST API for the Multimodal Media Analysis project.

Endpoints:
  GET  /                    - API info page (HTML)
  GET  /health              - liveness check
  POST /detect/faces        - MediaPipe face detection on a single image
  POST /detect/objects      - YOLOv8 object detection on a single image
  POST /transcribe          - Whisper transcription on audio/video
  POST /analyze/video       - batch pipeline (faces + objects + speech)
  GET  /ui                  - web UI
  POST /api/orchestrate     - consolidated multimodal endpoint

Run:  python app.py
"""

import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.exceptions import HTTPException

# Lazy-load models so startup is fast
_face_model    = None
_face_model_conf = None
_yolo_model    = None
_whisper_model = None


def get_face_model(min_confidence: float = 0.7):
    global _face_model, _face_model_conf
    if _face_model is None or _face_model_conf != float(min_confidence):
        from face_detector import FaceDetector
        _face_model = FaceDetector(min_confidence=float(min_confidence))
        _face_model_conf = float(min_confidence)
    return _face_model


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from object_detector import ObjectDetector
        _yolo_model = ObjectDetector(conf=0.40)
    return _yolo_model


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from speech_recognizer import SpeechRecognizer
        _whisper_model = SpeechRecognizer(model_size="base")
    return _whisper_model


# Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

ALLOWED_IMAGES = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_AUDIO  = {"wav", "mp3", "flac", "m4a"}
ALLOWED_VIDEO  = {"mp4", "mov", "avi", "mkv"}


def file_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@app.errorhandler(Exception)
def handle_unexpected_error(err):
    """Return JSON for uncaught errors instead of dropping the connection."""
    if isinstance(err, HTTPException):
        return jsonify({"error": err.description}), err.code
    app.logger.exception("Unhandled server error")
    return jsonify({"error": "Internal server error"}), 500


# Simple browser landing page
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Multimodal Media Analysis API</title>
  <style>
    body { font-family: monospace; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #0f0f0f; color: #e0e0e0; }
    h1   { color: #7ee8a2; border-bottom: 1px solid #333; padding-bottom: 12px; }
    h2   { color: #a0c4ff; margin-top: 32px; }
    .ep  { background: #1a1a1a; border-left: 3px solid #7ee8a2; padding: 12px 16px; margin: 12px 0; border-radius: 4px; }
    .new { border-left-color: #ffb347 !important; }
    .method { color: #ffb347; font-weight: bold; }
    .path   { color: #7ee8a2; }
    .desc   { color: #999; font-size: 0.85em; margin-top: 4px; }
    code    { background: #222; padding: 2px 6px; border-radius: 3px; color: #f9c74f; }
    a       { color: #7ee8a2; }
    .cta    { display: inline-block; margin: 8px 0; padding: 10px 18px; background: #7ee8a2; color: #0f0f0f; text-decoration: none; border-radius: 6px; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Multimodal Media Analysis API</h1>
  <p>End-to-end video analysis: face detection + object detection + speech recognition.</p>
  <a class="cta" href="/ui">Open Web UI →</a>

  <h2>Endpoints</h2>

  <div class="ep">
    <span class="method">GET</span> <span class="path">/health</span>
    <div class="desc">Check API status and loaded models</div>
  </div>

  <div class="ep">
    <span class="method">POST</span> <span class="path">/detect/faces</span>
    <div class="desc">Upload an image, returns face bounding boxes and confidence</div>
    <div class="desc">Form-data: <code>file</code> (jpg/png)</div>
  </div>

  <div class="ep">
    <span class="method">POST</span> <span class="path">/detect/objects</span>
    <div class="desc">Upload an image, returns YOLOv8 object detections</div>
    <div class="desc">Form-data: <code>file</code> (jpg/png), optional <code>conf</code> (default 0.4)</div>
  </div>

  <div class="ep">
    <span class="method">POST</span> <span class="path">/transcribe</span>
    <div class="desc">Upload audio/video, returns Whisper transcription with timestamps</div>
    <div class="desc">Form-data: <code>file</code> (wav/mp3/flac/m4a/mp4/mov/avi/mkv)</div>
    <div class="desc">Optional: <code>apply_corrections=true|false</code> (default true), <code>reference_text</code> for WER</div>
  </div>

  <div class="ep">
    <span class="method">POST</span> <span class="path">/analyze/video</span>
    <div class="desc">Batch pipeline (faces + objects + speech)</div>
    <div class="desc">Form-data: <code>file</code> (mp4/mov), optional <code>conf</code></div>
  </div>

  <div class="ep new">
    <span class="method">POST</span> <span class="path">/api/orchestrate</span>
    <div class="desc">Single consolidated multimodal endpoint.
        Returns a unified response with summary, timeline, transcript, and
        base64-encoded annotated preview frames.</div>
    <div class="desc">Form-data: <code>file</code>, optional <code>enable_faces</code>,
        <code>enable_objects</code>, <code>enable_speech</code>, <code>conf</code>,
        <code>sample_fps</code>, <code>max_preview_frames</code>.</div>
  </div>

  <div class="ep new">
    <span class="method">GET</span> <span class="path">/ui</span>
    <div class="desc">Web UI — drag-and-drop video,
        view annotated frames, transcript, and timeline in one page.</div>
  </div>

  <h2>Quick Test (curl)</h2>
  <pre style="background:#1a1a1a;padding:16px;border-radius:6px;overflow-x:auto;color:#ccc">
# Health check
curl http://localhost:5001/health

# Single consolidated multimodal call
curl -X POST -F "file=@Video_speech_object.mov" http://localhost:5001/api/orchestrate
  </pre>
</body>
</html>
"""


@app.route("/")
def home():
    return HOME_HTML


# GET /health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "face_detection":     "MediaPipe FaceDetection (full-range)",
            "object_detection":   "YOLOv8n (Ultralytics, COCO 80-class)",
            "speech_recognition": "OpenAI Whisper (base)",
        },
        "endpoints": [
            "GET /health",
            "POST /detect/faces",
            "POST /detect/objects",
            "POST /transcribe",
            "POST /analyze/video",
            "POST /api/orchestrate",
            "GET /ui",
        ],
    })


# POST /detect/faces
@app.route("/detect/faces", methods=["POST"])
def detect_faces():
    if "file" not in request.files:
        return jsonify({"error": "No file. Use form-data key: file"}), 400
    f = request.files["file"]
    if file_ext(f.filename) not in ALLOWED_IMAGES:
        return jsonify({"error": f"Allowed image types: {ALLOWED_IMAGES}"}), 400

    face_conf = float(request.form.get("face_conf", 0.70))

    with tempfile.NamedTemporaryFile(suffix="." + file_ext(f.filename), delete=False) as tmp:
        f.save(tmp.name)
        try:
            result = get_face_model(face_conf).detect(tmp.name)
        finally:
            os.unlink(tmp.name)

    return jsonify({
        "model":     "MediaPipe FaceDetection",
        "face_conf": face_conf,
        "num_faces": result["num_faces"],
        "faces":     result["faces"],
    })


# POST /detect/objects
@app.route("/detect/objects", methods=["POST"])
def detect_objects():
    if "file" not in request.files:
        return jsonify({"error": "No file. Use form-data key: file"}), 400
    f = request.files["file"]
    if file_ext(f.filename) not in ALLOWED_IMAGES:
        return jsonify({"error": f"Allowed image types: {ALLOWED_IMAGES}"}), 400

    conf = float(request.form.get("conf", 0.40))

    with tempfile.NamedTemporaryFile(suffix="." + file_ext(f.filename), delete=False) as tmp:
        f.save(tmp.name)
        try:
            result = get_yolo_model().detect(tmp.name, conf=conf)
        finally:
            os.unlink(tmp.name)

    return jsonify({
        "model":           "YOLOv8n",
        "conf_threshold":  conf,
        "num_objects":     result["num_objects"],
        "objects":         result["objects"],
    })


# POST /transcribe
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file. Use form-data key: file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file_ext(f.filename)
    allowed = ALLOWED_AUDIO | ALLOWED_VIDEO
    if ext not in allowed:
        return jsonify({"error": f"Allowed types: {sorted(allowed)}"}), 400

    with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as tmp:
        f.save(tmp.name)
        audio_file = tmp.name

    apply_corrections = parse_bool(request.form.get("apply_corrections"), default=True)
    reference_text = request.form.get("reference_text")
    if reference_text is not None:
        reference_text = reference_text.strip()
        if not reference_text:
            reference_text = None

    try:
        if ext in ALLOWED_VIDEO:
            wav_path = audio_file.replace("." + ext, ".wav")
            ffmpeg_result = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_file,
                 "-ac", "1", "-ar", "16000", "-vn", wav_path],
                capture_output=True, text=True,
            )
            if ffmpeg_result.returncode != 0:
                return jsonify({
                    "error": "Failed to extract audio from video",
                    "details": ffmpeg_result.stderr.strip()[-800:],
                }), 400
            os.unlink(audio_file)
            audio_file = wav_path

        result = get_whisper_model().transcribe(
            audio_file,
            apply_domain_corrections=apply_corrections,
            reference_text=reference_text,
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Transcription failed")
        return jsonify({"error": "Transcription failed", "details": str(exc)}), 500
    finally:
        if os.path.exists(audio_file):
            os.unlink(audio_file)

    return jsonify(result)


# POST /analyze/video
@app.route("/analyze/video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file. Use form-data key: file"}), 400
    f   = request.files["file"]
    ext = file_ext(f.filename)
    if ext not in ALLOWED_VIDEO:
        return jsonify({"error": f"Allowed video types: {ALLOWED_VIDEO}"}), 400

    conf    = float(request.form.get("conf", 0.40))
    workdir = Path(tempfile.mkdtemp())

    try:
        video_path = workdir / f"input.{ext}"
        f.save(str(video_path))

        frames_dir = workdir / "frames"
        frames_dir.mkdir()
        audio_wav  = workdir / "audio.wav"

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, int(round(fps)))
        frame_paths, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                ts  = idx / fps
                out = frames_dir / f"frame_{len(frame_paths):05d}_t{ts:.2f}s.jpg"
                cv2.imwrite(str(out), frame)
                frame_paths.append(str(out))
            idx += 1
        cap.release()

        audio_result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-ac", "1", "-ar", "16000", "-vn", str(audio_wav)],
            capture_output=True,
        )
        has_audio = audio_result.returncode == 0 and audio_wav.exists()

        face_det = get_face_model()
        obj_det  = get_yolo_model()
        per_frame = []
        for fp in frame_paths:
            fr = face_det.detect(fp)
            ob = obj_det.detect(fp, conf=conf)
            per_frame.append({
                "frame":       Path(fp).name,
                "num_faces":   fr["num_faces"],
                "faces":       fr["faces"],
                "num_objects": ob["num_objects"],
                "objects":     ob["objects"],
            })

        transcription = {"full_text": "", "language": "N/A", "segments": []}
        if has_audio:
            transcription = get_whisper_model().transcribe(str(audio_wav))

        all_objs = [o["label"] for fr in per_frame for o in fr["objects"]]

        return jsonify({
            "status":    "success",
            "timestamp": datetime.now().isoformat(),
            "video_metadata": {
                "filename": f.filename,
                "width":    w_v,
                "height":   h_v,
                "fps":      fps,
                "total_frames": total,
            },
            "summary": {
                "frames_analyzed":      len(per_frame),
                "total_faces":          sum(f["num_faces"]   for f in per_frame),
                "frames_with_faces":    sum(1 for f in per_frame if f["num_faces"] > 0),
                "total_objects":        sum(f["num_objects"] for f in per_frame),
                "frames_with_objects":  sum(1 for f in per_frame if f["num_objects"] > 0),
                "unique_object_classes": sorted(set(all_objs)),
                "object_counts":         dict(Counter(all_objs)),
                "transcript_words":      len(transcription.get("full_text", "").split()),
            },
            "per_frame_results": per_frame,
            "transcription":     transcription,
        })

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# POST /api/orchestrate
@app.route("/api/orchestrate", methods=["POST"])
def api_orchestrate():
    """Single-call multimodal analysis endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file. Use form-data key: file"}), 400
    f = request.files["file"]
    ext = file_ext(f.filename)
    if ext not in ALLOWED_VIDEO:
        return jsonify({"error": f"Allowed video types: {sorted(ALLOWED_VIDEO)}"}), 400

    enable_faces   = parse_bool(request.form.get("enable_faces"), default=True)
    enable_objects = parse_bool(request.form.get("enable_objects"), default=True)
    enable_speech  = parse_bool(request.form.get("enable_speech"), default=True)
    face_conf      = float(request.form.get("face_conf", 0.70))
    yolo_conf      = float(request.form.get("conf", 0.40))
    sample_fps     = float(request.form.get("sample_fps", 1.0))
    max_preview    = int(request.form.get("max_preview_frames", 6))

    workdir = Path(tempfile.mkdtemp(prefix="orch_upload_"))
    try:
        video_path = workdir / f"input.{ext}"
        f.save(str(video_path))

        from orchestrator import orchestrate_video
        result = orchestrate_video(
            str(video_path),
            enable_faces=enable_faces,
            enable_objects=enable_objects,
            enable_speech=enable_speech,
            face_conf=face_conf,
            yolo_conf=yolo_conf,
            sample_fps=sample_fps,
            max_preview_frames=max_preview,
        )
        result["video_metadata"]["filename"] = f.filename
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Orchestration failed")
        return jsonify({"error": "Orchestration failed", "details": str(exc)}), 500
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# GET /ui
@app.route("/assets/<path:filename>", methods=["GET"])
def assets(filename):
    """Serve bundled UI assets from the templates directory."""
    templates_dir = Path(__file__).parent / "templates"
    return send_from_directory(str(templates_dir), filename)


@app.route("/ui", methods=["GET"])
def ui():
    """Serve the static web UI page."""
    template_path = Path(__file__).parent / "templates" / "ui.html"
    if not template_path.exists():
        return "UI template not found at templates/ui.html", 500
    return template_path.read_text(encoding="utf-8")


# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n{'='*60}")
    print(f"  Starting Multimodal Media Analysis API on port {port}")
    print(f"  Landing page : http://localhost:{port}")
    print(f"  Web UI       : http://localhost:{port}/ui")
    print(f"  Health check : http://localhost:{port}/health")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
