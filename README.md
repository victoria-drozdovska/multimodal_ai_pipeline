# Final Project — Multimodal Media Analysis API

Multimodal video analysis with preprocessing, face detection, object
detection, speech recognition, full pipeline execution, Flask REST
API delivery, a **single-call orchestration endpoint**, and a web UI
for end users.

---

## Project Structure

```
project/
├── app.py                  # Flask REST API + web UI entry point
├── pipeline.py             # Run the full pipeline on a video file (CLI)
├── orchestrator.py         # Single-call orchestration logic
├── preprocessor.py         # Extracts frames + audio from video
├── face_detector.py        # MediaPipe face detection (from Milestone 1)
├── object_detector.py      # YOLOv8 object detection
├── speech_recognizer.py    # OpenAI Whisper transcription
├── visualize.py            # Generate plots + per-frame annotated outputs
├── requirements.txt        # All Python dependencies
├── templates/
│   └── ui.html             # Web UI (drag-and-drop)
├── tests/
│   └── test_api.py         # API endpoint tests
├── mini_milestone_3/       # Mini-Milestone 3 deliverables
│   ├── README.md
│   ├── SETUP_WALKTHROUGH.md
│   ├── ISSUES_AND_RESOLUTIONS.md
│   ├── VALIDATION_REPORT.md
│   ├── validation.py
│   ├── generate_screenshots.py
│   ├── results/            # one JSON per sample
│   └── screenshots/        # per-sample PNG cards
├── mini_milestone_4/       # Mini-Milestone 4 documentation
│   ├── README.md
│   ├── CHATGPT_PROMPTS.md
│   ├── CODE_CHANGES.md
│   └── DEBUGGING_LOG.md
└── processed/              # Created automatically at runtime
    ├── frames/             # Extracted video frames
    ├── audio/              # Extracted audio WAV
    └── annotated_frames/   # Frame-by-frame overlays + transcript text
```

---

## Setup (One Time)

### 1. Install Python 3.10+
Download from https://python.org if you don't have it.
Check: `python --version`

### 2. Install ffmpeg
- **Windows:** Download from https://ffmpeg.org/download.html → add to PATH
- **Mac:**     `brew install ffmpeg`
- **Linux:**   `sudo apt install ffmpeg`

Check: `ffmpeg -version`

### 3. Create virtual environment (recommended)
```bash
cd /path/to/this/repository
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 4. Install Python packages
```bash
pip install -r requirements.txt
```

YOLOv8 weights (~6 MB) and Whisper weights (~74 MB for "base") download
automatically on first run.

---

## Running the Pipeline (batch mode)

Run the full pipeline on a video file — outputs `milestone2_results.json`:

```bash
python pipeline.py path/to/your/video.mp4
```

Example used in this project:

```bash
python pipeline.py Video_speech_object.mov
```

Then generate annotated images:
```bash
python visualize.py
```

### Full Recapture (recommended)

Use this when you want all outputs regenerated from scratch:

```bash
source .venv/bin/activate
python pipeline.py Video_speech_object.mov
python visualize.py
```

### Generated Outputs

- `milestone2_results.json`
- `output_object_detection.png`
- `output_face_detection.png`
- `output_waveform.png`
- `processed/audio/audio.wav`
- `processed/annotated_frames/*.jpg`
- `processed/annotated_frames/speech_transcript.txt`

Each file in `processed/annotated_frames/` includes:
- Object bounding boxes + class confidence
- Object track ID across frames
- Face bounding boxes + confidence
- Bounding box coordinates
- Timestamp-matched speech subtitle text

---

## Milestone 2 Requirement Coverage

This repository satisfies Milestone 2 by implementing all required components:

- Preprocessing: `preprocessor.py` extracts video frames and audio from input videos.
- Face detection: `face_detector.py` runs MediaPipe face detection and returns bounding boxes/confidence.
- Object detection: `object_detector.py` runs YOLOv8n and returns class labels, boxes, confidence, and tracking IDs.
- Speech recognition: `speech_recognizer.py` uses OpenAI Whisper (`base`) with custom domain term corrections and optional WER comparison when a reference transcript is supplied.
- Full pipeline: `pipeline.py` executes preprocessing + all three AI models and aggregates synchronized JSON output.
- REST API: `app.py` exposes model endpoints for face, object, speech, and full video analysis.
- Outputs and visualization: `milestone2_results.json`, `processed/`, and `visualize.py` provide submission-ready artifacts (JSON, transcript text, and annotated frames/plots).

## Running the Flask API and UI

```bash
python app.py
```

Open your browser at: http://localhost:5001

Web UI: http://localhost:5001/ui

Health check: http://localhost:5001/health

Current local status:
- `GET /health` returns `status: ok`
- `GET /ui` loads the drag-and-drop video analysis page
- `POST /api/orchestrate` returns the consolidated multimodal response

---

## API Endpoints

| Method | Endpoint              | Description                                         |
|--------|-----------------------|-----------------------------------------------------|
| GET    | /                     | Landing page                                        |
| GET    | /health               | Check API status                                    |
| POST   | /detect/faces         | Face detection on an image                          |
| POST   | /detect/objects       | YOLOv8 object detection on an image                 |
| POST   | /transcribe           | Whisper transcription of audio/video                |
| POST   | /analyze/video        | Full pipeline on a video (all 3 models)             |
| **POST** | **/api/orchestrate** | **Single consolidated multimodal call** |
| **GET**  | **/ui**              | **Drag-and-drop web UI**       |

Current object detection defaults in pipeline:
- YOLO confidence threshold: `0.50`
- NMS IoU threshold: `0.50`
- Max detections per frame: `20`

Object tracking behavior:
- Lightweight IoU-based tracking assigns a persistent `track_id` to each detected object across frames.
- Track IDs are stored in each object record under `per_frame_results[].objects[].track_id`.
- Total number of distinct tracked objects is reported in `summary.unique_object_tracks`.

### Speech Customization (Mini-Milestone 2 evidence)

The speech module in `speech_recognizer.py` is based on OpenAI Whisper and includes a domain adaptation layer:
- Domain vocabulary correction dictionary (post-processing)
- Optional quality comparison using Word Error Rate (WER) when a reference transcript is provided

`/transcribe` supports two optional form fields:
- `apply_corrections=true|false` (default `true`)
- `reference_text=...` (ground truth transcript for evaluation)

When `reference_text` is provided, the API returns:
- `evaluation.raw_wer`
- `evaluation.corrected_wer`
- `evaluation.relative_improvement_pct`

### Example curl commands

```bash
# Health check
curl http://localhost:5001/health

# Face detection
curl -X POST -F "file=@frame.jpg" http://localhost:5001/detect/faces

# Object detection (custom confidence threshold)
curl -X POST -F "file=@frame.jpg" -F "conf=0.35" http://localhost:5001/detect/objects

# Speech transcription
curl -X POST -F "file=@audio.wav" http://localhost:5001/transcribe

# Speech transcription with domain corrections + WER comparison
curl -X POST \
    -F "file=@audio.wav" \
    -F "apply_corrections=true" \
  -F "reference_text=Hi, I am the presenter in this demo." \
    http://localhost:5001/transcribe

# Full video pipeline
curl -X POST -F "file=@Video_speech_object.mov" http://localhost:5001/analyze/video
```

### Example Python client

```python
import requests

BASE = "http://localhost:5001"

# Object detection
with open("frame.jpg", "rb") as f:
    resp = requests.post(f"{BASE}/detect/objects",
                         files={"file": f},
                         data={"conf": "0.4"})
print(resp.json())

# Transcription
with open("audio.wav", "rb") as f:
    resp = requests.post(f"{BASE}/transcribe", files={"file": f})
print(resp.json()["full_text"])
```

---

## Running Tests

With the Flask server running in one terminal, open another terminal:

```bash
python tests/test_api.py http://localhost:5001
```

For a quick manual UI check, open:

```text
http://localhost:5001/ui
```

Then upload a sample video such as `Video_speech_object.mov`.

---



## Video Sources Used

- Sample videos recorded for testing and demos were used.
- One sample video source was taken from Pexels.

---

## Models Used

| Task               | Model                          | Size   |
|--------------------|-------------------------------|--------|
| Face Detection     | MediaPipe FaceDetection        | ~9 MB  |
| Object Detection   | YOLOv8n (Ultralytics COCO)     | ~6 MB  |
| Speech Recognition | OpenAI Whisper base            | ~74 MB |

---

## Mini-Milestone 3 — Test, validate, demo

End-to-end validation across **3 different sample videos** with
captured screenshots and a structured demo recording plan.

```bash
# Start the API in one terminal
python app.py

# Run the validation script in another terminal
python mini_milestone_3/validation.py
```

This will:
- POST each of `Video.mp4`, `Video_speech.mov`, `Video_speech_object.mov`
  to `/api/orchestrate`
- Save the JSON response per sample to `mini_milestone_3/results/`
- Decode the base64 preview frames to PNGs in
  `mini_milestone_3/screenshots/<sample>/preview_NN.png`
- Build three report-ready PNG cards per sample (summary card, preview
  mosaic, transcript card)
- Write `mini_milestone_3/VALIDATION_REPORT.md`

**Run offline if the API isn't started** (uses the existing
`sample_outputs/`):
```bash
python mini_milestone_3/validation.py --offline
```

See `mini_milestone_3/README.md` for the full submission packet:
- `DEMO_PLAN.md` — minute-by-minute script for the demo recording
- `SETUP_WALKTHROUGH.md` — from-zero install steps
- `ISSUES_AND_RESOLUTIONS.md` — 6 real issues + fixes
- `VALIDATION_REPORT.md` — auto-generated validation summary

---

## Mini-Milestone 4 — Orchestration API + Web UI 

**Single API call** that hides all three AI models behind one endpoint:
```bash
curl -X POST -F "file=@Video_speech_object.mov" \
     http://localhost:5001/api/orchestrate
```

**Web UI** for non-developers:
```
http://localhost:5001/ui
```

Both are implemented in `orchestrator.py` (logic), `app.py` (route),
and `templates/ui.html` (UI).

Documentation for the orchestration work lives in `mini_milestone_4/`:
- `CHATGPT_PROMPTS.md` — prompt log and results
- `CODE_CHANGES.md` — implementation changes and rationale
- `DEBUGGING_LOG.md` — debugging notes

See `mini_milestone_4/README.md` for the full submission packet.
