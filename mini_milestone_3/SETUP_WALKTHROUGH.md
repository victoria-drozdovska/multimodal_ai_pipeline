# Mini-Milestone 3 — Setup Walkthrough

Guide to getting the project running on a fresh machine. Use this script during the demo recording when you say
"step-by-step walk-through of your final project setup.

---

## 1. System prerequisites

| Requirement | Why we need it                                      | How to check               |
|-------------|-----------------------------------------------------|----------------------------|
| Python ≥3.10| `match` statements + `int \| None` type hints      | `python --version`         |
| `ffmpeg`    | Audio extraction in `preprocessor.py` and `app.py` | `ffmpeg -version`          |
| ~2 GB free  | Whisper + YOLO weights are downloaded on first run | `df -h .`                  |
| Webcam not required | Sample videos ship with the project        | —                          |

If `ffmpeg` is missing:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install -y ffmpeg`
- Windows: download a static build from ffmpeg.org and add it to PATH

---

## 2. One-time installation

```bash
# 1. Clone or unzip the project
cd Multimodal-Media-Analysis-API-for-Face-Object-and-Speech-Recognition

# 2. Create an isolated virtual environment
python -m venv .venv

# 3. Activate it
#   macOS/Linux:
source .venv/bin/activate
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 4. Install Python dependencies (~3 min)
pip install --upgrade pip
pip install -r requirements.txt

# 5. (Optional) install requests just for the validation script
pip install requests
```

The first `python app.py` call will trigger:
- YOLOv8n weights download → `yolov8n.pt` (~6 MB) — already shipped here
- MediaPipe BlazeFace model → `blaze_face_short_range.tflite` (~225 KB) — already shipped here
- Whisper "base" weights → `~/.cache/whisper/base.pt` (~140 MB) — downloaded on first transcription

So the very first `/transcribe` or `/api/orchestrate` call will be slow.
Every call after that is fast because the models stay in memory.

---

## 3. Verify the install

```bash
# Sanity check — should print "OK" and the model parameter counts.
python -c "from face_detector import FaceDetector; \
           from object_detector import ObjectDetector; \
           print('OK')"
```

---

## 4. Start the API

```bash
python app.py
```

Expected output:
```
============================================================
  Starting Multimodal Media Analysis API on port 5001
  Landing page : http://localhost:5001
  Web UI       : http://localhost:5001/ui
  Health check : http://localhost:5001/health
============================================================
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
 * Running on http://192.168.x.x:5001
```

Open a browser at `http://localhost:5001` to confirm.

---

## 5. Smoke-test from a second terminal

```bash
# Health
curl http://localhost:5001/health

# Face detection on a frame the project ships with
curl -X POST -F "file=@processed/frames/frame_00010_t10.00s.jpg" \
     http://localhost:5001/detect/faces

# Full orchestration — single consolidated multimodal call
curl -X POST -F "file=@Video_speech_object.mov" \
     http://localhost:5001/api/orchestrate \
     -o orchestrate_result.json
```

`orchestrate_result.json` will contain:
- `video_metadata` – fps, resolution, duration
- `summary` – frame/face/object counts and unique tracks
- `transcription` – full text + timestamped segments
- `timeline` – one row per sampled frame with active subtitle
- `preview_frames` – base64 JPEGs with bounding boxes pre-drawn
- `per_frame_results` – exhaustive per-frame data

---

## 6. Open the Web UI (Mini-Milestone 4)

Browser → `http://localhost:5001/ui`

1. Drag any of the three sample videos onto the dropzone (or click).
2. Leave all toggles on.
3. Click **Analyze video** and wait for the progress strip to fill.
4. Scroll through the result panels (Summary → Annotated frames →
   Transcript → Timeline → Raw JSON).

---

## 7. Run the automated validation script

In a second terminal (with the API still running in the first):

```bash
python mini_milestone_3/validation.py
```

This:
- Posts each sample video to `/api/orchestrate`
- Saves the JSON response per sample
- Decodes the base64 preview frames into PNGs
- Builds three "report-ready" screenshot cards per sample
- Writes `mini_milestone_3/VALIDATION_REPORT.md` summarizing all three runs

If the API is NOT running you can still produce screenshots from the
saved sample outputs:
```bash
python mini_milestone_3/validation.py --offline
```

---

## 8. Stop the server

`Ctrl+C` in the terminal running `python app.py`. The Flask debug reloader
will exit cleanly.
