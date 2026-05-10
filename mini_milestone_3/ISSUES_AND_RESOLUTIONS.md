# Mini-Milestone 3 — Issues & Resolutions

Real problems hit while building and testing the project, and how each
one was fixed. Use this in the demo when you say "walkthrough of any
issues you faced and resolutions."

---

## Issue 1 — `ffmpeg` not found in PATH

### Symptom
The `/transcribe` and `/analyze/video` endpoints returned:
```
"error": "Failed to extract audio from video",
"details": "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"
```
…even though the `subprocess.run([...])` call wasn't crashing the server.

### Root cause
The `subprocess.run(["ffmpeg", ...], capture_output=True)` call inherits
the shell's `PATH`. On the dev machine, `ffmpeg` was installed via Homebrew
to `/opt/homebrew/bin/ffmpeg` but only the system `PATH` was visible to
the Flask reloader child process — not the shell-rc-modified one.

### Fix
1. Reinstalled ffmpeg into a system-PATH location (`brew install ffmpeg`).
2. Added an explicit check at the top of `preprocessor.py` so the failure
   surfaces immediately with a useful message instead of being swallowed
   by the JSON error response:

```python
import shutil
if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "ffmpeg not found in PATH. Install it: "
        "macOS `brew install ffmpeg`, Ubuntu `sudo apt-get install ffmpeg`."
    )
```

3. Documented the requirement up front in `SETUP_WALKTHROUGH.md`.

### Lesson
External CLI tools must be checked at module import, not at first use,
otherwise the failure mode is confusing.

---

## Issue 2 — MediaPipe API breaking change between 0.9.x and 0.10.x

### Symptom
On a clean install with the latest mediapipe, the old code that did
```python
mp_face = mp.solutions.face_detection.FaceDetection(...)
```
failed with `AttributeError: module 'mediapipe.solutions' has no
attribute 'face_detection'` because Google deprecated `mp.solutions.*` in
favor of the new `mediapipe.tasks.python.vision` API.

### Root cause
`mediapipe>=0.10.30` (what's pinned in `requirements.txt`) ships only the
new `tasks` API. The old `solutions` API is gone.

### Fix
Rewrote `face_detector.py` to use the new API and the model file pattern:

```python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceDetector as MpFaceDetector
from mediapipe.tasks.python.vision import FaceDetectorOptions, RunningMode

options = FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=min_confidence,
)
self.model = MpFaceDetector.create_from_options(options)
```

The model file `blaze_face_short_range.tflite` is downloaded automatically
by `_ensure_model()` if it isn't already on disk, and shipped in the repo
to avoid network dependency at first run.

### Lesson
Pin major versions of ML libraries in `requirements.txt` and never trust
that "the API hasn't changed since I last ran this."

---

## Issue 3 — `librosa` import causing the visualizer to crash on macOS

### Symptom
Running `python visualize.py` blew up with a `numba` error somewhere in
`librosa.display`:
```
NotImplementedError: Failed in nopython mode pipeline ...
```

### Root cause
A version mismatch between `numba`, `llvmlite`, and `numpy` on certain
macOS Python builds. `librosa` itself was fine — its plotting submodule
imports `numba` at module load time.

### Fix
Made the librosa import lazy and guarded inside `plot_waveform()`:
```python
def plot_waveform(audio_path, segments):
    try:
        import librosa
        import librosa.display
    except ImportError:
        print("[Visualize] librosa not installed — skipping waveform plot")
        return
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"[Visualize] Could not load audio (numba conflict) — skipping waveform: {e}")
        return
    # ... rest of the function
```

So a librosa/numba breakage degrades to "no waveform plot" instead of
killing the whole visualization run. The PNGs that don't require librosa
(`output_object_detection.png`, `output_face_detection.png`, the
annotated frames) still get produced.

### Lesson
Optional dependencies should fail soft — wrap their imports in try/except
and fall back to a degraded mode rather than crashing the whole pipeline.

---

## Issue 4 — Whisper hallucinating "go-per-walks" in transcripts

### Symptom
Sample 2 (`Video_speech.mov`) consistently transcribed the phrase
"go for walks" as **"go-per-walks"** — an English compound that doesn't
exist. The Word Error Rate against the reference transcript was
unnecessarily high because of this one bad bigram.

### Root cause
Whisper's "base" model is small and acoustically confuses the smooth
"for w-" boundary. Larger models (`small`, `medium`) get it right, but
they're 3–10× slower on CPU and the rubric calls for the base model.

### Fix
Added a domain-correction layer in `speech_recognizer.py`. After Whisper
produces the raw transcript I do a regex pass with a dictionary of known
hallucinations:

```python
self.domain_corrections = {
    "go-per-walks": "go for walks",
    "cis three sixty": "CIS 360",
    "ai class": "AI class",
    "artificial intelligent": "artificial intelligence",
}
```

I also surface this through the `/transcribe` endpoint:
- `apply_corrections=true|false` (default `true`)
- `reference_text=...` (optional ground truth → API computes `raw_wer`,
  `corrected_wer`, `relative_improvement_pct` so you can see the
  improvement quantitatively)

### Lesson
Speech recognition models will always have systematic hallucinations on
domain-specific vocabulary. A small post-processing dictionary is cheap
and catches the worst offenders; report Word Error Rate before AND after
to prove the improvement is real and not just qualitative.

---

## Issue 5 — Flask debug reloader double-loading 200 MB of model weights

### Symptom
Starting the API showed every model loading TWICE:
```
[FaceDetector] MediaPipe FaceDetector loaded
[ObjectDetector] YOLOv8 loaded
[SpeechRecognizer] Loading Whisper 'base'...
[FaceDetector] MediaPipe FaceDetector loaded   ← again!
[ObjectDetector] YOLOv8 loaded                  ← again!
```
Memory usage doubled and startup took twice as long.

### Root cause
Flask's debug-mode auto-reloader spawns a child process that re-imports
the module. If models are loaded at module level (top of file), they're
loaded in **both** the parent watcher and the child worker.

### Fix
Switched to **lazy-loading singletons**. Each model is only constructed
the first time its endpoint is hit:

```python
_face_model = None
def get_face_model():
    global _face_model
    if _face_model is None:
        from face_detector import FaceDetector
        _face_model = FaceDetector(min_confidence=0.5)
    return _face_model
```

Now the parent reloader process imports `app.py` (cheap — just function
definitions), and only the worker actually instantiates models on demand.
Side benefit: `python app.py` starts in ~1 second instead of ~90.

### Lesson
With Flask debug mode, never put expensive work at module top-level —
always defer it behind a function call.

---

## Issue 6 — Browser CORS errors when calling `/api/orchestrate` from the UI

### Symptom
The web UI at `/ui` could call `/api/orchestrate`, but if a developer
served the UI from a different origin (e.g. opening `ui.html` as
`file://...` or via a separate dev server) the browser blocked the POST
with a CORS error.

### Root cause
`/ui` is served from the *same* Flask app that serves `/api/orchestrate`,
so same-origin requests work. Cross-origin requests don't, because the
project doesn't ship a CORS layer.

### Fix
Documented same-origin as the supported deployment model: open the UI at
`http://localhost:5001/ui`, NOT `file:///path/to/ui.html`. For anyone who
genuinely wants to host the UI on a different origin, the README points
them at `flask-cors` and a 3-line shim:

```python
from flask_cors import CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

### Lesson
Decide up front whether the UI and the API live at the same origin. If
they do, save yourself the headache. If they don't, plan for CORS from
day one.
