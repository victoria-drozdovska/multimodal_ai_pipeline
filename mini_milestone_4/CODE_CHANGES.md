# Mini-Milestone 4 — Code Changes Made to ChatGPT's Output

ChatGPT's first drafts of `orchestrator.py` and `templates/ui.html`
were close but never drop-in. This file lists every meaningful change I
had to make, why, and what the original ChatGPT version looked like.

---

## Changes to `orchestrator.py`

### Change 1 — Replaced ChatGPT's invented model wrappers with the real ones

**ChatGPT version:**
```python
class FaceModel:
    def __init__(self):
        import mediapipe as mp
        self.detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5)
    def predict(self, image_path):
        ...

class ObjectModel:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
    ...
```

**Problems:**
1. `mp.solutions.face_detection` is the **deprecated 0.9.x API**. The
   project pins `mediapipe>=0.10.30` which removed it.
2. ChatGPT invented its own `predict()` method names. My existing
   classes already use `detect(image_path)`, returning a specific JSON
   shape that the rest of the codebase depends on.
3. There was no `_attach_track_ids` tracking logic — the existing
   `ObjectDetector` already implements IoU-based tracking and the
   pipeline relies on the `track_id` field.

**My fix:**
Throw the wrappers out entirely and import the existing classes:
```python
def _get_face_model():
    global _face_model
    if _face_model is None:
        from face_detector import FaceDetector
        _face_model = FaceDetector(min_confidence=0.5)
    return _face_model
```

The orchestrator never instantiates a model directly — it always goes
through these getters. This also gives us lazy loading for free.

---

### Change 2 — Fixed `_attach_track_ids` invocation

**ChatGPT version:**
```python
tracked = obj_det._attach_track_ids(per_frame)  # crashes
```

**Problem:** The actual signature in `object_detector.py` is:
```python
def _attach_track_ids(self, results: list[dict]) -> list[dict]:
    # expects: [{"objects": [...]}, {"objects": [...]}, ...]
```
But `per_frame` in my orchestrator is a richer record with `frame`,
`timestamp_sec`, `num_faces`, `faces`, etc. — `_attach_track_ids` was
silently iterating over those extra keys and only the `"objects"` slot
mattered, but the way ChatGPT wrote it produced unbound-track-id errors.

**My fix:**
Wrap the call so we hand it only what it needs, then merge the tracked
objects back in:
```python
tracked = obj_det._attach_track_ids(
    [{"objects": fr["objects"]} for fr in per_frame]
)
for fr, tr in zip(per_frame, tracked):
    fr["objects"] = tr["objects"]
```

---

### Change 3 — Added the "no audio track" branch

**ChatGPT version:**
```python
audio_wav = workdir / "audio.wav"
subprocess.run(["ffmpeg", "-i", video_path, str(audio_wav)])
transcription = whisper_model.transcribe(str(audio_wav))
```

**Problem:** Sample 1 (`Video.mp4`) has **no audio track**. ffmpeg
silently produces a zero-byte WAV in that case, which then crashes
Whisper with a cryptic numpy shape error.

**My fix:**
Capture the ffmpeg return code AND check the file size, then fall back
to a stub transcription:
```python
ffmpeg_res = subprocess.run([...], capture_output=True)
has_audio = (ffmpeg_res.returncode == 0
             and audio_wav.exists()
             and audio_wav.stat().st_size > 0)

if enable_speech and has_audio:
    transcription = _get_whisper_model().transcribe(str(audio_wav))
else:
    transcription = {
        "model": "skipped" if not enable_speech else "no_audio_track",
        "full_text": "", "language": "N/A",
        "word_count": 0, "segment_count": 0, "segments": [],
    }
```

This is what makes Sample 1 work end-to-end instead of 500-erroring.

---

### Change 4 — Added the `_pick_preview_frames` heuristic

**ChatGPT version:** picked the first 6 frames blindly:
```python
preview_indices = list(range(min(6, len(per_frame))))
```

**Problem:** the first 6 frames of many videos are intro shots with no
detections (think Sample 3, where the bottle doesn't enter the scene
until ~4s in). The "preview" was therefore boring and showed nothing
interesting.

**My fix:** wrote a 3-tier ranking — frames with BOTH faces and
objects first, then frames with at least one detection, then evenly-
spaced fallback. See `_pick_preview_frames(...)` in
`orchestrator.py`. This gives the UI a much more useful set of
previews.

---

### Change 5 — Annotated frame size + scaling math

**ChatGPT version:** ran annotation on the original 1620×1080 (or
2160×4096!) frame and base64-encoded the JPEG straight from disk. A
single response could top **10 MB** of base64 data.

**My fix:** downsized to a max width of **960 px** before drawing and
scaled the bounding-box coordinates to match:
```python
img = Image.open(frame_path).convert("RGB")
if img.width > max_width:
    new_h = int(img.height * (max_width / img.width))
    img = img.resize((max_width, new_h), Image.LANCZOS)

sx = img.width / Image.open(frame_path).width
sy = img.height / Image.open(frame_path).height
# ... then scale every (x1,y1,x2,y2) by (sx, sy)
```

Final response size dropped from ~10 MB to ~600 KB. The web UI loads
visibly faster.

---

### Change 6 — Stripping `frame_path` before returning

**ChatGPT version:** returned the absolute temp-dir path
(`/tmp/orchestrator_xyz/frames/frame_00000_t0.00s.jpg`) inside every
`per_frame` record.

**Problem:** that path leaks server filesystem layout and is useless to
the client (the temp dir is deleted by the time the response arrives).

**My fix:** pop it before returning:
```python
for fr in per_frame:
    fr.pop("frame_path", None)
```

Only the basename `frame` survives.

---

## Changes to `templates/ui.html`

### Change 7 — Switched from React to vanilla HTML/JS

**ChatGPT version:** assumed React + Tailwind via CDN.
**Problem:** the assignment runs from `python app.py` with no Node
toolchain. Adding a build step or relying on `cdn.jsdelivr.net` would
make the project fragile in offline grading environments.
**My fix:** rewrote the whole file as a single `templates/ui.html`
with inline `<style>` and one `<script>` block. Zero external
dependencies.

### Change 8 — Replaced `localStorage` with in-memory state

**ChatGPT version:** persisted the last response to `localStorage` so
a refresh would reload it.
**Problem:** payloads include base64 image data and routinely exceed
the 5 MB localStorage quota; the writes silently failed and the
"reload" feature did nothing.
**My fix:** kept everything in a `window.__lastResult__` JS variable
purely for dev-console inspection, and removed the persistence
feature. (Also matches the assignment's stateless API model.)

### Change 9 — Tightened the dropzone semantics

**ChatGPT version:** had a `<form>` element with the dropzone inside.
**Problem:** a `<form>` causes the page to reload on Enter when any
input has focus, which would lose the file selection.
**My fix:** removed the form. The button is plain `<button>`. The
dropzone is `tabindex="0"` and listens for Enter/Space on `keydown`
to open the file picker — keyboard-accessible without a form.

### Change 10 — Added the elapsed-timer and stage chips

**ChatGPT version:** had a single "Running…" message.
**My fix:** added the staged-progress UI described in `CHATGPT_PROMPTS.md`
prompt B3. The timeline gives users useful feedback during the 30-90s
that the orchestration call takes on CPU.

### Change 11 — Made the raw-JSON viewer hide base64 blobs

**ChatGPT version:** stringified the entire response, which made the
"Show raw JSON" expander unusable (megabytes of base64 in one block).
**My fix:**
```javascript
const light = JSON.parse(JSON.stringify(j));
(light.preview_frames || []).forEach(p => p.image_data_url = "<base64 hidden>");
$("rawJson").textContent = JSON.stringify(light, null, 2);
```
Now the raw JSON expander is actually readable and proves the schema
without flooding the page.

---

## Changes to `app.py`

### Change 12 — Wired `/api/orchestrate` and `/ui` into the existing app

**Reasoning:** the original `app.py` from Milestone 2 already had
lazy-loading getters, file-extension guards, and a global error
handler. Rather than letting ChatGPT scaffold a new Flask app, I added
just two routes (`/api/orchestrate` and `/ui`) and a couple of imports
to the existing file. This kept the Milestone 2 endpoints (`/health`,
`/detect/faces`, `/detect/objects`, `/transcribe`, `/analyze/video`)
unchanged so existing tests in `tests/test_api.py` still pass.

### Change 13 — Made `/api/orchestrate` reuse `orchestrator.orchestrate_video`

**ChatGPT version:** inlined the orchestration logic directly into the
Flask route handler.
**Problem:** the same logic is also reachable from the CLI
(`python orchestrator.py video.mp4`) and would have to be duplicated.
**My fix:** kept the route handler as a thin wrapper — parse form
fields, save the upload, call `orchestrate_video(path, ...)`, return
the result. Single source of truth.

---

## Summary of changes

| # | File                  | What changed                                       | Why                                |
|---|-----------------------|----------------------------------------------------|-------------------------------------|
| 1 | `orchestrator.py`     | Use real `FaceDetector` / `ObjectDetector` / `SpeechRecognizer` | ChatGPT's wrappers used the deprecated MediaPipe API |
| 2 | `orchestrator.py`     | Wrap `_attach_track_ids` input properly            | Signature mismatch                  |
| 3 | `orchestrator.py`     | Handle missing audio track                         | Sample 1 has no audio              |
| 4 | `orchestrator.py`     | Smarter preview frame picker                       | First 6 frames are usually boring  |
| 5 | `orchestrator.py`     | Resize frames + scale bbox coords before base64    | Response size 10 MB → 600 KB       |
| 6 | `orchestrator.py`     | Strip absolute paths from response                 | Privacy + cleanliness              |
| 7 | `templates/ui.html`   | Drop React/Tailwind, use vanilla HTML/JS           | No build step, offline-safe        |
| 8 | `templates/ui.html`   | Drop localStorage                                  | 5 MB quota issues                  |
| 9 | `templates/ui.html`   | Drop `<form>` element                              | Enter-key reload bug               |
|10 | `templates/ui.html`   | Add elapsed-timer + stage chips                    | UX during long sync calls          |
|11 | `templates/ui.html`   | Hide base64 in raw-JSON viewer                     | Make the expander usable           |
|12 | `app.py`              | Add `/api/orchestrate` + `/ui` routes only         | Don't break Milestone 2 endpoints  |
|13 | `app.py`              | Reuse `orchestrate_video()`                        | Single source of truth             |
