# Mini-Milestone 4 — ChatGPT-Assisted Orchestration API & Web UI

## Requirement 1 — Orchestration API

> *"Develop orchestration API for your AI models. The API should accept
> video clip as input and orchestrate calls to various AI models you are
> using in this project and return a consolidated final response."*

**Endpoint:** `POST /api/orchestrate`

**Input (form-data):**
- `file` (required): a video (`.mp4`, `.mov`, `.avi`, `.mkv`)
- `enable_faces` (bool, default `true`)
- `enable_objects` (bool, default `true`)
- `enable_speech` (bool, default `true`)
- `conf` (float, default `0.40`): YOLO confidence threshold
- `sample_fps` (float, default `1.0`): how many frames per second to sample
- `max_preview_frames` (int, default `6`): how many annotated previews to embed

**Output (single JSON):**
```jsonc
{
  "status": "success",
  "elapsed_sec": 87.4,
  "video_metadata": { "filename", "width", "height", "fps", "duration_sec", "has_audio", ... },
  "models_used":    { "face_detection", "object_detection", "speech_recognition" },
  "summary":        { "frames_analyzed", "total_faces_detected", "total_objects_detected",
                      "unique_object_classes", "unique_object_tracks",
                      "transcript_word_count", "transcript_language", ... },
  "transcription":  { "language", "full_text", "segments": [...], ... },
  "timeline":       [ { "t_sec", "num_faces", "num_objects",
                        "object_labels", "subtitle" } ],
  "preview_frames": [ { "t_sec", "image_data_url",   // base64 JPEG
                        "num_faces", "num_objects",
                        "object_labels", "subtitle" } ],
  "per_frame_results": [ ... full per-frame face + object detail ... ]
}
```

**The "consolidated" promise:** a frontend can render an entire results
dashboard from this one response with **zero** additional round-trips.
The base64 preview frames are pre-annotated server-side so the UI
doesn't need image processing.

**Quick test (use any local demo clip):**
```bash
curl -X POST -F "file=@/path/to/your_video.mov" \
     http://localhost:5001/api/orchestrate \
     -o orchestrate_result.json
```

---

## Requirement 2 — Web UI

> *"Develop a web UI that will use the API from above step so your end
> users can use your AI models API using a web UI."*

**Route:** `GET /ui` (`http://localhost:5001/ui`)

**Features:**
- Drag-and-drop video upload (or click-to-choose)
- Three checkbox toggles to enable/disable each of the 3 models
- Numeric inputs for YOLO confidence and # of preview frames
- Indeterminate progress bar with named stage chips
  (upload, frames, faces, objects, speech, merge)
- Live elapsed-seconds counter while the call runs
- Six summary KPI cards (frames, faces, objects, classes, tracks, words)
- Grid of base64-decoded annotated preview frames with timestamp captions
- Transcript panel with full text + per-segment timestamps
- Per-frame timeline panel with subtitles
- Collapsible raw-JSON viewer

**No build step. No CDN. Single `templates/ui.html` file** with inline
`<style>` and one `<script>` block.

---

## Requirement 3 — Document the prompts you used
See `CHATGPT_PROMPTS.md`. It covers:

- **Phase A — Designing the orchestration API**: 3 prompts (2 worked, 1 didn't)
- **Phase B — Designing the web UI**: 3 prompts (2 worked, 1 didn't)
- **Phase C — Debugging help**: 2 prompts (1 worked, 1 partially)

Each prompt is reproduced verbatim with a short note on **why it
worked or didn't**.

---

## Requirement 4 — Document changes made to ChatGPT's code

See `CODE_CHANGES.md`. 13 specific changes are listed with:
- the original ChatGPT version
- why it didn't work in this project
- the fix I applied

Highlights:
- Replaced ChatGPT's invented `FaceModel`/`ObjectModel` wrappers with
  the real `FaceDetector`/`ObjectDetector`/`SpeechRecognizer` classes
- Fixed `_attach_track_ids` invocation
- Added the "no audio track" branch (Sample 1 has no audio)
- Wrote a smarter `_pick_preview_frames` heuristic
- Resized frames + scaled bbox coordinates before base64 to drop
  response size from 9.8 MB to 600 KB
- Replaced React+Tailwind with vanilla HTML/JS for offline safety
- Removed `localStorage` (5 MB quota issues)
- Added the elapsed-timer + stage-chip UX

---

## Requirement 5 — Document debugging help from ChatGPT

See `DEBUGGING_LOG.md`. 6 real bugs are walked through with:
- the prompt I sent
- what ChatGPT diagnosed
- the actual root cause and resolution
- what was useful vs. what ChatGPT couldn't help with

---

## Folder layout

```
mini_milestone_4/
├── README.md                     ← this file
├── CHATGPT_PROMPTS.md            ← all prompts I used + which worked
├── CODE_CHANGES.md               ← 13 changes I made to ChatGPT's code
└── DEBUGGING_LOG.md              ← 6 debugging sessions with ChatGPT

(actual code is in the project root)
project_root/
├── orchestrator.py               ← Mini-Milestone 4 orchestration logic
├── app.py                        ← exposes /api/orchestrate and /ui
└── templates/
    └── ui.html                   ← the web UI
```

---

## Quick demo

```bash
# 1. Start the server
python app.py

# 2. Use the API (provide a local video path)
curl -X POST -F "file=@/path/to/your_video.mov" \
     http://localhost:5001/api/orchestrate

# 3. Or use the web UI
open http://localhost:5001/ui
```
