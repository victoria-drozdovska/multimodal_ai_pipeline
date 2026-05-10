# Mini-Milestone 4 — ChatGPT Prompt Log

This file is organized chronologically as a development journal.

---

## Phase A — Designing the orchestration API

### Prompt A1 — worked

> **My exact prompt:**
> *"I have a Flask app at `app.py` that exposes three separate endpoints
> — `/detect/faces` (MediaPipe), `/detect/objects` (YOLOv8), and
> `/transcribe` (OpenAI Whisper). Right now a user has to call all three
> separately and stitch the results together themselves. I want to add
> ONE new endpoint, `POST /api/orchestrate`, that accepts a single video
> upload and internally:*
>
> *1. extracts frames at 1 fps and audio at 16 kHz mono*
> *2. runs face + object detection on every frame*
> *3. runs Whisper on the audio*
> *4. aligns transcript segments with frames by timestamp*
> *5. returns ONE consolidated JSON with summary, per-frame results,
>    transcript segments, and a small carousel of annotated preview
>    frames embedded as base64 JPEGs.*
>
> *Don't write the implementation yet — first, design the JSON response
> schema. Show me the top-level keys and what each one contains, with a
> short comment on each. The key constraint: a frontend developer should
> be able to render an entire results dashboard from this one response
> without any second round-trip."*

**Why this prompt worked:** I explicitly described
1. the **existing context** (3 endpoints, what each does),
2. the **goal** (one consolidated endpoint),
3. the **steps** the implementation needs to perform internally,
4. the **constraint** (no second round-trip needed for the UI),
5. and what I wanted as the **output format** (just the schema, not the
   code yet).

ChatGPT returned a clean schema with `status`, `video_metadata`,
`models_used`, `summary`, `transcription`, `timeline`,
`preview_frames`, and `per_frame_results`. I used it almost verbatim —
see `orchestrator.py` for the final shape.

### Prompt A2 — didn't work

> **My exact prompt (first attempt):**
> *"Write me a Python orchestrator for my AI video pipeline."*

**Why this prompt didn't work:** Way too vague. ChatGPT produced a 200-line
class that:
- invented its own model wrapper interfaces (didn't match my
  `FaceDetector`, `ObjectDetector`, `SpeechRecognizer` classes)
- used `asyncio` for stages that are CPU-bound and synchronous in
  reality
- imported libraries I'd never installed
- had no error handling for the "no audio track" case my Sample 1 hits

**What I learned:** never use the word "orchestrator" without immediately
defining what it needs to wrap. ChatGPT treated it as a generic
microservices pattern instead of the very specific "wrap these three
existing classes" task I had in mind.

### Prompt A3 — worked (after fixing A2)

> **My exact prompt:**
> *"Here are my three existing model classes. Their public APIs are:*
>
>     `FaceDetector(min_confidence=0.5).detect(image_path) → {"frame": str, "num_faces": int, "faces": [{"bbox":[x1,y1,x2,y2], "confidence": float}]}`
>     `ObjectDetector(conf=0.4).detect(image_path, conf=0.4) → {"frame": str, "num_objects": int, "objects": [{"label": str, "class_id": int, "confidence": float, "bbox":[...]}]}`
>     `SpeechRecognizer(model_size="base").transcribe(audio_path) → {"language", "full_text", "segments": [{"start_sec","end_sec","text","confidence",...}], ...}`
>
> *Do NOT change any of these classes. Write a single function
> `orchestrate_video(video_path) → dict` that:*
> *1. uses ffmpeg to extract a 16 kHz mono WAV (skip transcription
>    gracefully if the video has no audio)*
> *2. uses OpenCV to sample frames at 1 fps*
> *3. calls the three classes with their actual public methods*
> *4. uses the YOLO `_attach_track_ids` helper that already exists
>    on `ObjectDetector` for tracking*
> *5. for the transcript, looks up the active segment for each frame
>    by matching `start_sec ≤ t ≤ end_sec`*
> *6. returns the schema we agreed on in the previous turn.*

**Why this prompt worked:** I pasted the **exact** function signatures
of the three classes, told ChatGPT what NOT to change, and pointed at a
specific helper (`_attach_track_ids`) that already existed. The result
was almost drop-in.

---

## Phase B — Designing the web UI

### Prompt B1 — worked

> **My exact prompt:**
> *"I need a single-file HTML page (no build step, no React, no
> bundler) at `templates/ui.html`. It will be served by Flask at GET
> `/ui`. The page consumes ONE backend endpoint: `POST /api/orchestrate`
> with form-data fields `file` (video), `enable_faces` (bool),
> `enable_objects` (bool), `enable_speech` (bool), `conf` (0.1-0.95),
> `max_preview_frames` (1-20).*
>
> *The response shape is in the JSON schema we agreed on earlier.*
>
> *Layout requirements:*
> *- Drag-and-drop video upload (or click to choose)*
> *- Three checkbox toggles for the three models*
> *- "Analyze video" button*
> *- Indeterminate progress bar with named pipeline stages while we wait*
> *- Six summary KPI cards (frames, faces, objects, classes, tracks, words)*
> *- Grid of base64-decoded annotated preview frames*
> *- Transcript panel with full text + per-segment timestamps*
> *- Timeline panel: one row per sampled frame*
> *- Collapsible raw-JSON viewer (with base64 image data hidden so it's readable)*
>
> *Style: dark theme, monospace + sans mix, accent green #7ee8a2,
> orange #ffb347 for objects, blue #a0c4ff for faces. Mobile-friendly.
> No external CSS frameworks, no CDN dependencies. Use only inline
> `<style>` and vanilla JS."*

**Why this prompt worked:**
- I pinned the **technical constraints** (single file, no build, no
  framework) up front.
- I gave the **API contract** so the JS knew exactly what to fetch.
- I gave a **complete layout spec** (8 panels) instead of "make it
  look nice."
- I gave **design tokens** (specific colors, theme).

The first response was 90% correct. I made a few small changes (see
`CODE_CHANGES.md`).

### Prompt B2 — didn't work

> **My exact prompt (first attempt):**
> *"Make the UI prettier."*

**Why it didn't work:** Empty subjective adjective. ChatGPT replaced
my color scheme with a Bootstrap-y one, broke the dark mode, removed
the monospace headings, and switched all the icons to emoji that don't
render in many fonts.

**What I did instead:** went back to **B1's specific design tokens**
and asked for one targeted change at a time:
> *"In `ui.html`, the `.card` element currently has no hover state.
> Add `transform: translateY(-1px)` and a subtle border-color shift
> on hover, but ONLY for `.card` — don't touch any other selector."*

That worked.

### Prompt B3 — worked

> **My exact prompt:**
> *"The progress card I have shows a static 'Running...' string. I want
> it to LOOK like the pipeline is progressing through stages even
> though `/api/orchestrate` is a single blocking POST. Add a row of
> stage chips (upload, frames, faces, objects, speech,
> merge). On click of "Analyze video", schedule each chip to flip to
> a 'done' style at staggered intervals (800ms, 2400ms, 4500ms, 7000ms,
> 10500ms) using `setTimeout`. Also start a 100ms-tick elapsed timer
> that displays `elapsed: X.Xs` underneath. Both stop when the response
> comes back."*

**Why this prompt worked:**
- I was **honest** about the constraint (the backend doesn't stream).
- I gave the **exact timings** so it didn't guess.
- I described **what state the UI should be in** before, during, and
  after the call.

---

## Phase C — Debugging help

### Prompt C1 — worked

> **My exact prompt:**
> *"My `/api/orchestrate` is returning HTTP 500. The Flask log shows:*
>
>     File "orchestrator.py", line 142, in orchestrate_video
>         tracked = obj_det._attach_track_ids(...)
>     AttributeError: 'ObjectDetector' object has no attribute '_attach_track_ids'
>
> *But I can see `_attach_track_ids` defined in `object_detector.py`.
> Here's the full class — what am I missing? [pasted entire `object_detector.py`]"*

**Why this prompt worked:** I gave the **exact traceback** AND the
**source of the class**. ChatGPT immediately spotted that I was calling
`_attach_track_ids(per_frame)` directly with a list of `per_frame_results`
dicts, but the method actually expects a list of `{"objects": [...]}`
records. The fix was a one-line list-comprehension wrapper. See
`DEBUGGING_LOG.md` for the resolution.

### Prompt C2 — partially worked

> **My exact prompt (first attempt):**
> *"My API is slow."*

**Why it didn't work:** No specifics. ChatGPT recommended generic
optimizations (asyncio, GPU, caching) that didn't apply to my situation
— most of the time was Whisper inference, which can't be made faster
without changing models.

**Better follow-up:**
> *"On Sample 3 (`Video_speech_object.mov`, 47s, 1620×1080), the
> orchestration call takes ~120s on CPU. I profiled it and 95% of that
> is `whisper.load_model('base')` + `model.transcribe()`. I'm running
> the model on CPU with no GPU available. I don't want to switch
> Whisper sizes. Two specific questions:*
> *1. Should I be loading the Whisper model once at startup vs.
>    lazy-loading it on first request?*
> *2. Is there a way to keep `model.transcribe()` results across
>    repeated calls with the same audio file?"*

ChatGPT answered both (yes to startup loading if memory allows, and
suggested an in-memory LRU cache keyed by audio SHA-1) and the prompt
was useful — but only because I gave concrete numbers, the profiler
output, and the specific questions.

---

## What I learned about prompting ChatGPT

| Pattern that worked                                                | Pattern that didn't                              |
|--------------------------------------------------------------------|--------------------------------------------------|
| Pasting the exact function signatures of my existing classes       | Asking for "an orchestrator" without context    |
| Giving the EXACT JSON schema I want back                           | "Make it pretty" / "make it better"             |
| Saying what NOT to change                                          | Asking for refactors that touch unrelated code  |
| One change per turn                                                | Asking for many changes in a single message     |
| Pasting the actual error traceback                                  | "Why is this broken?"                           |
| Numbers (47s, 1620×1080, 120s)                                     | "It's slow"                                     |
| Stating the constraint up front (no React, no CDN, single file)    | Letting GPT pick the stack                      |

**Three rules I now follow:**
1. **Pin the contract first.** Define inputs, outputs, schemas, and
   what NOT to change before asking for code.
2. **One concrete change per turn.** Don't ask for "polish" — ask for
   specific selector → specific style.
3. **Always include the actual error.** Tracebacks beat narrative
   descriptions every time.
