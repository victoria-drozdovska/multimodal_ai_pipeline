# Mini-Milestone 4 — Debugging Log: ChatGPT Assistance

Six debugging sessions with ChatGPT during Mini-Milestone 4 development.
Each one documents what I asked, what ChatGPT gave back, and what
was/wasn't useful.

---

## Bug 1 — `AttributeError: 'ObjectDetector' object has no attribute '_attach_track_ids'`

### Prompt I sent
> *"My `/api/orchestrate` is returning HTTP 500. The Flask log shows:*
>
>     File "orchestrator.py", line 142, in orchestrate_video
>         tracked = obj_det._attach_track_ids(...)
>     AttributeError: 'ObjectDetector' object has no attribute '_attach_track_ids'
>
> *But I can see `_attach_track_ids` defined in `object_detector.py`.
> Here's the full class — what am I missing? [pasted entire `object_detector.py`]"*

### What ChatGPT gave me
ChatGPT spotted that the method exists, but I was actually getting a
different error than I thought — the **first** call was to the lazy
getter `_get_yolo_model()` which had crashed on a different exception
that was being suppressed by my try/except. The "no attribute" error
was a follow-on from `obj_det` being `None`.

It pointed me at this line in my orchestrator:
```python
obj_det = _get_yolo_model() if enable_objects else None
```
and asked: "what does the call return when `enable_objects=True` but
the lazy loader raises?"

### My resolution
The actual root cause: my Flask route was overriding `enable_objects`
to a string `"True"` instead of a bool, so `bool("True")` was True but
ChatGPT's diagnostic question forced me to add proper boolean parsing
(`parse_bool(...)` helper) and to not silently catch model loading
exceptions. The `_attach_track_ids` invocation was wrong too (see
`CODE_CHANGES.md` change #2) and that became visible only after the
boolean issue was fixed.

### What was useful
- ChatGPT didn't take the prompt at face value — it asked "what
returns None?" instead of just rewriting the line.
- It suggested re-checking my error handling, which uncovered a real
bug.

### What ChatGPT couldn't help with
- (limitation) It couldn't see my actual Flask request body, so it didn't know
that `enable_objects` was arriving as the string `"True"` rather than
a Python bool. I had to figure that out by adding a `print(repr(...))`
log line.

---

## Bug 2 — Whisper crashes on Sample 1's no-audio video

### Prompt I sent
> *"`POST /api/orchestrate` works on Video_speech.mov but crashes on
> Video.mp4 (a video that has no audio track). The traceback is:*
>
>     File "speech_recognizer.py", line 97, in transcribe
>         result = self.model.transcribe(str(audio_path), ...)
>     ValueError: max() arg is an empty sequence
>
> *I think Whisper is being handed a zero-length WAV. ffmpeg returned
> exit code 0, so my code thinks the audio extraction succeeded. How do
> I detect 'video has no audio track' robustly without parsing ffmpeg's
> stderr?"*

### What ChatGPT gave me
Three options:
1. Run `ffprobe -i input.mp4 -show_streams -select_streams a` first
   and check if there's at least one audio stream.
2. After the ffmpeg call, check `audio_wav.exists() and
   audio_wav.stat().st_size > 0`.
3. Pass `-shortest` and a duration check to ffmpeg to force an error
   if no audio is present.

It recommended option 2 as the cheapest. It also pointed out that
ffmpeg with no audio input often **does** produce a tiny WAV header (~44
bytes) so I'd need a size threshold higher than 44.

### My resolution
Went with option 2 with a small twist:
```python
has_audio = (ffmpeg_res.returncode == 0
             and audio_wav.exists()
             and audio_wav.stat().st_size > 0)
```
…and gated the Whisper call on `has_audio`. See change #3 in
`CODE_CHANGES.md`. Sample 1 now returns a clean response with an empty
transcription stub.

### What was useful
- Specific options ranked by cost/complexity.
- The footnote about ffmpeg producing a 44-byte WAV header — that
saved me from a flaky `> 0` check.

### What ChatGPT couldn't help with
- (limitation) It didn't know whether my specific ffmpeg version on macOS
behaved exactly like that. I had to verify by running
`ls -la audio.wav` after the ffmpeg call on a no-audio video.

---

## Bug 3 — Web UI freezes during `/api/orchestrate` call

### Prompt I sent
> *"After clicking the 'Analyze video' button in `templates/ui.html`,
> the entire page becomes unresponsive for 60-90 seconds while the
> orchestration call runs. The progress bar I have is also frozen
> (it's a CSS animation that should be looping). Why?"*

### What ChatGPT gave me
First answer was wrong — it suggested moving the fetch into a Web
Worker. That doesn't fix the issue: a `fetch()` is already async, so
the UI shouldn't freeze just from waiting for the response.

After I followed up with:
> *"The CSS animation is also frozen. I'm running this in Chrome, no
> blocking JS in the page."*

It correctly diagnosed: the problem was that I had `<input type=file>`
inside a `<form>` element, and pressing Enter on the file dialog was
**reloading the page** mid-fetch. The "freeze" was actually a partial
page reload.

### My resolution
- Removed the `<form>` (change #9 in `CODE_CHANGES.md`).
- The dropzone is now a `<div tabindex="0">` that opens the file
  picker via `keydown`.
- The CSS animation runs smoothly because the page never reloads.

### What was useful
- When I gave it the symptom that ruled out its first hypothesis, it
correctly pivoted to the actual cause.

### What ChatGPT couldn't help with
- (limitation) Its first suggestion (Web Workers) was a generic "the UI is frozen
so use a worker" pattern-match that was wrong for my case. I had to
push back with more specifics before it found the real bug.

---

## Bug 4 — Response size too large

### Prompt I sent
> *"My `/api/orchestrate` response is around 9.8 MB on Sample 3
> (a 47-second 1620×1080 video). 95% of the bytes are in the
> `preview_frames[].image_data_url` base64 strings. The web UI takes 3-4
> seconds to render after the response arrives, presumably from
> JSON.parse. What are my options to shrink this without losing
> information?"*

### What ChatGPT gave me
Four ranked options:
1. **Resize frames before base64-encoding** (cheap, no API change).
2. Switch from JPEG quality 95 → 80 (smaller, no perceptible loss).
3. Return preview frames as separate static files at
   `/preview/<id>.jpg` (would require a stateful server).
4. Stream the response as NDJSON.

It recommended **1 + 2** combined.

### My resolution
Implemented #1 and #2 (change #5 in `CODE_CHANGES.md`):
- Max width 960px before drawing
- JPEG quality 80
- Scale bbox coords by `sx, sy` so they still align with the resized
  image

Response size dropped from 9.8 MB to ~600 KB on the same input. UI
renders in <100ms now.

### What was useful
- Concrete numbers in the prompt got concrete options back.
- It correctly noted that scaling the image requires scaling the
bounding-box coordinates too — easy to forget.

### What ChatGPT couldn't help with
- (limitation) It didn't predict that downsizing 4096-tall portraits (Sample 1)
would change the box aspect ratios slightly because of integer
rounding. I noticed by visual inspection and added the `LANCZOS`
resampling filter to minimize the artifact.

---

## Bug 5 — Lazy-loaded models duplicating in Flask debug mode

### Prompt I sent
> *"When I start `python app.py`, the startup log shows each model
> being loaded TWICE:*
>
>     [FaceDetector] MediaPipe FaceDetector loaded
>     [ObjectDetector] YOLOv8 loaded
>     [SpeechRecognizer] Loading Whisper 'base'...
>     [FaceDetector] MediaPipe FaceDetector loaded
>     [ObjectDetector] YOLOv8 loaded
>
> *Why? Models should load lazily on first endpoint hit, not at
> startup."*

### What ChatGPT gave me
Two diagnoses:
1. Flask's debug auto-reloader spawns a child worker process — module-
   level imports happen in both. (This was the right answer.)
2. `app.run(debug=True)` is what causes it; switching to
   `debug=False` or running with `flask --no-reload` makes it go away.

It also pointed out that the issue would NOT exist for true lazy-
loaded models like mine — but it asked: "are you SURE the loader is
lazy? Show me the call sequence."

When I traced it, I found that the test in `tests/test_api.py` was
calling `get_face_model()` at import time inside the test setup, and
the Flask debug reloader was importing both files.

### My resolution
- Moved the test-time model warmup inside a function that's only
  called by the test runner's `main()`, not at module import.
- Confirmed the lazy getters were correct.
- Documented the debug-reloader caveat in `mini_milestone_3/ISSUES_AND_RESOLUTIONS.md`
  Issue 5.

### What was useful
- ChatGPT's "are you sure?" question — instead of immediately
rewriting code, it forced me to verify my assumptions.

### What ChatGPT couldn't help with
- (limitation) It couldn't tell that my test file was the actual culprit — I had
to grep for `FaceDetector(` to find every place that instantiates the
class.

---

## Bug 6 — `requests.post` timing out at 60s on the validation script

### Prompt I sent
> *"My `mini_milestone_3/validation.py` calls
> `requests.post('http://localhost:5001/api/orchestrate', files=..., timeout=60)`
> and gets a `requests.exceptions.ReadTimeout` on Sample 3. The Flask
> server log shows the orchestration is still running successfully —
> just slowly. What's a robust timeout strategy for endpoints that take
> 30-120 seconds depending on input?"*

### What ChatGPT gave me
- Use a `(connect_timeout, read_timeout)` tuple — short connect, long
  read. E.g. `timeout=(5, 900)`.
- Or implement a polling pattern: have the server return a job ID
  immediately, then poll a separate `/jobs/<id>` endpoint. (Major
  refactor, not what I want for this assignment.)
- Optionally `requests.Session()` with a retry adapter, but only for
  idempotent GETs.

It recommended option 1 for my case.

### My resolution
Set `timeout=900` in `validation.py`. ChatGPT's tuple suggestion
(short-connect, long-read) is the textbook-correct version, but for a
local dev server a single 900s timeout is fine and simpler. See
`mini_milestone_3/validation.py`:
```python
r = requests.post(f"{base_url}/api/orchestrate",
                  files=files, data=data, timeout=900)
```

### What was useful
- Distinguishing connect-timeout from read-timeout — that's the
actual conceptual fix.

### What ChatGPT couldn't help with
- (limitation) It suggested a job-queue refactor as if it were a small
optimization. For an assignment this would be massive over-engineering;
I had to push back: "I don't want to add a job queue for an
assignment, just give me the smallest viable fix."

---

## Summary

| Bug | Prompt quality   | ChatGPT assistance | What it couldn't do                           |
|-----|------------------|--------------------|-----------------------------------------------|
| 1   | High (full TB)   | Forced re-think    | Couldn't see my actual request body           |
| 2   | High (specific)  | Three ranked options | Didn't know my ffmpeg version's exact behavior |
| 3   | Med→High after follow-up | Pivoted on more info | First hypothesis (Web Worker) was wrong      |
| 4   | High (numbers)   | Concrete fix       | Missed the resampling-quality detail          |
| 5   | High (logs)      | Right diagnosis    | Couldn't trace which file was the culprit     |
| 6   | High (specific)  | Right concept      | Suggested over-engineered refactor            |

**Pattern:** when I gave ChatGPT specific tracebacks, logs, or numbers,
it was useful 80% of the time. When I gave vague descriptions, the
first response was almost always wrong and I had to follow up with
more context.
