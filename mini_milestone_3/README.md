# Mini-Milestone 3 — Test, Validate & Demo

This folder contains the Mini-Milestone 3 validation and demo support
materials that are actually committed in this repository.

---

## 1 · End-to-end validation

The validation flow exercises the live API on three samples when the
video files are available locally, then writes the response JSON and
report-ready screenshots into this folder.

| # | Sample | Purpose | Audio |
|---|--------|---------|-------|
| 1 | `Video.mp4` | Multi-person scene, no speech | No |
| 2 | `Video_speech.mov` | Speaker only | Yes |
| 3 | `Video_speech_object.mov` | Full multimodal demo | Yes |

### How to run

In one terminal:
```bash
python app.py
```

In another terminal:
```bash
python mini_milestone_3/validation.py
```

The script:
1. Pings `/health` to confirm the API is up.
2. POSTs each sample to `/api/orchestrate`.
3. Saves each JSON response to `mini_milestone_3/results/`.
4. Decodes every preview frame into PNG screenshots.
5. Generates three report-ready PNG cards per sample:
   - `00_summary_card.png` — KPI dashboard
   - `01_preview_mosaic.png` — annotated preview frames grid
   - `02_transcript_card.png` — transcript with timestamps
6. Writes `mini_milestone_3/VALIDATION_REPORT.md` summarizing the run.

If the API is offline, you can still generate the report from the saved
response JSON files by rerunning validation after the API is back up.

### Saved outputs

The generated response JSON files and PNG screenshots are created
locally and are excluded from git:

- `mini_milestone_3/results/`
- `mini_milestone_3/screenshots/`

Examples:

- `mini_milestone_3/screenshots/sample_1_video/00_summary_card.png`
- `mini_milestone_3/screenshots/sample_3_video_speech_object/01_preview_mosaic.png`
- `mini_milestone_3/screenshots/sample_3_video_speech_object/02_transcript_card.png`

---

## 2 · Folder layout

```
mini_milestone_3/
├── README.md
├── SETUP_WALKTHROUGH.md
├── ISSUES_AND_RESOLUTIONS.md
├── VALIDATION_REPORT.md
├── validation.py
├── generate_screenshots.py
├── results/
│   ├── sample_1_video_response.json
│   ├── sample_2_video_speech_response.json
│   └── sample_3_video_speech_object_response.json
└── screenshots/
    ├── sample_1_video/
    ├── sample_2_video_speech/
    └── sample_3_video_speech_object/
```
