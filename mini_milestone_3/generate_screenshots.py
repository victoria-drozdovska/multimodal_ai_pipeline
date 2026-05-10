"""
mini_milestone_3/generate_screenshots.py
-----------------------------------------
Build report-ready PNG screenshots from an /api/orchestrate response.

Three screenshots are produced per sample:
  00_summary_card.png    - summary metrics dashboard
  01_preview_mosaic.png  - grid of annotated preview frames
  02_transcript_card.png - full transcript with timestamps
"""

from __future__ import annotations

import base64
import io
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


# Theme
BG       = "#0b0d10"
PANEL    = "#14171c"
PANEL_2  = "#1c2026"
BORDER   = "#262b33"
FG       = "#e8eaed"
MUTED    = "#8b95a3"
ACCENT   = "#7ee8a2"   # green
ACCENT_2 = "#ffb347"   # orange — objects
ACCENT_3 = "#a0c4ff"   # blue — faces
DANGER   = "#ff6b6b"


def _style_axes(ax, hide=True):
    ax.set_facecolor(BG)
    if hide:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def _decode_data_url(url: str) -> Image.Image | None:
    if not url or not url.startswith("data:image"):
        return None
    b64 = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# Summary card
def build_summary_card(response: dict, out_path: Path,
                       title: str = "Multimodal Analysis — Predicted Results") -> Path:
    """Render a one-page summary screenshot."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s    = response.get("summary", {})
    meta = response.get("video_metadata", {})
    models = response.get("models_used", {})
    transcript = response.get("transcription", {})

    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    fig.suptitle("")

    # Header
    ax_h = fig.add_axes([0.03, 0.83, 0.94, 0.14])
    _style_axes(ax_h)
    ax_h.text(0.0, 0.78, title, fontsize=18, fontweight="bold", color=ACCENT, va="top")
    sub = (f"{meta.get('filename','—')}   ·   "
           f"{meta.get('width','?')}×{meta.get('height','?')}   ·   "
           f"{meta.get('duration_sec','?')}s   ·   "
           f"{meta.get('fps','?'):.1f} fps   ·   "
           f"audio: {'yes' if meta.get('has_audio') else 'no'}")
    ax_h.text(0.0, 0.30, sub, fontsize=10, color=MUTED, va="top", family="monospace")

    # KPI strip — six cards
    kpis = [
        ("Frames",   s.get("frames_analyzed", 0),         f"of {meta.get('total_frames', 0)}", ACCENT),
        ("Faces",    s.get("total_faces_detected", 0),    f"in {s.get('frames_with_faces',0)} fr", ACCENT_3),
        ("Objects",  s.get("total_objects_detected", 0),  f"in {s.get('frames_with_objects',0)} fr", ACCENT_2),
        ("Classes",  len(s.get("unique_object_classes", [])),
                                                          ", ".join(s.get("unique_object_classes", [])) or "—", FG),
        ("Tracks",   s.get("unique_object_tracks", 0),    "object track IDs", ACCENT),
        ("Words",    s.get("transcript_word_count", 0),   f"lang: {s.get('transcript_language','—')}", ACCENT_3),
    ]
    n = len(kpis)
    pad = 0.012
    width = (0.94 - pad * (n - 1)) / n
    for i, (label, value, sub_, color) in enumerate(kpis):
        x0 = 0.03 + i * (width + pad)
        ax = fig.add_axes([x0, 0.55, width, 0.24])
        _style_axes(ax)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=PANEL, edgecolor=BORDER, linewidth=1,
                                    transform=ax.transAxes))
        ax.text(0.5, 0.78, label, fontsize=9, color=MUTED,
                ha="center", va="center", family="monospace")
        ax.text(0.5, 0.50, str(value), fontsize=22, color=color,
                fontweight="bold", ha="center", va="center")
        ax.text(0.5, 0.22, str(sub_)[:36], fontsize=8, color=MUTED,
                ha="center", va="center", family="monospace")

    # Models row
    ax_m = fig.add_axes([0.03, 0.45, 0.94, 0.07])
    _style_axes(ax_m)
    ax_m.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04",
                                  facecolor=PANEL, edgecolor=BORDER, linewidth=1,
                                  transform=ax_m.transAxes))
    model_lines = []
    icons = {"face_detection": "[face]", "object_detection": "[obj]",
             "speech_recognition": "[speech]"}
    for k in ("face_detection", "object_detection", "speech_recognition"):
        v = models.get(k, "—")
        model_lines.append(f"{icons.get(k,'•')} {v}")
    ax_m.text(0.02, 0.5, "    ".join(model_lines), fontsize=10, color=FG,
              va="center", family="monospace")

    # Object class bar chart
    counts = s.get("object_class_counts", {}) or {}
    ax_b = fig.add_axes([0.03, 0.18, 0.45, 0.22])
    ax_b.set_facecolor(PANEL)
    if counts:
        items = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        labels  = [k for k, _ in items]
        values  = [v for _, v in items]
        bars = ax_b.barh(labels, values, color=ACCENT_2, edgecolor=BORDER)
        for bar, v in zip(bars, values):
            ax_b.text(v + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                      str(v), va="center", color=FG, fontsize=9, family="monospace")
        ax_b.invert_yaxis()
        ax_b.tick_params(colors=MUTED, labelsize=9)
        for s_ in ax_b.spines.values():
            s_.set_color(BORDER)
        ax_b.set_title("Object class counts", color=FG, fontsize=10,
                       loc="left", pad=8)
        ax_b.set_xlabel("")
    else:
        _style_axes(ax_b)
        ax_b.text(0.5, 0.5, "no objects detected",
                  color=MUTED, ha="center", va="center",
                  family="monospace", fontsize=10)
        ax_b.set_title("Object class counts", color=FG, fontsize=10,
                       loc="left", pad=8)

    # Transcript preview
    ax_t = fig.add_axes([0.51, 0.05, 0.46, 0.36])
    _style_axes(ax_t)
    ax_t.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                  facecolor=PANEL, edgecolor=BORDER, linewidth=1,
                                  transform=ax_t.transAxes))
    full_text = transcript.get("full_text") or ""
    if full_text:
        ax_t.text(0.04, 0.92, "Transcript (preview)", fontsize=10,
                  color=ACCENT, fontweight="bold", va="top")
        wrapped = "\n".join(textwrap.wrap(full_text[:520], width=58))
        ax_t.text(0.04, 0.78, wrapped, fontsize=9, color=FG,
                  va="top", family="serif", linespacing=1.4)
        n_seg = len(transcript.get("segments", []))
        ax_t.text(0.04, 0.06,
                  f"{n_seg} segments  ·  {transcript.get('language','—')}  ·  "
                  f"model: {transcript.get('model','—')}",
                  fontsize=8, color=MUTED, family="monospace", va="bottom")
    else:
        ax_t.text(0.04, 0.92, "Transcript", fontsize=10,
                  color=ACCENT, fontweight="bold", va="top")
        ax_t.text(0.04, 0.50, "No audio track / speech disabled.",
                  fontsize=10, color=MUTED, va="center", family="monospace")

    fig.savefig(out_path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Preview mosaic
def build_preview_mosaic(response: dict, out_path: Path) -> Path:
    """Grid of the orchestrator's annotated preview frames."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pfs = response.get("preview_frames", []) or []
    if not pfs:
        # Empty placeholder
        fig = plt.figure(figsize=(8, 3), facecolor=BG)
        ax = fig.add_subplot(111); _style_axes(ax)
        ax.text(0.5, 0.5, "no preview frames available",
                color=MUTED, ha="center", va="center",
                family="monospace", fontsize=11)
        fig.savefig(out_path, dpi=140, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return out_path

    n = len(pfs)
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5.2 * cols, 4.0 * rows + 0.6), facecolor=BG)
    fig.suptitle("Annotated preview frames (faces · objects · speech subtitle)",
                 fontsize=12, color=ACCENT, fontweight="bold", y=0.99)

    for i, pf in enumerate(pfs):
        ax = fig.add_subplot(rows, cols, i + 1)
        _style_axes(ax)
        img = _decode_data_url(pf.get("image_data_url", ""))
        if img is not None:
            ax.imshow(img)
        title = (
            f"t={pf.get('t_sec',0):.2f}s  ·  "
            f"{pf.get('num_faces',0)} face(s)  ·  "
            f"{pf.get('num_objects',0)} obj"
        )
        ax.set_title(title, fontsize=9, color=FG, family="monospace", pad=6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Transcript card
def build_transcript_card(response: dict, out_path: Path) -> Path:
    """Render the full transcript + segment timestamps as an image."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transcript = response.get("transcription", {}) or {}
    segments = transcript.get("segments", []) or []

    # Auto-size based on number of segments
    seg_lines = max(1, len(segments))
    height = max(4.5, 1.8 + 0.32 * seg_lines + 1.2)
    fig = plt.figure(figsize=(12, height), facecolor=BG)

    # Header
    ax_h = fig.add_axes([0.03, 1 - 0.14 * (4.5 / height),
                          0.94, 0.12 * (4.5 / height)])
    _style_axes(ax_h)
    ax_h.text(0.0, 0.7, "Speech transcription (Whisper)",
              fontsize=15, fontweight="bold", color=ACCENT, va="center")
    info = (f"language: {transcript.get('language','—')}   ·   "
            f"{len(segments)} segments   ·   "
            f"{transcript.get('word_count','?')} words   ·   "
            f"model: {transcript.get('model','—')}")
    ax_h.text(0.0, 0.20, info, fontsize=9, color=MUTED,
              family="monospace", va="center")

    # Full text panel
    full = transcript.get("full_text", "")
    panel_top = 1 - 0.18 * (4.5 / height)
    full_h = 0.22
    ax_full = fig.add_axes([0.03, panel_top - full_h, 0.94, full_h])
    _style_axes(ax_full)
    ax_full.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                                     facecolor=PANEL, edgecolor=BORDER, linewidth=1,
                                     transform=ax_full.transAxes))
    if full:
        wrapped = "\n".join(textwrap.wrap(full, width=120))
        ax_full.text(0.015, 0.92, "Full text", fontsize=9,
                     color=ACCENT, family="monospace", va="top")
        ax_full.text(0.015, 0.74, wrapped, fontsize=10, color=FG,
                     va="top", linespacing=1.45, family="serif")
    else:
        ax_full.text(0.5, 0.5, "no speech transcribed",
                     fontsize=11, color=MUTED, ha="center", va="center",
                     family="monospace")

    # Segments table
    seg_h = panel_top - full_h - 0.04
    ax_seg = fig.add_axes([0.03, 0.04, 0.94, seg_h])
    _style_axes(ax_seg)
    ax_seg.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor=PANEL, edgecolor=BORDER, linewidth=1,
                                    transform=ax_seg.transAxes))
    ax_seg.text(0.015, 0.97, "Segments", fontsize=9,
                color=ACCENT, family="monospace", va="top")

    if segments:
        line_h = 0.92 / max(len(segments), 1)
        for i, seg in enumerate(segments):
            y = 0.92 - (i + 0.6) * line_h
            ts = f"[{seg.get('start_sec',0):6.2f}s — {seg.get('end_sec',0):6.2f}s]"
            ax_seg.text(0.020, y, ts, fontsize=8.5, color=ACCENT_3,
                        family="monospace", va="center")
            txt = (seg.get("text") or "").strip()
            if len(txt) > 110:
                txt = txt[:107] + "…"
            ax_seg.text(0.180, y, txt, fontsize=9, color=FG,
                        family="serif", va="center")
    else:
        ax_seg.text(0.5, 0.5, "(no segments)",
                    fontsize=10, color=MUTED, ha="center", va="center",
                    family="monospace")

    fig.savefig(out_path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path


# CLI: build cards directly from a saved JSON response
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(
        description="Build report-ready PNG cards from an /api/orchestrate response JSON.")
    parser.add_argument("response_json", help="Path to a saved /api/orchestrate response")
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--title", default="Multimodal Analysis", help="Card title")
    args = parser.parse_args()

    with open(args.response_json) as f:
        resp = json.load(f)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(build_summary_card(resp, out / "00_summary_card.png", title=args.title))
    print(build_preview_mosaic(resp, out / "01_preview_mosaic.png"))
    print(build_transcript_card(resp, out / "02_transcript_card.png"))
