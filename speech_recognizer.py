"""Speech recognition wrapper around OpenAI Whisper."""
from typing import Dict, List
import os

try:
    import whisper
except Exception:
    whisper = None


class SpeechRecognizer:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        if whisper is None:
            raise ImportError("openai-whisper is required for SpeechRecognizer")
        self._model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, object]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        res = self._model.transcribe(audio_path)
        segments = []
        for s in res.get("segments", []):
            segments.append({
                "start_sec": float(s.get("start", 0.0)),
                "end_sec": float(s.get("end", 0.0)),
                "text": s.get("text", "").strip(),
            })
        full = res.get("text", "")
        words = len(full.split())
        return {
            "model": f"whisper_{self.model_size}",
            "full_text": full,
            "language": res.get("language", "N/A"),
            "word_count": words,
            "segment_count": len(segments),
            "segments": segments,
        }
