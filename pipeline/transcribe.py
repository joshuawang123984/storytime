import json
from pathlib import Path

from faster_whisper import WhisperModel

def transcribe(audio_path: Path, out_dir: Path, model_size: str = "small") -> Path:
    out_path = out_dir / "transcript.json"
    if out_path.exists():
        return out_path

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), word_timestamps=True)

    words = []
    for segment in segments:
        for w in segment.words:
            words.append({
                "start": w.start,
                "end": w.end,
                "text": w.word.strip(),
            })

    out_path.write_text(json.dumps(words, indent=2), encoding="utf-8")
    return out_path