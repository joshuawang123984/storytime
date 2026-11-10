import json
from pathlib import Path

SENTENCE_END = (".", "?", "!", "…")
CLOSERS = "\"')]”’"
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs."}

def ends_sentence(text: str) -> bool:
    t = text.rstrip(CLOSERS)
    return t.endswith(SENTENCE_END) and t.lower() not in ABBREVIATIONS

def build_segments(words, pause_gap=0.8, min_seconds=1.5, max_seconds=20.0):
    if not words:
        return []

    spans = []
    first = 0
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        too_long = w["end"] - words[first]["start"] >= max_seconds

        if (
            nxt is None
            or ends_sentence(w["text"])
            or nxt["start"] - w["end"] >= pause_gap
            or too_long
        ):
            spans.append((first, i))
            first = i + 1

    merged = []
    for a, b in spans:
        short = words[b]["end"] - words[a]["start"] < min_seconds
        if merged and short:
            prev_a, _ = merged[-1]
            if words[b]["end"] - words[prev_a]["start"] <= max_seconds:
                merged[-1] = (prev_a, b)
                continue
        merged.append((a, b))

    return [
        {
           "id": n,
            "start": words[a]["start"],
            "end": words[b]["end"],
            "text": " ".join(w["text"] for w in words[a : b + 1]),
            "first_word": a,
            "last_word": b, 
        }
        for n, (a, b) in enumerate(merged)
    ]

def segment(transcript_path: Path, out_dir: Path) -> Path:
    out_path = out_dir / "segments.json"
    if out_path.exists():
        return out_path

    words = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    segments = build_segments(words)
    out_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    return out_path
