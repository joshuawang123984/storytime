import json
from pathlib import Path

import anthropic

MODEL = "claude-haiku-5-5"   # try other models like "claude-sonnet-5-5" for better judgment
BATCH = 40
OVERLAP = 5

PROMPT = """You are helping pick moments from a storytime YouTube video to turn into short clips.
Rate each numbered segment from 1-10 for how strong it would be as the START or CORE of a short clip.
High scores: a strong hook, tension, an emotional peak, a twist or payoff, and it makes sense without earlier context.
Low scores: filler, sponsor reads, setup that needs earlier context, rambling.
Return ONLY a JSON list like [{{"id": 0, "score": 7, "reason": "short reason"}}], one entry per segment.

Segments:
{segments}"""


def format_batch(batch):
    return "\n".join(f"[{s['id']}] {s['text']}" for s in batch)


def parse_scores(text, expected_ids):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    by_id = {int(item["id"]): item for item in data}
    missing = set(expected_ids) - set(by_id)
    if missing:
        raise ValueError(f"missing ids: {sorted(missing)}")
    return by_id


def score(segments_path: Path, out_dir: Path) -> Path:
    out_path = out_dir / "scores.json"
    if out_path.exists():
        return out_path

    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    client = anthropic.Anthropic() 

    results = {}
    step = BATCH - OVERLAP
    for start in range(0, len(segments), step):
        batch = segments[start : start + BATCH]
        ids = [s["id"] for s in batch]
        for attempt in range(3):
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4000,
                temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(segments=format_batch(batch))}],
            )
            try:
                scores = parse_scores(resp.content[0].text, ids)
                break
            except (ValueError, KeyError, json.JSONDecodeError):
                if attempt == 2:
                    raise
        for i, item in scores.items():
            results.setdefault(i, item) 

    out = [
        {"id": s["id"], "score": results[s["id"]]["score"], "reason": results[s["id"]].get("reason", "")}
        for s in segments
    ]
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path