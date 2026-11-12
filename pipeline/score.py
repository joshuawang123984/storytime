import json
import os
from functools import lru_cache
from pathlib import Path

BATCH = 20
OVERLAP = 3
N_CTX = 8192
MODEL_PATH_ENV = "STORYTIME_MODEL_PATH"

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "score": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["id", "score", "reason"],
    },
}

PROMPT = """You are helping pick moments from a storytime YouTube video to turn into short clips.
Rate each numbered segment from 1-10 for how strong it would be as the START or CORE of a short clip.
High scores: a strong hook, tension, an emotional peak, a twist or payoff, and it makes sense without earlier context.
Low scores: filler, sponsor reads, setup that needs earlier context, rambling.
Return ONLY a JSON list like [{{"id": 0, "score": 7, "reason": "short reason"}}], one entry per segment.

Segments:
{segments}"""


@lru_cache(maxsize=1)
def get_llm():
    path = os.environ.get(MODEL_PATH_ENV)
    if not path or not Path(path).exists():
        raise RuntimeError(f"Set {MODEL_PATH_ENV} to the path of a .gguf model file")
    from llama_cpp import Llama  

    return Llama(model_path=path, n_ctx=N_CTX, n_gpu_layers=-1, verbose=False)


def format_batch(batch):
    return "\n".join(f"[{s['id']}] {s['text']}" for s in batch)


def parse_scores(text, expected_ids):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    by_id = {}
    for item in data:
        by_id[int(item["id"])] = {
            "score": max(1, min(10, int(item["score"]))),
            "reason": str(item.get("reason", "")),
        }
    missing = set(expected_ids) - set(by_id)
    if missing:
        raise ValueError(f"missing ids: {sorted(missing)}")
    return by_id


def score(segments_path: Path, out_dir: Path) -> Path:
    out_path = out_dir / "scores.json"
    if out_path.exists():
        return out_path

    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    if not segments:
        out_path.write_text("[]", encoding="utf-8")
        return out_path

    llm = get_llm()

    results = {}
    step = BATCH - OVERLAP
    for start in range(0, len(segments), step):
        batch = segments[start : start + BATCH]
        ids = [s["id"] for s in batch]
        for attempt in range(3):
            resp = llm.create_chat_completion(
                messages=[{"role": "user", "content": PROMPT.format(segments=format_batch(batch))}],
                response_format={"type": "json_object", "schema": SCHEMA},
                temperature=0 if attempt == 0 else 0.3,  # a retry at temperature 0 would repeat itself
                max_tokens=1500,
            )
            try:
                scores = parse_scores(resp["choices"][0]["message"]["content"], ids)
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                if attempt == 2:
                    raise
        for i, item in scores.items():
            results.setdefault(i, item)  # keep the first score if overlap rescored it

    out = [
        {"id": s["id"], "score": results[s["id"]]["score"], "reason": results[s["id"]]["reason"]}
        for s in segments
    ]
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path