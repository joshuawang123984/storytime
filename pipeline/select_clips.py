import json
from pathlib import Path

def build_clips(
    segments,
    scores,
    min_seconds=20.0,
    max_seconds=60.0,
    max_clips=5,
    min_score=4.0,
    pad_start=0.2,
    pad_end=0.3,
    w_hook=0.5,
    w_mean=0.3,
    w_end=0.2,
):
    if not segments or max_clips <= 0:
        return []

    score_by_id = {s["id"]: s["score"] for s in scores}
    missing = [seg["id"] for seg in segments if seg["id"] not in score_by_id]
    if missing:
        raise ValueError(f"no score for segment ids: {missing}")

    vals = [score_by_id[seg["id"]] for seg in segments]
    video_end = segments[-1]["end"]

    # 1. every run of consecutive segments that fits the length limits is a candidate clip
    candidates = []
    for i in range(len(segments)):
        for j in range(i, len(segments)):
            length = segments[j]["end"] - segments[i]["start"]
            if length > max_seconds:
                break  # runs only get longer as j grows
            if length < min_seconds:
                continue
            run = vals[i : j + 1]
            clip_score = w_hook * vals[i] + w_mean * (sum(run) / len(run)) + w_end * vals[j]
            if clip_score < min_score:
                continue
            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "score": clip_score,
                    "start": max(0.0, segments[i]["start"] - pad_start),
                    "end": min(video_end, segments[j]["end"] + pad_end),
                }
            )

    # 2. greedily keep the best candidates that don't overlap anything already kept
    candidates.sort(key=lambda c: (-c["score"], c["start"]))
    chosen = []
    for c in candidates:
        if len(chosen) == max_clips:
            break
        if any(c["start"] < k["end"] and k["start"] < c["end"] for k in chosen):
            continue
        chosen.append(c)

    # 3. report in video order
    chosen.sort(key=lambda c: c["start"])
    return [
        {
            "id": n,
            "start": round(c["start"], 3),
            "end": round(c["end"], 3),
            "duration": round(c["end"] - c["start"], 3),
            "score": round(c["score"], 2),
            "first_segment": segments[c["i"]]["id"],
            "last_segment": segments[c["j"]]["id"],
            "text": " ".join(seg["text"] for seg in segments[c["i"] : c["j"] + 1]),
        }
        for n, c in enumerate(chosen)
    ]


def select_clips(segments_path: Path, scores_path: Path, out_dir: Path) -> Path:
    out_path = out_dir / "clips.json"
    if out_path.exists():
        return out_path

    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    scores = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    clips = build_clips(segments, scores)
    out_path.write_text(json.dumps(clips, indent=2), encoding="utf-8")
    return out_path