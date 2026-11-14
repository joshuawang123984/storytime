import json
import re
from types import SimpleNamespace

import pytest

import pipeline.score as score_mod
from pipeline.score import format_batch, parse_scores, score


# ---- helpers -------------------------------------------------------------

def make_segments(n):
    return [{"id": i, "start": float(i), "end": float(i + 1), "text": f"text {i}"} for i in range(n)]


def ids_in(prompt):
    """Segment ids that appear in a prompt, in order (lines like '[12] some text')."""
    return [int(x) for x in re.findall(r"^\[(\d+)\]", prompt, flags=re.M)]


def reply_for(prompt, score_value=5, with_reason=True):
    items = []
    for i in ids_in(prompt):
        item = {"id": i, "score": score_value}
        if with_reason:
            item["reason"] = f"reason {i}"
        items.append(item)
    return json.dumps(items)


def install_fake_llm(monkeypatch, reply_fn):
    """Replace get_llm() in score.py with a fake model.

    reply_fn(prompt, call_number) returns the text the fake model "says".
    Returns the list of recorded create_chat_completion calls.
    """
    calls = []

    def create_chat_completion(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        text = reply_fn(prompt, len(calls))
        return {"choices": [{"message": {"content": text}}]}

    fake_llm = SimpleNamespace(create_chat_completion=create_chat_completion)
    monkeypatch.setattr(score_mod, "get_llm", lambda: fake_llm)
    return calls


def write_segments(tmp_path, n):
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(make_segments(n)), encoding="utf-8")
    return path


def read_scores(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---- format_batch --------------------------------------------------------

def test_format_batch_uses_id_prefix_per_line():
    batch = [{"id": 0, "text": "hello"}, {"id": 7, "text": "world"}]
    assert format_batch(batch) == "[0] hello\n[7] world"


# ---- parse_scores --------------------------------------------------------

def test_parse_scores_reads_plain_json():
    text = '[{"id": 0, "score": 7, "reason": "good"}, {"id": 1, "score": 3}]'
    result = parse_scores(text, [0, 1])
    assert result[0] == {"score": 7, "reason": "good"}
    assert result[1] == {"score": 3, "reason": ""}


def test_parse_scores_strips_json_code_fence():
    text = '```json\n[{"id": 0, "score": 7}]\n```'
    assert parse_scores(text, [0])[0]["score"] == 7


def test_parse_scores_strips_plain_code_fence():
    text = '```\n[{"id": 0, "score": 7}]\n```'
    assert parse_scores(text, [0])[0]["score"] == 7


def test_parse_scores_accepts_string_ids():
    assert parse_scores('[{"id": "4", "score": 6}]', [4])[4]["score"] == 6


def test_parse_scores_clamps_scores_to_1_through_10():
    result = parse_scores('[{"id": 0, "score": 15}, {"id": 1, "score": 0}]', [0, 1])
    assert result[0]["score"] == 10
    assert result[1]["score"] == 1


def test_parse_scores_raises_when_ids_are_missing():
    with pytest.raises(ValueError, match="missing ids"):
        parse_scores('[{"id": 0, "score": 7}]', [0, 1, 2])


def test_parse_scores_raises_on_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        parse_scores("sorry, I can't do that", [0])


# ---- get_llm -------------------------------------------------------------

def test_get_llm_requires_model_path_env_var(monkeypatch):
    score_mod.get_llm.cache_clear()
    monkeypatch.delenv(score_mod.MODEL_PATH_ENV, raising=False)

    with pytest.raises(RuntimeError, match=score_mod.MODEL_PATH_ENV):
        score_mod.get_llm()


def test_get_llm_rejects_a_missing_model_file(tmp_path, monkeypatch):
    score_mod.get_llm.cache_clear()
    monkeypatch.setenv(score_mod.MODEL_PATH_ENV, str(tmp_path / "nope.gguf"))

    with pytest.raises(RuntimeError, match=score_mod.MODEL_PATH_ENV):
        score_mod.get_llm()


# ---- score ---------------------------------------------------

def test_score_writes_scores_json(tmp_path, monkeypatch):
    calls = install_fake_llm(monkeypatch, lambda prompt, n: reply_for(prompt, score_value=8))
    segments_path = write_segments(tmp_path, 3)

    out = score(segments_path, tmp_path)

    assert out == tmp_path / "scores.json"
    assert read_scores(out) == [
        {"id": 0, "score": 8, "reason": "reason 0"},
        {"id": 1, "score": 8, "reason": "reason 1"},
        {"id": 2, "score": 8, "reason": "reason 2"},
    ]
    assert len(calls) == 1


def test_score_asks_for_schema_constrained_json_at_temperature_zero(tmp_path, monkeypatch):
    calls = install_fake_llm(monkeypatch, lambda prompt, n: reply_for(prompt))
    score(write_segments(tmp_path, 2), tmp_path)

    assert calls[0]["response_format"] == {"type": "json_object", "schema": score_mod.SCHEMA}
    assert calls[0]["temperature"] == 0


def test_score_sends_segment_text_to_the_model(tmp_path, monkeypatch):
    calls = install_fake_llm(monkeypatch, lambda prompt, n: reply_for(prompt))
    score(write_segments(tmp_path, 2), tmp_path)

    prompt = calls[0]["messages"][0]["content"]
    assert "[0] text 0" in prompt
    assert "[1] text 1" in prompt


def test_score_defaults_missing_reason_to_empty_string(tmp_path, monkeypatch):
    install_fake_llm(monkeypatch, lambda prompt, n: reply_for(prompt, with_reason=False))

    out = score(write_segments(tmp_path, 2), tmp_path)

    assert [item["reason"] for item in read_scores(out)] == ["", ""]


def test_score_handles_code_fenced_replies(tmp_path, monkeypatch):
    install_fake_llm(monkeypatch, lambda prompt, n: f"```json\n{reply_for(prompt)}\n```")

    out = score(write_segments(tmp_path, 2), tmp_path)

    assert [item["score"] for item in read_scores(out)] == [5, 5]


def test_score_with_no_segments_never_loads_the_model(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("model should not be loaded for empty input")

    monkeypatch.setattr(score_mod, "get_llm", boom)

    out = score(write_segments(tmp_path, 0), tmp_path)

    assert read_scores(out) == []


def test_score_is_cached(tmp_path, monkeypatch):
    calls = install_fake_llm(monkeypatch, lambda prompt, n: reply_for(prompt))
    segments_path = write_segments(tmp_path, 3)

    score(segments_path, tmp_path)
    score(segments_path, tmp_path)

    assert len(calls) == 1  


def test_score_batches_with_overlap_and_keeps_first_score(tmp_path, monkeypatch):
    monkeypatch.setattr(score_mod, "BATCH", 40)
    monkeypatch.setattr(score_mod, "OVERLAP", 5)
    # Each call returns its own call number as the score, so we can tell which call scored what.
    calls = install_fake_llm(monkeypatch, lambda prompt, n: reply_for(prompt, score_value=n))

    out = score(write_segments(tmp_path, 100), tmp_path)

    # BATCH=40, OVERLAP=5 -> step 35 -> batches start at 0, 35, 70
    assert len(calls) == 3
    prompts = [c["messages"][0]["content"] for c in calls]
    assert ids_in(prompts[0]) == list(range(0, 40))
    assert ids_in(prompts[1]) == list(range(35, 75))
    assert ids_in(prompts[2]) == list(range(70, 100))

    scores = {item["id"]: item["score"] for item in read_scores(out)}
    assert len(scores) == 100
    assert scores[0] == 1     
    assert scores[36] == 1     
    assert scores[40] == 2   
    assert scores[72] == 2    
    assert scores[99] == 3  


def test_score_uses_default_batch_settings(tmp_path, monkeypatch):
    calls = install_fake_llm(monkeypatch, lambda prompt, n: reply_for(prompt))

    out = score(write_segments(tmp_path, 100), tmp_path)

    step = score_mod.BATCH - score_mod.OVERLAP
    assert len(calls) == len(range(0, 100, step))
    assert len(read_scores(out)) == 100


@pytest.mark.parametrize(
    "bad_reply",
    [
        "not json at all",               
        '[{"score": 5}]',                   
        '[{"id": 0}]',                        
        '[{"id": 0, "score": 5}]',         
    ],
)
def test_score_retries_after_a_bad_reply(tmp_path, monkeypatch, bad_reply):
    def reply(prompt, n):
        return bad_reply if n == 1 else reply_for(prompt, score_value=9)

    calls = install_fake_llm(monkeypatch, reply)

    out = score(write_segments(tmp_path, 3), tmp_path)

    assert len(calls) == 2
    assert [item["score"] for item in read_scores(out)] == [9, 9, 9]
    assert calls[1]["temperature"] > calls[0]["temperature"]  # retries must not repeat the same sample


def test_score_gives_up_after_three_bad_replies(tmp_path, monkeypatch):
    calls = install_fake_llm(monkeypatch, lambda prompt, n: "garbage")

    with pytest.raises(json.JSONDecodeError):
        score(write_segments(tmp_path, 3), tmp_path)

    assert len(calls) == 3
    assert not (tmp_path / "scores.json").exists()  # a failed run must not leave a cache file