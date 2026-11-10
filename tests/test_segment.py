import json
import pytest
from pipeline.segment import build_segments, ends_sentence, segment

def make_words(tokens, dur=0.8, gaps=None, start=0.0):
    gaps = gaps or {}
    words = []
    t = start
    for i, text in enumerate(tokens):
        words.append({"start": round(t, 3), "end": round(t + dur, 3), "text": text})
        t += dur + gaps.get(i, 0.0)
    return words


# ---- ends_sentence -------------------------------------------------------

@pytest.mark.parametrize("text", ["done.", "really?", "wow!", "so…", "stop.\"", "(end.)", "really?!"])
def test_ends_sentence_true(text):
    assert ends_sentence(text)


@pytest.mark.parametrize("text", ["hello", "and,", "Mr.", "dr.", "St.", "vs."])
def test_ends_sentence_false(text):
    assert not ends_sentence(text)


# ---- build_segments ------------------------------------------------------

def test_empty_input_gives_no_segments():
    assert build_segments([]) == []


def test_single_short_word_still_becomes_a_segment():
    segs = build_segments(make_words(["Hi."], dur=0.5))
    assert len(segs) == 1
    assert segs[0]["text"] == "Hi."


def test_splits_on_sentence_punctuation():
    words = make_words(["Hello", "there.", "How", "are", "you?"])
    segs = build_segments(words)

    assert [s["text"] for s in segs] == ["Hello there.", "How are you?"]


def test_segment_fields_and_indexes():
    words = make_words(["Hello", "there.", "How", "are", "you?"])
    segs = build_segments(words)

    assert segs[0] == {
        "id": 0,
        "start": words[0]["start"],
        "end": words[1]["end"],
        "text": "Hello there.",
        "first_word": 0,
        "last_word": 1,
    }
    assert segs[1]["id"] == 1
    assert segs[1]["start"] == words[2]["start"]
    assert segs[1]["end"] == words[4]["end"]
    assert (segs[1]["first_word"], segs[1]["last_word"]) == (2, 4)


def test_splits_on_long_pause_without_punctuation():
    words = make_words(["and", "then", "we", "left"], gaps={1: 1.0})
    segs = build_segments(words)

    assert [s["text"] for s in segs] == ["and then", "we left"]


def test_small_gap_does_not_split():
    words = make_words(["and", "then", "we", "left"], gaps={1: 0.3})
    segs = build_segments(words)

    assert [s["text"] for s in segs] == ["and then we left"]


def test_splits_when_max_length_is_reached():
    # 30 one-second words with no punctuation and no pauses
    words = make_words([f"w{i}" for i in range(30)], dur=1.0)
    segs = build_segments(words)

    assert len(segs) == 2
    assert segs[0]["end"] == 20.0
    assert segs[1]["start"] == 20.0
    assert segs[1]["end"] == 30.0


def test_abbreviation_does_not_split():
    words = make_words(["Mr.", "Smith", "left."])
    segs = build_segments(words)

    assert [s["text"] for s in segs] == ["Mr. Smith left."]


def test_short_segment_merges_into_previous():
    words = make_words(["I", "went", "home", "alone."])
    words += make_words(["Yeah."], dur=0.5, start=words[-1]["end"])
    segs = build_segments(words)

    assert len(segs) == 1
    assert segs[0]["text"] == "I went home alone. Yeah."
    assert segs[0]["last_word"] == 4


def test_short_segment_is_not_merged_if_it_would_exceed_the_cap():
    words = make_words(["a", "b", "c."], dur=1.0)
    words += make_words(["Ok."], dur=0.5, start=words[-1]["end"])
    segs = build_segments(words, max_seconds=3.2)

    assert [s["text"] for s in segs] == ["a b c.", "Ok."]


# ---- segment (file stage) ------------------------------------------------

def test_segment_writes_segments_json(tmp_path):
    words = make_words(["Hello", "there.", "How", "are", "you?"])
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps(words), encoding="utf-8")

    out = segment(transcript, tmp_path)

    assert out == tmp_path / "segments.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [s["text"] for s in data] == ["Hello there.", "How are you?"]


def test_segment_is_cached(tmp_path):
    first_words = make_words(["Hello", "there.", "How", "are", "you?"])
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps(first_words), encoding="utf-8")

    out = segment(transcript, tmp_path)
    before = out.read_text(encoding="utf-8")

    # change the input; a cached run must ignore it
    transcript.write_text(json.dumps(make_words(["Different", "words."])), encoding="utf-8")
    out_again = segment(transcript, tmp_path)

    assert out_again.read_text(encoding="utf-8") == before