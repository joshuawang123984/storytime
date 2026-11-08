import json
from types import SimpleNamespace

import pipeline.transcribe as transcribe_mod


def make_fake_model(segments, calls):

    class FakeModel:
        def __init__(self, *args, **kwargs):
            calls["init"] += 1

        def transcribe(self, audio, **kwargs):
            calls["transcribe_kwargs"] = kwargs
            return iter(segments), SimpleNamespace(language="en")

    return FakeModel

def word(start, end, text):
    return SimpleNamespace(start=start, end=end, word=text)

def test_transcribe_writes_word_timestamps(tmp_path, monkeypatch):
    calls = {"init": 0}
    segments = [
        SimpleNamespace(words=[word(0.0, 0.4, " Hello"), word(0.4, 0.9, " world")]),
        SimpleNamespace(words=[word(1.2, 1.6, " again")]),
    ]
    monkeypatch.setattr(transcribe_mod, "WhisperModel", make_fake_model(segments, calls))

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    out = transcribe_mod.transcribe(audio, tmp_path)

    assert out == tmp_path / "transcript.json"
    assert json.loads(out.read_text()) == [
        {"start": 0.0, "end": 0.4, "text": "Hello"},
        {"start": 0.4, "end": 0.9, "text": "world"},
        {"start": 1.2, "end": 1.6, "text": "again"},
    ]
    assert calls["transcribe_kwargs"]["word_timestamps"] is True


def test_transcribe_is_cached(tmp_path, monkeypatch):
    calls = {"init": 0}
    segments = [SimpleNamespace(words=[word(0.0, 0.5, " hi")])]
    monkeypatch.setattr(transcribe_mod, "WhisperModel", make_fake_model(segments, calls))

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    transcribe_mod.transcribe(audio, tmp_path)
    transcribe_mod.transcribe(audio, tmp_path)

    assert calls["init"] == 1 


def test_transcribe_handles_no_speech(tmp_path, monkeypatch):
    calls = {"init": 0}
    monkeypatch.setattr(transcribe_mod, "WhisperModel", make_fake_model([], calls))

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    out = transcribe_mod.transcribe(audio, tmp_path)

    assert json.loads(out.read_text()) == []