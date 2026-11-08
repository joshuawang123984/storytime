import json
import shutil
import subprocess

import pytest
from pipeline.audio import extract_audio

pytestmrk = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)

@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-shortest", str(path)],
            check=True, capture_output=True,
    )

    return path

def test_extract_audio_produces_16k_mono_wav(sample_video, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    audio_path = extract_audio(str(sample_video), out_dir)

    assert audio_path.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "json", str(audio_path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1

def test_extract_audio_is_cached(sample_video, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    first = extract_audio(str(sample_video), out_dir)
    mtime = first.stat().st_mtime_ns
    second = extract_audio(str(sample_video), out_dir)

    assert second == first
    assert second.stat().st_mtime_ns == mtime