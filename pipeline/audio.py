import subprocess
from pathlib import Path

def extract_audio(video_path: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audio.wav"
    if out_path.exists():
        return out_path

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_path)
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path