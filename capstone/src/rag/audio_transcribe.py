from pathlib import Path
import subprocess
from faster_whisper import WhisperModel


def mp4_to_wav(mp4_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def transcribe_wav(wav_path: Path, out_txt: Path, model_name: str = "small") -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    segments, info = model.transcribe(str(wav_path), beam_size=5)
    lines = [f"[LANG={info.language}]"]
    for seg in segments:
        lines.append(f"[{seg.start:0.2f}-{seg.end:0.2f}] {seg.text.strip()}")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
