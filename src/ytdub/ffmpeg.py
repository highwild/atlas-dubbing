"""ffmpeg / ffprobe helpers.

Every call goes through :func:`run`, which raises with the tail of ffmpeg's stderr so a
failure says why. Nothing here assumes the input has a video stream: callers probe
first with :func:`probe` and branch on ``has_video``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _exe(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"{name} not found on PATH. Install ffmpeg (sudo apt install ffmpeg).")
    return exe


def run(args: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run([_exe(args[0]), *args[1:]], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed: {' '.join(args[:8])} ...\n{proc.stderr[-3000:]}")
    return proc


@dataclass
class Probe:
    duration: float
    has_video: bool
    has_audio: bool


def probe(path: Path) -> Probe:
    """Container duration plus which stream types exist."""
    proc = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration:stream=codec_type", str(path),
    ])
    data = json.loads(proc.stdout or "{}")
    types = {s.get("codec_type") for s in data.get("streams", [])}
    try:
        duration = float(data.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    return Probe(duration=duration, has_video="video" in types, has_audio="audio" in types)


def extract_audio(src: Path, dst: Path, *, sample_rate: int, channels: int = 1) -> Path:
    """Decode the first audio stream to a PCM WAV (``-vn``: works for audio-only input too)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.wav")
    run([
        "ffmpeg", "-y", "-nostdin", "-i", str(src), "-map", "0:a:0", "-vn",
        "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(tmp),
    ])
    tmp.replace(dst)
    return dst


def atempo_chain(ratio: float) -> str:
    """``atempo`` accepts 0.5-2.0 per instance; chain for anything outside that."""
    ratio = max(0.25, min(4.0, ratio))
    stages: list[float] = []
    while ratio > 2.0:
        stages.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        stages.append(0.5)
        ratio /= 0.5
    stages.append(ratio)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def stretch_samples(samples: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Pitch-preserving tempo change: ``ratio`` > 1 shortens by that factor.

    Used as the fit stage's :data:`~ytdub.stages.fit.Stretcher`. The fitter forces the
    result to its planned length afterwards, so atempo's few-ms inaccuracy is harmless.
    """
    from ytdub.audio import read_mono, write_wav

    with tempfile.TemporaryDirectory(prefix="ytdub-stretch-") as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
        write_wav(src, samples, sr, subtype="FLOAT")
        run([
            "ffmpeg", "-y", "-nostdin", "-i", str(src), "-filter:a", atempo_chain(ratio),
            "-ar", str(sr), "-c:a", "pcm_f32le", str(dst),
        ])
        out, _ = read_mono(dst)
    return out


@dataclass
class Loudness:
    integrated: float  # LUFS
    true_peak: float  # dBTP
    lra: float  # LU
    threshold: float

    def to_dict(self) -> dict:
        return {"integrated_lufs": self.integrated, "true_peak_dbtp": self.true_peak,
                "lra_lu": self.lra, "threshold": self.threshold}


_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def measure_loudness(path: Path) -> Loudness:
    """EBU R128 measurement via loudnorm's analysis pass."""
    proc = run([
        "ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:a:0",
        "-af", "loudnorm=print_format=json", "-f", "null", "-",
    ])
    match = _LOUDNORM_JSON.search(proc.stderr)
    if not match:
        raise RuntimeError(f"could not parse loudnorm output for {path}")
    data = json.loads(match.group(0))
    return Loudness(
        integrated=float(data["input_i"]), true_peak=float(data["input_tp"]),
        lra=float(data["input_lra"]), threshold=float(data["input_thresh"]),
    )


def normalize_to(src: Path, dst: Path, *, target: Loudness, measured: Loudness,
                 sample_rate: int) -> Path:
    """Two-pass loudnorm of ``src`` towards the *source's* loudness, not a fixed target.

    ``linear=true`` applies a single gain where the true-peak ceiling allows it (no
    dynamic compression). The caller re-enforces the sample count afterwards.
    """
    target_i = max(-70.0, min(-5.0, target.integrated))
    target_tp = max(-9.0, min(-1.0, target.true_peak))
    target_lra = max(1.0, min(50.0, target.lra))
    af = (
        f"loudnorm=I={target_i:.2f}:TP={target_tp:.2f}:LRA={target_lra:.2f}"
        f":measured_I={measured.integrated:.2f}:measured_TP={measured.true_peak:.2f}"
        f":measured_LRA={measured.lra:.2f}:measured_thresh={measured.threshold:.2f}"
        ":linear=true:print_format=summary"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-nostdin", "-i", str(src), "-af", af,
        "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_f32le", str(dst),
    ])
    return dst


def mux_audio(video: Path, audio: Path, out: Path) -> Path:
    """Preview MP4: original video stream copied, dubbed audio as AAC.

    Only ever called after probing confirmed a video stream exists.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    run([
        "ffmpeg", "-y", "-nostdin", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
        "-movflags", "+faststart", str(tmp),
    ])
    tmp.replace(out)
    return out
