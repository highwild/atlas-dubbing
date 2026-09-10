"""Sample-level audio helpers (numpy + soundfile, no ffmpeg).

Everything that touches the length of the output lives here or in ``stages/fit.py`` and
works in integer sample counts, never in float seconds, so the total duration of the
dub can be made exact rather than approximately right.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read any soundfile-supported file as float32 mono ``(samples, sample_rate)``."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


def write_wav(path: Path, samples: np.ndarray, sr: int, subtype: str = "PCM_16") -> Path:
    """Write mono float samples to a WAV atomically (tmp file + rename)."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.wav")
    sf.write(str(tmp), np.clip(samples, -1.0, 1.0), sr, subtype=subtype)
    tmp.replace(path)
    return path


def num_samples(path: Path) -> tuple[int, int]:
    """``(frames, sample_rate)`` of a sound file without decoding it."""
    import soundfile as sf

    info = sf.info(str(path))
    return int(info.frames), int(info.samplerate)


def seconds_to_samples(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))


def trim_silence(
    samples: np.ndarray, sr: int, *, threshold_db: float = -40.0, frame_ms: float = 10.0,
    keep_ms: float = 20.0,
) -> np.ndarray:
    """Strip leading/trailing silence, keeping ``keep_ms`` of margin on each side.

    TTS clips usually carry 100-300 ms of dead air at each end. Leaving it in both
    delays the onset against the picture and wastes timeline slack the fitter needs.
    A clip that is silent throughout is returned unchanged (never emptied).
    """
    if len(samples) == 0:
        return samples
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = int(np.ceil(len(samples) / frame))
    padded = np.zeros(n_frames * frame, dtype=np.float32)
    padded[: len(samples)] = samples
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame) ** 2, axis=1) + 1e-12)
    peak = float(np.max(np.abs(samples))) or 1.0
    # Threshold relative to the clip's own peak, so quiet-but-valid TTS output survives.
    loud = np.nonzero(20 * np.log10(rms / peak) > threshold_db)[0]
    if len(loud) == 0:
        return samples
    keep = int(sr * keep_ms / 1000)
    start = max(0, loud[0] * frame - keep)
    end = min(len(samples), (loud[-1] + 1) * frame + keep)
    return samples[start:end]


def fade_edges(samples: np.ndarray, sr: int, fade_ms: float = 5.0) -> np.ndarray:
    """Short linear fade in/out so clips placed on the timeline never click."""
    n = min(len(samples) // 2, int(sr * fade_ms / 1000))
    if n <= 0:
        return samples
    out = samples.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited polyphase resampling (exact for integer ratios like 24k -> 48k)."""
    if sr_in == sr_out or len(samples) == 0:
        return samples.astype(np.float32)
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(samples, sr_out // g, sr_in // g).astype(np.float32)


def fit_length(samples: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or truncate to exactly ``n`` samples."""
    if len(samples) == n:
        return samples
    if len(samples) > n:
        return samples[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(samples)] = samples
    return out
