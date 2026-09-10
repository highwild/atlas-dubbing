"""Assemble stage: loudness match, exact-length WAV, SRT, optional preview mux.

Order matters: the WAV and SRT are written *before* any muxing, so a mux failure can
never cost the expensive work. Muxing is only attempted when probing found a video
stream (audio-only input is a primary use case, not an edge case).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ytdub.audio import fit_length, num_samples, read_mono, write_wav
from ytdub.ffmpeg import Loudness, measure_loudness, normalize_to
from ytdub.logging import stage_logger

log = stage_logger("assemble")


class DurationError(RuntimeError):
    pass


def finalize_audio(timeline: np.ndarray, *, sr: int, total_samples: int, out_path: Path,
                   work_dir: Path, source_loudness: Loudness | None) -> dict:
    """Loudness-match ``timeline`` to the source and write it at exactly ``total_samples``.

    Returns the loudness measurements for the report. Raises :class:`DurationError`
    if the written file is not exactly the required length.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    raw = write_wav(work_dir / "dub_raw.wav", timeline, sr, subtype="FLOAT")
    info: dict = {"source": source_loudness.to_dict() if source_loudness else None}
    samples = timeline
    if source_loudness is not None and math.isfinite(source_loudness.integrated) \
            and float(np.max(np.abs(timeline), initial=0.0)) > 0:
        measured = measure_loudness(raw)
        info["dub_before"] = measured.to_dict()
        if math.isfinite(measured.integrated):
            norm = normalize_to(raw, work_dir / "dub_norm.wav", target=source_loudness,
                                measured=measured, sample_rate=sr)
            samples, got_sr = read_mono(norm)
            if got_sr != sr:
                raise RuntimeError(f"loudnorm returned {got_sr} Hz, expected {sr}")
    # loudnorm (or anything else) may add or drop a few samples: re-impose exact length.
    samples = fit_length(samples, total_samples)
    write_wav(out_path, samples, sr, subtype="PCM_24")
    frames, got_sr = num_samples(out_path)
    if frames != total_samples or got_sr != sr:
        raise DurationError(f"{out_path.name}: {frames} samples @ {got_sr} Hz, "
                            f"expected exactly {total_samples} @ {sr} Hz")
    if source_loudness is not None and "dub_before" in info:
        info["dub_after"] = measure_loudness(out_path).to_dict()
        log.info(f"loudness: source {source_loudness.integrated:.1f} LUFS, dub "
                 f"{info['dub_before']['integrated_lufs']:.1f} -> "
                 f"{info['dub_after']['integrated_lufs']:.1f} LUFS "
                 f"(true peak {info['dub_after']['true_peak_dbtp']:.1f} dBTP)")
    return info
