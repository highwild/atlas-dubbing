"""Voice reference selection: one clean clip per speaker, chosen deliberately.

The cloned voice is only as good as its reference. The reference implementation
concatenated whatever came first; here we score candidate regions:

* contiguous speech from that speaker (consecutive segments with short gaps),
* high Whisper confidence,
* no other speaker within ``crosstalk_margin`` seconds (overlap/crosstalk risk),
* close to ``target_seconds`` long (Chatterbox conditions on ~10 s).

The choice is logged with its time range so a bad clone can be traced to its source.
A user-supplied clip per speaker always wins; reusing a known-good clip across videos
keeps a recurring presenter consistent between uploads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ytdub.audio import read_mono, write_wav
from ytdub.logging import stage_logger
from ytdub.models import Segment, SpeakerRef

log = stage_logger("refs")

_DEFAULT_CONF = 0.7  # for segments without a confidence value


@dataclass
class Region:
    spans: list[tuple[float, float]]  # one span, or several if concatenated
    indices: list[int]
    score: float

    @property
    def duration(self) -> float:
        return sum(e - s for s, e in self.spans)

    def describe(self) -> str:
        where = ", ".join(f"{s:.2f}-{e:.2f}s" for s, e in self.spans)
        return f"segments {self.indices[0]}-{self.indices[-1]} ({where})"


def _clean(seg: Segment, others: list[Segment], margin: float) -> bool:
    return not any(o.start < seg.end + margin and o.end > seg.start - margin for o in others)


def select_region(
    segments: list[Segment],
    speaker: str | None,
    *,
    target_seconds: float = 10.0,
    min_seconds: float = 4.0,
    max_gap: float = 0.5,
    crosstalk_margin: float = 0.3,
) -> Region | None:
    """Best reference region for ``speaker``. Pure; returns ``None`` if they never speak."""
    mine = [s for s in segments if s.speaker == speaker]
    others = [s for s in segments if s.speaker != speaker]
    clean = [s for s in mine if _clean(s, others, crosstalk_margin)] or mine
    if not clean:
        return None

    # Contiguous runs of clean segments (another speaker in between breaks a run,
    # because such segments were excluded above and leave a gap).
    runs: list[list[Segment]] = []
    for seg in sorted(clean, key=lambda s: s.start):
        if runs and seg.start - runs[-1][-1].end <= max_gap and seg.index == runs[-1][-1].index + 1:
            runs[-1].append(seg)
        else:
            runs.append([seg])

    def score(run: list[Segment]) -> float:
        dur = run[-1].end - run[0].start
        conf = float(np.mean([s.confidence if s.confidence is not None else _DEFAULT_CONF
                              for s in run]))
        return min(dur, target_seconds) / target_seconds * conf

    best = max(runs, key=score)
    start, end = best[0].start, best[-1].end
    if end - start >= min_seconds:
        # Trim to about target_seconds, cutting at a segment boundary when possible.
        cut = end
        if end - start > target_seconds * 1.2:
            boundaries = [s.end for s in best
                          if target_seconds * 0.8 <= s.end - start <= target_seconds * 1.2]
            cut = max(boundaries) if boundaries else start + target_seconds
        used = [s.index for s in best if s.start < cut]
        return Region(spans=[(start, cut)], indices=used, score=score(best))

    # No single run is long enough: concatenate the best segments.
    ranked = sorted(clean, key=lambda s: (s.confidence or _DEFAULT_CONF) * s.duration,
                    reverse=True)
    picked: list[Segment] = []
    for seg in ranked:
        picked.append(seg)
        if sum(s.duration for s in picked) >= target_seconds:
            break
    picked.sort(key=lambda s: s.start)
    return Region(spans=[(s.start, s.end) for s in picked], indices=[s.index for s in picked],
                  score=0.0)


def extract_region(audio_path: Path, region: Region, out_path: Path,
                   pad: float = 0.05, joint_silence: float = 0.15) -> float:
    """Cut ``region`` from the source audio into ``out_path``; returns its duration."""
    samples, sr = read_mono(audio_path)
    pieces = []
    for s, e in region.spans:
        a = max(0, int((s - pad) * sr))
        b = min(len(samples), int((e + pad) * sr))
        pieces += [samples[a:b], np.zeros(int(joint_silence * sr), dtype=np.float32)]
    clip = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)
    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if peak > 0:
        clip = clip * (0.7 / peak)  # consistent level for the speaker encoder
    write_wav(out_path, clip, sr)
    return len(clip) / sr


def build_references(
    segments: list[Segment],
    audio_path: Path,
    out_dir: Path,
    *,
    user_refs: dict[str | None, Path],
    target_seconds: float,
    min_seconds: float,
) -> dict[str | None, SpeakerRef]:
    """One :class:`SpeakerRef` per speaker label in ``segments``."""
    refs: dict[str | None, SpeakerRef] = {}
    speakers = sorted({s.speaker for s in segments}, key=lambda x: (x is None, x or ""))
    if None in user_refs and len(speakers) > 1:
        log.warning(f"--ref without a speaker label is ignored with {len(speakers)} speakers; "
                    f"use --ref SPK0=path (labels: {', '.join(map(str, speakers))})")
    unknown = [k for k in user_refs if k is not None and k not in speakers]
    if unknown:
        log.warning(f"--ref for unknown speaker(s) {unknown}; labels are {speakers}")
    for spk in speakers:
        name = spk or "voice"
        if spk in user_refs or (len(speakers) == 1 and None in user_refs):
            path = user_refs.get(spk, user_refs.get(None))
            samples, sr = read_mono(path)
            ref = SpeakerRef(spk, path, len(samples) / sr, "user")
            log.info(f"{name}: using supplied reference {path} ({ref.duration:.1f}s)")
        else:
            region = select_region(segments, spk, target_seconds=target_seconds,
                                   min_seconds=min_seconds)
            if region is None:
                continue
            path = out_dir / f"ref_{name}.wav"
            dur = extract_region(audio_path, region, path)
            ref = SpeakerRef(spk, path, dur, region.describe())
            msg = f"{name}: reference = {region.describe()}, {dur:.1f}s"
            if dur < min_seconds or len(region.spans) > 1:
                log.warning(msg + " (short or stitched; supply --ref for a better clone)")
            else:
                log.info(msg)
        refs[spk] = ref
    return refs
