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


def _merge_fold(merges: list[tuple[str, str]], protected: set[str]) -> dict[str, str]:
    """``{label: surviving label}`` for every label folded away, transitively.

    ``merges`` are ``(other, keeper)`` pairs; a chain (SPK3->SPK2, SPK2->SPK1) collapses to
    one target, because the alternative is a segment remapped onto a label that itself
    disappears. A label the caller named explicitly (``--ref SPK3=...``) is never folded
    away: that is a person stating which voice they want, and no similarity score
    outranks it.
    """
    parent: dict[str, str] = {label: label for pair in merges for label in pair}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for other, keeper in merges:
        if other in protected or keeper in protected:
            log.warning(f"not folding {other} into {keeper}: both were named explicitly")
            continue
        parent[find(other)] = find(keeper)
    return {label: find(label) for label in parent if find(label) != label}


def apply_matches(
    refs: dict[str | None, SpeakerRef],
    segments: list[Segment],
    *,
    matched: dict[str, Path],
    merges: list[tuple[str, str]],
    user_refs: dict[str | None, Path],
) -> dict[str, str]:
    """Attach voice-matched clips and fold split labels, in place.

    ``refs`` and ``segments`` are both rewritten, because everything downstream keys off
    the speaker label: the review grouping decides which lines may be merged, the prompt
    is told who is talking, and synthesis picks a voice per label. Folding a label in only
    one of those places leaves a speaker whose lines are grouped as one person but spoken
    as another — so the label is folded everywhere, or not at all.

    Returns the fold that was applied, for the caller to put in its cache keys.
    """
    fold = _merge_fold(merges, {k for k in user_refs if k is not None})
    if fold:
        for seg in segments:
            if seg.speaker in fold:
                seg.speaker = fold[seg.speaker]
    for speaker, clip in matched.items():
        if speaker not in refs:
            continue
        samples, sr = read_mono(clip)
        refs[speaker] = SpeakerRef(speaker, Path(clip).resolve(), len(samples) / sr, "matched")
        log.info(f"{speaker}: reference replaced by the stored clip "
                 f"({refs[speaker].duration:.1f}s)")
    for label in fold:
        if label in refs:
            log.info(f"{label}: folded into {fold[label]} (same voice); its reference is "
                     "no longer used")
            del refs[label]
    return fold


def build_references(
    segments: list[Segment],
    audio_path: Path,
    out_dir: Path,
    *,
    user_refs: dict[str | None, Path],
    target_seconds: float,
    min_seconds: float,
) -> dict[str | None, SpeakerRef]:
    """One :class:`SpeakerRef` per speaker label in ``segments``.

    A clip the caller named explicitly (``--ref SPK0=...``) always wins, because that is a
    person stating what they want. Clips identified *by voice* are applied afterwards, by
    :func:`apply_matches`, because identifying them requires these auto-cut references
    first — comparing a stored clip against a pipeline cut separates the same two speakers
    at 0.99 against 0.66, while comparing it against raw in-file audio puts them at 0.64
    against 0.61 (see :mod:`ytdub.stages.voices`).
    """
    refs: dict[str | None, SpeakerRef] = {}
    supplied = dict(user_refs)
    speakers = sorted({s.speaker for s in segments}, key=lambda x: (x is None, x or ""))
    if None in user_refs and len(speakers) > 1:
        log.warning(f"--ref without a speaker label is ignored with {len(speakers)} speakers; "
                    f"use --ref SPK0=path (labels: {', '.join(map(str, speakers))})")
    unknown = [k for k in supplied if k is not None and k not in speakers]
    if unknown:
        log.warning(f"reference for unknown speaker(s) {unknown}; labels are {speakers}")
    for spk in speakers:
        name = spk or "voice"
        if spk in supplied and (spk is not None or len(speakers) == 1):
            path = supplied.get(spk, supplied.get(None))
            samples, sr = read_mono(path)
            origin = "user"
            ref = SpeakerRef(spk, Path(path), len(samples) / sr, origin)
            log.info(f"{name}: using {origin} reference {path} ({ref.duration:.1f}s)")
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
