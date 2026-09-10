"""Synthesize stage: fragment merging, per-clip cached synthesis, progress reporting.

Clips are cached by a hash of everything that determines them (text, language,
reference clip content, TTS model and parameters), so a crash mid-language resumes
where it stopped and editing one line of a review SRT only resynthesizes that line.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from ytdub.audio import read_mono, trim_silence
from ytdub.cache import hash_obj
from ytdub.logging import stage_logger
from ytdub.models import Segment, SpeakerRef
from ytdub.plugins import load_class

log = stage_logger("tts")


BUILTIN = {"chatterbox": "ytdub.stages.tts.chatterbox:ChatterboxBackend"}


class TTSBackend(Protocol):
    """Voice-cloning TTS interface.

    Only Chatterbox ships, but any class with this shape can be dropped in with
    ``YTDUB_TTS_BACKEND=module.path:ClassName`` (or ``--tts``). It is constructed as
    ``cls(settings)``; its module must import cheaply (load the model lazily), because
    the CLI reads ``supported_languages`` from the class before a job starts.
    """

    name: str
    supported_languages: set[str]  # class attribute, ISO-639-1 codes

    def cache_identity(self) -> dict:
        """Model, version and parameters: anything that changes the audio. Part of every
        clip's cache key, so switching backend or upgrading one never reuses old clips."""

    def synthesize(self, text: str, ref: Path, language: str, out_path: Path, seed: int) -> Path:
        """Write a mono WAV of ``text`` in the voice of reference clip ``ref``."""

    def unload(self) -> None:
        """Free VRAM. Called once, after all languages."""


def tts_class(name: str) -> type:
    return load_class(name, BUILTIN, "TTS")


def get_tts(name: str, settings) -> TTSBackend:
    return tts_class(name)(settings)


def spoken_chars(text: str) -> int:
    """Letters, digits and combining marks (so Devanagari vowel signs count)."""
    return sum(1 for c in text if unicodedata.category(c)[0] in "LNM")


def merge_short_fragments(segments: list[Segment], *, min_chars: int = 3,
                          max_gap: float = 1.5) -> tuple[list[Segment], list[int]]:
    """Merge lines with fewer than ``min_chars`` spoken characters into a neighbour.

    Chatterbox's alignment analyzer takes a max over an empty slice on very short text
    (``IndexError: max(): Expected reduction dim 1 to have non-zero size``); padding
    with punctuation does not help. Merging preserves the words, adds no stutter and
    also frees a little timeline.

    Only an *adjacent* segment of the *same* speaker within ``max_gap`` seconds is a
    valid target (preferring the previous one), so speech order is never changed.
    Returns ``(segments, unmergeable_indices)``; unmergeable fragments are kept and
    synthesized alone, and reported if that fails.
    """
    segs = [replace(s, sources=s.sources or [s.index]) for s in segments]

    def joinable(a: Segment, b: Segment) -> bool:
        return a.speaker == b.speaker and b.start - a.end <= max_gap

    def merged(a: Segment, b: Segment) -> Segment:
        confs = [c for c in (a.confidence, b.confidence) if c is not None]
        return replace(
            a, end=max(a.end, b.end), text=f"{a.text} {b.text}".strip(),
            translated=f"{a.speech_text} {b.speech_text}".strip(),
            confidence=min(confs) if confs else None, sources=a.sources + b.sources,
        )

    stuck: set[int] = set()
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(segs):
            if spoken_chars(seg.speech_text) >= min_chars or seg.sources[0] in stuck:
                continue
            if i > 0 and joinable(segs[i - 1], seg):
                segs[i - 1:i + 1] = [merged(segs[i - 1], seg)]
            elif i + 1 < len(segs) and joinable(seg, segs[i + 1]):
                segs[i:i + 2] = [merged(seg, segs[i + 1])]
            else:
                stuck.add(seg.sources[0])
                continue
            changed = True
            break
    for i, seg in enumerate(segs):
        seg.index = i
    for seg in segs:
        if len(seg.sources) > 1:
            log.info(f"merged short fragment(s) into line {seg.index}: {seg.speech_text!r}")
    return segs, sorted(stuck)


def clip_key(text: str, language: str, ref: SpeakerRef, ref_hash: str, tts: TTSBackend,
             seed: int) -> str:
    return hash_obj({"text": text, "lang": language, "ref": ref_hash, "tts": tts.name,
                     "identity": tts.cache_identity(), "seed": seed})[:24]


def _looks_broken(path: Path, text: str) -> str | None:
    """Cheap sanity checks for the two common TTS failures: silence and runaway output."""
    samples, sr = read_mono(path)
    if len(samples) == 0 or float(abs(samples).max()) < 1e-3:
        return "silent"
    speech = trim_silence(samples, sr)
    expected = spoken_chars(text) / 12.0  # generous: slow speech is ~12 chars/s
    if len(speech) / sr > max(4.0, expected * 3.0):
        return f"runaway ({len(speech) / sr:.1f}s for {spoken_chars(text)} chars)"
    return None


def synthesize_all(
    segments: list[Segment],
    tts: TTSBackend,
    *,
    refs: dict[str | None, SpeakerRef],
    ref_hashes: dict[str | None, str],
    language: str,
    clip_dir: Path,
    seed: int,
    reuse: bool = True,
) -> tuple[dict[int, Path], list[int]]:
    """Synthesize (or reuse) a clip per segment. Returns ``(index -> clip, failed)``.

    One failed line never aborts the language; its full traceback is logged and the
    index returned in ``failed`` for the report. ``reuse=False`` (``--force``)
    resynthesizes clips that already exist.
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    default = next(iter(refs))
    clips: dict[int, Path] = {}
    failed: list[int] = []
    todo = []
    for seg in segments:
        spk = seg.speaker if seg.speaker in refs else default
        key = clip_key(seg.speech_text, language, refs[spk], ref_hashes[spk], tts, seed)
        path = clip_dir / f"{key}.wav"
        if reuse and path.exists():
            clips[seg.index] = path
        else:
            todo.append((seg, spk, path))
    if clips:
        log.info(f"[{language}] {len(clips)}/{len(segments)} clips reused from cache")

    started = time.monotonic()
    for n, (seg, spk, path) in enumerate(todo, start=1):
        text = seg.speech_text
        try:
            tmp = path.with_name(path.stem + ".tmp.wav")
            problem = None
            for attempt in range(2):
                tts.synthesize(text, refs[spk].path, language, tmp, seed + attempt)
                problem = _looks_broken(tmp, text)
                if not problem:
                    break
                log.warning(f"[{language}] line {seg.index}: {problem}, retrying with new seed")
            if problem:
                log.warning(f"[{language}] line {seg.index}: still {problem}; keeping it")
            tmp.replace(path)
            clips[seg.index] = path
        except Exception:
            log.exception(f"[{language}] TTS failed for line {seg.index} ({text[:60]!r})")
            failed.append(seg.index)
        elapsed = time.monotonic() - started
        eta = elapsed / n * (len(todo) - n)
        log.info(f"[{language}] synthesized line {n}/{len(todo)} "
                 f"({100 * n / len(todo):.0f}%) | elapsed {elapsed / 60:.1f}m | "
                 f"eta {eta / 60:.1f}m")
    return clips, failed
