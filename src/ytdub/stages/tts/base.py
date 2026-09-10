"""Synthesize stage: fragment merging, per-clip cached synthesis, progress reporting.

Clips are cached by a hash of everything that determines them (text, language,
reference clip content, TTS model and parameters), so a crash mid-language resumes
where it stopped and editing one line of a review SRT only resynthesizes that line.
"""

from __future__ import annotations

import re
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


# A CUDA context poisoned by a device-side assert (or any other sticky CUDA fault) fails
# every subsequent CUDA call in the process. It is not a per-line problem: the model can
# never produce audio again, so continuing turns one bad line into every remaining line of
# every remaining language, each logged as if it were independently broken. "CUDA out of
# memory" is deliberately NOT matched — that one is recoverable and keeps the context.
_CUDA_LOST = re.compile(r"CUDA error|device-side assert|illegal memory access",
                        re.IGNORECASE)


class CudaContextLost(RuntimeError):
    """The GPU is unusable for the rest of this process. Aborts the job, keeps the cache."""


def is_cuda_lost(exc: BaseException) -> bool:
    """True if ``exc`` (or anything it was raised from) is a sticky CUDA fault."""
    seen = 0
    while exc is not None and seen < 10:
        if _CUDA_LOST.search(str(exc)):
            return True
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False


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


def is_fragment(seg: Segment, min_chars: int) -> bool:
    """True if this line is too short to hand to the synthesizer on its own.

    Two ways to qualify: fewer than ``min_chars`` spoken characters, or a single word.
    The second is the one that matters. Chatterbox's alignment analyzer has two ways to
    die on a one-word line — a host-side ``IndexError: max(): Expected reduction dim 1 to
    have non-zero size`` in its pooling, and a device-side assert inside a CUDA kernel,
    which poisons the context for the whole process and costs every line after it.
    ``Jak?``, ``Would``, ``Tanguy``, ``Okay.``, ``No!`` are all one word; a character
    threshold of three only ever caught the shortest of them.
    """
    text = seg.speech_text.strip()
    return spoken_chars(text) < min_chars or len(text.split()) <= 1


def merge_short_fragments(segments: list[Segment], *, min_chars: int = 3,
                          max_gap: float = 1.5, stuck_gap: float = 3.0
                          ) -> tuple[list[Segment], list[int]]:
    """Merge fragment lines into a neighbour; see :func:`is_fragment` for what counts.

    Merging preserves the words, adds no stutter and also frees a little timeline, which
    is strictly better than the alternatives: a crash, or a line padded with punctuation
    that does not help.

    Three rounds, in order of preference:

    1. Into an adjacent line of the same speaker within ``max_gap`` (preferring the
       previous one), so speech order is never changed.
    2. A *one-word* line with nobody that near goes to the nearest line of its own speaker
       within ``stuck_gap``. Said a couple of seconds off, in the right voice, beats the
       two things that otherwise happen: the synthesizer dies on it, or it comes back as a
       six-second clip for a two-letter word and wrecks the timing around it.
    3. A one-word line with no line of its own speaker anywhere near it is **not spoken**.
       Its index comes back in the second element, and the caller reports it as a lost
       line. Nothing is gained by handing it to the model alone: that is the case that
       crashed, or looped, every time.

    Anything else too short to be safe is left in place and synthesized alone, as before.
    Returns ``(segments, unspoken_indices)``.
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

    def absorb(i: int, target: int) -> None:
        """Merge ``segs[i]`` into ``segs[target]``, keeping speech order."""
        if target < i:
            segs[target:i + 1] = [merged(segs[target], segs[i])]
        else:
            segs[i:target + 1] = [merged(segs[i], segs[target])]

    stuck: set[int] = set()
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(segs):
            if not is_fragment(seg, min_chars) or seg.sources[0] in stuck:
                continue
            if i > 0 and joinable(segs[i - 1], seg):
                absorb(i, i - 1)
            elif i + 1 < len(segs) and joinable(seg, segs[i + 1]):
                absorb(i, i + 1)
            else:
                stuck.add(seg.sources[0])
                continue
            changed = True
            break
    # Still stuck, and a single word: a line the model cannot say on its own. Speak it with
    # the nearest line of the same speaker, however far away that is. Being said slightly
    # early or late is a small fault; the alternatives are a crash inside the synthesizer,
    # a multi-second clip for a two-letter word, or a word that is never spoken at all.
    unspoken: set[int] = set()
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(segs):
            if seg.sources[0] not in stuck or seg.sources[0] in unspoken:
                continue
            if len(seg.speech_text.split()) > 1:
                continue  # a phrase is not fatal on its own; only a lone word is
            others = [j for j, other in enumerate(segs)
                      if j != i and other.speaker == seg.speaker]
            if not others:
                continue
            j = min(others, key=lambda k: _time_gap(seg, segs[k]))
            if _time_gap(seg, segs[j]) > stuck_gap:
                # Nowhere near a line of its own speaker, and this is one word: do not
                # speak it. See rule 3 in the docstring.
                unspoken.add(seg.sources[0])
                continue
            first, second = (i, j) if i < j else (j, i)
            joined = merged(segs[first], segs[second])
            # The words are spoken in the *target's* window, whatever the fragment's own
            # was. A span stretched from one to the other would block every line between
            # them from being placed anywhere near where it was said.
            segs[j] = replace(joined, start=segs[j].start, end=segs[j].end)
            del segs[i]
            stuck.discard(seg.sources[0])
            log.info(f"one-word line {seg.index} had no same-speaker neighbour to merge "
                     f"into; speaking it with line {segs[j].index} of the same speaker "
                     "instead (a word said early beats one not said)")
            changed = True
            break
    left_alone = sorted(stuck - unspoken)
    if left_alone:
        # Short, but not a lone word: a phrase the model can say on its own.
        log.warning(f"{len(left_alone)} short fragment(s) have no same-speaker neighbour "
                    f"to merge into (lines {left_alone}); synthesizing them alone")
    if unspoken:
        segs = [seg for seg in segs if seg.sources[0] not in unspoken]
    for i, seg in enumerate(segs):
        seg.index = i
    for seg in segs:
        if len(seg.sources) > 1:
            log.info(f"merged short fragment(s) into line {seg.index}: {seg.speech_text!r}")
    if unspoken:
        log.error(f"{len(unspoken)} one-word line(s) have no line of the same speaker "
                  f"within {stuck_gap:.1f}s and are not spoken: {sorted(unspoken)}. A lone "
                  "word is what makes the synthesizer crash or loop; merge it with a "
                  "neighbour in the review SRT to get it into the dub.")
    return segs, sorted(unspoken)


def _time_gap(a: Segment, b: Segment) -> float:
    """Seconds of silence between two segments; 0 if they touch or overlap."""
    return max(b.start - a.end, a.start - b.end, 0.0)


def spoken_text(text: str, language: str, *, expand_numbers: bool,
                protected: list[str] | None = None) -> tuple[str, int]:
    """``(text to synthesize, how many numbers were expanded)``.

    The digit-to-words transform (``stages/numbers.py``) belongs here, at the TTS
    boundary, and not in translation: the review SRT and the subtitle track keep the
    digits, because "7.8" is quicker to scan and reads the way the number is written,
    while the synthesizer gets "siedem przecinek osiem" — digits are unreliable to
    pronounce (wrong grammatical case, read as a date, or skipped outright) and none of
    that is visible in the SRT. Keeping it here also puts the changed text into the clip
    cache key, so toggling expansion resynthesizes exactly the lines it changes and
    nothing else.
    """
    if not expand_numbers:
        return text, 0
    from ytdub.stages.numbers import expand_text

    spoken, changes = expand_text(text, language, protected=protected)
    return spoken, len(changes)


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
    expand_numbers: bool = True,
    protected: list[str] | None = None,
) -> tuple[dict[int, Path], list[int]]:
    """Synthesize (or reuse) a clip per segment. Returns ``(index -> clip, failed)``.

    One failed line never aborts the language; its full traceback is logged and the
    index returned in ``failed`` for the report. ``reuse=False`` (``--force``)
    resynthesizes clips that already exist.

    ``expand_numbers`` applies the digit-to-words transform to what is sent to the
    synthesizer (never to ``seg.speech_text``, which is what the SRT is written from).
    The expanded text is what the clip is keyed on, so a line whose numbers changed is
    synthesized again and a line whose numbers did not is still reused.
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    default = next(iter(refs))
    clips: dict[int, Path] = {}
    failed: list[int] = []
    todo = []
    expanded: dict[int, tuple[str, int]] = {}
    for seg in segments:
        spk = seg.speaker if seg.speaker in refs else default
        to_speak, count = spoken_text(seg.speech_text, language,
                                      expand_numbers=expand_numbers, protected=protected)
        expanded[seg.index] = (to_speak, count)
        key = clip_key(to_speak, language, refs[spk], ref_hashes[spk], tts, seed)
        path = clip_dir / f"{key}.wav"
        if reuse and path.exists():
            clips[seg.index] = path
        else:
            todo.append((seg, spk, path))
    if clips:
        log.info(f"[{language}] {len(clips)}/{len(segments)} clips reused from cache")
    # Logged for every affected line, reused or not: on a resumed run the interesting
    # question is what the synthesizer was *given*, and "it came from the cache" is not
    # an answer. Every line's text also appears in the clip cache key, so this is a
    # record of what the audio actually says.
    for index in sorted(expanded):
        spoken_line, count = expanded[index]
        if count:
            log.debug(f"[{language}] line {index}: speaking {spoken_line!r} "
                      f"({count} number(s) written out; SRT text: "
                      f"{segments[index].speech_text!r})")

    started = time.monotonic()
    for n, (seg, spk, path) in enumerate(todo, start=1):
        text, _ = expanded[seg.index]
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
        except Exception as exc:
            if is_cuda_lost(exc):
                # Every later call fails too, so the loop is pointless: report the one
                # real failure with its traceback and let the pipeline stop the job.
                log.exception(f"[{language}] line {seg.index}: the CUDA context is dead; "
                              "giving up on this language (clips already written stay cached)")
                raise CudaContextLost(
                    f"[{language}] line {seg.index}: {type(exc).__name__}: {exc}") from exc
            log.exception(f"[{language}] TTS failed for line {seg.index} ({text[:60]!r})")
            failed.append(seg.index)
        elapsed = time.monotonic() - started
        eta = elapsed / n * (len(todo) - n)
        log.info(f"[{language}] synthesized line {n}/{len(todo)} "
                 f"({100 * n / len(todo):.0f}%) | elapsed {elapsed / 60:.1f}m | "
                 f"eta {eta / 60:.1f}m")
    spoken = [i for i, (_, count) in expanded.items() if count]
    if spoken:
        log.info(f"[{language}] {len(spoken)} line(s) had digits written out for speech "
                 f"({sum(expanded[i][1] for i in spoken)} number(s)); the SRT keeps the "
                 f"digits. Lines: {sorted(spoken)}")
    return clips, failed
