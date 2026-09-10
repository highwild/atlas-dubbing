"""Join Whisper's half-sentences back into the sentences they were cut out of.

The transcription splitter flushes on a pause longer than 0.6 s as well as on sentence
punctuation, because it has to: a segment is what diarization assigns a speaker to, and a
segment that spanned a speaker change would be labelled with one person's voice. That
leaves half-sentences everywhere — on hydro.wav **30 of 51 lines do not end a sentence**
("...for engineering and a" / "test track.", "...which we need" / "to react to.").

Nothing downstream wanted half-sentences, and each stage was paying for them separately:

* the translator saw fragments with no meaning to translate and echoed them back in the
  source language ("would", "okay i") or shifted whole lines by one;
* the synthesizer, handed one word or a dangling clause, crashes its alignment analyzer
  (a host-side ``IndexError``, or a device-side assert that poisons the CUDA context) or
  forces EOS and renders the leftover frames as breath and groan noise;
* the fitter compressed lines that were too long for a slot that was only half of theirs.

So they are joined once, here, straight after diarization — where the speaker of every
segment is known, and early enough that the references, the translation, the review file
and the synthesizer all see whole sentences. Joining happens only within one speaker, so
it can never move words into the wrong voice.

Pure and deterministic: no model, no audio, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ytdub.logging import stage_logger
from ytdub.models import Segment

log = stage_logger("sentences")

# Ends a sentence: a terminator, optionally followed by a closing quote or bracket.
_SENTENCE_END = re.compile(r"[.!?…。！？][\"'”’)\]]*$")


def ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))


def is_continuation(previous: Segment, following: Segment) -> bool:
    """True when ``following`` continues ``previous`` rather than starting something new.

    Two signals, either one enough: the previous line has no sentence-ending punctuation
    (the splitter cut it on a pause), or the following line starts lowercase (Whisper
    capitalises sentence starts, so lowercase means it is mid-sentence).
    """
    if not ends_sentence(previous.text):
        return True
    return following.text.strip()[:1].islower()


def _spoken_chars(text: str) -> int:
    from ytdub.stages.tts.base import spoken_chars

    return spoken_chars(text)


def join_fragments(
    segments: list[Segment],
    *,
    max_gap: float = 2.0,
    max_seconds: float = 20.0,
    max_chars: int = 300,
    min_chars: int = 12,
) -> list[Segment]:
    """One segment per sentence, per speaker. Returns a new list, renumbered from zero.

    A join needs all of: the same speaker, a gap no longer than ``max_gap``, and a reason —
    either the boundary is mid-sentence (:func:`is_continuation`) or one side is too short
    to stand alone (``min_chars``). Caps on length and duration stop a monologue with no
    sentence ends from becoming one enormous line.
    """
    if not segments:
        return []

    runs: list[list[Segment]] = [[segments[0]]]
    for seg in segments[1:]:
        run = runs[-1]
        previous = run[-1]
        gap = seg.start - previous.end
        joined_seconds = seg.end - run[0].start
        joined_chars = sum(_spoken_chars(s.text) for s in run) + _spoken_chars(seg.text)
        short = _spoken_chars(previous.text) < min_chars or _spoken_chars(seg.text) < min_chars
        if (seg.speaker == previous.speaker and gap <= max_gap
                and joined_seconds <= max_seconds and joined_chars <= max_chars
                and (is_continuation(previous, seg) or short)):
            run.append(seg)
        else:
            runs.append([seg])

    joined: list[Segment] = []
    for index, run in enumerate(runs):
        if len(run) == 1:
            joined.append(replace(run[0], index=index, sources=run[0].sources or [run[0].index]))
            continue
        parts = [s.text.strip() for s in run]
        joined.append(replace(
            run[0], index=index,
            start=run[0].start, end=run[-1].end,
            # One space between the pieces: Whisper's pieces already end where a pause was,
            # and the next one starts without knowing it is a continuation.
            text=" ".join(p for p in parts if p),
            confidence=min((s.confidence for s in run if s.confidence is not None),
                           default=None),
            sources=[i for s in run for i in (s.sources or [s.index])],
        ))
    if len(joined) < len(segments):
        log.info(f"joined {len(segments) - len(joined)} of {len(segments)} transcribed "
                 f"lines into whole sentences ({len(joined)} left)")
    return joined
