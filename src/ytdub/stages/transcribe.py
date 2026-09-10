"""Transcribe stage: faster-whisper with word timestamps, VAD and a confidence filter.

Whisper transcribes whatever it hears, including music and room noise, which then gets
translated and spoken as nonsense. Defences here: Silero VAD (``vad_filter``) drops
non-speech before decoding, and rebuilt segments whose mean word probability is low are
dropped and reported (``dropped`` list) so they can be reviewed rather than vanish.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ytdub.gpu import free_memory
from ytdub.logging import stage_logger
from ytdub.models import Segment

log = stage_logger("transcribe")

_SENTENCE_END = re.compile(r"[.!?…。！？]+[\"'”’)]?\s*$")

Word = tuple  # (start, end, text) or (start, end, text, probability)


def build_sentence_segments(
    words: list[Word],
    *,
    max_chars: int = 140,
    max_duration: float = 8.0,
    max_gap: float = 0.6,
) -> list[Segment]:
    """Group word timestamps into sentence/phrase segments.

    Splits on sentence punctuation, on pauses longer than ``max_gap``, and on length
    caps. ``confidence`` is the mean word probability when words carry one.
    """
    segments: list[Segment] = []
    cur: list[Word] = []

    def flush() -> None:
        if not cur:
            return
        text = "".join(w[2] for w in cur).strip()
        if text:
            probs = [w[3] for w in cur if len(w) > 3 and w[3] is not None]
            segments.append(Segment(
                index=len(segments), start=float(cur[0][0]), end=float(cur[-1][1]), text=text,
                confidence=round(sum(probs) / len(probs), 4) if probs else None,
            ))
        cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        text_so_far = "".join(x[2] for x in cur).strip()
        duration = cur[-1][1] - cur[0][0]
        ends_sentence = bool(_SENTENCE_END.search(w[2]))
        too_long = len(text_so_far) >= max_chars or duration >= max_duration
        pause_ahead = i + 1 < len(words) and (words[i + 1][0] - w[1]) > max_gap
        if ends_sentence or too_long or pause_ahead:
            flush()
    flush()
    return segments


def filter_low_confidence(segments: list[Segment], min_confidence: float):
    """Split into ``(kept, dropped)``; renumbers kept segments contiguously."""
    kept, dropped = [], []
    for seg in segments:
        has_words = any(ch.isalnum() for ch in seg.text)
        if not has_words or (min_confidence > 0 and seg.confidence is not None
                             and seg.confidence < min_confidence):
            dropped.append(seg)
        else:
            kept.append(seg)
    for i, seg in enumerate(kept):
        seg.index = i
    return kept, dropped


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    dropped: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"language": self.language,
                "segments": [s.to_dict() for s in self.segments],
                "dropped": [s.to_dict() for s in self.dropped]}

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        return cls(segments=[Segment.from_dict(s) for s in d["segments"]],
                   language=d["language"],
                   dropped=[Segment.from_dict(s) for s in d["dropped"]])


def transcribe(
    audio_path: Path,
    *,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    beam_size: int = 5,
    vad: bool = True,
    vad_min_silence_ms: int = 500,
    min_confidence: float = 0.45,
) -> Transcript:
    """Transcribe, then unload the model so its VRAM is free for the next stage.

    ``model`` may be a size name or a local CTranslate2 model directory.
    """
    from faster_whisper import WhisperModel

    fw_device = "cuda" if device == "cuda" else "cpu"
    log.info(f"Loading Whisper {model!r} on {fw_device} ({compute_type})")
    whisper = WhisperModel(model, device=fw_device, compute_type=compute_type)
    try:
        raw_segments, info = whisper.transcribe(
            str(audio_path), language=language, beam_size=beam_size, word_timestamps=True,
            vad_filter=vad, vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
            condition_on_previous_text=False,  # stops one hallucination seeding the next
        )
        total = float(getattr(info, "duration", 0.0)) or None
        words: list[Word] = []
        for seg in raw_segments:  # generator: decoding happens here
            for w in seg.words or []:
                if w.word:
                    words.append((float(w.start), float(w.end), w.word, float(w.probability)))
            if total:
                log.debug(f"transcribed {seg.end:.0f}/{total:.0f}s")
    finally:
        del whisper
        free_memory()

    segments, dropped = filter_low_confidence(build_sentence_segments(words), min_confidence)
    detected = info.language or language or "en"
    log.success(f"{len(segments)} segments, language={detected}, "
                f"{len(dropped)} dropped as low-confidence/non-speech")
    for seg in dropped:
        log.info(f"dropped {seg.start:7.2f}-{seg.end:7.2f} conf={seg.confidence}: {seg.text!r}")
    return Transcript(segments=segments, language=detected, dropped=dropped)
