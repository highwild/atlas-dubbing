"""Data model shared across pipeline stages.

Segments are plain dataclasses that round-trip through JSON, because every stage's
output is cached on disk and must survive a crash or reboot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass
class Segment:
    """A timed speech span on the *source* timeline (seconds)."""

    index: int
    start: float
    end: float
    text: str  # source-language text
    translated: str | None = None  # target-language text
    speaker: str | None = None  # diarization label, e.g. "SPK0"; None in single-voice mode
    confidence: float | None = None  # mean word probability from Whisper
    sources: list[int] = field(default_factory=list)  # source indices, if merged

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def speech_text(self) -> str:
        return (self.translated if self.translated is not None else self.text).strip()

    def with_translation(self, translated: str) -> Segment:
        return replace(self, translated=translated)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(**d)


@dataclass
class Source:
    """The acquired input file."""

    path: Path
    basename: str
    sha256: str
    duration: float
    has_video: bool


@dataclass
class SpeakerRef:
    """The voice reference clip chosen (or supplied) for one speaker."""

    speaker: str | None
    path: Path
    duration: float
    origin: str  # "user", or a description of the auto-selected region

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "path": str(self.path), "duration": self.duration,
                "origin": self.origin}

    @classmethod
    def from_dict(cls, d: dict) -> SpeakerRef:
        return cls(speaker=d["speaker"], path=Path(d["path"]), duration=d["duration"],
                   origin=d["origin"])
