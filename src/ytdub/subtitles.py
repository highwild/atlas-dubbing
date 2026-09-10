"""SRT writing and parsing.

Two kinds of SRT are produced: ``<lang>.review.srt`` on source timings right after
translation (for a native speaker to read and fix before any TTS runs), and the final
``<lang>.srt`` on the *fitted* timings, matching the dubbed audio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ytdub.cache import atomic_write_text


@dataclass
class Cue:
    start: float
    end: float
    text: str


def timestamp(seconds: float) -> str:
    millis = int(round(max(0.0, seconds) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt(cues: list[Cue]) -> str:
    blocks = [f"{i}\n{timestamp(c.start)} --> {timestamp(c.end)}\n{c.text.strip()}\n"
              for i, c in enumerate(cues, start=1)]
    return "\n".join(blocks)


def write_srt(cues: list[Cue], path: Path) -> Path:
    atomic_write_text(path, render_srt(cues))
    return path


_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def _seconds(ts: str) -> float:
    m = _TS.fullmatch(ts.strip())
    if not m:
        raise ValueError(f"bad SRT timestamp {ts!r}")
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def parse_srt(text: str) -> list[Cue]:
    """Parse SRT text (tolerates a BOM, CRLF and multi-line cue text)."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.split("\n")]
        if not lines or not lines[0].strip():
            continue
        if "-->" in lines[0]:
            timing, body = lines[0], lines[1:]
        elif len(lines) > 1 and "-->" in lines[1]:
            timing, body = lines[1], lines[2:]
        else:
            raise ValueError(f"SRT block without timing line: {block[:80]!r}")
        start, end = timing.split("-->")
        cues.append(Cue(_seconds(start), _seconds(end.split()[0]),
                        " ".join(ln.strip() for ln in body).strip()))
    return cues
