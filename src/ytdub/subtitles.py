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
from ytdub.logging import stage_logger

log = stage_logger("srt")


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


# --- review SRT -> transcript lines ------------------------------------------------


class ReviewError(ValueError):
    """A reviewed file that cannot be used as written. Raised instead of guessing."""


def mergeable_runs(speakers: list[str | None]) -> set[tuple[int, int]]:
    """Line-index ranges a reviewer may merge into a single cue.

    Maximal runs of consecutive lines from the same speaker. A merge is legal when the
    lines it covers sit *inside* one of these — see :func:`can_merge` — not when the
    range happens to equal a whole run, because a reviewer may merge two lines of a
    twelve-line run.
    """
    runs: set[tuple[int, int]] = set()
    start = 0
    for i in range(1, len(speakers) + 1):
        if i == len(speakers) or speakers[i] != speakers[start]:
            if i - 1 > start:
                runs.add((start, i - 1))
            start = i
    return runs


def can_merge(first: int, last: int, speakers: list[str | None]) -> bool:
    """Is merging lines ``first..last`` (inclusive, 0-based) legal?

    Legal when they are contiguous and all from the same speaker. Merging across a
    speaker change would put one person's words into another person's voice; merging
    lines that are not adjacent would reorder the video.
    """
    if last <= first:
        return True
    return speakers[first] == speakers[last] and \
        all(s == speakers[first] for s in speakers[first:last + 1])


def group_review(cues: list[Cue], starts: list[float], speakers: list[str | None],
                 name: str = "review file") -> list[tuple[int, list[int]]]:
    """``[(cue index, [line indices it covers]), ...]`` for a reviewed SRT.

    Whisper leaves fragments — a sentence split so that ``vacuum.`` gets its own 0.34 s
    line, which no language can say in the time available. A reviewer fixes that by
    joining the fragment to the line before it, and this is what accepts the result. A
    cue covering one line is the ordinary case; a cue covering several is a merge, and
    its text is spoken *once*, for the whole merged window, rather than being cut up and
    handed back to lines that each get a fraction of a second.

    Alignment is by time: cue *n* may only start at the line it matches, and each cue
    takes the contiguous lines whose boundaries fall inside it.
    """
    if not cues:
        raise ReviewError(f"{name} has no cues")
    groups: list[tuple[int, list[int]]] = []
    line = 0
    for ci, cue in enumerate(cues):
        if not cue.text.strip():
            raise ReviewError(f"{name}: cue {ci + 1} is empty")
        if line >= len(starts):
            raise ReviewError(f"{name}: more cues than transcript lines ({len(cues)} vs "
                              f"{len(starts)}); delete the file to regenerate it")
        best = min(range(line, len(starts)), key=lambda i: abs(starts[i] - cue.start))
        if best != line:
            raise ReviewError(
                f"{name}: cue {ci + 1} starts at {cue.start:.2f}s, which matches line "
                f"{best + 1}; {best - line} line(s) before it have no cue. Merge lines "
                "into the cue that starts them, or delete the file to regenerate it")
        covered = [line]
        while line + 1 < len(starts) and starts[line + 1] < cue.end:
            covered.append(line + 1)
            line += 1
        if len(covered) > 1:
            span = (covered[0], covered[-1])
            if not can_merge(span[0], span[1], speakers):
                who = {speakers[i] or "voice" for i in covered}
                raise ReviewError(
                    f"{name}: cue {ci + 1} covers lines {span[0] + 1}-{span[1] + 1}, "
                    f"which are not all the same speaker ({', '.join(sorted(who))}). "
                    "Merging those would put one person's words in another's voice")
            log.debug(f"{name}: cue {ci + 1} merges lines {span[0] + 1}-{span[1] + 1} "
                      f"({cue.end - cue.start:.2f}s for {len(covered)} lines)")
        groups.append((ci, covered))
        line += 1
    if line != len(starts):
        raise ReviewError(f"{name}: cues stop at line {line} of {len(starts)}; "
                          "delete the file to regenerate it")
    return groups
