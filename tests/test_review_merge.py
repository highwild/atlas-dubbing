"""Reviewing a fragment Whisper left with a fraction of a second: merging it into the
line before it, and having the pipeline accept that instead of refusing the file.

Timings are the real ones from a hydro.wav transcript, because they are what decides how
a cue aligns to a line — invented round numbers let a cue swallow a line it shouldn't.
"""

from __future__ import annotations

import pytest

from ytdub.subtitles import Cue, ReviewError, group_review, mergeable_runs

# lines 1-8: start, end, speaker. Line 7 is "vacuum." on 0.34s, the fragment a reviewer
# joins to line 6.
LINES = [
    (0.00, 8.90, "SPK0"), (8.90, 10.00, "SPK0"), (10.20, 20.60, "SPK0"),
    (20.62, 25.64, "SPK0"), (26.16, 34.64, "SPK1"), (34.64, 34.98, "SPK1"),
    (35.54, 43.58, "SPK1"), (44.34, 49.84, "SPK1"),
]
STARTS = [s for s, _, _ in LINES]
ENDS = [e for _, e, _ in LINES]
SPEAKERS = [k for _, _, k in LINES]
TEXTS = [f"line {i + 1}" for i in range(len(LINES))]


def cues_for(rows):
    return [Cue(start, end, text) for start, end, text in rows]


def review(texts=None):
    return cues_for([(s, e, t) for (s, e, _), t in zip(LINES, texts or TEXTS)])


def test_an_unmodified_review_file_is_read_line_for_line():
    groups = group_review(review(), STARTS, SPEAKERS)
    assert [lines for _, lines in groups] == [[i] for i in range(len(LINES))]


def test_a_merged_cue_covers_both_lines_and_keeps_its_text_whole():
    # Lines 5 and 6 ("...we can operate in air or" / "vacuum.") joined into one cue.
    rows = [(LINES[0][0], LINES[0][1], TEXTS[0]), (LINES[1][0], LINES[1][1], TEXTS[1]),
            (LINES[2][0], LINES[2][1], TEXTS[2]), (LINES[3][0], LINES[3][1], TEXTS[3]),
            (LINES[4][0], LINES[5][1], "we can operate in air or vacuum."),
            (LINES[6][0], LINES[6][1], TEXTS[6]), (LINES[7][0], LINES[7][1], TEXTS[7])]
    groups = group_review(cues_for(rows), STARTS, SPEAKERS)
    assert len(groups) == 7
    assert groups[4] == (4, [4, 5])          # the merged cue covers lines 5 and 6
    assert groups[5] == (5, [6])             # and the next cue still lands on line 7


def test_merging_across_a_speaker_change_is_refused():
    # A single-speaker transcript cannot test this: every merge is legal there. Two
    # speakers, then one cue spanning both lines.
    starts = [0.0, 10.0]
    speakers = ["SPK0", "SPK1"]
    rows = [(0.0, 20.0, "both speakers in one cue")]
    with pytest.raises(ReviewError, match="not all the same speaker .SPK0, SPK1."):
        group_review(cues_for(rows), starts, speakers)


def test_a_cue_that_skips_a_line_is_refused():
    # A cue starting at line 4 when line 3 has had no cue: the gap is named, because
    # silently shifting everything by one would put every later line in the wrong slot.

    rows = [(LINES[0][0], LINES[0][1], TEXTS[0]),
            (LINES[1][0], LINES[1][1], TEXTS[1]),
            (LINES[3][0], LINES[3][1], TEXTS[3]),
            (LINES[4][0], LINES[4][1], TEXTS[4]),
            (LINES[5][0], LINES[5][1], TEXTS[5]),
            (LINES[6][0], LINES[6][1], TEXTS[6]),
            (LINES[7][0], LINES[7][1], TEXTS[7])]
    with pytest.raises(ReviewError, match="have no cue"):
        group_review(cues_for(rows), STARTS, SPEAKERS)


def test_too_many_or_too_few_cues_is_refused():
    many = [(LINES[0][0] + i * 1.0, LINES[0][0] + i * 1.0 + 0.5, f"x{i}")
            for i in range(len(LINES) + 3)]
    with pytest.raises(ReviewError, match="more cues than transcript lines"):
        group_review(cues_for(many), STARTS, SPEAKERS)
    with pytest.raises(ReviewError, match="cues stop at line"):
        group_review(review(TEXTS[:3]), STARTS, SPEAKERS)
    with pytest.raises(ReviewError, match="has no cues"):
        group_review([], STARTS, SPEAKERS)


def test_an_empty_cue_is_refused():
    texts = list(TEXTS)
    texts[2] = "   "
    with pytest.raises(ReviewError, match="cue 3 is empty"):
        group_review(review(texts), STARTS, SPEAKERS)


def test_mergeable_runs_are_same_speaker_runs():
    assert mergeable_runs(["A", "A", "A"]) == {(0, 2)}
    assert mergeable_runs(["A", "B", "B"]) == {(1, 2)}
    assert mergeable_runs(["A", "B", "A"]) == set()   # nothing adjacent and equal
    assert mergeable_runs([None, None]) == {(0, 1)}   # single-voice content
