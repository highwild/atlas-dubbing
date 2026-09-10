"""Matching a stored voice clip to the speaker the diarizer happens to have called it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ytdub.models import Segment, SpeakerRef
from ytdub.stages.references import apply_matches
from ytdub.stages.voices import (
    DEFAULT_MARGIN,
    DEFAULT_THRESHOLD,
    MERGE_THRESHOLD,
    match_clips,
    voice_clips,
)


def unit(*values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return v / np.linalg.norm(v)


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "atlas.wav").write_bytes(b"x")
    (tmp_path / "guest.wav").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not audio")
    return tmp_path


def test_a_folder_yields_only_audio_files(folder):
    names = [p.name for p in voice_clips(folder)]
    assert names == ["atlas.wav", "guest.wav"]
    assert voice_clips(folder / "atlas.wav") == [folder / "atlas.wav"]
    assert voice_clips(folder / "missing.wav") == []


def test_the_closest_speaker_wins(folder):
    profiles = {"SPK0": unit(1, 0), "SPK1": unit(0, 1), "SPK2": unit(0.6, 0.8)}
    prints = {folder / "atlas.wav": unit(0.98, 0.02)}
    assigned, unmatched, merges = match_clips(profiles, prints)
    assert assigned == {"SPK0": folder / "atlas.wav"}
    assert unmatched == []
    assert merges == []


def test_a_clip_that_is_nobody_is_not_attached_to_anyone(folder):
    # Equidistant from every speaker, so below the threshold: the honest outcome is the
    # automatic reference, not a guess.
    profiles = {"SPK0": unit(1, 0, 1), "SPK1": unit(0, 1, 1)}
    # 0.707 to each and a 0.0 margin, so it fails on both counts.
    prints = {folder / "atlas.wav": unit(1, 1, 0)}
    assigned, unmatched, _ = match_clips(profiles, prints)
    assert assigned == {}
    assert "below" in unmatched[0]


def test_a_near_tie_is_refused(folder):
    # Two speakers almost equidistant from the clip: acting on that would be a coin flip,
    # and attaching your voice to the wrong speaker is worse than not attaching it.
    profiles = {"SPK0": unit(1, 0.01), "SPK1": unit(1, 0.03)}
    prints = {folder / "atlas.wav": unit(1, 0)}
    assigned, unmatched, _ = match_clips(profiles, prints)
    assert assigned == {}
    assert "ambiguous" in unmatched[0]


def test_two_clips_can_land_on_two_speakers(folder):
    # The two-microphone case: an outdoor clip and an internal clip, each matched to the
    # speaker it belongs to.
    profiles = {"SPK0": unit(1, 0), "SPK2": unit(0, 1)}
    prints = {folder / "atlas.wav": unit(0.99, 0.05), folder / "guest.wav": unit(0.05, 0.99)}
    assigned, _, _ = match_clips(profiles, prints)
    assert assigned == {"SPK0": folder / "atlas.wav", "SPK2": folder / "guest.wav"}


def test_no_speakers_is_reported_not_crashed(folder):
    assigned, unmatched, _ = match_clips({}, {folder / "atlas.wav": unit(1, 0)})
    assert assigned == {}
    assert "no speakers" in unmatched[0]


def test_a_split_presenter_is_folded_onto_one_voice(folder):
    # The hydro.wav case: the presenter's intro and his closing lines came out under two
    # labels. Both are closer to his clip than to anyone else's reference, so both are his.
    profiles = {"SPK0": unit(0, 1), "SPK2": unit(1, 0), "SPK3": unit(0.9, 0.436)}
    prints = {folder / "atlas.wav": unit(1, 0.01)}
    assigned, _, merges = match_clips(profiles, prints)
    assert assigned == {"SPK2": folder / "atlas.wav"}
    assert merges == [("SPK3", "SPK2")]


def test_a_speaker_near_the_merge_line_is_left_alone(folder):
    # 0.79 to the clip: enough to be picked as the best match, not enough to claim it is
    # the same person as the label that scored 0.99.
    profiles = {"SPK2": unit(1, 0), "SPK3": unit(0.79, 0.61)}
    prints = {folder / "atlas.wav": unit(1, 0)}
    assigned, _, merges = match_clips(profiles, prints)
    assert assigned == {"SPK2": folder / "atlas.wav"}
    assert merges == []


def test_thresholds_are_the_measured_ones():
    # These are not arbitrary. Against the pipeline's own reference cuts on this box the
    # presenter scored 0.99 against his own clip and 0.54-0.66 against the other four, so
    # anything in that gap separates them; the merge threshold sits above it because a
    # wrong merge silences a real voice.
    assert 0.70 <= DEFAULT_THRESHOLD <= 0.80
    assert 0 < DEFAULT_MARGIN <= 0.1
    assert DEFAULT_THRESHOLD < MERGE_THRESHOLD <= 0.95


# --------------------------------------------------------------- applying a match

def clip_file(tmp_path, name="atlas.wav") -> Path:
    from ytdub.audio import write_wav

    path = tmp_path / name
    write_wav(path, np.zeros(8000, dtype=np.float32), 16000)
    return path


def refs_for(*speakers) -> dict[str, SpeakerRef]:
    return {spk: SpeakerRef(spk, Path(f"/tmp/ref_{spk}.wav"), 5.0, "segments 0-2")
            for spk in speakers}


def segs(*speakers) -> list[Segment]:
    return [Segment(index=i, start=float(i), end=i + 1.0, text="hi", speaker=spk)
            for i, spk in enumerate(speakers)]


def test_a_fold_rewrites_the_labels_everywhere(tmp_path):
    # Both places matter: the review grouping and the prompt read the segment labels, and
    # synthesis picks a voice by label. Folding one and not the other splits a speaker.
    clip = clip_file(tmp_path)
    refs = refs_for("SPK2", "SPK3")
    segments = segs("SPK3", "SPK2")
    fold = apply_matches(refs, segments, matched={"SPK2": clip},
                         merges=[("SPK3", "SPK2")], user_refs={})
    assert fold == {"SPK3": "SPK2"}
    assert set(refs) == {"SPK2"}
    assert [s.speaker for s in segments] == ["SPK2", "SPK2"]


def test_a_chained_fold_ends_on_one_label(tmp_path):
    clip = clip_file(tmp_path)
    refs = refs_for("SPK1", "SPK2", "SPK3")
    segments = segs("SPK3", "SPK2", "SPK1")
    fold = apply_matches(refs, segments, matched={"SPK1": clip},
                         merges=[("SPK3", "SPK2"), ("SPK2", "SPK1")], user_refs={})
    assert fold == {"SPK3": "SPK1", "SPK2": "SPK1"}
    assert set(refs) == {"SPK1"}
    assert [s.speaker for s in segments] == ["SPK1", "SPK1", "SPK1"]


def test_an_explicit_reference_is_never_folded_away(tmp_path):
    # --ref SPK3=x/... is a person stating which voice goes on which label. A similarity
    # score does not get to overrule that.
    clip = clip_file(tmp_path)
    refs = refs_for("SPK2", "SPK3")
    segments = segs("SPK3", "SPK2")
    fold = apply_matches(refs, segments, matched={"SPK2": clip},
                         merges=[("SPK3", "SPK2")], user_refs={"SPK3": Path("/tmp/named.wav")})
    assert fold == {}
    assert set(refs) == {"SPK2", "SPK3"}
    assert [s.speaker for s in segments] == ["SPK3", "SPK2"]
