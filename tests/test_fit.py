"""Timeline fitting, proven in isolation on synthetic clips of known length.

No models, GPU or ffmpeg needed (the one ffmpeg test skips itself when it is absent).
These are the tests that guard the hard requirement (exact duration) and the primary
quality metric (how many segments get compressed, and how hard).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from ytdub.stages.fit import (
    FitError,
    FitItem,
    FitParams,
    budget_slots,
    placed_timings,
    plan,
    render,
    report,
    to_samples,
)

SR = 48_000
CLIP_SR = 24_000


def _check_constraints(placements, items, total, params=FitParams()):
    """The invariants every plan must satisfy, whatever the input."""
    by_key = {it.key: it for it in items}
    assert {p.key for p in placements} == set(by_key), "a segment was dropped"
    ordered = sorted(placements, key=lambda p: p.start)
    for p in ordered:
        assert p.start >= by_key[p.key].src_start - 1e-9, "clip starts before its source"
        assert p.length > 0
        assert p.ratio >= 1.0 - 1e-9, "clips are never stretched longer"
    for a, b in zip(ordered, ordered[1:]):
        assert a.end <= b.start + 1e-7, f"overlap between {a.key} and {b.key}"
    assert ordered[-1].end <= total + 1e-7, "last clip runs past the end"


def test_natural_length_when_it_fits():
    items = [FitItem(0, 0.0, 2.0, 1.5), FitItem(1, 3.0, 5.0, 1.8)]
    placements = plan(items, 6.0)
    _check_constraints(placements, items, 6.0)
    assert all(p.ratio == 1.0 for p in placements)
    assert [p.start for p in placements] == [0.0, 3.0]


def test_overflow_absorbed_by_following_gap():
    # Line 0 needs 0.4s more than its slot; 1.2s of silence follows.
    items = [FitItem(0, 0.0, 2.0, 2.4), FitItem(1, 3.2, 5.0, 1.8)]
    placements = plan(items, 6.0)
    _check_constraints(placements, items, 6.0)
    assert all(p.ratio == 1.0 for p in placements)
    assert placements[1].start == pytest.approx(3.2)  # the gap was enough; nothing moved


def test_borrows_from_gaps_further_along():
    # Back-to-back lines with no gap until a long pause after line 2.
    items = [
        FitItem(0, 0.0, 2.0, 2.5),
        FitItem(1, 2.0, 4.0, 2.5),
        FitItem(2, 4.0, 6.0, 2.5),
        FitItem(3, 9.0, 10.0, 1.0),
    ]
    placements = plan(items, 11.0)
    _check_constraints(placements, items, 11.0)
    assert all(p.ratio == 1.0 for p in placements), "slack downstream should absorb it"
    # Each line is pushed later rather than compressed; line 3 is untouched.
    assert placements[1].start == pytest.approx(2.5)
    assert placements[2].start == pytest.approx(5.0)
    assert placements[3].start == pytest.approx(9.0)


def test_prefers_nearest_gap():
    # A gap exists both right after line 0 and much later; only the first should be used.
    items = [
        FitItem(0, 0.0, 2.0, 2.3),
        FitItem(1, 3.0, 5.0, 2.0),
        FitItem(2, 5.0, 7.0, 2.0),
        FitItem(3, 12.0, 13.0, 1.0),
    ]
    placements = plan(items, 14.0)
    assert [p.start for p in placements] == pytest.approx([0.0, 3.0, 5.0, 12.0])


def test_compresses_only_when_slack_runs_out_and_prefers_long_clips():
    # 10s timeline, speech wall-to-wall, 0.6s too much audio in total. The long clip
    # should take the compression, not the short one.
    items = [FitItem(0, 0.0, 2.0, 2.3), FitItem(1, 2.0, 10.0, 8.3)]
    placements = plan(items, 10.0)
    _check_constraints(placements, items, 10.0)
    total_len = sum(p.length for p in placements)
    assert total_len == pytest.approx(10.0 - 0.0, abs=1e-6)  # slack used exactly
    short, long_ = placements
    assert long_.ratio > short.ratio
    assert long_.ratio <= FitParams().max_ratio


def test_never_starts_before_source_even_when_compressing():
    items = [FitItem(0, 1.0, 2.0, 1.5), FitItem(1, 2.0, 3.0, 1.5)]
    placements = plan(items, 3.5)
    _check_constraints(placements, items, 3.5)
    assert placements[0].start >= 1.0


def test_cap_exceeded_only_when_infeasible_and_reported():
    params = FitParams(max_ratio=1.2)
    # 3s of audio, 2s of room: needs 1.5x, over the cap but under hard max.
    items = [FitItem(0, 0.0, 1.0, 1.5), FitItem(1, 1.0, 2.0, 1.5)]
    placements = plan(items, 2.0, params)
    _check_constraints(placements, items, 2.0, params)
    rep = report(placements, items, 2.0, params)
    assert rep.over_cap >= 1
    assert rep.worst_ratio == pytest.approx(1.5, abs=0.01)


def test_truly_impossible_fit_raises():
    items = [FitItem(0, 0.0, 1.0, 10.0)]
    with pytest.raises(FitError):
        plan(items, 1.0, FitParams(hard_max_ratio=3.0))


def test_delay_cap_respected_when_compression_is_cheap_enough():
    # Pushing the tail back by 1.5s would avoid compression but break the 1s delay cap;
    # compressing line 0 by 1.083x (under the 1.2x cap) respects it.
    params = FitParams(max_delay=1.0)
    items = [FitItem(0, 0.0, 5.0, 6.5)] + [
        FitItem(k, 5.0 + (k - 1) * 1.0, 5.0 + k * 1.0, 1.0) for k in range(1, 6)
    ]
    placements = plan(items, 30.0, params)
    _check_constraints(placements, items, 30.0, params)
    assert max(p.delay for p in placements) <= 1.0 + 1e-6


def _speechlike_13():
    """13 segments modelled on real speech, translated into a longer language.

    Source timings have natural pauses (0.1s-1.6s); clip lengths are the source
    duration times a German/Polish-like expansion factor (mostly 1.1-1.35x, a few lines
    shorter). Includes a very short utterance (segment 6).
    """
    src = [
        (0.30, 3.10), (3.35, 5.20), (5.90, 9.80), (10.00, 11.40), (12.90, 16.20),
        (16.35, 18.00), (18.70, 18.95), (19.10, 23.60), (24.50, 26.10), (26.25, 29.90),
        (31.40, 33.00), (33.20, 36.80), (37.50, 40.20),
    ]
    factors = [1.25, 1.30, 1.15, 1.35, 1.20, 0.95, 1.80, 1.28, 1.10, 1.22, 0.90, 1.33, 1.18]
    items = [FitItem(i, s, e, round((e - s) * f, 3)) for i, ((s, e), f) in enumerate(zip(src, factors))]
    return items, 41.0


def test_13_segment_clip_minimal_compression():
    items, total = _speechlike_13()
    placements = plan(items, total)
    _check_constraints(placements, items, total)
    rep = report(placements, items, total)
    # The reference implementation's per-slot fitting compresses every longer line.
    assert rep.reference_style_compressed >= 10
    # Spec success criterion: low single digits.
    assert rep.compressed <= 3, rep.summary()
    assert rep.worst_ratio <= FitParams().max_ratio
    assert rep.over_cap == 0


def test_dense_long_language_compresses_evenly_within_cap():
    # Speech occupying ~90% of the timeline, translated 20% longer: compression is
    # unavoidable, but must stay within the cap and fill the timeline exactly.
    rng = np.random.default_rng(7)
    t, items = 0.0, []
    for k in range(40):
        dur = float(rng.uniform(1.0, 4.0))
        gap = float(rng.uniform(0.05, 0.4))
        items.append(FitItem(k, t, t + dur, dur * 1.2))
        t += dur + gap
    total = t
    params = FitParams(max_ratio=1.25)
    placements = plan(items, total, params)
    _check_constraints(placements, items, total, params)
    rep = report(placements, items, total, params)
    assert rep.over_cap == 0
    assert rep.worst_ratio <= 1.25 + 1e-6


# --- Rendering ----------------------------------------------------------------


def _tone(seconds: float, sr: int = CLIP_SR, freq: float = 220.0) -> np.ndarray:
    n = int(round(seconds * sr))
    return (0.3 * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)


def _sloppy_stretch(samples, sr, ratio):
    """Naive stand-in for atempo that misses its target length by 7 ms on purpose."""
    n_out = int(round(len(samples) / ratio)) + int(0.007 * sr)
    x = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(x, np.arange(len(samples)), samples).astype(np.float32)


def _onsets(timeline, sr, thresh=0.05):
    """Start sample of each non-silent run (on a 5 ms RMS envelope, not raw samples,
    which dip below any threshold at every zero crossing)."""
    win = int(0.005 * sr)
    env = np.sqrt(np.convolve(timeline ** 2, np.ones(win) / win, mode="same"))
    loud = env > thresh
    edges = np.flatnonzero(np.diff(loud.astype(np.int8)) == 1) + 1
    if loud[0]:
        edges = np.r_[0, edges]
    return edges


@pytest.mark.parametrize("total", [41.0, 41.0 + 1 / 48_000, 41.123457])
def test_render_duration_is_exact(total):
    items, _ = _speechlike_13()
    placements = plan(items, total)
    clips = {it.key: (_tone(it.clip_dur), CLIP_SR) for it in items}
    total_samples = int(round(total * SR))
    out = render(placements, clips, total_samples=total_samples, out_sr=SR, stretch=_sloppy_stretch)
    assert out.shape == (total_samples,)
    assert abs(len(out) / SR - total) < 0.001  # within 1 ms, as required


def test_render_places_every_clip_where_planned():
    items, total = _speechlike_13()
    placements = plan(items, total)
    clips = {it.key: (_tone(it.clip_dur), CLIP_SR) for it in items}
    out = render(placements, clips, total_samples=int(total * SR), out_sr=SR, stretch=_sloppy_stretch)
    timings = placed_timings(placements, SR, int(total * SR))
    onsets = _onsets(out, SR)
    # Adjacent clips with a zero gap merge into one run, so compare against gapped starts.
    expected = sorted(start for start, _ in timings.values())
    for o in onsets:
        assert min(abs(o / SR - e) for e in expected) < 0.015
    # No content lost: every planned clip has energy where it was placed.
    for start, end in timings.values():
        seg = out[int(start * SR):int(end * SR)]
        assert np.sqrt(np.mean(seg ** 2)) > 0.05


def test_to_samples_never_overlaps_or_starts_early():
    items = [FitItem(k, k * 0.3333333, k * 0.3333333 + 0.3333333, 0.3333333) for k in range(30)]
    total = 30 * 0.3333333 + 0.001
    placements = plan(items, total, FitParams(min_gap=0.0))
    total_samples = int(round(total * SR))
    rps = to_samples(placements, SR, total_samples)
    for a, b in zip(rps, rps[1:]):
        assert a.start_sample + a.length_samples <= b.start_sample
    for rp, it in zip(rps, items):
        assert rp.start_sample >= it.src_start * SR - 1e-6
    assert rps[-1].start_sample + rps[-1].length_samples <= total_samples


def test_budget_slots_include_borrowable_slack():
    starts, ends = [0.0, 3.0, 4.0], [2.0, 3.9, 5.0]
    slots = budget_slots(starts, ends, 20.0, min_gap=0.1, max_borrow=2.0)
    assert slots[0] == pytest.approx(2.9)   # source 2.0 + 0.9 of the following gap
    assert slots[1] == pytest.approx(0.9)   # no usable gap: next line starts at 4.0
    assert slots[2] == pytest.approx(3.0)   # long tail, capped at +2.0s


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_ffmpeg_atempo_stretch_hits_target(tmp_path):
    from ytdub.ffmpeg import stretch_samples

    clip = _tone(3.0)
    out = stretch_samples(clip, CLIP_SR, 1.2)
    assert abs(len(out) / CLIP_SR - 2.5) < 0.03
