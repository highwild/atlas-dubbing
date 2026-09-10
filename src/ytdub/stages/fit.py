"""Fit stage — place synthesized clips on the timeline. Replaces ``synchronize.py``.

The reference implementation squeezed every clip into its *own* source slot, so any
translated line longer than the English got time-compressed (60-85% of segments in
practice, with an audible warble). This stage treats fitting as one global problem:

    total output duration is a HARD constraint; per-segment duration is NOT.

Clips may run past their source end and borrow the silence that follows, pushing the
next line later; that line may in turn borrow from the gap after it, and so on. Only
when the slack genuinely runs out is audio compressed, and then preferentially on long
clips at small ratios, where it is least perceptible.

The problem is solved as a linear programme (scipy / HiGHS). Per segment ``i`` with
source start ``s_i``, natural clip length ``d_i`` and placed start ``t_i``:

    placed length   L_i = d_i - x1_i - x2_i - x3_i     (x = seconds removed by compression)
    no early start  t_i >= s_i
    no overlap      t_i + L_i + g_i <= t_{i+1}          (g_i = small breath gap)
    exact end       t_n + L_n <= T                       (then padded to exactly T)
    delay cap       t_i - s_i <= max_delay + z_i         (z_i = penalised overshoot)

The three compression bands give a convex, piecewise-linear cost on the ratio:
``x1`` up to ``imperceptible_ratio`` is cheap, ``x2`` up to ``max_ratio`` is expensive,
``x3`` beyond the cap is prohibitive and only used when nothing else is feasible (it is
reported loudly). Costs are per unit of ``x/d`` (= ``1 - 1/ratio``), so a second removed
from a long clip costs less than a second removed from a short one. Delay carries a tiny
per-second cost so the nearest gap is used before gaps further along.

Planning is pure (seconds in, placements out) and rendering works in integer samples,
so the output length is exact by construction and both halves are testable without any
model, GPU or ffmpeg.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from ytdub.audio import fade_edges, fit_length, resample
from ytdub.logging import stage_logger

log = stage_logger("fit")

# Objective weights. Only their ordering matters much: delay << band1 << band2 < excess
# delay < band3. See the module docstring.
_W_DELAY = 1e-4
_W_BAND1 = 1.0
_W_BAND2 = 10.0
_W_DELAY_EXCESS = 50.0
_W_BAND3 = 1000.0

# Ratios this close to 1.0 are solver noise, not compression.
_RATIO_EPS = 1e-6


class FitError(RuntimeError):
    """The clips cannot be placed at all (would need > hard_max_ratio compression)."""


@dataclass
class FitParams:
    max_ratio: float = 1.2
    """Compression cap. Exceeded only if the timeline is otherwise infeasible."""
    imperceptible_ratio: float = 1.05
    """Compression up to this ratio is treated as nearly free."""
    hard_max_ratio: float = 3.0
    """Absolute ceiling; beyond this the fit fails rather than produce garbage."""
    min_gap: float = 0.12
    """Breath gap kept between consecutive clips (never more than the source gap)."""
    max_delay: float = 2.0
    """Soft cap on how late a clip may start relative to its source segment."""


@dataclass
class FitItem:
    key: int
    src_start: float
    src_end: float
    clip_dur: float


@dataclass
class Placement:
    key: int
    src_start: float
    src_end: float
    clip_dur: float
    start: float
    length: float

    @property
    def end(self) -> float:
        return self.start + self.length

    @property
    def ratio(self) -> float:
        return self.clip_dur / self.length if self.length > 0 else 1.0

    @property
    def delay(self) -> float:
        return self.start - self.src_start


@dataclass
class FitReport:
    segments: int
    compressed: int
    compressed_perceptible: int
    over_cap: int
    worst_ratio: float
    mean_delay: float
    max_delay: float
    over_delay_cap: int
    total_seconds: float
    reference_style_compressed: int
    compressed_keys: list[int] = field(default_factory=list)
    over_cap_keys: list[int] = field(default_factory=list)
    over_delay_keys: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"compressed {self.compressed}/{self.segments} segments "
            f"(worst {self.worst_ratio:.3f}x, {self.compressed_perceptible} above "
            f"imperceptible, {self.over_cap} over cap) | delay mean {self.mean_delay:.2f}s "
            f"max {self.max_delay:.2f}s | reference-style fitting would have compressed "
            f"{self.reference_style_compressed}"
        )

    def warnings(self, params: FitParams | None = None) -> list[str]:
        """Human-readable problems, logged at WARNING and shown in the job summary."""
        p = params or FitParams()
        out = []
        if self.over_cap_keys:
            out.append(f"{len(self.over_cap_keys)} line(s) compressed beyond the "
                       f"{p.max_ratio}x cap (worst {self.worst_ratio:.2f}x): lines "
                       f"{self.over_cap_keys}. Their translations are too long for the "
                       "available time; shorten them in the review SRT.")
        if self.over_delay_keys:
            out.append(f"{len(self.over_delay_keys)} line(s) start more than "
                       f"{p.max_delay}s late (max {self.max_delay:.2f}s): lines "
                       f"{self.over_delay_keys}.")
        return out


def _gaps(items: list[FitItem], min_gap: float) -> list[float]:
    """Breath gap required after each item: ``min_gap``, but never more than the source had."""
    out = []
    for a, b in zip(items, items[1:]):
        out.append(max(0.0, min(min_gap, b.src_start - a.src_end)))
    out.append(0.0)
    return out


def _solve(items: list[FitItem], total: float, p: FitParams, gaps: list[float]):
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    n = len(items)
    # Variable layout: [t (n) | x1 (n) | x2 (n) | x3 (n) | z (n)]
    T0, X1, X2, X3, Z = (k * n for k in range(5))
    nv = 5 * n
    d = np.array([it.clip_dur for it in items])
    s = np.array([it.src_start for it in items])

    c = np.zeros(nv)
    c[T0:T0 + n] = _W_DELAY
    c[X1:X1 + n] = _W_BAND1 / d
    c[X2:X2 + n] = _W_BAND2 / d
    c[X3:X3 + n] = _W_BAND3 / d
    c[Z:Z + n] = _W_DELAY_EXCESS

    r1 = min(p.imperceptible_ratio, p.max_ratio)
    r2 = max(p.max_ratio, r1)
    r3 = max(p.hard_max_ratio, r2)
    bounds = (
        [(float(s[i]), float(total)) for i in range(n)]
        + [(0.0, float(d[i] * (1 - 1 / r1))) for i in range(n)]
        + [(0.0, float(d[i] * (1 / r1 - 1 / r2))) for i in range(n)]
        + [(0.0, float(d[i] * (1 / r2 - 1 / r3))) for i in range(n)]
        + [(0.0, None) for _ in range(n)]
    )

    a = lil_matrix((2 * n, nv))
    b = np.zeros(2 * n)
    for i in range(n):
        # t_i - x1 - x2 - x3 - t_{i+1} <= -d_i - g_i   (or <= T - d_n for the last)
        a[i, T0 + i] = 1.0
        a[i, X1 + i] = a[i, X2 + i] = a[i, X3 + i] = -1.0
        if i + 1 < n:
            a[i, T0 + i + 1] = -1.0
            b[i] = -d[i] - gaps[i]
        else:
            b[i] = total - d[i]
        # t_i - z_i <= s_i + max_delay
        a[n + i, T0 + i] = 1.0
        a[n + i, Z + i] = -1.0
        b[n + i] = s[i] + p.max_delay

    res = linprog(c, A_ub=a.tocsr(), b_ub=b, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    x = res.x
    starts = x[T0:T0 + n]
    removed = x[X1:X1 + n] + x[X2:X2 + n] + x[X3:X3 + n]
    return starts, d - removed


def plan(items: list[FitItem], total: float, params: FitParams | None = None) -> list[Placement]:
    """Choose a start and placed length for every clip. Pure; seconds in, seconds out."""
    p = params or FitParams()
    if not items:
        return []
    items = sorted(items, key=lambda it: (it.src_start, it.key))
    for it in items:
        if it.clip_dur <= 0:
            raise ValueError(f"segment {it.key}: clip duration must be positive")
        if it.src_start >= total:
            raise ValueError(f"segment {it.key} starts at/after the end of the timeline")

    gaps = _gaps(items, p.min_gap)
    solved = _solve(items, total, p, gaps)
    if solved is None:
        # Only reachable when even hard_max_ratio compression with breath gaps cannot
        # fit. Drop the breath gaps before giving up.
        log.warning("Timeline infeasible with breath gaps; retrying with zero gaps")
        solved = _solve(items, total, p, [0.0] * len(items))
    if solved is None:
        raise FitError(
            f"{len(items)} clips cannot fit in {total:.2f}s even at "
            f"{p.hard_max_ratio}x compression"
        )
    starts, lengths = solved

    out = []
    for it, t, length in zip(items, starts, lengths):
        if length >= it.clip_dur / (1 + _RATIO_EPS):
            length = it.clip_dur
        out.append(Placement(
            key=it.key, src_start=it.src_start, src_end=it.src_end,
            clip_dur=it.clip_dur, start=max(float(t), it.src_start), length=float(length),
        ))
    return out


def reference_style_compressed(items: list[FitItem], max_speedup: float = 1.4) -> int:
    """How many segments the reference implementation's per-slot fitting would compress.

    It squeezed every clip into its own source window. Reported next to our own count
    as the headline quality comparison.
    """
    return sum(
        1 for it in items
        if (it.src_end - it.src_start) > 0.05 and it.clip_dur > (it.src_end - it.src_start)
    )


def report(placements: list[Placement], items: list[FitItem], total: float,
           params: FitParams | None = None) -> FitReport:
    p = params or FitParams()
    ratios = [pl.ratio for pl in placements]
    delays = [pl.delay for pl in placements]
    compressed = [pl for pl in placements if pl.ratio > 1 + _RATIO_EPS]
    return FitReport(
        segments=len(placements),
        compressed=len(compressed),
        compressed_perceptible=sum(1 for r in ratios if r > p.imperceptible_ratio + 1e-6),
        over_cap=sum(1 for r in ratios if r > p.max_ratio + 1e-6),
        worst_ratio=round(max(ratios, default=1.0), 4),
        mean_delay=round(float(np.mean(delays)) if delays else 0.0, 4),
        max_delay=round(max(delays, default=0.0), 4),
        over_delay_cap=sum(1 for dl in delays if dl > p.max_delay + 1e-6),
        total_seconds=total,
        reference_style_compressed=reference_style_compressed(items),
        compressed_keys=[pl.key for pl in compressed],
        over_cap_keys=[pl.key for pl in placements if pl.ratio > p.max_ratio + 1e-6],
        over_delay_keys=[pl.key for pl in placements if pl.delay > p.max_delay + 1e-6],
    )



# --- Rendering ---------------------------------------------------------------

Stretcher = Callable[[np.ndarray, int, float], np.ndarray]
"""``(samples, sample_rate, ratio) -> samples`` shortened by ``ratio`` (> 1), pitch kept."""

# A stretcher may miss its target length slightly; beyond this we warn (it is still
# forced to the planned length, so the timeline stays exact either way).
_LENGTH_TOLERANCE_S = 0.03


@dataclass
class RenderedPlacement:
    key: int
    start_sample: int
    length_samples: int


def to_samples(placements: list[Placement], sr: int, total_samples: int) -> list[RenderedPlacement]:
    """Quantise a plan to integer samples, preserving every constraint exactly.

    Starts round *up* (never before the source), lengths round to nearest; any 1-sample
    collision from rounding pushes the later clip on by that sample.
    """
    out: list[RenderedPlacement] = []
    prev_end = 0
    for pl in sorted(placements, key=lambda q: q.start):
        start = max(int(np.ceil(pl.start * sr - 1e-6)), int(np.ceil(pl.src_start * sr - 1e-6)),
                    prev_end)
        length = int(round(pl.length * sr))
        if start + length > total_samples:
            overrun = start + length - total_samples
            if overrun > int(_LENGTH_TOLERANCE_S * sr):
                raise FitError(f"segment {pl.key} overruns the timeline by {overrun} samples")
            length -= overrun  # sub-tolerance rounding only; clips keep a silent tail margin
        out.append(RenderedPlacement(pl.key, start, max(0, length)))
        prev_end = start + length
    return out


def render(
    placements: list[Placement],
    clips: dict[int, tuple[np.ndarray, int]],
    *,
    total_samples: int,
    out_sr: int,
    stretch: Stretcher,
) -> np.ndarray:
    """Mix every clip onto a silent timeline of exactly ``total_samples`` samples.

    ``clips`` maps segment key to ``(samples, sample_rate)`` of the *trimmed* clip whose
    length was used for planning. Returns float32 mono at ``out_sr``.
    """
    timeline = np.zeros(total_samples, dtype=np.float32)
    by_key = {pl.key: pl for pl in placements}
    for rp in to_samples(placements, out_sr, total_samples):
        pl = by_key[rp.key]
        samples, sr = clips[rp.key]
        if pl.ratio > 1 + _RATIO_EPS:
            samples = stretch(samples, sr, pl.ratio)
        samples = resample(samples, sr, out_sr)
        miss = abs(len(samples) - rp.length_samples) / out_sr
        if miss > _LENGTH_TOLERANCE_S:
            log.warning(
                f"segment {rp.key}: stretched clip missed its planned length by "
                f"{miss * 1000:.0f} ms (forced to fit)"
            )
        samples = fade_edges(fit_length(samples, rp.length_samples), out_sr)
        timeline[rp.start_sample:rp.start_sample + rp.length_samples] += samples
    return timeline


def placed_timings(placements: list[Placement], sr: int, total_samples: int) -> dict[int, tuple[float, float]]:
    """Final ``key -> (start, end)`` in seconds, exactly as rendered. Used for the SRT."""
    return {
        rp.key: (rp.start_sample / sr, (rp.start_sample + rp.length_samples) / sr)
        for rp in to_samples(placements, sr, total_samples)
    }


def budget_slots(
    starts: list[float], ends: list[float], total: float, *, min_gap: float = 0.12,
    max_borrow: float = 2.0,
) -> list[float]:
    """Seconds of speech each segment can realistically occupy, used as translation budget.

    That is its source duration plus the silence after it that the fitter can borrow
    (up to the next segment, less a breath gap), capped at ``max_borrow`` extra seconds
    so a one-word line before a long pause is not invited to become a paragraph.
    """
    out = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        nxt = starts[i + 1] - min_gap if i + 1 < len(starts) else total
        avail = max(e - s, nxt - s)
        out.append(max(0.0, min(avail, (e - s) + max_borrow)))
    return out
