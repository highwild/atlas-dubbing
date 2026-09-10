"""Match a stored voice clip to the speaker the diarizer called it.

The problem this solves is not clone quality, it is the *naming*. A reference is attached
to a diarization label (``SPK0``, ``SPK2``), and those labels are assigned per file by
clustering — so on the next video the presenter can come out as a different label, and
``--ref SPK0=atlas.wav`` then clones someone else's voice onto your lines, silently.
Instead of naming the label, identify the voice.

Matching happens against the **reference clips the pipeline cut for itself**, not against
the raw audio, and that is the difference between working and not working. A stored clip
recorded on another rig loses about 0.3 of cosine similarity to level, bandwidth and room
differences when compared with raw in-file speech, which flattens every speaker into
0.54-0.66 — measured on this box, the right speaker scoring 0.64 against the wrong one's
0.61 is a coin flip. Comparing two clips that both came out of the pipeline's own cutting
and normalisation separates the same speakers at 0.99 against 0.66.

Everything else follows from that: a speaker whose reference does not match you is left
alone, and a *second* label that also matches you (a presenter the diarizer split in two)
is folded into the first, so a whole performance ends up on one voice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ytdub.logging import stage_logger

log = stage_logger("voices")

# Minimum similarity to accept a match. Measured against the pipeline's own cuts on this
# box: the presenter's own clip scored 0.99 against his reference and 0.54-0.66 against
# the other four. A threshold in the middle of that gap is not a close call — it is chosen
# so a mediocre clip (poor mic, short) still clears it while no other speaker can.
DEFAULT_THRESHOLD = 0.75
# And it must beat the runner-up by this much, so a two-way tie is not acted on.
DEFAULT_MARGIN = 0.05
# A second label counts as the same person when its own reference matches the stored clip
# this well. Higher than DEFAULT_THRESHOLD on purpose: merging two speakers is a bigger
# claim than picking one, and a wrong merge would silence a real voice.
MERGE_THRESHOLD = 0.80

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


@dataclass
class Match:
    """One stored clip, the speaker it matched, and any labels folded into it."""

    speaker: str
    path: Path
    score: float
    runner_up: float
    merged: list[str]  # other labels that also match this clip


def voice_clips(path: Path) -> list[Path]:
    """The audio files at ``path``: the file itself, or every audio file in the folder."""
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    return [path] if path.is_file() else []


def embed_file(path: Path, encoder):
    """Embed an audio file, peak-normalised.

    Both sides of every comparison go through this, so a quiet clip and a loud one are
    still comparable — the same normalisation refs.extract_region applies to the clips
    the pipeline cuts for itself."""
    import numpy as np
    from resemblyzer import preprocess_wav

    from ytdub.audio import read_mono

    samples, sr = read_mono(path)
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0:
        samples = samples * (0.7 / peak)
    return encoder.embed_utterance(preprocess_wav(samples, source_sr=sr))


def _cos(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def match_clips(ref_prints: dict[str, object], prints: dict[Path, object], *,
                threshold: float = DEFAULT_THRESHOLD,
                margin: float = DEFAULT_MARGIN,
                merge_threshold: float = MERGE_THRESHOLD
                ) -> tuple[dict[str, Path], list[str], list[tuple[str, str]]]:
    """``({speaker: clip}, unmatched reasons, merged pairs)``. Pure, so it is testable.

    ``ref_prints`` maps each speaker label to the embedding of the reference the pipeline
    cut for it; ``prints`` maps each stored clip to its embedding.
    """
    assigned: dict[str, Path] = {}
    reasons: list[str] = []
    merges: list[tuple[str, str]] = []
    for path, embedding in prints.items():
        scores = sorted(((_cos(embedding, v), k) for k, v in ref_prints.items()),
                        reverse=True)
        if not scores:
            reasons.append(f"{path.name}: no speakers to match against")
            continue
        best_score, best = scores[0]
        runner = scores[1][0] if len(scores) > 1 else 0.0
        if best_score < threshold:
            reasons.append(f"{path.name}: best match {best} only {best_score:.2f} "
                           f"(below {threshold:.2f})")
            continue
        if len(scores) > 1 and best_score - runner < margin:
            reasons.append(f"{path.name}: ambiguous, {best} {best_score:.2f} vs "
                           f"{scores[1][1]} {runner:.2f}")
            continue
        assigned[best] = path
        log.success(f"{best}: matched {path.name} ({best_score:.2f}, next {runner:.2f})")
        for score, other in scores[1:]:
            if score >= merge_threshold:
                merges.append((other, best))
                log.success(f"{other} is also {path.name} ({score:.2f}); its lines will "
                            f"be spoken as {best}")
    return assigned, reasons, merges


def match_voice(path: Path, refs, *, device: str = "cpu",
                threshold: float = DEFAULT_THRESHOLD,
                margin: float = DEFAULT_MARGIN,
                merge_threshold: float = MERGE_THRESHOLD
                ) -> tuple[dict[str, Path], list[tuple[str, str]]]:
    """``({speaker: clip}, merged pairs)`` for the stored voice(s) at ``path``.

    ``refs`` is the ``{speaker: SpeakerRef}`` the pipeline has just cut. Never raises: a
    missing file or an unreadable clip logs and returns nothing, because a reference we
    cannot match is not a reason to fail a job.
    """
    clips = voice_clips(path)
    if not clips:
        log.warning(f"--voice {path}: no audio file found; using the automatic references")
        return {}, []
    from resemblyzer import VoiceEncoder

    from ytdub.gpu import free_memory

    log.info(f"matching {len(clips)} stored voice clip(s) against the {len(refs)} "
             "reference(s) just cut")
    encoder = VoiceEncoder("cuda" if device == "cuda" else "cpu")
    try:
        ref_prints = {spk: embed_file(ref.path, encoder) for spk, ref in refs.items()}
        prints = {}
        for clip in clips:
            try:
                prints[clip] = embed_file(clip, encoder)
            except Exception:
                log.opt(exception=True).warning(f"could not read voice clip {clip}")
    finally:
        del encoder
        free_memory()
    assigned, reasons, merges = match_clips(ref_prints, prints, threshold=threshold,
                                            margin=margin, merge_threshold=merge_threshold)
    for reason in reasons:
        log.warning(f"--voice {reason}; keeping the automatic reference for that speaker")
    return assigned, merges
