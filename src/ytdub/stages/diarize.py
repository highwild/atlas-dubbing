"""Diarization: who speaks when (only with ``--speakers``).

Runs ONCE per source file and is cached; every language reuses the same labels and
the same speaker -> reference-clip mapping, so a person keeps their own cloned voice
in every language.

Backends:
  * ``embedding`` (default, token-free): Resemblyzer speaker embeddings per segment,
    then clustering. Apache-2.0, no Hugging Face account needed.
  * ``pyannote``: more accurate, but needs an HF token and terms acceptance, so it is
    offered, never required.

:func:`cluster_embeddings`, :func:`assign_speakers` and :func:`fill_short_labels` are
pure and unit-tested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from ytdub.logging import stage_logger
from ytdub.models import Segment

log = stage_logger("diarize")

# Embeddings of very short utterances are unreliable; those segments take the label
# of the nearest segment in time instead.
MIN_EMBED_SECONDS = 0.8


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """Tag each segment with the speaker whose turns overlap it most (nearest if none)."""
    if not turns:
        return segments
    out = []
    for seg in segments:
        best, best_overlap = None, 0.0
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best, best_overlap = turn.speaker, overlap
        if best is None:
            mid = (seg.start + seg.end) / 2
            best = min(turns, key=lambda t: abs((t.start + t.end) / 2 - mid)).speaker
        out.append(replace(seg, speaker=best))
    return out


def cluster_embeddings(embeddings, num_speakers: int = 0, threshold: float = 0.75) -> list[int]:
    """Cluster speaker embeddings. ``num_speakers`` > 0 forces k (cosine k-means with
    farthest-point seeding); 0 auto-estimates by average-linkage merging until the two
    closest clusters are less similar than ``threshold``."""
    import numpy as np

    x = np.asarray(embeddings, dtype=np.float64)
    n = len(x)
    if n <= 1:
        return [0] * n
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    if num_speakers and num_speakers >= 1:
        k = min(num_speakers, n)
        centroids = [x[0]]
        for _ in range(1, k):
            dist = 1.0 - (x @ np.array(centroids).T).max(axis=1)
            centroids.append(x[int(np.argmax(dist))])
        c = np.array(centroids)
        labels = np.full(n, -1)
        for _ in range(50):
            new = (x @ c.T).argmax(axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                members = x[labels == j]
                if len(members):
                    m = members.mean(axis=0)
                    c[j] = m / (np.linalg.norm(m) + 1e-9)
        return labels.tolist()

    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best_sim, bi, bj = -1.0, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = float(np.mean([x[a] @ x[b] for a in clusters[i] for b in clusters[j]]))
                if sim > best_sim:
                    best_sim, bi, bj = sim, i, j
        if best_sim < threshold:
            break
        clusters[bi] += clusters[bj]
        del clusters[bj]
    labels = [0] * n
    for lbl, members in enumerate(clusters):
        for idx in members:
            labels[idx] = lbl
    return labels


def parse_speaker_map(spec: str) -> list[tuple[int, int, str]]:
    """``"6-15=SPK3,22=SPK1"`` -> ``[(6, 15, "SPK3"), (22, 22, "SPK1")]``.

    Line numbers are the SRT's (1-based). Raises ``ValueError`` on anything malformed:
    a wrong boundary silently applied to the wrong lines is worse than a failed job.
    """
    entries: list[tuple[int, int, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"--speaker-map {chunk!r} is missing '='")
        where, _, speaker = chunk.partition("=")
        speaker = speaker.strip()
        if not speaker:
            raise ValueError(f"--speaker-map {chunk!r} has no speaker label")
        first, sep, last = where.strip().partition("-")
        try:
            start = int(first)
            end = int(last) if sep else start
        except ValueError as exc:
            raise ValueError(f"--speaker-map {chunk!r}: line numbers must be integers") from exc
        if start < 1 or end < start:
            raise ValueError(f"--speaker-map {chunk!r}: need 1 <= first <= last")
        entries.append((start, end, speaker))
    if not entries:
        raise ValueError("--speaker-map is empty")
    for (a, b, _), (c, d, _) in zip(entries, entries[1:]):
        if c <= b:
            raise ValueError(f"--speaker-map overlaps: {a}-{b} and {c}-{d}")
    return entries


def apply_speaker_map(segments: list[Segment], spec: str) -> list[Segment]:
    """Re-label the lines ``spec`` names, leaving every other line as diarized.

    Necessary because diarization gets boundaries wrong on crosstalk and on similar
    voices, and no amount of re-clustering fixes a stretch the model merged: the person
    who watched the video knows which lines are whose, and this is how they say so.
    """
    if not spec.strip():
        return segments
    entries = parse_speaker_map(spec)
    out = list(segments)
    for start, end, speaker in entries:
        if end > len(out):
            raise ValueError(f"--speaker-map {start}-{end} is past the end of the "
                             f"transcript ({len(out)} lines)")
        for i in range(start - 1, end):
            out[i] = replace(out[i], speaker=speaker)
    counts: dict[str, int] = {}
    for seg in out:
        counts[seg.speaker] = counts.get(seg.speaker, 0) + 1
    log.success(f"speaker map applied: {counts}")
    return out


def fill_short_labels(segments: list[Segment], labels: dict[int, int]) -> list[int]:
    """Label per segment; segments missing from ``labels`` take the nearest labelled one."""
    labelled = [s for s in segments if s.index in labels]
    out = []
    for seg in segments:
        if seg.index in labels:
            out.append(labels[seg.index])
            continue
        mid = (seg.start + seg.end) / 2
        nearest = min(labelled, key=lambda s: abs((s.start + s.end) / 2 - mid))
        out.append(labels[nearest.index])
    return out


def diarize_embedding(audio_path: Path, segments: list[Segment], *, num_speakers: int = 0,
                      device: str = "cpu") -> list[Segment]:
    import numpy as np
    from resemblyzer import VoiceEncoder, preprocess_wav

    from ytdub.audio import read_mono, resample
    from ytdub.gpu import free_memory

    samples, sr = read_mono(audio_path)
    if sr != 16000:
        samples, sr = resample(samples, sr, 16000), 16000
    encoder = VoiceEncoder("cuda" if device == "cuda" else "cpu")
    try:
        embeddable = [s for s in segments if s.duration >= MIN_EMBED_SECONDS] or segments
        embeddings = []
        for seg in embeddable:
            clip = samples[int(seg.start * sr):int(seg.end * sr)]
            wav = preprocess_wav(clip.astype(np.float32), source_sr=sr)
            if len(wav) < sr * 0.3:  # VAD trimmed almost everything; use the raw slice
                wav = clip.astype(np.float32)
            embeddings.append(encoder.embed_utterance(wav))
    finally:
        del encoder
        free_memory()
    raw = cluster_embeddings(embeddings, num_speakers=num_speakers)
    labels = fill_short_labels(segments, {s.index: lbl for s, lbl in zip(embeddable, raw)})
    out = [replace(s, speaker=f"SPK{lbl}") for s, lbl in zip(segments, labels)]
    counts = {f"SPK{k}": labels.count(k) for k in sorted(set(labels))}
    log.success(f"{len(counts)} voice(s): {counts}")
    return out


def _accept_legacy_hub_token() -> None:
    """Let pyannote 3.x talk to huggingface_hub 1.x.

    pyannote passes ``use_auth_token=`` to ``hf_hub_download``; huggingface_hub 1.0
    removed that name in favour of ``token``, so the call raises ``TypeError`` before any
    download starts. pyannote 4 fixes it but needs torch>=2.8, which the production
    Chatterbox pin (torch 2.6) forbids, and our transformers pin requires
    huggingface_hub>=1.3 — so neither side can move and the argument is translated
    instead. Every pyannote module that did ``from huggingface_hub import
    hf_hub_download`` holds its own reference, so each is patched where it looks.
    """
    import sys

    import huggingface_hub

    def shim_for(original):
        if getattr(original, "_ytdub_token_shim", False):
            return None

        def shim(*args, _original=original, use_auth_token=None, **kwargs):
            if use_auth_token is not None and "token" not in kwargs:
                kwargs["token"] = use_auth_token
            return _original(*args, **kwargs)

        shim._ytdub_token_shim = True
        shim._ytdub_wrapped = original
        return shim

    for name, module in list(sys.modules.items()):
        if not name.startswith("pyannote") or module is None:
            continue
        original = getattr(module, "hf_hub_download", None)
        if original is None:
            continue
        shim = shim_for(original)
        if shim is not None:
            setattr(module, "hf_hub_download", shim)
    # And the hub module itself, for anything that calls it as an attribute.
    shim = shim_for(huggingface_hub.hf_hub_download)
    if shim is not None:
        huggingface_hub.hf_hub_download = shim


def _allow_pyannote_checkpoints() -> None:
    """Let pyannote 3.x load its own checkpoints under torch>=2.6.

    torch 2.6 flipped ``torch.load``'s default to ``weights_only=True``; pyannote's
    released checkpoints carry a handful of ordinary pyannote classes, which the safe
    unpickler refuses unless they are allowlisted, so loading raises ``UnpicklingError``
    naming the class it wants. These are pyannote's official weights from the gated HF
    repos; the narrow fix is to allowlist the classes the loader asks for, one per round,
    rather than switching the safe loader off entirely.
    """
    import torch

    wanted = ["torch.torch_version.TorchVersion",
              "pyannote.audio.core.task.Specifications",
              "pyannote.audio.core.task.Problem",
              "pyannote.audio.core.task.Resolution"]
    allowed = []
    for dotted in wanted:
        module_name, _, class_name = dotted.rpartition(".")
        try:
            module = __import__(module_name, fromlist=[class_name])
            allowed.append(getattr(module, class_name))
        except Exception:
            log.debug(f"pyannote checkpoint class not available: {dotted}")
    if allowed:
        torch.serialization.add_safe_globals(allowed)


def _ensure_hub_token(token: str) -> None:
    """Export the token for the hub, in both the current and legacy variable names.

    pyannote's own download calls carry no token through to huggingface_hub, so the
    environment is what authenticates the gated model fetch.
    """
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def diarize_pyannote(audio_path: Path, segments: list[Segment], *, num_speakers: int = 0,
                     device: str = "cpu", hf_token: str | None = None) -> list[Segment]:
    import pyannote.audio.core.model  # noqa: F401  (imported so the token shim can reach it)
    from pyannote.audio import Pipeline

    from ytdub.gpu import free_memory

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("pyannote diarization needs a Hugging Face token (YTDUB_HF_TOKEN) "
                           "and accepted terms at hf.co/pyannote/speaker-diarization-3.1")
    _ensure_hub_token(token)
    _accept_legacy_hub_token()
    _allow_pyannote_checkpoints()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if device == "cuda":
        import torch

        pipeline.to(torch.device("cuda"))
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    try:
        result = pipeline(str(audio_path), **kwargs)
        turns = [SpeakerTurn(float(t.start), float(t.end), str(spk))
                 for t, _, spk in result.itertracks(yield_label=True)]
    finally:
        del pipeline
        free_memory()
    names = {spk: f"SPK{i}" for i, spk in enumerate(sorted({t.speaker for t in turns}))}
    turns = [SpeakerTurn(t.start, t.end, names[t.speaker]) for t in turns]
    return assign_speakers(segments, turns)
