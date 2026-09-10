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


def diarize_pyannote(audio_path: Path, segments: list[Segment], *, num_speakers: int = 0,
                     device: str = "cpu", hf_token: str | None = None) -> list[Segment]:
    from pyannote.audio import Pipeline

    from ytdub.gpu import free_memory

    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("pyannote diarization needs a Hugging Face token (HF_TOKEN) and "
                           "accepted terms at hf.co/pyannote/speaker-diarization-3.1")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
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
