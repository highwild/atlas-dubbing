"""On-disk, per-stage cache so long jobs are resumable and front-half work is shared.

Each stage result is stored under ``<work>/<stage>/<key>.json``, where ``key`` hashes
exactly the inputs that affect that stage (content hashes of files, model names, every
parameter that changes the output). Nothing is ever looked up by name or timestamp, so
a changed input can only produce a miss, never a stale hit.

When a stage misses but has run before, the log says *which* inputs changed, so
invalidation is visible rather than mysterious.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ytdub.logging import stage_logger

log = stage_logger("cache")

# Bump when a stage's output format or semantics change in code, to invalidate old
# results without users having to know about --force.
CACHE_VERSION = 1


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class StageCache:
    def __init__(self, root: Path, *, force: bool = False) -> None:
        self.root = root
        self.force = force

    @staticmethod
    def key(inputs: dict) -> str:
        return hash_obj({"cache_version": CACHE_VERSION, **inputs})[:24]

    def _path(self, stage: str, key: str) -> Path:
        return self.root / stage / f"{key}.json"

    def load(self, stage: str, inputs: dict) -> Any | None:
        """Cached data for ``inputs``, or ``None`` (always ``None`` with ``force``)."""
        key = self.key(inputs)
        path = self._path(stage, key)
        if self.force:
            return None
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))["data"]
            except (OSError, ValueError, KeyError):
                log.warning(f"{stage}: unreadable cache entry {path.name}, recomputing")
                return None
            log.info(f"{stage}: cached ({key})")
            return data
        self._explain_miss(stage, inputs)
        return None

    def save(self, stage: str, inputs: dict, data: Any) -> None:
        key = self.key(inputs)
        payload = json.dumps({"inputs": inputs, "data": data}, ensure_ascii=False,
                             default=str, indent=1)
        atomic_write_text(self._path(stage, key), payload)
        atomic_write_text(self.root / stage / "last_inputs.json",
                          json.dumps(inputs, ensure_ascii=False, default=str, indent=1))

    def _explain_miss(self, stage: str, inputs: dict) -> None:
        last = self.root / stage / "last_inputs.json"
        if not last.exists():
            return
        try:
            previous = json.loads(last.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        current = json.loads(json.dumps(inputs, default=str))
        changed = sorted(k for k in set(previous) | set(current)
                         if previous.get(k) != current.get(k))
        if changed:
            log.info(f"{stage}: recomputing, changed since last run: {', '.join(changed)}")
