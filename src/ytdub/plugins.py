"""Backend loading: a built-in short name, or any ``module.path:ClassName``.

This is what keeps translation and TTS pluggable. A replacement engine (say a second
TTS to keep working while a Chatterbox bug is diagnosed) is a class in any importable
module, selected with e.g. ``YTDUB_TTS_BACKEND=mytts.backend:MyTTS``; nothing in the
pipeline needs to change. Each interface is documented in its ``base.py``.
"""

from __future__ import annotations

import importlib


def load_class(spec: str, builtin: dict[str, str], kind: str) -> type:
    target = builtin.get(spec.lower(), spec)
    if ":" not in target:
        raise ValueError(f"unknown {kind} backend {spec!r}: use one of "
                         f"{sorted(builtin)} or 'module.path:ClassName'")
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        # Never swallowed: say which backend failed to import and why.
        raise ImportError(f"{kind} backend {spec!r}: cannot import {module_name}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{kind} backend {spec!r}: {module_name} has no {attr!r}") from exc
