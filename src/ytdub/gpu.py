"""Explicit VRAM release between stages.

Whisper, the Ollama model and Chatterbox never need to be resident together: each
stage loads its model, uses it, and frees it before the next one loads. This is what
lets a larger translation model be used on a 16 GB card.
"""

from __future__ import annotations

import gc
import sys

from ytdub.logging import stage_logger

log = stage_logger("gpu")


def free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")  # only if some stage already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.debug(f"CUDA allocated after free: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
