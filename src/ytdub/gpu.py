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
    """Best-effort VRAM release. Never raises.

    Releasing memory is housekeeping: the driver frees everything when the process exits,
    so failing here costs nothing — but raising costs a lot. A CUDA context that has been
    poisoned by a device-side assert (``RuntimeError: CUDA error: ...``) makes every later
    CUDA call fail, including ``empty_cache``, and this runs from the teardown path: an
    exception here skips the report the finished languages were supposed to be written
    into, which is the one thing a failed job must not lose.
    """
    gc.collect()
    torch = sys.modules.get("torch")  # only if some stage already imported it
    if torch is None:
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        log.debug(f"CUDA allocated after free: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    except RuntimeError as exc:
        log.debug(f"could not release CUDA memory ({exc}); the driver frees it at exit")
