"""Optional vocal isolation with Demucs (MIT), for when only a full mix is available.

A clean voice stem exported from the editor is always better; this is a convenience.
Runs Demucs in a subprocess so its torch usage is fully released afterwards.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ytdub.logging import stage_logger

log = stage_logger("separate")


def isolate_vocals(src: Path, out_dir: Path, *, model: str = "htdemucs",
                   device: str = "cuda") -> Path:
    vocals = out_dir / model / src.stem / "vocals.wav"
    if vocals.exists():
        log.info(f"Using cached vocal stem {vocals}")
        return vocals
    log.info(f"Separating vocals with demucs {model} (convenience; a clean stem is better)")
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", model,
           "-d", "cuda" if device == "cuda" else "cpu", "-o", str(out_dir), str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"demucs failed (is the [separate] extra installed?):\n"
                           f"{proc.stderr[-3000:]}")
    if not vocals.exists():
        raise FileNotFoundError(f"demucs did not produce {vocals}")
    return vocals
