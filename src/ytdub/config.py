"""Central configuration.

Every tunable lives here. Values come from (highest priority first) CLI flags, ``YTDUB_*``
environment variables, a ``.env`` file in the home directory, then these defaults.
Every model is a setting, so any of them can be swapped without code changes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LANGUAGES = ["de", "fr", "pl", "es", "nl", "hi"]


def _default_home() -> Path:
    """``YTDUB_HOME``, else the source checkout (editable install), else the cwd."""
    env = os.environ.get("YTDUB_HOME")
    if env:
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").exists() and (checkout / "synopses").is_dir():
        return checkout
    return Path.cwd().resolve()


@lru_cache(maxsize=1)
def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YTDUB_", env_file=".env", extra="ignore")

    # --- Layout -------------------------------------------------------------
    home: Path = Field(default_factory=_default_home)

    # --- Job ----------------------------------------------------------------
    languages: list[str] = Field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    style: str = "casual"
    source_lang: str | None = None  # None = let Whisper detect it
    speakers: int | None = None  # None = single voice; 0 = auto-detect count; N = N voices
    force: bool = False  # ignore every cache entry (results are still written)

    # --- Compute ------------------------------------------------------------
    device: str = Field(default_factory=detect_device)

    # --- Network ------------------------------------------------------------
    force_ipv4: bool = True
    net_timeout: float = 60.0
    cookies_from_browser: str | None = None  # yt-dlp: when YouTube asks "not a bot?"
    cookies_file: Path | None = None

    # --- Source separation (optional convenience) ---------------------------
    separate: bool = False  # run demucs to isolate vocals when only a full mix exists
    demucs_model: str = "htdemucs"

    # --- Transcription ------------------------------------------------------
    # A size name (downloads on first use) or a local model directory.
    asr_model: str = "large-v3"
    asr_compute_type: str | None = None  # None -> float16 on CUDA, int8 on CPU
    asr_beam_size: int = 5
    vad: bool = True
    vad_min_silence_ms: int = 500
    # Segments whose mean word probability is below this are dropped as probable
    # hallucinated noise (logged + written to dropped.txt for review). 0 disables.
    min_confidence: float = 0.45

    # --- Diarization --------------------------------------------------------
    diarize_method: str = "embedding"  # "embedding" (token-free) | "pyannote"
    hf_token: str | None = None

    # --- Voice references ---------------------------------------------------
    ref_target_seconds: float = 10.0  # Chatterbox conditions on ~10 s
    ref_min_seconds: float = 4.0

    # --- Translation --------------------------------------------------------
    # "ollama" (shipped) or "module.path:ClassName" for a drop-in backend.
    translator: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    # Context window sent with EVERY request. Qwen3-8B KV cache at f16 is ~0.14 MB per
    # token, so 8192 costs ~1.2 GB; it does not share VRAM with Whisper or Chatterbox
    # because those are unloaded between stages.
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.3
    ollama_timeout: float = 180.0  # per-read stall timeout on the streamed response
    translate_batch_lines: int = 30
    translate_context_lines: int = 4  # already-translated lines shown before a batch
    translate_lookahead_lines: int = 3  # source lines shown after a batch
    chars_per_second: float = 15.0
    budget_max_borrow: float = 2.0  # seconds of following silence a budget may include
    budget_tolerance: float = 1.15  # re-request lines longer than budget x this
    budget_retries: int = 2

    # --- Synthesis ------------------------------------------------------------
    # "chatterbox" (shipped) or "module.path:ClassName" for a drop-in backend.
    tts_backend: str = "chatterbox"
    tts_exaggeration: float = 0.5
    tts_cfg_weight: float = 0.5
    tts_temperature: float = 0.8
    tts_seed: int = 1234
    # Chatterbox crashes on utterances this short (IndexError in its alignment
    # analyzer); such lines are merged into an adjacent same-speaker line first.
    min_tts_chars: int = 3
    merge_max_gap: float = 1.5  # only merge into a neighbour this close in time

    # --- Fitting --------------------------------------------------------------
    max_ratio: float = 1.2
    imperceptible_ratio: float = 1.05
    hard_max_ratio: float = 3.0
    min_gap: float = 0.12
    max_delay: float = 2.0

    # --- Output ---------------------------------------------------------------
    sample_rate: int = 48_000
    match_loudness: bool = True
    # Match loudness to this file instead of the input. Useful when dubbing a clean
    # voice stem: the viewer switches between the dub and the *full mix* on YouTube.
    loudness_reference: Path | None = None
    mux_video: bool = True  # also write <lang>.mp4 preview when the input has video

    # --- Paths ------------------------------------------------------------------
    @property
    def input_dir(self) -> Path:
        return self.home / "input"

    @property
    def output_root(self) -> Path:
        return self.home / "output"

    @property
    def work_root(self) -> Path:
        return self.home / "work"

    @property
    def styles_dir(self) -> Path:
        return self.home / "synopses"

    @property
    def glossary_path(self) -> Path:
        return self.home / "glossary.txt"

    def compute_type(self) -> str:
        return self.asr_compute_type or ("float16" if self.device == "cuda" else "int8")


def available_styles(styles_dir: Path) -> list[str]:
    return sorted(p.stem for p in styles_dir.glob("*.txt")) if styles_dir.is_dir() else []
