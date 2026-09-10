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


def _env_files() -> tuple[str, ...]:
    """``.env`` from the project root, then the current directory.

    The project root matters: ``dub2 hydro.wav`` typed from anywhere else must still find
    the voice clip and the Hugging Face token, and a run that silently loses both is a run
    that clones the wrong voices. A ``.env`` in the working directory still wins, so a
    one-off override does not have to be written into the checkout.
    """
    return (str(_default_home() / ".env"), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YTDUB_", env_file=_env_files(),
                                      extra="ignore")

    # --- Layout -------------------------------------------------------------
    home: Path = Field(default_factory=_default_home)

    # --- Job ----------------------------------------------------------------
    languages: list[str] = Field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    style: str = "casual"
    source_lang: str | None = None  # None = let Whisper detect it
    # Multi-voice by default: 0 = let the diarizer count the voices. That is what makes
    # `dub file.wav` work with no flags on a file you have never heard — a single-speaker
    # file comes back as one label and behaves exactly like single-voice mode. Set 1 to
    # skip diarization's counting, or N to force a count the model got wrong.
    speakers: int | None = 0
    # A clip of your own voice, matched to whichever detected speaker sounds like it, so
    # the label the diarizer invented for you this time does not have to be named. A file
    # or a folder of files (see stages/voices.py). Needs multi-voice mode to match against.
    voice: Path | None = None
    # Minimum similarity to accept a match. Measured against the pipeline's own reference
    # cuts: the presenter's own clip scores 0.99, every other speaker 0.54-0.66, so this
    # sits in the gap rather than on either side's edge.
    voice_match_threshold: float = 0.75
    # Manual correction of diarization boundaries, "first-last=SPK,first-last=SPK"
    # (1-based SRT line numbers). For a stretch the model merged into the wrong speaker.
    speaker_map: str = ""
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
    # "pyannote" is the better model and the default, but it needs a free HF token with
    # the terms accepted at hf.co/pyannote/speaker-diarization-3.1; set it in .env as
    # YTDUB_HF_TOKEN. Without one the run says so and falls back to "embedding", which is
    # token-free and, measured on this box, cannot separate two similar male voices.
    diarize_method: str = "pyannote"  # "pyannote" | "embedding" (token-free)
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

    # Back-translation verification (DUAL-REFLECT): translate, translate the result back
    # into the source language, and revise any line whose meaning changed on the round
    # trip. Catches fluent-but-wrong output that reading the target language cannot.
    # Off by default: it multiplies translation requests by roughly three.
    verify: bool = False
    verify_batch_lines: int = 20  # lines per back-translation / comparison request
    # Verification is a check, not a translation, and it is affordable to use a bigger
    # model for it than the translator uses: the models are never resident together
    # (translation is unloaded before synthesis, verification runs inside translation).
    # Empty = verify with the translation model, which is the default.
    verify_ollama_model: str = ""
    verify_temperature: float = 0.8  # the second attempt, for a line that was echoed
    verify_echo_retries: int = 1  # 0 disables that second attempt
    # Set by the pipeline per language, not from the environment: where the report of
    # this pass goes (output/<file>/<lang>.verify.txt). None disables the file.
    verify_report_path: Path | None = None

    # --- Synthesis ------------------------------------------------------------
    # "chatterbox" (shipped) or "module.path:ClassName" for a drop-in backend.
    tts_backend: str = "chatterbox"
    tts_exaggeration: float = 0.5
    tts_cfg_weight: float = 0.5
    tts_temperature: float = 0.8
    tts_seed: int = 1234
    # Chatterbox crashes on utterances this short (IndexError in its alignment
    # analyzer); such lines are merged into an adjacent same-speaker line first.
    # A line is a fragment — and gets merged into a same-speaker neighbour — when it is
    # shorter than this in spoken characters, or when it is a single word (see
    # stages/tts/base.py: one-word lines are what crash the synthesizer).
    min_tts_chars: int = 3
    merge_max_gap: float = 1.5  # only merge into a neighbour this close in time
    # Write digits out as words for the synthesizer only ("7,8" -> "siedem przecinek
    # osiem"). The SRT keeps the digits; see stages/numbers.py for what is left alone.
    expand_numbers: bool = True

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

    # --- Derived ----------------------------------------------------------------
    @property
    def multi_voice(self) -> bool:
        """True when the file is treated as more than one voice.

        ``None`` and ``1`` both mean one speaker: no diarization, one reference, one voice.
        The diarizer's count is the only thing that makes the difference, so expressing it
        once here keeps ``--speakers 1`` from meaning "run the diarizer and hope it says 1".
        """
        return self.speakers is not None and self.speakers != 1

    # --- Paths ------------------------------------------------------------------
    @property
    def input_dir(self) -> Path:
        return self.home / "input"

    @property
    def voice_path(self) -> Path | None:
        """``voice`` as an absolute path, relative ones resolved against the project root.

        ``YTDUB_VOICE=./voices/atlas.wav`` has to mean the checkout's ``voices/`` wherever
        the command was typed from, because the alternative is matching your own clip
        against nothing and quietly cloning the diarizer's automatic reference instead.
        """
        if self.voice is None:
            return None
        path = Path(self.voice).expanduser()
        return path if path.is_absolute() else (self.home / path).resolve()

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

    @property
    def hints_path(self) -> Path:
        """``hints.txt``: user-written ``term = translation`` pairs, applied to every
        target language."""
        return self.home / "hints.txt"

    def hints_path_for(self, lang: str) -> Path:
        """``hints.<lang>.txt``: the same, for one target language, overriding the
        shared file. Domain vocabulary usually needs a different rendering per language,
        so this is where "patty" and "taste buds" get pinned down."""
        return self.home / f"hints.{lang}.txt"

    def verify_report_path_for(self, lang: str, out_dir: Path) -> Path:
        return out_dir / f"{lang}.verify.txt"

    def compute_type(self) -> str:
        return self.asr_compute_type or ("float16" if self.device == "cuda" else "int8")


def available_styles(styles_dir: Path) -> list[str]:
    return sorted(p.stem for p in styles_dir.glob("*.txt")) if styles_dir.is_dir() else []
