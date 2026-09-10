"""Chatterbox Multilingual (Resemble AI, MIT) voice-cloning backend."""

from __future__ import annotations

import os
from pathlib import Path

from ytdub.logging import stage_logger

log = stage_logger("tts")

# Mirrors chatterbox.mtl_tts.SUPPORTED_LANGUAGES (0.1.7) so languages can be validated
# before hours of work, without importing torch.
SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}


def _import_chatterbox():
    """Import Chatterbox, surfacing the failures that are otherwise silent.

    ``perth`` (the watermarker) imports ``pkg_resources``; setuptools 81+ removed it,
    perth swallows the ImportError and sets ``PerthImplicitWatermarker = None``, and
    every generate call then dies with "'NoneType' object is not callable". Check it
    here and say what is actually wrong.
    """
    # Chatterbox's per-step tqdm bar counts tokens against a 1000-token ceiling a
    # sentence never reaches and reads as "stuck". Our own per-line progress replaces it.
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        import perth
    except ImportError as exc:
        raise ImportError(f"resemble-perth (Chatterbox watermarker) failed to import: {exc}") from exc
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        try:
            import pkg_resources  # noqa: F401
            cause = "unknown (pkg_resources imports fine)"
        except ImportError as exc:
            cause = f"pkg_resources is missing ({exc}); install 'setuptools<80'"
        raise ImportError(f"perth.PerthImplicitWatermarker is None. Cause: {cause}")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS


class ChatterboxBackend:
    name = "chatterbox-multilingual"
    supported_languages = SUPPORTED_LANGUAGES

    def __init__(self, settings) -> None:
        self.device = "cuda" if settings.device == "cuda" else "cpu"
        self.exaggeration = settings.tts_exaggeration
        self.cfg_weight = settings.tts_cfg_weight
        self.temperature = settings.tts_temperature
        self._model = None

    def cache_identity(self) -> dict:
        from importlib.metadata import PackageNotFoundError, version

        try:
            pkg = version("chatterbox-tts")
        except PackageNotFoundError:
            pkg = "unknown"
        return {"model": "ChatterboxMultilingualTTS", "chatterbox_tts": pkg,
                "exaggeration": self.exaggeration, "cfg_weight": self.cfg_weight,
                "temperature": self.temperature}

    def _get_model(self):
        if self._model is None:
            cls = _import_chatterbox()
            log.info(f"Loading Chatterbox Multilingual on {self.device}")
            self._model = cls.from_pretrained(device=self.device)
        return self._model

    def synthesize(self, text: str, ref: Path, language: str, out_path: Path, seed: int) -> Path:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Chatterbox does not support {language!r}")
        import torch

        from ytdub.audio import write_wav

        model = self._get_model()
        torch.manual_seed(seed)
        wav = model.generate(
            text, language_id=language, audio_prompt_path=str(ref),
            exaggeration=self.exaggeration, cfg_weight=self.cfg_weight,
            temperature=self.temperature,
        )
        write_wav(out_path, wav.squeeze(0).detach().cpu().numpy(), model.sr, subtype="FLOAT")
        return out_path

    def unload(self) -> None:
        from ytdub.gpu import free_memory

        self._model = None
        free_memory()
