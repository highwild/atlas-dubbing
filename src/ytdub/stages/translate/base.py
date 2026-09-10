"""Translator interface.

Only the Ollama backend ships, but the pipeline talks to this interface alone, so
another engine can be dropped in with ``YTDUB_TRANSLATOR=module.path:ClassName``
(licence permitting). A backend is constructed as ``cls(settings, style_text,
glossary)`` and must provide:
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ytdub.plugins import load_class

if TYPE_CHECKING:
    from ytdub.config import Settings
    from ytdub.stages.translate.prompt import Hint, Line

BUILTIN = {"ollama": "ytdub.stages.translate.ollama:OllamaTranslator"}


class Translator(Protocol):
    name: str

    def cache_identity(self) -> dict:
        """Everything about this backend that changes its output (model, options, a hash
        of its code). Goes into the translation cache key."""

    def weights_fingerprint(self) -> str | None:
        """Identifies the model *weights* actually served (e.g. an Ollama digest), or
        None if unknown. A cached translation made with different weights is redone."""

    def translate(self, lines: list[Line], *, source_lang: str,
                  target_lang: str) -> tuple[list[str], dict]:
        """One translation per line, in order, never fewer; plus a stats dict."""

    def set_hints(self, hints: list[Hint]) -> None:
        """Optional. Called with the target language's ``hints.txt`` pairs just before
        that language is translated; a backend without this method is still valid and
        simply logs that the hints were ignored. The same list is in the language's
        cache key, so leaving this out cannot cause a stale cache hit."""

    def unload(self) -> None:
        """Free VRAM before synthesis. Called once, after all languages."""


def get_translator(settings: Settings, *, style_text: str, glossary: list[str]) -> Translator:
    cls = load_class(settings.translator, BUILTIN, "translator")
    return cls(settings, style_text, glossary)
