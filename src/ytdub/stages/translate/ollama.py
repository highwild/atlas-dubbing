"""Ollama backend — local LLM translation with glossary, synopsis + rolling context.

Unlike NLLB, an instruction-following LLM can be *told* what not to translate
(brand names, handles, in-game terms), can be given a synopsis of what the video
actually is, and can see what was said just before. That fixes the things NLLB
gets wrong on conversational content:

  * proper nouns getting literally translated ("Rec Room" -> "recreation room")
  * bar-for-bar renderings that are grammatical but not what a native would say
  * the opening line landing badly because nothing has established the register

Enabled with ``--translator ollama``. Requires Ollama running locally with a
model pulled (default ``qwen3:8b``).

Config via env vars:
  YTDUB_OLLAMA_URL     default http://localhost:11434
  YTDUB_OLLAMA_MODEL   default qwen3:8b
  YTDUB_GLOSSARY       default <repo>/glossary.txt
  YTDUB_SYNOPSIS       default <repo>/synopsis.txt
                       (can also be the synopsis text itself rather than a path)
  YTDUB_OLLAMA_UNLOAD  set to 1 to unload the model between calls (saves VRAM,
                       costs a few seconds per segment)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

from ytdub.logging import stage_logger

log = stage_logger("translate")

_URL = os.environ.get("YTDUB_OLLAMA_URL", "http://localhost:11434")
_MODEL = os.environ.get("YTDUB_OLLAMA_MODEL", "qwen3:8b")
_UNLOAD = os.environ.get("YTDUB_OLLAMA_UNLOAD", "").strip() in ("1", "true", "yes")

# How many previous segments to show the model as context.
_CONTEXT_WINDOW = 6

_LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "pl": "Polish", "nl": "Dutch", "hi": "Hindi", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese (Simplified)", "ar": "Arabic", "tr": "Turkish", "uk": "Ukrainian",
}


def _repo_root() -> Path:
    # .../src/ytdub/stages/translate/ollama.py -> repo root
    return Path(__file__).resolve().parents[4]


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


@lru_cache(maxsize=1)
def _load_glossary() -> list[str]:
    """Read the do-not-translate list. One term per line, # for comments."""
    path = Path(os.environ.get("YTDUB_GLOSSARY", _repo_root() / "glossary.txt"))
    if not path.exists():
        log.info(f"No glossary at {path} (continuing without one)")
        return []
    terms = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    log.info(f"Glossary: {len(terms)} protected term(s)")
    return terms


@lru_cache(maxsize=1)
def _load_synopsis() -> str:
    """Read a short description of what this video is.

    YTDUB_SYNOPSIS may be either a path to a file or the synopsis text itself.
    Falls back to <repo>/synopsis.txt.
    """
    raw = os.environ.get("YTDUB_SYNOPSIS", "").strip()

    if raw:
        candidate = Path(raw)
        # treat as literal text if it isn't a path that exists
        if not candidate.exists():
            log.info("Synopsis: using inline text from YTDUB_SYNOPSIS")
            return raw
        path = candidate
    else:
        path = _repo_root() / "synopsis.txt"

    if not path.exists():
        log.info(f"No synopsis at {path} (continuing without one)")
        return ""

    text = "\n".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ).strip()

    if text:
        log.info(f"Synopsis: {len(text)} chars from {path.name}")
    return text


def _strip_thinking(text: str) -> str:
    """Qwen3 and friends may emit <think>...</think> — drop it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _clean(text: str) -> str:
    """Strip the wrapper junk LLMs add even when told not to."""
    text = _strip_thinking(text)
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    if len(text) > 1 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    text = re.sub(r"^(translation|translated text|output)\s*:\s*", "", text, flags=re.I)
    return text.strip()


class OllamaTranslator:
    """Translate one segment at a time, with synopsis, glossary + recent context."""

    def __init__(self) -> None:
        self._history: list[tuple[str, str]] = []
        self._logged_model = False

    def _post(self, prompt: str, system: str) -> str:
        payload = {
            "model": _MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            },
        }
        if _UNLOAD:
            payload["keep_alive"] = 0

        req = urllib.request.Request(
            f"{_URL}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")

    def _system_prompt(self, source_lang: str, target_lang: str) -> str:
        src = _lang_name(source_lang)
        tgt = _lang_name(target_lang)
        glossary = _load_glossary()
        synopsis = _load_synopsis()

        rules = [
            f"You are a professional subtitle translator working from {src} into {tgt}.",
            "You are translating spoken dialogue from a YouTube video.",
        ]

        if synopsis:
            rules += ["", "About this video:", synopsis]

        rules += [
            "",
            "Rules:",
            f"- Output ONLY the {tgt} translation. No notes, no quotes, no explanation.",
            "- Translate the MEANING, not the words. Ask yourself how a native "
            f"{tgt} speaker would express this same idea out loud, and write THAT. "
            "A literal word-for-word rendering is wrong even when it is grammatical.",
            "- English idioms, filler and slang must become the natural equivalent "
            f"in {tgt}, not a literal translation. If there is no equivalent, "
            "rewrite the line so it sounds normal rather than translating it directly.",
            "- Keep the casual, spoken register of the original. This is a person "
            "talking to camera, not a formal document. Contractions and informality "
            "are correct here.",
            "- Never translate proper nouns: names of people, channels, games, "
            "brands, in-game currencies or features. Leave them exactly as written.",
            "- If a line is a fragment or trails off, translate it as a fragment. "
            "Do not invent a complete sentence or add content that is not there.",
        ]

        if glossary:
            rules += ["", "NEVER translate these terms, reproduce them verbatim:"]
            rules += [f"  {term}" for term in glossary]

        return "\n".join(rules)

    def translate(
        self, text: str, source_lang: str, target_lang: str, max_chars: int | None = None
    ) -> str:
        if source_lang == target_lang or not text.strip():
            return text

        if not self._logged_model:
            log.info(f"Ollama translation via {_MODEL} at {_URL}")
            self._logged_model = True

        parts = []

        if self._history:
            recent = self._history[-_CONTEXT_WINDOW:]
            parts.append("Earlier lines from this same video, for context only:")
            for src_line, tgt_line in recent:
                parts.append(f"  {src_line}  ->  {tgt_line}")
            parts.append("")
        else:
            parts.append(
                "This is the opening line of the video, so it sets the tone. "
                "Make it sound like a natural, natural-sounding opener in the "
                "target language."
            )
            parts.append("")

        parts.append("Now translate this line:")
        parts.append(text.strip())

        if max_chars:
            parts.append("")
            parts.append(
                f"Aim for roughly {max_chars} characters or fewer so it fits the "
                "original timing, but never drop meaning to hit that."
            )

        prompt = "\n".join(parts)
        system = self._system_prompt(source_lang, target_lang)

        try:
            raw = self._post(prompt, system)
        except Exception as exc:
            log.error(f"Ollama request failed ({exc}); falling back to source text")
            return text

        result = _clean(raw)
        if not result:
            return text

        self._history.append((text.strip(), result))
        return result
