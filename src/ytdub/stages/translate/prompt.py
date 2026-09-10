"""Prompt building, parsing and budget logic for document-level translation. Pure.

Nothing here talks to a model, so all of it is unit-tested directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Bump when prompt wording changes, so cached translations are invalidated.
PROMPT_VERSION = 1

LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "pl": "Polish",
    "nl": "Dutch", "hi": "Hindi", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Simplified)", "ar": "Arabic",
    "tr": "Turkish", "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
    "el": "Greek", "he": "Hebrew", "ms": "Malay", "sw": "Swahili",
}

# Used only when a style preset has no "Locale rules" section of its own.
DEFAULT_LOCALE_RULES = """Locale rules:
- Numbers: write numbers out as words, the way a native speaker would say them aloud.
  Never leave digits; the text is read by a speech synthesizer.
- Currency: keep the original currency and amount (pounds stay pounds, spelled out),
  and be consistent across the whole video.
- Dates: use the target language's natural spoken date order.
- Units: keep the original units unless they would be meaningless to the audience;
  if converting, round sensibly and do it consistently."""


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


@dataclass
class Line:
    n: int  # number shown to the model (1-based, unique within a language run)
    text: str  # source text
    budget: int  # character budget for the translation
    speaker: str | None = None


# --- Token estimation ----------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic token count (over-estimates, never under).

    Latin text runs ~4 characters per token in Qwen's tokenizer; we assume 3. Non-ASCII
    scripts (Devanagari, CJK, Cyrillic) can approach one token per character or worse,
    so each counts 1.2.
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return int(ascii_chars / 3.0 + (len(text) - ascii_chars) * 1.2) + 1


def lower_bound_tokens(text: str) -> int:
    """A count the real tokenizer can never go below (8+ chars/token doesn't happen)."""
    return len(text) // 8


def output_token_estimate(lines: list[Line], target_lang: str) -> int:
    """Pessimistic size of the JSON answer for ``lines``."""
    total = 0
    for line in lines:
        chars = max(line.budget, int(len(line.text) * 1.5))
        sample = ("x" if target_lang not in _NON_LATIN else "क") * chars
        total += estimate_tokens(sample) + 12  # {"n": 12, "text": "..."} overhead
    return total + 16


_NON_LATIN = {"hi", "zh", "ja", "ko", "ar", "ru", "uk", "el", "he", "th"}


# --- Prompts -------------------------------------------------------------------


def system_prompt(source_lang: str, target_lang: str, style_text: str,
                  glossary: list[str]) -> str:
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    parts = [
        f"You are a professional dubbing translator working from {src} into {tgt}.",
        "The text is spoken dialogue from a video. Your translation will be read aloud "
        "by a voice-cloning speech synthesizer and must fit the original timing.",
    ]
    if style_text.strip():
        parts += ["", "Video context:", style_text.strip()]
    if "locale rules" not in style_text.lower():
        parts += ["", DEFAULT_LOCALE_RULES]
    parts += [
        "",
        "Rules:",
        f"- Translate the meaning, not the words. Write what a native {tgt} speaker "
        "would actually say out loud in this situation. A literal rendering is wrong "
        "even when it is grammatical.",
        f"- Idioms, filler and slang become the natural {tgt} equivalent, never a "
        "literal translation.",
        "- Keep the register of the original: spoken, not written.",
        "- A fragment or trailing-off line stays a fragment. Never add content.",
        "- Never translate names of people, channels, games, brands, platforms or "
        "in-game currencies.",
        "- Lines tagged with a speaker label like [SPK1] are said by that person; use "
        "it for pronouns and grammatical agreement. Never output the tag.",
        "- Output only JSON in the requested shape. No notes or explanations.",
    ]
    if glossary:
        parts += ["", "Do not translate these terms. Reproduce them exactly as written:"]
        parts += [f"- {term}" for term in glossary]
    return "\n".join(parts)


def _fmt(line: Line, *, budget: bool = True) -> str:
    tag = f"[{line.speaker}] " if line.speaker else ""
    b = f"[≤{line.budget}] " if budget else ""
    return f"{line.n}. {b}{tag}{line.text.strip()}"


def batch_prompt(lines: list[Line], context: list[tuple[Line, str]],
                 lookahead: list[Line], target_lang: str) -> str:
    tgt = lang_name(target_lang)
    parts: list[str] = []
    if context:
        parts.append("Earlier lines, already translated (context only; do not return them):")
        parts += [f"{_fmt(line, budget=False)}  =>  {done}" for line, done in context]
        parts.append("")
    else:
        parts += ["These are the opening lines of the video.", ""]
    first, last = lines[0].n, lines[-1].n
    parts += [
        f"Translate lines {first}-{last} into {tgt}. Return every one of them with the "
        "same numbers.",
        "Each line has a character budget [≤N]: the most characters of translation "
        "that fit its time slot when spoken. Stay within it. You may move words between "
        "adjacent lines to help them fit, as long as meaning and order are preserved.",
        "",
    ]
    parts += [_fmt(line) for line in lines]
    if lookahead:
        parts += ["", "Following lines (context only; do not translate or return them):"]
        parts += [_fmt(line, budget=False) for line in lookahead]
    return "\n".join(parts)


def repair_prompt(window: list[tuple[Line, str]], too_long: set[int], source_lang: str,
                  target_lang: str) -> str:
    tgt = lang_name(target_lang)
    parts = [
        f"These {tgt} lines were translated from the {lang_name(source_lang)} source shown. "
        "Some are too long to be spoken in their time slot.",
        "Rewrite the lines marked TOO LONG so they fit their budget [≤N] characters. "
        "Shorten by choosing more concise natural phrasing, not by dropping meaning. "
        "You may move words into a neighbouring line that is under its budget, keeping "
        "the order of ideas. Return ALL lines listed, with the same numbers.",
        "",
    ]
    for line, current in window:
        flag = f"  TOO LONG ({len(current)} chars)" if line.n in too_long else ""
        parts.append(f"{line.n}. [≤{line.budget}] source: {line.text.strip()}")
        parts.append(f"    current: {current}{flag}")
    return "\n".join(parts)


def single_line_prompt(line: Line, context: list[tuple[Line, str]], target_lang: str) -> str:
    return batch_prompt([line], context, [], target_lang)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["n", "text"],
            },
        }
    },
    "required": ["lines"],
}


# --- Parsing -------------------------------------------------------------------


class ParseError(ValueError):
    pass


_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_TAG = re.compile(r"^\s*\[(?:SPK\d+|≤\s*\d+)\]\s*")


def strip_thinking(text: str) -> str:
    """Remove ``<think>`` blocks, even unterminated ones. Thinking is disabled per request,
    but a model or server version that ignores that must not corrupt the parse."""
    return _THINK.sub("", text).strip()


def parse_lines(raw: str, expected: list[int]) -> dict[int, str]:
    """Strictly parse a ``{"lines": [{"n", "text"}]}`` answer.

    Every expected number must appear exactly once with non-empty text; anything else
    is a :class:`ParseError` (the caller then falls back rather than losing lines).
    Echoed budget/speaker tags are stripped.
    """
    text = _FENCE.sub("", strip_thinking(raw)).strip()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ParseError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        raise ParseError("missing 'lines' array")
    out: dict[int, str] = {}
    for item in data["lines"]:
        if not isinstance(item, dict):
            raise ParseError("line entry is not an object")
        try:
            n = int(item["n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"bad line number in {item!r}") from exc
        value = item.get("text")
        if not isinstance(value, str):
            raise ParseError(f"line {n}: text is not a string")
        value = value.strip()
        while _TAG.match(value):
            value = _TAG.sub("", value, count=1)
        if n in out:
            raise ParseError(f"line {n} returned twice")
        out[n] = value
    missing = [n for n in expected if not out.get(n)]
    if missing:
        raise ParseError(f"missing or empty lines: {missing}")
    extra = sorted(set(out) - set(expected))
    if extra:
        raise ParseError(f"unexpected line numbers: {extra}")
    return out


# --- Budgets -------------------------------------------------------------------


def budget_chars(slot_seconds: float, chars_per_second: float) -> int:
    return max(8, int(slot_seconds * chars_per_second))


def overshoot(text: str, budget: int) -> int:
    return max(0, len(text) - budget)


def needs_repair(text: str, budget: int, tolerance: float) -> bool:
    """Materially over budget: beyond the tolerance ratio *and* by at least 5 chars."""
    return len(text) > budget * tolerance and len(text) - budget >= 5


def repair_groups(too_long: list[int], all_ns: list[int], max_group: int = 8) -> list[list[int]]:
    """Contiguous windows (each flagged line plus one neighbour either side), merged
    where they touch, so a line can hand words to a neighbour with slack."""
    valid = set(all_ns)
    wanted = sorted({m for n in too_long for m in (n - 1, n, n + 1) if m in valid})
    groups: list[list[int]] = []
    for n in wanted:
        if groups and n == groups[-1][-1] + 1 and len(groups[-1]) < max_group:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def glossary_misses(source: str, translation: str, glossary: list[str]) -> list[str]:
    """Protected terms present in the source line but absent from its translation."""
    misses = []
    for term in glossary:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.I)
        if pattern.search(source) and not pattern.search(translation):
            misses.append(term)
    return misses


def load_glossary(text: str) -> list[str]:
    """One term per line; ``#`` starts a comment line."""
    return [s for s in (line.strip() for line in text.splitlines())
            if s and not s.startswith("#")]


def load_style(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()
                     if not line.strip().startswith("#")).strip()
