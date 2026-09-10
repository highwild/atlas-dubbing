"""Prompt building, parsing and budget logic for document-level translation. Pure.

Nothing here talks to a model, so all of it is unit-tested directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Bump when prompt wording changes, so cached translations are invalidated.
# 2: hints (term = translation) section; back-translation verification.
PROMPT_VERSION = 2

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


@dataclass
class Hint:
    """A ``term = translation`` pair from ``hints.txt``: how to translate one domain
    term, as opposed to the glossary's do-not-translate-at-all."""

    term: str
    translation: str


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
                  glossary: list[str], hints: list[Hint] | None = None) -> str:
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
    if hints:
        parts += ["",
                  f"Use these specific {tgt} renderings whenever the term appears (or is "
                  "clearly implied by it). Match the inflected form the sentence needs:"]
        parts += [f"- {h.term} => {h.translation}" for h in hints]
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


def repeat_retry_prompt(line: Line, prev_text: str | None, next_text: str | None,
                        previous_reply: str, source_lang: str, target_lang: str) -> str:
    """Ask again for a line that came back the same as the line before it.

    Seen on hydro.wav in four languages: line 44 was given line 45's text ("test track")
    and line 45 was given it again, so line 44's own source — "completely out of action, a
    fantastic depot" — was never spoken in any of those tracks. The model tracks the
    numbered lines loosely when they are sentence fragments and catches up by repeating
    one, which loses the line it skipped. Naming both lines is the whole fix: this line,
    this source, not the one above it.
    """
    tgt, src = lang_name(target_lang), lang_name(source_lang)
    parts = [
        (f"Line {line.n} was answered with the same text as the line before it, but the two "
         f"lines are different {src} sentences."),
        "",
        f"the reply for line {line.n} was: {previous_reply.strip()}",
        "",
        (f"That belongs to the line before. Line {line.n} is this, and it needs its own "
         f"{tgt} translation:"),
        f"  {line.text.strip()}",
        "",
    ]
    if prev_text:
        parts.append(f"For reference, the line before ({src}): {prev_text.strip()}")
    if next_text:
        parts.append(f"And the line after ({src}): {next_text.strip()}")
    parts += ["",
              (f"Translate line {line.n} only. Do not repeat the line before it. Budget: at "
               f"most {line.budget} characters."),
              "",
              'Reply with JSON only: {"lines":[{"n":' + str(line.n)
              + ',"text":"..."}]}']
    return "\n".join(parts)


def fragment_retry_prompt(line: Line, prev_text: str | None, next_text: str | None,
                          source_lang: str, target_lang: str) -> str:
    """Ask again for a line that came back as the source written out.

    The lines this exists for are transcribe fragments — the words of one sentence split
    across two or three segments by a pause. A fragment on its own has no meaning to
    translate, and the model's safe answer is to hand the word back unchanged ("would",
    "okay i"), which is what ended up spoken inside the French and Spanish tracks. Naming
    the situation is the whole point of the prompt: the neighbouring lines are shown so the
    sentence is visible, and the reply is still just this line's part of it.
    """
    tgt, src = lang_name(target_lang), lang_name(source_lang)
    parts = [
        (f"Line {line.n} is a fragment: the {src} sentence it belongs to was split across "
         "several lines when the video was transcribed, so on its own it is not a sentence "
         "and its words do not stand alone."),
        ("The neighbouring lines are shown for context only. Return the part of the sentence "
         f"that line {line.n} carries, translated into {tgt}, in the same position: it must "
         "read correctly when spoken straight after the line before it and straight before "
         "the line after it."),
        ("Do not return the source text unchanged. Do not translate the neighbouring lines "
         "and do not repeat their words here."),
        "",
    ]
    if prev_text:
        parts.append(f"line before ({src}, already translated): {prev_text.strip()}")
    parts.append(f"line {line.n} to translate now ({src}): {line.text.strip()}")
    if next_text:
        parts.append(f"line after ({src}, not your job): {next_text.strip()}")
    parts += ["",
              (f"Budget for line {line.n}: at most {line.budget} characters, and it should "
               "stay as short as the source fragment it stands for."),
              "",
              'Reply with JSON only: {"lines":[{"n":' + str(line.n)
              + ',"text":"..."}]}']
    return "\n".join(parts)


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


class TruncatedAnswer(ParseError):
    """The answer ran into ``num_predict`` and was cut off mid-JSON.

    Separate from a plain :class:`ParseError` because it is the one failure that more room
    fixes: the answer was not wrong, it was unfinished. A malformed object, a missing line
    or an echo is a bad answer at any size.
    """

    def __init__(self, message: str, budget: int | None = None) -> None:
        super().__init__(message)
        self.budget = budget


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


def load_hints(text: str) -> list[Hint]:
    """``term = translation`` per line; ``#`` starts a comment line.

    The term may contain spaces (``taste buds = kubki smakowe``). A line without an
    ``=``, or with an empty side, is not a hint — it is skipped rather than guessed at,
    so a typo costs one entry instead of corrupting the prompt.
    """
    hints: list[Hint] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        term, _, translation = line.partition("=")
        term, translation = term.strip(), translation.strip()
        if term and translation:
            hints.append(Hint(term, translation))
    return hints


def merge_hints(base: list[Hint], override: list[Hint]) -> list[Hint]:
    """``override`` wins on the same term (case-insensitively), so a per-language file
    can correct the shared one. Ordered by term, so the prompt (and its cache key) does
    not depend on the order the files happen to list entries in."""
    merged: dict[str, Hint] = {}
    for hint in (*base, *override):
        merged[hint.term.casefold()] = hint
    return sorted(merged.values(), key=lambda h: h.term.casefold())


def load_style(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()
                     if not line.strip().startswith("#")).strip()
