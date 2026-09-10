"""Spoken-number expansion: digits to words, for the TTS input only. Pure.

The prompt asks for numbers written out as words and the models ignore it — "4.5",
"7.8", "12 mil" and "9 out of 10" all come through as digits, on qwen3:8b and on a 27B
model alike. That matters because the text is *spoken*: multilingual TTS reads digits in
the wrong grammatical case, treats them as dates, or skips them, and nothing in the
review SRT shows it. So this is done deterministically, after translation and before
synthesis, with no model call.

Two rules shape everything here:

* **The SRT keeps its digits.** Reviewers scan "7.8" faster than "siedem przecinek
  osiem", and the subtitle track should read the way the number is written. This is a
  transform on the way *into* the synthesizer, applied at the TTS boundary in
  ``pipeline.py``; it never touches the translated text or the review file.
* **When in doubt, leave it alone.** A confidently wrong number read aloud is worse than
  a digit the TTS might handle, because the SRT will not show it. Every rule below is a
  reason to skip: times, versions, dates, ranges, codes, units, anything inside a
  protected term.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ytdub.logging import stage_logger

log = stage_logger("numbers")

# num2words has no Hindi converter (checked on 0.5.14: every call raises
# NotImplementedError). Hindi is a default target language, so a missing converter is a
# normal outcome to report, not a bug: that language keeps its digits and says so.
SUPPORTED = frozenset({"pl", "de", "fr", "es", "nl", "en", "it", "pt", "ru", "uk", "sv",
                       "da", "no", "fi", "cs", "sk", "tr", "ro", "hu", "lt", "lv", "id",
                       "vi", "th", "ja", "ko"})

# A year is a 4-digit number in this range with a date cue next to it, before or after.
# The range is what decides it: without a cue, a bare 4-digit number is more likely a
# quantity ("2000 people"), and a quantity must not be read as a year.
YEAR_MIN, YEAR_MAX = 1100, 2199
CUE_WINDOW = 4  # words either side of the number to look for a date cue

_YEAR_CUES = {
    "en": {"in", "year", "years", "since", "until", "by", "back", "during"},
    "pl": {"w", "we", "roku", "rok", "lat", "od", "do", "przez", "podczas"},
    "de": {"im", "in", "jahr", "jahre", "jahres", "seit", "bis", "während"},
    "fr": {"en", "année", "an", "depuis", "jusqu", "pendant"},
    "es": {"en", "año", "años", "desde", "hasta", "durante"},
    "nl": {"in", "jaar", "sinds", "tot", "gedurende"},
}

# A code, not a quantity. Three shapes: a *known* unit next to the number ("5km",
# "12 GB"), letters glued to the digits ("4K", "10x", "1080p", "3rd"), or a model or
# product number (two or more capitals before the digits: "RTX 3080", "PS 5").
#
# Case-insensitivity is scoped to the unit list on purpose. Applied to the whole pattern
# it made the capital-letter run match ordinary words and swallowed the very numbers this
# exists for ("9 na 10", "4,5, bo"). Letters only count as glued when they actually
# touch: a *space* after the number is just the next word ("3 filmy"), and in French and
# Spanish that next word is often one letter ("9 à 10").
_UNITS = ("km", "kg", "cm", "mm", "ml", "gb", "mb", "kb", "tb", "fps", "hz", "khz",
          "mph", "kw", "kwh", "mah", "dpi", "ppi", "nm", "oz", "lb", "ft", "psi", "rpm",
          "ms", "px", "gbp", "usd", "eur", "pln", "zł", "°c", "°f")
# Matched case-insensitively ("12 GB", "5 km") and allowed to be spaced, because a unit
# written that way is unambiguous. Single-letter units are deliberately absent: the glued
# rule below covers "1080p" and "3m" without it, and a lone "i" or "p" with a space in
# front of it is a word — it is the Polish "and" in "4,5 i 9", and a case-insensitive
# single letter would have kept that line's numbers as digits.
# Each rule is anchored so it can only match text *touching* the number it is judging.
# That matters more than it looks: an unanchored scan over a window lets a designation
# earlier in the sentence mark a later, unrelated number — "Class 08 does 9 miles" was
# keeping the 9 as a digit because of the 08.
_CODE_BEFORE = re.compile(
    # A unit, spaced or glued: "12 GB", "5 km", "100%".
    r"(?:(?i:" + "|".join(_UNITS) + r")[-\u00a0 ]?$"
    # An all-caps run before the digits: "RTX 3080", "PS 5". Anything looser swallows
    # ordinary words — a case-insensitive single capital matches "W 2026 roku", "o 12 mil"
    # and "9 à 10" and would leave unexpanded the very numbers this pass exists for.
    r"|(?<![A-Za-zÀ-ÿ0-9])[A-ZÀ-Þ]{2,}[-\u00a0 ]$"
    # A class or model designation, in any case and inflected: "Class 08", "klasy 08",
    # "Typ 4", "Route 66". The number is part of the name — nobody says "Class huit" —
    # and the translation inflects the word, so the stems cover the pl/de/fr/es forms.
    r"|(?<![A-Za-zÀ-ÿ0-9])(?i:class\w*|klasse\w*|clase[s]?|klas\w*|type\w*|typ\w*"
    r"|series?|serie\w*|route|model\w*|mark\w*|variant\w*|wersj\w*|prototyp\w*)"
    r"[-\u00a0 ]$"
    # A single letter glued straight on, as in "1080p" or "3m": only counts with no gap.
    r"|(?<=\d)[A-Za-zÀ-ÿ]$)"
)
_CODE_AFTER = re.compile(
    # Letters glued onto the digits: "4K", "3rd", "1080p", "100%". The boundary test is
    # needed on the far side too, or "1080p" is read as the number 1080 followed by a word.
    r"^(?:[xX%°]|[A-Za-zÀ-ÿ]{2,4}(?![A-Za-zÀ-ÿ0-9])|[A-Za-zÀ-ÿ](?![A-Za-zÀ-ÿ0-9]))"
)

# A number with a separator and digits on *both* sides is part of a longer dotted run —
# a version or an IP, never one spoken number. The same test on either side with the
# match's own separators included catches "4.5.1" from the "4.5" match, and the whole
# chain is rejected, not the first piece of it.
_VERSION_BEFORE = re.compile(r"\d[.,]$")
_VERSION_AFTER = re.compile(r"[.,]\d")
# "7:30", and the 30 in it, are a clock time: neither piece is a quantity.
_TIME_BEFORE = re.compile(r"\d:\s*$")
_TIME_AFTER = re.compile(r"^\s*:\s*\d")


@dataclass
class Change:
    """One expansion, for the debug log: without it a wrong number is only findable by
    listening to the finished dub."""

    kind: str  # "integer" | "decimal" | "year"
    digits: str
    words: str
    start: int
    end: int


def expansion_pair(digits: str, lang: str, *, year: bool = False) -> str | None:
    """``digits`` as spoken words in ``lang``, or None when it cannot be done well.

    None covers the cases worth refusing: no converter for the language, a value
    num2words rejects, or (for a year) a result identical to the plain cardinal, which
    would mean the year form adds nothing and the cardinal is the safer read.
    """
    if lang not in SUPPORTED:
        return None
    if "," in digits:
        digits = digits.replace(",", ".")
    num2words = _num2words()
    try:
        if "." in digits:
            # A decimal is read as "seven point eight" / "siedem przecinek osiem"; the
            # separator word for the target language is num2words' business.
            return num2words(float(digits), lang=lang)
        value = int(digits)
        if year:
            spoken = num2words(value, lang=lang, to="year")
            return spoken or None
        return num2words(value, lang=lang)
    except (NotImplementedError, TypeError, ValueError, OverflowError, KeyError) as exc:
        log.debug(f"numbers: {digits!r} not expandable in {lang}: "
                  f"{type(exc).__name__}")
        return None


def _num2words():
    """Imported lazily so the pure-stages tests and any job without number expansion do
    not need the dependency present."""
    from num2words import num2words

    return num2words


# --- protection -----------------------------------------------------------------


def protected_spans(text: str, terms: list[str]) -> list[tuple[int, int]]:
    """Character spans of the protected terms present in ``text``.

    Glossary and hint terms are copied into the prompt and appear verbatim in the
    translation ("Rec Room Tokens", "PlayStation 5"), so a digit inside one is part of
    the name and must survive untouched.
    """
    spans: list[tuple[int, int]] = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        spans += [(m.start(), m.end()) for m in pattern.finditer(text)]
    return spans


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and span_start < end for span_start, span_end in spans)


# --- candidates -------------------------------------------------------------------


# --- case, where a language needs it ------------------------------------------------
#
# num2words only ever produces the nominative cardinal. For German, French, Spanish and
# Dutch that is the form used everywhere ("fünf Bücher", "cinq livres"), so nothing more
# is needed. Polish is the exception that matters here: after a preposition the number
# must be in the genitive, and "z dziesięć brązowych liści" — literally "out of ten
# brown leaves" — is the kind of confidently wrong read this pass exists to prevent. So
# the few prepositions that trigger it are handled explicitly, and a number whose
# genitive form is not in the table is *left as a digit* rather than spoken wrongly.
_PL_GENITIVE = {
    0: "zera", 1: "jednego", 2: "dwóch", 3: "trzech", 4: "czterech", 5: "pięciu",
    6: "sześciu", 7: "siedmiu", 8: "ośmiu", 9: "dziewięciu", 10: "dziesięciu",
    11: "jedenastu", 12: "dwunastu", 13: "trzynastu", 14: "czternastu", 15: "piętnastu",
    16: "szesnastu", 17: "siedemnastu", 18: "osiemnastu", 19: "dziewiętnastu",
    20: "dwudziestu", 30: "trzydziestu", 40: "czterdziestu", 50: "pięćdziesięciu",
    60: "sześćdziesięciu", 70: "siedemdziesięciu", 80: "osiemdziesięciu",
    90: "dziewięćdziesięciu", 100: "stu", 200: "dwustu", 300: "trzystu",
    400: "czterystu", 500: "pięciuset", 600: "sześciuset", 700: "siedmiuset",
    800: "ośmiuset", 900: "dziewięciuset",
}
# The prepositions that take the genitive in Polish. "z" is the one that appears in this
# content: "9 z 10" is a rating, and it is spoken "dziewięć z dziesięciu".
_PL_GENITIVE_TRIGGERS = {"z", "ze", "od", "do", "około", "ponad", "poniżej", "powyżej",
                         "więcej", "mniej", "wśród", "dla", "bez"}


def _pl_genitive(value: int) -> str | None:
    """The Polish genitive of ``value``, or None above 999 where it is not worth
    guessing. Both parts inflect independently ("dwadzieścia pięć" -> "dwudziestu
    pięciu"), which the small tables handle; thousands and above reintroduce the very
    guesswork this is meant to avoid, so those stay digits."""
    if value < 0 or value > 999:
        return None
    units = {n: word for n, word in _PL_GENITIVE.items() if n < 20 or n % 10 == 0}
    if value < 100:
        if value < 20:
            return units.get(value)
        # The tens inflect and the unit inflects with them: dwudziestu pięciu. A zero
        # unit is not spoken, which is what turns "dwadzieścia" plus "zero" into
        # "dwudziestu".
        tens, rest = divmod(value, 10)
        head = units.get(tens * 10)
        if head is None:
            return None
        return head if not rest else f"{head} {units.get(rest)}"
    hundreds, rest = divmod(value, 100)
    head = _PL_GENITIVE.get(hundreds * 100)
    if head is None:
        return None
    if not rest:
        return head
    tail = _pl_genitive(rest)
    return None if tail is None else f"{head} {tail}"


def _words(digits: str, lang: str, *, year: bool, preceded_by: str) -> str | None:
    """The spoken form, or None when it cannot be said correctly."""
    if lang == "pl" and not year and digits.isdigit() \
            and preceded_by.casefold() in _PL_GENITIVE_TRIGGERS:
        return _pl_genitive(int(digits))
    return expansion_pair(digits, lang, year=year)


def _trigger_before(text: str, start: int) -> str:
    """The word just before the number, which decides the case in Polish."""
    before = text[:start].rstrip()
    match = re.search(r"([^\W\d_]+)$", before, re.UNICODE)
    return match.group(1) if match else ""


def _is_year(text: str, start: int, end: int, lang: str) -> bool:
    """Is this 4-digit number a year here? Decided by a date cue on either side: Polish
    puts the cue after ("2019 roku"), English before ("in 2019")."""
    cues = _YEAR_CUES.get(lang, set())
    if not cues:
        return False
    before = re.findall(r"[\w'’-]+", text[:start].lower())[-CUE_WINDOW:]
    after = re.findall(r"[\w'’-]+", text[end:].lower())[:CUE_WINDOW]
    return any(word in cues for word in before + after)


def _skipped(text: str, start: int, end: int) -> str | None:
    """Why this number must stay a digit, or None when it may be expanded."""
    before, after = text[max(0, start - 8):start], text[end:end + 8]
    if _VERSION_BEFORE.search(before) or _VERSION_AFTER.match(after):
        return "version or date"  # 4.5.1, 192.168.0.1, 10.06.2024
    if _TIME_BEFORE.search(before) or _TIME_AFTER.match(after):
        return "clock time"  # 7:30
    if _CODE_BEFORE.search(text[max(0, start - 24):start]) or _CODE_AFTER.match(text[end:]):
        return "unit or code"  # 5km, 4K, 10x, 12GB, 100%, RTX 3080, Class 08
    if "-" in text[max(0, start - 1):end + 1] or "–" in text[max(0, start - 1):end + 1]:
        return "range"  # 3-4, 2020-2024
    return None


def _candidates(text: str, lang: str) -> list[Change]:
    found: list[Change] = []
    for match in re.finditer(r"\d+(?:[.,]\d+)?", text):
        start, end = match.span()
        digits = match.group()
        if digits[-1] in ".," or digits.count(".") + digits.count(",") > 1:
            continue  # trailing separator, or a version/IP with several
        if _skipped(text, start, end):
            continue
        year = False
        if len(digits) == 4 and digits.isdigit() and YEAR_MIN <= int(digits) <= YEAR_MAX:
            # In the plausible-year range the *cue* decides how it is read: with one,
            # the year form ("nineteen eighty-five" rather than "one thousand nine
            # hundred and eighty-five"); without one it is an ordinary quantity, and an
            # ordinary quantity is still written out.
            year = _is_year(text, start, end, lang)
        kind = "decimal" if ("." in digits or "," in digits) else (
            "year" if year else "integer")
        found.append(Change(kind=kind, digits=digits, words="", start=start, end=end))
    return found


# --- the transform ----------------------------------------------------------------


def expand_text(text: str, lang: str, *, protected: list[str] | None = None) -> tuple[str, list[Change]]:
    """``(text to speak, changes)``. ``text`` unchanged when nothing is expandable.

    Runs on the translated line on its way to the synthesizer, so the return value is
    TTS input only: the caller keeps the digit version for the SRT.
    """
    if not text or not re.search(r"\d", text) or lang not in SUPPORTED:
        return text, []
    spans = protected_spans(text, protected or [])
    out: list[str] = []
    changes: list[Change] = []
    cursor = 0
    for change in _candidates(text, lang):
        if _overlaps(change.start, change.end, spans):
            continue
        words = _words(change.digits, lang, year=change.kind == "year",
                       preceded_by=_trigger_before(text, change.start))
        if words is None:
            continue  # no form that is known to be right: the digit stays
        change.words = words
        out.append(text[cursor:change.start])
        out.append(words)
        cursor = change.end
        changes.append(change)
    if not changes:
        return text, []
    out.append(text[cursor:])
    return "".join(out), changes
