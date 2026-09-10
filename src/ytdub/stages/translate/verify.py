"""Back-translation verification (DUAL-REFLECT, arXiv 2406.07232) for the Ollama
translator. Prompt building and parsing only; nothing here talks to a model.

A translation can be fluent, idiomatic and wrong: qwen3 renders "on the actual patty"
as ``na kiełbasiu`` ("on the sausage") and "taste buds" as ``zębki`` ("little teeth").
Reviewing the Polish cannot catch that — it reads perfectly well. The error only exists
*against the source*, so the text has to come back the other way to become visible.

The pass is three steps, each one its own request:

1. **Back-translate** the translation into the source language, asking for a *literal*
   rendering. Polishing is the failure mode to avoid here: a back-translation that
   reads nicely has already hidden the drift.
2. **Compare** each back-translation with the source line it came from, and report the
   lines whose *meaning* changed. Legitimate paraphrase is not drift, and neither is
   word order, so this is a judgement call and has to be the model's, not a string
   similarity score (which would flag every correct translation).
3. **Revise** the flagged lines, showing the model the source, its first attempt and
   what that attempt actually says. That "what it actually says" is the whole point: it
   is the evidence the model never gets to see when it reviews its own output.

Every answer uses the translator's numbered-line JSON shape, so it goes through the same
strict parser and the same context-window safety checks as translation.

Two failure modes shaped the prompts, and both are worth knowing before editing them:

* **A back-translation that copies its input.** Asked to render Polish into English the
  model sometimes returns the Polish. That is not weak evidence, it is inverted evidence:
  the comparison sees the round trip "match" the source and confirms a wrong translation
  as correct, and a comparison told to assume the translation is right when ambiguous
  agrees. The answering rules are blunt about it, the comparison is told to report a
  back-translation that is not in the source language, an identical line is refused in
  code, and such lines are retried once with different instructions.
* **A comparison that reports synonyms and paraphrase as changes.** Left to describe the
  difference, qwen3:8b flagged "bun" → "bread" and passed "taste buds" → "little teeth".
  The worked examples in the comparison prompt are what fixed it (5/8 to 12/12 on real
  lines), so they are load-bearing, not decoration.

The pass is a diagnostic first and a repair second: it names the terms worth pinning, and
the repairs themselves are low-yield and kept only when verified.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ytdub.stages.translate.prompt import (
    Line,
    ParseError,
    _fmt,
    lang_name,
    strip_thinking,
)

# A ```json fence around the whole answer, same cleanup the translation parser does.
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

# What a hintable term can be: a word or a short phrase, never a clause. See in_source().
MAX_TERM_CHARS = 40
MAX_TERM_WORDS = 4


def lines_schema() -> dict:
    """A fresh ``{"lines": [{"n", "text"}]}`` schema, the same shape as
    :data:`~ytdub.stages.translate.prompt.RESPONSE_SCHEMA` (built per call so nothing
    can mutate a shared schema dict)."""
    return {
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


COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["ok", "drift"]},
                    "problem": {"type": "string"},
                    "term": {"type": "string"},
                },
                "required": ["n", "verdict", "problem", "term"],
            },
        }
    },
    "required": ["lines"],
}


@dataclass
class Verdict:
    """What the comparison said about one line."""

    drift: bool
    problem: str = ""
    # The source word or short phrase that drifted, copied from the source. Empty when
    # the model did not name one, or (see :meth:`parsed`) when the name it gave is not
    # actually in the source, which is a hallucinated term and worse than no term.
    term: str = ""


def in_source(source: str, term: str) -> bool:
    """Is ``term`` really a word or phrase of ``source``, and a plausible *term*?

    Terms come back from a small model, so they are checked before being offered as a
    hint: a term that is not in the source would send the next run looking for something
    that is not there, and a term that is a whole clause is not a term at all — pinning
    half a sentence produces a translation nobody wants. The length and word limits are
    what keep 'taste buds' a hint and 'probably it's probably a 7.8 today because it's
    all right' out of the file.
    """
    term = term.strip()
    if len(term) < 2 or len(term) > MAX_TERM_CHARS or len(term.split()) > MAX_TERM_WORDS:
        return False
    return _mentions(source, term)


def _mentions(text: str, term: str) -> bool:
    """``term`` appears in ``text`` as a whole word or phrase (no length limits)."""
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    return bool(pattern.search(text))


# `/no_think` is a control token some qwen builds leak into the answer text.
_NO_THINK = re.compile(r"/no_think", re.I)


def clean_text(text: str) -> str:
    """Strip leaked thinking control tokens from model output. They would otherwise end
    up in the report, and in a hint suggestion, where they are pure noise."""
    return _NO_THINK.sub("", strip_thinking(text)).strip()


# --- Back-translation -----------------------------------------------------------


def backtranslate_system_prompt(source_lang: str, target_lang: str) -> str:
    """The literal back-translation prompt.

    The blunt wording about copying is not decoration. Measured on qwen3:8b over real
    lines, a softer wording ("a line left in Polish is a failed answer") came back with
    the input unchanged for 3 of 17 lines of one clip and, on an unlucky draw, for every
    line of a batch — and an unchanged line is not weak evidence, it is the *opposite*
    evidence: the comparison sees it match the source and confirms a wrong translation.
    The three cases named at the end are the ones that actually went wrong.
    """
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    return "\n".join([
        f"You are a literal back-translator from {tgt} into {src}.",
        f"Every line of output must be {src}. A line left in {tgt} is a failed answer, "
        "and so is a line copied from the input: copying is never a translation, not "
        "even for a sound, a name, or a line that looks untranslatable.",
        "Your job is to expose what the text literally says, NOT to make it read well. "
        "Do not improve, smooth over, explain or correct it.",
        "Rules:",
        f"- Give the meaning word for word, keeping the original structure, even where "
        f"the result is clumsy, ungrammatical or meaningless in {src}.",
        f"- Idioms and metaphors become their literal meaning, never the equivalent idiom "
        f"in {src}.",
        f"- Names and brands stay as they are; every other word becomes {src}.",
        "- Invent nothing. If the text is incoherent, translate what it says anyway.",
        f"Where this goes wrong, and what to do instead (in {src}):",
        f"  a repeated sound -> not the same characters again, but the {src} spelling of "
        "the same sound",
        "  a name being called -> the name, with the words around it translated",
        "  a line that reads oddly -> still translated, word by word",
        "- Output only JSON in the requested shape. No notes or explanations.",
    ])


def backtranslate_retry_system_prompt(source_lang: str, target_lang: str) -> str:
    """Used once for a line that came back unchanged.

    Same task, different framing: an instruction not to copy can be ignored, but a
    translator who claims not to know the language cannot answer by copying."""
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    return "\n".join([
        f"You are an {src}-speaking translator with no {tgt} vocabulary. You are given "
        f"{tgt} lines and must render each one into {src} from the words you can work out.",
        f"You cannot answer by repeating the {tgt} text: what you write must be {src}. "
        "Guess the parts you are unsure of.",
        "Keep the structure and the names; translate every other word.",
        "Do not improve, explain or omit anything.",
        "Output only JSON in the requested shape.",
    ])


def backtranslate_prompt(items: list[tuple[int, str]], source_lang: str,
                         target_lang: str) -> str:
    tgt, src = lang_name(target_lang), lang_name(source_lang)
    parts = [
        f"Back-translate these {tgt} lines into {src}, literally. Every answer must be "
        f"in {src}.",
        "",
    ]
    parts += [f"{n}. {text.strip()}" for n, text in items]
    return "\n".join(parts)


def backtranslation_passed_through(target_text: str, back: str) -> bool:
    """True when the "back-translation" is just the input written back out.

    The failure this catches is real and quiet: asked to render Polish into English, the
    model sometimes copies the Polish line verbatim. A copied line then "matches" the
    source trivially, so the comparison calls it fine and a wrong translation is
    confirmed as correct — the opposite of what the pass is for. An identical line is
    never a translation of anything.
    """
    target, back = _normalise(target_text), _normalise(back)
    return bool(target) and target == back


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


# --- Comparison -----------------------------------------------------------------


def compare_system_prompt(source_lang: str, target_lang: str) -> str:
    """The prompt that decides drift or not. It carries worked examples because it has
    to: asked to describe the difference, a small model reliably reports synonyms and
    paraphrases as changes ("bun" -> "bread") and misses the failures this pass exists
    for. Measured on qwen3:8b with real burger-review lines, the rule-only wording got 5
    of 8 right and the examples below get 12 of 12, including every case from the failure
    report this feature was built for. The examples are English-to-Polish, which is the
    pair they were measured on; the task they demonstrate — different word, same thing
    versus a different thing — is the same for every target language."""
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    return "\n".join([
        f"You check a {tgt} translation against the {src} it was translated from, using a "
        f"literal {src} back-translation of the {tgt}.",
        "The back-translation is clumsy on purpose. You judge MEANING, and nothing else. "
        "Synonyms and paraphrase are never drift; only a change of *what is being talked "
        "about* is.",
        "",
        "Examples of ok (same meaning, different words):",
        "  source: the bun in that sense is just a bit dry",
        "  back:   the bread in this sense is a little dry            -> ok (same object)",
        "  source: it's all right, it's not the best",
        "  back:   it's good, it's not the best                       -> ok (paraphrase)",
        "  source: That gets a fucking U.",
        "  back:   That gets a f***ing U.                             -> ok (same swear)",
        "  source: Look at the bun on that",
        "  back:   the bun on that                                    -> ok (framing dropped)",
        "  source: give that a 4.5",
        "  back:   give it 4,5                                        -> ok (formatting)",
        "",
        "Examples of drift (the meaning changed):",
        "  source: taste buds",
        "  back:   little teeth                                       -> drift (tongue parts, "
        "not teeth)",
        "  source: the craftsmanship",
        "  back:   worked it out                                      -> drift (a skill became "
        "an action)",
        "  source: on the actual patty",
        "  back:   on the sausage                                     -> drift (a different food)",
        "  source: in moisture i give it a two",
        "  back:   in moisture I give it a ten                        -> drift (number changed)",
        "",
        f"A back-translation still in {tgt}, or one that copies its input, is always "
        f"drift: problem 'not translated back into {src}', and no term.",
        "Answer about every line listed, in the same order, with its number. For a drift, "
        "'problem' names the change in one short clause, and 'term' is the word or short "
        f"phrase *from the {src} source* that the change is about, copied exactly as the "
        "source writes it (e.g. 'patty', 'taste buds', 'craftsmanship'). For an ok line, "
        "'problem' must be an empty string and 'term' must be an empty string.",
        "Go through the lines one at a time and answer with JSON only.",
    ])


def compare_prompt(triples: list[tuple[int, str, str]], source_lang: str,
                   target_lang: str) -> str:
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    parts = [f"Check each {tgt} translation below against its {src} source.", ""]
    for n, source, back in triples:
        parts.append(f"{n}. source ({src}): {source.strip()}")
        parts.append(f"   back-translation ({src}): {back.strip()}")
    return "\n".join(parts)


def parse_comparison(raw: str, source_by_n: dict[int, str]) -> dict[int, Verdict]:
    """``{n: Verdict}``; an entry missing for a listed line is a drift with the reason
    recorded, because a comparison that says nothing about a line is not a verdict that
    the line is fine. A ``term`` that is not in the source is dropped."""
    text = _FENCE.sub("", strip_thinking(raw)).strip()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ParseError(f"comparison is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        raise ParseError("comparison is missing its 'lines' array")
    out: dict[int, Verdict] = {}
    for item in data["lines"]:
        if not isinstance(item, dict):
            raise ParseError("comparison entry is not an object")
        try:
            n = int(item["n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"bad line number in comparison entry {item!r}") from exc
        if n in out:
            raise ParseError(f"line {n} compared twice")
        verdict = str(item.get("verdict", "")).strip().lower()
        problem = clean_text(str(item.get("problem") or ""))
        term = clean_text(str(item.get("term") or ""))
        drifted = verdict != "ok" or bool(problem)
        if drifted and not problem:
            problem = "the round trip does not match the source"
        if not drifted or not in_source(source_by_n.get(n, ""), term):
            # An ok line has no term, and a term that is not in the source line is the
            # model inventing one: either way there is nothing to suggest pinning.
            term = ""
        out[n] = Verdict(drifted, problem, term)
    for n in source_by_n:
        out.setdefault(n, Verdict(True, "not compared by the model"))
    return out


# --- Revision -------------------------------------------------------------------


def revision_system_prompt(source_lang: str, target_lang: str) -> str:
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    return "\n".join([
        f"You repair {tgt} dubbing translations that drifted from their {src} source.",
        f"You are shown, for each line: the {src} source, the current {tgt} translation, "
        f"a literal back-translation of that translation into {src}, and the specific "
        "problem found by comparing the two.",
        "Rewrite the line so it means what the source says, fixing exactly that problem.",
        "Rules:",
        f"- Keep it natural, idiomatic {tgt} speech of the same register. Never translate "
        "word for word: the back-translation was clumsy on purpose, the repair must not be.",
        "- Change no more than the problem requires. If the rest of the line is already "
        "right, keep it.",
        "- Stay within the line's character budget [≤N]; the line is spoken aloud and "
        "must fit its time slot.",
        "- Reproduce any do-not-translate terms exactly as they are written.",
        "- Never add a comment, a note or an alternative. Return every line listed, with "
        "the same numbers, and nothing else in the text.",
        "Answer with JSON only.",
    ])


def revision_prompt(entries: list[tuple[Line, str, str, str]], source_lang: str,
                    target_lang: str) -> str:
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    parts = [
        f"These {tgt} lines do not mean what their {src} source says. Fix each one.",
        "",
    ]
    for line, current, back, problem in entries:
        parts.append(_fmt(line))
        parts.append(f"   current {tgt}: {current.strip()}")
        parts.append(f"   back-translation of the current line ({src}): {back.strip()}")
        parts.append(f"   problem: {problem}")
    return "\n".join(parts)


# --- Report ---------------------------------------------------------------------


@dataclass
class Drift:
    """One line the comparison flagged, and what happened to it."""

    n: int
    source: str
    attempt: str  # first-pass translation
    back: str  # what that attempt actually says
    problem: str
    term: str = ""  # the source word the problem is about, when the model named one
    revision: str | None = None
    revised_back: str | None = None
    remaining: list[str] = field(default_factory=list)
    kept: str = "attempt"  # attempt | revision
    note: str = ""  # why the revision was not kept


def collect_hints(drifts: list[Drift]) -> tuple[list[tuple[str, str, str]], list[str]]:
    """``(verified_pairs, unpinned)`` for the suggested-hints block.

    Each entry is ``(term, the first attempt's rendering of it, the verified revision)``
    for a term whose line was flagged, revised and verified. Only verified revisions
    count: a term paired with wording this pass already rejected would pin the mistake.

    The target-language rendering of the term is deliberately *not* extracted. Working
    out which Polish words render "taste buds" is the same judgement call the comparison
    model only manages about half the time; guessing it here would put confident wrong
    pairs in front of someone who is about to make them permanent. The verified line is
    given instead, and the term inside it is the reader's to spot.
    """
    seen: dict[str, tuple[str, str]] = {}
    unpinned: list[str] = []
    for drift in drifts:
        if not drift.term:
            continue
        key = drift.term.casefold()
        if drift.kept != "revision" or not drift.revision:
            unpinned.append(drift.term)
            continue
        if key not in seen:
            seen[key] = (drift.term, drift.attempt.strip(), drift.revision.strip())
    for term in unpinned:
        if term.casefold() in seen:  # pinned somewhere else in the run, so it is suggested
            continue
    unpinned = sorted({t for t in unpinned if t.casefold() not in seen})
    return list(seen.values()), unpinned


def render_hints_block(hints: list[tuple[str, str, str]], unpinned: list[str],
                       unverified: list[int]) -> list[str]:
    """The block the report ends with: the loop from the expensive pass to the cheap fix.

    It is intentionally one step short of paste-ready. What this pass knows for certain is
    the *term* (it was flagged on meaning, and checked against the source line) and the
    *line* that fixed it (verified by its own round trip). Which words of that line are
    the rendering of the term is not something it can be sure of, so it shows both and
    leaves the last step to the reader rather than writing a confident wrong pair.
    """
    out = ["", "=" * 78]
    if hints:
        out += ["# Suggested hints from this run.",
                "# Each entry is a term this pass flagged, and the line it verified as the",
                "# fix. Copy the term and the words of that line that render it into",
                "# hints.txt (every language) or hints.<lang>.txt (one language), then",
                "# rerun: changing a hint retranslates.",
                ""]
        out += [f"#   {term} = <- from: {revision}" for term, _, revision in hints]
        out += ["",
                "# The first attempt, for comparison - do not pin these, they are what",
                "# the pass found wrong:"]
        out += [f"#   {term} was: {attempt}" for term, attempt, _ in hints]
    else:
        out += ["# No lines were revised and verified this run, so there is no",
                "# `term = translation` pair with evidence behind it. The flagged lines",
                "# above still name the terms worth looking at by hand."]
    if unpinned:
        out += ["",
                "# Flagged but not pinned: the revision drifted too or could not be",
                "# verified, so check these by hand: " + ", ".join(unpinned)]
    if unverified:
        out += ["",
                "# Also worth a look: lines " + str(unverified) + " could not be checked",
                "# at all (the model would not translate them back), so a term there",
                "# could still be wrong."]
    return out


def render_report(source_lang: str, target_lang: str, lines: list[Line],
                  drifts: list[Drift], *, flagged: int, revised: int,
                  unverified: list[int], untranslated: list[int]) -> str:
    """The reviewable record of the pass: without it a silent no-op is invisible.

    Every flagged line is written out in full — source, first attempt, what the first
    attempt actually means, the revision and how it scored — so the pass can be judged
    on real content and switched off if it is not earning its time. It ends with the
    hints the run points at, which is what a diagnostic pass is actually for.
    """
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    hints, unpinned = collect_hints(drifts)
    out = [
        f"Back-translation verification: {src} -> {tgt}",
        f"flagged {flagged} line(s), kept a revision for {revised}",
    ]
    if unverified:
        out.append(f"not verifiable at all ({len(unverified)} line(s), left as "
                   f"translated): {unverified}")
    if untranslated:
        out.append(f"never translated, so not verified ({len(untranslated)} line(s), "
                   f"left in {src}): {untranslated}")
    if not drifts:
        out.append("")
        out.append("Meaning survived every round trip: nothing was flagged.")
        return "\n".join(out + render_hints_block([], [], unverified)) + "\n"

    by_n = {ln.n: ln for ln in lines}
    for drift in drifts:
        line = by_n.get(drift.n)
        budget = f" [≤{line.budget}]" if line else ""
        out += ["", "-" * 78, f"line {drift.n}{budget}"]
        out += [f"  source ({src}):           {drift.source}",
                f"  first attempt ({tgt}):    {drift.attempt}",
                f"  what that means ({src}):  {drift.back}",
                f"  problem:                {drift.problem}"]
        if drift.term:
            out.append(f"  term:                   {drift.term}")
        if drift.revision is not None:
            out.append(f"  revision ({tgt}):         {drift.revision}")
        if drift.revised_back is not None:
            out.append(f"  revision means ({src}):   {drift.revised_back}")
        if drift.remaining:
            out.append(f"  still drifting:         {'; '.join(drift.remaining)}")
        out.append(f"  kept:                   {drift.kept}"
                   + (f" ({drift.note})" if drift.note else ""))
    return "\n".join(out + render_hints_block(hints, unpinned, unverified)) + "\n"
