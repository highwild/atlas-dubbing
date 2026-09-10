"""Document-level translation: prompts, strict parsing, batching, fallbacks, budgets
and context-window safety. A fake client stands in for Ollama."""

from __future__ import annotations

import json
import re

import pytest

from ytdub.stages.translate.ollama import (
    BatchTranslator,
    ChatResult,
    ContextOverflowError,
    OllamaClient,
)
from ytdub.stages.translate.prompt import (
    Hint,
    Line,
    ParseError,
    batch_prompt,
    estimate_tokens,
    glossary_misses,
    load_glossary,
    load_hints,
    lower_bound_tokens,
    merge_hints,
    needs_repair,
    output_token_estimate,
    parse_lines,
    repair_groups,
    system_prompt,
)
from ytdub.stages.translate.verify import (
    Drift,
    Verdict,
    backtranslate_prompt,
    collect_hints,
    compare_prompt,
    parse_comparison,
    render_report,
    revision_prompt,
)

# Lines the model is asked to return carry a budget tag; context lines do not.
LINE_RE = re.compile(r"^(\d+)\. \[≤(\d+)\] (?:\[(SPK\d+)\] )?(?:source: )?(.*)$", re.M)


def asked(user: str) -> list[int]:
    return [int(m[0]) for m in LINE_RE.findall(user)]


class FakeClient:
    def __init__(self, responder, num_ctx: int = 8192):
        self.responder = responder
        self.num_ctx = num_ctx
        self.calls: list[str] = []

    def chat(self, system, user, *, num_predict, schema=None, label=""):
        self.calls.append(user)
        return ChatResult(self.responder(user), 100, 20, "stop")


def answer(texts: dict[int, str]) -> str:
    return json.dumps({"lines": [{"n": n, "text": t} for n, t in texts.items()]})


def echo(user: str) -> str:
    return answer({n: f"T{n}" for n in asked(user)})


def make_lines(n: int, budget: int = 60) -> list[Line]:
    return [Line(n=i + 1, text=f"source line {i + 1}", budget=budget) for i in range(n)]


def translator(client, **kw) -> BatchTranslator:
    return BatchTranslator(client, source_lang="en", target_lang="pl", style_text="Casual.",
                           glossary=["Rec Room"], **kw)


# --- parsing -------------------------------------------------------------------


def test_parse_strict_happy_path_and_cleanup():
    raw = "<think>hmm</think>```json\n" + answer({1: "Cześć", 2: "[≤40] [SPK1] No to lecimy"}) + "\n```"
    assert parse_lines(raw, [1, 2]) == {1: "Cześć", 2: "No to lecimy"}


def test_parse_unterminated_think_is_stripped():
    # Everything after an unclosed <think> is reasoning, not an answer.
    with pytest.raises(ParseError, match="not valid JSON"):
        parse_lines("<think>never closed " + answer({1: "x"}), [1])


@pytest.mark.parametrize("raw", [
    "not json",
    json.dumps({"lines": [{"n": 1, "text": "a"}]}),                     # line 2 missing
    json.dumps({"lines": [{"n": 1, "text": "a"}, {"n": 2, "text": " "}]}),  # empty
    json.dumps({"lines": [{"n": 1, "text": "a"}, {"n": 1, "text": "b"}, {"n": 2, "text": "c"}]}),
    json.dumps({"lines": [{"n": 1, "text": "a"}, {"n": 2, "text": "b"}, {"n": 3, "text": "c"}]}),
    json.dumps({"text": "a"}),
])
def test_parse_rejects_anything_incomplete(raw):
    with pytest.raises(ParseError):
        parse_lines(raw, [1, 2])


# --- prompt content --------------------------------------------------------------


def test_system_prompt_has_glossary_style_and_locale_rules():
    sp = system_prompt("en", "pl", "A casual gaming video.", ["Rec Room", "Atlas"])
    assert "Polish" in sp and "A casual gaming video." in sp
    assert "- Rec Room" in sp and "- Atlas" in sp
    assert "Locale rules" in sp  # default rules when the preset has none
    custom = system_prompt("en", "pl", "Locale rules:\n- convert miles", [])
    assert custom.count("Locale rules") == 1


def test_shipped_style_presets_carry_locale_rules():
    from pathlib import Path

    for preset in Path(__file__).resolve().parents[1].joinpath("synopses").glob("*.txt"):
        assert "locale rules" in preset.read_text(encoding="utf-8").lower(), preset.name


def test_batch_prompt_shows_budgets_context_and_lookahead():
    lines = make_lines(6)
    user = batch_prompt(lines[2:4], [(lines[1], "T2")], lines[4:6], "pl")
    assert asked(user) == [3, 4]
    assert "2. source line 2  =>  T2" in user
    assert "5. source line 5" in user and "do not translate" in user
    assert "[≤60]" in user


def test_glossary_file_format():
    assert load_glossary("# c\nAtlas\n\n  Rec Room  \n#x") == ["Atlas", "Rec Room"]


def test_hints_file_format_and_merge():
    parsed = load_hints("# comment\ntaste buds = kubki smakowe\n\nnot a hint\nbroken =\n"
                        "  patty  =  kotlet  \ncraftsmanship = rzemiosło")
    assert parsed == [Hint("taste buds", "kubki smakowe"), Hint("patty", "kotlet"),
                      Hint("craftsmanship", "rzemiosło")]
    merged = merge_hints([Hint("patty", "kotlet"), Hint("bun", "bułka")],
                         [Hint("Patty", "plaster mięsa")])
    assert {(h.term, h.translation) for h in merged} == {("Patty", "plaster mięsa"),
                                                         ("bun", "bułka")}


def test_hints_are_injected_into_the_system_prompt_beside_the_glossary():
    sp = system_prompt("en", "pl", "", ["Rec Room"], [Hint("patty", "kotlet")])
    assert "- Rec Room" in sp and "- patty => kotlet" in sp
    assert sp.index("Do not translate") < sp.index("patty => kotlet")
    assert "patty => kotlet" not in system_prompt("en", "pl", "", ["Rec Room"], [])


def test_hints_change_the_prompt_not_just_the_paragraph():
    # Same lines, different hints: the model must be told something different.
    a = system_prompt("en", "pl", "", [], [Hint("patty", "kotlet")])
    b = system_prompt("en", "pl", "", [], [Hint("patty", "plaster")])
    assert a != b


def test_glossary_misses():
    assert glossary_misses("I love Rec Room", "Kocham Rec Room", ["Rec Room"]) == []
    assert glossary_misses("I love Rec Room", "Kocham pokój rekreacyjny", ["Rec Room"]) == ["Rec Room"]
    assert glossary_misses("no terms", "brak", ["Rec Room"]) == []


def test_token_estimates_are_pessimistic():
    english = "So unless you've been living under a rock, you'll know what happened. " * 10
    # Real Qwen tokenization of English is ~4.2 chars/token.
    assert estimate_tokens(english) >= len(english) / 4.2
    hindi = "नमस्ते दोस्तों आज हम बात करेंगे" * 5
    assert estimate_tokens(hindi) >= len(hindi)
    assert lower_bound_tokens(english) <= len(english) / 6


def test_needs_repair_and_groups():
    assert not needs_repair("x" * 44, 40, 1.15)   # within tolerance
    assert not needs_repair("x" * 12, 8, 1.15)    # over by <5 chars
    assert needs_repair("x" * 60, 40, 1.15)
    assert repair_groups([3, 4, 10], list(range(1, 12))) == [[2, 3, 4, 5], [9, 10, 11]]
    assert repair_groups([1], [1, 2]) == [[1, 2]]


# --- batch translator ------------------------------------------------------------


def test_batches_carry_context_and_lookahead_across_boundaries():
    client = FakeClient(echo)
    out = translator(client, batch_lines=30, context_lines=4, lookahead_lines=3).translate(make_lines(70))
    assert out == [f"T{i}" for i in range(1, 71)]
    assert [asked(u) for u in client.calls] == [list(range(1, 31)), list(range(31, 61)),
                                               list(range(61, 71))]
    second = client.calls[1]
    assert "30. source line 30  =>  T30" in second and "26." not in second  # 4 lines of context
    assert "61. source line 61" in client.calls[0] or "33. source line 33" in client.calls[0]


def test_bad_batch_is_split_then_recovered():
    def flaky(user):
        return "garbage" if len(asked(user)) > 8 else echo(user)

    t = translator(FakeClient(flaky), batch_lines=30)
    out = t.translate(make_lines(30))
    assert out == [f"T{i}" for i in range(1, 31)]
    assert t.stats.batch_splits >= 2 and not t.stats.untranslated


def test_hopeless_lines_fall_back_to_source_never_dropped():
    t = translator(FakeClient(lambda user: "garbage"), batch_lines=4)
    lines = make_lines(5)
    out = t.translate(lines)
    assert out == [ln.text for ln in lines]
    assert t.stats.untranslated == [1, 2, 3, 4, 5]


def test_truncated_answer_counts_as_failure():
    class Cut(FakeClient):
        def chat(self, system, user, **kw):
            self.calls.append(user)
            reason = "length" if len(asked(user)) > 1 else "stop"
            return ChatResult(echo(user), 100, 20, reason)

    t = translator(Cut(None), batch_lines=4)
    assert t.translate(make_lines(4)) == ["T1", "T2", "T3", "T4"]
    assert t.stats.batch_splits >= 1


def test_a_cut_off_answer_is_given_more_room_before_the_batch_is_split():
    """More tokens fixes a truncated answer without losing the batch's context. Splitting
    is the fallback, not the first response — measured on tangi.wav, where a 30-line
    Polish batch that needed 955 tokens was allowed 923."""
    budgets: list[int] = []

    class CutOnce(FakeClient):
        def chat(self, system, user, *, num_predict, schema=None, label=""):
            self.calls.append(user)
            budgets.append(num_predict)
            # Cut off at every budget up to the escalation, then answer properly.
            if num_predict < 300:
                return ChatResult("", 100, num_predict, "length")
            return ChatResult(echo(user), 100, 20, "stop")

    t = translator(CutOnce(None), batch_lines=4)
    assert t.translate(make_lines(4)) == ["T1", "T2", "T3", "T4"]
    assert len(budgets) == 2 and budgets[1] > budgets[0], budgets
    assert t.stats.batch_splits == 0, "the batch was recoverable without splitting it"
    assert t.stats.answer_escalations == 1
    assert t.stats.untranslated == []


FRAG_N = re.compile(r"line (\d+) to translate now")


def test_a_fragment_that_came_back_untranslated_is_asked_again():
    """The real failure: Whisper split "okay I would like to purchase" into three lines, and
    the French track got "Okay i" and "Would" spoken in it. Framed as a fragment, the model
    answers "ok" and "voudrais" (measured against the live model on those two lines)."""
    def responder(user):
        if "is a fragment" in user:
            return answer({int(FRAG_N.search(user)[1]): "voudrais"})
        return answer({n: ("would" if n == 2 else f"T{n}") for n in asked(user)})

    lines = [Line(1, "okay i", 20), Line(2, "would", 20), Line(3, "like to purchase", 20)]
    t = translator(FakeClient(responder), batch_lines=3)
    out = t.translate(lines)
    assert out == ["T1", "voudrais", "T3"]
    assert t.stats.echo_lines == 1 and t.stats.echoes_fixed == 1


def test_a_line_that_stands_alone_is_not_retried_when_unchanged():
    """"Tanguy" coming back as "Tanguy" is a correct translation of a name. Only a line in
    the middle of a sentence is evidence that the model handed the source back."""
    def responder(user):
        assert "is a fragment" not in user, "a standalone line must not be retried"
        return answer({n: (f"T{n}" if n != 2 else "Tanguy") for n in asked(user)})

    lines = [Line(1, "He said something.", 30), Line(2, "Tanguy", 20),
             Line(3, "And then he left.", 30)]
    t = translator(FakeClient(responder), batch_lines=3)
    assert t.translate(lines) == ["T1", "Tanguy", "T3"]
    assert t.stats.echo_lines == 0


def test_a_retry_that_echoes_again_is_not_accepted():
    def responder(user):
        if "is a fragment" in user:
            return answer({int(FRAG_N.search(user)[1]): "would"})
        return answer({n: ("would" if n == 2 else f"T{n}") for n in asked(user)})

    lines = [Line(1, "okay i", 20), Line(2, "would", 20), Line(3, "like to purchase", 20)]
    t = translator(FakeClient(responder), batch_lines=3)
    assert t.translate(lines) == ["T1", "would", "T3"]
    assert t.stats.echoes_fixed == 0 and t.stats.echo_retries == 1


def test_a_retry_that_returns_the_whole_sentence_is_rejected():
    """A fragment given the whole sentence sometimes answers with the whole sentence, which
    would be spoken over a slot one word long."""
    def responder(user):
        if "is a fragment" in user:
            n = int(FRAG_N.search(user)[1])
            return answer({n: "je voudrais acheter un cylindre de butane de 70 litres " * 2})
        return answer({n: ("would" if n == 2 else f"T{n}") for n in asked(user)})

    lines = [Line(1, "okay i", 20), Line(2, "would", 20), Line(3, "like to purchase", 20)]
    t = translator(FakeClient(responder), batch_lines=3)
    assert t.translate(lines) == ["T1", "would", "T3"]
    assert t.stats.echoes_fixed == 0


def test_the_answer_budget_always_covers_the_estimate_and_stays_in_the_window():
    client = FakeClient(echo, num_ctx=4096)
    t = translator(client, batch_lines=4)
    lines = make_lines(4, budget=300)
    user = "x" * 3000
    est = output_token_estimate(lines, "pl")
    budget = t._answer_budget(user, lines)
    assert budget >= est
    assert t._prompt_tokens(user) + budget <= client.num_ctx
    # A prompt that leaves almost nothing still gets the estimator's number, never less.
    assert t._answer_budget("y" * 100_000, lines) >= est


def test_batches_shrink_to_fit_a_small_window():
    long_lines = [Line(n=i + 1, text="word " * 60, budget=300) for i in range(12)]
    client = FakeClient(echo, num_ctx=3000)
    translator(client, batch_lines=12).translate(long_lines)
    assert len(client.calls) > 1
    assert all(len(asked(u)) < 12 for u in client.calls)


def test_line_too_big_for_any_window_fails_loudly():
    client = FakeClient(echo, num_ctx=600)
    with pytest.raises(ContextOverflowError):
        translator(client).translate([Line(n=1, text="word " * 400, budget=2000)])


def test_over_budget_lines_are_shortened_and_only_improvements_kept():
    state = {"repairs": 0}

    def responder(user):
        if "TOO LONG" in user:
            state["repairs"] += 1
            # First repair improves line 2; second makes it worse (must be rejected).
            text = "short" if state["repairs"] == 1 else "x" * 200
            return answer({n: (text if n == 2 else f"T{n}") for n in asked(user)})
        return answer({n: ("y" * 80 if n == 2 else f"T{n}") for n in asked(user)})

    lines = make_lines(3, budget=30)
    t = translator(FakeClient(responder), budget_retries=2)
    out = t.translate(lines)
    assert out[1] == "short"
    assert t.stats.over_budget_initial == 1 and t.stats.over_budget_final == 0


def test_glossary_misses_are_recorded():
    lines = [Line(1, "Welcome to Rec Room", 60)]
    t = translator(FakeClient(lambda u: answer({1: "Witamy w pokoju"})))
    t.translate(lines)
    assert t.stats.glossary_misses == {1: ["Rec Room"]}


# --- back-translation verification ------------------------------------------------
#
# A stand-in model that really does round-trip: the first pass renders source lines
# "S<n>" as target lines "T<n>", the back-translation renders "T<n>" as "S<n>", and the
# comparison calls a round trip a drift unless the two agree. That is enough to drive
# the whole pass deterministically, including the case it exists for: a first pass that
# is fluent and wrong (line 1 -> "T1bad", which round-trips to "S1bad").


class VerifyFake:
    """Fake Ollama client whose three request kinds behave differently.

    ``round_trip`` is the model's opinion of what a target line means; it is what makes
    a first pass "fluent but wrong" (``T1`` means ``X1``, not ``S1``). A revision request
    answers with ``revised(n)``, which by default round-trips correctly — a fake model
    that is told what drifted produces a *better* second attempt, which is what the pass
    is for and what makes "keep the better attempt" testable.
    """

    def __init__(self, *, broken_numbers=(), fail_compare=frozenset(), revised=None):
        self.broken = set(broken_numbers)  # line numbers whose text means something else
        self.repaired = set()  # line numbers a revision request has answered for
        self.fail_compare = fail_compare  # None = every comparison fails
        self.revised = revised or (lambda n: f"T{n}rev")
        self.num_ctx = 8192
        self.calls: list[str] = []

    def round_trip(self, target_text: str) -> str:
        """What the model thinks ``target_text`` means. ``T<n>`` means ``S<n>``, unless
        line ``n`` is broken *and* unrepaired (then it means ``X<n>`` — fluent, and not
        what the source said, which is the whole failure this pass exists for)."""
        match = re.match(r"^T(\d+)(bad)?", target_text)
        if not match:
            return f"meaning of {target_text}"
        n = int(match.group(1))
        if match.group(2) and n in self.broken and n not in self.repaired:
            return f"X{n}"
        return f"S{n}"

    def chat(self, system, user, *, num_predict, schema=None, label="",
             temperature=None):
        self.calls.append(user)
        if "back-translator" in system:
            rows = re.findall(r"^(\d+)\. (.*)$", user, re.M)
            return ChatResult(answer({int(n): self.round_trip(t) for n, t in rows}),
                              100, 10, "stop")
        if "problem:" in user:  # a revision request (checked before the comparison)
            rows = re.findall(r"^(\d+)\. \[≤\d+\]", user, re.M)
            self.repaired.update(int(n) for n in rows)
            return ChatResult(answer({int(n): self.revised(int(n)) for n in rows}),
                              100, 10, "stop")
        if "Check each" in user:
            triples = re.findall(r"^(\d+)\. source \([^)]*\): (.*)\n +back-translation "
                                 r"\([^)]*\): (.*)$", user, re.M)
            if self.fail_compare is None or any(int(n) in self.fail_compare
                                                for n, _, _ in triples):
                return ChatResult("not json", 100, 10, "stop")
            return ChatResult(json.dumps({"lines": [
                {"n": int(n), "verdict": "drift" if a != b else "ok",
                 "problem": "" if a == b else f"{a} became {b}"}
                for n, a, b in triples]}), 100, 10, "stop")
        rows = re.findall(r"^(\d+)\. \[≤\d+\] (?:\[SPK\d\] )?(.*)$", user, re.M)
        return ChatResult(answer({int(n): f"T{n}bad" if int(n) in self.broken else f"T{n}"
                                  for n, _ in rows}), 100, 10, "stop")


def verified(client, **kw) -> BatchTranslator:
    return translator(client, verify=True, **kw)


def test_verification_flags_and_revises_a_line_that_does_not_round_trip():
    client = VerifyFake(broken_numbers={1})
    t = verified(client)
    lines = [Line(1, "S1", 60), Line(2, "S2", 60)]
    out = t.translate(lines)
    assert out == ["T1rev", "T2"]  # the bad line was replaced by its revision
    assert t.stats.flagged == 1 and t.stats.revised == 1 and t.stats.reverted == 0
    revision = next(u for u in client.calls if "problem:" in u)
    assert "current Polish: T1bad" in revision
    assert "back-translation of the current line (English): X1" in revision
    assert "problem: S1 became X1" in revision

def test_a_revision_that_does_not_help_is_not_kept():
    class StillWrong(VerifyFake):
        def round_trip(self, target_text: str) -> str:
            if target_text.endswith("rev"):
                return "still the wrong meaning"
            return super().round_trip(target_text)

    # The revision round-trips as badly as the first attempt (the verdict names it), so
    # there is no evidence it is an improvement and the first attempt stands.
    t = verified(StillWrong(broken_numbers={1}))
    out = t.translate([Line(1, "S1", 60)])
    assert out == ["T1bad"]
    assert t.stats.flagged == 1 and t.stats.revised == 0 and t.stats.reverted == 1


def test_verification_off_by_default_sends_no_extra_requests():
    client = VerifyFake()
    t = translator(client)
    t.translate([Line(1, "S1", 60), Line(2, "S2", 60)])
    assert len(client.calls) == 1  # one translation batch, nothing else
    assert not t.stats.verified and t.stats.flagged == 0


def test_verification_writes_a_report_of_what_it_found(tmp_path):
    path = tmp_path / "pl.verify.txt"
    t = verified(VerifyFake(broken_numbers={1}, revised=lambda n: "T1bad"),
                 verify_report_path=path)
    t.translate([Line(1, "on the actual patty", 60)])
    report = path.read_text(encoding="utf-8")
    assert "flagged 1 line(s)" in report
    for expected in ("on the actual patty", "first attempt (Polish)", "T1bad",
                     "what that means (English)", "X1", "problem:", "became X1",
                     "kept:"):
        assert expected in report, expected
    assert t.stats.verify_report == str(path)


def test_verification_report_is_written_even_when_nothing_is_flagged(tmp_path):
    path = tmp_path / "pl.verify.txt"
    t = verified(VerifyFake(), verify_report_path=path)
    t.translate([Line(1, "S1", 60)])
    assert "nothing was flagged" in path.read_text(encoding="utf-8")


def test_a_failed_comparison_is_not_a_drift_verdict():
    # No verdict about a line is no evidence, so it is left exactly as translated and
    # reported; it is never "revised" on the strength of a comparison that failed.
    t = verified(VerifyFake(fail_compare={1}))
    assert t.translate([Line(1, "S1", 60)]) == ["T1"]
    assert t.stats.unverifiable == [1] and t.stats.flagged == 0
    assert t.stats.verify_failures >= 1


def test_one_uncomparable_line_does_not_stop_the_others():
    t = verified(VerifyFake(fail_compare={1}))
    out = t.translate([Line(1, "S1", 60), Line(2, "S2", 60), Line(3, "S3", 60)])
    assert out == ["T1", "T2", "T3"]
    assert t.stats.unverifiable == [1] and t.stats.flagged == 0


def test_a_failed_back_translation_never_loses_a_line():
    class NoBack(VerifyFake):
        def chat(self, system, user, **kw):
            if "back-translator" in system:
                self.calls.append(user)
                return ChatResult("garbage", 100, 10, "stop")
            return super().chat(system, user, **kw)

    t = verified(NoBack())
    lines = [Line(1, "S1", 60), Line(2, "S2", 60)]
    out = t.translate(lines)
    assert out == ["T1", "T2"]
    assert t.stats.unverifiable == [1, 2] and t.stats.flagged == 0


def test_a_back_translation_that_echoes_the_input_is_not_evidence():
    # Asked to render Polish into English, the model sometimes copies the Polish line.
    # A copied line "matches" the source trivially, so treating it as a verdict would
    # confirm a wrong translation as correct - the opposite of the point.
    class Echoes(VerifyFake):
        def chat(self, system, user, **kw):
            if "back-translator" in system:
                self.calls.append(user)
                rows = re.findall(r"^(\d+)\. (.*)$", user, re.M)
                return ChatResult(answer({int(n): t for n, t in rows}), 100, 10, "stop")
            return super().chat(system, user, **kw)

    t = verified(Echoes(), verify_echo_retries=0)
    out = t.translate([Line(1, "S1", 60), Line(2, "S2", 60)])
    assert out == ["T1", "T2"]
    assert t.stats.unverifiable == [1, 2] and t.stats.flagged == 0
    assert t.stats.verify_comparisons == 0  # nothing usable to compare
    assert t.stats.verify_echo_retries == 0  # the second attempt was disabled


def test_an_echoed_line_is_retried_once_and_can_still_be_verified():
    """The failure worth fixing: a line the model would not translate back at all.

    The retry uses different instructions and a higher temperature, so it is a genuinely
    different attempt rather than the same request sent twice.
    """

    class EchoesFirst(VerifyFake):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.systems: list[str] = []
            self.temperatures: list[float | None] = []

        def chat(self, system, user, *, num_predict, schema=None, label="",
                 temperature=None):
            if "back-translator" in system or "no Polish vocabulary" in system:
                self.calls.append(user)
                self.systems.append(system)
                self.temperatures.append(temperature)
                rows = re.findall(r"^(\d+)\. (.*)$", user, re.M)
                if "no Polish vocabulary" in system:  # the retry: translate properly
                    return ChatResult(answer({int(n): self.round_trip(t)
                                              for n, t in rows}), 100, 10, "stop")
                return ChatResult(answer({int(n): t for n, t in rows}), 100, 10, "stop")
            return super().chat(system, user, num_predict=num_predict, schema=schema,
                                label=label, temperature=temperature)

    client = EchoesFirst(broken_numbers={1})
    t = verified(client)
    out = t.translate([Line(1, "S1", 60), Line(2, "S2", 60)])
    # One retry round for the first pass, one for the re-verification of the revision:
    # never a second retry of the same text.
    assert t.stats.verify_echo_retries == 2
    assert t.stats.unverifiable == []
    # Two attempts per round - the first pass and the re-verification of the revision -
    # and the retry is always a different prompt at a hotter setting.
    assert len(client.systems) == 4
    assert client.systems[0] != client.systems[1]
    assert client.temperatures[0] is None and client.temperatures[1] == 0.8
    # Line 1 is still flagged - but now on evidence instead of on silence.
    assert t.stats.flagged == 1
    assert out == ["T1rev", "T2"]


def test_passthrough_check_ignores_punctuation_and_case():
    from ytdub.stages.translate.verify import backtranslation_passed_through

    assert backtranslation_passed_through("Sztuczność tego burgera.",
                                          "sztuczność  tego burgera")
    assert not backtranslation_passed_through("Sztuczność tego burgera",
                                              "The artificiality of this burger")


def test_a_verified_revision_produces_a_suggested_hint(tmp_path):
    """The loop the pass exists for: a line drifts, the revision fixes it, and the report
    offers the term it was about together with the line that fixed it."""

    class Termed(VerifyFake):
        """A verifier that names the term, over a model that translates the first attempt
        wrongly and the revision correctly."""

        def round_trip(self, target_text: str) -> str:
            if target_text == "T1bad":
                return "little teeth are fine"
            if target_text == "T1rev":
                return "taste buds are fine"
            return "WRONG"

        def chat(self, system, user, *, num_predict, schema=None, label="",
                 temperature=None):
            if "Check each" in user:
                self.calls.append(user)
                triples = re.findall(r"^(\d+)\. source \([^)]*\): ([^\n]*)\n +back-translation "
                                     r"\([^)]*\): ([^\n]*)$", user, re.M)
                same = [(n, a, b) for n, a, b in triples if a != b]
                return ChatResult(json.dumps({"lines": [
                    {"n": int(n), "verdict": "drift", "problem": "the meaning changed",
                     "term": "taste buds"} for n, _, _ in same] + [
                    {"n": int(n), "verdict": "ok", "problem": "", "term": ""}
                    for n, a, b in triples if a == b]}), 100, 10, "stop")
            return super().chat(system, user, num_predict=num_predict, schema=schema,
                                label=label, temperature=temperature)

    path = tmp_path / "pl.verify.txt"
    t = verified(Termed(broken_numbers={1}), verify_report_path=path)
    t.translate([Line(1, "taste buds are fine", 60)])
    report = path.read_text(encoding="utf-8")
    assert t.stats.revised == 1
    assert "# Suggested hints from this run." in report
    # The term, the line the pass verified as the fix, and the attempt it replaced.
    assert "#   taste buds = <- from: T1rev" in report
    assert "#   taste buds was: T1bad" in report


def test_hints_are_only_suggested_from_a_verified_revision():
    # The evidence rule: the term is flagged, and the revision it points at round-tripped
    # clean. A first attempt is never suggested as the rendering to pin - the pass has
    # already decided that wording is wrong, and pinning it would make the error
    # permanent.
    verified_revision = Drift(n=1, source="taste buds it's all right",
                              attempt="kubki smakowe, jest ok",
                              back="taste buds, is ok", problem="taste buds became taste buds",
                              term="taste buds", revision="kubki smakowe są w porządku",
                              revised_back="taste buds are in order", kept="revision")
    reverted = Drift(n=2, source="the patty was dry", attempt="bułka była sucha",
                     back="the bun was dry", problem="patty became bun", term="patty",
                     revision="kotlet był suchy", revised_back="the cutlet was dry",
                     kept="attempt")
    hints, unpinned = collect_hints([verified_revision, reverted])
    assert hints == [("taste buds", "kubki smakowe, jest ok", "kubki smakowe są w porządku")]
    assert unpinned == ["patty"]

    text = render_report("en", "pl", [Line(1, "x", 10)], [verified_revision, reverted],
                         flagged=2, revised=1, unverified=[9], untranslated=[])
    suggested = text.split("Suggested hints")[1]
    assert "#   taste buds = <- from: kubki smakowe są w porządku" in suggested
    assert "#   taste buds was: kubki smakowe, jest ok" in suggested
    assert "check these by hand: patty" in suggested
    assert "lines [9] could not be checked" in suggested
    assert "patty was:" not in suggested  # an unverified attempt is never offered


def test_there_is_no_hint_block_without_a_verified_revision():
    # A run that flags lines but keeps no revision has nothing with evidence behind it,
    # and says so instead of offering the wording it just rejected.
    drift = Drift(n=1, source="on the actual patty", attempt="na kiełbasie",
                  back="on the sausage", problem="patty became sausage", term="patty",
                  revision="na kotlecie", revised_back="on the cutlet",
                  note="the revision drifted too; the first attempt was kept")
    hints, unpinned = collect_hints([drift])
    assert hints == [] and unpinned == ["patty"]
    text = render_report("en", "pl", [Line(1, "x", 10)], [drift], flagged=1, revised=0,
                         unverified=[], untranslated=[])
    block = text.split("=" * 78)[-1]
    assert "No lines were revised and verified this run" in block
    assert "na kiełbasie" not in block  # the rejected attempt is not offered as a hint
    assert "na kotlecie" not in block


def test_the_hint_block_is_written_even_when_nothing_drifted(tmp_path):
    path = tmp_path / "pl.verify.txt"
    t = verified(VerifyFake(), verify_report_path=path)
    t.translate([Line(1, "S1", 60)])
    report = path.read_text(encoding="utf-8")
    assert "nothing was flagged" in report
    assert "No lines were revised and verified this run" in report


def test_leaked_thinking_tokens_are_stripped_from_the_report():
    from ytdub.stages.translate.verify import clean_text

    assert clean_text("and green leaves /no_think") == "and green leaves"
    assert clean_text("<think>hmm</think> the bun") == "the bun"
