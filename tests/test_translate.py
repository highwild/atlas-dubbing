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
    Line,
    ParseError,
    batch_prompt,
    estimate_tokens,
    glossary_misses,
    load_glossary,
    lower_bound_tokens,
    needs_repair,
    parse_lines,
    repair_groups,
    system_prompt,
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


# --- Ollama client safety checks ----------------------------------------------------


def _client(**kw) -> OllamaClient:
    return OllamaClient("http://127.0.0.1:11434", "qwen3:8b", num_ctx=kw.pop("num_ctx", 8192),
                        temperature=0.3, timeout=5, **kw)


def test_every_request_sets_num_ctx_and_disables_thinking(monkeypatch):
    client = _client()
    sent = []

    def fake_stream(path, payload):
        sent.append(payload)
        yield {"message": {"content": '{"lines": []}'}, "done": False}
        yield {"message": {"content": ""}, "done": True, "done_reason": "stop",
               "prompt_eval_count": 500, "eval_count": 5}

    monkeypatch.setattr(client, "_stream", fake_stream)
    client.chat("system " * 50, "user " * 50, num_predict=100)
    client.chat("system " * 50, "user " * 50, num_predict=100)
    for payload in sent:
        assert payload["options"]["num_ctx"] == 8192
        assert payload["think"] is False
    # Unique prefix per request, so a cached prefix never shrinks prompt_eval_count.
    firsts = [p["messages"][0]["content"].splitlines()[0] for p in sent]
    assert firsts[0] != firsts[1]


def test_prompt_too_big_is_refused_before_sending(monkeypatch):
    client = _client(num_ctx=1000)
    monkeypatch.setattr(client, "_stream", lambda *a: pytest.fail("must not send"))
    with pytest.raises(ContextOverflowError):
        client.chat("s", "word " * 2000, num_predict=100)


def _respond(client, monkeypatch, prompt_eval_count, eval_count=5):
    def fake_stream(path, payload):
        yield {"message": {"content": "ok"}, "done": True, "done_reason": "stop",
               "prompt_eval_count": prompt_eval_count, "eval_count": eval_count}
    monkeypatch.setattr(client, "_stream", fake_stream)


def test_server_side_truncation_is_detected(monkeypatch):
    client = _client()
    # Server claims to have read 300 tokens of a ~9000-character prompt: truncated.
    _respond(client, monkeypatch, 300)
    with pytest.raises(ContextOverflowError, match="truncated"):
        client.chat("sys", "word " * 1800, num_predict=50)


def test_prompt_plus_answer_over_window_is_detected(monkeypatch):
    client = _client(num_ctx=8192)
    _respond(client, monkeypatch, 8000, eval_count=400)
    with pytest.raises(ContextOverflowError):
        client.chat("sys", "word " * 100, num_predict=50)


def test_context_calibration(monkeypatch):
    client = _client(num_ctx=8192)
    _respond(client, monkeypatch, 4096)  # server stuck at its default window
    with pytest.raises(ContextOverflowError, match="calibration"):
        client.verify_context_window()
    _respond(client, monkeypatch, 4950)  # read the whole ~4915-word probe
    client.verify_context_window()
