"""Document-level translation through a local Ollama server.

Replaces the reference implementation's one-line-at-a-time translator. Lines are sent
in numbered batches with a synopsis, glossary, character budgets, the previous few
translated lines and a peek at the lines that follow, so the model can see register,
references and where the slack is.

Context-window safety (Ollama silently truncates the *oldest* tokens, i.e. the system
prompt and glossary, when a prompt exceeds ``num_ctx``):

* ``num_ctx`` is set on every request, never left to server defaults.
* Every prompt is size-checked with a pessimistic token estimate before sending, and
  batches are shrunk until prompt + expected answer fit. Too big = hard error.
* Before the first batch, a calibration request of known size proves the server really
  honours the window (a server stuck at 2048/4096 fails here, not silently later).
* After every request ``prompt_eval_count`` is checked: fewer tokens than the prompt
  could possibly contain means truncation, and so does ``prompt + answer > num_ctx``.
  Each request starts with a unique ``Request #k`` line so a cached prefix never makes
  the count look short.

Robustness: output is constrained with a JSON schema and parsed strictly. A failed
batch is split in half and retried; a single line that still fails falls back to its
own request, and only then to the untranslated source text, which is logged. Lines are
never dropped.

Verification (``--verify``, off by default) adds the back-translation pass from
``verify.py``: the translation is rendered back into the source language, the round trip
is compared with the original line, and lines whose meaning changed are re-translated
with the drift named. It is a *revision*, never a replacement: a line is only overwritten
by a revision that parses and round-trips at least as well, so the pass can improve
wording but can never drop, blank or degrade a line. Budget shortening still runs after
it, because a revised line is only useful if it still fits its time slot.
"""

from __future__ import annotations

import itertools
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from ytdub.logging import stage_logger
from ytdub.stages.translate import verify as verify_mod
from ytdub.stages.translate.prompt import (
    RESPONSE_SCHEMA,
    Hint,
    Line,
    ParseError,
    TruncatedAnswer,
    batch_prompt,
    estimate_tokens,
    fragment_retry_prompt,
    glossary_misses,
    lower_bound_tokens,
    needs_repair,
    output_token_estimate,
    overshoot,
    parse_lines,
    repair_groups,
    repair_prompt,
    repeat_retry_prompt,
    single_line_prompt,
    strip_thinking,
    system_prompt,
)

log = stage_logger("translate")

# A source line ending like this closes a sentence, so the next line is not a continuation.
_SENTENCE_END = re.compile(r"[.!?…:;]\s*[\"')\]]*$")

# Chat-template tokens Ollama wraps around the messages.
_TEMPLATE_OVERHEAD = 32

# How much room to give the answer on top of the estimator's number. Measured, not
# guessed: for a 30-line Polish batch of tangi.wav the estimator said 923 tokens and the
# model used 955 — a 3% margin, so a batch that ran slightly long was cut off mid-JSON,
# thrown away, and redone in halves. The cost of headroom is nothing (the window is far
# from full at these batch sizes); the cost of being wrong is a wasted generation plus the
# context that the smaller batch loses.
ANSWER_HEADROOM = 1.6
# And if it is cut off anyway, this much more before giving up on the batch.
ANSWER_ESCALATION = 3.0


class OllamaError(RuntimeError):
    pass


class ContextOverflowError(OllamaError):
    """A prompt does not fit, or the server truncated one. Never retried silently."""


@dataclass
class ChatResult:
    content: str
    prompt_eval_count: int | None
    eval_count: int | None
    done_reason: str | None


class OllamaClient:
    def __init__(self, url: str, model: str, *, num_ctx: int, temperature: float,
                 timeout: float, keep_alive: str = "15m") -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._counter = itertools.count(1)

    # -- transport -----------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None, *, method: str = "POST"):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(f"{self.url}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.url}{path}: {exc}") from exc

    def _stream(self, path: str, payload: dict) -> Iterator[dict]:
        with self._request(path, payload) as resp:
            for raw in resp:
                raw = raw.strip()
                if raw:
                    chunk = json.loads(raw)
                    if "error" in chunk:
                        raise OllamaError(f"Ollama error: {chunk['error']}")
                    yield chunk

    # -- API -----------------------------------------------------------------
    def check_model(self) -> None:
        with self._request("/api/tags", method="GET") as resp:
            names = {m.get("name") for m in json.loads(resp.read()).get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaError(f"model {self.model!r} is not pulled. Run: ollama pull {self.model}")

    def model_digest(self) -> str | None:
        """Digest of the weights Ollama serves under this name (changes on re-pull)."""
        with self._request("/api/tags", method="GET") as resp:
            for m in json.loads(resp.read()).get("models", []):
                if m.get("name") in (self.model, f"{self.model}:latest"):
                    return m.get("digest")
        return None

    def chat(self, system: str, user: str, *, num_predict: int, schema: dict | None = None,
             label: str = "", temperature: float | None = None) -> ChatResult:
        system = f"Request #{next(self._counter)} {label}".strip() + "\n\n" + system
        estimate = estimate_tokens(system) + estimate_tokens(user) + _TEMPLATE_OVERHEAD
        if estimate + num_predict > self.num_ctx:
            raise ContextOverflowError(
                f"prompt ~{estimate} tokens + {num_predict} for the answer exceeds "
                f"num_ctx={self.num_ctx}"
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx, "num_predict": num_predict,
                        "temperature": self.temperature if temperature is None
                        else temperature},
        }
        if schema is not None:
            payload["format"] = schema
        parts: list[str] = []
        final: dict = {}
        for chunk in self._stream("/api/chat", payload):
            parts.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                final = chunk
        result = ChatResult(
            content=strip_thinking("".join(parts)),
            prompt_eval_count=final.get("prompt_eval_count"),
            eval_count=final.get("eval_count"),
            done_reason=final.get("done_reason"),
        )
        self._verify_not_truncated(result, system + user, num_predict)
        return result

    def _verify_not_truncated(self, result: ChatResult, prompt: str, num_predict: int) -> None:
        seen = result.prompt_eval_count
        if seen is None:
            log.warning("Ollama did not report prompt_eval_count; cannot verify no truncation")
            return
        floor = lower_bound_tokens(prompt)
        if seen < floor:
            raise ContextOverflowError(
                f"Ollama read only {seen} prompt tokens but the prompt has at least "
                f"{floor}: it was truncated (is num_ctx being overridden?)"
            )
        if seen + (result.eval_count or 0) > self.num_ctx:
            raise ContextOverflowError(
                f"prompt ({seen}) + answer ({result.eval_count}) tokens exceed "
                f"num_ctx={self.num_ctx}: context was truncated"
            )

    def verify_context_window(self) -> None:
        """Prove the server honours ``num_ctx`` before relying on it.

        Sends a prompt of ~60% of the window made of common one-token words and checks
        the server read (nearly) all of it. The probe must exceed Ollama's 2048/4096
        defaults to prove anything, and stays small enough to pass the (pessimistic)
        pre-send size check.
        """
        words = int(self.num_ctx * 0.6)
        if words <= 4200:
            log.warning(f"num_ctx={self.num_ctx} is at or below Ollama's default window; "
                        "calibration skipped")
            return
        sentence = "the cat sat on the mat and the dog ran to the red car ."
        filler = " ".join(itertools.islice(itertools.cycle(sentence.split()), words))
        result = self.chat("Reply with the single word OK.", filler, num_predict=2,
                           label="(context calibration)")
        seen = result.prompt_eval_count or 0
        if seen < words * 0.9:
            raise ContextOverflowError(
                f"context calibration failed: sent ~{words} tokens, Ollama read {seen}. "
                f"The server is not honouring num_ctx={self.num_ctx}."
            )
        log.info(f"Ollama context window verified: read {seen} tokens (num_ctx={self.num_ctx})")

    def unload(self) -> None:
        """Evict the model from VRAM and wait until Ollama reports it gone."""
        try:
            with self._request("/api/generate", {"model": self.model, "keep_alive": 0}):
                pass
            for _ in range(20):
                with self._request("/api/ps", method="GET") as resp:
                    loaded = {m.get("name") for m in json.loads(resp.read()).get("models", [])}
                if self.model not in loaded and f"{self.model}:latest" not in loaded:
                    log.info(f"Unloaded {self.model} from Ollama")
                    return
                time.sleep(0.5)
            log.warning(f"{self.model} still loaded in Ollama after unload request")
        except OllamaError:
            log.opt(exception=True).warning("Could not unload the Ollama model")


# --- Orchestration -------------------------------------------------------------


@dataclass
class TranslationStats:
    lines: int = 0
    batches: int = 0
    batch_splits: int = 0
    single_line_fallbacks: int = 0
    untranslated: list[int] = field(default_factory=list)
    answer_escalations: int = 0  # batches cut off at num_predict and retried with more room
    # Lines that came back in the source language because they are sentence fragments.
    echo_lines: int = 0
    echo_retries: int = 0
    echoes_fixed: int = 0
    # Lines that came back with the line above them: one line spoken twice, another lost.
    repeated_lines: int = 0
    repeat_retries: int = 0
    repeats_fixed: int = 0
    over_budget_initial: int = 0
    over_budget_final: int = 0
    repair_requests: int = 0
    glossary_misses: dict[int, list[str]] = field(default_factory=dict)
    # Back-translation verification (all zero when it is off).
    verified: bool = False
    verify_backtranslations: int = 0
    verify_comparisons: int = 0
    verify_failures: int = 0  # requests that failed; the first attempt was kept
    verify_echo_retries: int = 0  # second attempts after a line came back unchanged
    flagged: int = 0  # lines whose round trip did not match the source
    revised: int = 0  # flagged lines where the revision was kept
    reverted: int = 0  # flagged lines where the first attempt round-tripped better
    unverifiable: list[int] = field(default_factory=list)  # round trip never answered
    verify_report: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class BatchTranslator:
    def __init__(self, client: OllamaClient, *, source_lang: str, target_lang: str,
                 style_text: str, glossary: list[str], hints: list[Hint] | None = None,
                 batch_lines: int = 30, context_lines: int = 4, lookahead_lines: int = 3,
                 budget_tolerance: float = 1.15, budget_retries: int = 2,
                 verify: bool = False, verify_batch_lines: int = 20,
                 verify_revision_group: int = 8,
                 verify_client: OllamaClient | None = None,
                 verify_retry_temperature: float = 0.8,
                 verify_echo_retries: int = 1,
                 echo_retries: int = 1,
                 verify_report_path=None) -> None:
        self.client = client
        self.src = source_lang
        self.tgt = target_lang
        self.glossary = glossary
        self.hints = hints or []
        self.system = system_prompt(source_lang, target_lang, style_text, glossary, self.hints)
        self.batch_lines = batch_lines
        self.context_lines = context_lines
        self.lookahead_lines = lookahead_lines
        self.tolerance = budget_tolerance
        self.retries = budget_retries
        self.verify = verify
        self.verify_batch_lines = verify_batch_lines
        self.verify_revision_group = verify_revision_group
        # None = verify with the translation model (the default).
        self.verify_client = verify_client
        self.verify_retry_temperature = verify_retry_temperature
        self.verify_echo_retries = verify_echo_retries
        self.echo_retries = echo_retries
        self.verify_report_path = verify_report_path
        self.stats = TranslationStats(verified=verify)

    # -- sizing ----------------------------------------------------------------
    def _prompt_tokens(self, user: str) -> int:
        return (estimate_tokens(self.system) + estimate_tokens(user)
                + _TEMPLATE_OVERHEAD + 16)

    def _answer_budget(self, user: str, lines: list[Line], *,
                       factor: float = ANSWER_HEADROOM) -> int:
        """Tokens to allow the answer: the estimator times ``factor``, inside the window.

        Never below the estimator: ``_fits`` has already established that the prompt plus
        the estimator fits, so the room left is at least the estimator. The cap only ever
        trims the headroom, never the minimum the answer needs.
        """
        need = output_token_estimate(lines, self.tgt)
        room = self.client.num_ctx - self._prompt_tokens(user)
        return max(min(int(need * factor), room), need)

    def _fits(self, user: str, lines: list[Line]) -> tuple[bool, int]:
        need = output_token_estimate(lines, self.tgt)
        return (self._prompt_tokens(user) + need <= self.client.num_ctx,
                self._answer_budget(user, lines))

    def _context(self, lines: list[Line], start: int, done: dict[int, str]):
        ctx = [(ln, done[ln.n]) for ln in lines[max(0, start - self.context_lines):start]
               if ln.n in done]
        return ctx

    # -- main ------------------------------------------------------------------
    def translate(self, lines: list[Line]) -> list[str]:
        """Return one translation per line, in order. Never drops a line."""
        self.stats = TranslationStats(lines=len(lines), verified=self.verify)
        if not lines:
            return []
        done: dict[int, str] = {}
        pos = 0
        while pos < len(lines):
            size = min(self.batch_lines, len(lines) - pos)
            while True:
                batch = lines[pos:pos + size]
                look = lines[pos + size:pos + size + self.lookahead_lines]
                user = batch_prompt(batch, self._context(lines, pos, done), look, self.tgt)
                ok, _ = self._fits(user, batch)
                if ok:
                    break
                if size == 1:
                    raise ContextOverflowError(
                        f"line {batch[0].n} alone does not fit num_ctx={self.client.num_ctx}; "
                        "raise ollama_num_ctx or shorten the glossary/style"
                    )
                size = max(1, size // 2)
            self._translate_batch(lines, pos, size, done)
            log.info(f"[{self.tgt}] translated {min(pos + size, len(lines))}/{len(lines)} lines")
            pos += size
        self._retry_source_echoes(lines, done)
        self._repair_repeated_lines(lines, done)
        if self.verify:
            self._verify_translations(lines, done)
        self._repair_budgets(lines, done)
        self._check_glossary(lines, done)
        return [done[ln.n] for ln in lines]

    def _translate_batch(self, lines: list[Line], pos: int, size: int,
                         done: dict[int, str]) -> None:
        batch = lines[pos:pos + size]
        look = lines[pos + size:pos + size + self.lookahead_lines]
        user = batch_prompt(batch, self._context(lines, pos, done), look, self.tgt)
        label = f"[{self.tgt} lines {batch[0].n}-{batch[-1].n}]"
        # A cut-off answer gets more room before the batch is given up on: the answer was
        # unfinished, not wrong, and halving the batch loses the context that makes the
        # translation consistent with its neighbours.
        budgets = [self._answer_budget(user, batch),
                   self._answer_budget(user, batch, factor=ANSWER_ESCALATION)]
        for attempt, num_predict in enumerate(budgets):
            self.stats.batches += 1
            try:
                result = self.client.chat(self.system, user, num_predict=num_predict,
                                          schema=RESPONSE_SCHEMA, label=label)
                if result.done_reason == "length":
                    raise TruncatedAnswer("answer hit num_predict and was cut off",
                                          budget=num_predict)
                done.update(parse_lines(result.content, [ln.n for ln in batch]))
                return
            except TruncatedAnswer as exc:
                if attempt + 1 < len(budgets) and budgets[attempt + 1] > num_predict:
                    self.stats.answer_escalations += 1
                    log.warning(f"[{self.tgt}] batch {batch[0].n}-{batch[-1].n} {exc} at "
                                f"{num_predict} tokens; retrying with "
                                f"{budgets[attempt + 1]}")
                    continue
                log.warning(f"[{self.tgt}] batch {batch[0].n}-{batch[-1].n} unusable ({exc})")
            except ParseError as exc:
                # A malformed or echoing answer is not a sizing problem; more room would
                # only produce a longer wrong answer.
                log.warning(f"[{self.tgt}] batch {batch[0].n}-{batch[-1].n} unusable ({exc})")
                break
        if size > 1:
            self.stats.batch_splits += 1
            half = size // 2
            self._translate_batch(lines, pos, half, done)
            self._translate_batch(lines, pos + half, size - half, done)
            return
        self._translate_single(lines, pos, done)

    def _translate_single(self, lines: list[Line], pos: int, done: dict[int, str]) -> None:
        line = lines[pos]
        self.stats.single_line_fallbacks += 1
        user = single_line_prompt(line, self._context(lines, pos, done), self.tgt)
        for attempt in range(2):
            try:
                result = self.client.chat(self.system, user,
                                          num_predict=self._answer_budget(user, [line], factor=2.0),
                                          schema=RESPONSE_SCHEMA,
                                          label=f"[{self.tgt} line {line.n}, retry {attempt}]")
                done.update(parse_lines(result.content, [line.n]))
                return
            except ParseError as exc:
                log.warning(f"[{self.tgt}] line {line.n} attempt {attempt + 1} unusable ({exc})")
        log.error(f"[{self.tgt}] line {line.n} could not be translated; keeping source text: "
                  f"{line.text!r}")
        self.stats.untranslated.append(line.n)
        done[line.n] = line.text

    def _retry_source_echoes(self, lines: list[Line], done: dict[int, str]) -> None:
        """One second attempt for lines that came back in the source language.

        Only for lines that are a *continuation*: a word or phrase the transcription split
        off mid-sentence. Such a line has no meaning alone, so handing it back unchanged is
        the model's safe answer, and it is spoken as-is in the dub — "Would J'aimerais
        acheter..." in the French track. A line that stands on its own and comes back
        identical is left alone: "Tanguy", "Toyota", "Morrisons" are correct unchanged.

        The retry is accepted only if it differs from the source and is not wildly longer
        than the fragment's own budget, because the answer for a fragment that has been
        given the whole sentence is sometimes the whole sentence.
        """
        if self.echo_retries < 1:
            return
        candidates = [i for i, line in enumerate(lines)
                      if self._is_continuation(lines, i)
                      and verify_mod.translation_passed_through(line.text, done.get(line.n, ""))]
        if not candidates:
            return
        log.info(f"[{self.tgt}] {len(candidates)} line(s) came back untranslated "
                 f"(fragments of a split sentence); asking again: "
                 f"{[lines[i].n for i in candidates][:8]}")
        self.stats.echo_lines += len(candidates)
        for i in candidates:
            line = lines[i]
            user = fragment_retry_prompt(
                line,
                lines[i - 1].text if i else None,
                lines[i + 1].text if i + 1 < len(lines) else None,
                self.src, self.tgt)
            self.stats.echo_retries += 1
            try:
                result = self.client.chat(
                    self.system, user,
                    num_predict=self._answer_budget(user, [line], factor=2.0),
                    schema=RESPONSE_SCHEMA, label=f"[{self.tgt} fragment {line.n}]")
                answer = parse_lines(result.content, [line.n])
            except ParseError as exc:
                log.debug(f"[{self.tgt}] fragment {line.n} retry unusable ({exc})")
                continue
            got = answer.get(line.n, "").strip()
            if not got or verify_mod.translation_passed_through(line.text, got):
                log.debug(f"[{self.tgt}] fragment {line.n} came back unchanged again")
                continue
            if len(got) > max(line.budget, len(line.text) * 3) * 2:
                log.debug(f"[{self.tgt}] fragment {line.n} retry returned the whole sentence "
                          f"({len(got)} chars); keeping the original")
                continue
            done[line.n] = got
            self.stats.echoes_fixed += 1
            log.success(f"[{self.tgt}] line {line.n} was untranslated ({line.text.strip()!r}) "
                        f"-> {got!r}")

    @staticmethod
    def _content_words(text: str) -> set[str]:
        import re

        return {w for w in re.findall(r"[\w\u0900-\u097F]+", text.casefold())
                if len(w) > 3}

    def _repair_repeated_lines(self, lines: list[Line], done: dict[int, str]) -> None:
        """Re-ask for a line that came back with the line above it's text.

        Measured on hydro.wav: in German, Hindi, Polish and Spanish a line was answered
        with the *next* line's text and then that text was repeated, so one source line's
        content appeared nowhere in the track — German lost "completely out of action, a
        fantastic depot", Polish lost a carriage-shunting line, Spanish lost the closing
        sentence. The wording was fine everywhere; the mapping from line to line was not.

        Both lines of the pair are re-asked, because which of the two is the duplicate
        cannot be told from the text alone. A retry is accepted only if it no longer
        repeats its neighbour, so this can turn a repeated line into a distinct one and
        never the other way round.
        """
        if self.echo_retries < 1:
            return
        words = self._content_words
        pairs = []
        for i in range(1, len(lines)):
            mine, above = done.get(lines[i].n, ""), done.get(lines[i - 1].n, "")
            if not mine or not above:
                continue
            same = words(mine) & words(above)
            if not same or len(same) / max(1, len(words(mine) | words(above))) < 0.6:
                continue
            # Identical sources mean the duplicate is correct (Whisper repeats itself).
            src_same = words(lines[i].text) & words(lines[i - 1].text)
            if src_same and len(src_same) / max(1, len(words(lines[i].text)
                                                      | words(lines[i - 1].text))) >= 0.6:
                continue
            pairs.append((i - 1, i))
        if not pairs:
            return
        log.warning(f"[{self.tgt}] {len(pairs)} line(s) came back with the line above them "
                    f"({[lines[i].n for _, i in pairs]}); translating them again so no line "
                    "is spoken twice and none is skipped")
        self.stats.repeated_lines += len(pairs)
        for above_i, i in pairs:
            line = lines[i]
            user = repeat_retry_prompt(
                line,
                lines[above_i].text,
                lines[i + 1].text if i + 1 < len(lines) else None,
                done.get(line.n, ""), self.src, self.tgt)
            self.stats.repeat_retries += 1
            try:
                result = self.client.chat(
                    self.system, user,
                    num_predict=self._answer_budget(user, [line], factor=2.0),
                    schema=RESPONSE_SCHEMA, label=f"[{self.tgt} repeat {line.n}]")
                answer = parse_lines(result.content, [line.n])
            except ParseError as exc:
                log.debug(f"[{self.tgt}] line {line.n} repeat retry unusable ({exc})")
                continue
            got = answer.get(line.n, "").strip()
            if not got or verify_mod.translation_passed_through(line.text, got):
                continue
            still = words(got) & words(done.get(lines[above_i].n, ""))
            if still and len(still) / max(1, len(words(got) | words(done[lines[above_i].n]))) >= 0.6:
                log.debug(f"[{self.tgt}] line {line.n} repeated the line above again")
                continue
            done[line.n] = got
            self.stats.repeats_fixed += 1
            log.success(f"[{self.tgt}] line {line.n} was a copy of line {lines[above_i].n}; "
                        f"now {got!r}")

    @staticmethod
    def _is_continuation(lines: list[Line], i: int) -> bool:
        """True if line ``i`` is the middle of a sentence rather than a sentence itself."""
        prev = lines[i - 1].text.strip() if i else ""
        nxt = lines[i + 1].text.strip() if i + 1 < len(lines) else ""
        if prev and not _SENTENCE_END.search(prev):
            return True
        return bool(nxt) and nxt[0].islower()

    def _repair_budgets(self, lines: list[Line], done: dict[int, str]) -> None:
        by_n = {ln.n: ln for ln in lines}
        all_ns = [ln.n for ln in lines]
        flagged = lambda: [n for n in all_ns  # noqa: E731
                           if needs_repair(done[n], by_n[n].budget, self.tolerance)]
        initial = flagged()
        self.stats.over_budget_initial = len(initial)
        for _ in range(self.retries):
            too_long = flagged()
            if not too_long:
                break
            for group in repair_groups(too_long, all_ns):
                window = [(by_n[n], done[n]) for n in group]
                before = sum(overshoot(done[n], by_n[n].budget) for n in group)
                user = repair_prompt(window, set(too_long), self.src, self.tgt)
                self.stats.repair_requests += 1
                try:
                    result = self.client.chat(
                        self.system, user, schema=RESPONSE_SCHEMA,
                        num_predict=self._answer_budget(user, [by_n[n] for n in group]),
                        label=f"[{self.tgt} shorten {group[0]}-{group[-1]}]",
                    )
                    revised = parse_lines(result.content, group)
                except ParseError as exc:
                    log.debug(f"[{self.tgt}] shorten {group} unusable ({exc}); keeping previous")
                    continue
                after = sum(overshoot(revised[n], by_n[n].budget) for n in group)
                if after < before:  # keep the best result, never a worse one
                    done.update(revised)
        self.stats.over_budget_final = len(flagged())
        if initial:
            log.info(f"[{self.tgt}] over budget: {len(initial)} lines before shortening, "
                     f"{self.stats.over_budget_final} after")

    def _check_glossary(self, lines: list[Line], done: dict[int, str]) -> None:
        for ln in lines:
            misses = glossary_misses(ln.text, done[ln.n], self.glossary)
            if misses:
                self.stats.glossary_misses[ln.n] = misses
        if self.stats.glossary_misses:
            sample = list(self.stats.glossary_misses.items())[:5]
            log.warning(f"[{self.tgt}] protected terms missing from "
                        f"{len(self.stats.glossary_misses)} lines, e.g. {sample} "
                        f"(check the review SRT; may be legitimate inflection)")

    # -- back-translation verification ------------------------------------------
    @staticmethod
    def _chunks(numbers: list[int], size: int) -> Iterator[list[int]]:
        for start in range(0, len(numbers), size):
            yield numbers[start:start + size]

    def _verify_translations(self, lines: list[Line], done: dict[int, str]) -> None:
        """Round-trip every line, revise the ones whose meaning changed, cap at one round.

        Failing verification is never fatal and never loses a line: a line whose back
        translation cannot be obtained is left exactly as first translated and reported
        as unverifiable. A revision replaces a line only when its own round trip comes
        back clean, and it is re-verified together with the other revisions in its group,
        which keeps the comparison in context instead of judging lines one at a time.
        """
        by_n = {ln.n: ln for ln in lines}
        attempt = dict(done)  # every first-pass translation, kept for comparison
        # Lines left in the source language by the translation fallbacks are not
        # verified: a round trip of the source is not evidence about anything.
        numbers = [ln.n for ln in lines if ln.n not in self.stats.untranslated]
        if not numbers:
            return
        started = time.monotonic()
        back = self._backtranslate(numbers, attempt)
        back = self._retry_echoes(numbers, attempt, back)
        checkable = self._checkable(numbers, attempt, back)
        verdicts = self._compare(checkable, by_n, back)
        # A line the comparison could not be obtained for has no verdict, so it is not
        # flagged: it stays exactly as translated and the report says so.
        self.stats.unverifiable = sorted(set(self.stats.unverifiable)
                                         | (set(checkable) - set(verdicts)))
        flagged = [n for n in numbers if n in verdicts and verdicts[n].drift]
        self.stats.flagged = len(flagged)

        drifts = [verify_mod.Drift(n=n, source=by_n[n].text, attempt=attempt[n],
                                   back=back[n], problem=verdicts[n].problem,
                                   term=verdicts[n].term)
                  for n in flagged]
        by_drift = {d.n: d for d in drifts}
        for group in self._revision_groups(flagged):
            revised = self._revise(group, by_n, attempt, back,
                                   {n: by_drift[n].problem for n in group})
            if not revised:
                for n in group:
                    by_drift[n].note = ("the revision could not be parsed; the first "
                                        "attempt was kept")
                continue
            revised_back = self._backtranslate(sorted(revised), revised)
            revised_back = self._retry_echoes(sorted(revised), revised, revised_back)
            checkable = self._checkable(sorted(revised), revised, revised_back)
            revised_verdicts = self._compare(checkable, by_n, revised_back)
            for n in group:
                drift = by_drift[n]
                drift.revision = revised.get(n)
                drift.revised_back = revised_back.get(n)
                if n not in revised:
                    drift.note = ("the revision could not be parsed; the first attempt "
                                  "was kept")
                    continue
                if n not in revised_verdicts:
                    drift.note = ("the revision could not be checked, so it is unproven; "
                                  "the first attempt was kept")
                    continue
                if revised_verdicts[n].drift:
                    drift.remaining = [revised_verdicts[n].problem]
                    drift.note = "the revision drifted too; the first attempt was kept"
                    continue
                drift.kept = "revision"

        for drift in drifts:
            if drift.kept == "revision" and drift.revision:
                done[drift.n] = drift.revision
                self.stats.revised += 1
            else:
                self.stats.reverted += 1

        unverifiable = sorted(self.stats.unverifiable)
        self._write_verify_report(lines, drifts)
        log.info(f"[{self.tgt}] verify: {len(numbers)} lines round-tripped, "
                 f"{len(flagged)} flagged, {self.stats.revised} revised, "
                 f"{self.stats.reverted} kept as first translated"
                 + (f", {len(unverifiable)} not verifiable" if unverifiable else "")
                 + f" ({time.monotonic() - started:.0f}s, "
                 f"{self.stats.verify_backtranslations} back-translation and "
                 f"{self.stats.verify_comparisons} comparison request(s))")
        if self.stats.verify_failures:
            log.warning(f"[{self.tgt}] verify: {self.stats.verify_failures} request(s) "
                        "failed; those lines were left as first translated")
        if drifts and self.stats.revised == 0:
            log.warning(f"[{self.tgt}] verify flagged {len(drifts)} line(s) but kept no "
                        "revision: the drift may be beyond this model (check the report)")
        if unverifiable:
            log.warning(f"[{self.tgt}] verify could not round-trip {len(unverifiable)} "
                        f"line(s) {unverifiable[:8]}; they were left as translated")

    def _revision_groups(self, flagged: list[int]) -> list[list[int]]:
        """Flagged lines in contiguous groups, so neighbours are revised together (and
        a line can still hand words to the line beside it)."""
        groups: list[list[int]] = []
        for n in flagged:
            if groups and n == groups[-1][-1] + 1 \
                    and len(groups[-1]) < self.verify_revision_group:
                groups[-1].append(n)
            else:
                groups.append([n])
        return groups

    def _checkable(self, numbers: list[int], texts: dict[int, str],
                   back: dict[int, str]) -> list[int]:
        """Lines whose round trip is evidence: answered, and not just the input echoed
        back (see :func:`verify.backtranslation_passed_through`). A line that fails either
        test is recorded as unverifiable and left as translated."""
        usable, unusable = [], []
        for n in numbers:
            if n not in back:
                unusable.append(n)
            elif verify_mod.backtranslation_passed_through(texts[n], back[n]):
                unusable.append(n)
                log.warning(f"[{self.tgt}] verify: line {n} came back unchanged instead "
                            "of translated; left as translated")
            else:
                usable.append(n)
        self.stats.unverifiable = sorted(set(self.stats.unverifiable) | set(unusable))
        return usable

    def _backtranslate(self, numbers: list[int], texts: dict[int, str],
                       *, retry: bool = False) -> dict[int, str]:
        """Literal renderings of ``texts`` back into the source language, batched."""
        out: dict[int, str] = {}
        system = (verify_mod.backtranslate_retry_system_prompt(self.src, self.tgt)
                  if retry else verify_mod.backtranslate_system_prompt(self.src, self.tgt))

        def build(chunk: list[int]) -> str:
            return verify_mod.backtranslate_prompt([(n, texts[n]) for n in chunk],
                                                   self.src, self.tgt)

        for group in self._chunks(numbers, self.verify_batch_lines):
            chunk, user = self._shrink(
                system, group, build,
                lambda c: self._generate_budget([texts[n] for n in c], self.src))
            if chunk is None:
                self._verify_unfittable(group, "back-translated")
                continue
            answers = self._request_lines(
                system, user, chunk, self.src, schema=verify_mod.lines_schema(),
                label=(f"[{self.tgt} back-translation {chunk[0]}-{chunk[-1]}"
                       + (", retry" if retry else "") + "]"),
                count_stat="verify_backtranslations", client=self._verify_client(),
                temperature=self.verify_retry_temperature if retry else None)
            if answers:
                out.update(answers[0])
        return out

    def _retry_echoes(self, numbers: list[int], texts: dict[int, str],
                      back: dict[int, str]) -> dict[int, str]:
        """One second attempt for lines that came back unchanged, rephrased and hotter.

        A copied line is the one failure that actively misleads: the comparison sees the
        round trip "match" the source and confirms a wrong translation as correct. So
        rather than only recording it as unverifiable, the pass asks again — same task,
        framed as a translator who does not know the language, at a higher temperature,
        which is what broke the tie when this was measured. One retry, only for the lines
        that need it, and anything still unchanged is reported as unverifiable.
        """
        echoes = [n for n in numbers
                  if n in back and verify_mod.backtranslation_passed_through(texts[n],
                                                                            back[n])]
        if not echoes or self.verify_echo_retries < 1:
            return back
        log.info(f"[{self.tgt}] verify: retrying {len(echoes)} line(s) that came back "
                 f"untranslated: {echoes[:8]}")
        self.stats.verify_echo_retries += 1
        again = self._backtranslate(echoes, texts, retry=True)
        for n in echoes:
            if n in again and not verify_mod.backtranslation_passed_through(texts[n],
                                                                           again[n]):
                back[n] = again[n]
        return back

    def _compare(self, numbers: list[int], by_n: dict[int, Line],
                 back: dict[int, str]) -> dict[int, verify_mod.Verdict]:
        """``{n: Verdict}`` for the lines that could be round-tripped."""
        out: dict[int, verify_mod.Verdict] = {}
        system = verify_mod.compare_system_prompt(self.src, self.tgt)
        for group in self._chunks(numbers, self.verify_batch_lines):
            chunk, _ = self._shrink(
                system, group,
                lambda c: verify_mod.compare_prompt(
                    [(n, by_n[n].text, back[n]) for n in c], self.src, self.tgt),
                lambda c: self._generate_budget([back[n] for n in c], self.src))
            if chunk is None:
                self._verify_unfittable(group, "compared")
                continue
            num_predict = self._generate_budget([back[n] for n in chunk], self.src)
            out.update(self._compare_chunk(system, chunk, back, by_n, num_predict))
        return out

    def _compare_chunk(self, system: str, numbers: list[int], back: dict[int, str],
                       by_n: dict[int, Line],
                       num_predict: int) -> dict[int, verify_mod.Verdict]:
        """One comparison request, split in half while it keeps failing.

        A comparison that cannot be obtained is *not* a drift verdict: the line is
        reported as unverifiable rather than revised on no evidence. A retry of the same
        request is not a retry — the same prompt to a deterministic server gets the same
        unusable answer, so a failure narrows the chunk instead.
        """
        user = verify_mod.compare_prompt(
            [(n, by_n[n].text, back[n]) for n in numbers], self.src, self.tgt)
        sources = {n: by_n[n].text for n in numbers}
        try:
            result = self._verify_client().chat(
                system, user, num_predict=max(num_predict, 16 * len(numbers)),
                schema=verify_mod.COMPARISON_SCHEMA,
                label=f"[{self.tgt} verify compare {numbers[0]}-{numbers[-1]}]")
            self.stats.verify_comparisons += 1
            return verify_mod.parse_comparison(result.content, sources)
        except (ParseError, ContextOverflowError) as exc:
            self.stats.verify_failures += 1
            if len(numbers) > 1:
                half = len(numbers) // 2
                first = self._compare_chunk(system, numbers[:half], back, by_n, num_predict)
                second = self._compare_chunk(system, numbers[half:], back, by_n, num_predict)
                return {**first, **second}
            log.debug(f"[{self.tgt}] verify: comparison of line {numbers[0]} unusable "
                      f"({exc}); left as translated")
            return {}

    def _verify_client(self) -> OllamaClient:
        """The client used for back-translation and comparison: the main model unless a
        separate verification model is configured. Exactly one model is resident at a
        time here — the pipeline unloads the translation model between stages, so the
        verification model is loaded on the first verification request and used for the
        rest of the pass."""
        return self.verify_client or self.client

    def _request_lines(self, system: str, user: str, numbers: list[int], answer_lang: str,
                       *, schema: dict, label: str, count_stat: str | None = None,
                       client: OllamaClient | None = None,
                       temperature: float | None = None) -> list[dict[int, str]]:
        """Strict answers for ``numbers``; ``[]`` when none could be obtained.

        ``answer_lang`` is the language the answer is written in, which is what its size
        has to be estimated from: a Polish answer is longer than the English source it
        came from, and the line's character budget describes the translation, not the
        round trip.
        """
        lines = [Line(n=n, text="x" * 40, budget=40) for n in numbers]
        chat = client or self.client
        for attempt in range(2):
            try:
                result = chat.chat(
                    system, user, num_predict=output_token_estimate(lines, answer_lang) * 2,
                    schema=schema, label=f"{label}, retry {attempt}",
                    temperature=temperature)
                if count_stat:
                    setattr(self.stats, count_stat, getattr(self.stats, count_stat) + 1)
                return [parse_lines(result.content, numbers)]
            except (ParseError, ContextOverflowError) as exc:
                log.debug(f"[{self.tgt}] {label} unusable ({exc})")
                if len(numbers) > 1:
                    half = len(numbers) // 2
                    return (self._request_lines(system, user, numbers[:half], answer_lang,
                                                schema=schema, label=label,
                                                count_stat=count_stat, client=client,
                                                temperature=temperature)
                            + self._request_lines(system, user, numbers[half:], answer_lang,
                                                  schema=schema, label=label,
                                                  count_stat=count_stat, client=client,
                                                  temperature=temperature))
        self.stats.verify_failures += 1
        return []

    def _shrink(self, system: str, group: list[int], build, budget) -> tuple[list[int] | None, str]:
        """Largest prefix of ``group`` whose prompt and expected answer fit ``num_ctx``.

        ``budget`` maps a chunk to the answer size to allow for it. ``None`` when not
        even one line fits — the caller then reports those lines instead of sending a
        prompt Ollama would silently truncate.
        """
        size = len(group)
        while True:
            chunk = group[:size]
            user = build(chunk)
            if self._fits_prompt(system, user, budget(chunk)):
                return chunk, user
            if size == 1:
                return None, user
            size = max(1, size // 2)

    def _verify_unfittable(self, numbers: list[int], action: str) -> None:
        self.stats.verify_failures += 1
        log.warning(f"[{self.tgt}] verify: line {numbers[0]} cannot be {action} within "
                    f"num_ctx={self.client.num_ctx}; left as translated")

    def _revise(self, numbers: list[int], by_n: dict[int, Line], attempt: dict[int, str],
                back: dict[int, str], problems: dict[int, str]) -> dict[int, str]:
        """Re-translate the flagged lines, telling the model what its attempt means.

        The revision prompt carries the full translation system prompt (glossary, hints,
        style) as well as the repair instructions. Without it the model repairs the named
        drift while breaking a pinned term — rewriting a hinted "patty" as "burger" — and
        it is only asked to revise a line at all because it got something wrong.

        A revision request is neither a back-translation nor a comparison, so it is
        counted only as a failure when it cannot be obtained at all.
        """
        system = f"{self.system}\n\n{verify_mod.revision_system_prompt(self.src, self.tgt)}"
        user = verify_mod.revision_prompt(
            [(by_n[n], attempt[n], back[n], problems[n]) for n in numbers],
            self.src, self.tgt)
        answers = self._request_lines(system, user, numbers, self.tgt,
                                      schema=RESPONSE_SCHEMA,
                                      label=f"[{self.tgt} revision {numbers[0]}-"
                                            f"{numbers[-1]}]")
        return answers[0] if answers else {}

    def _generate_budget(self, texts: list[str], lang: str) -> int:
        """A pessimistic allowance for an answer that renders ``texts`` in ``lang``.

        A back-translation answers in the *source* language, which is not ``self.tgt``,
        and it can be longer than the text it came from, so its size is estimated from
        the text itself rather than from the line's character budget."""
        return output_token_estimate(
            [Line(n=i + 1, text=t, budget=len(t)) for i, t in enumerate(texts)], lang)

    def _fits_prompt(self, system: str, user: str, num_predict: int) -> bool:
        """The pre-send size check for a prompt that is not a translation request.

        Every prompt goes through one, because Ollama truncates silently rather than
        erroring, and a truncated verification prompt would produce a confident verdict
        about text the model never saw.
        """
        need = (estimate_tokens(system) + estimate_tokens(user) + num_predict
                + _TEMPLATE_OVERHEAD + 16)
        return need <= self.client.num_ctx

    def _write_verify_report(self, lines: list[Line], drifts: list[verify_mod.Drift]) -> None:
        if self.verify_report_path is None:
            return
        from ytdub.cache import atomic_write_text

        text = verify_mod.render_report(
            self.src, self.tgt, lines, drifts, flagged=self.stats.flagged,
            revised=self.stats.revised, unverified=self.stats.unverifiable,
            untranslated=list(self.stats.untranslated))
        try:
            atomic_write_text(self.verify_report_path, text)
            self.stats.verify_report = str(self.verify_report_path)
            log.info(f"[{self.tgt}] verification report: {self.verify_report_path}")
        except OSError as exc:
            log.warning(f"[{self.tgt}] could not write the verification report: {exc}")


def _code_hash() -> str:
    """Hash of the prompt, verification and orchestration code: editing any of them
    invalidates cached translations even if nobody remembers to bump PROMPT_VERSION."""
    from pathlib import Path

    from ytdub.cache import sha256_text

    here = Path(__file__).resolve().parent
    return sha256_text("".join((here / f).read_text(encoding="utf-8")
                               for f in ("prompt.py", "verify.py", "ollama.py")))[:16]


class OllamaTranslator:
    """The shipped :class:`~ytdub.stages.translate.base.Translator` backend."""

    name = "ollama"

    def __init__(self, settings, style_text: str, glossary: list[str]) -> None:
        self.s = settings
        self.style_text = style_text
        self.glossary = glossary
        # Hints are per target language, so they are set per call (see ``set_hints``).
        self.hints: list[Hint] = []
        self.client = OllamaClient(settings.ollama_url, settings.ollama_model,
                                   num_ctx=settings.ollama_num_ctx,
                                   temperature=settings.ollama_temperature,
                                   timeout=settings.ollama_timeout)
        # A separate model for verification only, when one is configured. Translation
        # stays on its own model; the two never need to be resident together, because
        # verification runs after a language's translation is finished.
        self.verify_model = verify_model(settings)
        self.verify_client = (
            OllamaClient(settings.ollama_url, self.verify_model,
                         num_ctx=settings.ollama_num_ctx,
                         temperature=settings.ollama_temperature,
                         timeout=settings.ollama_timeout)
            if self.verify_model != settings.ollama_model else None)
        self._ready = False
        self._verify_ready = False

    def set_hints(self, hints: list[Hint]) -> None:
        """``hints.<lang>.txt`` for the language about to be translated. The pipeline
        puts the same list in that language's cache key itself, so a changed hint file
        is a miss whether or not this is called."""
        self.hints = list(hints)

    def cache_identity(self) -> dict:
        from ytdub.stages.translate.prompt import PROMPT_VERSION

        s = self.s
        return {
            "model": s.ollama_model, "num_ctx": s.ollama_num_ctx,
            "temperature": s.ollama_temperature,
            "batch_lines": s.translate_batch_lines, "context_lines": s.translate_context_lines,
            "lookahead_lines": s.translate_lookahead_lines,
            "budget_tolerance": s.budget_tolerance, "budget_retries": s.budget_retries,
            # Verification changes the output, so it is part of the key: turning it on
            # must retranslate rather than serve a cached unverified translation. The
            # model that verifies is part of it for the same reason.
            "verify": s.verify, "verify_batch_lines": s.verify_batch_lines,
            "verify_model": self.verify_model,
            "verify_echo_retries": s.verify_echo_retries,
            "translation_echo_retries": s.translation_echo_retries,
            "prompt_version": PROMPT_VERSION, "code": _code_hash(),
        }

    def weights_fingerprint(self) -> str | None:
        try:
            return self.client.model_digest()
        except OllamaError:
            return None

    def translate(self, lines: list[Line], *, source_lang: str,
                  target_lang: str) -> tuple[list[str], dict]:
        if not self._ready:
            self.client.check_model()
            self.client.verify_context_window()
            self._ready = True
        verify_client = None
        if self.s.verify and self.verify_client is not None:
            if not self._verify_ready:
                # Fails loudly before any translation if the verification model is not
                # pulled: a pass that silently verifies with the wrong model is worse
                # than one that never starts.
                self.verify_client.check_model()
                self._verify_ready = True
            verify_client = self.verify_client
            log.info(f"[{target_lang}] verification model: {self.verify_model} "
                     f"(translation: {self.s.ollama_model})")
        bt = BatchTranslator(
            self.client, source_lang=source_lang, target_lang=target_lang,
            style_text=self.style_text, glossary=self.glossary, hints=self.hints,
            batch_lines=self.s.translate_batch_lines,
            context_lines=self.s.translate_context_lines,
            lookahead_lines=self.s.translate_lookahead_lines,
            budget_tolerance=self.s.budget_tolerance, budget_retries=self.s.budget_retries,
            verify=self.s.verify, verify_batch_lines=self.s.verify_batch_lines,
            verify_client=verify_client,
            verify_retry_temperature=self.s.verify_temperature,
            verify_echo_retries=self.s.verify_echo_retries,
            echo_retries=self.s.translation_echo_retries,
            verify_report_path=(self.s.verify_report_path if self.s.verify else None),
        )
        texts = bt.translate(lines)
        return texts, bt.stats.to_dict()

    def unload(self) -> None:
        if self._ready:
            self.client.unload()
        if self._verify_ready and self.verify_client is not None:
            # Nothing is left resident: the TTS stage wants the VRAM back.
            self.verify_client.unload()


def verify_model(settings) -> str:
    """The model used for the verification pass, or the translation model when no
    separate one is configured."""
    return settings.verify_ollama_model or settings.ollama_model
