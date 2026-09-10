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
"""

from __future__ import annotations

import itertools
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from ytdub.logging import stage_logger
from ytdub.stages.translate.prompt import (
    RESPONSE_SCHEMA,
    Line,
    ParseError,
    batch_prompt,
    estimate_tokens,
    glossary_misses,
    lower_bound_tokens,
    needs_repair,
    output_token_estimate,
    overshoot,
    parse_lines,
    repair_groups,
    repair_prompt,
    single_line_prompt,
    strip_thinking,
    system_prompt,
)

log = stage_logger("translate")

# Chat-template tokens Ollama wraps around the messages.
_TEMPLATE_OVERHEAD = 32


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
             label: str = "") -> ChatResult:
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
                        "temperature": self.temperature},
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
    over_budget_initial: int = 0
    over_budget_final: int = 0
    repair_requests: int = 0
    glossary_misses: dict[int, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BatchTranslator:
    def __init__(self, client: OllamaClient, *, source_lang: str, target_lang: str,
                 style_text: str, glossary: list[str], batch_lines: int = 30,
                 context_lines: int = 4, lookahead_lines: int = 3,
                 budget_tolerance: float = 1.15, budget_retries: int = 2) -> None:
        self.client = client
        self.src = source_lang
        self.tgt = target_lang
        self.glossary = glossary
        self.system = system_prompt(source_lang, target_lang, style_text, glossary)
        self.batch_lines = batch_lines
        self.context_lines = context_lines
        self.lookahead_lines = lookahead_lines
        self.tolerance = budget_tolerance
        self.retries = budget_retries
        self.stats = TranslationStats()

    # -- sizing ----------------------------------------------------------------
    def _fits(self, user: str, lines: list[Line]) -> tuple[bool, int]:
        num_predict = output_token_estimate(lines, self.tgt)
        need = (estimate_tokens(self.system) + estimate_tokens(user) + num_predict
                + _TEMPLATE_OVERHEAD + 16)
        return need <= self.client.num_ctx, num_predict

    def _context(self, lines: list[Line], start: int, done: dict[int, str]):
        ctx = [(ln, done[ln.n]) for ln in lines[max(0, start - self.context_lines):start]
               if ln.n in done]
        return ctx

    # -- main ------------------------------------------------------------------
    def translate(self, lines: list[Line]) -> list[str]:
        """Return one translation per line, in order. Never drops a line."""
        self.stats = TranslationStats(lines=len(lines))
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
        self._repair_budgets(lines, done)
        self._check_glossary(lines, done)
        return [done[ln.n] for ln in lines]

    def _translate_batch(self, lines: list[Line], pos: int, size: int,
                         done: dict[int, str]) -> None:
        batch = lines[pos:pos + size]
        look = lines[pos + size:pos + size + self.lookahead_lines]
        user = batch_prompt(batch, self._context(lines, pos, done), look, self.tgt)
        _, num_predict = self._fits(user, batch)
        self.stats.batches += 1
        try:
            result = self.client.chat(self.system, user, num_predict=num_predict,
                                      schema=RESPONSE_SCHEMA,
                                      label=f"[{self.tgt} lines {batch[0].n}-{batch[-1].n}]")
            if result.done_reason == "length":
                raise ParseError("answer hit num_predict and was cut off")
            done.update(parse_lines(result.content, [ln.n for ln in batch]))
            return
        except ParseError as exc:
            log.warning(f"[{self.tgt}] batch {batch[0].n}-{batch[-1].n} unusable ({exc})")
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
                                          num_predict=output_token_estimate([line], self.tgt) * 2,
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
                        num_predict=output_token_estimate([by_n[n] for n in group], self.tgt),
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


def _code_hash() -> str:
    """Hash of the prompt and orchestration code: editing either invalidates cached
    translations even if nobody remembers to bump PROMPT_VERSION."""
    from pathlib import Path

    from ytdub.cache import sha256_text

    here = Path(__file__).resolve().parent
    return sha256_text("".join((here / f).read_text(encoding="utf-8")
                               for f in ("prompt.py", "ollama.py")))[:16]


class OllamaTranslator:
    """The shipped :class:`~ytdub.stages.translate.base.Translator` backend."""

    name = "ollama"

    def __init__(self, settings, style_text: str, glossary: list[str]) -> None:
        self.s = settings
        self.style_text = style_text
        self.glossary = glossary
        self.client = OllamaClient(settings.ollama_url, settings.ollama_model,
                                   num_ctx=settings.ollama_num_ctx,
                                   temperature=settings.ollama_temperature,
                                   timeout=settings.ollama_timeout)
        self._ready = False

    def cache_identity(self) -> dict:
        from ytdub.stages.translate.prompt import PROMPT_VERSION

        s = self.s
        return {
            "model": s.ollama_model, "num_ctx": s.ollama_num_ctx,
            "temperature": s.ollama_temperature,
            "batch_lines": s.translate_batch_lines, "context_lines": s.translate_context_lines,
            "lookahead_lines": s.translate_lookahead_lines,
            "budget_tolerance": s.budget_tolerance, "budget_retries": s.budget_retries,
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
        bt = BatchTranslator(
            self.client, source_lang=source_lang, target_lang=target_lang,
            style_text=self.style_text, glossary=self.glossary,
            batch_lines=self.s.translate_batch_lines,
            context_lines=self.s.translate_context_lines,
            lookahead_lines=self.s.translate_lookahead_lines,
            budget_tolerance=self.s.budget_tolerance, budget_retries=self.s.budget_retries,
        )
        texts = bt.translate(lines)
        return texts, bt.stats.to_dict()

    def unload(self) -> None:
        if self._ready:
            self.client.unload()
