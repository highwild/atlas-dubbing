"""Job orchestration.

    ── once per source file, cached ─────────────────────────────
    acquire -> (separate) -> transcribe -> (diarize) -> voice references
    ── per language ─────────────────────────────────────────────
    translate (all languages, Ollama resident)  -> <lang>.review.srt
    [unload Ollama]                             -> stop here with --srt-only
    synthesize -> fit -> assemble  (all languages, Chatterbox resident)

Translating every language before synthesizing any means each model is loaded once
per job and never shares VRAM with another. Every stage result is cached on disk
(see ``cache.py``), so re-running an interrupted job resumes it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ytdub.cache import StageCache, atomic_write_text, sha256_file, sha256_text
from ytdub.config import Settings
from ytdub.logging import add_file_sink, stage_logger
from ytdub.models import Segment, Source, SpeakerRef

if TYPE_CHECKING:
    from ytdub.stages.translate.prompt import Hint

log = stage_logger("pipeline")

# Spec: output duration equals input duration "to within a few milliseconds". The
# sample count is exact by construction; this bounds container-level rounding.
DURATION_TOLERANCE_MS = 2.0


@dataclass
class LanguageResult:
    lang: str
    status: str = "pending"  # ok | degraded | failed | srt-only
    error: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    fit: dict | None = None
    translation: dict | None = None
    loudness: dict | None = None
    merged_fragments: int = 0
    lost_segments: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def model_fingerprint(spec: str) -> dict | None:
    """For a local model directory, file names + sizes + mtimes, so replacing the
    weights under the same path invalidates the transcript. None for a size name."""
    path = Path(spec).expanduser()
    if not path.is_dir():
        return None
    return {f.name: [f.stat().st_size, int(f.stat().st_mtime)]
            for f in sorted(path.iterdir()) if f.is_file()}


def voice_clip_paths(job) -> list[Path]:
    """The stored voice clips a job would match against, for the reference cache key."""
    if job.s.voice_path is None:
        return []
    from ytdub.stages.voices import voice_clips

    return voice_clips(job.s.voice_path)


def resolve_input(arg: str, settings: Settings) -> Path:
    from ytdub.stages.download import download, is_url

    if is_url(arg):
        return download(arg, settings.input_dir, force_ipv4=settings.force_ipv4,
                        timeout=settings.net_timeout,
                        cookies_from_browser=settings.cookies_from_browser,
                        cookies_file=settings.cookies_file)
    for candidate in (settings.input_dir / arg, Path(arg).expanduser()):
        if candidate.is_file():
            return candidate.resolve()
    have = sorted(p.name for p in settings.input_dir.glob("*")) if settings.input_dir.is_dir() else []
    raise FileNotFoundError(f"{arg!r} not found in {settings.input_dir} "
                            f"(available: {', '.join(have) or 'none'})")


class Job:
    def __init__(self, settings: Settings, input_arg: str, *, srt_only: bool = False,
                 from_review: bool = False, user_refs: dict[str | None, Path] | None = None) -> None:
        self.s = settings
        self.input_arg = input_arg
        self.srt_only = srt_only
        self.from_review = from_review
        self.user_refs = user_refs or {}
        self.results: dict[str, LanguageResult] = {}
        # {folded label: surviving label}, filled by _references when a stored voice clip
        # matched more than one diarization label.
        self.speaker_fold: dict[str, str] = {}

    # ------------------------------------------------------------------ front half
    def _acquire(self) -> None:
        from ytdub.ffmpeg import probe

        path = resolve_input(self.input_arg, self.s)
        log.info(f"Hashing {path.name}")
        sha = sha256_file(path)
        info = probe(path)
        if not info.has_audio:
            raise RuntimeError(f"{path} has no audio stream")
        self.src = Source(path=path, basename=path.stem, sha256=sha, duration=info.duration,
                          has_video=info.has_video)
        self.job_dir = self.s.work_root / f"{path.stem}-{sha[:12]}"
        self.out_dir = self.s.output_root / path.stem
        self.out_dir.mkdir(parents=True, exist_ok=True)
        add_file_sink(self.out_dir / "dub.log")
        self.cache = StageCache(self.job_dir / "cache", force=self.s.force)
        log.info(f"Source: {path} | {info.duration:.3f}s | "
                 f"{'video+audio' if info.has_video else 'audio only'} | sha256 {sha[:12]}")

    def _prepare_audio(self) -> None:
        from ytdub.audio import num_samples
        from ytdub.ffmpeg import Loudness, extract_audio, measure_loudness

        speech_src = self.src.path
        if self.s.separate:
            from ytdub.stages.separate import isolate_vocals

            speech_src = isolate_vocals(self.src.path, self.job_dir / "separated",
                                        model=self.s.demucs_model, device=self.s.device)
        tag = f"-{self.s.demucs_model}" if self.s.separate else ""
        self.asr_wav = self.job_dir / f"audio16k{tag}.wav"
        self.ref_wav = self.job_dir / f"audio24k{tag}.wav"
        for path, sr in ((self.asr_wav, 16000), (self.ref_wav, 24000)):
            if not path.exists():
                extract_audio(speech_src, path, sample_rate=sr)

        if self.src.duration <= 0:  # container without a duration: use decoded length
            frames, sr = num_samples(self.ref_wav)
            self.src.duration = frames / sr

        self.source_loudness = None
        if self.s.match_loudness:
            ref = Path(self.s.loudness_reference) if self.s.loudness_reference else self.src.path
            cache_file = self.job_dir / f"loudness-{sha256_text(str(ref))[:8]}.json"
            if cache_file.exists():
                self.source_loudness = Loudness(**json.loads(cache_file.read_text()))
            else:
                self.source_loudness = measure_loudness(ref)
                atomic_write_text(cache_file, json.dumps(asdict(self.source_loudness)))
            log.info(f"Loudness reference {ref.name}: {self.source_loudness.integrated:.1f} LUFS, "
                     f"TP {self.source_loudness.true_peak:.1f} dBTP, "
                     f"LRA {self.source_loudness.lra:.1f} LU")

    def _transcribe(self) -> None:
        from ytdub.stages.transcribe import Transcript, transcribe

        inputs = {
            "source_sha256": self.src.sha256,
            "separate": self.s.demucs_model if self.s.separate else None,
            "asr_model": self.s.asr_model, "asr_model_files": model_fingerprint(self.s.asr_model),
            "faster_whisper": package_version("faster-whisper"),
            "compute_type": self.s.compute_type(),
            "beam_size": self.s.asr_beam_size, "vad": self.s.vad,
            "vad_min_silence_ms": self.s.vad_min_silence_ms,
            "min_confidence": self.s.min_confidence, "source_lang": self.s.source_lang,
        }
        self.transcript_key = StageCache.key(inputs)
        cached = self.cache.load("transcribe", inputs)
        if cached is not None:
            self.transcript = Transcript.from_dict(cached)
        else:
            self.transcript = transcribe(
                self.asr_wav, model=self.s.asr_model, device=self.s.device,
                compute_type=self.s.compute_type(), language=self.s.source_lang,
                beam_size=self.s.asr_beam_size, vad=self.s.vad,
                vad_min_silence_ms=self.s.vad_min_silence_ms,
                min_confidence=self.s.min_confidence,
            )
            self.cache.save("transcribe", inputs, self.transcript.to_dict())
        if not self.transcript.segments:
            raise RuntimeError("Transcription produced no speech segments")

    def _diarize_method(self) -> str:
        """The method actually used: pyannote falls back when it cannot run.

        Falling back is better than failing, but it is not silent — the fallback model is
        measurably worse at telling two similar voices apart, and the run continues with
        labels that may be wrong.
        """
        if self.s.diarize_method != "pyannote":
            return self.s.diarize_method
        import os

        if self.s.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"):
            return "pyannote"
        log.warning("pyannote diarization needs a Hugging Face token with the terms "
                    "accepted at hf.co/pyannote/speaker-diarization-3.1; set "
                    "YTDUB_HF_TOKEN in .env. Falling back to the token-free embedding "
                    "diarizer, which separates similar voices poorly.")
        return "embedding"

    def _diarize(self) -> None:
        segs = self.transcript.segments
        # Skipped entirely for one speaker rather than run with num_speakers=1: the answer
        # is known, the diarizer's would only be the same answer after a minute of work,
        # and a count of one that comes back as two is a bug this never has to guard.
        method = self._diarize_method() if self.s.multi_voice else "none"
        inputs = {"transcript": self.transcript_key, "speakers": self.s.speakers,
                  "method": method, "speaker_map": self.s.speaker_map.strip()}
        self.diarize_key = StageCache.key(inputs)
        if not self.s.multi_voice:
            self.segments = [s for s in segs]
            return
        cached = self.cache.load("diarize", inputs)
        if cached is not None:
            self.segments = [Segment.from_dict(d) for d in cached]
            return
        from ytdub.stages import diarize

        if method == "pyannote":
            self.segments = diarize.diarize_pyannote(self.asr_wav, segs,
                                                     num_speakers=self.s.speakers,
                                                     device=self.s.device,
                                                     hf_token=self.s.hf_token)
        else:
            self.segments = diarize.diarize_embedding(self.asr_wav, segs,
                                                      num_speakers=self.s.speakers,
                                                      device=self.s.device)
        if self.s.speaker_map.strip():
            # Applied *before* the result is cached and before the key is used by anything
            # downstream: an out-of-band override that the key does not see would let a
            # later run load references built from the labels it corrects.
            self.segments = diarize.apply_speaker_map(self.segments, self.s.speaker_map)
        self.cache.save("diarize", inputs, [s.to_dict() for s in self.segments])

    def _references(self) -> None:
        from ytdub.audio import read_mono
        from ytdub.stages.references import apply_matches, build_references

        user = {k: sha256_file(v) for k, v in self.user_refs.items()}
        voice_files = {p.name: sha256_file(p) for p in voice_clip_paths(self)}
        inputs = {"diarize": self.diarize_key, "user_refs": user,
                  "target_seconds": self.s.ref_target_seconds,
                  "min_seconds": self.s.ref_min_seconds, "audio": self.ref_wav.name}
        cached = self.cache.load("references", inputs)
        if cached is not None and all(Path(d["path"]).exists() for d in cached):
            self.refs = {d["speaker"]: SpeakerRef.from_dict(d) for d in cached}
            for ref in self.refs.values():
                log.info(f"{ref.speaker or 'voice'}: reference {ref.origin}, {ref.duration:.1f}s")
        else:
            out = self.job_dir / "refs" / StageCache.key(inputs)
            self.refs = build_references(self.segments, self.ref_wav, out,
                                         user_refs=self.user_refs,
                                         target_seconds=self.s.ref_target_seconds,
                                         min_seconds=self.s.ref_min_seconds)
            self.cache.save("references", inputs, [r.to_dict() for r in self.refs.values()])

        # A stored clip of your own voice is identified by *voice*, not by label: the
        # diarizer's SPK numbers are assigned per file, so naming one is a trap (see
        # stages/voices.py). Matching needs the references above, so it runs after them,
        # and only when --voice / YTDUB_VOICE asks for it.
        self.speaker_fold: dict[str, str] = {}
        voice_path = self.s.voice_path
        if voice_path is not None:
            if not self.s.multi_voice:
                # One speaker: there is nothing to match against, and nothing to be wrong
                # about — the clip IS the voice for the whole file, and the automatic
                # reference cut is thrown away. Loading a voice encoder to decide what is
                # already known would only add a way to get it wrong.
                from ytdub.stages.voices import voice_clips

                clips = voice_clips(voice_path)
                if clips:
                    samples, sr = read_mono(clips[0])
                    self.refs[None] = SpeakerRef(None, clips[0], len(samples) / sr,
                                                 "your clip (single speaker)")
                    log.success(f"voice: using {clips[0].name} for the whole file "
                                f"({self.refs[None].duration:.1f}s)")
                else:
                    log.warning(f"--voice {voice_path}: no audio file found; using the "
                                "automatic reference")
            else:
                from ytdub.stages.voices import match_voice

                match_inputs = {"refs": {spk: sha256_file(r.path) for spk, r in self.refs.items()},
                                "voice_files": voice_files}
                matched = self.cache.load("voice_match", match_inputs)
                if matched is None:
                    assigned, merges = match_voice(voice_path, self.refs, device="cpu",
                                                   threshold=self.s.voice_match_threshold)
                    matched = {"assigned": {k: str(v) for k, v in assigned.items()},
                               "merges": [list(m) for m in merges]}
                    self.cache.save("voice_match", match_inputs, matched)
                self.speaker_fold = apply_matches(
                    self.refs, self.segments,
                    matched={k: Path(v) for k, v in matched["assigned"].items()},
                    merges=[tuple(m) for m in matched["merges"]], user_refs=self.user_refs)
        self.ref_hashes = {spk: sha256_file(r.path) for spk, r in self.refs.items()}

    def _write_source_files(self) -> None:
        from ytdub.subtitles import Cue, write_srt

        write_srt([Cue(s.start, s.end, (f"[{s.speaker}] " if s.speaker else "") + s.text)
                   for s in self.segments], self.out_dir / "source.srt")
        dropped = self.transcript.dropped
        lines = [f"{d.start:8.2f} {d.end:8.2f}  conf={d.confidence}  {d.text}" for d in dropped]
        atomic_write_text(self.out_dir / "dropped.txt",
                          "# Segments dropped as low-confidence / non-speech. Review these:\n"
                          "# if real speech was dropped, lower --min-confidence.\n"
                          + "\n".join(lines) + "\n")

    # ------------------------------------------------------------------ translation
    def _budgets(self) -> list[int]:
        from ytdub.stages.fit import budget_slots
        from ytdub.stages.translate.prompt import budget_chars

        slots = budget_slots([s.start for s in self.segments], [s.end for s in self.segments],
                             self.src.duration, min_gap=self.s.min_gap,
                             max_borrow=self.s.budget_max_borrow)
        return [budget_chars(sl, self.s.chars_per_second) for sl in slots]

    def _style_and_glossary(self) -> tuple[str, list[str], str, str]:
        from ytdub.stages.translate.prompt import load_glossary, load_style

        style_file = self.s.styles_dir / f"{self.s.style}.txt"
        style_raw = style_file.read_text(encoding="utf-8")
        gloss_raw = (self.s.glossary_path.read_text(encoding="utf-8")
                     if self.s.glossary_path.exists() else "")
        return load_style(style_raw), load_glossary(gloss_raw), style_raw, gloss_raw

    def _hints(self, lang: str) -> tuple[list[Hint], str]:
        """``hints.txt`` plus ``hints.<lang>.txt`` (which wins), and the raw text of
        both for the cache key. A missing or unreadable file is not an error: hints are
        an optional convenience, and neither language has to have any."""
        from ytdub.stages.translate.prompt import load_hints, merge_hints

        base_path, lang_path = self.s.hints_path, self.s.hints_path_for(lang)
        base_raw = base_path.read_text(encoding="utf-8") if base_path.exists() else ""
        lang_raw = lang_path.read_text(encoding="utf-8") if lang_path.exists() else ""
        hints = merge_hints(load_hints(base_raw), load_hints(lang_raw))
        if hints:
            log.info(f"[{lang}] {len(hints)} translation hint(s) from "
                     f"{' + '.join(p.name for p in (base_path, lang_path) if p.exists())}")
        return hints, base_raw + "\n# --- " + lang_path.name + " ---\n" + lang_raw

    def _protected_terms(self, lang: str) -> list[str]:
        """Terms a digit-to-words pass must not touch: the glossary, and the hints on
        both sides of the pair.

        A term that stays as written ("Rec Room Tokens") and a term pinned to a rendering
        ("PlayStation 5 = PlayStation 5") are names or fixed phrases either way, so a
        number inside one is part of the term and not a spoken quantity. Protected spans
        are checked against the *translated* line, so the target-language side of a hint
        has to be in the list too, not just the source term.
        """
        _, glossary, _, _ = self._style_and_glossary()
        terms = list(glossary)
        for path in (self.s.hints_path, self.s.hints_path_for(lang)):
            if path.exists():
                from ytdub.stages.translate.prompt import load_hints

                for hint in load_hints(path.read_text(encoding="utf-8")):
                    terms += [hint.term, hint.translation]
        return sorted({t.strip() for t in terms if t.strip()})

    def translation_inputs(self, lang: str, translator, style_raw: str, gloss_raw: str,
                           budgets: list[int], hints_raw: str = "") -> dict:
        """Everything that affects a language's translation. Any change is a cache miss.

        Model weights are checked separately (see ``_translate_all``) because reading
        them needs the server, and a cached result must stay usable when it is down.
        """
        return {
            "transcript": self.transcript_key,  # source text + timings
            "diarize": self.diarize_key,  # speaker tags appear in the prompt
            # Labels folded together by voice matching change the tags in the prompt,
            # and that happens after the diarize key was computed.
            "speaker_fold": self.speaker_fold,
            "source_lang": self.transcript.language, "target_lang": lang,
            "style": sha256_text(style_raw), "glossary": sha256_text(gloss_raw),
            # Verification on/off and its prompts are in the backend identity; the
            # per-language hints are here because they are read per language.
            "hints": sha256_text(hints_raw),
            "translator": translator.name, "backend": translator.cache_identity(),
            "budgets": budgets,  # derived from the three settings below + timings
            "chars_per_second": self.s.chars_per_second,
            "budget_max_borrow": self.s.budget_max_borrow, "min_gap": self.s.min_gap,
        }

    def _translate_all(self) -> dict[str, list[str]]:
        from ytdub.stages.translate import Line, get_translator

        style, glossary, style_raw, gloss_raw = self._style_and_glossary()
        budgets = self._budgets()
        src_lang = self.transcript.language
        translator = get_translator(self.s, style_text=style, glossary=glossary)
        weights: list[str | None] = []  # fetched once, lazily
        out: dict[str, list[str]] = {}
        try:
            for lang in self.s.languages:
                res = self.results[lang]
                try:
                    if lang == src_lang:
                        out[lang] = [s.text for s in self.segments]
                        self._write_review_srt(lang, out[lang])
                        continue
                    hints, hints_raw = self._hints(lang)
                    # Optional on the Translator interface: a backend that cannot use
                    # hints is still valid, it just does not get them.
                    set_hints = getattr(translator, "set_hints", None)
                    if set_hints is not None:
                        set_hints(hints)
                    elif hints:
                        log.warning(f"[{lang}] {translator.name} cannot take hints; "
                                    f"{len(hints)} hint(s) ignored by this backend")
                    # Where a --verify run writes what it found. Set on the settings the
                    # translator already holds, so the Translator interface is unchanged.
                    self.s.verify_report_path = (
                        self.s.verify_report_path_for(lang, self.out_dir)
                        if self.s.verify else None)
                    if not weights:
                        weights.append(translator.weights_fingerprint())
                    inputs = self.translation_inputs(lang, translator, style_raw, gloss_raw,
                                                     budgets, hints_raw)
                    stage = f"translate-{lang}"
                    cached = self.cache.load(stage, inputs)
                    if cached is not None and weights[0] and cached.get("weights") \
                            and cached["weights"] != weights[0]:
                        log.warning(f"{stage}: model weights changed since the cached "
                                    "translation (re-pulled?); recomputing")
                        cached = None
                    if cached is not None:
                        if weights[0] is None:
                            log.info(f"{stage}: cannot read model weights digest; "
                                     "using cached translation")
                        out[lang] = cached["translations"]
                        res.translation = cached["stats"]
                        self._drop_stale_verify_report(lang)
                    else:
                        lines = [Line(n=i + 1, text=s.text, budget=b, speaker=s.speaker)
                                 for i, (s, b) in enumerate(zip(self.segments, budgets))]
                        started = time.monotonic()
                        texts, stats = translator.translate(lines, source_lang=src_lang,
                                                            target_lang=lang)
                        if len(texts) != len(lines):
                            raise RuntimeError(f"translator returned {len(texts)} lines "
                                               f"for {len(lines)}")
                        out[lang], res.translation = texts, stats
                        self.cache.save(stage, inputs, {"translations": texts, "stats": stats,
                                                        "weights": weights[0]})
                        log.success(f"[{lang}] translated {len(lines)} lines in "
                                    f"{time.monotonic() - started:.0f}s")
                    self._write_review_srt(lang, out[lang])
                except Exception as exc:
                    log.exception(f"[{lang}] translation failed")
                    res.status, res.error = "failed", f"translate: {exc}"
                    out.pop(lang, None)
        finally:
            translator.unload()
        return out

    def _review_path(self, lang: str) -> Path:
        return self.out_dir / f"{lang}.review.srt"

    def _drop_stale_verify_report(self, lang: str) -> None:
        """A cached translation did not run the pass, so an older report for it would be
        a description of output that is no longer on disk. Remove it rather than let it
        look current."""
        path = self.s.verify_report_path_for(lang, self.out_dir)
        if path.exists():
            path.unlink()
            log.info(f"[{lang}] removed stale {path.name} (this translation came from "
                     "the cache; --force reruns verification)")

    def _write_review_srt(self, lang: str, texts: list[str]) -> None:
        """Write the review SRT, but never overwrite one a reviewer has edited."""
        from ytdub.subtitles import Cue, render_srt

        path = self._review_path(lang)
        record_file = self.job_dir / "review_written.json"
        record = json.loads(record_file.read_text()) if record_file.exists() else {}
        content = render_srt([Cue(s.start, s.end, t) for s, t in zip(self.segments, texts)])
        if path.exists():
            current = sha256_text(path.read_text(encoding="utf-8"))
            if current == sha256_text(content):
                return
            if record.get(lang) != current:
                log.warning(f"[{lang}] {path.name} has been edited; leaving it alone. Use "
                            "--from-review to dub from it, or delete it to regenerate.")
                return
        atomic_write_text(path, content)
        record[lang] = sha256_text(content)
        atomic_write_text(record_file, json.dumps(record, indent=1))
        self.results[lang].outputs["review_srt"] = str(path)

    def _load_reviews(self) -> dict[str, list[tuple[str, list[int]]]]:
        """Reviewed text per language as ``[(text, line indices), ...]``.

        One entry per cue. A cue standing for several lines is a reviewer's merge — the
        fix for a fragment Whisper left with a fraction of a second — and its text stays
        together, spoken once for the whole merged window, rather than being cut up
        between lines that each have no time. See subtitles.group_review.
        """
        from ytdub.subtitles import group_review, parse_srt

        out: dict[str, list[tuple[str, list[int]]]] = {}
        starts = [s.start for s in self.segments]
        speakers = [s.speaker for s in self.segments]
        for lang in self.s.languages:
            path = self._review_path(lang)
            try:
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found (run with --srt-only first)")
                cues = parse_srt(path.read_text(encoding="utf-8"))
                groups = group_review(cues, starts, speakers, path.name)
                merged = sum(1 for _, lines in groups if len(lines) > 1)
                if merged:
                    log.info(f"[{lang}] {merged} merged cue(s) in {path.name}; the merged "
                             "text is spoken over the whole window")
                moved = sum(1 for cue, (_, lines) in zip(cues, groups)
                            if abs(cue.start - starts[lines[0]]) > 0.5)
                if moved:
                    log.warning(f"[{lang}] {moved} cue timings differ from the transcript; "
                                "timings are ignored (fitting decides them)")
                out[lang] = [(cues[i].text.strip(), lines) for i, lines in groups]
                log.info(f"[{lang}] using reviewed text from {path.name}")
            except Exception as exc:
                log.exception(f"[{lang}] cannot use review SRT")
                self.results[lang].status, self.results[lang].error = "failed", f"review: {exc}"
        return out

    # ------------------------------------------------------------------ synthesis
    def _dub_language(self, lang: str, texts: list[tuple[str, list[int]]], tts) -> None:
        from ytdub.audio import read_mono, trim_silence
        from ytdub.ffmpeg import mux_audio, probe, stretch_samples
        from ytdub.stages import fit
        from ytdub.stages.assemble import DurationError, finalize_audio
        from ytdub.stages.tts.base import merge_short_fragments, synthesize_all
        from ytdub.subtitles import Cue, write_srt

        res = self.results[lang]
        started = time.monotonic()
        # One group per cue. A merged cue becomes ONE segment carrying the whole text and
        # spanning every line it covers, so the fragment-merger below never gets a chance
        # to move that text onto a different speaker's line.
        segs = [replace(self.segments[lines[0]], end=self.segments[lines[-1]].end,
                        translated=text, sources=list(lines))
                for text, lines in texts]
        covered = {i for _, lines in texts for i in lines}
        if len(texts) == len(self.segments):
            merged, stuck = merge_short_fragments(segs, min_chars=self.s.min_tts_chars,
                                                  max_gap=self.s.merge_max_gap)
        else:
            # Merge already done by the reviewer; doing it again would fight the file.
            merged, stuck = segs, []
        res.merged_fragments = len(covered) - len(merged)
        if stuck:
            log.warning(f"[{lang}] {len(stuck)} short fragment(s) had no same-speaker neighbour "
                        f"to merge into (lines {stuck}); synthesizing them alone")

        clips, failed = synthesize_all(
            merged, tts, refs=self.refs, ref_hashes=self.ref_hashes, language=lang,
            clip_dir=self.job_dir / "clips" / lang, seed=self.s.tts_seed, reuse=not self.s.force,
            expand_numbers=self.s.expand_numbers,
            protected=self._protected_terms(lang),
        )

        arrays, items = {}, []
        for seg in merged:
            if seg.index not in clips:
                continue
            samples, sr = read_mono(clips[seg.index])
            samples = trim_silence(samples, sr)
            arrays[seg.index] = (samples, sr)
            items.append(fit.FitItem(seg.index, seg.start, seg.end, len(samples) / sr))
        if not items:
            raise RuntimeError("no clips were synthesized")

        params = fit.FitParams(max_ratio=self.s.max_ratio,
                               imperceptible_ratio=self.s.imperceptible_ratio,
                               hard_max_ratio=self.s.hard_max_ratio, min_gap=self.s.min_gap,
                               max_delay=self.s.max_delay)
        total = self.src.duration
        sr_out = self.s.sample_rate
        total_samples = int(round(total * sr_out))
        placements = fit.plan(items, total, params)
        rep = fit.report(placements, items, total, params)
        res.fit = rep.to_dict()
        for warning in rep.warnings(params):
            log.warning(f"[{lang}] {warning}")
            res.warnings.append(warning)
        timeline = fit.render(placements, arrays, total_samples=total_samples, out_sr=sr_out,
                              stretch=stretch_samples)

        wav = self.out_dir / f"{lang}.wav"
        res.loudness = finalize_audio(timeline, sr=sr_out, total_samples=total_samples,
                                      out_path=wav, work_dir=self.job_dir / "assemble" / lang,
                                      source_loudness=self.source_loudness)
        res.outputs["wav"] = str(wav)
        # Hard requirement, checked on every real run (not an assert: survives -O).
        # finalize_audio already verified the sample count it wrote; this re-measures
        # the finished file independently with ffprobe against the probed source.
        written = probe(wav).duration
        drift_ms = abs(written - total) * 1000
        if drift_ms > DURATION_TOLERANCE_MS:
            raise DurationError(f"{wav.name} is {written:.6f}s but the source is {total:.6f}s "
                                f"({drift_ms:.2f} ms off); output rejected")

        # SRT on the fitted timings, written before any muxing.
        timings = fit.placed_timings(placements, sr_out, total_samples)
        cues = [Cue(*timings[s.index], s.speech_text) for s in merged if s.index in timings]
        srt = write_srt(cues, self.out_dir / f"{lang}.srt")
        res.outputs["srt"] = str(srt)

        spoken_lines = {i for s in merged if s.index in clips for i in s.sources}
        res.lost_segments = sorted(set(range(len(self.segments))) - spoken_lines)
        res.status = "degraded" if (res.lost_segments or failed) else "ok"

        log.success(f"[{lang}] FIT: {rep.summary()}")
        log.success(f"[{lang}] wrote {wav.name} ({total_samples} samples = {total:.3f}s, "
                    f"exact) + {srt.name}")
        if res.lost_segments:
            log.error(f"[{lang}] {len(res.lost_segments)} source line(s) have NO audio: "
                      f"{res.lost_segments}")

        if self.src.has_video and self.s.mux_video:
            try:
                mp4 = mux_audio(self.src.path, wav, self.out_dir / f"{lang}.mp4")
                res.outputs["mp4"] = str(mp4)
            except Exception as exc:
                log.exception(f"[{lang}] preview mux failed (wav + srt are already written)")
                res.error = f"mux: {exc}"
        res.seconds = time.monotonic() - started

    # ------------------------------------------------------------------ driver
    def run(self) -> list[LanguageResult]:
        self.results = {lang: LanguageResult(lang) for lang in self.s.languages}
        self._acquire()
        self._prepare_audio()
        self._transcribe()
        self._diarize()
        self._references()
        self._write_source_files()

        if self.from_review:
            groups = self._load_reviews()
        else:
            # Translation produces one string per line: wrap it in the same
            # (text, line indices) shape a reviewed file yields, so there is one path.
            translated = self._translate_all()
            groups = {lang: [(text, [i]) for i, text in enumerate(texts)]
                      for lang, texts in translated.items()}
        if self.srt_only:
            for lang in groups:
                self.results[lang].status = "srt-only"
            return self._finish()

        from ytdub.stages.tts.base import get_tts

        tts = None
        try:
            for lang in self.s.languages:
                if lang not in groups:
                    continue
                try:
                    if tts is None:
                        tts = get_tts(self.s.tts_backend, self.s)
                    log.info(f"===== [{lang}] synthesis + fitting =====")
                    self._dub_language(lang, groups[lang], tts)
                except Exception as exc:
                    log.exception(f"[{lang}] dubbing failed")
                    self.results[lang].status = "failed"
                    self.results[lang].error = f"{type(exc).__name__}: {exc}"
        finally:
            if tts is not None:
                tts.unload()
        return self._finish()

    def _finish(self) -> list[LanguageResult]:
        results = list(self.results.values())
        report = {"source": str(self.src.path), "duration": self.src.duration,
                  "sha256": self.src.sha256, "segments": len(self.segments),
                  "dropped_segments": len(self.transcript.dropped),
                  "speakers": {str(k): v.to_dict() for k, v in self.refs.items()},
                  "languages": [asdict(r) for r in results]}
        atomic_write_text(self.out_dir / "report.json", json.dumps(report, indent=1, default=str))
        log.info("=" * 78)
        log.info(f"SUMMARY  {self.src.path.name}  ({self.src.duration:.2f}s, "
                 f"{len(self.segments)} lines, {len(self.transcript.dropped)} dropped)")
        for r in results:
            if r.fit:
                f = r.fit
                detail = (f"compressed {f['compressed']}/{f['segments']} worst {f['worst_ratio']:.3f}x "
                          f"| max delay {f['max_delay']:.2f}s | lost {len(r.lost_segments)}")
            else:
                detail = r.error or ""
            (log.info if r.status in ("ok", "srt-only") else log.error)(
                f"  {r.lang:>3}  {r.status:<9} {detail}")
            for warning in r.warnings:
                log.warning(f"  {r.lang:>3}  WARNING: {warning}")
        log.info(f"Outputs: {self.out_dir}")
        log.info("=" * 78)
        return results


def run_job(settings: Settings, input_arg: str, **kwargs) -> list[LanguageResult]:
    from ytdub.net import configure_network

    configure_network(force_ipv4=settings.force_ipv4, timeout=settings.net_timeout)
    return Job(settings, input_arg, **kwargs).run()
