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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ytdub.cache import StageCache, atomic_write_text, sha256_file, sha256_text
from ytdub.config import Settings
from ytdub.logging import add_file_sink, stage_logger
from ytdub.models import Segment, Source, SpeakerRef

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

    def _diarize(self) -> None:
        segs = self.transcript.segments
        inputs = {"transcript": self.transcript_key, "speakers": self.s.speakers,
                  "method": self.s.diarize_method}
        self.diarize_key = StageCache.key(inputs)
        if self.s.speakers is None:
            self.segments = [s for s in segs]
            return
        cached = self.cache.load("diarize", inputs)
        if cached is not None:
            self.segments = [Segment.from_dict(d) for d in cached]
            return
        from ytdub.stages import diarize

        if self.s.diarize_method == "pyannote":
            self.segments = diarize.diarize_pyannote(self.asr_wav, segs, num_speakers=self.s.speakers,
                                                     device=self.s.device, hf_token=self.s.hf_token)
        else:
            self.segments = diarize.diarize_embedding(self.asr_wav, segs,
                                                      num_speakers=self.s.speakers,
                                                      device=self.s.device)
        self.cache.save("diarize", inputs, [s.to_dict() for s in self.segments])

    def _references(self) -> None:
        from ytdub.stages.references import build_references

        user = {k: sha256_file(v) for k, v in self.user_refs.items()}
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

    def translation_inputs(self, lang: str, translator, style_raw: str, gloss_raw: str,
                           budgets: list[int]) -> dict:
        """Everything that affects a language's translation. Any change is a cache miss.

        Model weights are checked separately (see ``_translate_all``) because reading
        them needs the server, and a cached result must stay usable when it is down.
        """
        return {
            "transcript": self.transcript_key,  # source text + timings
            "diarize": self.diarize_key,  # speaker tags appear in the prompt
            "source_lang": self.transcript.language, "target_lang": lang,
            "style": sha256_text(style_raw), "glossary": sha256_text(gloss_raw),
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
                    if not weights:
                        weights.append(translator.weights_fingerprint())
                    inputs = self.translation_inputs(lang, translator, style_raw, gloss_raw,
                                                     budgets)
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

    def _load_reviews(self) -> dict[str, list[str]]:
        from ytdub.subtitles import parse_srt

        out = {}
        for lang in self.s.languages:
            path = self._review_path(lang)
            try:
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found (run with --srt-only first)")
                cues = parse_srt(path.read_text(encoding="utf-8"))
                if len(cues) != len(self.segments):
                    raise ValueError(f"{path.name} has {len(cues)} cues but the transcript has "
                                     f"{len(self.segments)} lines; edit text only, do not "
                                     "add, remove or merge cues")
                empty = [i + 1 for i, c in enumerate(cues) if not c.text.strip()]
                if empty:
                    raise ValueError(f"{path.name}: cues {empty} are empty")
                moved = sum(1 for c, s in zip(cues, self.segments) if abs(c.start - s.start) > 0.5)
                if moved:
                    log.warning(f"[{lang}] {moved} cue timings differ from the transcript; "
                                "timings are ignored (fitting decides them)")
                out[lang] = [c.text for c in cues]
                log.info(f"[{lang}] using reviewed text from {path.name}")
            except Exception as exc:
                log.exception(f"[{lang}] cannot use review SRT")
                self.results[lang].status, self.results[lang].error = "failed", f"review: {exc}"
        return out

    # ------------------------------------------------------------------ synthesis
    def _dub_language(self, lang: str, texts: list[str], tts) -> None:
        from ytdub.audio import read_mono, trim_silence
        from ytdub.ffmpeg import mux_audio, probe, stretch_samples
        from ytdub.stages import fit
        from ytdub.stages.assemble import DurationError, finalize_audio
        from ytdub.stages.tts.base import merge_short_fragments, synthesize_all
        from ytdub.subtitles import Cue, write_srt

        res = self.results[lang]
        started = time.monotonic()
        segs = [s.with_translation(t) for s, t in zip(self.segments, texts)]
        merged, stuck = merge_short_fragments(segs, min_chars=self.s.min_tts_chars,
                                              max_gap=self.s.merge_max_gap)
        res.merged_fragments = len(segs) - len(merged)
        if stuck:
            log.warning(f"[{lang}] {len(stuck)} short fragment(s) had no same-speaker neighbour "
                        f"to merge into (lines {stuck}); synthesizing them alone")

        clips, failed = synthesize_all(
            merged, tts, refs=self.refs, ref_hashes=self.ref_hashes, language=lang,
            clip_dir=self.job_dir / "clips" / lang, seed=self.s.tts_seed, reuse=not self.s.force,
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

        covered = {i for s in merged if s.index in clips for i in s.sources}
        res.lost_segments = sorted(set(range(len(self.segments))) - covered)
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

        texts = self._load_reviews() if self.from_review else self._translate_all()
        if self.srt_only:
            for lang in texts:
                self.results[lang].status = "srt-only"
            return self._finish()

        from ytdub.stages.tts.base import get_tts

        tts = None
        try:
            for lang in self.s.languages:
                if lang not in texts:
                    continue
                try:
                    if tts is None:
                        tts = get_tts(self.s.tts_backend, self.s)
                    log.info(f"===== [{lang}] synthesis + fitting =====")
                    self._dub_language(lang, texts[lang], tts)
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
