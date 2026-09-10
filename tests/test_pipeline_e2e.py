"""Whole-job tests with every model replaced by a fake.

Whisper, the Ollama model, the diarizer and Chatterbox are swapped for deterministic
stand-ins; everything else (ffmpeg audio extraction, reference selection, fragment
merging, caching, fitting, loudness matching, SRT and mux) is the real code. Proves
wiring, exact duration, no lost content, cache/resume and the review loop.

Uses a short two-speaker clip that includes a very short utterance, the case that
crashed the reference implementation. Requires ffmpeg on PATH (skipped otherwise).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from ytdub.config import Settings
from ytdub.models import Segment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

DURATION = 20.0
CALLS: Counter = Counter()

# (start, end, speaker, text)
SCRIPT = [
    (0.40, 2.60, "SPK0", "So unless you've been living under a rock,"),
    (2.70, 4.10, "SPK0", "you'll know what happened."),
    (4.15, 4.35, "SPK0", "So."),                        # -> "to": short, merges into line 1
    (5.20, 7.60, "SPK1", "Honestly I did not see that coming at all."),
    (7.70, 7.90, "SPK1", "Oh"),                         # -> "O": short, merges into line 3
    (8.60, 8.80, "SPK0", "Hm"),                         # -> "H": no same-speaker neighbour
    (9.40, 12.80, "SPK1", "It was a mad past, I worked there five years in Rec Room."),
    (13.50, 15.00, "SPK0", "I am now unemployed."),
    (15.10, 18.20, "SPK0", "Anyway, let's get into the video, shall we?"),
]
SHORT = {"So.": "to", "Oh": "O", "Hm": "H"}


def translate_text(src: str, lang: str) -> str:
    return SHORT.get(src, f"{src} ({lang} und so weiter)")


class FakeOllama:
    digest = "sha256:aaaa"

    def __init__(self, url, model, *, num_ctx, **kw):
        self.num_ctx = num_ctx

    def model_digest(self):
        return FakeOllama.digest

    def check_model(self):
        pass

    def verify_context_window(self):
        pass

    def unload(self):
        CALLS["unload"] += 1

    def chat(self, system, user, *, num_predict, schema=None, label=""):
        from ytdub.stages.translate.ollama import ChatResult

        CALLS["translate"] += 1
        lang = "pl" if "Polish" in system else "de"
        rows = re.findall(r"^(\d+)\. \[≤\d+\] (?:\[SPK\d\] )?(?:source: )?(.*)$", user, re.M)
        out = [{"n": int(n), "text": translate_text(t.strip(), lang)} for n, t in rows]
        return ChatResult(json.dumps({"lines": out}), 1000, 100, "stop")


class FakeTTS:
    """Loaded by the pipeline through its ``module:Class`` spec, exactly as a real
    drop-in backend would be (see ``settings()`` below)."""

    name = "fake-tts"
    supported_languages = {"de", "pl"}
    fail_on: set[str] = set()
    chars_per_sec = 14.0

    def __init__(self, settings):
        self.settings = settings

    def cache_identity(self):
        return {"model": "fake", "version": 1}

    def synthesize(self, text, ref, language, out_path, seed):
        from ytdub.audio import write_wav
        from ytdub.stages.tts.base import spoken_chars

        CALLS["tts"] += 1
        if text in self.fail_on:
            raise RuntimeError("simulated TTS crash")
        assert ref.exists()
        sr = 24000
        speech = max(0.25, spoken_chars(text) / self.chars_per_sec)
        t = np.arange(int(speech * sr)) / sr
        tone = 0.3 * np.sin(2 * np.pi * 180 * t)
        pad = np.zeros(int(0.2 * sr))
        write_wav(out_path, np.concatenate([pad, tone, pad]).astype(np.float32), sr)
        return out_path

    def unload(self):
        pass


def fake_transcribe(audio_path, **kw):
    from ytdub.stages.transcribe import Transcript

    CALLS["transcribe"] += 1
    segs = [Segment(i, s, e, text, confidence=0.9) for i, (s, e, _, text) in enumerate(SCRIPT)]
    dropped = [Segment(0, 18.5, 19.0, "[music]", confidence=0.1)]
    return Transcript(segments=segs, language="en", dropped=dropped)


def fake_diarize(audio_path, segments, **kw):
    from dataclasses import replace

    CALLS["diarize"] += 1
    return [replace(s, speaker=SCRIPT[s.index][2]) for s in segments]


def _patch_models(monkeypatch):
    """Whisper and the diarizer are replaced in place; the real OllamaTranslator and
    BatchTranslator run against a fake HTTP client."""
    from ytdub.stages import diarize, transcribe
    from ytdub.stages.translate import ollama

    monkeypatch.setattr(transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(diarize, "diarize_embedding", fake_diarize)
    monkeypatch.setattr(ollama, "OllamaClient", FakeOllama)


@pytest.fixture
def home(tmp_path, monkeypatch):
    CALLS.clear()
    FakeTTS.fail_on = set()
    FakeTTS.chars_per_sec = 14.0
    FakeOllama.digest = "sha256:aaaa"
    _patch_models(monkeypatch)

    (tmp_path / "synopses").mkdir()
    (tmp_path / "synopses" / "casual.txt").write_text("A casual video.", encoding="utf-8")
    (tmp_path / "glossary.txt").write_text("Rec Room\n", encoding="utf-8")
    (tmp_path / "input").mkdir()
    sr = 44100
    audio = np.zeros(int(DURATION * sr), dtype=np.float32)
    for s, e, spk, _ in SCRIPT:
        t = np.arange(int((e - s) * sr)) / sr
        audio[int(s * sr):int(s * sr) + len(t)] = 0.2 * np.sin(2 * np.pi * (140 if spk == "SPK0" else 220) * t)
    sf.write(tmp_path / "input" / "talk.wav", audio, sr)
    return tmp_path


def settings(home, **kw) -> Settings:
    base = dict(home=home, languages=["de", "pl"], speakers=2, device="cpu", style="casual",
                tts_backend=f"{__name__}:FakeTTS")
    return Settings(**{**base, **kw})


def run(home, input_name="talk.wav", **kw):
    from ytdub.pipeline import Job

    job_kw = {k: kw.pop(k) for k in ("srt_only", "from_review") if k in kw}
    job = Job(settings(home, **kw), input_name, **job_kw)
    return {r.lang: r for r in job.run()}, job


def srt_cues(path):
    from ytdub.subtitles import parse_srt

    return parse_srt(path.read_text(encoding="utf-8"))


def test_audio_only_job(home):
    results, job = run(home)
    out = home / "output" / "talk"
    for lang in ("de", "pl"):
        r = results[lang]
        assert r.status == "ok", r
        info = sf.info(out / f"{lang}.wav")
        # Hard requirement: exact duration.
        assert info.samplerate == 48000
        assert info.frames == round(DURATION * 48000)
        assert abs(info.frames / info.samplerate - DURATION) < 0.001
        # No content lost; short fragments merged where a same-speaker neighbour exists.
        assert r.lost_segments == []
        assert r.merged_fragments == 2
        # Audio in -> audio + SRT out, no mux attempted.
        assert not (out / f"{lang}.mp4").exists()
        cues = srt_cues(out / f"{lang}.srt")
        assert len(cues) == len(SCRIPT) - 2
        for a, b in zip(cues, cues[1:]):
            assert a.end <= b.start + 0.001
        assert all(c.end <= DURATION + 0.001 for c in cues)
        assert (out / f"{lang}.review.srt").exists()
        assert r.fit["segments"] == len(SCRIPT) - 2
        assert r.fit["worst_ratio"] <= 1.2
        # Loudness matched to the source, not a fixed target.
        src_lufs = r.loudness["source"]["integrated_lufs"]
        assert abs(r.loudness["dub_after"]["integrated_lufs"] - src_lufs) < 1.0
    assert CALLS["transcribe"] == 1 and CALLS["diarize"] == 1  # once for all languages
    assert CALLS["unload"] == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["dropped_segments"] == 1 and "[music]" in (out / "dropped.txt").read_text(encoding="utf-8")
    assert set(report["speakers"]) == {"SPK0", "SPK1"}


def test_rerun_resumes_everything_from_cache(home):
    run(home)
    before = CALLS.copy()
    results, _ = run(home)
    assert all(r.status == "ok" for r in results.values())
    assert CALLS["transcribe"] == before["transcribe"]
    assert CALLS["diarize"] == before["diarize"]
    assert CALLS["translate"] == before["translate"]
    assert CALLS["tts"] == before["tts"]


def test_glossary_change_invalidates_translation_not_transcription(home):
    run(home)
    before = CALLS.copy()
    (home / "glossary.txt").write_text("Rec Room\nAtlas\n", encoding="utf-8")
    run(home)
    assert CALLS["transcribe"] == before["transcribe"]
    assert CALLS["translate"] > before["translate"]
    # Same resulting text -> the per-clip cache still serves every clip.
    assert CALLS["tts"] == before["tts"]


def _edit_style(home):
    (home / "synopses" / "casual.txt").write_text("A casual video. More jokes.", encoding="utf-8")


def _repull_model(home):
    FakeOllama.digest = "sha256:bbbb"


# Every input that changes translation output must invalidate translation, and none
# of them may invalidate transcription.
@pytest.mark.parametrize("change", [
    {"_edit": _edit_style},
    {"_edit": lambda home: (home / "glossary.txt").write_text("Atlas\n", encoding="utf-8")},
    {"_edit": _repull_model},
    {"ollama_model": "qwen3:14b"},
    {"ollama_num_ctx": 16384},
    {"ollama_temperature": 0.5},
    {"chars_per_second": 11.0},
    {"budget_max_borrow": 0.5},
    {"translate_batch_lines": 5},
    {"translate_context_lines": 1},
    {"budget_tolerance": 1.3},
    {"budget_retries": 0},
])
def test_translation_inputs_invalidate_translation_only(home, change):
    run(home, languages=["de"], srt_only=True)
    before = CALLS.copy()
    change = dict(change)
    if "_edit" in change:
        change.pop("_edit")(home)
    run(home, languages=["de"], srt_only=True, **change)
    assert CALLS["translate"] > before["translate"], f"{change} did not retranslate"
    assert CALLS["transcribe"] == before["transcribe"]
    assert CALLS["diarize"] == before["diarize"]


def test_unrelated_settings_keep_the_cached_translation(home):
    run(home, languages=["de"], srt_only=True)
    before = CALLS.copy()
    run(home, languages=["de"], srt_only=True, max_ratio=1.1, ollama_url="http://127.0.0.2:1")
    assert CALLS["translate"] == before["translate"]


def test_cached_translation_survives_ollama_being_down(home, monkeypatch):
    run(home, languages=["de"], srt_only=True)
    before = CALLS.copy()
    from ytdub.stages.translate.ollama import OllamaError

    def unreachable(self):
        raise OllamaError("connection refused")

    monkeypatch.setattr(FakeOllama, "model_digest", unreachable)
    results, _ = run(home, languages=["de"], srt_only=True)
    assert results["de"].status == "srt-only"
    assert CALLS["translate"] == before["translate"]


class EchoTranslator:
    """A complete drop-in translator, selected by module:Class spec."""

    name = "echo"

    def __init__(self, settings, style_text, glossary):
        pass

    def cache_identity(self):
        return {}

    def weights_fingerprint(self):
        return None

    def translate(self, lines, *, source_lang, target_lang):
        CALLS["echo"] += 1
        return [f"[{target_lang}] {ln.text}" for ln in lines], {}

    def unload(self):
        pass


def test_drop_in_translator_backend(home):
    results, _ = run(home, languages=["de"], translator=f"{__name__}:EchoTranslator")
    assert results["de"].status == "ok" and CALLS["echo"] == 1 and CALLS["translate"] == 0
    assert "[de] I am now unemployed." in (home / "output" / "talk" / "de.srt").read_text("utf-8")


def test_wrong_length_output_fails_the_language(home, monkeypatch):
    import ytdub.stages.assemble as assemble

    real = assemble.fit_length
    monkeypatch.setattr(assemble, "fit_length", lambda s, n: real(s, n - 480))  # 10 ms short
    results, _ = run(home, languages=["de"])
    assert results["de"].status == "failed"
    assert "DurationError" in results["de"].error


def test_over_cap_is_a_visible_warning(home):
    from loguru import logger

    FakeTTS.chars_per_sec = 8.0  # ~2.4x more audio than time: the 1.2x cap cannot hold

    records = []
    sink = logger.add(lambda m: records.append(m.record), level="INFO")  # normal CLI level
    try:
        results, _ = run(home, languages=["de"])
    finally:
        logger.remove(sink)
    r = results["de"]
    assert r.fit["over_cap"] > 0 and r.warnings
    warned = [x["message"] for x in records if x["level"].name == "WARNING"]
    assert any("beyond the 1.2x cap" in m for m in warned)  # at the stage...
    assert any("WARNING:" in m and "cap" in m for m in warned)  # ...and in the summary


def test_force_recomputes(home):
    run(home, languages=["de"])
    before = CALLS.copy()
    run(home, languages=["de"], force=True)
    assert CALLS["transcribe"] == before["transcribe"] + 1
    assert CALLS["tts"] > before["tts"]


def test_srt_only_then_dub_from_edited_review(home):
    results, _ = run(home, srt_only=True)
    out = home / "output" / "talk"
    assert all(r.status == "srt-only" for r in results.values())
    assert CALLS["tts"] == 0 and not (out / "de.wav").exists()

    review = out / "de.review.srt"
    edited = review.read_text(encoding="utf-8").replace(
        "I am now unemployed. (de und so weiter)", "Jetzt bin ich arbeitslos.")
    review.write_text(edited, encoding="utf-8")

    run(home, srt_only=True)  # a normal re-run must not clobber the reviewer's edit
    assert review.read_text(encoding="utf-8") == edited

    before = CALLS.copy()
    results, _ = run(home, languages=["de"], from_review=True)
    assert results["de"].status == "ok"
    assert CALLS["translate"] == before["translate"]
    assert "Jetzt bin ich arbeitslos." in (out / "de.srt").read_text(encoding="utf-8")


def test_one_tts_failure_is_reported_not_fatal(home):
    FakeTTS.fail_on = {translate_text(SCRIPT[7][3], "de")}
    results, _ = run(home, languages=["de"])
    r = results["de"]
    assert r.status == "degraded" and r.lost_segments == [7]
    info = sf.info(home / "output" / "talk" / "de.wav")
    assert info.frames == round(DURATION * 48000)  # still exact


def _make_video(home):
    src = home / "input" / "talk.wav"
    dst = home / "input" / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c=black:s=160x90:d={DURATION}", "-i", str(src),
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "aac",
                    "-t", str(DURATION), str(dst)], check=True)
    return dst


def test_video_input_writes_wav_and_srt_before_mux(home, monkeypatch):
    import ytdub.ffmpeg as ff
    from ytdub.ffmpeg import probe

    _make_video(home)
    total = probe(home / "input" / "clip.mp4").duration

    def broken_mux(*a, **k):
        raise RuntimeError("simulated mux failure")

    monkeypatch.setattr(ff, "mux_audio", broken_mux)
    results, _ = run(home, "clip.mp4", languages=["de"])
    out = home / "output" / "clip"
    assert (out / "de.wav").exists() and (out / "de.srt").exists()
    assert results["de"].error.startswith("mux:")
    assert sf.info(out / "de.wav").frames == round(total * 48000)

    monkeypatch.undo()
    _patch_models(monkeypatch)
    results, _ = run(home, "clip.mp4", languages=["de"])
    assert results["de"].status == "ok" and (out / "de.mp4").exists()
