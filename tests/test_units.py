"""Unit tests for the smaller pure pieces: cache, SRT, fragment merging, reference
selection, transcription post-processing, diarization clustering, download options,
network hardening and the CLI."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

from ytdub.config import DEFAULT_LANGUAGES
from ytdub.models import Segment


def seg(i, start, end, text="hello there", speaker=None, conf=0.9, translated=None):
    return Segment(index=i, start=start, end=end, text=text, speaker=speaker,
                   confidence=conf, translated=translated)


# --- cache ---------------------------------------------------------------------


def test_cache_hits_misses_and_force(tmp_path):
    from ytdub.cache import StageCache

    cache = StageCache(tmp_path)
    inputs = {"source": "abc", "model": "large-v3"}
    assert cache.load("transcribe", inputs) is None
    cache.save("transcribe", inputs, {"x": 1})
    assert cache.load("transcribe", inputs) == {"x": 1}
    assert cache.load("transcribe", {**inputs, "model": "small"}) is None  # input changed
    assert StageCache(tmp_path, force=True).load("transcribe", inputs) is None


def test_translation_key_ignores_nothing_that_matters():
    from ytdub.cache import StageCache

    base = {"transcript": "t", "style": "s1", "glossary": "g1", "model": "qwen3:8b"}
    keys = {StageCache.key(base), StageCache.key({**base, "style": "s2"}),
            StageCache.key({**base, "glossary": "g2"}), StageCache.key({**base, "model": "x"})}
    assert len(keys) == 4


# --- subtitles -----------------------------------------------------------------


def test_srt_roundtrip():
    from ytdub.subtitles import Cue, parse_srt, render_srt, timestamp

    assert timestamp(3661.5) == "01:01:01,500"
    cues = [Cue(0.0, 1.5, "Cześć wszystkim"), Cue(2.0, 3.25, "Druga\nlinia")]
    parsed = parse_srt("﻿" + render_srt(cues).replace("\n", "\r\n"))
    assert [(c.start, c.end) for c in parsed] == [(0.0, 1.5), (2.0, 3.25)]
    assert parsed[1].text == "Druga linia"


# --- fragment merging (known bug 1) ------------------------------------------------


def test_short_fragment_merges_into_same_speaker_neighbour():
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 2, translated="No więc słuchajcie", speaker="SPK0"),
            seg(1, 2.1, 2.3, translated="to", speaker="SPK0"),
            seg(2, 3, 5, translated="Druga osoba mówi", speaker="SPK1")]
    merged, stuck = merge_short_fragments(segs, min_chars=3)
    assert [m.speech_text for m in merged] == ["No więc słuchajcie to", "Druga osoba mówi"]
    assert merged[0].sources == [0, 1] and merged[0].end == 2.3
    assert stuck == []


def test_fragment_never_merges_across_speakers():
    """Putting your word into someone else's mouth is worse than any timing fault."""
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 2, translated="Pierwsza osoba", speaker="SPK0"),
            seg(1, 2.1, 2.3, translated="No!", speaker="SPK1"),
            seg(2, 2.5, 5, translated="Znowu pierwsza", speaker="SPK0")]
    merged, unspoken = merge_short_fragments(segs, min_chars=3, max_gap=1.5)
    assert [m.speech_text for m in merged] == ["Pierwsza osoba", "No!", "Znowu pierwsza"]
    assert unspoken == [], "it is still spoken — alone, in its own voice"


def test_a_one_word_line_near_a_line_of_its_speaker_is_spoken_with_it():
    """The last resort, and the reason it exists: a one-word line handed to the
    synthesizer alone either crashes it (IndexError, or a device-side assert that poisons
    the CUDA context) or loops into a multi-second clip. Saying it early, in the right
    voice, is the least bad option left."""
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 2, translated="Pierwsza osoba", speaker="SPK0"),
            seg(1, 2.1, 2.3, translated="No!", speaker="SPK1"),
            seg(2, 2.5, 5, translated="Znowu pierwsza", speaker="SPK0"),
            seg(3, 7.0, 7.2, translated="ok", speaker="SPK0")]
    merged, unspoken = merge_short_fragments(segs, min_chars=3, max_gap=1.5, stuck_gap=3.0)
    assert unspoken == [], "the cross-speaker line is still refused a home, and spoken alone"
    assert [m.speech_text for m in merged] == ["Pierwsza osoba", "No!", "Znowu pierwsza ok"]
    # It is spoken in the target line's window, not in a span stretching across the gap:
    # a window from 2.5s to 6.2s would block everything said in between.
    # Spoken in the target line's window, not in a span stretching to the fragment's own:
    # 2.5s to 7.2s would block everything said in between.
    assert (merged[-1].start, merged[-1].end) == (2.5, 5.0)
    assert merged[-1].sources == [2, 3]


def test_a_lone_word_with_no_line_of_its_speaker_near_it_is_not_spoken():
    """Handing it to the model alone is what crashed, or looped, every time it happened.
    One word of audio is cheaper than any of those outcomes, and the line is reported."""
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 2, translated="Pierwsza osoba", speaker="SPK0"),
            seg(1, 2.1, 2.3, translated="No!", speaker="SPK1"),
            seg(2, 2.5, 5, translated="Znowu pierwsza", speaker="SPK0"),
            seg(3, 9.0, 9.2, translated="ok", speaker="SPK0")]
    merged, unspoken = merge_short_fragments(segs, min_chars=3, max_gap=1.5, stuck_gap=3.0)
    assert unspoken == [3], "the lone word is the line that goes unspoken"
    assert len(merged) == 3, "the cross-speaker word is still spoken, alone"
    assert not any(m.speech_text.endswith("ok") for m in merged)


def test_a_phrase_with_no_neighbour_is_still_spoken_alone():
    """Only a lone word is fatal. A short phrase is left to the synthesizer, as before."""
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 2, translated="Pierwsza osoba", speaker="SPK0"),
            seg(1, 9.0, 9.4, translated="do widzenia", speaker="SPK0")]
    merged, unspoken = merge_short_fragments(segs, min_chars=3, max_gap=1.5, stuck_gap=3.0)
    assert unspoken == [] and len(merged) == 2


def test_a_single_word_is_a_fragment_however_many_letters_it_has():
    """The ones that crashed the synthesizer were not the shortest — "Would", "Tanguy",
    "Jak?" are four to six characters and one word each."""
    from ytdub.stages.tts.base import is_fragment

    assert is_fragment(seg(0, 0, 1, translated="Would"), 3)
    assert is_fragment(seg(0, 0, 1, translated="Tanguy"), 3)
    assert is_fragment(seg(0, 0, 1, translated="Jak?"), 3)
    assert is_fragment(seg(0, 0, 1, translated="No!"), 3)
    assert is_fragment(seg(0, 0, 1, translated="to"), 3)
    assert not is_fragment(seg(0, 0, 1, translated="Dziękuję bardzo"), 3)
    assert not is_fragment(seg(0, 0, 1, translated="ok, ja"), 3)


def test_a_one_word_line_merges_into_its_neighbour_instead_of_crashing():
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 2, translated="Odbudowana lokomotywa", speaker="SPK0"),
            seg(1, 2.1, 2.6, translated="Jak?", speaker="SPK0"),
            seg(2, 2.8, 5, translated="To działa tak", speaker="SPK0")]
    merged, stuck = merge_short_fragments(segs, min_chars=3)
    assert [m.speech_text for m in merged] == ["Odbudowana lokomotywa Jak?", "To działa tak"]
    assert stuck == []


def test_a_one_word_line_still_never_crosses_speakers():
    """Merging it anywhere would be better than crashing, except onto another voice: that
    would put your word in their mouth."""
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 2, translated="Pierwsza osoba", speaker="SPK0"),
            seg(1, 2.1, 2.6, translated="Tanguy", speaker="SPK1"),
            seg(2, 2.8, 5, translated="Znowu pierwsza", speaker="SPK0")]
    merged, unspoken = merge_short_fragments(segs, min_chars=3)
    assert len(merged) == 3 and unspoken == []


def test_fragment_merges_forward_when_it_opens_a_turn():
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 0.2, translated="A", speaker="SPK1"),
            seg(1, 0.3, 2, translated="potem reszta", speaker="SPK1")]
    merged, _ = merge_short_fragments(segs)
    assert [m.speech_text for m in merged] == ["A potem reszta"]
    assert merged[0].start == 0 and merged[0].end == 2


def test_spoken_chars_counts_devanagari_marks():
    from ytdub.stages.tts.base import spoken_chars

    assert spoken_chars("to.") == 2
    assert spoken_chars("है") == 2  # consonant + vowel sign


# --- voice references ------------------------------------------------------------


def test_reference_prefers_long_clean_confident_speech():
    from ytdub.stages.references import select_region

    segs = [
        seg(0, 0.0, 1.0, speaker="SPK0", conf=0.95),                 # short
        seg(1, 1.1, 1.9, speaker="SPK1"),                          # crosstalk neighbour
        seg(2, 2.0, 5.0, speaker="SPK0", conf=0.9),                 # 0.1s after SPK1: crosstalk risk
        seg(3, 5.2, 8.0, speaker="SPK0", conf=0.9),
        seg(4, 8.1, 12.5, speaker="SPK0", conf=0.85),               # clean run 5.2-12.5
        seg(5, 14.0, 20.0, speaker="SPK0", conf=0.3),               # long but low confidence
    ]
    region = select_region(segs, "SPK0", target_seconds=10, min_seconds=4)
    assert region.indices == [3, 4]
    assert region.spans == [(5.2, 12.5)]


def test_reference_falls_back_to_stitching_short_segments():
    from ytdub.stages.references import select_region

    segs = [seg(i, i * 3.0, i * 3.0 + 1.0, speaker="SPK1") for i in range(6)]
    region = select_region(segs, "SPK1", target_seconds=4, min_seconds=3)
    assert len(region.spans) >= 4


def test_user_reference_wins(tmp_path):
    from ytdub.audio import write_wav
    from ytdub.stages.references import build_references

    user = write_wav(tmp_path / "atlas.wav", np.zeros(24000, dtype=np.float32), 24000)
    refs = build_references([seg(0, 0, 5)], tmp_path / "unused.wav", tmp_path,
                            user_refs={None: user}, target_seconds=10, min_seconds=4)
    assert refs[None].origin == "user" and refs[None].path == user


# --- transcription post-processing (known bug 3) -----------------------------------


def test_sentence_segments_carry_confidence_and_split_on_pause():
    from ytdub.stages.transcribe import build_sentence_segments

    words = [(0.0, 0.4, "Hello", 0.9), (0.4, 0.8, " there", 0.7),
             (3.0, 3.4, "Next", 0.5), (3.4, 3.8, " bit.", 0.5)]
    segs = build_sentence_segments(words, max_gap=0.6)
    assert [s.text for s in segs] == ["Hello there", "Next bit."]
    assert segs[0].confidence == pytest.approx(0.8)


def test_low_confidence_segments_are_dropped_and_reported():
    from ytdub.stages.transcribe import filter_low_confidence

    segs = [seg(0, 0, 1, conf=0.9), seg(1, 1, 2, conf=0.2), seg(2, 2, 3, text="...", conf=0.9),
            seg(3, 3, 4, conf=None)]
    kept, dropped = filter_low_confidence(segs, 0.45)
    assert [s.index for s in kept] == [0, 1]  # renumbered contiguously
    assert len(dropped) == 2


# --- diarization -------------------------------------------------------------------


def test_cluster_embeddings_two_speakers():
    from ytdub.stages.diarize import cluster_embeddings

    embs = [[1, 0, 0], [0.95, 0.05, 0], [0, 1, 0], [0.05, 0.95, 0]]
    labels = cluster_embeddings(embs, num_speakers=2)
    assert labels[0] == labels[1] != labels[2] == labels[3]
    assert len(set(cluster_embeddings(embs, num_speakers=0))) == 2


def test_short_segments_take_nearest_label():
    from ytdub.stages.diarize import fill_short_labels

    segs = [seg(0, 0, 2), seg(1, 2.1, 2.3), seg(2, 5, 7)]
    assert fill_short_labels(segs, {0: 0, 2: 1}) == [0, 0, 1]


# --- download / network (known bugs 4 and 5) ---------------------------------------


def test_single_ydl_options_dict_has_js_runtime_ipv4_and_timeout(tmp_path):
    from ytdub.stages.download import ydl_options

    opts = ydl_options(tmp_path, force_ipv4=True, timeout=30)
    assert opts["js_runtimes"] == {"node": {}}
    assert opts["remote_components"] == ["ejs:github"]
    assert opts["source_address"] == "0.0.0.0"
    assert opts["socket_timeout"] == 30
    assert "source_address" not in ydl_options(tmp_path, force_ipv4=False)


def test_ipv4_resolver_filters_to_inet(monkeypatch):
    from ytdub import net

    seen = {}

    def fake(host, port, family=0, *rest):
        seen["family"] = family
        return [(family, socket.SOCK_STREAM, 6, "", ("1.2.3.4", port))]

    monkeypatch.setattr(net, "_original_getaddrinfo", fake)
    net._ipv4_getaddrinfo("example.com", 443)
    assert seen["family"] == socket.AF_INET


# --- audio helpers -------------------------------------------------------------------


def test_trim_silence_keeps_speech_and_never_empties():
    from ytdub.audio import trim_silence

    sr = 24000
    tone = 0.3 * np.sin(np.arange(sr) * 0.05).astype(np.float32)
    clip = np.concatenate([np.zeros(sr // 4), tone, np.zeros(sr // 2)]).astype(np.float32)
    out = trim_silence(clip, sr)
    assert abs(len(out) / sr - 1.04) < 0.03
    silent = np.zeros(1000, dtype=np.float32)
    assert len(trim_silence(silent, sr)) == 1000


# --- plugins ------------------------------------------------------------------------


def test_backend_loader_builtin_spec_and_errors():
    from ytdub.plugins import load_class
    from ytdub.stages.tts.base import BUILTIN, tts_class

    assert tts_class("chatterbox").__name__ == "ChatterboxBackend"  # no torch import needed
    assert load_class("collections:OrderedDict", BUILTIN, "TTS").__name__ == "OrderedDict"
    with pytest.raises(ValueError):
        load_class("nope", BUILTIN, "TTS")
    with pytest.raises(ImportError, match="no_such_module"):
        load_class("no_such_module:X", BUILTIN, "TTS")


# --- CLI ----------------------------------------------------------------------------


def _home(tmp_path):
    (tmp_path / "synopses").mkdir()
    for name in ("casual", "duo"):
        (tmp_path / "synopses" / f"{name}.txt").write_text("x", encoding="utf-8")
    (tmp_path / "input").mkdir()
    return tmp_path


def test_cli_without_args_prints_usage_and_styles(tmp_path, monkeypatch, capsys):
    from ytdub import cli

    monkeypatch.setenv("YTDUB_HOME", str(_home(tmp_path)))
    assert cli.main([]) == 1
    out = capsys.readouterr().out
    assert "casual" in out and "duo" in out


def test_bare_cli_dubs_the_newest_file_in_input(tmp_path, monkeypatch, capsys):
    """`dub2` with nothing after it means "the file I just dropped in" — the newest one,
    announced before anything expensive starts."""
    import os
    import time

    from ytdub import cli, pipeline

    home = _home(tmp_path)
    monkeypatch.setenv("YTDUB_HOME", str(home))
    for name, age in (("old.wav", 3600), ("new.wav", 60)):
        (home / "input" / name).write_bytes(b"x")
        os.utime(home / "input" / name, (time.time() - age, time.time() - age))
    captured = {}

    def fake_run(settings, input_arg, **kw):
        captured["input"] = input_arg
        captured["languages"] = settings.languages
        return []

    monkeypatch.setattr(pipeline, "run_job", fake_run)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main([]) == 0
    assert captured["input"] == "new.wav"
    assert captured["languages"] == DEFAULT_LANGUAGES, "no languages named means all of them"
    assert "new.wav" in capsys.readouterr().out


def test_bare_cli_says_nothing_to_do_when_input_is_empty(tmp_path, monkeypatch, capsys):
    from ytdub import cli

    monkeypatch.setenv("YTDUB_HOME", str(_home(tmp_path)))
    assert cli.main([]) == 1
    assert "Nothing to dub" in capsys.readouterr().out


def test_bare_cli_does_not_ask_when_several_files_are_there(tmp_path, monkeypatch, capsys):
    """The one-word command must not want a second word: it announces the pick instead."""
    import os
    import time

    from ytdub import cli, pipeline

    home = _home(tmp_path)
    monkeypatch.setenv("YTDUB_HOME", str(home))
    for name, age in (("a.wav", 3600), ("b.wav", 60)):
        (home / "input" / name).write_bytes(b"x")
        os.utime(home / "input" / name, (time.time() - age, time.time() - age))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input",
                        lambda prompt="": pytest.fail(f"must not prompt ({prompt})"))
    captured = {}
    monkeypatch.setattr(pipeline, "run_job",
                        lambda settings, arg, **k: captured.update(input=arg) or [])
    assert cli.main([]) == 0
    assert captured["input"] == "b.wav"
    assert "dubbing the newest" in capsys.readouterr().out


def test_cli_flags_in_any_order(tmp_path, monkeypatch):
    from ytdub import cli, pipeline

    monkeypatch.setenv("YTDUB_HOME", str(_home(tmp_path)))
    captured = {}

    def fake_run(settings, input_arg, **kw):
        captured.update(settings=settings, input=input_arg, **kw)
        return []

    monkeypatch.setattr(pipeline, "run_job", fake_run)
    assert cli.main(["talk.wav", "de", "--style", "duo", "fr", "--speakers", "2", "--srt-only"]) == 0
    s = captured["settings"]
    assert captured["input"] == "talk.wav" and s.languages == ["de", "fr"]
    assert s.style == "duo" and s.speakers == 2 and captured["srt_only"]


def test_a_relative_ref_is_found_from_the_project_root_too(tmp_path, monkeypatch):
    """`dub2 x.wav de --ref SPK0=voices/atlas.wav` is typed from wherever the user is,
    and the clip lives in the project. Only looking in the current directory made that
    "not found" for anyone who had not cd'd there."""
    from ytdub import cli
    from ytdub.config import Settings

    home = _home(tmp_path)
    (home / "voices").mkdir()
    # A name that does not also exist in the working directory, or the current-directory
    # rule would answer first and the fallback would go untested.
    clip = home / "voices" / "test-only-clip.wav"
    clip.write_bytes(b"x")
    monkeypatch.setenv("YTDUB_HOME", str(home))
    settings = Settings(_env_file=None, home=home)
    assert cli._parse_refs(["SPK0=voices/test-only-clip.wav"],
                           settings) == {"SPK0": clip.resolve()}
    # An absolute path is still taken as given.
    assert cli._parse_refs([f"SPK1={clip}"], settings) == {"SPK1": clip.resolve()}
    with pytest.raises(SystemExit):
        cli._parse_refs(["SPK0=voices/not-here-either.wav"], settings)


def test_cli_rejects_unknown_style_and_language(tmp_path, monkeypatch):
    from ytdub import cli

    monkeypatch.setenv("YTDUB_HOME", str(_home(tmp_path)))
    assert cli.main(["talk.wav", "--style", "nope"]) == 1
    assert cli.main(["talk.wav", "xx"]) == 1


def test_free_memory_never_raises_even_with_a_dead_cuda_context(monkeypatch):
    """It runs from the teardown path: raising here skips writing the report, which is how
    a poisoned CUDA context cost the record of an otherwise finished job."""
    import sys
    import types

    from ytdub import gpu

    fake = types.ModuleType("torch")

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            raise RuntimeError("CUDA error: device-side assert triggered")

        @staticmethod
        def memory_allocated():
            return 0

    fake.cuda = Cuda
    monkeypatch.setitem(sys.modules, "torch", fake)
    gpu.free_memory()  # must not raise


def test_sticky_cuda_faults_are_told_apart_from_recoverable_ones():
    from ytdub.stages.tts.base import is_cuda_lost

    assert is_cuda_lost(RuntimeError("CUDA error: device-side assert triggered"))
    assert is_cuda_lost(RuntimeError("CUDA error: an illegal memory access was encountered"))
    # Caused by a CUDA fault several frames down, as the real traceback was.
    deep = RuntimeError("max(): Expected reduction dim 1 to have non-zero size")
    deep.__cause__ = RuntimeError("CUDA error: device-side assert triggered")
    assert is_cuda_lost(deep)
    # Out of memory keeps the context alive, so the next line can still be synthesized.
    assert not is_cuda_lost(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_cuda_lost(RuntimeError("simulated TTS crash"))
    assert not is_cuda_lost(ValueError("nope"))


def test_hindi_gets_its_own_character_budget():
    """Measured on tangi.wav: the synthesizer speaks 6.6 Hindi characters per second where
    Spanish manages 8.2, so the shared 15 asked for twice the text a Hindi slot holds and
    the fitter compressed to its 3x cap — a fast-forward, not speech."""
    from ytdub.config import Settings
    from ytdub.models import Source
    from ytdub.pipeline import Job
    from ytdub.stages.translate.prompt import budget_chars

    settings = Settings(_env_file=None, chars_per_second=15.0,
                        chars_per_second_by_language={"hi": 12.0})
    job = Job.__new__(Job)
    job.s = settings
    job.segments = [seg(0, 0.0, 4.0), seg(1, 4.0, 8.0)]
    job.src = Source(path=Path("/tmp/x.wav"), basename="x", sha256="x", duration=10.0,
                     has_video=False)
    default = job._budgets()
    hindi = job._budgets_for("hi", default)
    spanish = job._budgets_for("es", default)
    assert spanish == default, "a language with no measured rate is untouched"
    assert hindi[0] == budget_chars(4.0, 12.0) < default[0]
    assert all(h < d for h, d in zip(hindi, default))


def test_a_configured_rate_of_zero_leaves_the_budget_alone():
    from ytdub.config import Settings
    from ytdub.models import Source
    from ytdub.pipeline import Job

    job = Job.__new__(Job)
    job.s = Settings(_env_file=None, chars_per_second_by_language={"hi": 0})
    job.segments = [seg(0, 0.0, 4.0)]
    job.src = Source(path=Path("/tmp/x.wav"), basename="x", sha256="x", duration=10.0,
                     has_video=False)
    default = job._budgets()
    assert job._budgets_for("hi", default) == default


def test_a_normal_hindi_clip_is_not_mistaken_for_a_loop(tmp_path):
    """Six real Hindi lines were dropped as "runaway" because the check assumed the Latin
    rate: 19 characters at 12 chars/s gives a 4.8s ceiling, and ordinary Hindi takes 5s."""
    from ytdub.audio import write_wav
    from ytdub.stages.tts.base import _looks_broken

    def clip(tmp_path, seconds, name):
        sr = 24000
        t = np.arange(int(seconds * sr)) / sr
        path = tmp_path / name
        write_wav(path, (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr)
        return path

    text = "नहीं, वह गैस वाला है ब्रो।"          # 19 spoken characters
    natural = clip(tmp_path, 5.0, "natural.wav")
    loop = clip(tmp_path, 11.0, "loop.wav")
    assert _looks_broken(natural, text, chars_per_second=8.0) is None
    assert _looks_broken(loop, text, chars_per_second=8.0).startswith("runaway")
    # The same 5s clip judged by the Latin default is what went wrong before.
    assert _looks_broken(natural, text, chars_per_second=12.0).startswith("runaway")


def test_silence_is_still_caught_by_the_sanity_check(tmp_path):
    from ytdub.audio import write_wav
    from ytdub.stages.tts.base import _looks_broken

    path = tmp_path / "silent.wav"
    write_wav(path, np.zeros(24000, dtype=np.float32), 24000)
    assert _looks_broken(path, "anything", chars_per_second=8.0) == "silent"
