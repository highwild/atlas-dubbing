"""Unit tests for the smaller pure pieces: cache, SRT, fragment merging, reference
selection, transcription post-processing, diarization clustering, download options,
network hardening and the CLI."""

from __future__ import annotations

import socket

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


def test_fragment_never_merges_across_speakers_or_big_gaps():
    from ytdub.stages.tts.base import merge_short_fragments

    segs = [seg(0, 0, 2, translated="Pierwsza osoba", speaker="SPK0"),
            seg(1, 2.1, 2.3, translated="No!", speaker="SPK1"),
            seg(2, 2.5, 5, translated="Znowu pierwsza", speaker="SPK0"),
            seg(3, 9.0, 9.2, translated="ok", speaker="SPK0")]
    merged, stuck = merge_short_fragments(segs, min_chars=3, max_gap=1.5)
    assert len(merged) == 4 and sorted(stuck) == [1, 3]


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
