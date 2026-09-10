"""``dub``: one command per job.

    dub2                                        # newest file in input/, all languages
    dub2 myfile.wav                             # all default languages
    dub2 myfile.wav --style duo --speakers 2    # conversation, two voices
    dub2 myfile.wav --style professional de fr  # subset of languages
    dub2 --help                                 # usage + available styles

Flags and languages may come in any order.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ytdub.config import DEFAULT_LANGUAGES, Settings, available_styles


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dub", description="Dub a video or audio file into other languages in the "
        "speaker's own cloned voice. Runs entirely locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="file in input/ (or a path, or a YouTube URL)")
    p.add_argument("langs", nargs="*", help=f"target languages (default: {' '.join(DEFAULT_LANGUAGES)})")
    job = p.add_argument_group("job")
    job.add_argument("--style", help="style preset from synopses/ (default: casual)")
    job.add_argument("--speakers", type=int, metavar="N",
                     help="number of voices: 0 = count them with the diarizer (default), "
                          "1 = treat the file as one speaker, N = force N")
    job.add_argument("--source", metavar="LANG", help="source language (default: auto-detect)")
    job.add_argument("--ref", action="append", default=[], metavar="[SPK=]PATH",
                     help="voice reference clip, e.g. --ref atlas.wav or --ref SPK1=guest.wav")
    job.add_argument("--speaker-map", dest="speaker_map", metavar="LINES=SPK,...",
                     help="re-label lines after diarization, e.g. "
                          "'6-15=SPK1,22=SPK3' (1-based review-SRT line numbers)")
    job.add_argument("--voice", metavar="PATH",
                     help="your own voice clip (file or folder): matched to whichever "
                          "speaker sounds like it, so the per-file SPK label never has to "
                          "be named. Also YTDUB_VOICE")
    review = p.add_argument_group("review loop")
    review.add_argument("--srt-only", action="store_true",
                        help="stop after translation; write <lang>.review.srt for checking")
    review.add_argument("--from-review", action="store_true",
                        help="dub from the (edited) <lang>.review.srt files, skipping translation")
    review.add_argument("--verify", action="store_true",
                        help="back-translate the translation and revise lines whose meaning "
                             "drifted (about three times the translation time; writes "
                             "<lang>.verify.txt). Also YTDUB_VERIFY=1")
    review.add_argument("--verify-model", dest="verify_ollama_model", metavar="MODEL",
                        help="model for the verification pass only (default: the "
                             "translation model). A bigger model is affordable here: it is "
                             "only ever resident after translation is done")
    cache = p.add_argument_group("cache")
    cache.add_argument("--force", "--no-cache", dest="force", action="store_true",
                       help="ignore every cached result and recompute")
    tune = p.add_argument_group("tuning")
    tune.add_argument("--separate", action="store_true",
                      help="isolate vocals with demucs first (only for full mixes)")
    tune.add_argument("--asr-model", help="Whisper size name or local model directory")
    tune.add_argument("--ollama-model", help="translation model (default qwen3:8b)")
    tune.add_argument("--tts", dest="tts_backend",
                      help="TTS backend: 'chatterbox' or module.path:ClassName")
    tune.add_argument("--translator", help="translator: 'ollama' or module.path:ClassName")
    tune.add_argument("--num-ctx", type=int, dest="ollama_num_ctx", help="Ollama context window")
    tune.add_argument("--max-ratio", type=float, help="compression cap (default 1.2)")
    tune.add_argument("--max-delay", type=float, help="soft cap on lateness, seconds")
    tune.add_argument("--cps", type=float, dest="chars_per_second",
                      help="speaking rate for translation budgets (default 15)")
    tune.add_argument("--min-confidence", type=float,
                      help="drop transcribed segments below this mean word probability")
    tune.add_argument("--loudness-ref", dest="loudness_reference", metavar="PATH",
                      help="match loudness to this file (e.g. the full mix when dubbing a stem)")
    tune.add_argument("--no-loudness", action="store_true", help="skip loudness matching")
    tune.add_argument("--no-mux", action="store_true", help="never write the preview .mp4")
    tune.add_argument("--no-number-expansion", dest="no_number_expansion", action="store_true",
                      help="send digits to the synthesizer as digits instead of writing "
                           "them out as words (also YTDUB_EXPAND_NUMBERS=0). The SRT is "
                           "unaffected either way")
    tune.add_argument("--device", help="cuda or cpu (default: auto)")
    net = p.add_argument_group("network / running")
    net.add_argument("--allow-ipv6", action="store_true", help="do not force IPv4")
    net.add_argument("--cookies-from-browser", metavar="BROWSER", help="yt-dlp cookies source")
    net.add_argument("--detach", action="store_true",
                     help="run in the background, surviving SSH disconnects; logs to output/")
    net.add_argument("--log-level", default="INFO")
    return p


def _input_candidates(settings: Settings) -> list[Path]:
    """Files in ``input/``, newest first. Empty when the directory is missing or empty."""
    if not settings.input_dir.is_dir():
        return []
    files = [f for f in settings.input_dir.iterdir()
             if f.is_file() and not f.name.startswith(".")]
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def _pick_input(settings: Settings) -> str | None:
    """What ``dub2`` with no file should dub: the newest file in ``input/``.

    "The file I just dropped in" is the only reading of a bare command that is never a
    guess in practice, so newest wins over alphabetical order.

    It does not ask for confirmation. It used to, and that was wrong: a one-word command
    that immediately wants a second word is not a one-word command, and the answer to the
    question is Enter every time. The pick is announced instead, as the first line of the
    run, so a wrong file is visible in the first second and Ctrl-C costs nothing — much
    less than a prompt on every run costs when the pick is right.
    """
    candidates = _input_candidates(settings)
    if not candidates:
        return None
    picked = candidates[0]
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(picked.stat().st_mtime))
    print(f"No file named; dubbing the newest in {settings.input_dir}: "
          f"{picked.name} ({when})", flush=True)
    return picked.name


def _usage(p: argparse.ArgumentParser, settings: Settings) -> None:
    p.print_help()
    styles = available_styles(settings.styles_dir)
    print("\nAvailable styles:" + ("".join(f"\n  {s}" for s in styles) or "\n  (none found in "
                                   f"{settings.styles_dir})"))
    if settings.input_dir.is_dir():
        files = sorted(f.name for f in settings.input_dir.iterdir() if f.is_file())
        print(f"\nFiles in {settings.input_dir}:" + ("".join(f"\n  {f}" for f in files) or "\n  (empty)"))
    print("\nExamples:\n  dub2                       # newest file in input/, all languages\n"
          "  dub2 doc.wav --style professional de fr\n  dub2 talk.wav --srt-only de pl   "
          "# then edit output/talk/de.review.srt\n  dub2 talk.wav --from-review de pl\n"
          "  dub2 solo.wav --srt-only pl --verify   # back-translation check + "
          "output/solo/pl.verify.txt")


def _parse_refs(values: list[str]) -> dict[str | None, Path]:
    refs: dict[str | None, Path] = {}
    for v in values:
        spk, _, path = v.rpartition("=") if "=" in v else ("", "", v)
        p = Path(path).expanduser()
        if not p.is_file():
            raise SystemExit(f"--ref: {p} not found")
        refs[spk or None] = p.resolve()
    return refs


def _detach(argv: list[str], settings: Settings, input_arg: str) -> int:
    log_dir = settings.output_root / Path(input_arg).stem
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dub-{time.strftime('%Y%m%d-%H%M%S')}.log"
    args = [a for a in argv if a != "--detach"]
    with open(log_path, "ab") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "ytdub.cli", *args], stdout=fh,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    print(f"Running detached (pid {proc.pid}); safe to disconnect.\n  tail -f {log_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_intermixed_args(argv)

    overrides = {k: v for k, v in {
        "style": args.style, "speakers": args.speakers, "source_lang": args.source,
        "asr_model": args.asr_model, "ollama_model": args.ollama_model,
        "ollama_num_ctx": args.ollama_num_ctx, "max_ratio": args.max_ratio,
        "max_delay": args.max_delay, "chars_per_second": args.chars_per_second,
        "min_confidence": args.min_confidence, "loudness_reference": args.loudness_reference,
        "device": args.device, "cookies_from_browser": args.cookies_from_browser,
        "tts_backend": args.tts_backend, "translator": args.translator,
        "verify_ollama_model": args.verify_ollama_model, "voice": args.voice,
        "speaker_map": args.speaker_map or None,
    }.items() if v is not None}
    if args.langs:
        overrides["languages"] = [lang.lower() for lang in args.langs]
    for flag, key, value in ((args.force, "force", True), (args.separate, "separate", True),
                             (args.no_loudness, "match_loudness", False),
                             (args.no_mux, "mux_video", False),
                             (args.allow_ipv6, "force_ipv4", False),
                             (args.verify, "verify", True),
                             (args.no_number_expansion, "expand_numbers", False)):
        if flag:
            overrides[key] = value
    settings = Settings(**overrides)

    if not args.input:
        picked = _pick_input(settings)
        if picked is None:
            _usage(parser, settings)
            print(f"\nNothing to dub: {settings.input_dir} is empty. Drop a file in "
                  "there and run dub2 again, or name one.")
            return 1
        args.input = picked
    styles = available_styles(settings.styles_dir)
    if settings.style not in styles:
        print(f"No style called {settings.style!r}. Available: {', '.join(styles) or 'none'}")
        return 1
    if args.srt_only and args.from_review:
        print("--srt-only and --from-review are mutually exclusive")
        return 1
    if not args.srt_only:
        from ytdub.stages.tts.base import tts_class

        supported = tts_class(settings.tts_backend).supported_languages
        bad = [lang for lang in settings.languages if lang not in supported]
        if bad:
            print(f"Unsupported by the {settings.tts_backend} TTS: {', '.join(bad)}. "
                  f"Supported: {' '.join(sorted(supported))}")
            return 1
    refs = _parse_refs(args.ref)
    if args.detach:
        return _detach(argv, settings, args.input)

    from ytdub.logging import setup_logging
    from ytdub.pipeline import run_job

    setup_logging(args.log_level.upper(), force=True)
    results = run_job(settings, args.input, srt_only=args.srt_only,
                      from_review=args.from_review, user_refs=refs)
    return 0 if all(r.status in ("ok", "srt-only") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
