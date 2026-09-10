# atlas-dubbing

Local voice-cloning dubbing. Give it a video or audio file and it writes an alternate
audio track per language, in the original speaker's cloned voice, with **exactly the
same duration as the source**, ready to upload to YouTube as a multi-language track,
plus a matching subtitle file.

Everything runs on one machine: no cloud APIs, no per-use cost. This started as a
patched fork of [mazzasaverio/youtube-auto-dub](https://github.com/mazzasaverio/youtube-auto-dub)
and has been rebuilt per [`DUBBING_PIPELINE_SPEC.md`](DUBBING_PIPELINE_SPEC.md).
Install and usage are in [`SETUP.md`](SETUP.md).

```bash
dub myfile.wav                              # all default languages (de fr pl es nl hi)
dub myfile.wav --style duo --speakers 2     # conversation, two cloned voices
dub myfile.wav --style professional de fr   # subset of languages
dub                                         # usage + available styles
```

## Pipeline

```
input/<file>  (video or audio; a clean voice-only stem gives the best results)
  │
  ├─ once per source file, cached ─────────────────────────────────────────
  │   acquire → [demucs] → transcribe (faster-whisper + VAD + confidence filter)
  │           → [diarize] → pick one voice reference per speaker
  │
  ├─ translate every language  (Qwen3 via Ollama, batched, with context)
  │   → output/<file>/<lang>.review.srt              ← stop here with --srt-only
  │   [Ollama model unloaded]
  │
  └─ per language  (Chatterbox multilingual)
      synthesize → fit onto the timeline → loudness-match → <lang>.wav + <lang>.srt
                                                         → <lang>.mp4 (video input only)
```

Each model is loaded, used and freed before the next one loads, so the job never
needs Whisper, the LLM and Chatterbox resident together.

## What changed from the reference implementation

**Timeline fitting (`stages/fit.py`).** The reference squeezed each translated clip
into its own source slot, time-compressing 60-85% of lines. Fitting is now one global
optimisation: total duration is exact, but a clip may run past its slot into the
silence that follows, pushing later lines back, as far along the timeline as there is
slack. Compression is a last resort, capped (default 1.2x), and goes preferentially to
long clips at small ratios where it is least audible. Clips never start before their
source line and never overlap. It is solved as a small linear programme (HiGHS via
scipy) and rendered in integer samples, so output length is exact by construction.
Each run logs how many lines were compressed and the worst ratio, next to how many
the reference approach would have compressed.

**Document-level translation (`stages/translate/`).** Lines go to the LLM in numbered
batches of ~30 with a style synopsis, glossary, per-line character budgets (which
include borrowable silence), the previous lines' translations and a peek at the next
lines. Output is JSON-schema constrained and parsed strictly; a bad batch is split and
retried, then translated line by line, and only then left in the source language (and
reported). Lines that overshoot their budget are re-requested with their neighbours so
words can move to a line with slack; a revision is only kept if it is shorter.
`num_ctx` is sent on every request, prompts are size-checked before sending, a
calibration request proves the server honours the window, and `prompt_eval_count` is
checked after every request, so silent context truncation fails loudly.

**Cache and resume (`cache.py`).** Every stage result is cached under `work/`, keyed
on a hash of exactly the inputs that affect it. Transcription, diarization and voice
references run once and are shared by all languages; translations and every
synthesized clip survive a crash. Re-running a job skips whatever is complete, and a
cache miss logs which inputs changed. `--force` recomputes everything.

**Review loop.** `--srt-only` stops after translation. Edit
`output/<file>/<lang>.review.srt`, then `--from-review` dubs from the edited text; only
changed lines are resynthesized. A normal re-run never overwrites an edited review file.

## Outputs

`output/<basename>/`:

| File | What |
|---|---|
| `<lang>.wav` | Dubbed track: 48 kHz mono 24-bit PCM, exactly the source's length, loudness-matched to the source |
| `<lang>.srt` | Subtitles on the *fitted* timings (matches the audio) |
| `<lang>.review.srt` | Translation on source timings, for review and correction |
| `<lang>.mp4` | Preview mux (video input only; written after the WAV and SRT) |
| `source.srt` | The transcript, with speaker labels |
| `dropped.txt` | Transcribed segments dropped as probable noise; review these |
| `report.json` | Per-language fit metrics, loudness, translation stats, lost lines |
| `dub.log` | Full log of every run |

## Tests

```bash
uv pip install -e ".[dev]"
pytest
```

The tests need no GPU, models or Ollama. `tests/test_fit.py` proves the fitting stage
on synthetic clips of known length (exact duration, no overlap, no early starts,
minimal compression: on a 13-line speech-like clip it compresses 3 lines, all by at most
1.05x, where the reference approach would compress 11). `tests/test_pipeline_e2e.py`
runs whole jobs with fake models but real ffmpeg (skipped if ffmpeg is missing).

Translation quality can't be unit tested and needs a native speaker.

## Licences

Everything used permits commercial use: faster-whisper and Whisper weights (MIT),
Qwen3 (Apache-2.0), Chatterbox (MIT), Resemblyzer (Apache-2.0), yt-dlp (Unlicense),
Demucs (MIT, optional), pyannote (MIT code, gated weights; optional). The reference
implementation's NLLB-200 (CC-BY-NC) and XTTS (Coqui Public Model Licence,
non-commercial) backends were removed for that reason.
