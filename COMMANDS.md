# COMMANDS — every command, flag and setting

Reference for this box as it stands. State at the time of writing: the rebuild in
`/home/atlas/atlas-dubbing-rebuild` (branch `rebuild/spec-pipeline`), **172 tests passing**,
and the last real run being the no-flags one this file is built around:

```bash
.venv/bin/dub hydro.wav pl fr
```

→ `output/hydro/{pl,fr}.wav`, both exactly 335.519 s, 51 lines, 0 dropped. The diarizer
found 5 voices, the clip in `.env` matched `SPK2` at 0.99 (the presenter's 9 lines) and the
other four speakers kept their own auto-cut references. Spot-checking the synthesized audio
of every speaker's first lines against all five references put each line closest to its own
voice (0.82–0.92 against 0.44–0.69), so the file is genuinely five voices, not one.

Flags that run needed: none.

---

## The box

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop, 16 GB VRAM, driver 595.84 |
| Python | 3.11.16, in `.venv/` |
| Package manager | `uv` 0.12.9 at `~/.local/bin/uv` |
| Ollama | `http://127.0.0.1:11434`, running |
| Whisper | `/home/atlas/models/faster-whisper-large-v3` (set in `.env`) |
| Project | `/home/atlas/atlas-dubbing-rebuild` |
| Hints / glossary | `hints.txt`, `hints.pl.txt`, `glossary.txt` in the project root |

### `dub2` is the command. `dub` is not.

`~/.local/bin/dub2` → `.venv/bin/dub` in **this** repo. Use it.

`~/.local/bin/dub` → `~/youtube-auto-dub/duball.sh`, the **legacy** fork's shell script.
It reads `~/youtube-auto-dub/input/`, writes `~/youtube-auto-dub/output/`, and has none of
the rebuild in it. Do not use it, and do not be surprised when it "works" but ignores
`--verify` and hints.

`~/.local/bin` is on PATH from `~/.bashrc`, so `dub2` works in an interactive shell. In a
script or a bare `bash -c`, use the full path or the module form.

Three equivalent ways to run it, all from the repo root (so `.env` and `synopses/` resolve):

```bash
cd /home/atlas/atlas-dubbing-rebuild
dub2 cheese.wav pl                              # if ~/.local/bin is on PATH
.venv/bin/dub cheese.wav pl                     # always works
.venv/bin/python -m ytdub.cli cheese.wav pl     # bypasses the entry point entirely
```

`dub2` with no arguments runs the newest file in `input/`, in every language:

```bash
dub2                    # == dub2 <newest file in input/> de fr pl es nl hi
dub2 --help             # usage, available styles, what is in input/
```

That is the whole flow: copy a file into `input/`, type `dub2`, walk away. Nothing else is
needed — voices are counted, references are cut, your clip in `.env` is matched by voice,
every language is translated and dubbed.

It says which file it is about to use, and starts work immediately — no confirmation,
because a one-word command that immediately wants a second word is not a one-word command.
To dub something else, name it: `dub2 hydro.wav`. An empty `input/` prints usage and runs
nothing.

Which file counts as newest is by modification time, so a file you just copied over is the
one it picks. The choice is the first line of the log.

---

## Argument reference

Positional, and they may be in any order with the flags:

```bash
dub2 <input> <lang...>
```

Both are optional: no `input` means the newest file in `input/`, no languages means all of
`de fr pl es nl hi`.


| Positional | Meaning |
|---|---|
| `input` | file name inside `input/`, a path, or a YouTube URL |
| `lang...` | target languages, ISO codes. Default: `de fr pl es nl hi` |

Per-language codes: the shipped TTS (Chatterbox 0.1.7) accepts `ar da de el en es fi fr he
hi it ja ko ms nl no pl pt ru sv sw tr zh`. Anything outside that set is rejected before
the job starts, naming what is supported — no hours wasted before finding out.

### job

| Flag | Default | Notes |
|---|---|---|
| `--style NAME` | `casual` | a preset from `synopses/`. Currently `casual`, `duo`, `professional`. Drop a new `.txt` in and it appears here. |
| `--speakers N` | `0` | `0` = count the voices with the diarizer (default), `1` = one speaker, `N` = force N. Multi-voice mode tags every line with its `SPK` label and gives each speaker their own cloned voice. |
| `--source LANG` | auto | source language; skip Whisper's detection when you know it |
| `--ref [SPK=]PATH` | auto | voice reference clip, repeatable: `--ref atlas.wav` or `--ref SPK1=guest.wav`. Without it, a reference is cut from the input itself. |
| `--voice PATH` | `YTDUB_VOICE` | **your own voice clip** (file or a folder of clips), matched to whichever speaker sounds like it — so the per-file `SPK` label never has to be named. A second label that also matches you is folded into the first. Also `YTDUB_VOICE`. |

### review loop

| Flag | What it does |
|---|---|
| `--srt-only` | stop after translation. Writes `<lang>.review.srt`, skips synthesis. Seconds to a couple of minutes instead of half an hour. |
| `--from-review` | dub from the (edited) `<lang>.review.srt` files, skipping transcription and translation. Mutually exclusive with `--srt-only`. |
| `--verify` | back-translation verification. ~3× the translation time. Writes `<lang>.verify.txt`. Also `YTDUB_VERIFY=1`. |
| `--verify-model MODEL` | model for the verification pass only; translation stays on `--ollama-model`. `qwen27-24k:latest` is the one worth using — it flagged 3 lines where 8b flagged 7, and its flags were the accurate ones. |

### cache

| Flag | What it does |
|---|---|
| `--force`, `--no-cache` | ignore every cached stage result and recompute. Front half (transcription, diarization, references), translations, and every synthesized clip. |

### tuning

| Flag | Default | Notes |
|---|---|---|
| `--separate` | off | run Demucs first to isolate vocals. Only for a full mix with music under the speech. |
| `--asr-model SPEC` | `.env` value | Whisper size name (`large-v3`) or a local model directory |
| `--ollama-model MODEL` | `qwen3:8b` | translation model |
| `--num-ctx N` | `8192` | Ollama context window. Sent on every request; a calibration request proves the server honours it. |
| `--tts SPEC` | `chatterbox` | TTS backend, or `module.path:ClassName` |
| `--translator SPEC` | `ollama` | translator backend, or `module.path:ClassName` |
| `--max-ratio R` | `1.2` | hard cap on time-compression per clip |
| `--max-delay S` | `2.0` | soft cap on how late a line may start |
| `--cps N` | `15.0` | speaking rate used for translation character budgets |
| `--min-confidence P` | `0.45` | drop transcribed segments below this mean word probability as probable noise |
| `--loudness-ref PATH` | input file | match loudness to this instead: `--loudness-ref mix.wav` when dubbing a clean stem |
| `--no-loudness` | — | skip loudness matching |
| `--no-mux` | — | never write the preview `<lang>.mp4` |
| `--no-number-expansion` | off | send digits to the synthesizer as digits instead of words. Also `YTDUB_EXPAND_NUMBERS=0`. The SRT is the same either way. |
| `--device cuda\|cpu` | auto | auto = cuda if torch sees a GPU |

### network / running

| Flag | What it does |
|---|---|
| `--allow-ipv6` | do not force IPv4 (IPv4 is forced process-wide by default) |
| `--cookies-from-browser BROWSER` | yt-dlp cookies, for when YouTube says "not a bot?" |
| `--detach` | run in the background, surviving SSH disconnect. Logs to `output/<name>/dub-<timestamp>.log`. |
| `--log-level LEVEL` | `INFO` default. `DEBUG` prints the exact text sent to the synthesizer for every line whose numbers were written out. |

---

## Settings without a flag

These come from `YTDUB_*` environment variables or `.env` in the project root (flag wins,
then env, then `.env`, then the default). Full list in `src/ytdub/config.py`.

```bash
YTDUB_DIARIZE_METHOD=embedding        # token-free fallback diarizer
YTDUB_HF_TOKEN=hf_...                 # pyannote diarization (terms must be accepted)
YTDUB_VOICE=./voices/atlas.wav        # your clip, matched by voice
YTDUB_VERIFY=1                        # same as --verify
YTDUB_VERIFY_OLLAMA_MODEL=qwen27-24k:latest
YTDUB_EXPAND_NUMBERS=0                # same as --no-number-expansion
YTDUB_OLLAMA_MODEL=qwen3:14b
YTDUB_LANGUAGES='["de","fr"]'         # the default language set, without passing codes
YTDUB_MIN_CONFIDENCE=0.5
YTDUB_TTS_SEED=4321                   # different take, same everything else
YTDUB_TTS_EXAGGERATION=0.7            # more expressive read; affects every clip's cache key
YTDUB_MAX_RATIO=1.15
YTDUB_HOME=/somewhere/else            # moves input/, output/, work/, synopses/, hints
```

Precedence, highest first: a command-line flag, then `YTDUB_*` in the environment, then
`.env`, then the default in `config.py`. A flag only overrides when you pass it, with one
exception worth knowing: `--verify` is a store-true flag, so omitting it leaves
`YTDUB_VERIFY` in charge, while passing it forces verification on. The `--no-*` flags work
the same way in reverse — passing them forces the feature off and wins over `.env`.

Current `.env`:

```
YTDUB_ASR_MODEL=/home/atlas/models/faster-whisper-large-v3
YTDUB_VOICE=./voices/atlas.wav
YTDUB_HF_TOKEN=hf_...   (real token; the file is gitignored)
```

Validated at startup, before any model loads: a bad style name, an unsupported language,
`--srt-only` together with `--from-review`, or a missing `--ref` file exits immediately
with a message and no work done.

---

## Workflows

### The one you will mostly use

```bash
cd /home/atlas/atlas-dubbing-rebuild
dub2 cheese.wav pl
```

Audio in `input/`, outputs in `output/cheese/`. Re-running reuses everything already done.

### Review before the expensive part

```bash
dub2 cheese.wav --srt-only pl            # translate, stop, write pl.review.srt
$EDITOR output/cheese/pl.review.srt      # edit text only: no added, removed or merged cues
dub2 cheese.wav --from-review pl         # dub from the edited file
```

Timing edits are ignored (the fitter decides timings) and warned about; empty cues are an
error. A normal re-run never overwrites a review file you have edited — delete it to
regenerate.

### Diagnostic verification, with the good checker

```bash
dub2 cheese.wav --srt-only pl --verify --verify-model qwen27-24k:latest
$EDITOR output/cheese/pl.verify.txt
```

The report ends with the terms it flagged and the verified line that fixed each one, which
is what you copy into `hints.pl.txt`, then rerun. Changing a hint retranslates.

### Two voices

```bash
dub2 chat.wav --style duo --speakers 2
dub2 panel.wav --style duo --speakers 0          # auto-detect the count
dub2 chat.wav --style duo --speakers 2 --ref SPK1=guest.wav
```

### Most target languages at once

```bash
dub2 cheese.wav                       # all six, one front half, one translation pass
dub2 cheese.wav de fr pl
dub2 cheese.wav --style professional de fr
```

### A long job over SSH

```bash
dub2 two-hour-video.wav de fr pl --detach
tail -f output/two-hour-video/dub-*.log
```

### From YouTube

```bash
dub2 'https://www.youtube.com/watch?v=XXXX' pl
dub2 'https://...' pl --cookies-from-browser firefox     # if it says "not a bot?"
```

### Recompute from scratch

```bash
dub2 cheese.wav pl --force
```

### Faster iteration on translation only

```bash
dub2 cheese.wav --srt-only pl --force          # retranslate, no synthesis
dub2 cheese.wav --srt-only pl de fr --verify   # check three languages in one pass
```

---

## What a run writes

`output/<input name>/`:

| File | What |
|---|---|
| `<lang>.wav` | the dub: 48 kHz mono 24-bit PCM, **exactly** the source's length, loudness-matched |
| `<lang>.srt` | subtitles on the fitted timings, digits kept as digits |
| `<lang>.review.srt` | translation on source timings, for review/editing |
| `<lang>.verify.txt` | with `--verify`: every flagged line in full + suggested hints |
| `<lang>.mp4` | preview mux, video input only, written after the WAV and SRT |
| `source.srt` | the transcript, with speaker labels |
| `dropped.txt` | segments dropped as low-confidence/non-speech — read this if speech went missing |
| `report.json` | per-language fit metrics, loudness, translation stats, lost lines |
| `dub.log` | every run, appended |

`work/<name>-<sha12>/` holds the resumable cache: `cache/` (`transcribe`, `diarize`,
`references`, `translate-<lang>`), `clips/<lang>/` (one WAV per line), `refs/` (voice
references), `audio16k.wav` (Whisper) + `audio24k.wav` (voice references), and
`review_written.json` (which review files you have edited, so they are never overwritten).

Delete `work/` to force everything; `--force` does the same without deleting history.

Exit code is `0` when every language is `ok` or `srt-only`, `1` otherwise. A language is
`degraded` (exit 1) if a line's audio is missing, `failed` on an exception.

---

## Failure behaviour worth knowing

- **Duration.** After writing each `<lang>.wav` the pipeline re-measures it with ffprobe
  against the probed source. More than 2 ms off fails that language, loudly. It does not
  happen: the sample count is exact by construction.
- **Compression cap.** If the timeline cannot fit within `--max-ratio`, the language still
  completes and logs a warning naming the lines and worst ratio. Same for lines pushed
  later than `--max-delay`. Both appear in `report.json`.
- **Lost lines.** A TTS failure marks the language `degraded`, logs the line numbers as
  ERROR and repeats them in the summary.
- **Context truncation.** `num_ctx` is sent on every request, every prompt is size-checked
  before sending, and `prompt_eval_count` is checked after. A server that ignores the
  window fails loudly instead of silently truncating the glossary off the prompt.
- **Hints and glossary.** Protected terms missing from a translation are warned about;
  a protected term is never rewritten by number expansion.

---

## Health checks

```bash
# Is Ollama up, and what is pulled?
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool | grep '"name"'
ollama ps                                    # what is resident in VRAM right now

# What actually got sent to the synthesizer, per line
dub2 cheese.wav pl --log-level DEBUG 2>&1 | grep speaking

# What number expansion would do to a line, without running anything
.venv/bin/python -c "
from ytdub.stages.numbers import expand_text
for line in ['z 10 brązowych liści', 'Daję 4,5', 'Ocena 9 na 10', 'o 7:30', 'wersja 4.5.1']:
    print(f'{line!r} -> {expand_text(line, \"pl\")[0]!r}')"

# The whole test suite (no GPU, no models, no Ollama needed)
.venv/bin/pytest -q

# Restore one file to its committed content
git checkout 273d703 -- synopses/casual.txt
```

Models on this box right now: `qwen3:8b` (5.2 GB), `qwen3:14b` (9.3 GB),
`gpt-oss:20b` (13.8 GB), `qwen27-24k:latest` / `qwen27-16k` / `qwen27-32k` (15.2 GB each),
plus `nomic-embed-text`. Two models are never resident at once: the pipeline unloads
Ollama before synthesis starts.

---

## Dropping in a file

```bash
dub2 hydro.wav pl fr
```

or, for the file you just dropped in, every language:

```bash
dub2
```

That is the whole command. Transcription, diarization, one reference clip per detected
speaker, translation, synthesis, fitting and assembly all happen with no flags: the
diarizer counts the voices by itself (`--speakers 0` is the default), and the voice clip
named in `.env` is matched to whichever speaker sounds like you.

`.env` is read from the project root whatever directory the command is typed from, so
`YTDUB_VOICE` and `YTDUB_HF_TOKEN` apply everywhere, not only from the repo root.

## Diarization

| | |
|---|---|
| `pyannote` (default) | `pyannote/speaker-diarization-3.1`. Needs a free Hugging Face token **with the terms accepted** at `hf.co/pyannote/speaker-diarization-3.1`, in `.env` as `YTDUB_HF_TOKEN=hf_...`. Measured on `hydro.wav`: 5 voices, and the presenter's own 9 lines came back as one label. |
| `embedding` (fallback) | Resemblyzer embeddings + clustering. Token-free, no account. On the same file it merged the presenter with a second man — 0.76 similarity where the threshold is 0.75, i.e. it genuinely cannot separate those two voices. |

With no token the run says so in one warning and falls back to `embedding` rather than
failing. That fallback is not equivalent, which is why it is a warning and not a note.

Force the method per run with `YTDUB_DIARIZE_METHOD=pyannote|embedding`.

Labels are assigned per file by clustering, so the same person is a different `SPK` number
on the next video. Never hard-code one.

### When the count comes out wrong

```bash
dub2 hydro.wav pl --speakers 5                    # force the count
dub2 hydro.wav pl --speaker-map "22=SPK3"         # move one line's label (1-based review-SRT numbers)
```

`--speaker-map` re-labels lines after diarization; it accepts ranges
(`"6-15=SPK1,22=SPK3"`) and is applied before the result is cached, so nothing downstream
can read the uncorrected labels.

## Your own voice, matched by voice

A diarization label (`SPK0`, `SPK2`) is invented per file, so `--ref SPK0=you.wav` is a
trap: on the next video you may be `SPK1` and the clip silently clones the wrong person.

`--voice` removes the label from the equation. The clip is embedded and compared against
the reference clips the pipeline cut for each speaker; it is attached to whoever sounds
like it.

```bash
dub2 hydro.wav pl fr --voice ./voices
```

A folder works, so you can keep several clips (different microphones) and each one is
matched independently:

```
voices/atlas-internal.wav
voices/atlas-outdoor.wav
```

**What it is compared against matters more than the threshold.** The comparison is against
the pipeline's own cuts — already trimmed, level-normalised, one per speaker — not against
raw in-file audio. Measured with the presenter's clip on `hydro.wav`: against the cuts, his
own speaker scored **0.99** and the nearest other speaker **0.66**; against raw audio the
same comparison came out **0.64** and **0.61**, which is a coin flip. That is why matching
happens after the references exist.

Matching is conservative. Two rules, both must hold:

| Rule | Default | Meaning |
|---|---|---|
| `voice_match_threshold` | `0.75` | best speaker must score at least this |
| margin | `0.05` | and beat the runner-up by at least this |

Otherwise the automatic reference is kept and the log says which rule failed. Attaching
your voice to the wrong speaker is worse than not attaching it.

If a **second** label also matches you at `0.80` or better, it is folded into the first:
the segment labels are rewritten, the reference is dropped, and both stretches of your
performance come out in your voice instead of one of them being cloned from a stranger.
That merge never overrules an `--ref SPK3=...` you typed yourself.

Set the path once in `.env` with `YTDUB_VOICE=./voices/atlas.wav` and it applies to every
future job.

Keep clips to **about 10 seconds** of clean solo speech — Chatterbox truncates the
reference to its first 10 s (`DEC_COND_LEN`), so anything longer is discarded, and under
~4 s the clone comes out thin and generic.

## `.env`

```bash
YTDUB_ASR_MODEL=/home/atlas/models/faster-whisper-large-v3
YTDUB_VOICE=./voices/atlas.wav
YTDUB_HF_TOKEN=hf_...            # pyannote diarization; accept terms at hf.co/pyannote/speaker-diarization-3.1
```

`.env` is gitignored, and every value in it can be overridden per run by the matching
`--flag` or by a `YTDUB_*` environment variable.

## Number expansion, in one place

Digits are written out as words for the synthesizer only. The SRT and the review file keep
the digits.

```
SRT:    Daję 4,5, bo dziś czuje się szorstko
audio:  Daję cztery przecinek pięć, bo dziś czuje się szorstko
```

Kept as digits on purpose: clock times (`7:30`), versions and IPs (`4.5.1`), dates
(`10.06.2024`), ranges (`3-4`), percentages (`100%`), units and model numbers (`12GB`,
`1080p`, `RTX 3080`, `PS 5`), and any digit inside a glossary or hints term.

Polish gets the genitive after `z`, `około`, `od`, `do` and friends, because
`num2words` only produces the nominative and `z dziesięć brązowych liści` is wrong:
`z 10` → `z dziesięciu`. The table covers 0–999; above that a number stays a digit rather
than being said wrongly. **That table is the one thing here worth a native speaker's eye** —
`src/ytdub/stages/numbers.py`, `_PL_GENITIVE`.

Hindi has no `num2words` converter, so it keeps its digits and logs that fact. German,
French, Spanish and Dutch need no case handling.

Turn it off per run with `--no-number-expansion`, or in `.env` with
`YTDUB_EXPAND_NUMBERS=0`.

---

## Reinstalling / syncing

```bash
cd /home/atlas/atlas-dubbing-rebuild
~/.local/bin/uv sync --python 3.11 --extra ml --extra dev
```

The `ml` extra carries `chatterbox-tts==0.1.7` with `transformers==5.2.0`,
`diffusers==0.29.0` and `setuptools<80`. Do not let a resolver move those independently:
Chatterbox breaks on other versions, and without `setuptools<80` every TTS call fails with
`'NoneType' object is not callable`. After any change, run one real synthesis before
trusting it.
