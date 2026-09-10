# Setup

For Ubuntu with an NVIDIA GPU (16 GB VRAM), Ollama installed and running, Python 3.11,
`uv`, and ffmpeg on PATH.

## 1. Clone and install

```bash
git clone <your-repo-url> atlas-dubbing
cd atlas-dubbing
sudo apt install build-essential python3.11-dev   # webrtcvad (Resemblyzer) has no wheel

uv sync --python 3.11 --extra ml --extra dev      # installs exactly what uv.lock pins
source .venv/bin/activate
```

The project is pinned to Python 3.11 (`requires-python = "==3.11.*"`) and `uv.lock`
records the exact versions resolved for it on Linux x86_64 (torch 2.6.0 + CUDA 12.4 /
cuDNN 9.1 wheels, transformers 5.2.0, diffusers 0.29.0, numpy 1.26.4, faster-whisper
1.2.1 / ctranslate2 4.8.2, setuptools 79). The resolution has been checked; the
install itself and the GPU stack have not (see "Unverified" below).

The `ml` extra carries the pins that matter, with the reasoning inline in
`pyproject.toml`:

- `setuptools<80`: `perth` (Chatterbox's watermarker) imports `pkg_resources`, which
  setuptools 81+ removed. Without the pin, every TTS call fails with
  `'NoneType' object is not callable`. The TTS backend now checks for this at import
  time and names the cause.
- `chatterbox-tts==0.1.7` with `transformers==5.2.0` and `diffusers==0.29.0` restated:
  Chatterbox breaks on other versions of either. Change them together, and only after a
  real synthesis test.

Optional extras: `pyannote` (more accurate diarization; needs a Hugging Face token)
and `separate` (Demucs vocal isolation, only for when you have no clean stem).

## 2. Download the Whisper model

Models are gitignored (several GB). Downloading Whisper by hand to a local directory
is the most reliable option:

```bash
mkdir -p ~/models/faster-whisper-large-v3 && cd ~/models/faster-whisper-large-v3
for f in model.bin config.json tokenizer.json preprocessor_config.json vocabulary.json; do
  curl -4 -L --max-time 3600 -O "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/$f"
done
cd -
```

Then point the pipeline at it, once, in `.env` in the repo root:

```bash
echo 'YTDUB_ASR_MODEL=/home/<you>/models/faster-whisper-large-v3' >> .env
```

(Or pass `--asr-model` per run. A bare size name such as `large-v3` also works and
downloads on first use.)

## 3. Pull the translation model

```bash
curl -s http://localhost:11434/api/tags   # Ollama is running?
ollama pull qwen3:8b
```

The pipeline sets the context window (`num_ctx`, default 8192) on every request itself
and verifies the server honours it, so no Modelfile or `OLLAMA_CONTEXT_LENGTH` changes
are needed.

## 4. Chatterbox

Chatterbox downloads its weights (~3 GB) to `~/.cache/huggingface/` on the first run.

## 5. Working folders and the `dub` command

```bash
mkdir -p input output
mkdir -p ~/.local/bin && ln -sf "$(pwd)/.venv/bin/dub" ~/.local/bin/dub
dub          # prints usage and the available styles
```

`dub` finds its home directory (`input/`, `output/`, `work/`, `synopses/`,
`glossary.txt`) from the editable install, so it works from any directory. Set
`YTDUB_HOME` to use a different location.

## IPv6

Not needed any more: the pipeline forces IPv4 for every connection in its own process
(yt-dlp, Hugging Face, Ollama) and puts timeouts on every network call, so a broken
IPv6 route surfaces as an error instead of a silent hang. `--allow-ipv6` opts out.
Tools run *outside* the pipeline (e.g. `curl` above) still need `-4`.

---

# Usage

Put a clean voice-only audio file (no music, no SFX, exported from your editor) in
`input/`. This is the intended workflow: it gives markedly better transcription and
voice cloning than a full mix, and no phantom words from background noise.

```bash
dub myfile.wav                              # all languages, casual style
dub myfile.wav --style duo --speakers 2     # conversation, two voices
dub myfile.wav --style professional de fr   # professional, German + French
dub https://youtu.be/VIDEO_ID de            # download first (saved to input/)
```

Outputs land in `output/<filename>/`: `<lang>.wav` and `<lang>.srt` for upload, plus
`<lang>.review.srt`, `report.json`, `dropped.txt` and `dub.log` (see the README).

At the end of each language the log prints the primary quality metric:

```
[de] FIT: compressed 2/143 segments (worst 1.048x, 0 above imperceptible, 0 over cap) | ...
```

and a summary table for the whole job.

## Review before the expensive part

Translation takes seconds; synthesis takes ~25-30 minutes per language for a 9-minute
video. So:

```bash
dub talk.wav --srt-only de pl          # translate only
# a native speaker edits output/talk/de.review.srt (text only; keep the cues)
dub talk.wav --from-review de pl       # dub from the reviewed text
```

Only edited lines are resynthesized if clips from an earlier run exist.

## Voices

Voices are counted by the diarizer by default, so a multi-speaker file needs no flags.
Speakers are labelled `SPK0`, `SPK1`, ... once per source file and the same labels and
voice references are used for every language. The chosen reference for each speaker is
logged with its time range.

Diarization defaults to `pyannote`, which needs a free Hugging Face token in `.env` as
`YTDUB_HF_TOKEN=hf_...` **and** the terms accepted at
`hf.co/pyannote/speaker-diarization-3.1`. Without a token the run says so and falls back
to the token-free embedding diarizer, which is measurably worse at separating two similar
voices. Force either per run with `YTDUB_DIARIZE_METHOD=pyannote|embedding`.

To use a known-good recording instead (recommended for a recurring presenter, for
consistency between uploads) — named by label, or attached by voice so the per-file label
never has to be known:

```bash
dub2 solo.wav --ref voices/atlas.wav
dub2 chat.wav --speakers 2 --ref SPK0=voices/atlas.wav
dub2 chat.wav --voice voices/atlas.wav        # matched to whoever sounds like you
```

`--speakers 1` means one speaker: no diarization, and `--voice` then applies to the whole
file. `--speakers 0` (the default) counts the voices.

## Long jobs

```bash
dub myfile.wav --detach       # runs in its own session; safe to close SSH
tail -f output/myfile/dub-*.log
```

(`tmux` or `nohup dub ... &` work too.) If a run dies for any reason, run the same
command again: completed stages, translations and synthesized clips are reused.
`--force` recomputes everything.

## Editable config (no code changes)

- `glossary.txt`: terms never to translate (channel names, games, brands, handles).
- `hints.txt`, `hints.<lang>.txt`: how to translate specific terms (`patty = kotlet`),
  for every language or for one. See "Translation quality" below.
- `synopses/*.txt`: style presets. Each has a description of the video and its tone,
  and a "Locale rules" section (numbers, currency, dates, units). A new `.txt` file
  appears as a `--style` option automatically.
- `.env` / `YTDUB_*` environment variables: every setting in `src/ytdub/config.py`,
  e.g. `YTDUB_MAX_RATIO=1.15`, `YTDUB_OLLAMA_MODEL=qwen3:14b`,
  `YTDUB_LANGUAGES='["de","fr"]'`, `YTDUB_MIN_CONFIDENCE=0.5`, `YTDUB_VERIFY=1`,
  `YTDUB_EXPAND_NUMBERS=0`.

## Spoken numbers

Digits are written out as words for the synthesizer, deterministically, with no model
call. The translation prompt asks for this and both models ignore it, and the failure is
only audible at the end of the pipeline: multilingual TTS reads a digit in the wrong
grammatical case, mistakes it for a date, or skips it — and the review SRT shows nothing.

```
SRT (you and the viewer):   Daję 4,5, bo dziś czuje się szorstko
audio (the synthesizer):    Daję cztery przecinek pięć, bo dziś czuje się szorstko
```

It runs at the TTS boundary — after translation, after verification, after budget
shortening — so it never touches the translated text, the review file or the subtitle
track. Budgets are decided before it: expansion only ever makes a line longer, and the
fitter already exists to absorb a longer line into the silence around it, so the extra
characters are its job rather than something to plan the budgets around. Every affected
line is logged (debug for the exact text, info for the count and line numbers), so a wrong
expansion is diagnosable from `dub.log` without listening to the whole dub.

Left as digits on purpose: clock times (`7:30`), versions and IPs (`4.5.1`), dates
(`10.06.2024`), ranges (`3-4`), percentages (`100%`), units and model numbers (`12GB`,
`1080p`, `RTX 3080`), and any digit inside a glossary or hints term. Turn the whole thing
off with `--no-number-expansion` or `YTDUB_EXPAND_NUMBERS=0`; the SRT is identical either
way, and only the lines whose numbers changed are resynthesized.

**Language coverage.** `num2words` has a converter for pl, de, fr, es and nl. It does
**not** have one for Hindi (every call raises `NotImplementedError` on 0.5.14), so Hindi
keeps its digits. That is logged rather than guessed at: a wrong number read confidently
is worse than a digit the TTS might handle.

Polish needs more than `num2words` can give, because its cardinals inflect: after `z`,
`około`, `od`, `do` and similar prepositions the number must be genitive, and
`num2words` only produces the nominative. "10 brown leaves" as a rating is `z dziesięciu`,
not `z dziesięć`. A small table covers 0-999, which is the range this content uses;
anything larger is left as a digit rather than said wrongly. If you speak Polish, that
table is the thing to check — it is in `stages/numbers.py` next to the prepositions that
trigger it. German, French, Spanish and Dutch need nothing: their cardinals are the same
form in every position.

## Translation quality

A translation can be fluent, idiomatic and wrong. `hints.txt` pins the recurring
vocabulary; `--verify` catches the rest.

`hints.txt` holds `term = translation` pairs for domain words that come up across videos:
`patty`, `taste buds`, `craftsmanship`. They are injected into the translation prompt, so
they beat the model's own guess on the day. A per-language override goes in
`hints.<lang>.txt` (`hints.pl.txt`, `hints.de.txt`), which wins over the shared file for
the same term. Both are optional: `#` starts a comment line, a line without an `=` is
skipped rather than guessed at, and a missing or empty file is not an error.

`--verify` (or `YTDUB_VERIFY=1`) adds *back-translation verification*. After translating,
each line is rendered back into the source language **literally**, that round trip is
compared with the original line, and lines whose meaning changed are translated again with
the drift named to the model:

```
source:          on the actual patty
current:         na kiełbasiu
back-translation: on the sausage
problem:         patty became sausage
```

The revision is then verified the same way and kept only if its own round trip comes back
clean; otherwise the first attempt stands, so the pass can improve a line but never
silently degrade one. It costs roughly three times the translation time — small next to
synthesis, which is why `--srt-only` makes it cheap to try:

```bash
dub cheese.wav --srt-only pl --verify     # writes output/cheese/pl.verify.txt
```

**Use it as a diagnostic.** Expect it to flag more lines than it can fix — on real
content the repairs are low-yield — but the flags are the point: the report ends with the
terms it caught, next to the one line it managed to verify as the fix.

```
# Suggested hints from this run.
# Each entry is a term this pass flagged, and the line it verified as the
# fix. Copy the term and the words of that line that render it into
# hints.txt ...
#   craftsmanship = <- from: Rzemiosło tego burgera
#   taste buds = <- from: kubki smakowe są w porządku
```

That block stops one step short of paste-ready on purpose. What the pass knows for
certain is the term (flagged on meaning, checked against the source line) and the line
that fixed it (verified by its own round trip). Which words of that line render the term
is the judgement call the comparison model itself only manages about half the time, so
guessing it here would put confident wrong pairs in front of someone about to make them
permanent. On a run that flags lines but keeps no revision, the block says so and lists
the terms to check by hand instead.

Lines the pass could not check at all are listed at the top of the report. That happens
when the model answers the back-translation request by copying its input instead of
translating it; the pass retries those lines once, rephrased and at a higher temperature
(`verify_temperature`, one retry per round, `verify_echo_retries=0` to disable), and
reports whatever is still unanswered rather than pretending silence is agreement.

### A bigger model for the check, not for the translation

Verification is a check, not a translation, and the two never need to be resident at the
same time: translation finishes and unloads before the pass runs. So the pass can use a
model the translator cannot afford:

```bash
dub video.wav --srt-only pl --verify --verify-model qwen27-24k:latest
# or YTDUB_VERIFY_OLLAMA_MODEL=... , which needs no --verify-model
```

Translation stays on `ollama_model` (`qwen3:8b` by default). With no verification model
set — the default — the translator verifies with its own model, exactly as before.
Whichever model is used is part of the translation cache key, so changing it retranslates.
If the named model is not pulled, the job fails before translating rather than quietly
verifying with the wrong one.

Verification on/off, its prompts, its batch size and its model are part of the translation
cache key, as are both hint files, so toggling it or editing a hint retranslates rather
than serving a result from before the change.

## What invalidates what

Each stage's cache key is a hash of the inputs listed here. Any change is a miss, and
the log names the inputs that changed. Tests cover each translation input.

- **Transcription:** source file sha256, Demucs model (with `--separate`), `asr_model`,
  the file sizes/mtimes of a local model directory, the faster-whisper version, compute
  type, beam size, VAD on/off and min-silence, `min_confidence`, `source_lang`.
- **Diarization:** the transcription key, `speakers`, `diarize_method`.
- **Voice references:** the diarization key, sha256 of each `--ref` file, reference
  target/min seconds.
- **Translation (per language):** the transcription and diarization keys (source text,
  timings, speaker tags), source and target language, sha256 of the style file, of
  `glossary.txt` and of that language's hints (shared file + `hints.<lang>.txt`),
  translator backend name, and the backend's own identity. For Ollama that is the model
  name, `num_ctx`, temperature, batch/context/lookahead line counts, budget tolerance and
  retries, `verify`, `verify_batch_lines`, the verification model, `verify_echo_retries`,
  `PROMPT_VERSION`, and a hash of the translator's source code (prompts, verification and
  orchestration). It also includes the per-line character budgets and the settings they
  come from: `chars_per_second`, `budget_max_borrow`, `min_gap`. The Ollama *weights
  digest* is stored with the result: a re-pulled model with the same name is
  retranslated. If Ollama is unreachable the digest can't be checked and the cached
  translation is used (logged).
- **Synthesized clips (per line):** the exact text, language, sha256 of the speaker's
  reference clip, TTS backend name and identity (for Chatterbox: package version,
  exaggeration, cfg weight, temperature), seed.
- **Fitting, loudness, WAV/SRT/MP4:** not cached. They are cheap and rerun every time,
  so changing `--max-ratio` and similar never needs `--force`.

Settings that don't affect a stage (e.g. `max_ratio`, `ollama_url`) are deliberately
not in its key.

## Swapping a model backend

Only Chatterbox (TTS) and Ollama (translation) ship. Both sit behind small interfaces,
`stages/tts/base.py:TTSBackend` and `stages/translate/base.py:Translator`, and any
class with that shape can be dropped in without touching the pipeline:

```bash
dub talk.wav de --tts mypkg.tts:OtherTTS          # or YTDUB_TTS_BACKEND=...
dub talk.wav de --translator mypkg.mt:OtherMT     # or YTDUB_TRANSLATOR=...
```

A TTS backend is built as `cls(settings)` and needs `name`, a class-level
`supported_languages`, `cache_identity()`, `synthesize(text, ref, language, out_path,
seed)` and `unload()`. A translator is built as `cls(settings, style_text, glossary)` and
needs `name`, `cache_identity()`, `weights_fingerprint()`, `translate(lines,
source_lang=..., target_lang=...)` and `unload()`. `set_hints(hints)` is optional: a
translator without it still works, it just logs that the hints were ignored (the hints are
in the cache key either way, so that cannot cause a stale hit). The tests load their fake
TTS and a fake translator exactly this way. Check the licence of anything you drop in:
XTTS and NLLB are non-commercial.

## Loudness

The dub is loudness-matched to the input file. When the input is a voice stem, the
YouTube original is the full mix, which is louder; match that instead:

```bash
dub stem.wav --loudness-ref mix.wav
```

---

# Failure behaviour

- **Duration:** after writing each `<lang>.wav` the pipeline checks the sample count
  it wrote, then re-measures the file with ffprobe against the probed source. More than
  2 ms off fails that language (logged with traceback, `failed` in the summary, exit
  code 1). This is a runtime check on every real job, not a test-only assert.
- **Compression cap:** if the timeline can't fit within `--max-ratio`, the language
  still completes but logs a WARNING naming the lines and the worst ratio, and repeats it
  in the end-of-job summary. The same applies to lines pushed later than `--max-delay`.
  Both appear at the default log level and in `report.json`.
- **Lost lines** (TTS failed on a line): the language is marked `degraded`, the line
  numbers are logged as ERROR, and the exit code is 1.

# Unverified on real hardware

The unit and fake-model tests pass on Python 3.11. Nothing here has run on a GPU,
with real models, or against a real Ollama. Specifically unverified: the `uv sync`
install itself; ctranslate2 4.8.2 loading alongside torch 2.6's cuDNN 9.1; Chatterbox
and Whisper output; Ollama honouring `think: false` with a JSON schema; the
`prompt_eval_count` checks against a real server; `--detach`.

# Known limitations

- Chatterbox crashes on utterances of only a couple of characters. They are merged
  into an adjacent line from the same speaker (within 1.5 s). A short line with no
  such neighbour (e.g. a one-word reaction between two other speakers) is synthesized
  alone; if that fails it is reported as a lost line in the log and `report.json`.
- Low-confidence segments are dropped as probable noise. Check `dropped.txt`; if real
  speech was dropped, lower `--min-confidence`.
- LLMs treat character budgets loosely. Over-budget lines are re-requested, and the
  fitter absorbs the rest into silence, compressing only when it must.
- `--verify` needs the model twice per line and is only as good as the model's judgement:
  it can miss a drift it does not recognise, and it will not overwrite a line it cannot
  prove is better, so a stubborn line stays as first translated and is named in
  `<lang>.verify.txt`. It also runs *after* translation and *before* budget shortening,
  so a kept revision can still be shortened afterwards.

# Local patches from the fork, and where they went

1. **`pipeline.py`, audio-only inputs skip the mux.** Redundant as a patch: the
   rebuilt pipeline probes for a video stream and only muxes when one exists, and
   writes the WAV and SRT *before* any muxing, so a mux failure can't lose them.
2. **`stages/tts/base.py`, merge fragments under 3 chars.** Kept and tightened: the
   merge now only joins an *adjacent* line from the *same* speaker within 1.5 s (the
   old version could carry a fragment to a later line of that speaker, past other
   people's lines), counts letters rather than characters, and the threshold is
   configurable (`YTDUB_MIN_TTS_CHARS`).
3. **`stages/tts/base.py`, full tracebacks on TTS failure.** Kept and generalised:
   every stage and language failure logs a full traceback, and the Chatterbox import
   explicitly detects the silent perth / `pkg_resources` failure.
4. **`stages/download.py`, IPv4 + Node JS runtime in both option dicts.** Kept, and
   the two-dict problem is designed out: one options builder, one yt-dlp call (audio is
   extracted with ffmpeg afterwards). IPv4 is now also forced process-wide, which covers
   Hugging Face downloads, and every network call has a timeout.
5. **`stages/translate/ollama.py`, LLM backend with glossary, synopsis and rolling
   context.** Superseded by the batch translator, which keeps the glossary and synopsis
   and replaces rolling context with full batch context.
6. **`stages/translate/base.py`, `--translator ollama`.** Redundant: Ollama is the
   only translator. NLLB (non-commercial licence) and Argos were removed.

Also replaced: `duball.sh` (now the `dub` entry point, which runs the shared front
half once instead of once per language), the manual `setuptools<80` install (now in
the `ml` extra) and the IPv6 sysctl step (see above).
