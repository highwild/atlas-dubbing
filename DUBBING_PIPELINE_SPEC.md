# Local Voice-Cloning Dubbing Pipeline — Build Spec

## What this is

A rebuild of an existing tool (`mazzasaverio/youtube-auto-dub`, referred to below as
"the reference implementation") that dubs a video or audio file into other languages
using the original speaker's cloned voice, running entirely locally on one machine.

The reference implementation works, and its overall stage design is sound. This spec
keeps that design and fixes three architectural problems plus a set of known bugs. Read
the reference implementation before starting — most of the stage logic can be reused
or adapted rather than written from scratch.

---

## Hard requirements

These are non-negotiable. Everything else is open to judgement.

1. **Output duration must exactly match input duration.** The dubbed audio is uploaded
   to YouTube as an alternate audio track over the original video. If a 2:34 source
   produces 2:33 or 2:35 of dubbed audio, it desyncs from the picture and is unusable.
   This is the single most important constraint in the system.

2. **Runs fully locally.** No cloud APIs, no per-use cost, no data leaving the machine.

3. **Single command per job.** `dub myfile.wav --style casual --speakers 2` and it
   produces everything. No multi-step manual process.

4. **Commercially usable licences only.** Every model and library must permit
   commercial use. The reference implementation defaults to NLLB-200, which is
   CC-BY-NC (non-commercial) — that is disqualifying. Qwen 3 (Apache 2.0) is a known
   acceptable alternative for translation.

---

## Target environment

- Linux (Ubuntu), single machine
- NVIDIA GPU, **16GB VRAM** — this is a real ceiling, see VRAM budget below
- Ollama already installed and running locally, serving `qwen3:8b`
- Python 3.11, `uv` for dependency management
- ffmpeg 8.x available on PATH

### VRAM budget

A run must fit in 16GB with all models resident. Measured from the reference
implementation:

| Component | Approx VRAM |
|---|---|
| Whisper large-v3 (faster-whisper, float16) | ~3GB |
| Chatterbox multilingual TTS | ~3GB |
| Qwen3 8B via Ollama (Q4) | ~5GB |
| Overhead / fragmentation | ~3GB |
| **Total** | **~14.6GB of 16GB** |

This leaves almost no headroom. Qwen3 14B (~9GB) does **not** fit alongside the
others. If a larger model is wanted, the pipeline must unload models between stages
rather than holding them all resident. Design for stage-by-stage loading and explicit
unloading; do not assume everything can co-exist.

---

## Pipeline stages

```
input (video or audio)
  ↓
── run ONCE per source file, cached ──────────────
1. acquire      — local file, or download via yt-dlp
2. transcribe   — Whisper large-v3 → timestamped segments
3. diarize      — optional: identify distinct speakers
──────────────────────────────────────────────────
  ↓
── then per target language ──────────────────────
4. translate    — batch/document-level LLM translation  ← REDESIGNED
5. synthesize   — Chatterbox TTS, voice cloned per speaker
6. fit          — place clips on the timeline           ← REDESIGNED
7. assemble     — mux onto video, or emit audio + SRT
──────────────────────────────────────────────────
```

Stages 1-3 depend only on the source file, not the target language, so they run once
and are shared across every language in the job. The reference implementation reruns
them per language, which is pure waste — see Redesign 3.

Stages 1, 2, 3, 5 and 7 can otherwise follow the reference implementation closely.
Stages 4 and 6 are where the redesign lives.

---

## Redesign 1 — Document-level translation

### The problem

The reference implementation translates **one segment at a time**, each in an isolated
call. Native-speaker testers reported the output was grammatically correct but
obviously machine-produced: English phrasing rendered word-for-word into the target
language, idioms translated literally, phrases "you wouldn't actually say."

Root causes:
- The model never sees surrounding dialogue, so it cannot tell what a line refers to
  or what register the piece is in
- It cannot move content between lines — a long line must be compressed or fail,
  even when the next line is short
- Terminology drifts, because each call is independent

### The design

Translate in **batches with full context**, not per-sentence.

**Input to the model:** a numbered block of source lines, each tagged with a character
budget, preceded by a synopsis of the video and a glossary of protected terms.

The **character budget** is how many characters of translated text will fit that
segment's time slot at a natural speaking pace — roughly `slot_duration ×
chars_per_second`, where chars_per_second is a tunable around 15. A 3-second slot gives
roughly 45 characters. It is the pipeline's way of telling the model how much room a
line actually has. Under the redesigned timeline fitting below, the slot includes any
adjacent slack the segment can borrow, not just its original source duration.

```
Video context: <synopsis — who is speaking, what about, what tone>

Do not translate these terms: <glossary list>

Translate lines 1-30 into Polish. Return the same numbering.
Each line has a character budget. You may move words between adjacent lines
to help them fit, as long as the meaning and order are preserved.

1. [≤82] So unless you've been living under a rock, you'll know...
2. [≤41] I am now unemployed.
3. [≤95] It was a mad past, I've worked there five years...
```

**Expected output:** the same numbering, translated, respecting budgets.

**Batch size:** around 20-40 lines. Long videos (150+ segments) will not fit in one
prompt. Overlap batches by a few lines so context carries across boundaries.

### Context window — a silent failure mode

**Ollama defaults to a 4096-token context window (2048 in some builds), far below what
the model actually supports, and silently truncates anything longer.** No error, no
warning, nothing in the response indicates it happened.

Truncation drops the *oldest* content first — which is where the system prompt,
glossary and synopsis sit. The symptom is the model appearing to ignore its
instructions: protected terms getting translated, tone guidance not applied. It looks
like a bad prompt, not a config problem, which makes it expensive to diagnose.

Requirements:
- Set `num_ctx` explicitly on every request. Do not rely on the server default, the
  `OLLAMA_CONTEXT_LENGTH` env var, or a Modelfile parameter, since any of these can be
  overridden elsewhere in the stack.
- **Estimate tokens before sending** and assert the prompt fits the configured window.
  Fail loudly if it does not, rather than letting the server truncate.
- Verify the setting actually took effect. `prompt_eval_count` in the API response
  shows how many tokens the model really read; if it is lower than the prompt sent,
  truncation happened.
- Size batches against the real window, not an assumed one. Remember the budget covers
  system prompt + glossary + synopsis + source lines + **the generated translation**,
  which roughly doubles the requirement.

Raising `num_ctx` increases KV cache memory use, which competes with the VRAM budget
above. Size it deliberately rather than setting it to the model maximum.

**Robustness:** LLMs return malformed output sometimes. Parse strictly, and on failure
fall back to translating that batch line-by-line rather than dropping it. Never let a
parse failure lose content — the reference implementation's habit of falling back to
untranslated source text is the right instinct.

**Important caveat on character budgets.** An LLM treats a stated budget as a soft
suggestion and frequently ignores it. This was measured: tightening the budget in the
reference implementation's per-line prompt produced *no* reduction in time-compression,
and marginally more. Do not assume the returned line respects the budget.

Instead, verify and iterate:
- Measure the returned line length against its budget
- For lines that materially overshoot, re-request just those with explicit instruction
  to shorten, and/or offer the model the option to move the overflow into a
  neighbouring line that has slack
- Cap the number of retries and accept the best result rather than looping forever

Batch translation makes this tractable in a way per-line translation was not, because
the model can see where the slack is. But it still has to be checked rather than
trusted.

### Prompt inputs

Three user-editable files, read at runtime, no code changes needed to edit:

- **Glossary** — flat list of terms never to translate: channel names, game titles,
  brands, in-game currencies, people's handles. Real failures seen without one:
  "Rec Room" became "recreation room", "Atlas" was translated as the noun.
- **Hints** (`hints.txt`, `hints.<lang>.txt`) — `term = translation` pairs for domain
  vocabulary that recurs across videos and that the model gets wrong often enough to be
  worth pinning ("patty", "taste buds", "craftsmanship"). The glossary says *never
  translate this*; hints say *translate it exactly this way*. Per-language, because the
  right rendering differs by language.
- **Synopsis / style presets** — a short description of what the video is and how it
  should sound. Multiple presets, selected per run (e.g. casual solo, multi-person
  conversation, more formal/informative). This measurably improves the opening line,
  which otherwise has no context at all to work from.

### Back-translation verification (`--verify`)

Fluent target-language text that means something else is the failure class a review of
the target text cannot catch: the error only exists *against the source*. With `--verify`
every line is rendered back into the source language literally, that round trip is
compared with the original line, and lines whose meaning changed are translated again
with the drift named (source, current attempt, what that attempt actually says, and the
specific change). The revision is verified the same way and kept only if its own round
trip comes back clean, so the pass can improve a line but can never silently degrade one.
One revision round, capped. Off by default: it roughly triples translation requests.

Two things make it work rather than produce noise. The comparison prompt carries worked
examples of the distinction it has to draw — different word, same thing (ok) versus a
different thing (drift) — because without them a small model reports synonyms and
paraphrase as changes; measured on qwen3:8b over real burger-review lines that took the
verdict accuracy from 5/8 to 12/12. And a back-translation that comes back unchanged, or
still in the target language, is treated as *no evidence* rather than as a match: a copied
line would otherwise "confirm" a wrong translation as correct. Those lines are retried
once with different instructions, and anything still unanswered is reported.

Its real output is a **diagnostic**, not a repair: the revisions are low-yield, but every
flagged line is written to `<lang>.verify.txt` with the source, the attempt, what that
attempt means and the specific change — which is a list of terms worth pinning in
`hints.txt`. The pass also suggests the terms it verified a fix for, with the line that
fixed them, since a term plus a rendering it has actually confirmed is the one thing it
knows better than the reader does.

Because verification is a check rather than a translation, and never has to be resident
with the translator, it can run on a larger model than the translation does
(`verify_ollama_model`).

### Why this fixes the literalness

The model can see that line 2 is tight and line 3 has slack, so it can move a clause
between them. Per-sentence it can only compress or fail. It also gets consistent
terminology and working pronoun reference for free.

---

## Redesign 2 — Timeline fitting

### The problem

**This is the biggest quality issue in the reference implementation.**

It fits each segment into its *own original time slot*. Any translated line longer
than its English source gets time-compressed to fit. In practice 60-85% of segments
end up stretched, and heavy pitch-preserving compression produces a warbly, stuttery
artifact that native listeners described as the audio "freezing" and sounding like "a
fever dream."

Target languages are frequently longer than English — Polish and German especially —
so this is the normal case, not an edge case.

### The insight

**Total duration is a hard constraint. Per-segment duration is not.**

Real speech contains substantial slack: pauses between sentences, breaths, silence
while something is shown on screen. If a translated line needs an extra 0.4s and the
gap after it is 1.2s of silence, the line can simply borrow from the gap. Total
timeline length is unchanged and no compression is needed.

The reference implementation ignores this entirely and treats every segment boundary
as rigid.

### The design

Treat fitting as a **global optimisation over the whole timeline**, subject to total
duration being exactly preserved.

Roughly, in order of preference:

1. **Use the natural length** of the synthesized clip where it fits
2. **Absorb overflow into adjacent silence** — shift the following segment's start
   later, consuming gap rather than compressing audio
3. **Redistribute slack globally** — a segment can borrow from gaps further along the
   timeline, not just the one immediately after, provided nothing overlaps
4. **Compress, as a last resort**, and only by the minimum needed, prioritising
   compression of segments where it is least perceptible
5. **Pad with silence** if the total comes in short, so the output length is exact

Constraints to enforce:
- Total output duration == total input duration, exactly
- Segments must not overlap
- A segment must not start before its source segment started (drifting earlier is more
  perceptible than drifting later)
- Cap maximum compression at a configurable ratio; the reference implementation
  defaults to 1.4x, which is audibly too aggressive

Report per-run how many segments needed compression and the worst ratio applied. This
is the primary quality metric for the whole system and should be visible in the logs
without needing to grep for it.

### Success criterion

Substantially fewer compressed segments than the reference implementation on the same
input, with identical total duration. On a 13-segment test clip the reference
implementation compressed 8-10 of them; a good implementation should be in the low
single digits.

---

## Redesign 3 — Cache and resume

The reference implementation re-runs **every stage for every language**. On a
six-language job it transcribes the same audio six times, diarizes it six times, and
loads Whisper six times, despite those steps being identical regardless of target
language. This is visible in its logs and is pure waste.

Worse, there is no resume. A six-language run takes hours; if language five fails, or
the machine reboots, or SSH drops, everything is lost.

Requirements:

- **Run language-independent stages once.** Acquire, transcribe and diarize depend only
  on the source file. Do them a single time, cache the result keyed on a hash of the
  input file plus the relevant settings, and reuse across all languages.
- **Cache per stage, per language, on disk.** Translation output and synthesized clips
  should survive a crash. Re-running a job should skip anything already complete.
- **Make invalidation correct and obvious.** Changing the glossary, synopsis, style or
  model must invalidate translation but not transcription. Hash the inputs that
  actually affect each stage. Getting this wrong in the safe direction (recomputing
  unnecessarily) is far better than serving stale output.
- **Provide an explicit `--force` / `--no-cache`** to bypass it.

This changes a six-language job from six full pipelines into one shared front half plus
six cheap tails, and makes long runs interruptible. It is worth more in practice than
either of the quality redesigns.

---

## Human review loop

Translation quality cannot be verified automatically and needs a native speaker. The
expensive stage (TTS) runs *after* the cheap one (translation), so the workflow should
let a person intervene between them:

- **`--srt-only` mode**: run through translation, write the SRT, stop. Seconds rather
  than half an hour.
- **Accept a corrected SRT as input**: having reviewed or fixed the text, re-run from
  synthesis using the edited file, skipping transcription and translation entirely.

Without this, correcting a single mistranslated line means regenerating the entire dub.
With it, review becomes cheap and the tool becomes usable for content that actually
matters.

---

## YouTube delivery

The output is uploaded to YouTube as a multi-language audio track alongside the
original video. A few things follow from that and should be handled by the tool rather
than left to the user.

### Loudness

YouTube normalises audio, and viewers can switch between the original and dubbed track
mid-video. If the dub is noticeably quieter or louder than the original, that switch is
jarring.

Measure the source loudness and normalise the dubbed output to match it, rather than to
a fixed target. ffmpeg's `loudnorm` filter does this. Report the measured values.

### Subtitles

The SRT produced for each language is directly uploadable as a subtitle track. Generate
it **after** the fitting stage, not before — timings shift when segments borrow slack,
and a subtitle file that disagrees with the audio is worse than none.

---

## Locale handling in translation

Spoken numbers, dates, currencies and units are a common dubbing failure and worth
explicit prompt instruction:

- **Currency:** decide whether to convert or keep the original, and be consistent
- **Dates:** British "the fifth of June" ordering differs from other conventions
- **Units:** miles, feet and stone mean little in most target markets; converting is a
  judgement call but should be a deliberate, documented one
- **Numbers read aloud:** TTS pronounces "2026" differently across languages. Spelling
  numbers out in the translated text is often safer than leaving digits

Put these rules in the translation prompt and make them part of the style presets, so
they can differ per video type.

---

## Voice reference selection

The quality of the cloned voice depends heavily on which snippet of source audio is
used as the reference clip. A short sample, or one containing background noise, music
or two people talking over each other, degrades every line that speaker produces.

Do not take the first available segment. Select deliberately:

- Prefer longer continuous speech from that speaker
- Prefer high-confidence, clearly voiced regions
- Avoid segments that overlap other speakers
- Log which segment was chosen and its duration, so a bad clone can be diagnosed rather
  than guessed at

Allow an explicit reference clip to be supplied per speaker. For a recurring presenter,
a known-good reference recording beats anything auto-selected, and reusing the same one
across videos gives consistency between uploads — which matters when a viewer watches
more than one.

---

## Known bugs to design out

All of these were hit in production use of the reference implementation. They are
listed with cause and fix so the rebuild does not reintroduce them.

### 1. Short segments crash the TTS

Chatterbox's alignment analyzer raises `IndexError: max(): Expected reduction dim 1 to
have non-zero size` on very short utterances (seen on a 2-character Polish word, "to")
because its internal reduction window ends up empty.

Padding the text with punctuation does **not** fix it. Working fix: merge fragments
below a threshold (~3 characters) into an adjacent segment from the same speaker
before synthesis. Preserves the words, introduces no stutter, and as a side effect
reduces time-compression slightly.

Make the threshold configurable.

### 2. Audio-only inputs crash at the mux stage

The reference implementation assumes a video stream exists and calls ffmpeg with
`-map 0:v:0`, which fails with `Stream map '' matches no streams` when the input is a
`.wav` or `.mp3`. Worse, this crash happens *before* the SRT is written, so a run that
did all its expensive work still produces nothing.

Audio-only input is a **primary use case**, not an edge case — feeding a clean
voice-only stem (no music, no SFX) produces markedly better transcription and voice
cloning than a full mix. Design for it:

- Probe for a video stream; branch cleanly on the result
- Audio in → audio + SRT out, no muxing attempted
- Write the SRT **before** any muxing, so a mux failure never costs the transcript

### 3. Background noise transcribed as speech

Whisper transcribes whatever it hears. On a full mix with music, sound effects or room
noise, it produces phantom words that are then translated and spoken aloud in the dub —
audible as random nonsense lines that were never said.

Mitigations, in order of effectiveness:

- **Prefer a clean voice-only stem as input.** This is the intended workflow and the
  main reason audio-only input matters. Document it prominently.
- **Enable voice activity detection (VAD)** on the transcription stage to drop
  non-speech regions before they become segments. faster-whisper supports this
  directly; the reference implementation does not enable it.
- **Filter low-confidence segments.** Whisper returns per-segment confidence; segments
  well below threshold are usually hallucinated noise. Make the threshold configurable
  and log what was dropped so it can be reviewed.
- Optionally offer built-in source separation (e.g. Demucs) for when only a full mix is
  available, but treat it as a convenience — a stem exported from the editor will
  always be cleaner.

### 4. Large downloads hang silently on IPv6

On a machine with a broken IPv6 route, small HTTP requests succeed but large transfers
hang forever with no error and no timeout — this affected both yt-dlp and HuggingFace
model downloads, and presented as the tool being frozen with no diagnostic.

Force IPv4 on all outbound HTTP by default, with an opt-out. Set explicit timeouts on
every network call so a stall surfaces as an error rather than a hang.

### 5. yt-dlp needs a JS runtime and does not inherit CLI config

Two related traps:
- YouTube requires solving a JS challenge; yt-dlp needs a JS runtime (Node) explicitly
  enabled plus remote components fetched, or it hangs at "Downloading webpage"
- When yt-dlp is used as a **library** rather than a CLI, it ignores
  `~/.config/yt-dlp/config` entirely, so any user config there silently does nothing

Set the equivalent options programmatically in every `YoutubeDL` options dict, and
apply them consistently — the reference implementation has two separate option dicts
and fixing only one produces a confusing partial failure.

### 6. Dependency version traps

- `setuptools` 81+ removed `pkg_resources`, which `perth` (Chatterbox's watermarker)
  imports. Perth catches the ImportError silently and sets its main class to `None`,
  producing a downstream `'NoneType' object is not callable` on every TTS segment with
  no indication of the real cause. Pin `setuptools<80` or vendor around it.
- Chatterbox is sensitive to `transformers` and `diffusers` versions. Pin explicitly
  and document why.

Prefer current versions of everything else, but pin what actually matters and record
the reasoning inline.

### 7. Errors that hide their cause

The single most time-consuming problem in debugging the reference implementation was
`log.error(f"...: {exc}")` printing only an exception message with no traceback,
combined with a silent `try/except ImportError` upstream. The visible symptom pointed
nowhere near the actual cause.

Log full tracebacks on stage failures by default. Never swallow an ImportError without
recording what failed and why.

---

## Model choices

Sensible defaults, all commercially licensed, all configurable:

| Stage | Default | Notes |
|---|---|---|
| Transcription | faster-whisper `large-v3` | Same weights as OpenAI Whisper, faster inference. Support passing a local model directory, not just a size name — auto-download is a common failure point. |
| Translation | Qwen 3 8B via Ollama | Apache 2.0. Disable reasoning/thinking mode, and strip `<think>` blocks defensively. Low temperature (~0.3). |
| Voice cloning | Chatterbox multilingual | Best local option found. Supports the target languages. |
| Diarization | pyannote, falling back to embedding clustering | pyannote is the default because it separates similar voices and the embedding clusterer does not (measured: it merged two different men at 0.76 similarity against a 0.75 threshold). It needs a HuggingFace token and terms acceptance, so with no token the run warns once and falls back to the token-free clusterer — offered and default, but still not required. |

Every model choice should be swappable via config. Assume better options will exist in
six months.

---

## CLI

Single command, sensible defaults, flags in any order:

```bash
dub myfile.wav                                  # all default languages
dub myfile.wav --style duo --speakers 2         # conversation, two voices
dub myfile.wav --style professional de fr       # subset of languages
dub                                             # usage + available styles
```

Behaviour:
- Input files read from a fixed `input/` directory
- Outputs written to `output/<basename>/<lang>.wav` and `<lang>.srt`
- Languages processed sequentially (they compete for VRAM otherwise)
- One language failing must not abort the rest; report failures in a summary at the end
- Style presets discovered from a directory of text files, listed automatically in help
- Long jobs must survive an SSH disconnect, or document how to run detached

**Default language set:** German, French, Polish, Spanish, Dutch, Hindi
(`de fr pl es nl hi`). Configurable, but these are the working defaults. Note that
German and Polish in particular run longer than English, which is what makes the
timeline fitting problem acute.

**Output format:** Chatterbox generates at 24kHz mono. Resample to a standard rate for
upload (48kHz) rather than shipping 24kHz, and keep it uncompressed — the file is
being uploaded, not streamed, so there is no reason to introduce a lossy generation.

### Speaker consistency across languages

In multi-speaker mode, each language is a separate run, and diarization assigns speaker
labels independently each time. There is no guarantee that "speaker 0" in the German
run is the same person as "speaker 0" in the Polish run — which would put one person's
cloned voice on another person's lines, differently in each language.

Fix this by diarizing **once** and reusing the speaker-to-reference-clip mapping across
every language in the job, rather than recomputing per language. This also saves time.

Expect runs to take hours. A 9-minute video takes roughly 25-30 minutes per language.
Progress reporting should make it clear which stage is running and how far through it
is — the reference implementation's TTS progress bars are widely misread as being
stuck, because they show token generation against a 1000-token ceiling that a normal
sentence never approaches.

---

## Testing

A working test needs to prove three things:

1. **Duration is exact.** Assert output duration equals input duration to within a few
   milliseconds, on every language. This is the hard requirement and should fail loudly.
2. **No content is lost.** Segment count in equals segment count out (after any
   deliberate merging), no silently dropped lines.
3. **Compression is minimal.** Track how many segments needed time-compression and the
   worst ratio. Regressions here are the main quality risk.

Include a short multi-speaker test clip with at least one very short utterance, since
that is the case that broke the reference implementation.

Quality of translation itself needs a native speaker; it cannot be unit tested. Make
the SRT easy to inspect before committing to a full TTS run — reviewing text takes
seconds, listening to a generated dub takes half an hour.

---

## Explicitly out of scope

- Lip-sync / mouth re-rendering. The output is an alternate audio track over unchanged
  video; nothing needs to match mouth movement.
- Any cloud service or paid API.
- A GUI.
- Real-time or streaming operation.

---

## Notes on approach

The reference implementation is a good starting point and its stage boundaries are
sensible. This is a targeted rebuild, not a rejection — reuse what works.

The three redesigns are where essentially all the value lives. Document-level
translation and global timeline fitting fix the quality problems; cache-and-resume
fixes the practical one, and is the change most likely to be underestimated — a
six-language job currently redoes identical work six times and cannot survive an
interruption.

The bug list is table stakes; getting those right just means not losing hours to
problems that are already solved and documented above.

If a trade-off has to be made, prioritise in this order: exact duration, then no lost
content, then minimal compression, then translation naturalness, then speed.

**Suggested build order.** Build the timeline fitting stage first and test it in
isolation with synthetic clips of known length — it is where the quality lives, it is
the easiest thing to verify without a native speaker, and it is the part most likely to
be glossed over if everything is built at once. Get exact-duration output with minimal
compression proven before wiring in the real models.
