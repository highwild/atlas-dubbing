# Setup

Everything here assumes Ubuntu with an NVIDIA GPU and Ollama already installed.

## 1. Clone and create the venv

```bash
git clone <your-repo-url> atlas-dubbing
cd atlas-dubbing

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[chatterbox,nllb,diarize]"
uv pip install "setuptools<80"
```

The `setuptools<80` pin is required. Version 81+ removed `pkg_resources`, which
`perth` (Chatterbox's watermarker) imports. Without it, every TTS segment fails with
`'NoneType' object is not callable` and nothing tells you why.

## 2. Create the working folders

```bash
mkdir -p input output models
```

## 3. Download the Whisper model

Models are gitignored (several GB). Download them manually.

**Force IPv4 with `-4`.** Without it, large downloads hang forever with no error on
some networks.

```bash
mkdir -p ~/models/faster-whisper-large-v3
cd ~/models/faster-whisper-large-v3

curl -4 -L -O "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin"
curl -4 -L -O "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json"
curl -4 -L -O "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json"
curl -4 -L -O "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json"
curl -4 -L -O "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json"
```

`model.bin` is ~3GB, the rest are small.

## 4. Pull the translation model

```bash
ollama pull qwen3:8b
```

~5GB. Check Ollama is running first:

```bash
curl -s http://localhost:11434/api/tags
```

## 5. Chatterbox (automatic)

Chatterbox downloads its own weights (~3GB) on first run and caches them in
`~/.cache/huggingface/`. Nothing to do, but the first dub will be slower.

## 6. Disable IPv6

Required, or model downloads and yt-dlp will hang.

```bash
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1
```

Resets on reboot. To make it permanent:

```bash
echo -e "net.ipv6.conf.all.disable_ipv6 = 1\nnet.ipv6.conf.default.disable_ipv6 = 1" | sudo tee -a /etc/sysctl.conf
```

## 7. Install the `dub` command

```bash
chmod +x duball.sh
mkdir -p ~/.local/bin
ln -sf "$(pwd)/duball.sh" ~/.local/bin/dub

echo $PATH | grep -q "$HOME/.local/bin" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 8. Check it works

```bash
dub
```

Should print usage and list the available styles.

---

# Usage

Drop a clean voice-only audio file (no music, no SFX) in `input/`, then:

```bash
dub myfile.wav                              # all languages, casual style
dub myfile.wav --style duo --speakers 2     # conversation, two voices
dub myfile.wav --style professional de fr   # professional, German + French
```

Outputs land in `output/<filename>/<lang>.wav` and `<lang>.srt`.

Expect roughly 25-30 minutes per language for a 9-minute video. Run detached for long
jobs:

```bash
nohup dub myfile.wav > dub.log 2>&1 &
tail -f dub.log
```

## Editable config

- `glossary.txt` — terms never to translate (channel names, games, brands)
- `synopses/*.txt` — tone presets. Add a new `.txt` and it appears as a `--style`
  option automatically.

---

# Known issues

- Background noise gets transcribed as speech. Use a clean vocal stem.
- Most segments get time-compressed to fit the original timing, which causes a warbly
  artifact. `YTDUB_MAX_SPEEDUP=1.2` reduces it at the cost of some drift.
- `--target-cps` does nothing useful with the Ollama translator; the LLM treats the
  budget as a suggestion.

## Local patches

This repo is a patched fork of `mazzasaverio/youtube-auto-dub`. Changes:

1. `pipeline.py` — audio-only inputs skip the video mux instead of crashing
2. `stages/tts/base.py` — merges fragments under 3 chars into adjacent segments, which
   otherwise crash Chatterbox
3. `stages/tts/base.py` — logs full tracebacks on TTS failure
4. `stages/download.py` — forces IPv4 and enables the Node JS runtime in both
   `video_opts` and `audio_opts`, since yt-dlp used as a library ignores
   `~/.config/yt-dlp/config`
5. `stages/translate/ollama.py` — new LLM translation backend with glossary, synopsis
   and rolling context
6. `stages/translate/base.py` — registers `--translator ollama`
