#!/usr/bin/env bash
# duball.sh - dub one audio/video file into multiple languages with ytdub
#
# Usage:
#   dub myfile.wav                              # casual style, all languages
#   dub myfile.wav --style duo --speakers 2     # conversation, 2 voices
#   dub myfile.wav --style professional de fr   # professional, German + French
#
# Input files live in  ~/youtube-auto-dub/input/
# Outputs land in      ~/youtube-auto-dub/output/<filename>/<lang>.wav + <lang>.srt

set -uo pipefail

YTDUB_DIR="$HOME/youtube-auto-dub"
INPUT_DIR="$YTDUB_DIR/input"
OUTPUT_DIR="$YTDUB_DIR/output"
SYNOPSIS_DIR="$YTDUB_DIR/synopses"
ASR_MODEL="$HOME/models/faster-whisper-large-v3"
DEFAULT_LANGS=(de fr pl es nl hi)
DEFAULT_STYLE="casual"

# activate the venv so this works when called from anywhere
source "$YTDUB_DIR/.venv/bin/activate"

# ---- args ----
if [ $# -lt 1 ]; then
  echo "Usage: $(basename "$0") <filename> [--style NAME] [--speakers N] [langs...]"
  echo ""
  echo "  <filename>      file in $INPUT_DIR"
  echo "  --style NAME    tone preset (default: $DEFAULT_STYLE)"
  echo "  --speakers N    multi-voice mode: N speakers (0 = auto-detect)"
  echo "  [langs]         optional; defaults to ${DEFAULT_LANGS[*]}"
  echo ""
  echo "Available styles:"
  if [ -d "$SYNOPSIS_DIR" ]; then
    for f in "$SYNOPSIS_DIR"/*.txt; do
      [ -e "$f" ] && echo "  $(basename "$f" .txt)"
    done
  else
    echo "  (none - $SYNOPSIS_DIR not found)"
  fi
  echo ""
  echo "Examples:"
  echo "  dub solo.wav"
  echo "  dub chat.wav --style duo --speakers 2"
  echo "  dub doc.wav --style professional de fr"
  exit 1
fi

INFILE_NAME="$1"
shift

STYLE="$DEFAULT_STYLE"
SPEAKERS=""

# flags can come in either order
while [ $# -gt 0 ]; do
  case "${1:-}" in
    --style)
      STYLE="${2:-$DEFAULT_STYLE}"
      shift 2
      ;;
    --speakers)
      SPEAKERS="${2:-0}"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [ $# -gt 0 ]; then
  LANGS=("$@")
else
  LANGS=("${DEFAULT_LANGS[@]}")
fi

INFILE="$INPUT_DIR/$INFILE_NAME"
if [ ! -f "$INFILE" ]; then
  echo "ERROR: $INFILE not found"
  exit 1
fi

SYNOPSIS_FILE="$SYNOPSIS_DIR/$STYLE.txt"
if [ ! -f "$SYNOPSIS_FILE" ]; then
  echo "ERROR: no style called '$STYLE' ($SYNOPSIS_FILE not found)"
  echo "Available:"
  for f in "$SYNOPSIS_DIR"/*.txt; do
    [ -e "$f" ] && echo "  $(basename "$f" .txt)"
  done
  exit 1
fi
export YTDUB_SYNOPSIS="$SYNOPSIS_FILE"

BASENAME="${INFILE_NAME%.*}"
DEST="$OUTPUT_DIR/$BASENAME"
mkdir -p "$DEST"

echo "=========================================="
echo "Input:     $INFILE"
echo "Style:     $STYLE"
echo "Languages: ${LANGS[*]}"
if [ -n "$SPEAKERS" ]; then
  if [ "$SPEAKERS" = "0" ]; then
    echo "Mode:      multi-voice (auto-detect speakers)"
  else
    echo "Mode:      multi-voice ($SPEAKERS speakers)"
  fi
else
  echo "Mode:      single voice"
fi
echo "Output:    $DEST"
echo "=========================================="

cd "$YTDUB_DIR" || exit 1

FAILED=()

for lang in "${LANGS[@]}"; do
  echo ""
  echo "--- [$lang] starting $(date +%H:%M:%S) ---"

  if [ -n "$SPEAKERS" ]; then
    ytdub dub "$INFILE" \
      --target "$lang" \
      --asr-model "$ASR_MODEL" \
      --translator ollama \
      --diarize --speakers "$SPEAKERS"
    RC=$?
  else
    ytdub dub "$INFILE" \
      --target "$lang" \
      --asr-model "$ASR_MODEL" \
      --translator ollama
    RC=$?
  fi

  if [ $RC -ne 0 ]; then
    echo "--- [$lang] FAILED ---"
    FAILED+=("$lang")
    continue
  fi

  PRODUCED=$(ls -t "$YTDUB_DIR/data/output/${BASENAME}.${lang}."* 2>/dev/null)

  if [ -z "$PRODUCED" ]; then
    echo "--- [$lang] no output found ---"
    FAILED+=("$lang")
    continue
  fi

  MEDIA=$(echo "$PRODUCED" | grep -E '\.(mp4|mkv|wav|mp3|m4a)$' | head -1)
  SRT=$(echo "$PRODUCED" | grep -E '\.srt$' | head -1)

  if [ -n "$MEDIA" ]; then
    if [[ "$MEDIA" == *.wav ]]; then
      cp "$MEDIA" "$DEST/${lang}.wav"
    else
      ffmpeg -y -loglevel error -i "$MEDIA" -vn -acodec pcm_s16le "$DEST/${lang}.wav"
    fi
    echo "--- [$lang] wrote $DEST/${lang}.wav ---"
  fi

  if [ -n "$SRT" ]; then
    cp "$SRT" "$DEST/${lang}.srt"
    echo "--- [$lang] wrote $DEST/${lang}.srt ---"
  fi

  echo "--- [$lang] done $(date +%H:%M:%S) ---"
done

echo ""
echo "=========================================="
echo "FINISHED  (style: $STYLE)"
echo "Output: $DEST"
ls -lh "$DEST"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "FAILED languages: ${FAILED[*]}"
fi
echo "=========================================="
