"""Pipeline stages. Heavy dependencies (torch, faster-whisper, Chatterbox, yt-dlp) are
imported lazily inside the functions that need them, so importing a stage module, and
running the unit tests, never requires a GPU or model weights."""
