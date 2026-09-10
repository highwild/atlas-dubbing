"""Acquire a source from a URL via yt-dlp (local files skip this entirely).

Two traps, both handled here in the single options dict used for every call:

* YouTube's JS challenge needs a JS runtime (Node) enabled plus yt-dlp's remote solver
  components, or downloads hang at "Downloading webpage".
* yt-dlp used as a *library* ignores ``~/.config/yt-dlp/config``, so nothing there
  applies. The reference implementation had two option dicts and fixing one gave a
  confusing partial failure; there is now exactly one (:func:`ydl_options`) and only
  one download call, since audio is extracted from the result with ffmpeg afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path

from ytdub.logging import stage_logger

log = stage_logger("download")

_ID_RE = re.compile(r"(?:v=|/shorts/|youtu\.be/|/)([0-9A-Za-z_-]{11})(?:[?&/]|$)")


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def extract_video_id(url: str) -> str | None:
    match = _ID_RE.search(url)
    return match.group(1) if match else None


def ydl_options(out_dir: Path, *, force_ipv4: bool = True, timeout: float = 60.0,
                cookies_from_browser: str | None = None,
                cookies_file: Path | None = None) -> dict:
    opts: dict = {
        "format": ("bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best[vcodec^=avc1]/"
                   "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noprogress": True,
        "no_warnings": False,
        "socket_timeout": timeout,
        "retries": 5,
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
    }
    if force_ipv4:
        opts["source_address"] = "0.0.0.0"
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file:
        opts["cookiefile"] = str(cookies_file)
    return opts


def download(url: str, out_dir: Path, **opts) -> Path:
    """Download ``url`` into ``out_dir`` and return the media file path."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {url}")
    with yt_dlp.YoutubeDL(ydl_options(out_dir, **opts)) as ydl:
        info = ydl.extract_info(url, download=True)
    path = out_dir / f"{info['id']}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"yt-dlp did not produce {path}")
    log.success(f"Downloaded {info.get('title', info['id'])!r} -> {path.name}")
    return path
