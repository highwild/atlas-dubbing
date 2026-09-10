"""Network hardening: IPv4 by default and timeouts on everything.

On a machine with a broken IPv6 route, small requests succeed but large transfers hang
forever with no error. That hit yt-dlp and Hugging Face downloads alike. Two defences,
applied once at startup for the whole process:

* Resolve hostnames to IPv4 only (opt out with ``--allow-ipv6`` / ``YTDUB_FORCE_IPV4=0``).
  Patching ``getaddrinfo`` covers urllib, requests, httpx and anything else in-process.
* A default socket timeout, so a stalled transfer raises instead of hanging. This is a
  per-read timeout, not a total one, so large downloads that keep flowing are fine.

Subprocesses (e.g. demucs downloading its own weights) are not covered by the patch;
the env vars set here cover Hugging Face in child processes too.
"""

from __future__ import annotations

import os
import socket

from ytdub.logging import stage_logger

log = stage_logger("net")

_original_getaddrinfo = socket.getaddrinfo
_configured = False


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family not in (0, socket.AF_UNSPEC):
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    try:
        return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    except socket.gaierror:
        # IPv6-only host: fall through rather than make it unreachable.
        log.warning(f"{host} has no IPv4 address; using IPv6 for it")
        return _original_getaddrinfo(host, port, family, type, proto, flags)


def configure_network(*, force_ipv4: bool = True, timeout: float = 60.0) -> None:
    """Idempotent process-wide network setup."""
    global _configured
    if _configured:
        return
    socket.setdefaulttimeout(timeout)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(int(timeout)))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(int(min(timeout, 30))))
    if force_ipv4:
        socket.getaddrinfo = _ipv4_getaddrinfo
    log.debug(f"network: ipv4_only={force_ipv4} socket_timeout={timeout}s")
    _configured = True
