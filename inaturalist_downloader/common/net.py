"""HTTP and download helpers for iNaturalist API and image files."""

import json
import time
from http.client import IncompleteRead
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import API_BASE, DEFAULT_TIMEOUT, USER_AGENT


def http_get_bytes(url: str, params: Optional[dict] = None, retries: int = 5) -> bytes:
    """Fetch bytes from a URL with retry/backoff."""
    final_url = url
    if params:
        final_url = f"{url}?{urlencode(params)}"

    last_error = None
    for attempt in range(1, retries + 1):
        request = Request(final_url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError, IncompleteRead) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(attempt, 5))

    raise RuntimeError(f"Request failed for {final_url}: {last_error}")


def api_get(path: str, retries: int = 5, **params) -> dict:
    """Fetch and decode JSON from an iNaturalist API path."""
    payload = http_get_bytes(f"{API_BASE}{path}", params=params, retries=retries)
    return json.loads(payload.decode("utf-8"))


def http_stream_to_file(url: str, destination: Path, retries: int = 5) -> None:
    """Stream a URL response to disk using a temporary file and retries."""
    last_error = None
    for attempt in range(1, retries + 1):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        tmp_file = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                with open(tmp_file, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp_file.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError, IncompleteRead) as exc:
            last_error = exc
            if tmp_file.exists():
                tmp_file.unlink()
            if attempt == retries:
                break
            time.sleep(min(attempt, 5))

    raise RuntimeError(f"Streaming download failed for {url}: {last_error}")


def download_file(url: str, destination: Path, overwrite: bool, retries: int = 5) -> bool:
    """Download one file unless it already exists and overwrite is disabled."""
    if destination.exists() and not overwrite:
        return False

    http_stream_to_file(url, destination, retries=retries)
    return True
