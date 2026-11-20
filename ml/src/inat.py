"""Small, polite client for the public iNaturalist API (no account needed).

iNaturalist asks API users to stay around 1 request/second and ~10k requests/day,
and to keep media downloads well under ~5 GB/hour. This client enforces the
1 req/s limit for API calls and retries on rate-limit / server errors.
"""

from __future__ import annotations

import time

import requests

API_URL = "https://api.inaturalist.org/v1"
USER_AGENT = "BirdIdentifier-learning-project/0.1 (+https://github.com)"

# Well-known iNaturalist IDs
EUROPE_PLACE_ID = 97391
AVES_TAXON_ID = 3


class INatClient:
    def __init__(self, min_interval: float = 1.0, max_retries: int = 5):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def get(self, path: str, **params) -> dict:
        """GET an API endpoint and return the decoded JSON."""
        for attempt in range(self.max_retries):
            self._throttle()
            backoff = 2 ** attempt * 5
            try:
                resp = self.session.get(f"{API_URL}/{path}", params=params, timeout=30)
            except (requests.ConnectionError, requests.Timeout) as e:
                print(f"  network error ({type(e).__name__}), retrying in {backoff}s...")
                time.sleep(backoff)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                print(f"  HTTP {resp.status_code}, retrying in {backoff}s...")
                time.sleep(backoff)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Giving up on {path} after {self.max_retries} attempts")

    def download(self, url: str, dest, timeout: float = 30) -> None:
        """Download a file (photos are served from S3, not the API)."""
        resp = self.session.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
