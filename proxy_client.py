"""HTTP client with proxy rotation for ketabonline.com API requests."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
    "Referer": "https://ketabonline.com/",
    "Origin": "https://ketabonline.com",
}

USER_AGENTS = [
    DEFAULT_HEADERS["User-Agent"],
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    ),
]

RETRYABLE_STATUS_CODES = {403, 407, 408, 429, 500, 502, 503, 504}


def load_proxy_list(proxy_file: str | None = None) -> list[str]:
    """Load proxies from PROXY_LIST env, proxies.txt, or HTTP(S)_PROXY."""
    proxies: list[str] = []

    env_list = os.environ.get("PROXY_LIST", "").strip()
    if env_list:
        proxies.extend(p.strip() for p in env_list.split(",") if p.strip())

    file_path = Path(proxy_file or os.environ.get("PROXY_FILE", "proxies.txt"))
    if file_path.is_file():
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                proxies.append(line)

    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(key, "").strip()
        if value and value not in proxies:
            proxies.append(value)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for proxy in proxies:
        if proxy not in seen:
            seen.add(proxy)
            unique.append(proxy)
    return unique


def _normalize_proxy(proxy: str) -> dict[str, str]:
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    return {"http": proxy, "https": proxy}


class ProxySession:
    """Requests session that rotates proxies and retries on block responses."""

    def __init__(
        self,
        proxies: list[str] | None = None,
        max_retries: int = 5,
        request_delay: float = 1.0,
        timeout: int = 60,
    ) -> None:
        self.proxies = proxies if proxies is not None else load_proxy_list()
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.timeout = timeout
        self._proxy_index = 0
        self._last_request_at = 0.0

    def _next_proxy(self) -> dict[str, str] | None:
        if not self.proxies:
            return None
        proxy = self.proxies[self._proxy_index % len(self.proxies)]
        self._proxy_index += 1
        return _normalize_proxy(proxy)

    def _throttle(self) -> None:
        if self.request_delay <= 0:
            return
        elapsed = time.time() - self._last_request_at
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        headers = dict(DEFAULT_HEADERS)
        headers.update(kwargs.pop("headers", {}))
        headers["User-Agent"] = random.choice(USER_AGENTS)

        last_error: Exception | None = None
        attempts = max(self.max_retries, len(self.proxies) or 1)

        for attempt in range(attempts):
            self._throttle()
            proxy_dict = self._next_proxy()
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=proxy_dict,
                    timeout=kwargs.pop("timeout", self.timeout),
                    **kwargs,
                )
                self._last_request_at = time.time()

                if response.status_code in RETRYABLE_STATUS_CODES:
                    proxy_label = proxy_dict["http"] if proxy_dict else "direct"
                    print(
                        f"  [proxy] HTTP {response.status_code} for {url} "
                        f"(attempt {attempt + 1}/{attempts}, proxy={proxy_label})"
                    )
                    time.sleep(min(2 ** attempt, 30))
                    continue

                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                proxy_label = proxy_dict["http"] if proxy_dict else "direct"
                print(
                    f"  [proxy] Request failed for {url} "
                    f"(attempt {attempt + 1}/{attempts}, proxy={proxy_label}): {exc}"
                )
                time.sleep(min(2 ** attempt, 30))

        if last_error:
            raise last_error
        raise requests.HTTPError(f"Failed to fetch {url} after {attempts} attempts")
