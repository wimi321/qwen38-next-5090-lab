"""Small standard-library OpenAI-compatible HTTP/SSE client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class Q38HTTPError(RuntimeError):
    pass


@dataclass(frozen=True)
class SSEEvent:
    payload: dict[str, Any]
    elapsed_seconds: float


@dataclass(frozen=True)
class SSEResponse:
    events: tuple[SSEEvent, ...]
    done: bool
    elapsed_seconds: float


class OpenAIHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid HTTP base URL: {base_url!r}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base URL must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base URL must not contain a query or fragment")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._urlopen = urlopen
        self._monotonic = monotonic

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/" + path.lstrip("/"),
            data=body,
            method=method,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": "q38lab/0.1",
            },
        )
        try:
            return self._urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except OSError:
                detail = ""
            raise Q38HTTPError(f"HTTP {exc.code} {path}: {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise Q38HTTPError(f"request failed for {path}: {exc}") from exc

    def get_json(self, path: str) -> dict[str, Any]:
        with self._request("GET", path) as response:
            raw = response.read()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise Q38HTTPError(f"{path} did not return JSON") from exc
        if not isinstance(parsed, dict):
            raise Q38HTTPError(f"{path} returned a non-object JSON response")
        return parsed

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._request("POST", path, payload) as response:
            raw = response.read()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise Q38HTTPError(f"{path} did not return JSON") from exc
        if not isinstance(parsed, dict):
            raise Q38HTTPError(f"{path} returned a non-object JSON response")
        if "error" in parsed:
            raise Q38HTTPError(f"{path} returned an API error: {parsed['error']}")
        return parsed

    def post_sse(self, path: str, payload: dict[str, Any]) -> SSEResponse:
        start = self._monotonic()
        events: list[SSEEvent] = []
        done = False
        data_lines: list[str] = []

        def consume() -> None:
            nonlocal done
            if not data_lines:
                return
            data = "\n".join(data_lines)
            data_lines.clear()
            if data == "[DONE]":
                done = True
                return
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as exc:
                raise Q38HTTPError(f"invalid SSE JSON: {data[:200]}") from exc
            if not isinstance(parsed, dict):
                raise Q38HTTPError("SSE data was not a JSON object")
            if "error" in parsed:
                raise Q38HTTPError(f"stream returned an API error: {parsed['error']}")
            events.append(SSEEvent(parsed, self._monotonic() - start))

        with self._request("POST", path, payload) as response:
            for raw_line in response:
                try:
                    line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise Q38HTTPError("SSE stream contains invalid UTF-8") from exc
                if line == "":
                    consume()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            consume()

        elapsed = self._monotonic() - start
        if not done:
            raise Q38HTTPError("SSE stream ended without [DONE]")
        return SSEResponse(tuple(events), done=True, elapsed_seconds=elapsed)
