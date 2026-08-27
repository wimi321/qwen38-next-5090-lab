# Modified by Qwen3.8 Next 5090 Lab contributors in 2026; see MODIFICATIONS.md.
"""Safe image acquisition for OpenAI-compatible structured chat messages."""

from __future__ import annotations

import asyncio
import base64
import binascii
import http.client
import io
import ipaddress
import json
import os
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote_to_bytes, urljoin, urlsplit

from freetoken.multimodal import MediaPayload


MAX_IMAGES = 4
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 40 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 10.0
MAX_REDIRECTS = 3
_READ_CHUNK = 64 * 1024
DOH_FALLBACK_ENV = "Q38LAB_DOH_FALLBACK"
# The fallback never resolves its own endpoint through the host resolver.  The
# addresses below are Cloudflare's documented anycast resolver addresses; TLS
# still authenticates the fixed hostname rather than the numeric peer.
_DOH_HOSTNAME = "cloudflare-dns.com"
_DOH_APPROVED_IPS = ("1.1.1.1", "1.0.0.1")
_DOH_TARGET = "/dns-query"
_DOH_MAX_BODY_BYTES = 64 * 1024
_DOH_MAX_ANSWER_RECORDS = 32
_DOH_MAX_CNAME_DEPTH = 4
# Python exposes no portable cancellation primitive for a getaddrinfo already
# inside libc/NSS.  Keep those calls behind deadline-bounded daemon helpers and
# cap the number that can remain stuck after a timeout.  The outer image-fetch
# worker therefore returns at the request deadline, but this is deliberately
# reported as a soft-cancellation limitation by q38lab doctor/evidence.
SYSTEM_DNS_HARD_CANCEL_SUPPORTED = False
_SYSTEM_RESOLVER_SLOTS = 4
_system_resolver_slots = threading.BoundedSemaphore(_SYSTEM_RESOLVER_SLOTS)
_ALLOWED_MIME_TYPES = frozenset(
    {
        "image/bmp",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


class MediaError(ValueError):
    """Client-classifiable media validation/fetch failure."""

    def __init__(self, message: str, *, code: str = "invalid_image_url") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _FetchResult:
    status: int
    mime_type: str
    body: bytes
    location: str | None


class _DownloadBudget:
    """Thread-safe request-wide allowance for retained encoded image bytes."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.used = 0
        self._lock = threading.Lock()

    def reserve(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("download budget reservations must be non-negative")
        with self._lock:
            if self.used + amount > self.limit:
                raise MediaError(
                    f"images exceed the {self.limit // (1024 * 1024)} MiB total limit",
                    code="image_too_large",
                )
            self.used += amount

    def release(self, amount: int) -> None:
        with self._lock:
            self.used -= amount
            if self.used < 0:  # pragma: no cover - invariant guard
                raise RuntimeError("download budget released more bytes than reserved")


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection whose TCP peer is the address we already audited.

    TLS SNI and certificate verification still use the original hostname.  This
    closes the validate-then-resolve DNS-rebinding hole present when a normal
    high-level client performs a second lookup during connect.
    """

    def __init__(
        self,
        hostname: str,
        approved_ip: str,
        port: int,
        timeout: float,
        *,
        deadline: float | None = None,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self._approved_ip = approved_ip
        self._deadline = time.monotonic() + timeout if deadline is None else deadline

    def connect(self) -> None:  # pragma: no cover - exercised through mocked socket in unit tests
        raw = socket.create_connection(
            (self._approved_ip, self.port), _remaining_timeout(self._deadline)
        )
        # Publish the raw socket before TLS so the hard-deadline watchdog can
        # interrupt a stalled handshake as well as later HTTP reads.
        self.sock = raw
        try:
            raw.settimeout(_remaining_timeout(self._deadline))
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


class _ConnectionDeadline:
    """Best-effort hard stop for blocking TLS/HTTP internals.

    Socket timeouts are inactivity timers.  ``http.client.getresponse`` may
    perform several reads internally, so merely setting the remaining timeout
    before the call does not stop a slowloris that drips header bytes.  A daemon
    watchdog closes/shuts down the pinned socket at the shared absolute
    deadline.  It is cancelled after the hop completes.
    """

    def __init__(self, connection: http.client.HTTPSConnection, deadline: float) -> None:
        self._connection = connection
        self._timer = threading.Timer(
            _remaining_timeout(deadline), self._abort_connection
        )
        self._timer.daemon = True

    def _abort_connection(self) -> None:
        sock = getattr(self._connection, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            self._connection.close()
        except OSError:
            pass

    def __enter__(self) -> _ConnectionDeadline:
        self._timer.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._timer.cancel()


def _normalized_mime(raw: str | None) -> str:
    return (raw or "").split(";", 1)[0].strip().lower()


def _validate_mime(mime_type: str) -> str:
    mime_type = _normalized_mime(mime_type)
    if mime_type not in _ALLOWED_MIME_TYPES:
        raise MediaError(
            f"unsupported image MIME type {mime_type or '(missing)'!r}; supported types are "
            + ", ".join(sorted(_ALLOWED_MIME_TYPES)),
            code="unsupported_content_type",
        )
    return mime_type


def _public_ip(value: str) -> bool:
    try:
        addr = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_global


def _doh_fallback_enabled() -> bool:
    return os.getenv(DOH_FALLBACK_ENV) == "1"


def _system_getaddrinfo(
    hostname: str, port: int, deadline: float
) -> list[tuple[object, ...]]:
    """Run libc/NSS resolution behind a bounded deadline.

    ``socket.getaddrinfo`` has no per-call timeout and cannot be killed
    portably.  A daemon helper keeps the caller deadline-bounded; a semaphore
    limits pathological stuck resolver calls to four for the process lifetime.
    This is soft cancellation, intentionally exposed by doctor/evidence.
    """

    if not _system_resolver_slots.acquire(blocking=False):
        raise MediaError(
            "system DNS resolver capacity is exhausted after timed-out lookups",
            code="image_fetch_timeout",
        )
    done = threading.Event()
    result: dict[str, object] = {}

    def resolve() -> None:
        try:
            result["answers"] = socket.getaddrinfo(
                hostname, port, type=socket.SOCK_STREAM
            )
        except BaseException as exc:  # carried back to the request thread
            result["error"] = exc
        finally:
            _system_resolver_slots.release()
            done.set()

    worker = threading.Thread(
        target=resolve,
        name="q38lab-system-dns",
        daemon=True,
    )
    worker.start()
    if not done.wait(_remaining_timeout(deadline)):
        raise MediaError("image fetch timed out during DNS resolution", code="image_fetch_timeout")
    error = result.get("error")
    if error is not None:
        if isinstance(error, OSError):
            raise MediaError(f"could not resolve image host {hostname!r}: {error}") from error
        raise MediaError(
            f"system DNS resolver failed for image host {hostname!r}: {type(error).__name__}"
        ) from error
    answers = result.get("answers")
    if not isinstance(answers, list):  # pragma: no cover - invariant guard
        raise MediaError(f"system DNS resolver returned no result for {hostname!r}")
    return answers


def _normalize_dns_name(value: str) -> str:
    raw = value.strip().rstrip(".")
    if not raw:
        raise MediaError("DoH returned an empty DNS name")
    try:
        normalized = raw.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise MediaError("DoH returned an invalid internationalized DNS name") from exc
    labels = normalized.split(".")
    if len(normalized) > 253 or any(not label or len(label) > 63 for label in labels):
        raise MediaError("DoH returned an invalid DNS name")
    return normalized


def _read_small_response(
    connection: http.client.HTTPSConnection,
    response: http.client.HTTPResponse,
    deadline: float,
    limit: int,
) -> bytes:
    declared_raw = response.getheader("Content-Length")
    if declared_raw is not None:
        try:
            declared = int(declared_raw)
        except ValueError as exc:
            raise MediaError("DoH returned an invalid Content-Length") from exc
        if declared < 0 or declared > limit:
            raise MediaError("DoH response exceeded its bounded body limit")
    body = io.BytesIO()
    total = 0
    while True:
        _arm_socket_deadline(connection, deadline)
        chunk = response.read(min(_READ_CHUNK, limit + 1 - total))
        if not chunk:
            break
        body.write(chunk)
        total += len(chunk)
        if total > limit:
            raise MediaError("DoH response exceeded its bounded body limit")
    return body.getvalue()


def _doh_query(name: str, rr_type: str, deadline: float) -> dict[str, Any]:
    from urllib.parse import urlencode

    target = _DOH_TARGET + "?" + urlencode(
        {"name": name, "type": rr_type, "do": "false", "cd": "false"}
    )
    last_error: BaseException | None = None
    for approved_ip in _DOH_APPROVED_IPS:
        if not _public_ip(approved_ip):  # pragma: no cover - constant audit guard
            raise RuntimeError("configured DoH peer is not a public IP")
        connection = _PinnedHTTPSConnection(
            _DOH_HOSTNAME,
            approved_ip,
            443,
            _remaining_timeout(deadline),
            deadline=deadline,
        )
        try:
            with _ConnectionDeadline(connection, deadline):
                connection.connect()
                _arm_socket_deadline(connection, deadline)
                connection.request(
                    "GET",
                    target,
                    headers={
                        "Accept": "application/dns-json",
                        "User-Agent": "qwen38-next-5090-lab/0.2 image-fetcher",
                        "Connection": "close",
                    },
                )
                _arm_socket_deadline(connection, deadline)
                response = connection.getresponse()
                if response.status in _REDIRECT_CODES:
                    raise MediaError("DoH redirects are rejected")
                if response.status < 200 or response.status >= 300:
                    raise MediaError(f"DoH returned HTTP {response.status}")
                if _normalized_mime(response.getheader("Content-Type")) != "application/dns-json":
                    raise MediaError("DoH returned an unexpected content type")
                payload = _read_small_response(
                    connection, response, deadline, _DOH_MAX_BODY_BYTES
                )
            try:
                document = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MediaError("DoH returned malformed JSON") from exc
            if not isinstance(document, dict):
                raise MediaError("DoH response must be a JSON object")
            return document
        except MediaError:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise MediaError("image fetch timed out during DoH", code="image_fetch_timeout") from exc
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            if time.monotonic() >= deadline:
                raise MediaError(
                    "image fetch timed out during DoH", code="image_fetch_timeout"
                ) from exc
            last_error = exc
        finally:
            connection.close()
    raise MediaError(f"could not reach fixed DoH resolver: {last_error}")


def _parse_doh_answer(
    document: dict[str, Any], query_name: str, rr_type: str
) -> tuple[list[str], str | None, int]:
    status = document.get("Status")
    if type(status) is not int:
        raise MediaError("DoH response has no integer DNS status")
    if status == 3:  # NXDOMAIN
        return [], None, 0
    if status != 0 or document.get("TC") is True:
        raise MediaError(f"DoH DNS query failed with status {status}")
    records = document.get("Answer", [])
    if records is None:
        records = []
    if not isinstance(records, list) or len(records) > _DOH_MAX_ANSWER_RECORDS:
        raise MediaError("DoH response exceeded its answer-record limit")

    address_type = 1 if rr_type == "A" else 28
    addresses: dict[str, list[str]] = {}
    aliases: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise MediaError("DoH answer record must be an object")
        try:
            owner = _normalize_dns_name(str(record["name"]))
            record_type = int(record["type"])
            data = str(record["data"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MediaError("DoH answer record is malformed") from exc
        if record_type in (1, 28):
            try:
                address = str(ipaddress.ip_address(data.split("%", 1)[0]))
            except ValueError as exc:
                raise MediaError("DoH returned a malformed IP address") from exc
            # Reject the whole trusted answer set, not merely the selected
            # canonical owner, if it contains any non-global address.
            if not _public_ip(address):
                raise MediaError(
                    "DoH resolved the image host to a private, loopback, link-local, or reserved address",
                    code="unsafe_image_url",
                )
            if record_type == address_type:
                addresses.setdefault(owner, []).append(address)
        elif record_type == 5:
            aliases.setdefault(owner, set()).add(_normalize_dns_name(data))

    current = _normalize_dns_name(query_name)
    seen: set[str] = set()
    hops = 0
    for _ in range(_DOH_MAX_CNAME_DEPTH + 1):
        if current in seen:
            raise MediaError("DoH returned a CNAME loop")
        seen.add(current)
        if current in addresses:
            return list(dict.fromkeys(addresses[current])), None, hops
        targets = aliases.get(current, set())
        if not targets:
            return [], (current if hops else None), hops
        if len(targets) != 1:
            raise MediaError("DoH returned conflicting CNAME targets")
        current = next(iter(targets))
        hops += 1
    raise MediaError("DoH CNAME chain exceeded its depth limit")


def _resolve_doh_addresses(hostname: str, deadline: float) -> list[str]:
    normalized = _normalize_dns_name(hostname)
    results: list[str] = []
    for rr_type in ("A", "AAAA"):
        current = normalized
        seen: set[str] = set()
        total_hops = 0
        for _ in range(_DOH_MAX_CNAME_DEPTH + 1):
            if current in seen:
                raise MediaError("DoH returned a CNAME loop")
            seen.add(current)
            document = _doh_query(current, rr_type, deadline)
            addresses, next_name, hops = _parse_doh_answer(document, current, rr_type)
            total_hops += hops
            if total_hops > _DOH_MAX_CNAME_DEPTH:
                raise MediaError("DoH CNAME chain exceeded its depth limit")
            if addresses:
                results.extend(addresses)
                break
            if next_name is None:
                break
            current = next_name
        else:  # pragma: no cover - loop bound documents the invariant
            raise MediaError("DoH CNAME chain exceeded its depth limit")
    results = list(dict.fromkeys(results))
    if not results:
        raise MediaError(f"fixed DoH resolver found no public address for {hostname!r}")
    return results


def _resolve_public_addresses(
    hostname: str, port: int, deadline: float | None = None
) -> list[str]:
    if deadline is None:
        deadline = time.monotonic() + FETCH_TIMEOUT_SECONDS
    answers = _system_getaddrinfo(hostname, port, deadline)
    addresses = list(dict.fromkeys(answer[4][0] for answer in answers))
    if not addresses:
        raise MediaError(f"image host {hostname!r} has no addresses")
    # Reject the hostname if *any* answer is non-public.  Picking only a public
    # answer would permit split-horizon/rebinding hosts to smuggle an internal peer.
    blocked = [address for address in addresses if not _public_ip(address)]
    if not blocked:
        return addresses
    # Mixed public/private answers remain an unconditional rejection.  Falling
    # back in that case would weaken the existing rebinding defense.  Numeric
    # literals also never go through DoH.
    try:
        ipaddress.ip_address(hostname.split("%", 1)[0])
        numeric_literal = True
    except ValueError:
        numeric_literal = False
    if (
        len(blocked) == len(addresses)
        and not numeric_literal
        and _doh_fallback_enabled()
    ):
        return _resolve_doh_addresses(hostname, deadline)
    if blocked:
        raise MediaError(
            f"image URL resolves to a private, loopback, link-local, or reserved address",
            code="unsafe_image_url",
        )
    raise AssertionError("unreachable")


def _validate_https_url(url: str) -> tuple[str, int, str]:
    if not isinstance(url, str) or not url:
        raise MediaError("image_url.url must be a non-empty string")
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https":
        if parsed.scheme.lower() in ("file", "http", "ftp") or not parsed.scheme:
            raise MediaError(
                "only HTTPS and base64 data image URLs are supported; local files and HTTP are rejected",
                code="unsafe_image_url",
            )
        raise MediaError(f"unsupported image URL scheme {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise MediaError("image URLs containing credentials are rejected", code="unsafe_image_url")
    if not parsed.hostname:
        raise MediaError("image URL must include a hostname")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise MediaError(f"invalid image URL port: {exc}") from exc
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return parsed.hostname.rstrip("."), port, target


def _remaining_timeout(deadline: float) -> float:
    """Return the remaining request budget or raise a client-classifiable timeout."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise MediaError("image fetch timed out", code="image_fetch_timeout")
    return remaining


def _arm_socket_deadline(connection: http.client.HTTPSConnection, deadline: float) -> float:
    """Apply the absolute deadline to the next blocking socket operation.

    ``http.client`` otherwise keeps the timeout chosen at connect time.  Resetting
    it before every request/response/read prevents a peer from consuming the full
    timeout repeatedly by sending one small chunk at a time.
    """

    remaining = _remaining_timeout(deadline)
    if connection.sock is not None:
        connection.sock.settimeout(remaining)
    return remaining


def _request_once(
    url: str, deadline: float, *, budget: _DownloadBudget | None = None
) -> _FetchResult:
    hostname, port, target = _validate_https_url(url)
    _remaining_timeout(deadline)
    addresses = _resolve_public_addresses(hostname, port, deadline)
    _remaining_timeout(deadline)
    last_error: BaseException | None = None
    for address in addresses:
        remaining = _remaining_timeout(deadline)
        connection = _PinnedHTTPSConnection(
            hostname, address, port, remaining, deadline=deadline
        )
        reserved = 0
        retain_reservation = False
        try:
            with _ConnectionDeadline(connection, deadline):
                # Connect explicitly so the TCP connect and TLS handshake both
                # consume the same absolute request budget.
                connection.connect()
                _arm_socket_deadline(connection, deadline)
                connection.request(
                    "GET",
                    target,
                    headers={
                        "Accept": "image/webp,image/png,image/jpeg,image/gif,image/bmp",
                        "User-Agent": "qwen38-next-5090-lab/0.2 image-fetcher",
                        "Connection": "close",
                    },
                )
                _arm_socket_deadline(connection, deadline)
                response = connection.getresponse()
                location = response.getheader("Location")
                mime_type = _normalized_mime(response.getheader("Content-Type"))
                if response.status in _REDIRECT_CODES:
                    response.read(0)
                    return _FetchResult(response.status, mime_type, b"", location)
                if response.status < 200 or response.status >= 300:
                    _arm_socket_deadline(connection, deadline)
                    response.read(min(_READ_CHUNK, MAX_IMAGE_BYTES))
                    raise MediaError(f"image server returned HTTP {response.status}")
                mime_type = _validate_mime(mime_type)
                content_length = response.getheader("Content-Length")
                if content_length is not None:
                    try:
                        declared = int(content_length)
                    except ValueError as exc:
                        raise MediaError("image server returned an invalid Content-Length") from exc
                    if declared < 0 or declared > MAX_IMAGE_BYTES:
                        raise MediaError(
                            f"image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB per-image limit",
                            code="image_too_large",
                        )
                # BytesIO.getvalue() avoids the second full-size join allocation
                # that a list of chunks would create at the 40 MiB request edge.
                body = io.BytesIO()
                total = 0
                while True:
                    _arm_socket_deadline(connection, deadline)
                    chunk = response.read(min(_READ_CHUNK, MAX_IMAGE_BYTES + 1 - total))
                    if not chunk:
                        break
                    chunk_size = len(chunk)
                    if budget is not None:
                        budget.reserve(chunk_size)
                        reserved += chunk_size
                    body.write(chunk)
                    total += chunk_size
                    if total > MAX_IMAGE_BYTES:
                        raise MediaError(
                            f"image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB per-image limit",
                            code="image_too_large",
                        )
            if total == 0:
                raise MediaError("image response was empty")
            retain_reservation = True
            return _FetchResult(response.status, mime_type, body.getvalue(), None)
        except MediaError:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise MediaError("image fetch timed out", code="image_fetch_timeout") from exc
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            if time.monotonic() >= deadline:
                raise MediaError("image fetch timed out", code="image_fetch_timeout") from exc
            last_error = exc
        finally:
            connection.close()
            if budget is not None and reserved and not retain_reservation:
                budget.release(reserved)
    raise MediaError(f"could not fetch image: {last_error}")


def _fetch_https_sync(
    url: str,
    *,
    deadline: float | None = None,
    budget: _DownloadBudget | None = None,
) -> MediaPayload:
    if deadline is None:
        deadline = time.monotonic() + FETCH_TIMEOUT_SECONDS
    current = url
    for redirect in range(MAX_REDIRECTS + 1):
        _remaining_timeout(deadline)
        result = _request_once(current, deadline, budget=budget)
        if result.status not in _REDIRECT_CODES:
            return MediaPayload(result.mime_type, result.body, "https")
        if redirect == MAX_REDIRECTS:
            raise MediaError(f"image URL exceeded {MAX_REDIRECTS} redirects")
        if not result.location:
            raise MediaError("image redirect did not include a Location header")
        current = urljoin(current, result.location)
        # Validate immediately; _request_once repeats DNS validation after the
        # redirect so no hop can target localhost or a private network.
        _validate_https_url(current)
    raise AssertionError("unreachable")


def _decode_data_url(
    url: str, *, budget: _DownloadBudget | None = None
) -> MediaPayload:
    header, separator, encoded = url.partition(",")
    if not separator or not header.lower().startswith("data:"):
        raise MediaError("malformed data image URL")
    metadata = header[5:].split(";")
    mime_type = _validate_mime(metadata[0])
    flags = {item.lower() for item in metadata[1:] if item}
    if flags != {"base64"}:
        raise MediaError("data image URLs must use base64 encoding")
    # Refuse obviously oversized input before allocating the decoded buffer.
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise MediaError(
            f"image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB per-image limit",
            code="image_too_large",
        )
    encoded_bytes = unquote_to_bytes(encoded)
    padding = len(encoded_bytes) - len(encoded_bytes.rstrip(b"="))
    unpadded = encoded_bytes[:-padding] if padding else encoded_bytes
    if len(encoded_bytes) % 4 or padding > 2 or b"=" in unpadded:
        raise MediaError("data image URL contains invalid base64")
    decoded_size = len(encoded_bytes) // 4 * 3 - padding
    if decoded_size > MAX_IMAGE_BYTES:
        raise MediaError(
            f"image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB per-image limit",
            code="image_too_large",
        )
    reserved = False
    if budget is not None:
        budget.reserve(decoded_size)
        reserved = True
    try:
        data = base64.b64decode(encoded_bytes, validate=True)
    except (binascii.Error, ValueError) as exc:
        if reserved:
            budget.release(decoded_size)  # type: ignore[union-attr]
        raise MediaError("data image URL contains invalid base64") from exc
    if not data:
        if reserved:
            budget.release(decoded_size)  # type: ignore[union-attr]
        raise MediaError("data image URL is empty")
    if len(data) > MAX_IMAGE_BYTES:
        if reserved:
            budget.release(decoded_size)  # type: ignore[union-attr]
        raise MediaError(
            f"image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB per-image limit",
            code="image_too_large",
        )
    return MediaPayload(mime_type, data, "data")


async def load_image_url(
    url: str,
    *,
    deadline: float | None = None,
    budget: _DownloadBudget | None = None,
) -> MediaPayload:
    if url.lower().startswith("data:"):
        return _decode_data_url(url, budget=budget)
    if deadline is None:
        deadline = time.monotonic() + FETCH_TIMEOUT_SECONDS
    try:
        remaining = _remaining_timeout(deadline)
        # Cancelling ``to_thread`` cannot kill an in-flight worker.  The same
        # absolute deadline is therefore passed into the synchronous fetcher,
        # whose connect, response, and every read are socket-bounded by it.
        return await asyncio.wait_for(
            asyncio.to_thread(
                _fetch_https_sync, url, deadline=deadline, budget=budget
            ),
            timeout=remaining,
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise MediaError("image fetch timed out", code="image_fetch_timeout") from exc


def _image_url_from_part(part: dict[str, Any]) -> str:
    value = part.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value:
        raise MediaError("image_url content parts require image_url.url")
    return value


async def prepare_image_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[MediaPayload]]:
    """Fetch image parts and return URL-free processor-ready messages.

    Text-only conversations are returned without creating tasks.  Images are
    fetched concurrently under one ten-second request deadline and retain their
    conversation order.
    """

    normalized: list[dict[str, Any]] = []
    urls: list[str] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if not isinstance(content, list):
            normalized.append(copied)
            continue
        parts: list[dict[str, Any]] = []
        for raw_part in content:
            if not isinstance(raw_part, dict):
                raise MediaError("message content parts must be objects")
            part = dict(raw_part)
            part_type = part.get("type")
            if part_type == "text":
                parts.append({"type": "text", "text": part.get("text") or ""})
            elif part_type in ("image_url", "input_image"):
                urls.append(_image_url_from_part(part))
                # No source URL crosses the frontend/worker boundary.
                parts.append({"type": "image"})
            elif part_type in ("audio", "audio_url", "input_audio", "video", "video_url"):
                raise MediaError(
                    f"content part type {part_type!r} is not supported; this release accepts images only",
                    code="unsupported_content_type",
                )
            else:
                raise MediaError(
                    f"unsupported content part type {part_type!r}",
                    code="unsupported_content_type",
                )
        copied["content"] = parts
        normalized.append(copied)

    if not urls:
        return normalized, []
    if len(urls) > MAX_IMAGES:
        raise MediaError(f"at most {MAX_IMAGES} images are allowed per request", code="too_many_images")
    deadline = time.monotonic() + FETCH_TIMEOUT_SECONDS
    budget = _DownloadBudget(MAX_TOTAL_IMAGE_BYTES)
    tasks = [
        asyncio.create_task(load_image_url(url, deadline=deadline, budget=budget))
        for url in urls
    ]
    try:
        payloads = await asyncio.wait_for(
            asyncio.gather(*tasks), timeout=_remaining_timeout(deadline)
        )
    except asyncio.TimeoutError as exc:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise MediaError("image fetch timed out", code="image_fetch_timeout") from exc
    except BaseException:
        # ``gather`` propagates the first error while siblings keep running
        # unless cancelled explicitly. Rejected requests must not retain sockets
        # or partially downloaded image bodies in the background.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    oversized = next((payload for payload in payloads if len(payload.data) > MAX_IMAGE_BYTES), None)
    if oversized is not None:
        raise MediaError(
            f"image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB per-image limit",
            code="image_too_large",
        )
    total = sum(len(payload.data) for payload in payloads)
    if total > MAX_TOTAL_IMAGE_BYTES:
        raise MediaError(
            f"images exceed the {MAX_TOTAL_IMAGE_BYTES // (1024 * 1024)} MiB total limit",
            code="image_too_large",
        )
    return normalized, payloads
