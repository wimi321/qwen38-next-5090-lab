from __future__ import annotations

import asyncio
import base64
import threading
import time

import pytest

from freetoken.multimodal import MediaPayload
from freetoken.server import media
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.request_logger import _to_payload


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_data_url_is_decoded_and_source_url_is_removed() -> None:
    url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
            ],
        }
    ]

    prepared, payloads = asyncio.run(media.prepare_image_messages(messages))

    assert prepared == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image"},
            ],
        }
    ]
    assert payloads == [MediaPayload("image/png", _PNG, "data")]
    assert url not in repr(prepared)


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("http://example.com/image.png", "unsafe_image_url"),
        ("file:///etc/passwd", "unsafe_image_url"),
        ("/tmp/image.png", "unsafe_image_url"),
        ("data:text/plain;base64,SGVsbG8=", "unsupported_content_type"),
    ],
)
def test_unsafe_or_non_image_sources_are_rejected(url: str, code: str) -> None:
    with pytest.raises(media.MediaError) as caught:
        asyncio.run(media.load_image_url(url))
    assert caught.value.code == code


def test_dns_answer_set_rejects_any_private_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        media.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (media.socket.AF_INET, media.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (media.socket.AF_INET, media.socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )
    with pytest.raises(media.MediaError, match="private") as caught:
        media._resolve_public_addresses("rebinding.example", 443)
    assert caught.value.code == "unsafe_image_url"


def _addrinfo(*addresses: str) -> list[tuple[object, ...]]:
    return [
        (media.socket.AF_INET, media.socket.SOCK_STREAM, 6, "", (address, 443))
        for address in addresses
    ]


def test_doh_fallback_is_explicit_and_only_for_all_nonpublic_system_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        media,
        "_system_getaddrinfo",
        lambda *_args: _addrinfo("198.18.0.1"),
    )
    called: list[str] = []
    monkeypatch.setattr(
        media,
        "_resolve_doh_addresses",
        lambda hostname, _deadline: called.append(hostname) or ["93.184.216.34"],
    )

    monkeypatch.delenv(media.DOH_FALLBACK_ENV, raising=False)
    with pytest.raises(media.MediaError) as caught:
        media._resolve_public_addresses("fake-ip.example", 443)
    assert caught.value.code == "unsafe_image_url"
    assert called == []

    monkeypatch.setenv(media.DOH_FALLBACK_ENV, "1")
    assert media._resolve_public_addresses("fake-ip.example", 443) == ["93.184.216.34"]
    assert called == ["fake-ip.example"]

    monkeypatch.setattr(
        media,
        "_system_getaddrinfo",
        lambda *_args: _addrinfo("93.184.216.34", "198.18.0.1"),
    )
    with pytest.raises(media.MediaError) as caught:
        media._resolve_public_addresses("mixed.example", 443)
    assert caught.value.code == "unsafe_image_url"
    assert called == ["fake-ip.example"], "mixed answers must never fall back to DoH"


def test_doh_answer_rejects_any_nonpublic_address_and_bounds_cnames() -> None:
    poisoned = {
        "Status": 0,
        "Answer": [
            {"name": "image.example.", "type": 1, "data": "93.184.216.34"},
            {"name": "unrelated.example.", "type": 1, "data": "127.0.0.1"},
        ],
    }
    with pytest.raises(media.MediaError) as caught:
        media._parse_doh_answer(poisoned, "image.example", "A")
    assert caught.value.code == "unsafe_image_url"

    cname_only = {
        "Status": 0,
        "Answer": [
            {"name": "image.example.", "type": 5, "data": "cdn.example."},
        ],
    }
    addresses, next_name, hops = media._parse_doh_answer(
        cname_only, "image.example", "A"
    )
    assert addresses == [] and next_name == "cdn.example" and hops == 1

    too_deep = {
        "Status": 0,
        "Answer": [
            {"name": f"n{index}.example.", "type": 5, "data": f"n{index + 1}.example."}
            for index in range(media._DOH_MAX_CNAME_DEPTH + 1)
        ],
    }
    with pytest.raises(media.MediaError, match="depth"):
        media._parse_doh_answer(too_deep, "n0.example", "A")


def test_doh_resolution_combines_public_a_and_aaaa_under_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deadlines: list[float] = []

    def query(name: str, rr_type: str, deadline: float):
        assert name == "image.example"
        deadlines.append(deadline)
        address = "93.184.216.34" if rr_type == "A" else "2606:4700:4700::1111"
        record_type = 1 if rr_type == "A" else 28
        return {
            "Status": 0,
            "Answer": [{"name": "image.example.", "type": record_type, "data": address}],
        }

    monkeypatch.setattr(media, "_doh_query", query)
    deadline = time.monotonic() + 1
    assert media._resolve_doh_addresses("image.example", deadline) == [
        "93.184.216.34",
        "2606:4700:4700::1111",
    ]
    assert deadlines == [deadline, deadline]


def test_system_dns_timeout_is_fail_closed_and_returns_without_waiting_for_libc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()

    def stuck_getaddrinfo(*_args, **_kwargs):
        release.wait(1)
        return _addrinfo("93.184.216.34")

    monkeypatch.setattr(media.socket, "getaddrinfo", stuck_getaddrinfo)
    started = time.monotonic()
    try:
        with pytest.raises(media.MediaError) as caught:
            media._system_getaddrinfo("slow-resolver.example", 443, started + 0.03)
        assert caught.value.code == "image_fetch_timeout"
        assert time.monotonic() - started < 0.2
        assert media.SYSTEM_DNS_HARD_CANCEL_SUPPORTED is False
    finally:
        release.set()


def test_image_count_and_total_byte_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    too_many = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": f"data:image/png;base64,{base64.b64encode(_PNG).decode()}"}
                for _ in range(media.MAX_IMAGES + 1)
            ],
        }
    ]
    with pytest.raises(media.MediaError) as caught:
        asyncio.run(media.prepare_image_messages(too_many))
    assert caught.value.code == "too_many_images"

    async def fake_load(_url: str, **_kwargs) -> MediaPayload:
        return MediaPayload("image/png", b"abc", "https")

    monkeypatch.setattr(media, "load_image_url", fake_load)
    monkeypatch.setattr(media, "MAX_TOTAL_IMAGE_BYTES", 5)
    two = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": "https://one.example/a.png"},
                {"type": "image_url", "image_url": "https://two.example/b.png"},
            ],
        }
    ]
    with pytest.raises(media.MediaError) as caught:
        asyncio.run(media.prepare_image_messages(two))
    assert caught.value.code == "image_too_large"


@pytest.mark.parametrize("part_type", ["audio", "audio_url", "input_audio", "video", "video_url"])
def test_audio_and_video_parts_are_explicitly_rejected(part_type: str) -> None:
    messages = [{"role": "user", "content": [{"type": part_type, "audio_url": "x"}]}]
    with pytest.raises(media.MediaError) as caught:
        asyncio.run(media.prepare_image_messages(messages))
    assert caught.value.code == "unsupported_content_type"


def test_redirect_is_revalidated_and_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    deadlines: list[float] = []

    def fake_request(url: str, _deadline: float, **_kwargs) -> media._FetchResult:
        calls.append(url)
        deadlines.append(_deadline)
        if len(calls) == 1:
            return media._FetchResult(302, "", b"", "https://cdn.example/image.png")
        return media._FetchResult(200, "image/png", _PNG, None)

    monkeypatch.setattr(media, "_request_once", fake_request)
    payload = media._fetch_https_sync("https://origin.example/start")
    assert payload == MediaPayload("image/png", _PNG, "https")
    assert calls == ["https://origin.example/start", "https://cdn.example/image.png"]
    assert deadlines[0] == deadlines[1]


def test_data_url_size_limit_accepts_boundary_and_rejects_one_byte_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media, "MAX_IMAGE_BYTES", 4)
    exact = "data:image/png;base64," + base64.b64encode(b"1234").decode()
    over = "data:image/png;base64," + base64.b64encode(b"12345").decode()

    assert asyncio.run(media.load_image_url(exact)).data == b"1234"
    with pytest.raises(media.MediaError) as caught:
        asyncio.run(media.load_image_url(over))
    assert caught.value.code == "image_too_large"


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def shutdown(self, _how: int) -> None:
        return None


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        mime_type: str = "image/png",
        content_length: str | None = None,
        on_read=None,
    ) -> None:
        self.status = 200
        self._body = bytearray(body)
        self._mime_type = mime_type
        self._content_length = content_length
        self._on_read = on_read

    def getheader(self, name: str):
        if name == "Content-Type":
            return self._mime_type
        if name == "Content-Length":
            return self._content_length
        return None

    def read(self, amount: int) -> bytes:
        if self._on_read is not None:
            self._on_read()
        take = min(amount, 1) if self._on_read is not None else amount
        chunk = bytes(self._body[:take])
        del self._body[:take]
        return chunk


class _FakeConnection:
    def __init__(self, response: _FakeResponse) -> None:
        self.sock: _FakeSocket | None = None
        self.response = response

    def connect(self) -> None:
        self.sock = _FakeSocket()

    def request(self, *_args, **_kwargs) -> None:
        return None

    def getresponse(self) -> _FakeResponse:
        return self.response

    def close(self) -> None:
        return None


def _install_fake_https(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> list[_FakeConnection]:
    connections: list[_FakeConnection] = []

    def connection_factory(*_args, **_kwargs):
        connection = _FakeConnection(response)
        connections.append(connection)
        return connection

    monkeypatch.setattr(media, "_resolve_public_addresses", lambda *_args: ["93.184.216.34"])
    monkeypatch.setattr(media, "_PinnedHTTPSConnection", connection_factory)
    return connections


def test_pinned_tls_handshake_rearms_remaining_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _FakeSocket()
    raw.close = lambda: None  # type: ignore[attr-defined]
    connect_timeouts: list[float] = []

    def create_connection(_peer, timeout):
        connect_timeouts.append(timeout)
        return raw

    class _Context:
        def wrap_socket(self, value, *, server_hostname):
            assert value is raw
            assert server_hostname == "images.example"
            return value

    clock = iter((0.0, 3.0))
    monkeypatch.setattr(media.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(media.socket, "create_connection", create_connection)
    connection = media._PinnedHTTPSConnection(
        "images.example", "93.184.216.34", 443, 10.0, deadline=10.0
    )
    connection._context = _Context()
    connection.connect()

    assert connect_timeouts == [pytest.approx(10.0)]
    assert raw.timeouts == [pytest.approx(7.0)]


def test_doh_uses_fixed_pinned_peer_and_rejects_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(b"", mime_type="application/dns-json")
    response.status = 302
    created: list[tuple[object, ...]] = []

    def connection_factory(hostname, approved_ip, port, timeout, *, deadline):
        created.append((hostname, approved_ip, port, timeout, deadline))
        return _FakeConnection(response)

    monkeypatch.setattr(media, "_PinnedHTTPSConnection", connection_factory)
    with pytest.raises(media.MediaError, match="redirect"):
        media._doh_query("image.example", "A", time.monotonic() + 1)

    assert len(created) == 1
    hostname, approved_ip, port, timeout, deadline = created[0]
    assert hostname == media._DOH_HOSTNAME
    assert approved_ip == media._DOH_APPROVED_IPS[0]
    assert port == 443 and 0 < timeout <= 1
    assert deadline > time.monotonic()


def test_https_rejects_wrong_mime_before_retaining_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(b"not-an-image", mime_type="text/plain")
    _install_fake_https(monkeypatch, response)

    with pytest.raises(media.MediaError) as caught:
        media._request_once("https://images.example/a", time.monotonic() + 1)
    assert caught.value.code == "unsupported_content_type"


def test_https_read_aborts_at_shared_budget_and_releases_partial_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(b"abcdef", on_read=lambda: None)
    _install_fake_https(monkeypatch, response)
    budget = media._DownloadBudget(5)

    with pytest.raises(media.MediaError) as caught:
        media._request_once(
            "https://images.example/large",
            time.monotonic() + 1,
            budget=budget,
        )
    assert caught.value.code == "image_too_large"
    assert budget.used == 0


def test_each_https_read_uses_decreasing_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]

    def monotonic() -> float:
        return clock[0]

    def slow_read() -> None:
        clock[0] += 0.6

    response = _FakeResponse(b"ab", on_read=slow_read)
    connections = _install_fake_https(monkeypatch, response)
    monkeypatch.setattr(media.time, "monotonic", monotonic)

    with pytest.raises(media.MediaError) as caught:
        media._request_once("https://images.example/slow", 1.0)

    assert caught.value.code == "image_fetch_timeout"
    assert connections[0].sock is not None
    timeouts = connections[0].sock.timeouts
    assert timeouts[0] == pytest.approx(1.0)
    assert any(value == pytest.approx(0.4) for value in timeouts)


def test_async_timeout_leaves_no_long_running_fetch_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished = threading.Event()

    def fake_fetch(_url: str, *, deadline: float, **_kwargs) -> MediaPayload:
        try:
            while time.monotonic() < deadline:
                time.sleep(0.001)
            raise media.MediaError("image fetch timed out", code="image_fetch_timeout")
        finally:
            finished.set()

    monkeypatch.setattr(media, "FETCH_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(media, "_fetch_https_sync", fake_fetch)

    with pytest.raises(media.MediaError) as caught:
        asyncio.run(media.load_image_url("https://images.example/slow"))
    assert caught.value.code == "image_fetch_timeout"
    assert finished.wait(0.2), "cancelled asyncio request left its fetch worker running"


def test_concurrent_fetches_fail_at_shared_total_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)

    def fake_fetch(
        _url: str,
        *,
        deadline: float,
        budget: media._DownloadBudget,
    ) -> MediaPayload:
        barrier.wait(timeout=1)
        budget.reserve(3)
        return MediaPayload("image/png", b"abc", "https")

    monkeypatch.setattr(media, "MAX_TOTAL_IMAGE_BYTES", 5)
    monkeypatch.setattr(media, "_fetch_https_sync", fake_fetch)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": "https://one.example/a.png"},
                {"type": "image_url", "image_url": "https://two.example/b.png"},
            ],
        }
    ]

    with pytest.raises(media.MediaError) as caught:
        asyncio.run(media.prepare_image_messages(messages))
    assert caught.value.code == "image_too_large"


def test_request_logging_redacts_inline_and_signed_media_sources() -> None:
    inline = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    signed = "https://cdn.example/image.png?secret=do-not-log"
    request = ChatCompletionRequest(
        model="unit",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": inline, "detail": "auto"}},
                    {"type": "image_url", "image_url": signed},
                ],
            }
        ],
    )

    payload = _to_payload(request)
    serialized = str(payload)

    assert "describe" in serialized
    assert "detail" in serialized
    assert inline not in serialized
    assert "do-not-log" not in serialized
    assert serialized.count("[redacted media source]") == 2
