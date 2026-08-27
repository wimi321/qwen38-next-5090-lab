from __future__ import annotations

import io
import urllib.error

import pytest

from q38lab.http import OpenAIHttpClient, Q38HTTPError


def test_client_rejects_credentials_and_ambiguous_urls():
    with pytest.raises(ValueError, match="credentials"):
        OpenAIHttpClient("http://user:secret@127.0.0.1:1919")
    with pytest.raises(ValueError, match="query or fragment"):
        OpenAIHttpClient("http://127.0.0.1:1919?model=x")


def test_json_and_multiline_sse_are_parsed_strictly():
    responses = iter([
        io.BytesIO(b'{"status":"ok"}'),
        io.BytesIO(
            b'data: {"choices":[\n'
            b'data: {"delta":{"content":"ok"}}]}\n\n'
            b'data: [DONE]\n\n'
        ),
    ])
    client = OpenAIHttpClient(
        "http://127.0.0.1:1919", urlopen=lambda *args, **kwargs: next(responses),
        monotonic=iter([0.0, 0.1, 0.2]).__next__,
    )
    assert client.get_json("/health") == {"status": "ok"}
    response = client.post_sse("/v1/chat/completions", {"stream": True})
    assert response.done
    assert response.events[0].payload["choices"][0]["delta"]["content"] == "ok"


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"data: not-json\n\ndata: [DONE]\n\n", "invalid SSE JSON"),
        (b'data: {"choices":[]}\n\n', r"without \[DONE\]"),
        (b"data: \xff\n\n", "invalid UTF-8"),
    ],
)
def test_sse_rejects_malformed_or_incomplete_streams(payload, message):
    client = OpenAIHttpClient(
        "http://127.0.0.1:1919", urlopen=lambda *args, **kwargs: io.BytesIO(payload),
    )
    with pytest.raises(Q38HTTPError, match=message):
        client.post_sse("/v1/chat/completions", {"stream": True})


def test_transport_errors_are_redacted_to_bounded_http_details():
    body = io.BytesIO(b"failure " + b"x" * 1000)
    error = urllib.error.HTTPError("http://local", 500, "boom", {}, body)
    client = OpenAIHttpClient(
        "http://127.0.0.1:1919", urlopen=lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    with pytest.raises(Q38HTTPError) as caught:
        client.get_json("/health")
    assert len(str(caught.value)) < 600
