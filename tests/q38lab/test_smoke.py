from __future__ import annotations

from q38lab.http import Q38HTTPError, SSEEvent, SSEResponse
from q38lab.smoke import run_smoke


class FakeClient:
    def __init__(
        self, stream_text: str = "q38lab-ready", *,
        disabled_reasoning: str | None = None,
        thinking_reasoning: str | None = "2+2 is 4",
        tool_arguments: str = '{"city":"Shanghai"}',
    ) -> None:
        self.stream_text = stream_text
        self.disabled_reasoning = disabled_reasoning
        self.thinking_reasoning = thinking_reasoning
        self.tool_arguments = tool_arguments
        self.image_payloads = []

    def get_json(self, path):
        if path == "/health":
            return {"status": "ok"}
        assert path == "/v1/models"
        return {"data": [{"id": "qwen-test"}]}

    def post_json(self, path, payload):
        assert path == "/v1/chat/completions"
        content = (payload.get("messages") or [{}])[0].get("content")
        if isinstance(content, list):
            self.image_payloads.append(payload)
            for part in content:
                if part.get("type") in {"input_audio", "video_url"}:
                    raise Q38HTTPError("unsupported_content_type")
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                if isinstance(image_url, str) and image_url.startswith(
                    ("http://", "file:", "https://127.", "https://169.254.")
                ):
                    raise Q38HTTPError("unsafe_image_url")
            message = {"content": "image-ok"}
        elif payload.get("tools"):
            assert payload.get("reasoning_effort") == "none"
            assert payload.get("tool_choice") == "required"
            message = {
                "content": None,
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": self.tool_arguments},
                }],
            }
        elif payload.get("chat_template_kwargs", {}).get("enable_thinking"):
            message = {"content": "4", "reasoning_content": self.thinking_reasoning}
        else:
            message = {
                "content": "q38lab-ready",
                "reasoning_content": self.disabled_reasoning,
            }
        return {"choices": [{"message": message}]}

    def post_sse(self, path, payload):
        content = (payload.get("messages") or [{}])[0].get("content")
        stream_text = "image-stream" if isinstance(content, list) else self.stream_text
        return SSEResponse(
            events=(SSEEvent({"choices": [{"delta": {"content": stream_text}}]}, 0.2),),
            done=True,
            elapsed_seconds=0.4,
        )


def test_smoke_covers_all_api_modes():
    report = run_smoke(FakeClient())
    assert report.passed
    assert [check.name for check in report.checks] == [
        "health", "models", "non_streaming", "streaming", "thinking", "tool_call"
    ]


def test_smoke_rejects_stream_nonstream_drift():
    report = run_smoke(FakeClient(stream_text="different"))
    assert not report.passed
    assert next(check for check in report.checks if check.name == "streaming").passed is False


def test_smoke_requires_real_thinking_toggle_and_schema_valid_tool_arguments():
    report = run_smoke(FakeClient(disabled_reasoning="should be absent"))
    assert not report.passed
    assert next(check for check in report.checks if check.name == "non_streaming").passed is False

    report = run_smoke(FakeClient(thinking_reasoning=None))
    assert not report.passed
    assert next(check for check in report.checks if check.name == "thinking").passed is False

    report = run_smoke(FakeClient(tool_arguments='{"city":"Hangzhou"}'))
    assert not report.passed
    assert next(check for check in report.checks if check.name == "tool_call").passed is False


def test_parameterized_image_smoke_covers_data_https_four_stream_and_rejections():
    client = FakeClient()

    report = run_smoke(
        client,
        include_images=True,
        https_image_url="https://images.example/public.png",
    )

    assert report.passed
    image_checks = [check.name for check in report.checks if check.name.startswith("image_")]
    assert image_checks == [
        "image_data_url",
        "image_streaming",
        "image_four",
        "image_https",
        "image_security",
        "image_only_modalities",
    ]
    counts = [
        sum(1 for part in payload["messages"][0]["content"] if part["type"] == "image_url")
        for payload in client.image_payloads
    ]
    assert 1 in counts and 4 in counts
