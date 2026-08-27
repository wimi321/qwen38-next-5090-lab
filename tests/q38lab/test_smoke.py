from __future__ import annotations

from q38lab.http import SSEEvent, SSEResponse
from q38lab.smoke import run_smoke


class FakeClient:
    def __init__(
        self, stream_text: str = "q38lab-ready", *,
        disabled_reasoning: str | None = None,
        thinking_reasoning: str | None = "2+2 is 4",
        tool_arguments: str = '{"city":"Hangzhou"}',
    ) -> None:
        self.stream_text = stream_text
        self.disabled_reasoning = disabled_reasoning
        self.thinking_reasoning = thinking_reasoning
        self.tool_arguments = tool_arguments

    def get_json(self, path):
        if path == "/health":
            return {"status": "ok"}
        assert path == "/v1/models"
        return {"data": [{"id": "qwen-test"}]}

    def post_json(self, path, payload):
        assert path == "/v1/chat/completions"
        if payload.get("tools"):
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
        return SSEResponse(
            events=(SSEEvent({"choices": [{"delta": {"content": self.stream_text}}]}, 0.2),),
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

    report = run_smoke(FakeClient(tool_arguments='{"city":"Shanghai"}'))
    assert not report.passed
    assert next(check for check in report.checks if check.name == "tool_call").passed is False
