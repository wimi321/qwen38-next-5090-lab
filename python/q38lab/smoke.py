"""OpenAI API lifecycle, streaming, reasoning and tool-call smoke checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .http import SSEResponse


class SmokeClient(Protocol):
    def get_json(self, path: str) -> dict[str, Any]: ...
    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...
    def post_sse(self, path: str, payload: dict[str, Any]) -> SSEResponse: ...


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SmokeReport:
    schema_version: int
    model: str | None
    passed: bool
    checks: tuple[SmokeCheck, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def _first_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no assistant message")
    return message


def _stream_text(response: SSEResponse) -> str:
    pieces: list[str] = []
    for event in response.events:
        choices = event.payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                pieces.append(delta["content"])
    return "".join(pieces)


def run_smoke(client: SmokeClient, *, requested_model: str | None = None) -> SmokeReport:
    checks: list[SmokeCheck] = []
    model: str | None = requested_model

    try:
        health = client.get_json("/health")
        status = health.get("status")
        if status != "ok":
            raise ValueError(f"server status is {status!r}, not 'ok'")
        checks.append(SmokeCheck("health", True, "server reports status=ok"))
    except Exception as exc:  # report all externally visible smoke failures uniformly
        checks.append(SmokeCheck("health", False, str(exc)))
        return SmokeReport(1, model, False, tuple(checks))

    try:
        models = client.get_json("/v1/models")
        data = models.get("data")
        ids = [item.get("id") for item in data or [] if isinstance(item, dict)]
        ids = [item for item in ids if isinstance(item, str)]
        if not ids:
            raise ValueError("/v1/models returned no model ids")
        if model is None:
            model = ids[0]
        if model not in ids:
            raise ValueError(f"requested model {model!r} is not served; available: {ids}")
        checks.append(SmokeCheck("models", True, f"using {model}"))
    except Exception as exc:
        checks.append(SmokeCheck("models", False, str(exc)))
        return SmokeReport(1, model, False, tuple(checks))

    assert model is not None
    base_request = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: q38lab-ready"}],
        "max_tokens": 32,
        "temperature": 0.0,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    nonstream_text: str | None = None
    try:
        response = client.post_json("/v1/chat/completions", {**base_request, "stream": False})
        message = _first_message(response)
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("non-streaming response has empty content")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            raise ValueError("thinking-disabled response unexpectedly contains reasoning")
        nonstream_text = content
        checks.append(SmokeCheck("non_streaming", True, f"received {len(content)} characters"))
    except Exception as exc:
        checks.append(SmokeCheck("non_streaming", False, str(exc)))

    try:
        response = client.post_sse(
            "/v1/chat/completions",
            {
                **base_request,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        content = _stream_text(response)
        if not content:
            raise ValueError("streaming response has empty content")
        if nonstream_text is not None and content != nonstream_text:
            raise ValueError(
                "streaming and non-streaming greedy output differ: "
                f"{content!r} != {nonstream_text!r}"
            )
        checks.append(SmokeCheck("streaming", True, f"received {len(content)} characters"))
    except Exception as exc:
        checks.append(SmokeCheck("streaming", False, str(exc)))

    try:
        response = client.post_json(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
                "max_tokens": 128,
                "temperature": 0.0,
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        message = _first_message(response)
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("thinking-enabled response has no reasoning_content")
        checks.append(SmokeCheck("thinking", True, "thinking reasoning_content received"))
    except Exception as exc:
        checks.append(SmokeCheck("thinking", False, str(exc)))

    try:
        response = client.post_json(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in Shanghai?",
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.0,
                "top_k": 1,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Return weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "reasoning_effort": "none",
            },
        )
        tool_calls = _first_message(response).get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError("forced tool request returned no tool_calls")
        first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        function = first.get("function") if isinstance(first, dict) else {}
        if not isinstance(function, dict) or function.get("name") != "get_weather":
            raise ValueError("tool response did not call get_weather")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict) or arguments.get("city") != "Shanghai":
            raise ValueError("tool arguments do not satisfy the required Shanghai city schema")
        checks.append(SmokeCheck("tool_call", True, "get_weather(Shanghai) parsed"))
    except Exception as exc:
        checks.append(SmokeCheck("tool_call", False, str(exc)))

    return SmokeReport(
        schema_version=1,
        model=model,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def format_smoke_report(report: SmokeReport) -> str:
    lines = [f"q38lab smoke: {'PASS' if report.passed else 'FAIL'}"]
    for check in report.checks:
        lines.append(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    return "\n".join(lines)
