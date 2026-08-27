#!/usr/bin/env python3
"""Create, validate, checksum, and render sanitized RTX 5090 evidence bundles.

The release workflow accepts only a bundle whose ``summary.json`` says
``status=verified`` and which passes every hardware gate below.  The tool uses
only the Python standard library so hosted CI can validate metadata without the
checkpoint, CUDA, or the FreeToken runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
BEGIN_MARKER = "<!-- BEGIN GENERATED BENCHMARK SUMMARY -->"
END_MARKER = "<!-- END GENERATED BENCHMARK SUMMARY -->"
REQUIRED_FILES = (
    "environment.json",
    "resolved-config.json",
    "summary.json",
    "requests.jsonl",
    "latency.csv",
    "resource-samples.csv",
    "pytest.txt",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RELEASE_ONLY_ALLOWED_FILES = set(REQUIRED_FILES) | {"SHA256SUMS"}
RUNTIME_PATHS = (
    "python",
    "profiles",
    "pyproject.toml",
    "setup.py",
    "scripts/release",
)
ALLOWED_POST_RUNTIME_FILES = {
    "README.md",
    "README.zh-CN.md",
    "CHANGELOG.md",
}
ALLOWED_POST_RUNTIME_PREFIX = "results/rtx5090-"
EXPECTED_MODEL = {
    "repository": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    "manifest_sha256": "6cc22b628ca575785e5dfdcab3c7056e79a7eac798969a145341ed1530c2a3a8",
    "file_count": 419,
    "total_bytes": 135_253_622_894,
}
EXPECTED_SETTINGS = {
    "host": "127.0.0.1",
    "port": 1919,
    "tp_size": 1,
    "max_running_requests": 1,
    "max_seq_len": 8192,
    "max_prefill_length": 8192,
    "num_tokens": 8192,
    "memory_ratio": 0.89,
    "cache_type": "naive",
    "attention_backend": "qsa_triton",
    "cuda_graph": False,
    "moe_backend": "offload",
    "moe_cache_auto": True,
    "moe_cpu_layers": None,
    "nvfp4_backend": "auto",
}
EXPECTED_EXECUTION = {
    "attention_backend": "qsa_triton",
    "cache_type": "naive",
    "context_tokens": 8192,
    "cuda_graph": False,
    "quantization": "W4A16 compatibility",
    "text_only": True,
    "tp_size": 1,
}
PRIVATE_PATTERNS = (
    re.compile(r"(?i)[a-z]:[\\/]users[\\/][^\\/\s]+"),
    re.compile(r"/home/[^/\s]+"),
    re.compile(r"/mnt/[a-z]/", re.IGNORECASE),
    re.compile(r"(?i)\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{12,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
)
IPV4_RE = re.compile(r"(?<![\w.-])(?:\d{1,3}\.){3}\d{1,3}(?![\w.-])")
IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
HOSTNAME_FIELD_RE = re.compile(r"(?i)(?:^|[,{\s\"'])hostname(?:$|[,}:=\s\"'])")


class EvidenceError(ValueError):
    """An evidence bundle is incomplete, unsafe, or fails release gates."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{path}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path}: top level must be an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _walk_strings(value: Any, prefix: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{prefix}[{index}]")


def assert_sanitized(value: Any, source: str) -> None:
    forbidden_keys = {"hostname", "username", "user", "home", "model_path", "command_line"}
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in forbidden_keys:
                raise EvidenceError(f"{source}: private key is not allowed: {key}")
            assert_sanitized(item, source)
    elif isinstance(value, list):
        for item in value:
            assert_sanitized(item, source)
    for path, text in _walk_strings(value):
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                raise EvidenceError(f"{source}: private-looking value at {path}")


def assert_text_sanitized(text: str, source: str) -> None:
    """Scan every tracked text artifact, not only selected JSON fields."""

    for pattern in PRIVATE_PATTERNS:
        match = pattern.search(text)
        if match:
            raise EvidenceError(f"{source}: private-looking text is not allowed")
    if HOSTNAME_FIELD_RE.search(text):
        raise EvidenceError(f"{source}: hostname fields are not allowed")
    for pattern in (IPV4_RE, IPV6_RE):
        for match in pattern.finditer(text):
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            if not address.is_loopback:
                raise EvidenceError(f"{source}: non-loopback IP address is not allowed")


def _git_output(root: Path, arguments: list[str], *, binary: bool = False) -> bytes | str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, stderr=subprocess.STDOUT,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", b"" if binary else "")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise EvidenceError(f"git {' '.join(arguments)} failed: {str(detail).strip()}") from exc


def runtime_tree_sha256(root: Path, commit: str = "HEAD") -> str:
    """Hash canonical Git tree entries for the release runtime scope."""

    if not COMMIT_RE.fullmatch(str(_git_output(root, ["rev-parse", f"{commit}^{{commit}}"])).strip()):
        raise EvidenceError(f"runtime commit does not resolve to a 40-hex commit: {commit}")
    raw = _git_output(
        root,
        ["ls-tree", "-r", "-z", "--full-tree", commit, "--", *RUNTIME_PATHS],
        binary=True,
    )
    assert isinstance(raw, bytes)
    if not raw:
        raise EvidenceError("runtime tree scope is empty")
    return hashlib.sha256(raw).hexdigest()


def validate_tag_binding(
    summary: dict[str, Any], *, tag_commit: str, repo_root: Path
) -> None:
    """Bind evidence from runtime commit C to evidence-only tag commit E."""

    tag = str(_git_output(repo_root, ["rev-parse", f"{tag_commit}^{{commit}}"])).strip()
    if not COMMIT_RE.fullmatch(tag):
        raise EvidenceError("tag commit must resolve to an exact 40-hex commit")
    runtime = str(summary.get("source", {}).get("validated_runtime_commit", ""))
    if not COMMIT_RE.fullmatch(runtime):
        raise EvidenceError("validated runtime commit must be exact 40-hex")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", runtime, tag], cwd=repo_root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise EvidenceError("validated runtime commit is not an ancestor of the release tag")
    changed = str(_git_output(repo_root, ["diff", "--name-only", f"{runtime}..{tag}"])).splitlines()
    forbidden = [
        path for path in changed
        if path not in ALLOWED_POST_RUNTIME_FILES
        and not (
            path.startswith(ALLOWED_POST_RUNTIME_PREFIX)
            and "/" in path[len(ALLOWED_POST_RUNTIME_PREFIX):]
        )
    ]
    if forbidden:
        raise EvidenceError(
            "release tag changes runtime files after validation: " + ", ".join(forbidden)
        )
    recorded_tree = str(summary.get("source", {}).get("runtime_tree_sha256", ""))
    runtime_tree = runtime_tree_sha256(repo_root, runtime)
    tag_tree = runtime_tree_sha256(repo_root, tag)
    if recorded_tree != runtime_tree or tag_tree != runtime_tree:
        raise EvidenceError("runtime tree digest does not match evidence/runtime/tag")


def _required(mapping: dict[str, Any], path: str, expected: type | tuple[type, ...]) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise EvidenceError(f"summary.json: missing {path}")
        current = current[part]
    if isinstance(current, bool) and expected in (int, float, (int, float)):
        raise EvidenceError(f"summary.json: {path} has wrong type")
    if not isinstance(current, expected):
        raise EvidenceError(f"summary.json: {path} has wrong type")
    return current


def _validate_summary(
    summary: dict[str, Any], *, release: bool, expected_commit: str | None = None
) -> None:
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"summary.json: schema_version must be {SCHEMA_VERSION}")
    assert_sanitized(summary, "summary.json")
    _required(summary, "run_id", str)
    _required(summary, "measured_at", str)
    runtime_commit = _required(summary, "source.validated_runtime_commit", str)
    _required(summary, "source.upstream_base", str)
    _required(summary, "model.repository", str)
    revision = _required(summary, "model.revision", str)
    if len(revision) != 40 or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise EvidenceError("summary.json: model.revision must be a pinned 40-hex commit")
    manifest = _required(summary, "model.manifest_sha256", str)
    if not SHA256_RE.fullmatch(manifest):
        raise EvidenceError("summary.json: model.manifest_sha256 must be 64 lowercase hex")
    _required(summary, "execution.quantization", str)
    _required(summary, "execution.context_tokens", int)
    _required(summary, "execution.text_only", bool)
    _required(summary, "execution.tp_size", int)
    _required(summary, "gates.pytest.passed", int)
    _required(summary, "gates.pytest.failed", int)
    prompt_cases = _required(summary, "gates.prompt_cases", list)
    _required(summary, "gates.steady_decode.observed_tokens", int)
    _required(summary, "gates.api.stream_nonstream_match", bool)
    _required(summary, "gates.api.thinking_none", bool)
    _required(summary, "gates.api.thinking_high", bool)
    _required(summary, "gates.api.tool_call", bool)
    _required(summary, "gates.stability.succeeded", int)
    _required(summary, "gates.stability.attempted", int)
    _required(summary, "gates.continuous_run_seconds", (int, float))
    _required(summary, "resources.peak_vram_mib", (int, float))
    _required(summary, "resources.peak_wsl_rss_kib", (int, float))
    _required(summary, "resources.wsl_rss_source", str)
    _required(summary, "resources.wsl_swap_kib", (int, float))
    _required(summary, "resources.windows_pagefile_note", str)
    _required(summary, "caveats", list)

    if summary.get("status") != "synthetic" and not COMMIT_RE.fullmatch(runtime_commit):
        raise EvidenceError("summary.json: validated_runtime_commit must be exact 40-hex")
    if expected_commit is not None:
        if not COMMIT_RE.fullmatch(expected_commit):
            raise EvidenceError("--expected-commit must be exact 40-hex")
        if runtime_commit != expected_commit:
            raise EvidenceError(
                f"validated runtime commit {runtime_commit} != expected {expected_commit}"
            )

    if not release:
        return
    if summary.get("status") != "verified":
        raise EvidenceError("release evidence must have status=verified")
    if summary.get("source", {}).get("release_compatible") is not True:
        raise EvidenceError("release evidence needs a human-reviewed source.release_compatible=true")
    if not COMMIT_RE.fullmatch(runtime_commit):
        raise EvidenceError("release evidence runtime commit must be exact 40-hex")
    runtime_tree = _required(summary, "source.runtime_tree_sha256", str)
    if not SHA256_RE.fullmatch(runtime_tree):
        raise EvidenceError("release evidence runtime_tree_sha256 must be 64 lowercase hex")
    if summary["execution"] != EXPECTED_EXECUTION:
        raise EvidenceError("release execution contract does not match the v0.1 preview")
    for key, expected in EXPECTED_MODEL.items():
        if summary.get("model", {}).get(key) != expected:
            raise EvidenceError(f"release model.{key} does not match the pinned checkpoint")
    pytest_gate = summary["gates"]["pytest"]
    if pytest_gate["passed"] < 1454 or pytest_gate["failed"] != 0:
        raise EvidenceError("pytest release gate requires >=1454 passed and zero failed")
    targets = {int(case.get("rendered_prompt_tokens", -1)) for case in prompt_cases}
    if not {13, 128, 2048, 8176}.issubset(targets):
        raise EvidenceError("prompt gates must include rendered lengths 13, 128, 2048, and 8176")
    one_token_cases = [
        case for case in prompt_cases
        if case.get("content_prompt_tokens") == 1
        and case.get("rendered_prompt_tokens") == 13
    ]
    if len(one_token_cases) != 1:
        raise EvidenceError("prompt gates must include content=1/rendered=13 exactly once")
    gates = summary["gates"]
    if gates["steady_decode"]["observed_tokens"] < 256:
        raise EvidenceError("steady decode gate requires at least 256 observed tokens")
    if not all(gates["api"].values()):
        raise EvidenceError("all stream/thinking/tool API gates must pass")
    stability = gates["stability"]
    if stability["attempted"] < 100 or stability["succeeded"] != stability["attempted"]:
        raise EvidenceError("stability gate requires at least 100/100 successful requests")
    if gates["continuous_run_seconds"] < 1800:
        raise EvidenceError("continuous run gate requires at least 1800 seconds")
    resources = summary["resources"]
    if resources["peak_vram_mib"] >= 31 * 1024:
        raise EvidenceError("peak VRAM must be below 31 GiB")
    if resources["peak_wsl_rss_kib"] >= 105 * 1024 * 1024:
        raise EvidenceError("peak WSL RSS must be below 105 GiB")
    if resources["wsl_swap_kib"] != 0:
        raise EvidenceError("WSL swap must be zero")
    if resources["wsl_rss_source"] not in {
        "Windows vmmemWSL working set",
        "WSL MemTotal-MemAvailable fallback",
    }:
        raise EvidenceError("release evidence must identify a supported WSL RSS source")


def _validate_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not required_columns.issubset(fields):
                missing = ", ".join(sorted(required_columns - fields))
                raise EvidenceError(f"{path}: missing CSV columns: {missing}")
            return list(reader)
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot read CSV: {exc}") from exc


def _finite_float(value: Any, source: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{source}: expected a number") from exc
    if not math.isfinite(result):
        raise EvidenceError(f"{source}: number must be finite")
    return result


def _numeric_version(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(value or "")))


def _integer(value: Any, source: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{source}: expected an integer") from exc
    if str(result) != str(value).strip():
        raise EvidenceError(f"{source}: expected an integer")
    return result


def _close(left: float, right: float, *, tolerance: float = 0.002) -> bool:
    return abs(left - right) <= tolerance


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise EvidenceError("cannot calculate percentile of an empty sequence")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _parse_pytest_counts(text: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "deselected": 0}
    for key in counts:
        matches = re.findall(rf"(?<!\d)(\d+)\s+{key}\b", text)
        if matches:
            counts[key] = int(matches[-1])
    return counts


def _read_requests(path: Path, *, release: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "case", "iteration", "warmup", "stream", "success", "http_status",
        "prompt_tokens", "completion_tokens", "ttft_ms", "total_ms",
        "content_tokens", "started_elapsed_s", "finished_elapsed_s", "recorded_at",
        "proof",
    }
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"requests.jsonl:{number}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise EvidenceError(f"requests.jsonl:{number}: expected object")
        forbidden = {"prompt", "messages", "content", "output", "response", "reasoning_content"}
        if forbidden.intersection(item):
            raise EvidenceError(f"requests.jsonl:{number}: request/response text must not be tracked")
        assert_sanitized(item, f"requests.jsonl:{number}")
        if release:
            missing = required - set(item)
            if missing:
                raise EvidenceError(f"requests.jsonl:{number}: missing fields {sorted(missing)}")
            if not isinstance(item["case"], str) or not item["case"]:
                raise EvidenceError(f"requests.jsonl:{number}: case must be non-empty")
            for key in ("iteration", "http_status", "prompt_tokens", "completion_tokens"):
                if isinstance(item[key], bool) or not isinstance(item[key], int):
                    raise EvidenceError(f"requests.jsonl:{number}: {key} must be an integer")
            for key in ("warmup", "stream", "success"):
                if not isinstance(item[key], bool):
                    raise EvidenceError(f"requests.jsonl:{number}: {key} must be boolean")
            if not isinstance(item["proof"], dict):
                raise EvidenceError(f"requests.jsonl:{number}: proof must be an object")
            total = _finite_float(item["total_ms"], f"requests.jsonl:{number}.total_ms")
            started = _finite_float(item["started_elapsed_s"], f"requests.jsonl:{number}.started_elapsed_s")
            finished = _finite_float(item["finished_elapsed_s"], f"requests.jsonl:{number}.finished_elapsed_s")
            if total < 0 or started < 0 or finished < started:
                raise EvidenceError(f"requests.jsonl:{number}: invalid request timing")
            if not _close((finished - started) * 1000, total, tolerance=5.0):
                raise EvidenceError(f"requests.jsonl:{number}: elapsed timing disagrees with total_ms")
            try:
                datetime.fromisoformat(str(item["recorded_at"]).replace("Z", "+00:00"))
            except ValueError as exc:
                raise EvidenceError(f"requests.jsonl:{number}: invalid recorded_at") from exc
        rows.append(item)
    if release and not rows:
        raise EvidenceError("requests.jsonl: release evidence must contain requests")
    return rows


def _resource_rows(path: Path, *, release: bool) -> list[dict[str, Any]]:
    raw = _validate_csv(
        path,
        {
            "elapsed_s", "gpu_memory_mib", "wsl_rss_kib", "wsl_rss_source",
            "wsl_swap_kib", "minor_faults", "major_faults", "fault_processes",
        },
    )
    if not release:
        return []
    if not raw:
        raise EvidenceError("resource-samples.csv: telemetry is empty")
    rows: list[dict[str, Any]] = []
    for number, item in enumerate(raw, 2):
        converted: dict[str, Any] = {}
        for key in (
            "elapsed_s", "gpu_memory_mib", "wsl_rss_kib", "wsl_swap_kib",
            "minor_faults", "major_faults", "fault_processes",
        ):
            if item.get(key, "").strip() == "":
                raise EvidenceError(f"resource-samples.csv:{number}: missing telemetry {key}")
            converted[key] = _finite_float(item[key], f"resource-samples.csv:{number}.{key}")
            if converted[key] < 0:
                raise EvidenceError(f"resource-samples.csv:{number}: {key} must be non-negative")
        source = item.get("wsl_rss_source", "").strip()
        if source not in {
            "Windows vmmemWSL working set",
            "WSL MemTotal-MemAvailable fallback",
        }:
            raise EvidenceError(f"resource-samples.csv:{number}: invalid WSL RSS source")
        converted["wsl_rss_source"] = source
        if converted["gpu_memory_mib"] <= 0 or converted["wsl_rss_kib"] <= 0:
            raise EvidenceError(f"resource-samples.csv:{number}: GPU and WSL RSS must be positive")
        if converted["minor_faults"] <= 0 or converted["fault_processes"] < 1:
            raise EvidenceError(
                f"resource-samples.csv:{number}: page-fault telemetry lacks a live process"
            )
        rows.append(converted)
    elapsed = [row["elapsed_s"] for row in rows]
    if any(right <= left for left, right in zip(elapsed, elapsed[1:])):
        raise EvidenceError("resource-samples.csv: elapsed_s must increase strictly")
    return rows


def detect_monotonic_rss_leak(
    resources: list[dict[str, Any]], soak_start: float, soak_end: float
) -> bool:
    """Detect sustained >64 MiB growth over ten soak windows."""

    duration = soak_end - soak_start
    if duration <= 0:
        return True
    medians: list[float] = []
    for index in range(10):
        left = soak_start + duration * index / 10
        right = soak_start + duration * (index + 1) / 10
        values = [
            row["wsl_rss_kib"] for row in resources
            if left <= row["elapsed_s"] <= right
        ]
        if not values:
            return True
        medians.append(float(sorted(values)[len(values) // 2]))
    nearly_monotonic = all(left <= right + 4096 for left, right in zip(medians, medians[1:]))
    return nearly_monotonic and medians[-1] - medians[0] > 64 * 1024


def _crosscheck_release_raw(
    directory: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    requests: list[dict[str, Any]],
    latency_rows: list[dict[str, str]],
    resources: list[dict[str, Any]],
) -> None:
    """Recompute every objective gate from raw records and reject drift."""

    settings = config.get("settings")
    if settings != EXPECTED_SETTINGS:
        raise EvidenceError("resolved-config.json does not match the v0.1 preview profile")
    if config.get("profile") != "rtx5090-wsl2":
        raise EvidenceError("resolved-config.json profile must be rtx5090-wsl2")
    if config.get("served_model_name") != "qwen3.8-flash-next-nvfp4":
        raise EvidenceError("resolved-config.json served model name is incorrect")

    pytest_text = (directory / "pytest.txt").read_text(encoding="utf-8")
    pytest_counts = _parse_pytest_counts(pytest_text)
    if pytest_counts != summary["gates"]["pytest"]:
        raise EvidenceError("summary pytest counts do not match pytest.txt")
    if pytest_counts["passed"] < 1454 or pytest_counts["failed"] != 0:
        raise EvidenceError("pytest.txt does not prove >=1454 passed and zero failed")

    keys_seen: set[tuple[str, int]] = set()
    for number, item in enumerate(requests, 1):
        key = (item["case"], item["iteration"])
        if key in keys_seen:
            raise EvidenceError(f"requests.jsonl:{number}: duplicate case/iteration {key}")
        keys_seen.add(key)
        if item["http_status"] != 200 and item["success"]:
            raise EvidenceError(f"requests.jsonl:{number}: non-200 request cannot be successful")

    measured_requests = [item for item in requests if not item["warmup"]]
    if len(latency_rows) != len(measured_requests):
        raise EvidenceError("latency.csv row count does not match non-warmup requests")
    latency_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for number, item in enumerate(latency_rows, 2):
        key = (item["case"], _integer(item["iteration"], f"latency.csv:{number}.iteration"))
        if key in latency_by_key:
            raise EvidenceError(f"latency.csv:{number}: duplicate case/iteration {key}")
        latency_by_key[key] = item
    for item in measured_requests:
        key = (item["case"], item["iteration"])
        latency = latency_by_key.get(key)
        if latency is None:
            raise EvidenceError(f"latency.csv is missing request {key}")
        for field in ("prompt_tokens", "completion_tokens"):
            if _integer(latency[field], f"latency.csv {key}.{field}") != item[field]:
                raise EvidenceError(f"latency.csv disagrees with requests.jsonl for {key}.{field}")
        for field in ("ttft_ms", "total_ms"):
            raw_value = item[field]
            csv_value = latency.get(field, "")
            if raw_value is None:
                if csv_value.strip() not in {"", "None"}:
                    raise EvidenceError(f"latency.csv disagrees with requests.jsonl for {key}.{field}")
            elif not _close(_finite_float(csv_value, f"latency.csv {key}.{field}"), float(raw_value)):
                raise EvidenceError(f"latency.csv disagrees with requests.jsonl for {key}.{field}")

    def api_pair(case: str) -> list[dict[str, Any]]:
        rows = [item for item in requests if item["case"] == case and not item["warmup"]]
        if not (
            len(rows) == 2
            and {bool(row["stream"]) for row in rows} == {False, True}
            and all(row["success"] and row["http_status"] == 200 for row in rows)
        ):
            return []
        return rows

    parity_rows = api_pair("stream-parity")
    parity_hashes = [row["proof"].get("text_sha256") for row in parity_rows]
    parity_ok = bool(
        parity_rows
        and all(row["proof"].get("text_present") is True for row in parity_rows)
        and all(isinstance(value, str) and SHA256_RE.fullmatch(value) for value in parity_hashes)
        and len(set(parity_hashes)) == 1
    )

    thinking_none_rows = api_pair("thinking-none")
    thinking_none_ok = bool(
        thinking_none_rows
        and all(
            row["proof"].get("visible_text_present") is True
            and row["proof"].get("reasoning_present") is False
            for row in thinking_none_rows
        )
    )
    thinking_high_rows = api_pair("thinking-high")
    thinking_high_ok = bool(
        thinking_high_rows
        and all(
            row["proof"].get("visible_text_present") is True
            and row["proof"].get("reasoning_present") is True
            for row in thinking_high_rows
        )
    )

    tool_rows = api_pair("tool-call")
    tool_hashes = [row["proof"].get("arguments_sha256") for row in tool_rows]
    tool_cities = [str(row["proof"].get("tool_city") or "").casefold() for row in tool_rows]
    tool_ok = bool(
        tool_rows
        and all(row["proof"].get("tool_name") == "get_weather" for row in tool_rows)
        and all(city == "shanghai" or city == "上海" for city in tool_cities)
        and len(set(tool_cities)) == 1
        and all(isinstance(value, str) and SHA256_RE.fullmatch(value) for value in tool_hashes)
        and len(set(tool_hashes)) == 1
    )
    recomputed_api = {
        "stream_nonstream_match": parity_ok,
        "thinking_none": thinking_none_ok,
        "thinking_high": thinking_high_ok,
        "tool_call": tool_ok,
    }
    if recomputed_api != summary["gates"]["api"] or not all(recomputed_api.values()):
        raise EvidenceError("summary API gates do not match successful raw request pairs")

    prompt_cases: list[dict[str, Any]] = []
    for target in (13, 128, 2048, 8176):
        case = f"prompt-{target}"
        warmups = [item for item in requests if item["case"] == case and item["warmup"]]
        measured = [item for item in requests if item["case"] == case and not item["warmup"]]
        if len(warmups) < 3 or len(measured) < 10:
            raise EvidenceError(f"{case} requires at least 3 warmups and 10 measurements")
        if not all(
            item["success"] and item["http_status"] == 200
            and item["prompt_tokens"] == target and item["stream"]
            for item in warmups + measured
        ):
            raise EvidenceError(f"{case} raw requests do not prove the target")
        if target == 13 and any(item.get("content_tokens") != 1 for item in warmups + measured):
            raise EvidenceError("prompt-13 must prove one content token and 13 rendered tokens")
        completion = [int(item["completion_tokens"]) for item in measured]
        content_lengths = [int(item["content_tokens"]) for item in measured]
        if len(set(content_lengths)) != 1:
            raise EvidenceError(f"{case} content token counts are inconsistent")
        ttft = [_finite_float(item["ttft_ms"], f"{case}.ttft_ms") for item in measured]
        total = [_finite_float(item["total_ms"], f"{case}.total_ms") for item in measured]
        prompt_cases.append({
            "content_prompt_tokens": content_lengths[0],
            "rendered_prompt_tokens": target,
            "completion_tokens": int(statistics.median(completion)),
            "ttft_p50_ms": round(_percentile(ttft, 0.50), 3),
            "ttft_p95_ms": round(_percentile(ttft, 0.95), 3),
            "total_p50_ms": round(_percentile(total, 0.50), 3),
            "total_p95_ms": round(_percentile(total, 0.95), 3),
        })
    if prompt_cases != summary["gates"]["prompt_cases"]:
        raise EvidenceError("summary prompt cases do not match raw request measurements")

    decode = [item for item in requests if item["case"] == "steady-decode" and not item["warmup"]]
    if len(decode) != 1 or not decode[0]["success"] or not decode[0]["stream"]:
        raise EvidenceError("raw evidence must contain one successful steady-decode stream")
    steady_summary = summary["gates"]["steady_decode"]
    if not 256 <= int(steady_summary.get("requested_tokens", 0)) <= 512:
        raise EvidenceError("steady decode requested_tokens must be in [256, 512]")
    if decode[0]["completion_tokens"] != steady_summary.get("observed_tokens"):
        raise EvidenceError("steady decode summary does not match the raw request")
    if decode[0]["completion_tokens"] < 256:
        raise EvidenceError("steady decode raw request observed fewer than 256 tokens")
    decode_time_ms = float(decode[0]["total_ms"]) - float(decode[0]["ttft_ms"])
    if decode_time_ms <= 0:
        raise EvidenceError("steady decode timing must be longer than TTFT")
    decode_tps = round((decode[0]["completion_tokens"] - 1) / (decode_time_ms / 1000), 3)
    if not _close(float(steady_summary.get("decode_tokens_per_second", -1)), decode_tps):
        raise EvidenceError("steady decode throughput does not match the raw request")

    stability = [item for item in requests if item["case"] == "stability" and not item["warmup"]]
    succeeded = sum(bool(item["success"] and item["http_status"] == 200) for item in stability)
    stability_summary = summary["gates"]["stability"]
    if stability_summary.get("attempted") != len(stability) or stability_summary.get("succeeded") != succeeded:
        raise EvidenceError("stability summary does not match raw requests")
    if len(stability) < 100 or succeeded != len(stability):
        raise EvidenceError("raw stability gate requires at least 100/100 successes")

    soak = [item for item in requests if item["case"] == "soak" and not item["warmup"]]
    if len(soak) < 61 or not all(item["success"] and item["http_status"] == 200 for item in soak):
        raise EvidenceError("soak must contain periodic successful generations")
    soak = sorted(soak, key=lambda item: float(item["started_elapsed_s"]))
    soak_start = float(soak[0]["started_elapsed_s"])
    soak_end = float(soak[-1]["finished_elapsed_s"])
    soak_duration = soak_end - soak_start
    start_gaps = [
        float(right["started_elapsed_s"]) - float(left["started_elapsed_s"])
        for left, right in zip(soak, soak[1:])
    ]
    if soak_duration < 1800 or max(start_gaps, default=math.inf) > 30:
        raise EvidenceError("soak raw generation coverage must span >=1800s with <=30s gaps")
    soak_summary = summary["gates"].get("soak")
    expected_soak = {
        "attempted": len(soak),
        "succeeded": len(soak),
        "started_elapsed_s": round(soak_start, 3),
        "finished_elapsed_s": round(soak_end, 3),
        "duration_seconds": round(soak_duration, 3),
        "max_start_gap_seconds": round(max(start_gaps, default=0), 3),
    }
    if soak_summary != expected_soak:
        raise EvidenceError("soak summary does not match raw generation timestamps")
    if not _close(float(summary["gates"]["continuous_run_seconds"]), soak_duration):
        raise EvidenceError("continuous_run_seconds does not match raw soak duration")

    elapsed = [row["elapsed_s"] for row in resources]
    gaps = [right - left for left, right in zip(elapsed, elapsed[1:])]
    resource_summary = summary["resources"]
    acceptance_start = _finite_float(
        resource_summary.get("acceptance_window_start_elapsed_s"),
        "summary.resources.acceptance_window_start_elapsed_s",
    )
    acceptance_end = _finite_float(
        resource_summary.get("acceptance_window_end_elapsed_s"),
        "summary.resources.acceptance_window_end_elapsed_s",
    )
    first_api_start = min(
        float(item["started_elapsed_s"])
        for item in requests if item["case"] == "stream-parity"
    )
    if acceptance_start > first_api_start or first_api_start - acceptance_start > 1.0:
        raise EvidenceError("acceptance window must start with the initial API gates")
    if not _close(acceptance_end, soak_end):
        raise EvidenceError("acceptance window must end with the final soak generation")
    if elapsed[0] > acceptance_start or elapsed[-1] < acceptance_end:
        raise EvidenceError("resource telemetry does not cover the complete acceptance window")
    if max(gaps, default=math.inf) > 2.5:
        raise EvidenceError("resource telemetry sampling gap exceeds 2.5 seconds")
    covered = [row for row in resources if soak_start <= row["elapsed_s"] <= soak_end]
    if len(covered) < math.floor(soak_duration / 2):
        raise EvidenceError("resource telemetry has too few samples for the soak duration")
    acceptance = [
        row for row in resources
        if acceptance_start <= row["elapsed_s"] <= acceptance_end
    ]
    if not acceptance:
        raise EvidenceError("resource telemetry has no acceptance-window samples")
    raw_sources = {str(row["wsl_rss_source"]) for row in acceptance}
    if raw_sources != {resource_summary.get("wsl_rss_source")}:
        raise EvidenceError("wsl_rss_source does not match acceptance telemetry")
    peak_vram = max(row["gpu_memory_mib"] for row in acceptance)
    peak_rss = max(row["wsl_rss_kib"] for row in acceptance)
    peak_swap = max(row["wsl_swap_kib"] for row in acceptance)
    if not _close(float(resource_summary["peak_vram_mib"]), peak_vram):
        raise EvidenceError("peak_vram_mib does not match telemetry")
    if not _close(float(resource_summary["peak_wsl_rss_kib"]), peak_rss):
        raise EvidenceError("peak_wsl_rss_kib does not match telemetry")
    if not _close(float(resource_summary["wsl_swap_kib"]), peak_swap):
        raise EvidenceError("wsl_swap_kib does not match telemetry")
    page_faults = {
        "minor_delta": max(0, acceptance[-1]["minor_faults"] - acceptance[0]["minor_faults"]),
        "major_delta": max(0, acceptance[-1]["major_faults"] - acceptance[0]["major_faults"]),
    }
    if resource_summary.get("page_faults") != page_faults:
        raise EvidenceError("page fault deltas do not match acceptance-window telemetry")
    leak = detect_monotonic_rss_leak(resources, soak_start, soak_end)
    if summary["gates"].get("memory_leak_detected") is not leak:
        raise EvidenceError("memory leak summary does not match telemetry")
    if leak:
        raise EvidenceError("resource telemetry indicates monotonic WSL RSS growth")


def write_checksums(directory: Path) -> Path:
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = []
    for path in files:
        relative = path.relative_to(directory).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}")
    target = directory / "SHA256SUMS"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def verify_checksums(directory: Path) -> None:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise EvidenceError(f"{directory}: missing SHA256SUMS")
    expected_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    listed: set[str] = set()
    for number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", line)
        if not match:
            raise EvidenceError(f"SHA256SUMS:{number}: malformed line")
        digest, relative = match.groups()
        path = directory / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise EvidenceError(f"SHA256SUMS:{number}: unsafe path")
        if not path.is_file():
            raise EvidenceError(f"SHA256SUMS:{number}: missing {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise EvidenceError(f"SHA256SUMS:{number}: checksum mismatch for {relative}")
        listed.add(relative)
    if listed != expected_files:
        missing = sorted(expected_files - listed)
        extra = sorted(listed - expected_files)
        raise EvidenceError(f"SHA256SUMS is not exhaustive (missing={missing}, extra={extra})")


def validate_directory(
    directory: Path,
    *,
    release: bool = False,
    expected_commit: str | None = None,
    tag_commit: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if directory.is_symlink():
        raise EvidenceError(f"{directory}: evidence root must not be a symlink")
    if not directory.is_dir():
        raise EvidenceError(f"not an evidence directory: {directory}")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise EvidenceError(f"{directory}: evidence bundles must not contain symlinks")
    for name in REQUIRED_FILES:
        if not (directory / name).is_file():
            raise EvidenceError(f"{directory}: missing {name}")
    if release:
        relative_files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*") if path.is_file()
        }
        if relative_files != RELEASE_ONLY_ALLOWED_FILES:
            raise EvidenceError(
                "release evidence contains missing/unexpected files: "
                f"expected={sorted(RELEASE_ONLY_ALLOWED_FILES)}, got={sorted(relative_files)}"
            )
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            if release:
                raise EvidenceError(f"{path}: evidence artifacts must be UTF-8 text") from exc
            continue
        assert_text_sanitized(text, path.relative_to(directory).as_posix())
    environment = read_json(directory / "environment.json")
    config = read_json(directory / "resolved-config.json")
    summary = read_json(directory / "summary.json")
    for name, value in (("environment.json", environment), ("resolved-config.json", config)):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise EvidenceError(f"{name}: schema_version must be {SCHEMA_VERSION}")
        assert_sanitized(value, name)
    _validate_summary(summary, release=release, expected_commit=expected_commit)
    if release:
        hardware = environment.get("hardware") or {}
        software = environment.get("software") or {}
        if not isinstance(hardware, dict) or not isinstance(software, dict):
            raise EvidenceError("release environment hardware/software must be objects")
        gpu_name = str(hardware.get("gpu_name", ""))
        if "RTX 5090" not in gpu_name or environment.get("synthetic") is not False:
            raise EvidenceError("release evidence must come from a real RTX 5090 environment")
        compute_capability = str(hardware.get("compute_capability", ""))
        if compute_capability not in {"12.0", "12.0 (SM120)"}:
            raise EvidenceError("release evidence must record RTX 5090 compute capability 12.0/SM120")
        if hardware.get("gpu_index") != 0:
            raise EvidenceError("release evidence must bind the measured GPU to index 0")
        gpu_memory = _finite_float(hardware.get("gpu_memory_mib"), "environment.hardware.gpu_memory_mib")
        if not 31 * 1024 <= gpu_memory <= 33 * 1024:
            raise EvidenceError("release GPU must expose a 32GB-class framebuffer")
        wsl_memory = _finite_float(hardware.get("wsl_memory_gib"), "environment.hardware.wsl_memory_gib")
        if wsl_memory < 100:
            raise EvidenceError("release WSL2 environment must expose at least 100 GiB RAM")
        if hardware.get("wsl_processors") != 32:
            raise EvidenceError("release WSL2 environment must expose exactly 32 processors")
        kernel = str(software.get("kernel") or "").lower()
        if "microsoft" not in kernel or "wsl2" not in kernel:
            raise EvidenceError("release evidence must come from WSL2")
        if software.get("os_id") != "ubuntu" or software.get("os_version_id") != "24.04":
            raise EvidenceError("release evidence must come from Ubuntu 24.04")
        if _numeric_version(software.get("nvidia_driver")) < (591, 86):
            raise EvidenceError("release NVIDIA driver must be at least 591.86")
        if _numeric_version(software.get("cuda_toolkit"))[:1] != (13,):
            raise EvidenceError("release CUDA toolkit must be 13.x")
        if not str(software.get("torch") or "").startswith("2.11"):
            raise EvidenceError("release Torch must be 2.11.x")
        if _numeric_version(software.get("torch_cuda"))[:1] != (13,):
            raise EvidenceError("release Torch runtime must use CUDA 13.x")
        if software.get("triton") != "3.6.0":
            raise EvidenceError("release Triton must be 3.6.0")
        if software.get("cuda_runtime_probe") is not True:
            raise EvidenceError("release CUDA tensor runtime probe must pass on SM120")
        verification = summary.get("verification") or {}
        if verification != {
            "checkpoint_full_sha256": True,
            "checkpoint_shape": True,
            "server_profile": True,
            "launch_attestation": True,
            "runtime_clean_tree": True,
            "server_port_owner": True,
        }:
            raise EvidenceError("checkpoint/runtime/server launch verification is incomplete")
        if summary.get("gates", {}).get("memory_leak_detected") is not False:
            raise EvidenceError("release evidence reports a monotonic memory leak")
        if summary.get("errors") != []:
            raise EvidenceError("release evidence must contain an empty errors list")
    latency_rows = _validate_csv(
        directory / "latency.csv",
        {"case", "iteration", "prompt_tokens", "completion_tokens", "ttft_ms", "total_ms"},
    )
    requests = _read_requests(directory / "requests.jsonl", release=release)
    resources = _resource_rows(directory / "resource-samples.csv", release=release)
    if release:
        _crosscheck_release_raw(directory, summary, config, requests, latency_rows, resources)
    verify_checksums(directory)
    if tag_commit is not None:
        if not release:
            raise EvidenceError("tag binding is only valid for release evidence")
        validate_tag_binding(summary, tag_commit=tag_commit, repo_root=repo_root or Path.cwd())
    return summary


def render_markdown(summary: dict[str, Any], environment: dict[str, Any]) -> str:
    resources = summary["resources"]
    gates = summary["gates"]
    hardware = environment.get("hardware", {})
    prompt_8176 = next(
        (case for case in gates.get("prompt_cases", []) if case.get("rendered_prompt_tokens") == 8176),
        {},
    )
    ttft_p50 = float(prompt_8176.get("ttft_p50_ms") or 0)
    ttft_p95 = float(prompt_8176.get("ttft_p95_ms") or 0)
    effective_prefill = 8176 / (ttft_p50 / 1000) if ttft_p50 > 0 else 0
    decode_tps = float(gates.get("steady_decode", {}).get("decode_tokens_per_second") or 0)
    lines = [
        "| Verified field | Result |",
        "|---|---:|",
        f"| Runtime commit | `{summary['source']['validated_runtime_commit'][:12]}` |",
        f"| Checkpoint revision | `{summary['model']['revision'][:12]}` |",
        f"| GPU | {hardware.get('gpu_name', 'not recorded')} |",
        f"| Executed expert path | {summary['execution']['quantization']} |",
        f"| Peak VRAM | {resources['peak_vram_mib'] / 1024:.3f} GiB |",
        f"| Peak WSL RSS | {resources['peak_wsl_rss_kib'] / 1024 / 1024:.3f} GiB |",
        f"| WSL swap | {resources['wsl_swap_kib'] / 1024:.0f} MiB |",
        f"| 8176 rendered-token TTFT (p50 / p95) | {ttft_p50:.1f} / {ttft_p95:.1f} ms |",
        f"| 8176 effective prefill (p50) | {effective_prefill:.1f} tok/s |",
        f"| 256-token steady decode | {decode_tps:.1f} tok/s |",
        f"| Sequential requests | {gates['stability']['succeeded']}/{gates['stability']['attempted']} |",
        f"| Continuous run | {gates['continuous_run_seconds'] / 60:.1f} min |",
        f"| Tests | {gates['pytest']['passed']} passed, {gates['pytest']['failed']} failed |",
    ]
    return "\n".join(lines)


def update_marked_text(text: str, generated: str) -> str:
    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise EvidenceError("README must contain exactly one generated benchmark marker pair")
    before, rest = text.split(BEGIN_MARKER, 1)
    _old, after = rest.split(END_MARKER, 1)
    return f"{before}{BEGIN_MARKER}\n{generated}\n{END_MARKER}{after}"


def find_release_input(
    root: Path,
    *,
    expected_commit: str | None = None,
    tag_commit: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    candidates: list[Path] = []
    failures: list[str] = []
    for directory in sorted(root.glob("rtx5090-*"), reverse=True):
        if not directory.is_dir():
            continue
        try:
            directory.resolve().relative_to(root.resolve())
        except ValueError:
            failures.append(f"{directory.name}: evidence directory escapes results root")
            continue
        try:
            validate_directory(
                directory,
                release=True,
                expected_commit=expected_commit,
                tag_commit=tag_commit,
                repo_root=repo_root,
            )
        except EvidenceError as exc:
            failures.append(f"{directory.name}: {exc}")
        else:
            candidates.append(directory)
    if not candidates:
        detail = "\n".join(failures) if failures else "no results/rtx5090-* directory exists"
        raise EvidenceError(f"no verified release evidence found under {root}:\n{detail}")
    return candidates[0]


def _command(args: argparse.Namespace) -> int:
    if args.command == "validate":
        validate_directory(
            Path(args.directory),
            release=args.release,
            expected_commit=args.expected_commit,
            tag_commit=args.tag_commit,
            repo_root=Path(args.repo_root) if args.repo_root else None,
        )
        print(f"valid: {args.directory}")
    elif args.command == "checksums":
        print(write_checksums(Path(args.directory)))
    elif args.command == "validate-example":
        root = Path(__file__).resolve().parents[2]
        validate_directory(root / "results" / "example-synthetic")
        print("synthetic evidence example is valid (not release-eligible)")
    elif args.command == "release-input":
        directory = find_release_input(
            Path(args.root),
            expected_commit=args.expected_commit,
            tag_commit=args.tag_commit,
            repo_root=Path(args.repo_root) if args.repo_root else None,
        )
        if args.print_path:
            print(directory.as_posix())
        else:
            print(f"release evidence: {directory}")
    elif args.command == "table":
        directory = Path(args.directory)
        summary = validate_directory(directory, release=False)
        environment = read_json(directory / "environment.json")
        generated = render_markdown(summary, environment)
        if args.readme:
            readme = Path(args.readme)
            current = readme.read_text(encoding="utf-8")
            updated = update_marked_text(current, generated)
            if args.check:
                if current != updated:
                    raise EvidenceError(f"{readme}: generated benchmark table is stale")
            else:
                readme.write_text(updated, encoding="utf-8")
        else:
            print(generated)
    else:  # pragma: no cover - argparse prevents this
        raise AssertionError(args.command)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate one evidence bundle")
    validate.add_argument("directory")
    validate.add_argument("--release", action="store_true", help="also enforce release gates")
    validate.add_argument("--expected-commit")
    validate.add_argument("--tag-commit")
    validate.add_argument("--repo-root")
    checksums = commands.add_parser("checksums", help="rewrite an exhaustive SHA256SUMS")
    checksums.add_argument("directory")
    commands.add_parser("validate-example", help="validate the tracked synthetic example")
    release = commands.add_parser("release-input", help="select the newest release-eligible bundle")
    release.add_argument("--root", default="results")
    release.add_argument("--print-path", action="store_true")
    release.add_argument("--expected-commit")
    release.add_argument("--tag-commit")
    release.add_argument("--repo-root")
    table = commands.add_parser("table", help="render or check the README benchmark table")
    table.add_argument("directory")
    table.add_argument("--readme")
    table.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return _command(build_parser().parse_args(argv))
    except EvidenceError as exc:
        print(f"evidence error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
