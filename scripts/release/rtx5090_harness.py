#!/usr/bin/env python3
"""Run the private RTX 5090 release gate against an already-running local server.

This program is intentionally never invoked by GitHub Actions.  It hashes the
full pinned checkpoint, runs the local test suite, measures the OpenAI API, and
writes only aggregate/sanitized evidence.  Prompts and generated text are kept
in memory and never written to the bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from evidence import (  # noqa: E402
    COMMIT_RE,
    SCHEMA_VERSION,
    EvidenceError,
    detect_monotonic_rss_leak,
    runtime_tree_sha256,
    validate_directory,
    write_checksums,
    write_json,
)
from q38lab.checkpoint import (  # noqa: E402
    CheckpointVerificationError,
    verify_checkpoint as verify_q38lab_checkpoint,
)
from q38lab.config import resolve_serve_config  # noqa: E402
from q38lab.constants import (  # noqa: E402
    EXPECTED_FILE_COUNT as MODEL_FILE_COUNT,
    EXPECTED_MANIFEST_SHA256 as MODEL_MANIFEST_SHA256,
    EXPECTED_TOTAL_BYTES as MODEL_TOTAL_BYTES,
    MODEL_REPO as MODEL_REPOSITORY,
    MODEL_REVISION,
    RTX5090_WSL2_PROFILE,
    SERVED_MODEL_NAME as SERVED_MODEL,
)


UPSTREAM_BASE = "9ef3651309fe4058672f2cc92069238dea06be1b"
PROMPT_TARGETS = (13, 128, 2048, 8176)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def command_output(command: list[str], default: str | None = None) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return default


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def numeric_version(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(value or "")))


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with LOCAL_OPENER.open(request, timeout=timeout) as response:
            data = json.load(response)
            if response.status != 200 or not isinstance(data, dict):
                raise EvidenceError(f"POST {url} did not return a 200 JSON object")
            return response.status, data, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise EvidenceError(f"HTTP {exc.code} from {url}: {detail}") from exc


def get_json(url: str, timeout: float = 30) -> dict[str, Any]:
    try:
        with LOCAL_OPENER.open(url, timeout=timeout) as response:
            value = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"GET {url} failed: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"GET {url} did not return a JSON object")
    return value


@dataclass
class StreamResult:
    status: int
    text: str
    reasoning: str
    tool_name: str
    tool_arguments: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float


def post_sse(url: str, payload: dict[str, Any], timeout: float) -> StreamResult:
    payload = dict(payload)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.perf_counter()
    first_delta: float | None = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_name = ""
    tool_arguments: list[str] = []
    usage: dict[str, Any] = {}
    done = False
    try:
        with LOCAL_OPENER.open(request, timeout=timeout) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise EvidenceError(f"stream returned invalid JSON: {data[:200]}") from exc
                if not isinstance(chunk, dict):
                    raise EvidenceError("stream data event must contain a JSON object")
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                tool_calls = delta.get("tool_calls") or []
                changed = bool(content or reasoning or tool_calls)
                if changed and first_delta is None:
                    first_delta = time.perf_counter()
                text_parts.append(content)
                reasoning_parts.append(reasoning)
                for call in tool_calls:
                    function = call.get("function") or {}
                    if function.get("name"):
                        tool_name = function["name"]
                    if function.get("arguments"):
                        tool_arguments.append(str(function["arguments"]))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise EvidenceError(f"HTTP {exc.code} from {url}: {detail}") from exc
    ended = time.perf_counter()
    if not done:
        raise EvidenceError("stream ended without the required [DONE] marker")
    if first_delta is None:
        raise EvidenceError("stream completed without a content, reasoning, or tool delta")
    return StreamResult(
        status=status,
        text="".join(text_parts),
        reasoning="".join(reasoning_parts),
        tool_name=tool_name,
        tool_arguments="".join(tool_arguments),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        ttft_ms=(first_delta - started) * 1000,
        total_ms=(ended - started) * 1000,
    )


def completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "")


def completion_tool(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices") or []
    if not choices:
        return "", ""
    calls = (choices[0].get("message") or {}).get("tool_calls") or []
    if not calls:
        return "", ""
    function = calls[0].get("function") or {}
    arguments = function.get("arguments") or ""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return str(function.get("name") or ""), str(arguments)


def normalize_json_arguments(value: str) -> str:
    try:
        return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))
    except json.JSONDecodeError:
        return value.strip()


def valid_city_arguments(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    city = parsed.get("city")
    if not isinstance(city, str) or not city.strip():
        return None
    normalized = city.strip().casefold()
    if "shanghai" not in normalized and "上海" not in normalized:
        return None
    return parsed


def thinking_response_valid(effort: str, content: str, reasoning: str) -> bool:
    if effort == "none":
        return bool(content) and not reasoning
    if effort == "high":
        return bool(content) and bool(reasoning)
    return False


def load_tokenizer(model_dir: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise EvidenceError("transformers is required to construct exact rendered prompts") from exc
    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False, local_files_only=True)


def rendered_length(tokenizer: Any, content: str) -> int:
    messages = [{"role": "user", "content": content}]
    # Match ``reasoning_effort=none`` after the OpenAI frontend maps it to the
    # exact template kwargs consumed by TokenizeManager.  Qwen's disabled
    # template inserts different control tokens from its default thinking mode.
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        thinking_mode="disabled",
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return len(token_ids)


def exact_prompt(tokenizer: Any, target: int) -> str:
    # Qwen's token for " x" composes additively under the chat template.  Verify
    # rather than assume that property, because a tokenizer/template update must
    # stop the pinned benchmark instead of silently changing prompt length.
    base = "x"
    baseline = rendered_length(tokenizer, base)
    if target < baseline:
        raise EvidenceError(f"rendered target {target} is below the template minimum {baseline}")
    estimate = target - baseline
    for count in range(max(0, estimate - 8), estimate + 9):
        candidate = base + " x" * count
        if rendered_length(tokenizer, candidate) == target:
            return candidate
    raise EvidenceError(f"could not construct an exact {target}-token rendered Qwen chat prompt")


def verify_checkpoint(model_dir: Path) -> None:
    """Use the same pinned canonical manifest verifier as ``q38lab download``."""

    try:
        verification = verify_q38lab_checkpoint(model_dir, full=True)
    except CheckpointVerificationError as exc:
        raise EvidenceError(str(exc)) from exc
    if not verification.full_verify or verification.manifest_sha256 != MODEL_MANIFEST_SHA256:
        raise EvidenceError("q38lab checkpoint verifier did not perform the full pinned hash")


def process_command(pid: int) -> list[str]:
    try:
        return [part.decode("utf-8", "replace") for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
    except OSError as exc:
        raise EvidenceError(f"cannot read server process {pid}: {exc}") from exc


def _flag_values(command: list[str]) -> dict[str, str | bool]:
    values: dict[str, str | bool] = {}
    for index, part in enumerate(command):
        if not part.startswith("--"):
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
        elif index + 1 < len(command) and not command[index + 1].startswith("--"):
            values[part] = command[index + 1]
        else:
            values[part] = True
    return values


def verify_launch_argv(
    command: list[str], *, model_dir: Path, host: str, port: int
) -> None:
    values = _flag_values(command)
    expected = {
        "--gpu": "0",
        "--tp-size": "1",
        "--max-running-requests": "1",
        "--max-seq-len-override": "8192",
        "--max-prefill-length": "8192",
        "--num-tokens": "8192",
        "--memory-ratio": "0.89",
        "--cache-type": "naive",
        "--attention-backend": "qsa_triton",
        "--graph": "0",
        "--moe-backend": "offload",
        "--nvfp4-backend": "auto",
        "--served-model-name": SERVED_MODEL,
        "--host": host,
        "--port": str(port),
    }
    for flag, wanted in expected.items():
        if str(values.get(flag)) != wanted:
            raise EvidenceError(f"server process must include {flag} {wanted}")
    if values.get("--moe-cache-auto") is not True:
        raise EvidenceError("server process must include --moe-cache-auto")
    if "--moe-cpu-layers" in values:
        raise EvidenceError("--moe-cpu-layers must be omitted so the memory budget resolver decides")
    model_value = values.get("--model", values.get("--model-path"))
    if not isinstance(model_value, str) or Path(model_value).resolve() != model_dir.resolve():
        raise EvidenceError("server launch argv model path does not match --model-dir")
    if host != "127.0.0.1" or port != RTX5090_WSL2_PROFILE.port:
        raise EvidenceError("release harness requires 127.0.0.1:1919 exactly")


def verify_server_command(pid: int, *, model_dir: Path, host: str, port: int) -> None:
    """Optional direct ``ft`` process validation (q38lab normally uses attestation)."""

    verify_launch_argv(process_command(pid), model_dir=model_dir, host=host, port=port)


def process_start_ticks(pid: int) -> int:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (OSError, ValueError, IndexError) as exc:
        raise EvidenceError(f"cannot read process start ticks for PID {pid}") from exc


def process_cwd(pid: int) -> Path:
    try:
        return Path(f"/proc/{pid}/cwd").resolve()
    except OSError as exc:
        raise EvidenceError(f"cannot resolve cwd for PID {pid}") from exc


def verify_clean_runtime(root: Path, expected_commit: str) -> str:
    if not COMMIT_RE.fullmatch(expected_commit):
        raise EvidenceError("--expected-commit must be exact 40-hex")
    commit = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    if commit != expected_commit:
        raise EvidenceError(f"runtime checkout {commit} != expected commit {expected_commit}")
    status = command_output([
        "git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"
    ])
    if status is None or status:
        raise EvidenceError("release harness requires a completely clean runtime checkout")
    return runtime_tree_sha256(root, expected_commit)


def _expected_public_config(model_dir: Path, host: str, port: int) -> dict[str, Any]:
    profile = RTX5090_WSL2_PROFILE
    resolved = resolve_serve_config(
        profile_name=profile.name,
        cli={
            "model_dir": model_dir.resolve(),
            "served_model_name": SERVED_MODEL,
            "gpu": "0",
            "host": host,
            "port": port,
            "memory_ratio": 0.89,
            "num_tokens": 8192,
            "max_seq_len": 8192,
            "max_prefill_length": 8192,
            "unsafe_non_loopback": False,
        },
        env={},
        profile=profile,
    )
    return resolved.public_dict()


def verify_launch_attestation(
    path: Path,
    *,
    pid: int,
    root: Path,
    expected_commit: str,
    model_dir: Path,
    host: str,
    port: int,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read q38lab launch attestation {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise EvidenceError("q38lab launch attestation schema_version must be 1.0")
    if value.get("pid") != pid or value.get("proc_start_ticks") != process_start_ticks(pid):
        raise EvidenceError("launch attestation PID/start ticks do not match the live server")
    if value.get("runtime_commit") != expected_commit or value.get("clean_tree") is not True:
        raise EvidenceError("launch attestation is not bound to the clean expected runtime commit")
    if Path(str(value.get("model_realpath", ""))).resolve() != model_dir.resolve():
        raise EvidenceError("launch attestation model realpath does not match --model-dir")
    config = value.get("resolved_config")
    if config != _expected_public_config(model_dir, host, port):
        raise EvidenceError("launch attestation resolved config does not match the release profile")
    if not isinstance(config, dict) or config.get("profile_contract_verified") is not True:
        raise EvidenceError("launch attestation must prove the exact release profile contract")
    argv = value.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise EvidenceError("launch attestation argv must be a string array")
    verify_launch_argv(argv, model_dir=model_dir, host=host, port=port)
    if process_cwd(pid) != root.resolve():
        raise EvidenceError("live server process cwd is not the validated runtime checkout")
    return value


def proc_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().split()
            parents[int(fields[0])] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def verify_listening_port(root_pid: int, host: str, port: int) -> None:
    """Prove that the attested process tree owns the loopback listening socket."""

    if host != "127.0.0.1" or port != RTX5090_WSL2_PROFILE.port:
        raise EvidenceError("release server must bind 127.0.0.1:1919 exactly")
    inodes: set[str] = set()
    for proc_net in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = proc_net.read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                address_hex, port_hex = fields[1].rsplit(":", 1)
                local_port = int(port_hex, 16)
                if len(address_hex) == 8:
                    packed = bytes.fromhex(address_hex)[::-1]
                elif len(address_hex) == 32:
                    packed = b"".join(
                        bytes.fromhex(address_hex[index:index + 8])[::-1]
                        for index in range(0, 32, 8)
                    )
                else:
                    continue
                local_address = ipaddress.ip_address(packed)
            except (ValueError, IndexError):
                continue
            if local_port == port and local_address == ipaddress.ip_address(host):
                inodes.add(fields[9])
    if not inodes:
        raise EvidenceError(f"no listener found on the exact address {host}:{port}")
    for pid in proc_tree(root_pid):
        try:
            descriptors = Path(f"/proc/{pid}/fd").iterdir()
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return
    raise EvidenceError(f"port {port} is not owned by the attested server process tree")


def process_faults(root_pid: int) -> tuple[int, int, int]:
    minor = major = 0
    sampled = 0
    for pid in proc_tree(root_pid):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            minor += int(fields[9])
            major += int(fields[11])
            sampled += 1
        except (OSError, ValueError, IndexError):
            continue
    if sampled == 0:
        raise EvidenceError("cannot sample page faults from the attested process tree")
    return minor, major, sampled


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            values[key] = int(rest.strip().split()[0])
    except (OSError, ValueError, IndexError) as exc:
        raise EvidenceError(f"cannot read /proc/meminfo: {exc}") from exc
    required = {"MemTotal", "MemAvailable", "SwapTotal"}
    missing = required - set(values)
    if missing:
        raise EvidenceError(f"/proc/meminfo is missing {sorted(missing)}")
    return values


def wsl_working_set_kib() -> tuple[int, str]:
    raw = command_output([
        "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
        "[int64]((Get-Process -Name vmmemWSL -ErrorAction Stop).WorkingSet64 / 1KB)",
    ])
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw), "Windows vmmemWSL working set"
    info = meminfo()
    working_set = info["MemTotal"] - info["MemAvailable"]
    if working_set <= 0:
        raise EvidenceError("WSL RSS fallback produced a non-positive working set")
    return working_set, "WSL MemTotal-MemAvailable fallback"


class ResourceSampler:
    def __init__(self, pid: int, target: Path, *, origin: float) -> None:
        self.pid = pid
        self.target = target
        self.started = origin
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float | int | str | None]] = []
        self.rss_source = "unknown"
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="release-resource-sampler", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        self._write()

    def _sample(self) -> None:
        gpu_raw = command_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"
        ])
        try:
            gpu_mib = float((gpu_raw or "").splitlines()[0].strip())
        except (ValueError, IndexError):
            raise EvidenceError("nvidia-smi did not return GPU memory telemetry")
        if gpu_mib <= 0:
            raise EvidenceError("GPU memory telemetry must be positive")
        rss_kib, rss_source = wsl_working_set_kib()
        if self.rss_source not in {"unknown", rss_source}:
            raise EvidenceError("WSL RSS telemetry source changed during the acceptance run")
        self.rss_source = rss_source
        info = meminfo()
        minor, major, fault_processes = process_faults(self.pid)
        if minor <= 0:
            raise EvidenceError("attested process tree reported no minor page faults")
        self.samples.append({
            "elapsed_s": round(time.monotonic() - self.started, 3),
            "gpu_memory_mib": gpu_mib,
            "wsl_rss_kib": rss_kib,
            "wsl_rss_source": rss_source,
            "wsl_swap_kib": info["SwapTotal"],
            "minor_faults": minor,
            "major_faults": major,
            "fault_processes": fault_processes,
            "pcie_rx_mib_s": None,
            "pcie_tx_mib_s": None,
        })

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self._sample()
                self.stop_event.wait(1)
            self._sample()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()

    def _write(self) -> None:
        fields = [
            "elapsed_s", "gpu_memory_mib", "wsl_rss_kib", "wsl_rss_source",
            "wsl_swap_kib", "minor_faults", "major_faults", "fault_processes",
            "pcie_rx_mib_s", "pcie_tx_mib_s",
        ]
        with self.target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)


def sanitize_test_log(text: str, model_dir: Path) -> str:
    text = text.replace(str(model_dir), "<MODEL_DIR>")
    text = re.sub(r"/home/[^/\s]+", "<HOME>", text)
    text = re.sub(r"(?i)[a-z]:[\\/]users[\\/][^\\/\s]+", "<USER_HOME>", text)
    return text


def run_pytest(root: Path, model_dir: Path, target: Path) -> dict[str, int]:
    command = [sys.executable, "-m", "pytest", "-m", "not slow"]
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    target.write_text(sanitize_test_log(result.stdout, model_dir), encoding="utf-8")
    counts = {"passed": 0, "failed": 0, "skipped": 0, "deselected": 0}
    for key in counts:
        matches = re.findall(rf"(\d+) {key}", result.stdout)
        if matches:
            counts[key] = int(matches[-1])
    if result.returncode and counts["failed"] == 0:
        counts["failed"] = 1
    return counts


def environment_document() -> dict[str, Any]:
    info = meminfo()
    os_release: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')
    except OSError:
        pass
    gpu_line = command_output([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits", "--id=0"
    ], "unknown, 0, unknown, unknown") or "unknown, 0, unknown, unknown"
    parts = [part.strip() for part in gpu_line.split(",")]
    try:
        gpu_memory = int(float(parts[1]))
    except (ValueError, IndexError):
        gpu_memory = 0
    nvcc = command_output(["nvcc", "--version"], "not found") or "not found"
    cuda_match = re.search(r"release ([0-9.]+)", nvcc)
    try:
        import torch
        torch_version = torch.__version__
        torch_cuda_version = str(torch.version.cuda) if torch.version.cuda else None
        cuda_runtime_probe = False
    except ImportError:
        torch_version = None
        torch_cuda_version = None
        cuda_runtime_probe = False
    else:
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0):
                probe = torch.ones(1, device="cuda") + 1
                torch.cuda.synchronize(0)
                cuda_runtime_probe = probe.item() == 2
        except Exception:
            cuda_runtime_probe = False
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": utc_now(),
        "synthetic": False,
        "hardware": {
            "gpu_name": parts[0] if parts else "unknown",
            "gpu_memory_mib": gpu_memory,
            "gpu_index": 0,
            "compute_capability": parts[3] if len(parts) > 3 else "unknown",
            "wsl_memory_gib": round(info.get("MemTotal", 0) / 1024 / 1024, 3),
            "wsl_processors": os.cpu_count() or 0,
        },
        "software": {
            "os": platform.platform(),
            "kernel": platform.release(),
            "os_id": os_release.get("ID"),
            "os_version_id": os_release.get("VERSION_ID"),
            "nvidia_driver": parts[2] if len(parts) > 2 else "unknown",
            "cuda_toolkit": cuda_match.group(1) if cuda_match else "not found",
            "python": platform.python_version(),
            "torch": torch_version,
            "torch_cuda": torch_cuda_version,
            "triton": triton_version,
            "cuda_runtime_probe": cuda_runtime_probe,
        },
    }


def resolved_config(model_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": "rtx5090-wsl2",
        "served_model_name": SERVED_MODEL,
        "model_basename": model_dir.name,
        "settings": {
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
        },
    }


class Recorder:
    def __init__(self, directory: Path, *, origin: float) -> None:
        self.directory = directory
        self.origin = origin
        self.requests = (directory / "requests.jsonl").open("w", encoding="utf-8")
        self.latency_handle = (directory / "latency.csv").open("w", newline="", encoding="utf-8")
        self.latency = csv.DictWriter(self.latency_handle, fieldnames=["case", "iteration", "prompt_tokens", "completion_tokens", "ttft_ms", "total_ms"])
        self.latency.writeheader()

    def record(
        self,
        *,
        case: str,
        iteration: int,
        warmup: bool,
        stream: bool,
        success: bool,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_ms: float | None,
        total_ms: float,
        http_status: int = 200,
        content_tokens: int | None = None,
        proof: dict[str, Any] | None = None,
        started_elapsed_s: float | None = None,
        finished_elapsed_s: float | None = None,
    ) -> dict[str, Any]:
        if (started_elapsed_s is None) != (finished_elapsed_s is None):
            raise ValueError("explicit request timing requires both start and finish")
        if started_elapsed_s is None:
            finished_elapsed = time.monotonic() - self.origin
            started_elapsed = max(0.0, finished_elapsed - total_ms / 1000)
        else:
            started_elapsed = started_elapsed_s
            finished_elapsed = finished_elapsed_s
            total_ms = max(0.0, (finished_elapsed - started_elapsed) * 1000)
        item = {
            "case": case,
            "iteration": iteration,
            "warmup": warmup,
            "stream": stream,
            "success": success,
            "http_status": http_status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "content_tokens": content_tokens,
            "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
            "total_ms": round(total_ms, 3),
            "started_elapsed_s": round(started_elapsed, 3),
            "finished_elapsed_s": round(finished_elapsed, 3),
            "recorded_at": utc_now(),
            "proof": dict(proof or {}),
        }
        self.requests.write(json.dumps(item, sort_keys=True) + "\n")
        self.requests.flush()
        if not warmup:
            self.latency.writerow({key: item[key] for key in self.latency.fieldnames})
            self.latency_handle.flush()
        return item

    def close(self) -> None:
        self.requests.close()
        self.latency_handle.close()


def chat_payload(content: str, max_tokens: int, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    payload.update(extra)
    return payload


def run_api_gates(args: argparse.Namespace, recorder: Recorder, tokenizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    api = {"stream_nonstream_match": False, "thinking_none": False, "thinking_high": False, "tool_call": False}
    prompt_summaries = []

    deterministic = chat_payload("Reply with exactly: freetoken-ok", 32, reasoning_effort="none")
    status, nonstream, total_ms = post_json(endpoint, deterministic, args.request_timeout)
    usage = nonstream.get("usage") or {}
    stream = post_sse(endpoint, deterministic, args.request_timeout)
    nonstream_text = completion_text(nonstream)
    parity_ok = nonstream_text == stream.text and bool(stream.text)
    api["stream_nonstream_match"] = parity_ok
    nonstream_hash = hashlib.sha256(nonstream_text.encode("utf-8")).hexdigest()
    stream_hash = hashlib.sha256(stream.text.encode("utf-8")).hexdigest()
    recorder.record(case="stream-parity", iteration=0, warmup=False, stream=False, success=parity_ok, prompt_tokens=int(usage.get("prompt_tokens") or 0), completion_tokens=int(usage.get("completion_tokens") or 0), ttft_ms=None, total_ms=total_ms, http_status=status, proof={"text_present": bool(nonstream_text), "text_sha256": nonstream_hash})
    recorder.record(case="stream-parity", iteration=1, warmup=False, stream=True, success=parity_ok, prompt_tokens=stream.prompt_tokens, completion_tokens=stream.completion_tokens, ttft_ms=stream.ttft_ms, total_ms=stream.total_ms, http_status=stream.status, proof={"text_present": bool(stream.text), "text_sha256": stream_hash})

    for effort in ("none", "high"):
        payload = chat_payload("Explain why the sky is blue in one sentence.", 128, reasoning_effort=effort)
        status, response, total_ms = post_json(endpoint, payload, args.request_timeout)
        usage = response.get("usage") or {}
        message = ((response.get("choices") or [{}])[0].get("message") or {})
        stream_thinking = post_sse(endpoint, payload, args.request_timeout)
        nonstream_reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
        thinking_ok = (
            thinking_response_valid(effort, str(message.get("content") or ""), nonstream_reasoning)
            and thinking_response_valid(effort, stream_thinking.text, stream_thinking.reasoning)
        )
        api[f"thinking_{effort}"] = thinking_ok
        nonstream_content = str(message.get("content") or "")
        recorder.record(case=f"thinking-{effort}", iteration=0, warmup=False, stream=False, success=thinking_ok, prompt_tokens=int(usage.get("prompt_tokens") or 0), completion_tokens=int(usage.get("completion_tokens") or 0), ttft_ms=None, total_ms=total_ms, http_status=status, proof={"reasoning_present": bool(nonstream_reasoning), "visible_text_present": bool(nonstream_content)})
        recorder.record(case=f"thinking-{effort}", iteration=1, warmup=False, stream=True, success=thinking_ok, prompt_tokens=stream_thinking.prompt_tokens, completion_tokens=stream_thinking.completion_tokens, ttft_ms=stream_thinking.ttft_ms, total_ms=stream_thinking.total_ms, http_status=stream_thinking.status, proof={"reasoning_present": bool(stream_thinking.reasoning), "visible_text_present": bool(stream_thinking.text)})

    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Read weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
    tool_payload = chat_payload("What is the weather in Shanghai?", 128, reasoning_effort="none", tools=[tool], tool_choice="required")
    status, nonstream_tool, total_ms = post_json(endpoint, tool_payload, args.request_timeout)
    usage = nonstream_tool.get("usage") or {}
    stream_tool = post_sse(endpoint, tool_payload, args.request_timeout)
    ns_name, ns_args = completion_tool(nonstream_tool)
    ns_city = valid_city_arguments(ns_args)
    stream_city = valid_city_arguments(stream_tool.tool_arguments)
    tool_ok = (
        ns_name == stream_tool.tool_name == "get_weather"
        and ns_city is not None and stream_city is not None
        and normalize_json_arguments(ns_args) == normalize_json_arguments(stream_tool.tool_arguments)
    )
    api["tool_call"] = tool_ok
    ns_normalized = normalize_json_arguments(ns_args)
    stream_normalized = normalize_json_arguments(stream_tool.tool_arguments)
    ns_city_value = str((ns_city or {}).get("city") or "").strip().casefold()
    stream_city_value = str((stream_city or {}).get("city") or "").strip().casefold()
    recorder.record(case="tool-call", iteration=0, warmup=False, stream=False, success=tool_ok, prompt_tokens=int(usage.get("prompt_tokens") or 0), completion_tokens=int(usage.get("completion_tokens") or 0), ttft_ms=None, total_ms=total_ms, http_status=status, proof={"arguments_sha256": hashlib.sha256(ns_normalized.encode("utf-8")).hexdigest(), "tool_city": ns_city_value, "tool_name": ns_name})
    recorder.record(case="tool-call", iteration=1, warmup=False, stream=True, success=tool_ok, prompt_tokens=stream_tool.prompt_tokens, completion_tokens=stream_tool.completion_tokens, ttft_ms=stream_tool.ttft_ms, total_ms=stream_tool.total_ms, http_status=stream_tool.status, proof={"arguments_sha256": hashlib.sha256(stream_normalized.encode("utf-8")).hexdigest(), "tool_city": stream_city_value, "tool_name": stream_tool.tool_name})

    for target in PROMPT_TARGETS:
        content = exact_prompt(tokenizer, target)
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)
        content_tokens = len(content_token_ids)
        if target == 13 and content_tokens != 1:
            raise EvidenceError("the 13-token rendered prompt must contain exactly one content token")
        payload = chat_payload(content, 7, reasoning_effort="none", ignore_eos=True)
        for iteration in range(args.warmups + args.measurements):
            warmup = iteration < args.warmups
            result = post_sse(endpoint, payload, args.request_timeout)
            if result.prompt_tokens != target:
                raise EvidenceError(f"server reported {result.prompt_tokens} prompt tokens for target {target}")
            recorder.record(case=f"prompt-{target}", iteration=iteration, warmup=warmup, stream=True, success=True, prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens, ttft_ms=result.ttft_ms, total_ms=result.total_ms, http_status=result.status, content_tokens=content_tokens)
            if not warmup:
                prompt_summaries.append({"target": target, "content_tokens": content_tokens, "ttft_ms": result.ttft_ms, "total_ms": result.total_ms, "completion_tokens": result.completion_tokens})

    prompt_cases = []
    for target in PROMPT_TARGETS:
        rows = [row for row in prompt_summaries if row["target"] == target]
        prompt_cases.append({
            "content_prompt_tokens": int(statistics.median(row["content_tokens"] for row in rows)),
            "rendered_prompt_tokens": target,
            "completion_tokens": int(statistics.median(row["completion_tokens"] for row in rows)),
            "ttft_p50_ms": round(percentile([row["ttft_ms"] for row in rows], 0.50) or 0, 3),
            "ttft_p95_ms": round(percentile([row["ttft_ms"] for row in rows], 0.95) or 0, 3),
            "total_p50_ms": round(percentile([row["total_ms"] for row in rows], 0.50) or 0, 3),
            "total_p95_ms": round(percentile([row["total_ms"] for row in rows], 0.95) or 0, 3),
        })

    decode_payload = chat_payload("Continue with short numbered facts.", args.decode_tokens, reasoning_effort="none", ignore_eos=True)
    decode = post_sse(endpoint, decode_payload, args.request_timeout)
    recorder.record(case="steady-decode", iteration=0, warmup=False, stream=True, success=True, prompt_tokens=decode.prompt_tokens, completion_tokens=decode.completion_tokens, ttft_ms=decode.ttft_ms, total_ms=decode.total_ms, http_status=decode.status)
    decode_seconds = max(0.001, (decode.total_ms - decode.ttft_ms) / 1000)
    steady = {
        "requested_tokens": args.decode_tokens,
        "observed_tokens": decode.completion_tokens,
        "decode_tokens_per_second": round(max(0, decode.completion_tokens - 1) / decode_seconds, 3),
    }
    return {"api": api, "prompt_cases": prompt_cases, "steady_decode": steady}, {"endpoint": endpoint}


def run_stability(args: argparse.Namespace, recorder: Recorder) -> dict[str, Any]:
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    payload = chat_payload("Reply with OK", 8, reasoning_effort="none")
    durations: list[float] = []
    succeeded = 0
    for iteration in range(args.sequential_requests):
        status, response, total_ms = post_json(endpoint, payload, args.request_timeout)
        usage = response.get("usage") or {}
        success = status == 200 and bool(completion_text(response))
        succeeded += int(success)
        durations.append(total_ms)
        recorder.record(case="stability", iteration=iteration, warmup=False, stream=False, success=success, prompt_tokens=int(usage.get("prompt_tokens") or 0), completion_tokens=int(usage.get("completion_tokens") or 0), ttft_ms=None, total_ms=total_ms, http_status=status)
        if not success:
            break
    return {
        "attempted": len(durations),
        "succeeded": succeeded,
        "p50_ms": round(percentile(durations, 0.50) or 0, 3),
        "p95_ms": round(percentile(durations, 0.95) or 0, 3),
        "max_ms": round(max(durations, default=0), 3),
    }


def run_soak(args: argparse.Namespace, recorder: Recorder) -> dict[str, Any]:
    """Start only after all initial gates, then periodically perform generation."""

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    payload = chat_payload("Reply with OK", 8, reasoning_effort="none")
    soak_started = time.monotonic()
    iteration = 0
    succeeded = 0
    raw_starts: list[float] = []
    raw_finishes: list[float] = []
    while True:
        request_started = time.monotonic()
        status, response, total_ms = post_json(endpoint, payload, args.request_timeout)
        usage = response.get("usage") or {}
        success = status == 200 and bool(completion_text(response))
        request_finished = time.monotonic()
        recorded = recorder.record(
            case="soak", iteration=iteration, warmup=False, stream=False,
            success=success,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            ttft_ms=None, total_ms=total_ms, http_status=status,
            started_elapsed_s=request_started - recorder.origin,
            finished_elapsed_s=request_finished - recorder.origin,
        )
        raw_starts.append(float(recorded["started_elapsed_s"]))
        raw_finishes.append(float(recorded["finished_elapsed_s"]))
        succeeded += int(success)
        iteration += 1
        if not success:
            raise EvidenceError("soak generation failed")
        elapsed = request_finished - soak_started
        if elapsed >= args.duration_seconds:
            break
        next_start = soak_started + iteration * args.soak_interval_seconds
        time.sleep(max(0.0, min(next_start - time.monotonic(), args.soak_interval_seconds)))
    gaps = [right - left for left, right in zip(raw_starts, raw_starts[1:])]
    return {
        "attempted": iteration,
        "succeeded": succeeded,
        "started_elapsed_s": round(raw_starts[0], 3),
        "finished_elapsed_s": round(raw_finishes[-1], 3),
        "duration_seconds": round(raw_finishes[-1] - raw_starts[0], 3),
        "max_start_gap_seconds": round(max(gaps, default=0.0), 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--soak-interval-seconds", type=float, default=15.0)
    parser.add_argument("--sequential-requests", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--measurements", type=int, default=10)
    parser.add_argument("--decode-tokens", type=int, choices=range(256, 513), default=256, metavar="256..512")
    parser.add_argument("--request-timeout", type=float, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = ROOT
    if not COMMIT_RE.fullmatch(args.expected_commit):
        print("--expected-commit must be exact 40-hex", file=sys.stderr)
        return 2
    if args.duration_seconds < 1800:
        print("--duration-seconds must be at least 1800 for release evidence", file=sys.stderr)
        return 2
    if not 0 < args.soak_interval_seconds <= 25:
        print("--soak-interval-seconds must be in (0, 25] to leave scheduling margin", file=sys.stderr)
        return 2
    if args.sequential_requests < 100:
        print("--sequential-requests must be at least 100", file=sys.stderr)
        return 2
    if args.warmups < 3:
        print("--warmups must be at least 3", file=sys.stderr)
        return 2
    if args.measurements < 10:
        print("--measurements must be at least 10", file=sys.stderr)
        return 2
    parsed_url = urllib.parse.urlsplit(args.base_url)
    host = parsed_url.hostname or ""
    port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    if parsed_url.scheme != "http" or parsed_url.path not in {"", "/"}:
        print("--base-url must be a plain local http origin", file=sys.stderr)
        return 2
    if host != "127.0.0.1" or port != RTX5090_WSL2_PROFILE.port:
        print("--base-url must be exactly http://127.0.0.1:1919", file=sys.stderr)
        return 2
    try:
        runtime_tree = verify_clean_runtime(root, args.expected_commit)
    except EvidenceError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    if args.out.exists() and any(args.out.iterdir()):
        print(f"refusing to overwrite non-empty evidence directory: {args.out}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    measured_at = utc_now()
    errors: list[str] = []
    checkpoint_verified = False
    server_profile_verified = False
    attestation_verified = False
    server_port_verified = False
    runtime_clean_verified = True
    recorder = Recorder(args.out, origin=started)
    sampler = ResourceSampler(args.server_pid, args.out / "resource-samples.csv", origin=started)
    pytest_counts = {"passed": 0, "failed": 1, "skipped": 0, "deselected": 0}
    api_results = {
        "api": {"stream_nonstream_match": False, "thinking_none": False, "thinking_high": False, "tool_call": False},
        "prompt_cases": [],
        "steady_decode": {"requested_tokens": args.decode_tokens, "observed_tokens": 0},
    }
    stability = {"attempted": args.sequential_requests, "succeeded": 0}
    soak = {
        "attempted": 0, "succeeded": 0, "started_elapsed_s": 0.0,
        "finished_elapsed_s": 0.0, "duration_seconds": 0.0,
        "max_start_gap_seconds": 0.0,
    }
    acceptance_window_start = 0.0
    acceptance_window_end = 0.0
    env_doc = environment_document()
    sampler.start()
    try:
        hardware = env_doc["hardware"]
        software = env_doc["software"]
        if "RTX 5090" not in str(hardware["gpu_name"]):
            raise EvidenceError("GPU index 0 is not an RTX 5090")
        if str(hardware.get("compute_capability")) != "12.0":
            raise EvidenceError("GPU index 0 does not report SM120/compute capability 12.0")
        if not 31 * 1024 <= int(hardware.get("gpu_memory_mib") or 0) <= 33 * 1024:
            raise EvidenceError("GPU index 0 does not expose a 32GB-class framebuffer")
        if float(hardware.get("wsl_memory_gib") or 0) < 100:
            raise EvidenceError("WSL2 must expose at least 100 GiB RAM")
        if int(hardware.get("wsl_processors") or 0) != 32:
            raise EvidenceError("the release profile requires exactly 32 WSL processors")
        kernel = str(software.get("kernel") or "").lower()
        if "microsoft" not in kernel or "wsl2" not in kernel:
            raise EvidenceError("release evidence must run under WSL2")
        if software.get("os_id") != "ubuntu" or software.get("os_version_id") != "24.04":
            raise EvidenceError("release evidence requires Ubuntu 24.04")
        if numeric_version(software.get("nvidia_driver")) < (591, 86):
            raise EvidenceError("NVIDIA driver must be at least 591.86")
        if numeric_version(software.get("cuda_toolkit"))[:1] != (13,):
            raise EvidenceError("CUDA toolkit 13.x is required")
        if not str(software.get("torch") or "").startswith("2.11"):
            raise EvidenceError("Torch 2.11.x is required")
        if numeric_version(software.get("torch_cuda"))[:1] != (13,):
            raise EvidenceError("Torch must use the CUDA 13 runtime")
        if software.get("triton") != "3.6.0":
            raise EvidenceError("Triton 3.6.0 is required")
        if software.get("cuda_runtime_probe") is not True:
            raise EvidenceError("the SM120 CUDA tensor runtime probe did not pass")
        attestation_path = args.attestation or (
            Path.home() / ".cache" / "q38lab" / f"serve-{port}.json"
        )
        verify_launch_attestation(
            attestation_path,
            pid=args.server_pid,
            root=root,
            expected_commit=args.expected_commit,
            model_dir=args.model_dir,
            host=host,
            port=port,
        )
        attestation_verified = True
        verify_listening_port(args.server_pid, host, port)
        server_port_verified = True
        server_profile_verified = True
        health = get_json(args.base_url.rstrip("/") + "/health")
        stats = get_json(args.base_url.rstrip("/") + "/v1/stats")
        models = get_json(args.base_url.rstrip("/") + "/v1/models")
        if health.get("status") != "ok":
            raise EvidenceError(f"server health is not ok: {health.get('status')}")
        if (stats.get("model") or {}).get("ctx") != 8192:
            raise EvidenceError("server /v1/stats does not report ctx=8192")
        model_ids = {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}
        if model_ids != {SERVED_MODEL}:
            raise EvidenceError(f"/v1/models must contain only {SERVED_MODEL}")
        verify_checkpoint(args.model_dir)
        checkpoint_verified = True
        pytest_counts = run_pytest(root, args.model_dir, args.out / "pytest.txt")
        tokenizer = load_tokenizer(args.model_dir)
        acceptance_window_start = time.monotonic() - started
        api_results, _private = run_api_gates(args, recorder, tokenizer)
        stability = run_stability(args, recorder)
        if not (
            pytest_counts["passed"] >= 1454 and pytest_counts["failed"] == 0
            and all(api_results["api"].values())
            and stability["attempted"] >= 100
            and stability["succeeded"] == stability["attempted"]
            and api_results["steady_decode"].get("observed_tokens", 0) >= 256
        ):
            raise EvidenceError("an initial release gate failed; soak was not started")
        soak = run_soak(args, recorder)
        acceptance_window_end = float(soak["finished_elapsed_s"])
    except Exception as exc:  # produce an inspectable incomplete bundle on every failure
        errors.append(sanitize_test_log(f"{type(exc).__name__}: {exc}", args.model_dir))
        if not (args.out / "pytest.txt").exists():
            (args.out / "pytest.txt").write_text("Test suite did not run because an earlier gate failed.\n", encoding="utf-8")
    finally:
        recorder.close()
        sampler.stop()
    if sampler.error:
        errors.append(sanitize_test_log(f"resource telemetry failed: {sampler.error}", args.model_dir))

    elapsed = float(soak["duration_seconds"])
    write_json(args.out / "environment.json", env_doc)
    write_json(args.out / "resolved-config.json", resolved_config(args.model_dir))
    numeric_samples = [
        {
            "elapsed_s": float(row["elapsed_s"]),
            "gpu_memory_mib": float(row["gpu_memory_mib"]),
            "wsl_rss_kib": float(row["wsl_rss_kib"]),
            "wsl_swap_kib": float(row["wsl_swap_kib"]),
            "minor_faults": float(row["minor_faults"]),
            "major_faults": float(row["major_faults"]),
        }
        for row in sampler.samples
    ]
    sample_elapsed = [row["elapsed_s"] for row in numeric_samples]
    sample_gaps = [right - left for left, right in zip(sample_elapsed, sample_elapsed[1:])]
    soak_start = float(soak["started_elapsed_s"])
    soak_end = float(soak["finished_elapsed_s"])
    soak_samples = [row for row in numeric_samples if soak_start <= row["elapsed_s"] <= soak_end]
    acceptance_samples = [
        row for row in numeric_samples
        if acceptance_window_start <= row["elapsed_s"] <= acceptance_window_end
    ]
    telemetry_ok = (
        len(numeric_samples) == len(sampler.samples)
        and bool(numeric_samples)
        and bool(acceptance_samples)
        and sampler.error is None
        and sampler.rss_source in {
            "Windows vmmemWSL working set",
            "WSL MemTotal-MemAvailable fallback",
        }
        and all(row["gpu_memory_mib"] > 0 for row in numeric_samples)
        and all(row["wsl_rss_kib"] > 0 for row in numeric_samples)
        and all(row["minor_faults"] > 0 for row in numeric_samples)
        and all(row.get("fault_processes", 0) >= 1 for row in sampler.samples)
        and sample_elapsed[0] <= acceptance_window_start
        and sample_elapsed[-1] >= acceptance_window_end
        and max(sample_gaps, default=float("inf")) <= 2.5
        and len(soak_samples) >= int(max(0, elapsed) // 2)
    )
    leak = (
        detect_monotonic_rss_leak(numeric_samples, soak_start, soak_end)
        if telemetry_ok else True
    )
    gates = {
        "pytest": pytest_counts,
        "prompt_cases": api_results["prompt_cases"],
        "steady_decode": api_results["steady_decode"],
        "api": api_results["api"],
        "stability": stability,
        "soak": soak,
        "continuous_run_seconds": round(elapsed, 3),
        "memory_leak_detected": leak,
    }
    resources = {
        "acceptance_window_start_elapsed_s": round(acceptance_window_start, 3),
        "acceptance_window_end_elapsed_s": round(acceptance_window_end, 3),
        "peak_vram_mib": max((row["gpu_memory_mib"] for row in acceptance_samples), default=0),
        "peak_wsl_rss_kib": max((row["wsl_rss_kib"] for row in acceptance_samples), default=0),
        "wsl_rss_source": sampler.rss_source,
        "wsl_swap_kib": max((row["wsl_swap_kib"] for row in acceptance_samples), default=0),
        "page_faults": {
            "minor_delta": max(0, acceptance_samples[-1]["minor_faults"] - acceptance_samples[0]["minor_faults"]) if acceptance_samples else 0,
            "major_delta": max(0, acceptance_samples[-1]["major_faults"] - acceptance_samples[0]["major_faults"]) if acceptance_samples else 0,
        },
        "pcie_traffic": {"status": "unavailable", "reason": "portable nvidia-smi sampling does not expose RX/TX counters"},
        "windows_pagefile_note": "A pre-existing Windows pagefile may be active; the harness does not modify it and does not claim zero host paging.",
    }
    preliminary = {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete",
        "run_id": args.out.name,
        "measured_at": measured_at,
        "source": {
            "validated_runtime_commit": args.expected_commit,
            "runtime_tree_sha256": runtime_tree,
            "upstream_base": UPSTREAM_BASE,
            "release_compatible": False,
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "manifest_sha256": MODEL_MANIFEST_SHA256,
            "file_count": MODEL_FILE_COUNT,
            "total_bytes": MODEL_TOTAL_BYTES,
        },
        "execution": {
            "attention_backend": "qsa_triton",
            "cache_type": "naive",
            "context_tokens": 8192,
            "cuda_graph": False,
            "quantization": "W4A16 compatibility",
            "text_only": True,
            "tp_size": 1,
        },
        "verification": {
            "checkpoint_full_sha256": checkpoint_verified,
            "checkpoint_shape": checkpoint_verified,
            "server_profile": server_profile_verified,
            "launch_attestation": attestation_verified,
            "runtime_clean_tree": runtime_clean_verified,
            "server_port_owner": server_port_verified,
        },
        "gates": gates,
        "resources": resources,
        "benchmarks": {"prompt_classes": api_results["prompt_cases"], "steady_decode": api_results["steady_decode"]},
        "caveats": [
            "The checkpoint declares W4A4, but this release executes the W4A16 compatibility path.",
            "The Windows host pagefile exists; zero host paging is not claimed.",
            "Full-model Transformers logits parity has not been established.",
            "PCIe counters are unavailable in this portable harness and must not be inferred.",
        ],
        "errors": errors,
    }
    objective_pass = (
        not errors
        and pytest_counts["passed"] >= 1454
        and pytest_counts["failed"] == 0
        and {case.get("rendered_prompt_tokens") for case in api_results["prompt_cases"]} >= set(PROMPT_TARGETS)
        and api_results["steady_decode"].get("observed_tokens", 0) >= 256
        and all(api_results["api"].values())
        and stability.get("attempted", 0) >= 100
        and stability.get("succeeded") == stability.get("attempted")
        and soak.get("attempted", 0) >= 61
        and soak.get("succeeded") == soak.get("attempted")
        and elapsed >= 1800
        and telemetry_ok
        and resources["peak_vram_mib"] < 31 * 1024
        and resources["peak_wsl_rss_kib"] < 105 * 1024 * 1024
        and resources["wsl_swap_kib"] == 0
        and not leak
    )
    preliminary["status"] = "verified" if objective_pass else "incomplete"
    write_json(args.out / "summary.json", preliminary)
    write_checksums(args.out)
    try:
        validate_directory(args.out, release=False)
    except EvidenceError as exc:
        print(f"wrote invalid evidence bundle: {exc}", file=sys.stderr)
        return 2
    if objective_pass:
        print(f"objective gates passed; review {args.out}/summary.json and set source.release_compatible=true before release")
        return 0
    print(f"release gates did not pass; inspect {args.out}/summary.json", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
