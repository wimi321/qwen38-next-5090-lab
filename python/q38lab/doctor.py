"""Read-only host, CUDA and checkpoint preflight."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from .checkpoint import CheckpointVerificationError, verify_checkpoint_receipt

GiB = 1024**3
Status = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class DoctorSnapshot:
    is_wsl2: bool
    kernel_release: str
    os_id: str | None
    os_version_id: str | None
    source_dir: str
    source_filesystem: str | None
    model_dir: str
    model_filesystem: str | None
    gpu_name: str | None
    compute_capability: str | None
    driver_version: str | None
    gpu_memory_mib: int | None
    gpu_memory_free_mib: int | None
    nvcc_version: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    torch_cuda_available: bool
    torch_cuda_probe: str
    triton_version: str | None
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    swap_total_bytes: int | None
    disk_free_bytes: int | None
    model_exists: bool
    checkpoint_file_count: int | None
    checkpoint_total_bytes: int | None
    checkpoint_safetensors_count: int | None
    checkpoint_error: str | None
    checkpoint_verification_mode: str | None
    q38lab_distribution_version: str | None
    upstream_freetoken_distribution_version: str | None
    port: int
    port_available: bool


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    schema_version: int
    ready: bool
    checks: tuple[DoctorCheck, ...]
    snapshot: DoctorSnapshot

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
            "snapshot": asdict(self.snapshot),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _run(command: list[str], *, runner: CommandRunner) -> str | None:
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _nearest_existing(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _filesystem(path: Path, *, runner: CommandRunner) -> str | None:
    output = _run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(_nearest_existing(path))],
        runner=runner,
    )
    return output.splitlines()[0].strip() if output else None


def _meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            match = re.search(r"(\d+)", raw)
            if match:
                result[key] = int(match.group(1)) * 1024
    except (OSError, ValueError):
        pass
    return result


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.startswith("#"):
                continue
            key, raw = line.split("=", 1)
            values[key] = raw.strip().strip('"')
    except OSError:
        pass
    return values


def _numeric_version(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", value))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_cuda_version() -> str | None:
    try:
        import torch

        return str(torch.version.cuda) if torch.version.cuda else None
    except Exception:  # CUDA probing must never crash doctor
        return None


def _torch_cuda_runtime_probe() -> tuple[bool, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is false"
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        probe = torch.ones(1, device="cuda") + 1
        torch.cuda.synchronize(0)
        if probe.item() != 2:
            return False, "CUDA tensor probe returned an unexpected value"
        return True, f"{name}; SM {capability[0]}.{capability[1]}; tensor probe passed"
    except Exception as exc:  # doctor must report a failed probe, never crash
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"


def _port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def collect_doctor_snapshot(
    *,
    source_dir: Path,
    model_dir: Path,
    port: int,
    runner: CommandRunner = subprocess.run,
) -> DoctorSnapshot:
    kernel_release = platform.release()
    marker = kernel_release.lower()
    is_wsl2 = "microsoft-standard" in marker or "wsl2" in marker
    os_release = _os_release()

    gpu_output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
    )
    gpu_name = compute_capability = driver_version = None
    gpu_memory_mib: int | None = None
    gpu_memory_free_mib: int | None = None
    if gpu_output:
        fields = [part.strip() for part in gpu_output.splitlines()[0].split(",")]
        if len(fields) >= 5:
            gpu_name, compute_capability, driver_version = fields[:3]
            try:
                gpu_memory_mib = int(float(fields[3]))
                gpu_memory_free_mib = int(float(fields[4]))
            except ValueError:
                gpu_memory_mib = None
                gpu_memory_free_mib = None

    nvcc_output = _run(["nvcc", "--version"], runner=runner)
    nvcc_match = re.search(r"release\s+([\d.]+)", nvcc_output or "", re.IGNORECASE)
    nvcc_version = nvcc_match.group(1) if nvcc_match else None
    meminfo = _meminfo()
    disk_target = _nearest_existing(model_dir)
    try:
        disk_free = shutil.disk_usage(disk_target).free
    except OSError:
        disk_free = None

    checkpoint_file_count = checkpoint_total_bytes = checkpoint_safetensors_count = None
    checkpoint_error = None
    checkpoint_verification_mode = None
    if model_dir.is_dir():
        try:
            verification = verify_checkpoint_receipt(model_dir, require_full=False)
            checkpoint_file_count = verification.file_count
            checkpoint_total_bytes = verification.total_bytes
            checkpoint_safetensors_count = verification.safetensors_count
            checkpoint_verification_mode = verification.verification_mode
        except CheckpointVerificationError as exc:
            checkpoint_error = str(exc)

    torch_cuda_available, torch_cuda_probe = _torch_cuda_runtime_probe()
    return DoctorSnapshot(
        is_wsl2=is_wsl2,
        kernel_release=kernel_release,
        os_id=os_release.get("ID"),
        os_version_id=os_release.get("VERSION_ID"),
        source_dir=str(source_dir.expanduser()),
        source_filesystem=_filesystem(source_dir, runner=runner),
        model_dir=str(model_dir.expanduser()),
        model_filesystem=_filesystem(model_dir, runner=runner),
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        driver_version=driver_version,
        gpu_memory_mib=gpu_memory_mib,
        gpu_memory_free_mib=gpu_memory_free_mib,
        nvcc_version=nvcc_version,
        torch_version=_package_version("torch"),
        torch_cuda_version=_torch_cuda_version(),
        torch_cuda_available=torch_cuda_available,
        torch_cuda_probe=torch_cuda_probe,
        triton_version=_package_version("triton"),
        memory_total_bytes=meminfo.get("MemTotal"),
        memory_available_bytes=meminfo.get("MemAvailable"),
        swap_total_bytes=meminfo.get("SwapTotal"),
        disk_free_bytes=disk_free,
        model_exists=model_dir.is_dir(),
        checkpoint_file_count=checkpoint_file_count,
        checkpoint_total_bytes=checkpoint_total_bytes,
        checkpoint_safetensors_count=checkpoint_safetensors_count,
        checkpoint_error=checkpoint_error,
        checkpoint_verification_mode=checkpoint_verification_mode,
        q38lab_distribution_version=_package_version("qwen38-next-5090-lab"),
        upstream_freetoken_distribution_version=_package_version("freetoken"),
        port=port,
        port_available=_port_available(port),
    )


def evaluate_doctor(snapshot: DoctorSnapshot) -> DoctorReport:
    checks: list[DoctorCheck] = []

    def add(name: str, status: Status, detail: str) -> None:
        checks.append(DoctorCheck(name, status, detail))

    add(
        "wsl2",
        "pass" if snapshot.is_wsl2 else "fail",
        f"kernel {snapshot.kernel_release}",
    )
    add(
        "distribution",
        "pass" if snapshot.os_id == "ubuntu" and snapshot.os_version_id == "24.04" else "fail",
        f"{snapshot.os_id or 'unknown'} {snapshot.os_version_id or 'unknown'} "
        "(expected Ubuntu 24.04)",
    )
    add(
        "source_filesystem",
        "pass" if snapshot.source_filesystem == "ext4" else "fail",
        f"{snapshot.source_dir}: {snapshot.source_filesystem or 'unknown'}",
    )
    add(
        "model_filesystem",
        "pass" if snapshot.model_filesystem == "ext4" else "fail",
        f"{snapshot.model_dir}: {snapshot.model_filesystem or 'unknown'}",
    )

    gpu_ok = bool(
        snapshot.gpu_name
        and "RTX 5090" in snapshot.gpu_name.upper()
        and snapshot.gpu_memory_mib is not None
        and 31 * 1024 <= snapshot.gpu_memory_mib <= 33 * 1024
    )
    add(
        "gpu",
        "pass" if gpu_ok else "fail",
        f"{snapshot.gpu_name or 'not found'}; {snapshot.gpu_memory_mib or 0} MiB total; "
        f"driver {snapshot.driver_version or 'unknown'}",
    )
    driver_ok = _numeric_version(snapshot.driver_version) >= (591, 86)
    add(
        "driver",
        "pass" if driver_ok else "fail",
        f"{snapshot.driver_version or 'unknown'} (validated floor 591.86)",
    )
    gpu_free = snapshot.gpu_memory_free_mib
    add(
        "gpu_memory_available",
        "pass" if gpu_free is not None and gpu_free >= 30 * 1024 else "fail",
        f"{gpu_free or 0} MiB free (expected at least 30720 MiB before launch)",
    )
    capability = snapshot.compute_capability
    add(
        "compute_capability",
        "pass" if capability in {"12.0", "12"} else "fail",
        f"SM {capability or 'unknown'} (expected SM 12.0)",
    )
    nvcc_ok = bool(snapshot.nvcc_version and snapshot.nvcc_version.split(".")[0] == "13")
    torch_cuda_ok = bool(
        snapshot.torch_cuda_version and snapshot.torch_cuda_version.split(".")[0] == "13"
    )
    add(
        "cuda_toolkit",
        "pass" if nvcc_ok and torch_cuda_ok else "fail",
        f"nvcc {snapshot.nvcc_version or 'not found'}; torch CUDA "
        f"{snapshot.torch_cuda_version or 'unknown'}",
    )
    add(
        "torch_cuda_runtime",
        "pass" if snapshot.torch_cuda_available else "fail",
        snapshot.torch_cuda_probe,
    )
    stack_ok = bool(
        snapshot.torch_version
        and snapshot.torch_version.startswith("2.11")
        and snapshot.triton_version == "3.6.0"
    )
    add(
        "python_stack",
        "pass" if stack_ok else "fail",
        f"torch {snapshot.torch_version or 'not found'} (expected 2.11.x); "
        f"triton {snapshot.triton_version or 'not found'} (expected 3.6.0)",
    )
    official_dist = snapshot.upstream_freetoken_distribution_version
    project_dist = snapshot.q38lab_distribution_version
    distribution_status: Status = "fail" if official_dist else ("pass" if project_dist else "warn")
    add(
        "distribution_identity",
        distribution_status,
        (
            f"qwen38-next-5090-lab {project_dist or 'not installed'}; "
            f"upstream freetoken distribution {official_dist or 'not installed'}"
        ),
    )

    memory = snapshot.memory_total_bytes
    add(
        "memory",
        "pass" if (
            memory is not None and memory >= 100 * GiB
            and snapshot.memory_available_bytes is not None
            and snapshot.memory_available_bytes >= 80 * GiB
        ) else "fail",
        f"{(memory or 0) / GiB:.2f} GiB total; "
        f"{(snapshot.memory_available_bytes or 0) / GiB:.2f} GiB available "
        "(expected at least 80 GiB before launch)",
    )
    swap = snapshot.swap_total_bytes
    add(
        "swap",
        "pass" if swap == 0 else "fail",
        f"{(swap or 0) / GiB:.2f} GiB configured (expected 0)",
    )
    disk_floor = 10 * GiB if snapshot.model_exists else 150 * GiB
    disk = snapshot.disk_free_bytes
    add(
        "disk",
        "pass" if disk is not None and disk >= disk_floor else "fail",
        f"{(disk or 0) / GiB:.2f} GiB free; floor {disk_floor / GiB:.0f} GiB",
    )

    checkpoint_ok = snapshot.model_exists and snapshot.checkpoint_error is None
    checkpoint_detail = (
        f"{snapshot.checkpoint_file_count} files, {snapshot.checkpoint_total_bytes} bytes, "
        f"{snapshot.checkpoint_safetensors_count} safetensors; "
        f"proof={snapshot.checkpoint_verification_mode}"
        if checkpoint_ok
        else snapshot.checkpoint_error or "checkpoint not downloaded"
    )
    add("checkpoint", "pass" if checkpoint_ok else "fail", checkpoint_detail)
    add(
        "port",
        "pass" if snapshot.port_available else "fail",
        f"127.0.0.1:{snapshot.port} is "
        f"{'available' if snapshot.port_available else 'already in use'}",
    )

    return DoctorReport(
        schema_version=1,
        ready=not any(check.status == "fail" for check in checks),
        checks=tuple(checks),
        snapshot=snapshot,
    )


def format_doctor_report(report: DoctorReport) -> str:
    lines = [f"q38lab doctor: {'READY' if report.ready else 'NOT READY'}"]
    glyph = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    for check in report.checks:
        lines.append(f"[{glyph[check.status]}] {check.name}: {check.detail}")
    return "\n".join(lines)
