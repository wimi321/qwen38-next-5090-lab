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
from .constants import PROFILE_NAME, SERVE_PROFILES

GiB = 1024**3
GPU_FREE_FLOOR_MIB = 30_000
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
    requested_profile: str = PROFILE_NAME
    ple_linux: bool = False
    ple_odirect: bool = False
    ple_kernel_io_uring: bool = False
    ple_native_extension: bool = False
    ple_production_ready: bool = False
    ple_capability_detail: str = "not probed"
    memlock_soft_bytes: int | None = None
    memory_budget: dict[str, int] | None = None
    cuda_toolkit_path: str | None = None
    qsa_native_topk_ready: bool = False
    qsa_native_topk_detail: str = "not probed"
    media_doh_fallback_enabled: bool = False
    media_system_dns_hard_cancel_supported: bool = False


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


def _probe_nvcc(*, runner: CommandRunner) -> tuple[str | None, str | None]:
    candidates = ["nvcc", "/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.3/bin/nvcc"]
    if Path("/usr/local").is_dir():
        candidates.extend(
            str(path / "bin" / "nvcc")
            for path in sorted(Path("/usr/local").glob("cuda-13*"), reverse=True)
        )
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        output = _run([candidate, "--version"], runner=runner)
        if output:
            return output, candidate
    return None, None


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
    profile_name: str = PROFILE_NAME,
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

    nvcc_output, nvcc_path = _probe_nvcc(runner=runner)
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
    profile = SERVE_PROFILES.get(profile_name, SERVE_PROFILES[PROFILE_NAME])
    try:
        from freetoken.models.qwen4_exp.ple_io import probe_ple_streaming_capability

        ple_capability = probe_ple_streaming_capability(model_dir)
    except Exception as exc:
        ple_capability = None
        ple_detail = f"{type(exc).__name__}: {str(exc)[:200]}"
    else:
        ple_detail = ple_capability.detail
    if profile.qsa_require_native_topk:
        try:
            from freetoken.kernel.qsa_fast_topk import probe_qsa_fast_topk_native

            qsa_capability = probe_qsa_fast_topk_native()
        except Exception as exc:
            qsa_capability = None
            qsa_detail = f"{type(exc).__name__}: {str(exc)[:200]}"
        else:
            qsa_detail = qsa_capability.detail
    else:
        qsa_capability = None
        qsa_detail = "profile does not require the native SM120 selector"
    try:
        import resource

        memlock_soft = int(resource.getrlimit(resource.RLIMIT_MEMLOCK)[0])
    except (ImportError, OSError, ValueError):
        memlock_soft = None
    ple_staging_bytes = (
        profile.ple_staging_buffers
        * profile.ple_max_batch_pages
        * 4096
    )
    pinned_staging_bytes = ple_staging_bytes + profile.moe_bounce_staging_bytes
    if profile.moe_total_layers:
        moe_locked_bytes = (
            profile.moe_bank_bytes
            * profile.moe_locked_layers
            // profile.moe_total_layers
        )
    else:
        moe_locked_bytes = 0
    moe_registered_bytes = max(0, profile.moe_bank_bytes - moe_locked_bytes)
    # Keep this in lockstep with freetoken.engine.engine._pin_budget_bytes(): WSL
    # shares a WDDM CUDA-host-registration quota, conservatively budgeted at 40%
    # of guest RAM. CUDA pinned allocations are distinct from RLIMIT_MEMLOCK.
    wddm_pin_budget_bytes = int(0.4 * (meminfo.get("MemTotal") or 0))
    wddm_pinned_planned_bytes = moe_registered_bytes + pinned_staging_bytes
    gpu_total_bytes = (
        int(gpu_memory_mib) * 1024**2 if gpu_memory_mib is not None else 0
    )
    gpu_free_bytes = (
        int(gpu_memory_free_mib) * 1024**2
        if gpu_memory_free_mib is not None
        else 0
    )
    gpu_current_used = max(0, gpu_total_bytes - gpu_free_bytes)
    gpu_ratio_managed = int(profile.memory_ratio * gpu_free_bytes)
    gpu_planned_peak = gpu_current_used + gpu_ratio_managed
    gpu_envelope_headroom = (
        profile.gpu_memory_envelope_bytes - gpu_planned_peak
        if profile.gpu_memory_envelope_bytes
        else 0
    )
    memory_budget = {
        "gpu_qsa_cache_bytes": profile.qsa_cache_bytes,
        "gpu_selector_workspace_bytes": profile.selector_workspace_bytes,
        "gpu_vision_weights_bytes": profile.vision_weights_bytes,
        "gpu_fixed_profile_bytes": (
            profile.qsa_cache_bytes
            + profile.selector_workspace_bytes
            + profile.vision_weights_bytes
        ),
        "host_ple_lru_bytes": profile.ple_cache_bytes,
        "host_ple_pinned_staging_bytes": ple_staging_bytes,
        "host_moe_bounce_staging_bytes": profile.moe_bounce_staging_bytes,
        "host_pinned_staging_bytes": pinned_staging_bytes,
        "host_moe_registered_banks_bytes": moe_registered_bytes,
        "host_wddm_pinned_planned_bytes": wddm_pinned_planned_bytes,
        "host_wddm_pin_budget_bytes": wddm_pin_budget_bytes,
        "host_moe_os_locked_bytes": moe_locked_bytes,
        "gpu_memory_envelope_bytes": profile.gpu_memory_envelope_bytes,
        "gpu_runtime_reserve_bytes": profile.gpu_runtime_reserve_bytes,
        "gpu_current_used_bytes": gpu_current_used,
        "gpu_ratio_managed_bytes": gpu_ratio_managed,
        "gpu_planned_peak_bytes": gpu_planned_peak,
        "gpu_envelope_headroom_bytes": gpu_envelope_headroom,
    }
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
        requested_profile=profile.name,
        ple_linux=bool(getattr(ple_capability, "linux", False)),
        ple_odirect=bool(getattr(ple_capability, "odirect_read", False)),
        ple_kernel_io_uring=bool(
            getattr(ple_capability, "kernel_io_uring", False)
        ),
        ple_native_extension=bool(
            getattr(ple_capability, "native_extension", False)
        ),
        ple_production_ready=bool(
            getattr(ple_capability, "production_ready", False)
        ),
        ple_capability_detail=ple_detail,
        memlock_soft_bytes=memlock_soft,
        memory_budget=memory_budget,
        cuda_toolkit_path=nvcc_path,
        qsa_native_topk_ready=bool(
            getattr(qsa_capability, "production_ready", False)
        ),
        qsa_native_topk_detail=qsa_detail,
        media_doh_fallback_enabled=os.getenv("Q38LAB_DOH_FALLBACK") == "1",
        # CPython/libc expose no portable hard-cancel for a getaddrinfo already
        # running inside NSS. The media layer deadline-bounds a capped daemon
        # helper instead; keep this limitation explicit in machine output.
        media_system_dns_hard_cancel_supported=False,
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

    profile = SERVE_PROFILES.get(snapshot.requested_profile)
    requires_native_ple = bool(profile and profile.ple_require_native_io_uring)
    requires_native_topk = bool(profile and profile.qsa_require_native_topk)
    if profile and profile.load_vision:
        add(
            "media_dns_policy",
            (
                "pass"
                if snapshot.media_system_dns_hard_cancel_supported
                else "warn"
            ),
            "fixed pinned DoH fallback "
            f"{'enabled' if snapshot.media_doh_fallback_enabled else 'disabled'} via "
            "Q38LAB_DOH_FALLBACK; system getaddrinfo uses deadline-bounded, "
            "four-slot soft cancellation (no portable hard cancel)",
        )
    ple_status: Status = (
        "pass"
        if snapshot.ple_production_ready
        else ("fail" if requires_native_ple else "warn")
    )
    add(
        "ple_streaming",
        ple_status,
        f"profile={snapshot.requested_profile}; O_DIRECT={snapshot.ple_odirect}; "
        f"kernel_io_uring={snapshot.ple_kernel_io_uring}; "
        f"native_extension={snapshot.ple_native_extension}; "
        f"{snapshot.ple_capability_detail}",
    )
    add(
        "qsa_native_fast_topk",
        (
            "pass"
            if snapshot.qsa_native_topk_ready
            else ("fail" if requires_native_topk else "warn")
        ),
        f"profile={snapshot.requested_profile}; {snapshot.qsa_native_topk_detail}",
    )
    budget = snapshot.memory_budget or {}
    planned_peak = int(budget.get("gpu_planned_peak_bytes", 0))
    envelope = int(budget.get("gpu_memory_envelope_bytes", 0))
    budget_fits = bool(budget) and (
        not requires_native_ple
        or (envelope > 0 and planned_peak > 0 and planned_peak < envelope)
    )
    add(
        "profile_memory_budget",
        "pass" if budget_fits else ("fail" if requires_native_ple else "warn"),
        ", ".join(
            f"{name}={value / GiB:.3f}GiB" for name, value in sorted(budget.items())
        ) if budget else "memory budget unavailable",
    )
    if requires_native_ple:
        planned_pinned = int(budget.get("host_wddm_pinned_planned_bytes", 0))
        pin_budget = int(budget.get("host_wddm_pin_budget_bytes", 0))
        pin_ok = pin_budget > 0 and planned_pinned < pin_budget
        add(
            "wddm_pin_budget",
            "pass" if pin_ok else "fail",
            f"registered banks + CUDA pinned staging require {planned_pinned} bytes; "
            f"40% WSL budget is {pin_budget} bytes",
        )
        required_locked = int(budget.get("host_moe_os_locked_bytes", 0))
        memlock = snapshot.memlock_soft_bytes
        # RLIM_INFINITY is commonly represented by -1. This limit applies to
        # mlock'd CPU expert banks, not cudaMallocHost/cudaHostRegister storage.
        lock_ok = memlock == -1 or (
            memlock is not None and memlock >= required_locked
        )
        add(
            "locked_memory",
            "pass" if lock_ok else "fail",
            f"soft limit {memlock}; OS-locked MoE banks require {required_locked} bytes",
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
        "pass" if gpu_free is not None and gpu_free >= GPU_FREE_FLOOR_MIB else "fail",
        f"{gpu_free or 0} MiB free (expected at least {GPU_FREE_FLOOR_MIB} MiB "
        "before launch; WDDM display reservation is allowed)",
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
        f"nvcc {snapshot.nvcc_version or 'not found'} "
        f"({snapshot.cuda_toolkit_path or 'PATH/fallback not found'}); torch CUDA "
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
