"""Lazy, replaceable side effects used by the command layer."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .checkpoint import verify_checkpoint, verify_checkpoint_receipt
from .attestation import remove_launch_attestation, write_launch_attestation
from .doctor import collect_doctor_snapshot
from .http import OpenAIHttpClient


@contextmanager
def temporary_environment(values: Mapping[str, str]):
    """Apply model-construction settings for one in-process server launch."""

    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def cuda_toolkit_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve the WSL CUDA toolkit even when the login PATH omits nvcc."""

    current = os.environ if environ is None else environ
    candidates: list[Path] = []
    for name in ("CUDA_HOME", "CUDA_PATH"):
        raw = current.get(name)
        if raw:
            candidates.append(Path(raw))
    candidates.extend((Path("/usr/local/cuda"), Path("/usr/local/cuda-13.3")))
    candidates.extend(
        sorted(Path("/usr/local").glob("cuda-13*"), reverse=True)
        if Path("/usr/local").is_dir()
        else ()
    )
    toolkit = next(
        (path for path in candidates if (path / "bin" / "nvcc").is_file()),
        None,
    )
    if toolkit is None:
        return {}
    binary = str(toolkit / "bin" / "nvcc")
    bin_dir = str(toolkit / "bin")
    old_path = current.get("PATH", os.defpath)
    path = old_path if old_path.split(os.pathsep)[0] == bin_dir else (
        bin_dir + os.pathsep + old_path
    )
    return {
        "CUDA_HOME": str(toolkit),
        "CUDA_PATH": str(toolkit),
        "CUDACXX": binary,
        "PATH": path,
    }


def _launch_server(argv: list[str], prog: str) -> None:
    from freetoken.server import launch_server

    launch_server(argv=argv, prog=prog)


def _snapshot_download(**kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _release_harness(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "release" / "rtx5090_harness.py"
    if not script.is_file():
        raise RuntimeError(
            "q38lab bench requires the source checkout containing "
            "scripts/release/rtx5090_harness.py"
        )
    return subprocess.run([sys.executable, str(script), *argv], cwd=root).returncode


def _probe_ple_streaming(path: Path):
    from freetoken.models.qwen4_exp.ple_io import probe_ple_streaming_capability

    return probe_ple_streaming_capability(path)


def _probe_qsa_native_fast_topk():
    from freetoken.kernel.qsa_fast_topk import probe_qsa_fast_topk_native

    return probe_qsa_fast_topk_native()


def _probe_ple_checkpoint_rows(path: Path) -> dict[str, Any]:
    """Run the source-only release probe before worker processes are spawned."""

    root = Path(__file__).resolve().parents[2]
    release_dir = root / "scripts" / "release"
    script = release_dir / "ple_checkpoint_probe.py"
    if not script.is_file():
        raise RuntimeError(
            "the 256K release profile requires the source checkout containing "
            "scripts/release/ple_checkpoint_probe.py"
        )
    release_path = str(release_dir)
    if release_path not in sys.path:
        sys.path.insert(0, release_path)
    from ple_checkpoint_probe import run_probe

    return run_probe(path)


@dataclass
class Dependencies:
    """Dependency-injection seam for GPU/network-free command tests."""

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    launch_server: Callable[[list[str], str], None] = _launch_server
    snapshot_download: Callable[..., str] = _snapshot_download
    verify_checkpoint: Callable[..., Any] = verify_checkpoint
    verify_checkpoint_receipt: Callable[..., Any] = verify_checkpoint_receipt
    doctor_collector: Callable[..., Any] = collect_doctor_snapshot
    http_client_factory: Callable[..., Any] = OpenAIHttpClient
    release_harness: Callable[[list[str]], int] = _release_harness
    attestation_writer: Callable[..., Path] = write_launch_attestation
    attestation_remover: Callable[..., None] = remove_launch_attestation
    ple_capability_probe: Callable[..., Any] = _probe_ple_streaming
    qsa_native_topk_probe: Callable[..., Any] = _probe_qsa_native_fast_topk
    ple_checkpoint_probe: Callable[..., dict[str, Any]] = _probe_ple_checkpoint_rows
