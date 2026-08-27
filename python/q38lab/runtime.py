"""Lazy, replaceable side effects used by the command layer."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .checkpoint import verify_checkpoint, verify_checkpoint_receipt
from .attestation import remove_launch_attestation, write_launch_attestation
from .doctor import collect_doctor_snapshot
from .http import OpenAIHttpClient


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
