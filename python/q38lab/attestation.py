"""Atomic launch attestations consumed by the local release harness.

The FreeToken server is launched in-process, so ``/proc/<pid>/cmdline`` only
shows the public ``q38lab serve`` invocation.  This file records the exact
resolved low-level arguments without treating self-reported data as sufficient:
the release harness independently checks the PID start time, Git checkout,
model path, listening port, API model id, and hardware before accepting it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import ResolvedServeConfig


SCHEMA_VERSION = "1.0"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def default_attestation_path(port: int) -> Path:
    return Path.home() / ".cache" / "q38lab" / f"serve-{port}.json"


def _proc_start_ticks(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError(f"cannot read Linux process start time for pid {pid}") from exc


def _source_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / ".git").exists() else None


def _git_identity(root: Path | None) -> tuple[str | None, bool]:
    if root is None:
        return None, False
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, False
    return (commit if COMMIT_RE.fullmatch(commit) else None, status == "")


def build_launch_attestation(
    config: ResolvedServeConfig,
    argv: list[str],
    *,
    pid: int | None = None,
) -> dict[str, Any]:
    process_id = os.getpid() if pid is None else pid
    source_root = _source_root()
    commit, clean = _git_identity(source_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "pid": process_id,
        "proc_start_ticks": _proc_start_ticks(process_id),
        "runtime_commit": commit,
        "clean_tree": clean,
        "source_root": str(source_root.resolve()) if source_root else None,
        "model_realpath": str(config.model_dir.resolve()),
        "resolved_config": config.public_dict(),
        "argv": list(argv),
    }


def write_launch_attestation(
    config: ResolvedServeConfig,
    argv: list[str],
    *,
    target: Path | None = None,
) -> Path:
    path = (target or default_attestation_path(config.port)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = build_launch_attestation(config, argv)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def remove_launch_attestation(path: Path, *, pid: int | None = None) -> None:
    """Remove only the caller's still-current attestation, never another run's."""

    process_id = os.getpid() if pid is None else pid
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if value.get("pid") == process_id:
        path.unlink(missing_ok=True)
