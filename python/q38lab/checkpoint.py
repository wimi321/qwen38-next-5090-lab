"""Pinned checkpoint download and local integrity verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .constants import (
    EXPECTED_FILE_COUNT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SAFETENSORS_COUNT,
    EXPECTED_TOTAL_BYTES,
    MODEL_REPO,
    MODEL_REVISION,
)


class CheckpointVerificationError(RuntimeError):
    pass


RECEIPT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CheckpointExpectation:
    file_count: int = EXPECTED_FILE_COUNT
    total_bytes: int = EXPECTED_TOTAL_BYTES
    safetensors_count: int = EXPECTED_SAFETENSORS_COUNT
    manifest_sha256: str = EXPECTED_MANIFEST_SHA256


@dataclass(frozen=True)
class CheckpointVerification:
    root: Path
    file_count: int
    total_bytes: int
    safetensors_count: int
    full_verify: bool
    manifest_sha256: str | None
    verification_mode: str = "shape-only"

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "safetensors_count": self.safetensors_count,
            "full_verify": self.full_verify,
            "manifest_sha256": self.manifest_sha256,
            "verification_mode": self.verification_mode,
        }


def iter_checkpoint_files(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield canonical relative paths, excluding Hugging Face's local cache."""

    if not root.is_dir():
        return
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if ".cache" in rel.parts:
            continue
        files.append((rel.as_posix(), path))
    yield from sorted(files, key=lambda item: item[0].encode("utf-8"))


def _file_sha256(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_receipt_path(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    return resolved.parent / f".{resolved.name}.q38lab-verification.json"


def _stat_fingerprint(files: Iterable[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for rel, path in files:
        stat = path.stat()
        digest.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _control_hashes(files: Iterable[tuple[str, Path]]) -> dict[str, str]:
    """Hash every non-weight file that can influence parsing or code execution."""

    return {
        rel: _file_sha256(path)
        for rel, path in files
        if not rel.endswith((".safetensors", ".bin", ".gguf"))
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_manifest_sha256(files: Iterable[tuple[str, Path]]) -> str:
    """Hash the same manifest produced by the documented GNU sha256sum recipe.

    Each line is ``<file-sha256>  ./<relative-posix-path>\n``.  The manifest is
    kept outside the checkpoint and is therefore not self-referential.
    """

    manifest = hashlib.sha256()
    for rel, path in files:
        line = f"{_file_sha256(path)}  ./{rel}\n"
        manifest.update(line.encode("utf-8"))
    return manifest.hexdigest()


def verify_checkpoint(
    root: Path,
    *,
    full: bool = False,
    expectation: CheckpointExpectation = CheckpointExpectation(),
) -> CheckpointVerification:
    root = root.expanduser()
    if not root.is_dir():
        raise CheckpointVerificationError(f"checkpoint directory does not exist: {root}")

    files = list(iter_checkpoint_files(root))
    file_count = len(files)
    total_bytes = sum(path.stat().st_size for _, path in files)
    safetensors_count = sum(rel.endswith(".safetensors") for rel, _ in files)
    mismatches: list[str] = []
    if file_count != expectation.file_count:
        mismatches.append(f"files {file_count} != {expectation.file_count}")
    if total_bytes != expectation.total_bytes:
        mismatches.append(f"bytes {total_bytes} != {expectation.total_bytes}")
    if safetensors_count != expectation.safetensors_count:
        mismatches.append(
            f"safetensors {safetensors_count} != {expectation.safetensors_count}"
        )
    if mismatches:
        raise CheckpointVerificationError("checkpoint shape mismatch: " + "; ".join(mismatches))

    manifest_sha256 = canonical_manifest_sha256(files) if full else None
    if full and manifest_sha256 != expectation.manifest_sha256:
        raise CheckpointVerificationError(
            "checkpoint manifest mismatch: "
            f"{manifest_sha256} != {expectation.manifest_sha256}"
        )
    return CheckpointVerification(
        root=root,
        file_count=file_count,
        total_bytes=total_bytes,
        safetensors_count=safetensors_count,
        full_verify=full,
        manifest_sha256=manifest_sha256,
        verification_mode="full-sha256" if full else "shape-only",
    )


def write_verification_receipt(
    verification: CheckpointVerification,
    *,
    trusted_pinned_download: bool,
    expectation: CheckpointExpectation = CheckpointExpectation(),
) -> Path:
    """Record a pinned-download or full-hash proof outside the checkpoint tree."""

    files = list(iter_checkpoint_files(verification.root))
    if not trusted_pinned_download and not verification.full_verify:
        raise CheckpointVerificationError(
            "a receipt requires a pinned download or full SHA-256 verification"
        )
    mode = "full-sha256" if verification.full_verify else "pinned-hf-download"
    document = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "root_realpath": str(verification.root.resolve()),
        "mode": mode,
        "file_count": verification.file_count,
        "total_bytes": verification.total_bytes,
        "safetensors_count": verification.safetensors_count,
        "expected_manifest_sha256": expectation.manifest_sha256,
        "verified_manifest_sha256": verification.manifest_sha256,
        "stat_fingerprint": _stat_fingerprint(files),
        "control_hashes": _control_hashes(files),
    }
    target = checkpoint_receipt_path(verification.root)
    _write_json_atomic(target, document)
    return target


def verify_checkpoint_receipt(
    root: Path,
    *,
    require_full: bool = False,
    expectation: CheckpointExpectation = CheckpointExpectation(),
) -> CheckpointVerification:
    """Validate a prior pinned-download proof and detect subsequent local drift."""

    resolved = root.expanduser().resolve()
    receipt = checkpoint_receipt_path(resolved)
    try:
        document = json.loads(receipt.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CheckpointVerificationError(
            f"missing pinned-checkpoint receipt {receipt}; run q38lab download "
            "--accept-qwen-license first"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointVerificationError(f"cannot read checkpoint receipt {receipt}: {exc}") from exc
    if not isinstance(document, dict):
        raise CheckpointVerificationError("checkpoint receipt must be a JSON object")
    expected_identity = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "root_realpath": str(resolved),
        "file_count": expectation.file_count,
        "total_bytes": expectation.total_bytes,
        "safetensors_count": expectation.safetensors_count,
        "expected_manifest_sha256": expectation.manifest_sha256,
    }
    for key, wanted in expected_identity.items():
        if document.get(key) != wanted:
            raise CheckpointVerificationError(
                f"checkpoint receipt {key} mismatch: {document.get(key)!r} != {wanted!r}"
            )
    mode = document.get("mode")
    if mode not in {"pinned-hf-download", "full-sha256"}:
        raise CheckpointVerificationError(f"unsupported checkpoint receipt mode: {mode!r}")
    if require_full and mode != "full-sha256":
        raise CheckpointVerificationError(
            "release verification requires q38lab download --full-verify"
        )
    quick = verify_checkpoint(resolved, full=False, expectation=expectation)
    files = list(iter_checkpoint_files(resolved))
    if document.get("stat_fingerprint") != _stat_fingerprint(files):
        raise CheckpointVerificationError(
            "checkpoint files changed after the pinned verification receipt was written"
        )
    controls = document.get("control_hashes")
    if not isinstance(controls, dict) or controls != _control_hashes(files):
        raise CheckpointVerificationError(
            "checkpoint configuration/tokenizer/control-file hashes changed"
        )
    manifest = document.get("verified_manifest_sha256")
    if mode == "full-sha256" and manifest != expectation.manifest_sha256:
        raise CheckpointVerificationError("full checkpoint receipt has the wrong manifest digest")
    return CheckpointVerification(
        root=quick.root,
        file_count=quick.file_count,
        total_bytes=quick.total_bytes,
        safetensors_count=quick.safetensors_count,
        full_verify=mode == "full-sha256",
        manifest_sha256=manifest if isinstance(manifest, str) else None,
        verification_mode=mode,
    )


SnapshotDownload = Callable[..., str]


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_download_destination(
    destination: Path,
    *,
    expectation: CheckpointExpectation = CheckpointExpectation(),
    enforce_capacity: bool = True,
) -> None:
    """Reject source-tree destinations and undersized filesystems before I/O."""

    resolved = destination.expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if _inside(resolved, source_root):
        raise CheckpointVerificationError(
            f"refusing to download model files inside the project source tree: {resolved}"
        )
    for ancestor in (resolved, *resolved.parents):
        if (ancestor / ".git").exists():
            raise CheckpointVerificationError(
                f"refusing to download model files inside Git worktree {ancestor}"
            )
    if not enforce_capacity:
        return
    nearest = resolved
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent
    existing_bytes = 0
    if resolved.is_dir():
        existing_bytes = sum(path.stat().st_size for _, path in iter_checkpoint_files(resolved))
    remaining = max(0, expectation.total_bytes - existing_bytes)
    safety_margin = 10 * 1024**3
    available = shutil.disk_usage(nearest).free
    required = remaining + safety_margin
    if available < required:
        raise CheckpointVerificationError(
            "insufficient free space for the pinned checkpoint: "
            f"need {required} bytes including a 10 GiB safety margin, "
            f"have {available} bytes"
        )


def download_checkpoint(
    model_dir: Path,
    *,
    full_verify: bool,
    snapshot_download: SnapshotDownload,
    expectation: CheckpointExpectation = CheckpointExpectation(),
    enforce_capacity: bool = True,
) -> CheckpointVerification:
    """Download exactly the audited commit, then reject any shape/hash drift."""

    destination = model_dir.expanduser()
    validate_download_destination(
        destination, expectation=expectation, enforce_capacity=enforce_capacity,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=str(destination),
        ))
    except Exception as exc:
        raise CheckpointVerificationError(
            "pinned checkpoint download failed "
            f"({type(exc).__name__}); no fallback repository or revision was attempted"
        ) from exc
    # Hugging Face normally returns local_dir.  Accept an equivalent resolved path
    # but never verify an unrelated cache snapshot by accident.
    if downloaded.resolve() != destination.resolve():
        raise CheckpointVerificationError(
            f"downloader returned unexpected path {downloaded}; expected {destination}"
        )
    verification = verify_checkpoint(destination, full=full_verify, expectation=expectation)
    write_verification_receipt(
        verification, trusted_pinned_download=True, expectation=expectation,
    )
    return CheckpointVerification(
        root=verification.root,
        file_count=verification.file_count,
        total_bytes=verification.total_bytes,
        safetensors_count=verification.safetensors_count,
        full_verify=verification.full_verify,
        manifest_sha256=verification.manifest_sha256,
        verification_mode=("full-sha256" if full_verify else "pinned-hf-download"),
    )
