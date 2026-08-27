from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from q38lab.checkpoint import (
    CheckpointExpectation,
    CheckpointVerificationError,
    canonical_manifest_sha256,
    download_checkpoint,
    checkpoint_receipt_path,
    iter_checkpoint_files,
    verify_checkpoint,
    verify_checkpoint_receipt,
    write_verification_receipt,
)
from q38lab.constants import MODEL_REPO, MODEL_REVISION


def _fixture(root: Path) -> CheckpointExpectation:
    (root / "nested").mkdir(parents=True)
    (root / ".cache").mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "nested" / "model.safetensors").write_bytes(b"weights")
    (root / ".cache" / "ignored").write_bytes(b"not checkpoint data")
    files = list(iter_checkpoint_files(root))
    return CheckpointExpectation(
        file_count=2,
        total_bytes=sum(path.stat().st_size for _, path in files),
        safetensors_count=1,
        manifest_sha256=canonical_manifest_sha256(files),
    )


def test_quick_and_full_checkpoint_verification(tmp_path):
    expectation = _fixture(tmp_path)
    quick = verify_checkpoint(tmp_path, expectation=expectation)
    assert quick.file_count == 2
    assert quick.manifest_sha256 is None

    full = verify_checkpoint(tmp_path, full=True, expectation=expectation)
    assert full.manifest_sha256 == expectation.manifest_sha256
    assert full.total_bytes == expectation.total_bytes


def test_full_checkpoint_verification_rejects_manifest_drift(tmp_path):
    expectation = _fixture(tmp_path)
    bad = CheckpointExpectation(
        file_count=expectation.file_count,
        total_bytes=expectation.total_bytes,
        safetensors_count=expectation.safetensors_count,
        manifest_sha256="0" * 64,
    )
    with pytest.raises(CheckpointVerificationError, match="manifest mismatch"):
        verify_checkpoint(tmp_path, full=True, expectation=bad)


def test_receipt_binds_pinned_download_and_rejects_later_drift(tmp_path):
    expectation = _fixture(tmp_path)
    verification = verify_checkpoint(tmp_path, full=True, expectation=expectation)
    write_verification_receipt(
        verification, trusted_pinned_download=False, expectation=expectation,
    )
    result = verify_checkpoint_receipt(tmp_path, require_full=True, expectation=expectation)
    assert result.verification_mode == "full-sha256"
    (tmp_path / "config.json").write_text("[]\n", encoding="utf-8")
    # Preserve aggregate bytes to demonstrate that receipt verification is stronger
    # than the shape-only check.
    assert sum(path.stat().st_size for _, path in iter_checkpoint_files(tmp_path)) == expectation.total_bytes
    with pytest.raises(CheckpointVerificationError, match="changed"):
        verify_checkpoint_receipt(tmp_path, expectation=expectation)


def test_receipt_is_kept_outside_checkpoint_shape(tmp_path):
    expectation = _fixture(tmp_path)
    verification = verify_checkpoint(tmp_path, full=True, expectation=expectation)
    receipt = write_verification_receipt(
        verification, trusted_pinned_download=False, expectation=expectation,
    )
    assert receipt == checkpoint_receipt_path(tmp_path)
    assert receipt.parent == tmp_path.parent
    assert len(list(iter_checkpoint_files(tmp_path))) == expectation.file_count


def test_downloader_is_pinned_to_audited_repo_and_revision(tmp_path):
    destination = tmp_path / "checkpoint"
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        destination.mkdir()
        expectation_holder.append(_fixture(destination))
        return str(destination)

    expectation_holder: list[CheckpointExpectation] = []
    # Populate once so the expectation can be passed without weakening production defaults.
    destination.mkdir()
    expectation = _fixture(destination)
    for path in sorted(destination.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    destination.rmdir()

    result = download_checkpoint(
        destination,
        full_verify=True,
        snapshot_download=fake_snapshot_download,
        expectation=expectation,
        enforce_capacity=False,
    )
    assert result.full_verify
    assert seen == {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "local_dir": str(destination),
    }


def test_downloader_refuses_source_tree_before_network_or_disk_writes(tmp_path):
    calls = []
    source_root = Path(__file__).resolve().parents[2]
    target = source_root / "models-should-never-exist"
    with pytest.raises(CheckpointVerificationError, match="source tree"):
        download_checkpoint(
            target,
            full_verify=False,
            snapshot_download=lambda **kwargs: calls.append(kwargs),
            enforce_capacity=False,
        )
    assert calls == []
    assert not target.exists()


def test_downloader_wraps_network_failures_without_trying_another_revision(tmp_path):
    def fail(**kwargs):
        assert kwargs["repo_id"] == MODEL_REPO
        assert kwargs["revision"] == MODEL_REVISION
        raise ConnectionError("private transport detail")

    with pytest.raises(
        CheckpointVerificationError,
        match="pinned checkpoint download failed .*no fallback",
    ):
        download_checkpoint(
            tmp_path / "checkpoint",
            full_verify=False,
            snapshot_download=fail,
            enforce_capacity=False,
        )


def test_manifest_format_matches_documented_sha256sum_recipe(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    file_hash = hashlib.sha256(b"abc").hexdigest()
    expected = hashlib.sha256(f"{file_hash}  ./a.txt\n".encode()).hexdigest()
    assert canonical_manifest_sha256(iter_checkpoint_files(tmp_path)) == expected
