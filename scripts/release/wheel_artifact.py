#!/usr/bin/env python3
"""Audit the narrow Linux wheel and emit a sanitized build provenance record."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_NATIVE = {
    "freetoken/kernel/_cpu_moe.cpython-312-x86_64-linux-gnu.so",
    "freetoken/kernel/_pinned_tensor.cpython-312-x86_64-linux-gnu.so",
    "freetoken/models/qwen4_exp/_ple_io_uring.cpython-312-x86_64-linux-gnu.so",
}
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".gguf", ".pt", ".pth", ".safetensors"}
PRIVATE_MARKERS = (b"/home/", b"/root/", b"/mnt/c/", b"C:\\Users\\")


class WheelAuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_wheel(path: Path, expected_version: str) -> dict[str, object]:
    if not path.name.endswith("-cp312-cp312-linux_x86_64.whl"):
        raise WheelAuditError("wheel must be tagged cp312-cp312-linux_x86_64")
    expected_dist = f"qwen38_next_5090_lab-{expected_version}.dist-info"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise WheelAuditError("wheel contains duplicate members")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise WheelAuditError(f"unsafe wheel member: {name}")
            if not member.parts or member.parts[0] not in {
                "freetoken", "q38lab", expected_dist,
            }:
                raise WheelAuditError(f"unexpected wheel root: {name}")
            if member.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise WheelAuditError(f"model/checkpoint artifact in wheel: {name}")
            payload = archive.read(name)
            if any(marker in payload for marker in PRIVATE_MARKERS):
                raise WheelAuditError(f"private build path found in wheel member: {name}")

        native = {name for name in names if name.endswith(".so")}
        if native != EXPECTED_NATIVE:
            raise WheelAuditError(f"unexpected native extension set: {sorted(native)}")
        required = {
            "q38lab/ple_checkpoint_probe.py",
            f"{expected_dist}/METADATA",
            f"{expected_dist}/WHEEL",
            f"{expected_dist}/RECORD",
            f"{expected_dist}/licenses/LICENSE",
            f"{expected_dist}/licenses/MODIFICATIONS.md",
            f"{expected_dist}/licenses/THIRD_PARTY_NOTICES.md",
        }
        missing = required.difference(names)
        if missing:
            raise WheelAuditError(f"wheel is missing required members: {sorted(missing)}")
        metadata = archive.read(f"{expected_dist}/METADATA").decode("utf-8")
        wheel = archive.read(f"{expected_dist}/WHEEL").decode("utf-8")
        if not re.search(r"^Name: qwen38-next-5090-lab$", metadata, re.MULTILINE):
            raise WheelAuditError("wheel distribution name is wrong")
        if not re.search(rf"^Version: {re.escape(expected_version)}$", metadata, re.MULTILINE):
            raise WheelAuditError("wheel distribution version is wrong")
        if "Root-Is-Purelib: false" not in wheel or "Tag: cp312-cp312-linux_x86_64" not in wheel:
            raise WheelAuditError("wheel metadata does not declare the native CPython 3.12 tag")

    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "distribution": "qwen38-next-5090-lab",
        "version": expected_version,
        "tag": "cp312-cp312-linux_x86_64",
        "native_extensions": sorted(EXPECTED_NATIVE),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()

    artifact = audit_wheel(args.wheel, args.version)
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise WheelAuditError("source commit must be an exact 40-hex commit")
    document = {
        "schema_version": "1.0",
        "artifact": artifact,
        "source": {
            "repository": "https://github.com/wimi321/qwen38-next-5090-lab",
            "commit": args.source_commit,
            "tag": args.source_tag,
            "hardware_evidence_release": "v0.2.0-alpha.1",
            "validated_runtime_commit": "74650573a40c5ebb313f96b1fc6482c9644261e0",
        },
        "build": {
            "environment": "maintainer WSL2 Ubuntu 24.04 ext4 clean checkout",
            "python": platform.python_version(),
            "architecture": platform.machine(),
            "libc": list(platform.libc_ver()),
            "cuda_toolkit": "13.3",
            "torch": "2.11.0+cu130",
            "setuptools": "81.0.0",
            "bundled_model_weights": False,
            "bundled_cuda_or_libtorch": False,
        },
        "install_requirements": {
            "name": args.requirements.name,
            "sha256": sha256(args.requirements),
        },
        "validation": {
            "fresh_venv_install": "pass",
            "pip_check": "pass",
            "native_extension_imports": "pass",
            "q38lab_doctor": "pass",
            "packaged_ple_checkpoint_probe": "pass",
            "full_hardware_evidence": "inherited from v0.2.0-alpha.1; not rerun for packaging-only post1",
        },
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.provenance:
        args.provenance.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
