#!/usr/bin/env python3
"""Fail CI on unsafe release plumbing, private data, weights, or broken local links."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAX_TRACKED_BYTES = 20 * 1024 * 1024
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "PyPI token": re.compile(r"\bpypi-[A-Za-z0-9_-]{20,}"),
}
ACTION_RE = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
LOCAL_LINK_RE = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
FORBIDDEN_REBRAND_PATHS = {
    "install.sh",
    "assets/desktop-console.png",
    "assets/freetoken-icon.svg",
    "assets/freetoken-wechatgroup.png",
    "scripts/build-release-wheels.sh",
    "scripts/publish-wheels.sh",
    "scripts/ci/manylinux-build.sh",
    "scripts/ci/retag-manylinux.py",
}
FORBIDDEN_REBRAND_PREFIXES = (
    "assets/freetoken-logo",
    "freetoken-kernel-cache/",
)


def tracked_files(root: Path = ROOT) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _link_target(raw: str) -> str | None:
    target = raw.strip().split(" ", 1)[0].strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    return target.split("#", 1)[0]


def forbidden_rebrand_paths(relative_paths: list[str]) -> list[str]:
    """Return upstream publishing/branding paths forbidden in this downstream."""

    return sorted(
        relative
        for relative in relative_paths
        if relative in FORBIDDEN_REBRAND_PATHS
        or any(relative.startswith(prefix) for prefix in FORBIDDEN_REBRAND_PREFIXES)
    )


def audit(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    files = tracked_files(root)
    # ``git ls-files`` retains index entries for working-tree deletions until
    # they are staged. Audit what would actually be present in the source tree;
    # the committed CI checkout has no such transitional entries.
    relative_files = [
        path.relative_to(root).as_posix() for path in files if path.exists()
    ]
    for relative in forbidden_rebrand_paths(relative_files):
        errors.append(f"forbidden upstream branding/publishing path is tracked: {relative}")
    for required in ("LICENSE", "MODIFICATIONS.md", "THIRD_PARTY_NOTICES.md"):
        if not (root / required).is_file():
            errors.append(f"missing required attribution file: {required}")

    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 20 MiB: {relative} ({size} bytes)")
        if path.suffix.lower() in WEIGHT_SUFFIXES or path.name.endswith(".bin"):
            errors.append(f"model/checkpoint artifact must not be tracked: {relative}")
        text = _text(path)
        if text is None:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {relative}")
        if path.suffix.lower() == ".md":
            for raw in LOCAL_LINK_RE.findall(text):
                target = _link_target(raw)
                if target is None or target.startswith("data:"):
                    continue
                resolved = (path.parent / target).resolve()
                try:
                    resolved.relative_to(root.resolve())
                except ValueError:
                    errors.append(f"local Markdown link escapes repository: {relative} -> {target}")
                else:
                    if not resolved.exists():
                        errors.append(f"broken local Markdown link: {relative} -> {target}")

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    if project.get("name") != "qwen38-next-5090-lab":
        errors.append("pyproject project.name must be qwen38-next-5090-lab")
    urls = project.get("urls", {})
    if "wimi321/qwen38-next-5090-lab" not in urls.get("Repository", ""):
        errors.append("pyproject Repository URL must point at wimi321/qwen38-next-5090-lab")
    if urls.get("Upstream") != "https://github.com/FlashML-org/FreeToken":
        errors.append("pyproject must retain the FreeToken upstream URL")
    sglang_source = (
        pyproject.get("tool", {}).get("uv", {}).get("sources", {}).get("sglang-kernel")
    )
    expected_sglang_source = {
        "url": (
            "https://github.com/sgl-project/whl/releases/download/v0.4.5/"
            "sglang_kernel-0.4.5%2Bcu130-cp310-abi3-manylinux2014_x86_64.whl"
        ),
        "marker": "sys_platform == 'linux' and platform_machine == 'x86_64'",
    }
    if sglang_source != expected_sglang_source:
        errors.append("pyproject must pin the audited SGLang cu130 wheel URL")
    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        errors.append("the verified Linux x86_64 dependency lock uv.lock must be tracked")
    else:
        lock_text = lock_path.read_text(encoding="utf-8")
        expected_sglang_digest = (
            "sha256:f482a5fdf287d85cfc9434eaa0faff757d6fee31272f1c3e4408bd79aef189b5"
        )
        if expected_sglang_digest not in lock_text:
            errors.append("uv.lock must retain the audited SGLang cu130 wheel SHA256")

    workflows = root / ".github" / "workflows"
    for path in workflows.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        if "pull_request_target:" in text:
            errors.append(f"pull_request_target is forbidden: {relative}")
        for forbidden in ("twine upload", "docker push", "runs-on: [self-hosted", "runs-on: self-hosted"):
            if forbidden in text:
                errors.append(f"forbidden release operation {forbidden!r} in {relative}")
        for action in ACTION_RE.findall(text):
            if action.startswith("./"):
                continue
            if not re.fullmatch(r"[^@]+@[0-9a-f]{40}", action):
                errors.append(f"GitHub Action is not SHA-pinned in {relative}: {action}")
    return errors


def main() -> int:
    try:
        errors = audit()
    except (OSError, subprocess.CalledProcessError, tomllib.TOMLDecodeError) as exc:
        print(f"repository audit could not run: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("repository audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
