#!/usr/bin/env python3
"""Generate an SPDX 2.3 JSON inventory for the tracked source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]


def tracked_files(root: Path = ROOT) -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root, stderr=subprocess.DEVNULL
    )
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def build_document(name: str, version: str, root: Path = ROOT) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "NOASSERTION"
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is None and commit != "NOASSERTION":
        try:
            epoch = subprocess.check_output(
                ["git", "show", "-s", "--format=%ct", "HEAD"], cwd=root, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            epoch = None
    created = datetime.fromtimestamp(int(epoch or 0), timezone.utc).replace(microsecond=0)
    namespace = f"https://github.com/wimi321/qwen38-next-5090-lab/spdx/{quote(version)}/{commit}"
    files = []
    relationships = [{"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": "SPDXRef-Package"}]
    for index, path in enumerate(tracked_files(root), 1):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        spdx_id = f"SPDXRef-File-{index}"
        files.append({
            "SPDXID": spdx_id,
            "fileName": f"./{relative}",
            "checksums": [{"algorithm": "SHA256", "checksumValue": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "copyrightText": "NOASSERTION",
        })
        relationships.append({"spdxElementId": "SPDXRef-Package", "relationshipType": "CONTAINS", "relatedSpdxElement": spdx_id})
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{name}-{version}-source",
        "documentNamespace": namespace,
        "creationInfo": {
            "created": created.isoformat().replace("+00:00", "Z"),
            "creators": ["Tool: qwen38-next-5090-lab-source-sbom/1.0"],
        },
        "packages": [{
            "name": name,
            "SPDXID": "SPDXRef-Package",
            "versionInfo": version,
            "downloadLocation": "https://github.com/wimi321/qwen38-next-5090-lab",
            "filesAnalyzed": True,
            "licenseConcluded": "Apache-2.0",
            "licenseDeclared": "Apache-2.0",
            "copyrightText": "NOASSERTION",
            "externalRefs": [{
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:github/wimi321/qwen38-next-5090-lab@{quote(version)}",
            }],
        }],
        "files": files,
        "relationships": relationships,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="qwen38-next-5090-lab")
    parser.add_argument("--version", default="0.2.0a1.post1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_document(args.name, args.version), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
