#!/usr/bin/env python3
"""Validate the tracked evidence example against every published JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "results" / "schema"
EXAMPLE_ROOT = ROOT / "results" / "example-synthetic"


def main() -> int:
    pairs = {
        "environment.json": "environment.schema.json",
        "resolved-config.json": "resolved-config.schema.json",
        "summary.json": "summary.schema.json",
    }
    for document_name, schema_name in pairs.items():
        document = json.loads((EXAMPLE_ROOT / document_name).read_text(encoding="utf-8"))
        schema = json.loads((SCHEMA_ROOT / schema_name).read_text(encoding="utf-8"))
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
        print(f"schema valid: {document_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
