#!/usr/bin/env python3
# Copyright 2026 Qwen3.8 Next 5090 Lab contributors.
# SPDX-License-Identifier: Apache-2.0
"""Source-checkout compatibility wrapper for the packaged PLE probe."""

from q38lab.ple_checkpoint_probe import *  # noqa: F401,F403
from q38lab.ple_checkpoint_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
