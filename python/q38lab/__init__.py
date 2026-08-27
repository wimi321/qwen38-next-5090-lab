"""Reproducibility tooling for the Qwen3.8 Next RTX 5090 profile.

The runtime remains importable as :mod:`freetoken`.  This package is a thin,
torch-free command layer which pins the public checkpoint and the hardware
configuration used for the published evidence.
"""

from .constants import MODEL_REPO, MODEL_REVISION, PROFILE_NAME

__all__ = ["MODEL_REPO", "MODEL_REVISION", "PROFILE_NAME"]

