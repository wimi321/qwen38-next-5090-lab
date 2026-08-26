"""Qwen4-Exp routed MoE.

Its router and expert equations are the same softmax/top-k + gated shared expert
used by Qwen3.5, so reuse the already-optimized FreeToken implementation.  A
separate class keeps the model package and future Qwen4-specific changes
independent without duplicating the offload/cache-critical code.
"""

from __future__ import annotations

from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE


class Qwen4ExpMoE(Qwen3_5MoE):
    pass


__all__ = ["Qwen4ExpMoE"]
