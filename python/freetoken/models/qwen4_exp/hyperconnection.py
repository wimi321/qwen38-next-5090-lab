"""Qwen4-Exp gated-residual (hyper-connection) inference primitives.

The checkpoint carries four residual streams concatenated on the feature axis.  At
each attention/MLP site, a low-rank read mixer collapses those streams to one model
hidden state and a scalar write gate chooses how much of the block output to inject
back into each stream.  The model head reuses the read mixer with ``use_combine=False``
to perform the final four-to-one collapse.

The equations and parameter names intentionally match Transformers'
``Qwen4ExpTextGatedResidual`` so the official weights load without a rename:

    x_norm = grouped_rms_norm(x)
    read = sigmoid(up(silu(down(x_norm) / streams)))
    mixed = mean(read * x_norm.reshape(..., streams, hidden), dim=streams)
    write = 2 * sigmoid(block_inject(x_norm) / streams)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated


class Qwen4ExpGroupedRMSNorm(BaseOP):
    """Per-stream ``(1 + weight)`` RMSNorm used by Qwen4-Exp.

    ``group_size`` is the model hidden size.  Consequently a ``streams * hidden``
    feature vector is normalized independently for every stream rather than across
    the concatenated hyper-connection width.  Normalization and scale application
    happen in fp32 before converting back to the input dtype, exactly as in HF.
    """

    def __init__(self, dim: int, group_size: int, eps: float = 1e-6):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if group_size <= 0 or dim % group_size != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by positive group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        # Qwen4-Exp stores the offset from one (Gemma-style norm semantics).
        self.weight = torch.empty(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.numel():
            raise ValueError(
                f"Expected {self.weight.numel()} features, got {x.shape[-1]}"
            )
        grouped = x.float().reshape(*x.shape[:-1], -1, self.group_size)
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized.flatten(start_dim=-2)
        return (normalized * (1.0 + self.weight.float())).to(x.dtype)


class Qwen4ExpGatedResidual(BaseOP):
    """Four-stream Qwen4-Exp read/write mixer.

    The defaults are the released Qwen3.8-Flash-Next values (four streams and rank
    320), while explicit arguments keep tiny correctness models inexpensive.

    When ``use_combine`` is true, :meth:`forward` returns
    ``(mixed_input, hyper_input, injection_weights)``.  A caller applies a block
    output ``y`` using ``hyper_input + (y[..., None, :] *
    injection_weights[..., :, None]).flatten(-2)``.  When it is false, only the
    final mixed input is returned.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        rms_norm_eps: float = 1e-6,
        use_combine: bool = True,
    ):
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if hc_count <= 0:
            raise ValueError(f"hc_count must be positive, got {hc_count}")
        if hc_lowrank <= 0:
            raise ValueError(f"hc_lowrank must be positive, got {hc_lowrank}")

        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hc_lowrank = hc_lowrank
        hc_hidden_size = hc_count * hidden_size

        self.hc_norm = Qwen4ExpGroupedRMSNorm(
            hc_hidden_size, group_size=hidden_size, eps=rms_norm_eps
        )
        self.input_mix_weight_down = LinearReplicated(
            hc_hidden_size, hc_lowrank, has_bias=False
        )
        self.input_mix_weight_up = LinearReplicated(
            hc_lowrank, hc_hidden_size, has_bias=False
        )
        self.block_inject_weight = (
            LinearReplicated(hc_hidden_size, hc_count, has_bias=False)
            if use_combine
            else None
        )

    def forward(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = self.hc_count * self.hidden_size
        if hyper_input.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} hyper-connection features, got {hyper_input.shape[-1]}"
            )

        hyper_input_normed = self.hc_norm.forward(hyper_input)
        input_mix_weight = F.silu(
            self.input_mix_weight_down.forward(hyper_input_normed) / self.hc_count
        )
        input_mix_weight = torch.sigmoid(
            self.input_mix_weight_up.forward(input_mix_weight)
        ).unflatten(-1, (self.hc_count, self.hidden_size))
        streams = hyper_input_normed.unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        mixed_input = (input_mix_weight * streams).mean(dim=-2)

        if self.block_inject_weight is None:
            return mixed_input

        injection_weights = 2 * torch.sigmoid(
            self.block_inject_weight.forward(hyper_input_normed) / self.hc_count
        )
        return mixed_input, hyper_input, injection_weights


__all__ = ["Qwen4ExpGatedResidual", "Qwen4ExpGroupedRMSNorm"]
