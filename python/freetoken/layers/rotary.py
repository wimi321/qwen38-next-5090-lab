from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
        proportional: bool = False,
        attention_factor: float = 1.0,
        is_neox: bool = True,
        mrope_section: Tuple[int, ...] | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # NeoX (half-rotation, HF default) vs GPT-J interleaved (adjacent pairs,
        # ``rope_interleave`` models: GLM MLA lineage). Both underlying kernels
        # accept the flag; the cos/sin cache layout is identical.
        self.is_neox = is_neox
        self.mrope_section = mrope_section
        self.mrope_interleaved = bool(mrope_interleaved)
        if proportional:
            assert 0 < rotary_dim <= head_size
            assert rotary_dim % 2 == 0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size)
            )
            if rotary_dim < head_size:
                inv_freq[rotary_dim // 2 :] = 0.0
        else:
            # Standard (NeoX) rope. Supports partial rotary (rotary_dim < head_size):
            # rope is applied to the first ``rotary_dim`` dims of each head, the rest pass
            # through. Frequencies are spaced over ``rotary_dim`` (matches HF default
            # partial rope, e.g. Qwen3.5 partial_rotary_factor, MiniMax-M2's
            # ``apply_rotary_pos_emb``). Full rope is rotary_dim == head_size and is
            # unaffected. ``head_size`` is passed to flashinfer separately so it rotates
            # only the first ``rotary_dim`` dims.
            assert 0 < rotary_dim <= head_size
            assert rotary_dim % 2 == 0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
            )
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer import apply_rope_with_cos_sin_cache_inplace
        else:
            from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 2:
            return self._forward_mrope(positions, query, key)
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
            is_neox=self.is_neox,
        )
        return query, key

    def _forward_mrope(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Qwen4-Exp's interleaved temporal/height/width RoPE."""

        if positions.shape[0] != 3:
            raise ValueError(f"mRoPE positions must be [3, tokens], got {tuple(positions.shape)}")
        if self.mrope_section is None or not self.mrope_interleaved:
            raise ValueError("three-axis positions require an interleaved mrope_section")
        if sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError(
                f"mrope_section {self.mrope_section} does not cover rotary dim {self.rotary_dim}"
            )
        if not self.is_neox:
            raise NotImplementedError("Qwen4-Exp mRoPE currently requires NeoX half rotation")
        token_count = positions.shape[1]
        if query.shape[0] != token_count or key.shape[0] != token_count:
            raise ValueError("mRoPE token positions must match query and key rows")
        if token_count == 0:
            return query, key

        flat_positions = positions.to(device=self._cos_sin_cache.device, dtype=torch.long)
        if int(flat_positions.min().item()) < 0 or int(flat_positions.max().item()) >= len(
            self._cos_sin_cache
        ):
            raise ValueError("mRoPE position is outside the configured context window")
        axis = self._cos_sin_cache.index_select(0, flat_positions.reshape(-1)).view(
            3, token_count, self.rotary_dim
        )
        half = self.rotary_dim // 2
        cos = axis[0, :, :half].clone()
        sin = axis[0, :, half:].clone()
        for dim, offset in ((1, 1), (2, 2)):
            stop = self.mrope_section[dim] * 3
            lane = slice(offset, stop, 3)
            cos[:, lane] = axis[dim, :, :half][:, lane]
            sin[:, lane] = axis[dim, :, half:][:, lane]
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)

        def rotate_in_place(tensor: torch.Tensor) -> None:
            shaped = tensor.reshape(token_count, -1, self.head_size)
            rope = shaped[..., : self.rotary_dim]
            left, right = rope.chunk(2, dim=-1)
            rotated_half = torch.cat((-right, left), dim=-1)
            rotated = rope * cos[:, None].to(rope.dtype) + rotated_half * sin[:, None].to(
                rope.dtype
            )
            shaped[..., : self.rotary_dim].copy_(rotated)

        rotate_in_place(query)
        rotate_in_place(key)
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    is_neox: bool = True,
    mrope_section: Tuple[int, ...] | None = None,
    mrope_interleaved: bool = False,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(
            head_dim,
            rotary_dim,
            max_position,
            base,
            is_neox=is_neox,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox=is_neox)

        case "proportional":
            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                proportional=True,
                is_neox=is_neox,
            )

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(
                head_dim, rotary_dim, max_position, base, post_process, is_neox=is_neox
            )

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def get_mscale(scale: float, mscale: float = 1.0) -> float:
                if scale <= 1:
                    return 1.0
                return 0.1 * mscale * math.log(scale) + 1.0

            attention_factor = rope_scaling.get("attention_factor")
            if attention_factor is None:
                mscale = rope_scaling.get("mscale")
                mscale_all_dim = rope_scaling.get("mscale_all_dim")
                # Truthiness, not presence: HF falls back to get_mscale(factor) when
                # mscale_all_dim is 0 (a real DeepSeek-lineage default).
                if mscale and mscale_all_dim:
                    attention_factor = get_mscale(factor, mscale) / get_mscale(
                        factor, mscale_all_dim
                    )
                else:
                    attention_factor = get_mscale(factor)

            def _find_correction_dim(num_rotations: float) -> float:
                return (
                    rotary_dim
                    * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = _find_correction_dim(beta_fast)
            high = _find_correction_dim(beta_slow)
            if rope_scaling.get("truncate", True):
                low = math.floor(low)
                high = math.ceil(high)
            low = max(low, 0)
            # rotary_dim - 1, per HF's find_correction_range and this repo's own faithful copy in
            # models/deepseek_v4/ops.py. Clamping to rotary_dim//2 - 1 instead forces the ramp to
            # reach 1.0 at the last entry, fully interpolating the longest-wavelength dims that
            # the reference deliberately leaves partly extrapolated.
            high = min(high, rotary_dim - 1)
            if low == high:  # HF nudges instead of flooring the gap at 1 ("truncate": false)
                high += 0.001

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / (high - low),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                post_process,
                attention_factor=float(attention_factor),
                is_neox=is_neox,
            )

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
    is_neox: bool = True,
    mrope_section: Tuple[int, ...] | None = None,
    mrope_interleaved: bool = False,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(
                head_dim,
                rotary_dim,
                max_position,
                base,
                rope_map,
                is_neox,
                mrope_section,
                mrope_interleaved,
            )
    return _get_rope(
        head_dim,
        rotary_dim,
        max_position,
        base,
        rope_map,
        is_neox,
        mrope_section,
        mrope_interleaved,
    )


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
