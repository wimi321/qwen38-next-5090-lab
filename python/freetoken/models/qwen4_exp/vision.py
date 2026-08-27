# Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
# Copyright 2026 Qwen3.8 Next 5090 Lab contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen4-Exp image tower.

The tensor layout and parameter names are adapted from Transformers 5.16.1's
``Qwen4ExpVisionModel``.  The implementation uses FreeToken ``BaseOP`` objects
so the ordinary resident-weight loader can materialize the checkpoint without
constructing a second Transformers model.  Video is intentionally not exposed
by the serving API in the v0.2 milestone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated, OPList

if TYPE_CHECKING:
    from .config import Qwen4ExpVisionConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _LayerNorm(BaseOP):
    """LayerNorm with checkpoint-compatible ``weight``/``bias`` names."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = torch.empty(hidden_size)
        self.bias = torch.empty(hidden_size)
        self._shape = (hidden_size,)
        self._eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self._shape, self.weight, self.bias, self._eps)


class Qwen4ExpVisionRotaryEmbedding:
    def __init__(self, dim: int, theta: float = 10_000.0):
        self._dim = int(dim)
        self._theta = float(theta)
        self._inv_freq: torch.Tensor | None = None

    def _inv(self, device: torch.device) -> torch.Tensor:
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = 1.0 / (
                self._theta
                ** (torch.arange(0, self._dim, 2, dtype=torch.float32, device=device) / self._dim)
            )
        return self._inv_freq

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.float().unsqueeze(-1) * self._inv(position_ids.device)).flatten(1)


class Qwen4ExpVisionPatchEmbed(BaseOP):
    def __init__(self, config: Qwen4ExpVisionConfig):
        self.proj = _Conv3dPatchProjection(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(hidden_states)


class _Conv3dPatchProjection(BaseOP):
    """Conv3d represented as plain tensors for FreeToken's state-dict walker."""

    def __init__(self, config: Qwen4ExpVisionConfig):
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.weight = torch.empty(config.hidden_size, config.in_channels, *kernel)
        self.bias = torch.empty(config.hidden_size)
        self._in_channels = config.in_channels
        self._kernel = kernel
        self._embed_dim = config.hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.view(-1, self._in_channels, *self._kernel)
        x = F.conv3d(x.to(self.weight.dtype), self.weight, self.bias, stride=self._kernel)
        return x.view(-1, self._embed_dim)


class Qwen4ExpVisionMLP(BaseOP):
    def __init__(self, config: Qwen4ExpVisionConfig):
        self.linear_fc1 = LinearReplicated(
            config.hidden_size, config.intermediate_size, has_bias=True
        )
        self.linear_fc2 = LinearReplicated(
            config.intermediate_size, config.hidden_size, has_bias=True
        )
        self._hidden_act = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.linear_fc1.forward(hidden_states)
        if self._hidden_act in {"gelu_pytorch_tanh", "gelu_tanh"}:
            x = F.gelu(x, approximate="tanh")
        elif self._hidden_act == "gelu":
            x = F.gelu(x)
        else:
            raise ValueError(f"unsupported Qwen4-Exp vision activation {self._hidden_act!r}")
        return self.linear_fc2.forward(x)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = query.dtype, key.dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q = query.float() * cos + _rotate_half(query.float()) * sin
    k = key.float() * cos + _rotate_half(key.float()) * sin
    return q.to(q_dtype), k.to(k_dtype)


class Qwen4ExpVisionAttention(BaseOP):
    def __init__(self, config: Qwen4ExpVisionConfig):
        self.qkv = LinearReplicated(config.hidden_size, config.hidden_size * 3, has_bias=True)
        self.proj = LinearReplicated(config.hidden_size, config.hidden_size, has_bias=True)
        self._num_heads = config.num_heads
        self._head_dim = config.hidden_size // config.num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: tuple[int, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        q, k, v = (
            self.qkv.forward(hidden_states)
            .reshape(seq_len, 3, self._num_heads, self._head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = _apply_rotary(q, k, cos, sin)
        outputs: list[torch.Tensor] = []
        offset = 0
        for length in lengths:
            end = offset + int(length)
            q_i = q[offset:end].transpose(0, 1).unsqueeze(0)
            k_i = k[offset:end].transpose(0, 1).unsqueeze(0)
            v_i = v[offset:end].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False)
            outputs.append(out.squeeze(0).transpose(0, 1))
            offset = end
        if offset != seq_len:
            raise ValueError(f"vision grid describes {offset} patches but received {seq_len}")
        merged = torch.cat(outputs, dim=0).reshape(seq_len, -1)
        return self.proj.forward(merged)


class Qwen4ExpVisionBlock(BaseOP):
    def __init__(self, config: Qwen4ExpVisionConfig):
        self.norm1 = _LayerNorm(config.hidden_size)
        self.norm2 = _LayerNorm(config.hidden_size)
        self.attn = Qwen4ExpVisionAttention(config)
        self.mlp = Qwen4ExpVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: tuple[int, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn.forward(
            self.norm1.forward(hidden_states), lengths, cos, sin
        )
        return hidden_states + self.mlp.forward(self.norm2.forward(hidden_states))


class Qwen4ExpVisionPatchMerger(BaseOP):
    def __init__(self, config: Qwen4ExpVisionConfig):
        merged_size = config.hidden_size * config.spatial_merge_size**2
        self.norm = _LayerNorm(config.hidden_size)
        self.linear_fc1 = LinearReplicated(merged_size, merged_size, has_bias=True)
        self.linear_fc2 = LinearReplicated(merged_size, config.out_hidden_size, has_bias=True)
        self._merged_size = merged_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(hidden_states).view(-1, self._merged_size)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


class Qwen4ExpVisionModel(BaseOP):
    """Flattened processor pixels -> merged text-width image embeddings."""

    def __init__(self, config: Qwen4ExpVisionConfig):
        self.patch_embed = Qwen4ExpVisionPatchEmbed(config)
        self.pos_embed = _Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = OPList([Qwen4ExpVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen4ExpVisionPatchMerger(config)
        self._spatial_merge_size = config.spatial_merge_size
        self._num_grid_per_side = int(math.isqrt(config.num_position_embeddings))
        if self._num_grid_per_side**2 != config.num_position_embeddings:
            raise ValueError("Qwen4-Exp vision learned position table must form a square grid")
        self._rotary = Qwen4ExpVisionRotaryEmbedding(
            (config.hidden_size // config.num_heads) // 2
        )

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        from transformers.vision_utils import (
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        grid = image_grid_thw.to(device=pixel_values.device, dtype=torch.long)
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid,
            num_grid_per_side=self._num_grid_per_side,
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=self._spatial_merge_size,
        )
        position_ids = get_vision_position_ids(grid, self._spatial_merge_size)
        lengths = tuple(int(value) for value in grid.prod(-1).tolist())

        hidden = self.patch_embed.forward(pixel_values)
        position = (
            self.pos_embed.forward(interp_indices)
            * interp_weights.to(self.pos_embed.weight.dtype).unsqueeze(-1)
        ).sum(1)
        hidden = hidden + position.to(hidden.dtype)
        rotary = self._rotary.forward(position_ids).reshape(hidden.shape[0], -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        cos, sin = rotary.cos(), rotary.sin()
        for block in self.blocks.op_list:
            hidden = block.forward(hidden, lengths, cos, sin)
        return self.merger.forward(hidden)


class _Embedding(BaseOP):
    def __init__(self, num_embeddings: int, hidden_size: int):
        self.weight = torch.empty(num_embeddings, hidden_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.weight)


__all__ = [
    "Qwen4ExpVisionAttention",
    "Qwen4ExpVisionBlock",
    "Qwen4ExpVisionModel",
    "Qwen4ExpVisionPatchEmbed",
    "Qwen4ExpVisionPatchMerger",
]
