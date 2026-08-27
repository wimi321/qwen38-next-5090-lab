"""Qwen4-Exp gated GQA with Query-Selective Attention (QSA)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP
from freetoken.layers.rotary import get_rope
from freetoken.models.qwen3_5_moe.quant_linear import make_col_merged, make_replicated
from freetoken.utils import nvtx_annotate

from .hyperconnection import Qwen4ExpGroupedRMSNorm

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig, QSAAttentionGroupConfig


class Qwen4ExpQSAIndexer(BaseOP):
    """Projection/norm/RoPE portion of the QSA indexer.

    Raw index keys are intentionally returned before normalization and RoPE.
    The backend pools each complete four-token block first; ``transform_keys``
    then applies the key norm and partial RoPE at the block's first position.
    """

    def __init__(self, config: ModelConfig, group: QSAAttentionGroupConfig):
        self.num_q_heads = group.indexer_n_heads
        self.num_kv_heads = group.indexer_kv_heads
        self.head_dim = group.indexer_head_dim
        self._split = [self.num_q_heads * self.head_dim, self.num_kv_heads * self.head_dim]
        self.index_qk_proj = make_replicated(
            config, config.hidden_size, sum(self._split), has_bias=False
        )
        self.q_layernorm = Qwen4ExpGroupedRMSNorm(
            self.head_dim, group_size=self.head_dim, eps=config.rms_norm_eps
        )
        self.k_layernorm = Qwen4ExpGroupedRMSNorm(
            self.head_dim, group_size=self.head_dim, eps=config.rms_norm_eps
        )
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=group.rotary_config.rotary_dim,
            max_position=group.rotary_config.max_position,
            base=group.rotary_config.base,
            rope_scaling=None,
            mrope_section=group.rotary_config.mrope_section,
            mrope_interleaved=group.rotary_config.mrope_interleaved,
        )

    def project(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        qk = self.index_qk_proj.forward(hidden_states)
        q, raw_k = torch.split(qk, self._split, dim=-1)
        q = q.view(-1, self.num_q_heads, self.head_dim)
        raw_k = raw_k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_layernorm.forward(q)
        # The RoPE primitive consumes flattened head dimensions.
        q_flat = q.reshape(q.shape[0], -1)
        unused_key = raw_k.new_zeros(raw_k.shape[0], self.head_dim)
        q_flat, _ = self.rotary.forward(positions, q_flat, unused_key)
        return q_flat.view_as(q), raw_k.squeeze(-2)

    def transform_keys(
        self, pooled_keys: torch.Tensor, logical_block_starts: torch.Tensor
    ) -> torch.Tensor:
        original_shape = pooled_keys.shape
        if pooled_keys.ndim not in (2, 3):
            raise ValueError(
                "QSA pooled keys must be [blocks, dim] or [batch, blocks, dim], "
                f"got {tuple(original_shape)}"
            )
        positions = logical_block_starts.to(torch.int32)
        if pooled_keys.ndim == 3:
            # The legacy/raw cache pools independently for each request and
            # passes starts as [B,N].  In particular B==3 is still a text batch,
            # not a three-axis mRoPE tensor.
            if positions.shape != original_shape[:-1]:
                raise ValueError(
                    "batched QSA key positions must match [batch, blocks], got "
                    f"{tuple(positions.shape)} for keys {tuple(original_shape)}"
                )
            positions = positions.reshape(-1)
        elif positions.ndim == 1:
            if positions.numel() != original_shape[0]:
                raise ValueError("QSA scalar block positions do not match pooled keys")
            positions = positions.reshape(-1)
        elif positions.ndim != 2 or positions.shape != (3, original_shape[0]):
            raise ValueError(
                "compressed QSA key positions must be [blocks] or [3, blocks], got "
                f"{tuple(positions.shape)}"
            )

        pooled_keys = self.k_layernorm.forward(pooled_keys).reshape(-1, self.head_dim)
        # Avoid aliasing query/key inputs: the in-place RoPE kernel rotates both.
        unused_query = pooled_keys.clone()
        _, pooled_keys = self.rotary.forward(
            positions, unused_query, pooled_keys
        )
        return pooled_keys.reshape(original_shape)


class Qwen4ExpAttention(BaseOP):
    """Gated QSA layer matching ``Qwen4ExpTextAttention`` parameter names."""

    def __init__(self, config: ModelConfig, layer_id: int):
        from freetoken.models.config import QSAAttentionGroupConfig

        group = config.attention_group_for_layer(layer_id)
        if not isinstance(group, QSAAttentionGroupConfig):
            raise TypeError(f"Qwen4ExpAttention layer {layer_id} requires a QSA group")
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = group.num_kv_heads
        self.head_dim = group.head_dim
        self.qo_dim = self.num_q * self.head_dim
        self.kv_dim = self.num_kv * self.head_dim
        self._qkv_split = [2 * self.qo_dim, self.kv_dim, self.kv_dim]

        self.qkv_proj = make_col_merged(
            config, config.hidden_size, self._qkv_split, has_bias=False
        )
        self.q_norm = Qwen4ExpGroupedRMSNorm(
            self.head_dim, group_size=self.head_dim, eps=config.rms_norm_eps
        )
        self.k_norm = Qwen4ExpGroupedRMSNorm(
            self.head_dim, group_size=self.head_dim, eps=config.rms_norm_eps
        )
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=group.rotary_config.rotary_dim,
            max_position=group.rotary_config.max_position,
            base=group.rotary_config.base,
            rope_scaling=None,
            mrope_section=group.rotary_config.mrope_section,
            mrope_interleaved=group.rotary_config.mrope_interleaved,
        )
        self.indexer = Qwen4ExpQSAIndexer(config, group)
        self.o_proj = make_replicated(
            config, self.qo_dim, config.hidden_size, has_bias=False
        )

    def _project(self, hidden_states: torch.Tensor):
        batch = get_global_ctx().batch
        positions = getattr(batch, "mrope_positions", None)
        if positions is None:
            positions = batch.positions
        qkv = self.qkv_proj.forward(hidden_states)
        qg, key, value = torch.split(qkv, self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, 2 * self.head_dim)
        query = self.q_norm.forward(qg[..., : self.head_dim].contiguous())
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_dim)
        key = self.k_norm.forward(key.view(-1, self.num_kv, self.head_dim).contiguous())

        query_flat = query.reshape(-1, self.qo_dim)
        key_flat = key.reshape(-1, self.kv_dim)
        query_flat, key_flat = self.rotary.forward(positions, query_flat, key_flat)
        query = query_flat.view(-1, self.num_q, self.head_dim)
        key = key_flat.view(-1, self.num_kv, self.head_dim)
        value = value.view(-1, self.num_kv, self.head_dim).contiguous()
        index_q, raw_index_k = self.indexer.project(hidden_states, positions)
        return query, key, value, gate, index_q, raw_index_k

    @nvtx_annotate("QSA")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        query, key, value, gate, index_q, raw_index_k = self._project(hidden_states)
        qsa_forward = getattr(ctx.attn_backend, "qsa_forward", None)
        if qsa_forward is None:
            raise RuntimeError(
                f"attention backend {type(ctx.attn_backend).__name__} has no QSA path"
            )
        output = qsa_forward(
            query,
            key,
            value,
            index_q,
            raw_index_k,
            self.layer_id,
            ctx.batch,
            index_key_transform=self.indexer.transform_keys,
        )
        output = output.reshape(-1, self.qo_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(output)


__all__ = ["Qwen4ExpAttention", "Qwen4ExpQSAIndexer"]
