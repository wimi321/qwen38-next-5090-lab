"""Pure-PyTorch Qwen Query-Selective Attention correctness oracle.

This module intentionally contains no Triton/CUDA specialization.  It mirrors
the released Qwen4-Exp selection rule closely enough to serve as an independent
test oracle and as a slow bring-up backend:

* causally visible token indices are grouped in four-token microblocks;
* raw index keys are mean pooled, RMS-normalized, and optionally partial-RoPE'd
  at the first token position of each block;
* block score is ``sum(relu(q @ k)) / sqrt(index_head_dim)`` across index heads;
* the highest ``token_budget / compress_ratio`` complete blocks are selected;
* the final incomplete causal tail is always selected.

The implementation follows logical token indices.  Paged physical addressing
belongs to the backend/pool and is tested separately.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def qsa_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Qwen4-Exp RMSNorm; checkpoint ``weight`` is the delta in ``1 + weight``."""
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def qsa_apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to the leading ``cos.shape[-1]`` dims and preserve the rest."""
    rotary_dim = int(cos.shape[-1])
    if rotary_dim <= 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError(
            f"invalid QSA rotary dim {rotary_dim} for head dim {x.shape[-1]}"
        )
    rope, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rope * cos.to(rope.dtype) + _rotate_half(rope) * sin.to(rope.dtype)
    return torch.cat((rotated, passthrough), dim=-1)


def qsa_causal_visibility(
    query_positions: torch.Tensor,
    kv_length: int,
) -> torch.Tensor:
    """Boolean ``[..., query, key]`` causal visibility in logical positions."""
    if kv_length < 0:
        raise ValueError(f"kv_length must be non-negative, got {kv_length}")
    keys = torch.arange(kv_length, device=query_positions.device)
    return keys <= query_positions.to(torch.long).unsqueeze(-1)


def _as_batched(
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    visible_mask: torch.Tensor,
    key_cos: torch.Tensor | None,
    key_sin: torch.Tensor | None,
):
    squeeze = index_query.ndim == 3
    if squeeze:
        index_query = index_query.unsqueeze(0)
        raw_index_key = raw_index_key.unsqueeze(0)
        visible_mask = visible_mask.unsqueeze(0)
        if key_cos is not None:
            key_cos = key_cos.unsqueeze(0)
        if key_sin is not None:
            key_sin = key_sin.unsqueeze(0)
    if index_query.ndim != 4 or raw_index_key.ndim != 3 or visible_mask.ndim != 3:
        raise ValueError(
            "QSA selector expects q [B,Q,H,D], raw_k [B,K,D], mask [B,Q,K] "
            f"(or unbatched forms), got {tuple(index_query.shape)}, "
            f"{tuple(raw_index_key.shape)}, {tuple(visible_mask.shape)}"
        )
    bq, _, _, dim = index_query.shape
    bk, kv_len, key_dim = raw_index_key.shape
    if bq != bk or key_dim != dim or visible_mask.shape != (bq, index_query.shape[1], kv_len):
        raise ValueError("QSA selector batch/query/key dimensions do not agree")
    if key_cos is None or key_sin is None:
        if key_cos is not None or key_sin is not None:
            raise ValueError("QSA key_cos and key_sin must be supplied together")
    else:
        if key_cos.shape[:2] != (bq, kv_len) or key_sin.shape != key_cos.shape:
            raise ValueError("QSA key RoPE tensors must have shape [B,K,rotary_dim]")
    return index_query, raw_index_key, visible_mask.bool(), key_cos, key_sin, squeeze


def qsa_select_token_mask(
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int = 4,
    key_norm_weight: torch.Tensor | None = None,
    key_norm_eps: float = 1e-6,
    key_cos: torch.Tensor | None = None,
    key_sin: torch.Tensor | None = None,
    key_transform: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return the exact boolean QSA token mask.

    ``index_query`` is expected after query RMSNorm and partial RoPE.  The raw
    index keys are transformed after microblock pooling, matching the model: the
    optional key norm weight uses Qwen's ``1 + weight`` convention and optional
    ``key_cos/key_sin`` are indexed at each block's first logical token.
    """
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if token_budget < compress_ratio or token_budget % compress_ratio:
        raise ValueError(
            f"token_budget must be a positive multiple of {compress_ratio}, got {token_budget}"
        )
    if key_transform is not None and (
        key_norm_weight is not None or key_cos is not None or key_sin is not None
    ):
        raise ValueError(
            "key_transform is mutually exclusive with built-in key norm/RoPE arguments"
        )

    q, raw_k, visible, key_cos, key_sin, squeeze = _as_batched(
        index_query, raw_index_key, visible_mask, key_cos, key_sin
    )
    batch, query_len, _, index_dim = q.shape
    kv_len = raw_k.shape[1]
    block_topk = token_budget // compress_ratio
    selected = torch.zeros((batch, query_len, kv_len), dtype=torch.bool, device=q.device)

    # Deliberately straightforward loops: this is the numerical oracle against
    # which the optimized kernel will be tested, not the production hot path.
    for batch_idx in range(batch):
        for query_idx in range(query_len):
            visible_indices = torch.nonzero(
                visible[batch_idx, query_idx], as_tuple=False
            ).flatten()
            num_complete = visible_indices.numel() // compress_ratio
            if num_complete:
                block_tokens = visible_indices[: num_complete * compress_ratio].view(
                    num_complete, compress_ratio
                )
                pooled = raw_k[batch_idx].index_select(0, block_tokens.flatten())
                pooled = pooled.view(num_complete, compress_ratio, index_dim)
                pooled = pooled.float().mean(dim=1).to(raw_k.dtype)
                starts = block_tokens[:, 0]
                if key_transform is not None:
                    pooled = key_transform(pooled, starts, batch_idx)
                else:
                    pooled = qsa_rms_norm(pooled, key_norm_weight, eps=key_norm_eps)
                    if key_cos is not None:
                        pooled = qsa_apply_partial_rope(
                            pooled,
                            key_cos[batch_idx].index_select(0, starts),
                            key_sin[batch_idx].index_select(0, starts),
                        )
                if pooled.shape != (num_complete, index_dim):
                    raise ValueError(
                        "QSA key_transform must preserve [num_blocks, index_head_dim], "
                        f"got {tuple(pooled.shape)}"
                    )
                scores = q[batch_idx, query_idx].float() @ pooled.float().transpose(0, 1)
                scores = torch.relu(scores).sum(dim=0) / math.sqrt(index_dim)
                picked_blocks = torch.topk(
                    scores, k=min(block_topk, num_complete), dim=0
                ).indices
                picked_tokens = block_tokens.index_select(0, picked_blocks).flatten()
                selected[batch_idx, query_idx, picked_tokens] = True

            # A query inside a microblock must see every visible token in that
            # incomplete block; dropping it is the common tail/causality bug.
            tail = visible_indices[num_complete * compress_ratio :]
            selected[batch_idx, query_idx, tail] = True

    return selected[0] if squeeze else selected


def qsa_reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Dense materialization of GQA attention under a QSA token mask."""
    squeeze = query.ndim == 3
    if squeeze:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        selected_mask = selected_mask.unsqueeze(0)
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("QSA attention expects q/k/v [B,T,H,D] (or unbatched forms)")
    batch, query_len, query_heads, dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != dim:
        raise ValueError("QSA attention q/k dimensions do not agree")
    kv_len, kv_heads = key.shape[1], key.shape[2]
    if query_heads % kv_heads:
        raise ValueError(f"QSA GQA heads do not divide: q={query_heads}, kv={kv_heads}")
    if selected_mask.shape != (batch, query_len, kv_len):
        raise ValueError("QSA selected mask shape does not match q/k lengths")
    if not torch.all(selected_mask.any(dim=-1)):
        raise ValueError("every QSA query must select at least one visible token")

    repeats = query_heads // kv_heads
    key = key.repeat_interleave(repeats, dim=2)
    value = value.repeat_interleave(repeats, dim=2)
    scale = dim**-0.5 if sm_scale is None else sm_scale
    logits = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    logits.masked_fill_(~selected_mask[:, None], torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probs, value.float()).to(query.dtype)
    return output[0] if squeeze else output


def qsa_reference_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int = 4,
    key_norm_weight: torch.Tensor | None = None,
    key_norm_eps: float = 1e-6,
    key_cos: torch.Tensor | None = None,
    key_sin: torch.Tensor | None = None,
    key_transform: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select then attend, returning ``(output, selected_token_mask)``."""
    selected = qsa_select_token_mask(
        index_query,
        raw_index_key,
        visible_mask,
        token_budget=token_budget,
        compress_ratio=compress_ratio,
        key_norm_weight=key_norm_weight,
        key_norm_eps=key_norm_eps,
        key_cos=key_cos,
        key_sin=key_sin,
        key_transform=key_transform,
    )
    return (
        qsa_reference_attention(query, key, value, selected, sm_scale=sm_scale),
        selected,
    )


__all__ = [
    "qsa_apply_partial_rope",
    "qsa_causal_visibility",
    "qsa_reference_attention",
    "qsa_reference_forward",
    "qsa_rms_norm",
    "qsa_select_token_mask",
]
