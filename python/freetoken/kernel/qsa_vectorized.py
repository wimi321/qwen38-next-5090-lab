"""Vectorized Torch/CUDA QSA primitives used by the bring-up ``qsa_triton`` backend.

The operations are fixed-shape and contain no per-query Python loop, which
keeps a future CUDA-graph path possible.  The backend does not advertise graph
support until capture/replay is validated on the target GPU.  They are kept
separate from the scalar oracle in ``qsa_reference.py`` so the oracle remains
an independent parity target.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def qsa_pool_index_keys_vectorized(
    raw_index_key: torch.Tensor,
    *,
    compress_ratio: int,
    key_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Mean-pool contiguous raw keys and apply post-pool norm/partial-RoPE.

    ``raw_index_key`` is ``[B, padded_kv_width, D]``.  Its width must be a
    multiple of ``compress_ratio`` (FreeToken's page table is 32-aligned, hence
    true for Qwen's ratio four).  ``key_transform`` receives pooled keys
    ``[B,num_blocks,D]`` and logical block starts ``[B,num_blocks]``.
    """
    if raw_index_key.ndim != 3:
        raise ValueError(f"raw QSA keys must be [B,K,D], got {tuple(raw_index_key.shape)}")
    if compress_ratio <= 0 or raw_index_key.shape[1] % compress_ratio:
        raise ValueError(
            f"QSA padded KV width {raw_index_key.shape[1]} must be divisible by "
            f"compress_ratio={compress_ratio}"
        )
    batch, width, dim = raw_index_key.shape
    num_blocks = width // compress_ratio
    pooled = raw_index_key.view(batch, num_blocks, compress_ratio, dim)
    pooled = pooled.float().mean(dim=2).to(raw_index_key.dtype)
    starts = (
        torch.arange(num_blocks, device=raw_index_key.device, dtype=torch.long)
        .mul(compress_ratio)
        .view(1, num_blocks)
        .expand(batch, -1)
    )
    transformed = key_transform(pooled, starts)
    if transformed.shape != pooled.shape:
        raise ValueError(
            "QSA key_transform must preserve pooled shape, got "
            f"{tuple(transformed.shape)} != {tuple(pooled.shape)}"
        )
    return transformed


def qsa_select_token_indices_vectorized(
    index_query: torch.Tensor,
    pooled_index_key: torch.Tensor,
    query_positions: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-width top-block + causal-tail selection.

    Returns ``(logical_indices, valid_mask)`` with width
    ``token_budget + compress_ratio - 1``. Invalid/padded indices are ``-1``.
    Shapes: q ``[B,Q,Hi,D]``, pooled K ``[B,N,D]``, positions ``[B,Q]``,
    lengths ``[B]``.
    """
    if index_query.ndim != 4 or pooled_index_key.ndim != 3:
        raise ValueError("vectorized QSA expects q [B,Q,H,D] and pooled K [B,N,D]")
    batch, query_len, _, dim = index_query.shape
    if pooled_index_key.shape[0] != batch or pooled_index_key.shape[-1] != dim:
        raise ValueError("vectorized QSA query/key dimensions do not agree")
    if query_positions.shape != (batch, query_len) or kv_lens.shape != (batch,):
        raise ValueError("vectorized QSA positions/length shapes do not agree")
    if token_budget < compress_ratio or token_budget % compress_ratio:
        raise ValueError("QSA token budget must be a positive multiple of compress ratio")

    block_topk = token_budget // compress_ratio
    num_blocks = pooled_index_key.shape[1]
    scores = torch.einsum(
        "bqhd,bkd->bqhk", index_query.float(), pooled_index_key.float()
    )
    scores = torch.relu(scores).sum(dim=2) / math.sqrt(dim)

    # A block becomes visible only when all of its tokens are causal.  Clamp by
    # request length as well so padded page-table columns never participate.
    complete = torch.minimum(
        torch.div(query_positions.to(torch.long) + 1, compress_ratio, rounding_mode="floor"),
        torch.div(kv_lens.to(torch.long)[:, None], compress_ratio, rounding_mode="floor"),
    )
    block_ids = torch.arange(num_blocks, device=scores.device).view(1, 1, -1)
    scores = scores.masked_fill(block_ids >= complete.unsqueeze(-1), -torch.inf)

    # Keep K fixed even for tiny test/context widths: padded -inf columns become
    # invalid slots and preserve the decode graph's output shape.
    if num_blocks < block_topk:
        scores = torch.nn.functional.pad(
            scores, (0, block_topk - num_blocks), value=-torch.inf
        )
    top_scores, picked_blocks = torch.topk(scores, k=block_topk, dim=-1)
    picked_valid = torch.isfinite(top_scores)

    offsets = torch.arange(compress_ratio, device=scores.device, dtype=torch.long)
    block_tokens = picked_blocks.unsqueeze(-1) * compress_ratio + offsets
    block_tokens = block_tokens.flatten(-2)
    block_valid = picked_valid.unsqueeze(-1).expand(-1, -1, -1, compress_ratio).flatten(-2)

    # At most ratio-1 visible tokens remain after the complete blocks.
    tail_offsets = torch.arange(
        compress_ratio - 1, device=scores.device, dtype=torch.long
    ).view(1, 1, -1)
    tail = complete.unsqueeze(-1) * compress_ratio + tail_offsets
    tail_valid = (tail <= query_positions.to(torch.long).unsqueeze(-1)) & (
        tail < kv_lens.to(torch.long)[:, None, None]
    )

    indices = torch.cat((block_tokens, tail), dim=-1)
    valid = torch.cat((block_valid, tail_valid), dim=-1)
    indices = torch.where(valid, indices, torch.full_like(indices, -1))
    return indices, valid


def qsa_paged_gqa_attention_vectorized(
    query: torch.Tensor,
    key_rows: torch.Tensor,
    value_rows: torch.Tensor,
    page_rows: torch.Tensor,
    logical_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Gather selected physical rows and run vectorized GQA.

    ``key_rows/value_rows`` are the pool's global row-flat slabs ``[R,Hkv,D]``;
    ``page_rows`` maps each request's logical history ``[B,K]`` to those rows.
    No full main-K/V history is materialized.
    """
    if query.ndim != 4 or key_rows.ndim != 3 or value_rows.shape != key_rows.shape:
        raise ValueError("QSA paged attention expects q [B,Q,H,D], KV rows [R,H,D]")
    batch, query_len, query_heads, dim = query.shape
    kv_heads = key_rows.shape[1]
    if key_rows.shape[-1] != dim or query_heads % kv_heads:
        raise ValueError("QSA paged GQA head geometry does not agree")
    if logical_indices.shape != valid_mask.shape or logical_indices.shape[:2] != (
        batch,
        query_len,
    ):
        raise ValueError("QSA selected-index shapes do not agree")
    if page_rows.shape[0] != batch:
        raise ValueError("QSA page rows batch does not agree")
    safe_logical = logical_indices.clamp_min(0)
    expanded_rows = page_rows[:, None, :].expand(-1, query_len, -1)
    physical = torch.gather(expanded_rows, dim=2, index=safe_logical)
    selected_k = key_rows[physical.to(torch.long)]
    selected_v = value_rows[physical.to(torch.long)]
    # Invalid fixed-width slots may map to an unwritten torch.empty KV row.
    # Zero them *before* matmul/reduction: masking logits alone is insufficient
    # because IEEE 0 * NaN in the value reduction is still NaN.
    lane_valid = valid_mask[..., None, None]
    selected_k.masked_fill_(~lane_valid, 0)
    selected_v.masked_fill_(~lane_valid, 0)

    groups = query_heads // kv_heads
    q_grouped = query.view(batch, query_len, kv_heads, groups, dim)
    logits = torch.einsum(
        "bqghd,bqsgd->bqghs", q_grouped.float(), selected_k.float()
    )
    logits.mul_(dim**-0.5 if sm_scale is None else sm_scale)
    logits.masked_fill_(~valid_mask[:, :, None, None, :], torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1)
    output = torch.einsum(
        "bqghs,bqsgd->bqghd", probs, selected_v.float()
    ).to(query.dtype)
    return output.reshape(batch, query_len, query_heads, dim)


__all__ = [
    "qsa_paged_gqa_attention_vectorized",
    "qsa_pool_index_keys_vectorized",
    "qsa_select_token_indices_vectorized",
]
