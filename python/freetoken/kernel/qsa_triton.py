"""Triton kernels for Qwen Query-Selective Attention.

The scalar oracle in :mod:`freetoken.kernel.qsa_reference` remains the source
of truth.  This module owns the CUDA hot path for the released 8K geometry:

* index-query/block-key scoring;
* deterministic top-k block selection and token-index expansion;
* selected paged-GQA logits, softmax, and value reduction.

Pooling raw index keys and applying their model-owned RMSNorm/partial-RoPE stay
outside these kernels because those projections own checkpoint parameters.  A
vectorized Torch fallback is selected by the backend for unsupported shapes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# The first hardware milestone is 8K: 8192 / four-token microblocks == 2048.
# Larger contexts deliberately fall back until the selector is hierarchically
# tiled; a one-program bitonic top-k beyond this width has poor occupancy.
_MAX_SELECTOR_BLOCKS = 2048
_MAX_SELECTED_WIDTH = 4096


@triton.jit
def _qsa_index_score_kernel(
    index_query_ptr,
    pooled_key_ptr,
    score_ptr,
    query_len,
    index_scale,
    num_blocks,
    index_heads: tl.constexpr,
    index_dim: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    block_h: tl.constexpr,
):
    block_pid = tl.program_id(0)
    bq_pid = tl.program_id(1)
    batch_idx = bq_pid // query_len

    d = tl.arange(0, block_d)
    h = tl.arange(0, block_h)
    block = block_pid * block_n + tl.arange(0, block_n)

    q_offsets = bq_pid * index_heads * index_dim + h[:, None] * index_dim + d[None, :]
    q = tl.load(
        index_query_ptr + q_offsets,
        mask=(h[:, None] < index_heads) & (d[None, :] < index_dim),
        other=0.0,
    ).to(tl.float32)
    key_offsets = (
        batch_idx * num_blocks * index_dim
        + d[:, None] * 1
        + block[None, :] * index_dim
    )
    key = tl.load(
        pooled_key_ptr + key_offsets,
        mask=(d[:, None] < index_dim) & (block[None, :] < num_blocks),
        other=0.0,
    ).to(tl.float32)

    # Pad the released four index heads to a tensor-core friendly 16 rows.  The
    # masked rows are zero, so their post-ReLU contribution is exactly zero.
    per_head = tl.dot(q, key, input_precision="ieee")
    scores = tl.sum(tl.maximum(per_head, 0.0), axis=0) * index_scale
    tl.store(
        score_ptr + bq_pid * num_blocks + block,
        scores,
        mask=block < num_blocks,
    )


@triton.jit
def _qsa_topk_tokens_kernel(
    score_ptr,
    position_ptr,
    kv_len_ptr,
    logical_out_ptr,
    valid_out_ptr,
    query_len,
    num_blocks,
    token_budget: tl.constexpr,
    compress_ratio: tl.constexpr,
    block_topk: tl.constexpr,
    out_width: tl.constexpr,
    selector_width: tl.constexpr,
):
    bq_pid = tl.program_id(0)
    batch_idx = bq_pid // query_len
    position = tl.load(position_ptr + bq_pid).to(tl.int64)
    kv_len = tl.load(kv_len_ptr + batch_idx).to(tl.int64)
    complete = tl.minimum((position + 1) // compress_ratio, kv_len // compress_ratio)
    complete = tl.minimum(complete, num_blocks)

    block = tl.arange(0, selector_width)
    eligible = (block < num_blocks) & (block < complete)
    scores = tl.load(
        score_ptr + bq_pid * num_blocks + block,
        mask=eligible,
        other=-float("inf"),
    )

    # tl.topk returns values only.  Recover a stable exact-K mask from its last
    # value: take every strictly-greater score, then the lowest-id threshold ties
    # until K is full.  This matters when ReLU produces many exact zero scores.
    top_values = tl.topk(scores, block_topk)
    threshold = tl.min(top_values, axis=0)
    greater = eligible & (scores > threshold)
    num_greater = tl.sum(greater.to(tl.int32), axis=0)
    ties = eligible & (scores == threshold)
    tie_rank = tl.cumsum(ties.to(tl.int32), axis=0)
    take_tie = ties & (tie_rank <= (block_topk - num_greater))
    selected = eligible & ((complete <= block_topk) | greater | take_tie)
    selected_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1

    micro = tl.arange(0, compress_ratio)
    output_slot = selected_rank[:, None] * compress_ratio + micro[None, :]
    token = block[:, None] * compress_ratio + micro[None, :]
    selected_2d = selected[:, None]
    out_base = bq_pid * out_width
    tl.store(logical_out_ptr + out_base + output_slot, token, mask=selected_2d)
    tl.store(valid_out_ptr + out_base + output_slot, 1, mask=selected_2d)

    # The causal incomplete microblock is always present after the top-k budget.
    # Keeping it at a fixed offset (rather than compacting after a short top-k)
    # preserves graph shapes; invalid gaps remain -1/False.
    tail_offset = tl.arange(0, compress_ratio)
    tail_token = complete * compress_ratio + tail_offset
    tail_valid = (
        (tail_offset < compress_ratio - 1)
        & (tail_token <= position)
        & (tail_token < kv_len)
    )
    tl.store(
        logical_out_ptr + out_base + token_budget + tail_offset,
        tail_token,
        mask=tail_valid,
    )
    tl.store(
        valid_out_ptr + out_base + token_budget + tail_offset,
        1,
        mask=tail_valid,
    )


@triton.jit
def _qsa_selected_logits_kernel(
    query_ptr,
    key_ptr,
    page_rows_ptr,
    logical_ptr,
    valid_ptr,
    logits_ptr,
    query_len,
    query_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_width,
    selected_width: tl.constexpr,
    sm_scale,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
):
    selected_pid = tl.program_id(0)
    row_pid = tl.program_id(1)  # flattened [B,Q,Hq]
    head_idx = row_pid % query_heads
    bq_idx = row_pid // query_heads
    batch_idx = bq_idx // query_len
    groups = query_heads // kv_heads
    kv_head_idx = head_idx // groups

    selected = selected_pid * block_s + tl.arange(0, block_s)
    d = tl.arange(0, block_d)
    logical_offsets = bq_idx * selected_width + selected
    lane_valid = (selected < selected_width) & tl.load(
        valid_ptr + logical_offsets,
        mask=selected < selected_width,
        other=0,
    )
    logical = tl.load(
        logical_ptr + logical_offsets,
        mask=selected < selected_width,
        other=0,
    ).to(tl.int64)
    lane_valid = lane_valid & (logical >= 0) & (logical < page_width)
    safe_logical = tl.where(lane_valid, logical, 0)
    physical = tl.load(
        page_rows_ptr + batch_idx * page_width + safe_logical,
        mask=lane_valid,
        other=0,
    ).to(tl.int64)

    q = tl.load(
        query_ptr + row_pid * head_dim + d,
        mask=d < head_dim,
        other=0.0,
    ).to(tl.float32)
    key_offsets = (
        physical[:, None] * kv_heads * head_dim
        + kv_head_idx * head_dim
        + d[None, :]
    )
    key = tl.load(
        key_ptr + key_offsets,
        mask=lane_valid[:, None] & (d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(key * q[None, :], axis=1) * sm_scale
    logits = tl.where(lane_valid, logits, -float("inf"))
    tl.store(
        logits_ptr + row_pid * selected_width + selected,
        logits,
        mask=selected < selected_width,
    )


@triton.jit
def _qsa_softmax_inplace_kernel(
    logits_ptr,
    selected_width: tl.constexpr,
    softmax_width: tl.constexpr,
):
    row_pid = tl.program_id(0)
    selected = tl.arange(0, softmax_width)
    logits = tl.load(
        logits_ptr + row_pid * selected_width + selected,
        mask=selected < selected_width,
        other=-float("inf"),
    )
    logits -= tl.max(logits, axis=0)
    numerator = tl.exp(logits)
    probs = numerator / tl.sum(numerator, axis=0)
    tl.store(
        logits_ptr + row_pid * selected_width + selected,
        probs,
        mask=selected < selected_width,
    )


@triton.jit
def _qsa_selected_value_kernel(
    probs_ptr,
    value_ptr,
    page_rows_ptr,
    logical_ptr,
    valid_ptr,
    output_ptr,
    query_len,
    query_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_width,
    selected_width: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
):
    dim_pid = tl.program_id(0)
    row_pid = tl.program_id(1)  # flattened [B,Q,Hq]
    head_idx = row_pid % query_heads
    bq_idx = row_pid // query_heads
    batch_idx = bq_idx // query_len
    groups = query_heads // kv_heads
    kv_head_idx = head_idx // groups
    d = dim_pid * block_d + tl.arange(0, block_d)
    accumulator = tl.zeros((block_d,), dtype=tl.float32)

    for selected_start in tl.range(0, selected_width, block_s, num_stages=2):
        selected = selected_start + tl.arange(0, block_s)
        logical_offsets = bq_idx * selected_width + selected
        lane_valid = (selected < selected_width) & tl.load(
            valid_ptr + logical_offsets,
            mask=selected < selected_width,
            other=0,
        )
        logical = tl.load(
            logical_ptr + logical_offsets,
            mask=selected < selected_width,
            other=0,
        ).to(tl.int64)
        lane_valid = lane_valid & (logical >= 0) & (logical < page_width)
        safe_logical = tl.where(lane_valid, logical, 0)
        physical = tl.load(
            page_rows_ptr + batch_idx * page_width + safe_logical,
            mask=lane_valid,
            other=0,
        ).to(tl.int64)
        probs = tl.load(
            probs_ptr + row_pid * selected_width + selected,
            mask=selected < selected_width,
            other=0.0,
        ).to(tl.float32)
        value_offsets = (
            physical[:, None] * kv_heads * head_dim
            + kv_head_idx * head_dim
            + d[None, :]
        )
        values = tl.load(
            value_ptr + value_offsets,
            mask=lane_valid[:, None] & (d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(values * probs[:, None], axis=0)

    tl.store(
        output_ptr + row_pid * head_dim + d,
        accumulator,
        mask=d < head_dim,
    )


def can_use_qsa_selection_triton(
    index_query: torch.Tensor,
    pooled_index_key: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
) -> bool:
    """Whether the released-shape CUDA selector can serve these tensors."""
    if not index_query.is_cuda or not pooled_index_key.is_cuda:
        return False
    if index_query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if pooled_index_key.dtype != index_query.dtype:
        return False
    if pooled_index_key.device != index_query.device:
        return False
    if index_query.ndim != 4 or pooled_index_key.ndim != 3:
        return False
    index_heads, index_dim = index_query.shape[-2:]
    num_blocks = pooled_index_key.shape[1]
    block_topk = token_budget // compress_ratio if compress_ratio else 0
    return (
        compress_ratio == 4
        and token_budget >= compress_ratio
        and token_budget % compress_ratio == 0
        and block_topk > 0
        and block_topk & (block_topk - 1) == 0
        and num_blocks > 0
        and num_blocks <= _MAX_SELECTOR_BLOCKS
        and index_heads <= 16
        and index_dim <= 256
        and pooled_index_key.shape[0] == index_query.shape[0]
        and pooled_index_key.shape[-1] == index_dim
    )


def qsa_select_token_indices_triton(
    index_query: torch.Tensor,
    pooled_index_key: torch.Tensor,
    query_positions: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Triton score/top-k/token expansion path.

    The returned fixed-width logical ids are int64 and invalid lanes are -1,
    matching :func:`qsa_select_token_indices_vectorized`.
    """
    if not can_use_qsa_selection_triton(
        index_query,
        pooled_index_key,
        token_budget=token_budget,
        compress_ratio=compress_ratio,
    ):
        raise ValueError("QSA Triton selector does not support this tensor geometry")
    batch, query_len, index_heads, index_dim = index_query.shape
    if query_positions.shape != (batch, query_len) or kv_lens.shape != (batch,):
        raise ValueError("QSA Triton positions/length shapes do not agree")
    if query_positions.device != index_query.device or kv_lens.device != index_query.device:
        raise ValueError("QSA Triton selector inputs must share one CUDA device")

    index_query = index_query.contiguous()
    pooled_index_key = pooled_index_key.contiguous()
    query_positions = query_positions.contiguous()
    kv_lens = kv_lens.contiguous()
    num_blocks = pooled_index_key.shape[1]
    score_block_n = 32
    scores = torch.empty(
        (batch, query_len, num_blocks), dtype=torch.float32, device=index_query.device
    )
    score_grid = (triton.cdiv(num_blocks, score_block_n), batch * query_len)
    _qsa_index_score_kernel[score_grid](
        index_query,
        pooled_index_key,
        scores,
        query_len,
        index_dim**-0.5,
        num_blocks=num_blocks,
        index_heads=index_heads,
        index_dim=index_dim,
        block_n=score_block_n,
        block_d=max(16, triton.next_power_of_2(index_dim)),
        block_h=16,
        num_warps=4,
        num_stages=2,
    )

    out_width = token_budget + compress_ratio - 1
    logical = torch.full(
        (batch, query_len, out_width),
        -1,
        dtype=torch.int64,
        device=index_query.device,
    )
    valid = torch.zeros(
        (batch, query_len, out_width), dtype=torch.bool, device=index_query.device
    )
    block_topk = token_budget // compress_ratio
    selector_width = max(triton.next_power_of_2(num_blocks), block_topk)
    _qsa_topk_tokens_kernel[(batch * query_len,)](
        scores,
        query_positions,
        kv_lens,
        logical,
        valid,
        query_len,
        num_blocks=num_blocks,
        token_budget=token_budget,
        compress_ratio=compress_ratio,
        block_topk=block_topk,
        out_width=out_width,
        selector_width=selector_width,
        num_warps=8,
        num_stages=2,
    )
    return logical, valid


def can_use_qsa_attention_triton(
    query: torch.Tensor,
    key_rows: torch.Tensor,
    logical_indices: torch.Tensor,
) -> bool:
    if not query.is_cuda or not key_rows.is_cuda or not logical_indices.is_cuda:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16) or key_rows.dtype != query.dtype:
        return False
    if key_rows.device != query.device or logical_indices.device != query.device:
        return False
    if query.ndim != 4 or key_rows.ndim != 3 or logical_indices.ndim != 3:
        return False
    return (
        query.shape[-1] == key_rows.shape[-1]
        and query.shape[-1] <= 256
        and logical_indices.shape[-1] <= _MAX_SELECTED_WIDTH
    )


def qsa_paged_gqa_attention_triton(
    query: torch.Tensor,
    key_rows: torch.Tensor,
    value_rows: torch.Tensor,
    page_rows: torch.Tensor,
    logical_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Triton selected-token paged GQA (logits + softmax + value)."""
    if not can_use_qsa_attention_triton(query, key_rows, logical_indices):
        raise ValueError("QSA Triton attention does not support this tensor geometry")
    if value_rows.shape != key_rows.shape or value_rows.dtype != key_rows.dtype:
        raise ValueError("QSA Triton K/V slabs do not agree")
    batch, query_len, query_heads, head_dim = query.shape
    kv_heads = key_rows.shape[1]
    if query_heads % kv_heads:
        raise ValueError("QSA Triton GQA query heads must divide by KV heads")
    selected_width = logical_indices.shape[-1]
    if logical_indices.shape[:2] != (batch, query_len):
        raise ValueError("QSA Triton selected indices do not match query batch")
    if valid_mask.shape != logical_indices.shape or page_rows.shape[0] != batch:
        raise ValueError("QSA Triton selected/page masks do not agree")
    if not page_rows.is_cuda or not valid_mask.is_cuda or not value_rows.is_cuda:
        raise ValueError("QSA Triton attention requires all inputs on CUDA")
    if any(
        tensor.device != query.device
        for tensor in (value_rows, page_rows, valid_mask)
    ):
        raise ValueError("QSA Triton attention inputs must share one CUDA device")

    query = query.contiguous()
    key_rows = key_rows.contiguous()
    value_rows = value_rows.contiguous()
    page_rows = page_rows.contiguous()
    logical_indices = logical_indices.contiguous()
    valid_mask = valid_mask.contiguous()
    page_width = page_rows.shape[1]
    row_count = batch * query_len * query_heads
    logits = torch.empty(
        (batch, query_len, query_heads, selected_width),
        dtype=torch.float32,
        device=query.device,
    )
    logit_block_s = 16
    _qsa_selected_logits_kernel[
        (triton.cdiv(selected_width, logit_block_s), row_count)
    ](
        query,
        key_rows,
        page_rows,
        logical_indices,
        valid_mask,
        logits,
        query_len,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_width=page_width,
        selected_width=selected_width,
        sm_scale=head_dim**-0.5 if sm_scale is None else sm_scale,
        block_s=logit_block_s,
        block_d=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=2,
    )
    _qsa_softmax_inplace_kernel[(row_count,)](
        logits,
        selected_width=selected_width,
        softmax_width=triton.next_power_of_2(selected_width),
        num_warps=8,
        num_stages=2,
    )

    output = torch.empty_like(query)
    value_block_d = 32
    _qsa_selected_value_kernel[
        (triton.cdiv(head_dim, value_block_d), row_count)
    ](
        logits,
        value_rows,
        page_rows,
        logical_indices,
        valid_mask,
        output,
        query_len,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_width=page_width,
        selected_width=selected_width,
        block_s=32,
        block_d=value_block_d,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = [
    "can_use_qsa_attention_triton",
    "can_use_qsa_selection_triton",
    "qsa_paged_gqa_attention_triton",
    "qsa_select_token_indices_triton",
]
