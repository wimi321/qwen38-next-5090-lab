"""Vectorized QSA CUDA seam against the independent scalar Torch oracle."""

import pytest
import torch

from freetoken.kernel.qsa_reference import (
    qsa_causal_visibility,
    qsa_reference_attention,
    qsa_rms_norm,
    qsa_select_token_mask,
)
from freetoken.kernel.qsa_vectorized import (
    qsa_paged_gqa_attention_vectorized,
    qsa_pool_index_keys_vectorized,
    qsa_select_token_indices_vectorized,
)
from freetoken.kernel.qsa_triton import (
    can_use_qsa_attention_triton,
    can_use_qsa_selection_triton,
    qsa_paged_gqa_attention_triton,
    qsa_select_token_indices_triton,
)


def _indices_to_mask(indices, valid, width):
    counts = torch.zeros(*indices.shape[:2], width, dtype=torch.int32)
    safe = indices.clamp_min(0)
    counts.scatter_add_(2, safe, valid.to(torch.int32))
    return counts > 0


def test_pool_transform_contract_is_batched_and_uses_block_starts():
    raw = torch.arange(2 * 12 * 3, dtype=torch.float32).view(2, 12, 3)
    seen = {}

    def transform(pooled, starts):
        seen["pooled_shape"] = pooled.shape
        seen["starts"] = starts.clone()
        return pooled

    pooled = qsa_pool_index_keys_vectorized(
        raw,
        compress_ratio=4,
        key_transform=transform,
    )

    assert pooled.shape == (2, 3, 3)
    assert seen["pooled_shape"] == (2, 3, 3)
    assert torch.equal(seen["starts"], torch.tensor([[0, 4, 8], [0, 4, 8]]))


def test_vectorized_selection_matches_scalar_oracle_across_boundaries():
    torch.manual_seed(7)
    batch, query_len, width, heads, dim = 2, 5, 16, 3, 4
    raw = torch.randn(batch, width, dim)
    index_q = torch.randn(batch, query_len, heads, dim)
    positions = torch.tensor([[0, 3, 4, 7, 9], [0, 3, 4, 8, 12]])
    kv_lens = torch.tensor([10, 13])

    pooled = qsa_pool_index_keys_vectorized(
        raw,
        compress_ratio=4,
        key_transform=lambda values, _starts: qsa_rms_norm(values),
    )
    indices, valid = qsa_select_token_indices_vectorized(
        index_q,
        pooled,
        positions,
        kv_lens,
        token_budget=8,
        compress_ratio=4,
    )
    assert indices.shape == valid.shape == (batch, query_len, 11)
    vector_mask = _indices_to_mask(indices, valid, width)

    for batch_idx in range(batch):
        kv_len = int(kv_lens[batch_idx])
        visible = qsa_causal_visibility(positions[batch_idx], kv_len)
        expected = qsa_select_token_mask(
            index_q[batch_idx],
            raw[batch_idx, :kv_len],
            visible,
            token_budget=8,
            compress_ratio=4,
        )
        assert torch.equal(vector_mask[batch_idx, :, :kv_len], expected)
        assert not vector_mask[batch_idx, :, kv_len:].any()


def test_vectorized_paged_gqa_matches_reference_attention():
    torch.manual_seed(11)
    batch, query_len, width, query_heads, kv_heads, dim = 2, 3, 8, 4, 2, 4
    # Non-contiguous physical maps prove logical selected ids are translated
    # through page_rows before the global slabs are gathered.
    page_rows = torch.tensor([[7, 1, 9, 3, 11, 5, 13, 15], [2, 4, 6, 8, 10, 12, 14, 0]])
    key_rows = torch.randn(16, kv_heads, dim)
    value_rows = torch.randn(16, kv_heads, dim)
    query = torch.randn(batch, query_len, query_heads, dim)
    logical = torch.tensor(
        [
            [[0, 1, -1], [0, 2, 3], [1, 5, 7]],
            [[0, -1, -1], [1, 2, -1], [0, 4, 6]],
        ]
    )
    valid = logical >= 0
    output = qsa_paged_gqa_attention_vectorized(
        query,
        key_rows,
        value_rows,
        page_rows,
        logical,
        valid,
        sm_scale=0.5,
    )

    for batch_idx in range(batch):
        selected = _indices_to_mask(
            logical[batch_idx : batch_idx + 1],
            valid[batch_idx : batch_idx + 1],
            width,
        )[0]
        expected = qsa_reference_attention(
            query[batch_idx],
            key_rows[page_rows[batch_idx]],
            value_rows[page_rows[batch_idx]],
            selected,
            sm_scale=0.5,
        )
        assert torch.allclose(output[batch_idx], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_triton_selection_cuda_matches_scalar_oracle():
    """Exercise the production tensor path on CUDA when GPU CI is available."""
    device = torch.device("cuda")
    width, query_len, heads, dim = 16, 5, 4, 16
    block_scale = torch.arange(1, 5, device=device, dtype=torch.bfloat16)
    raw = block_scale[:, None, None].expand(4, 4, dim).reshape(1, width, dim)
    index_q = torch.ones(
        1, query_len, heads, dim, device=device, dtype=torch.bfloat16
    )
    positions = torch.tensor([[0, 3, 4, 9, 12]], device=device)
    kv_lens = torch.tensor([13], device=device)
    pooled = qsa_pool_index_keys_vectorized(
        raw,
        compress_ratio=4,
        key_transform=lambda values, _starts: values,
    )
    assert can_use_qsa_selection_triton(
        index_q, pooled, token_budget=8, compress_ratio=4
    )
    indices, valid = qsa_select_token_indices_triton(
        index_q,
        pooled,
        positions,
        kv_lens,
        token_budget=8,
        compress_ratio=4,
    )
    actual = _indices_to_mask(indices.cpu(), valid.cpu(), width)[0, :, :13]
    expected = qsa_select_token_mask(
        index_q[0].cpu(),
        raw[0, :13].cpu(),
        qsa_causal_visibility(positions[0].cpu(), 13),
        token_budget=8,
        compress_ratio=4,
        key_transform=lambda values, _starts, _batch: values,
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_selected_paged_gqa_triton_matches_reference():
    torch.manual_seed(23)
    device = torch.device("cuda")
    batch, query_len, width, query_heads, kv_heads, dim = 1, 2, 8, 4, 2, 16
    page_rows = torch.tensor(
        [[7, 1, 9, 3, 11, 5, 13, 15]], device=device, dtype=torch.int32
    )
    key_rows = torch.randn(16, kv_heads, dim, device=device, dtype=torch.bfloat16)
    value_rows = torch.randn_like(key_rows)
    query = torch.randn(
        batch, query_len, query_heads, dim, device=device, dtype=torch.bfloat16
    )
    logical = torch.tensor(
        [[[0, 1, -1, -1], [0, 4, 6, 7]]], device=device, dtype=torch.int64
    )
    valid = logical >= 0
    assert can_use_qsa_attention_triton(query, key_rows, logical)
    output = qsa_paged_gqa_attention_triton(
        query,
        key_rows,
        value_rows,
        page_rows,
        logical,
        valid,
        sm_scale=0.25,
    )

    selected = _indices_to_mask(logical.cpu(), valid.cpu(), width)[0]
    expected = qsa_reference_attention(
        query[0].cpu(),
        key_rows[page_rows[0].long()].cpu(),
        value_rows[page_rows[0].long()].cpu(),
        selected,
        sm_scale=0.25,
    )
    assert torch.allclose(output[0].cpu(), expected, atol=2e-2, rtol=2e-2)
