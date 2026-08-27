"""Focused oracle/boundary tests for the SM120 QSA selector."""

import importlib

import pytest
import torch

from freetoken.kernel.qsa_fast_topk import (
    QSA_BLOCK_TOPK,
    _dispatch_native_or_fallback,
    _jit_qsa_fast_topk_module,
    qsa_fast_topk,
)
from freetoken.kernel.qsa_triton import (
    qsa_paged_gqa_attention_triton,
    qsa_select_token_indices_sm120,
)
from freetoken.kernel.qsa_vectorized import (
    qsa_paged_gqa_attention_vectorized,
    qsa_select_token_indices_vectorized,
)


def _selection_as_mask(
    logical: torch.Tensor, valid: torch.Tensor, *, width: int
) -> torch.Tensor:
    """Compare selector results without depending on unspecified top-k order."""

    sentinel = torch.full_like(logical, width)
    scatter = torch.where(valid, logical, sentinel)
    mask = torch.zeros(
        (*logical.shape[:-1], width + 1),
        dtype=torch.bool,
        device=logical.device,
    )
    return mask.scatter_(-1, scatter, True)[..., :width]


def test_fast_topk_handles_all_65536_compressed_blocks_on_cpu():
    scores = torch.arange(65_536, dtype=torch.float32).view(1, -1)
    selected = qsa_fast_topk(scores, torch.tensor([65_536], dtype=torch.int32))
    assert selected.shape == (1, QSA_BLOCK_TOPK)
    assert selected.dtype == torch.int32
    expected = torch.arange(65_536 - QSA_BLOCK_TOPK, 65_536, dtype=torch.int32)
    assert torch.equal(selected[0].sort().values, expected)


def test_fast_topk_short_row_uses_minus_one_for_unused_slots():
    scores = torch.arange(16, dtype=torch.float32).view(1, -1)
    selected = qsa_fast_topk(scores, torch.tensor([7], dtype=torch.int32))[0]
    assert torch.equal(selected[selected >= 0].sort().values, torch.arange(7))
    assert int((selected == -1).sum()) == QSA_BLOCK_TOPK - 7


def test_native_dispatch_records_success(monkeypatch):
    module = importlib.import_module("freetoken.kernel.qsa_fast_topk")

    class Native:
        @staticmethod
        def launch(scores, lengths, output):
            output.fill_(23)

    monkeypatch.setattr(module, "_jit_qsa_fast_topk_module", lambda: Native())
    telemetry = []
    scores = torch.zeros((1, 16), dtype=torch.float32)
    lengths = torch.tensor([16], dtype=torch.int32)
    output = torch.empty((1, QSA_BLOCK_TOPK), dtype=torch.int32)
    observed = _dispatch_native_or_fallback(
        scores,
        lengths,
        output,
        require_native=True,
        telemetry=telemetry.append,
    )
    assert observed is output
    assert torch.all(observed == 23)
    assert telemetry == ["native"]


def test_optional_native_dispatch_records_visible_fallback(monkeypatch):
    module = importlib.import_module("freetoken.kernel.qsa_fast_topk")

    class BrokenNative:
        @staticmethod
        def launch(scores, lengths, output):
            raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(module, "_jit_qsa_fast_topk_module", lambda: BrokenNative())
    telemetry = []
    scores = torch.arange(16, dtype=torch.float32).view(1, -1)
    lengths = torch.tensor([7], dtype=torch.int32)
    output = torch.empty((1, QSA_BLOCK_TOPK), dtype=torch.int32)
    observed = _dispatch_native_or_fallback(
        scores,
        lengths,
        output,
        require_native=False,
        telemetry=telemetry.append,
    )
    assert torch.equal(observed[0][observed[0] >= 0].sort().values, torch.arange(7))
    assert telemetry == ["error", "fallback"]


def test_required_native_dispatch_fails_closed_without_fallback(monkeypatch):
    module = importlib.import_module("freetoken.kernel.qsa_fast_topk")

    class BrokenNative:
        @staticmethod
        def launch(scores, lengths, output):
            raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(module, "_jit_qsa_fast_topk_module", lambda: BrokenNative())
    telemetry = []
    scores = torch.zeros((1, 16), dtype=torch.float32)
    lengths = torch.tensor([16], dtype=torch.int32)
    output = torch.empty((1, QSA_BLOCK_TOPK), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="required.*failed"):
        _dispatch_native_or_fallback(
            scores,
            lengths,
            output,
            require_native=True,
            telemetry=telemetry.append,
        )
    assert telemetry == ["error"]


def test_required_native_public_api_rejects_cpu_without_fallback():
    telemetry = []
    with pytest.raises(RuntimeError, match="not on CUDA"):
        qsa_fast_topk(
            torch.zeros((1, 16), dtype=torch.float32),
            torch.tensor([16], dtype=torch.int32),
            require_native=True,
            telemetry=telemetry.append,
        )
    assert telemetry == ["error"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_fast_topk_exactly_rescans_oversized_coarse_bin():
    device = torch.device("cuda")
    scores = torch.zeros((1, 65_536), dtype=torch.float32, device=device)
    # All 16K candidates occupy the same high byte after conversion to FP16,
    # while the exact-FP32 winners live late in the row.  The inherited 4096
    # candidate cache clipped this case before its exact radix passes.
    candidates = torch.linspace(1.0, 1.1, 16_384, device=device)
    scores[0, 49_152:] = candidates
    half_bits = candidates.half().view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.unique(half_bits >> 8).numel() == 1

    lengths = torch.tensor([65_536], dtype=torch.int32, device=device)
    observed = torch.empty((1, QSA_BLOCK_TOPK), dtype=torch.int32, device=device)
    # Call the native module directly so a compile/launch failure cannot be
    # hidden by qsa_fast_topk's visible torch fallback.
    _jit_qsa_fast_topk_module().launch(scores, lengths, observed)
    expected = torch.topk(scores, QSA_BLOCK_TOPK, dim=1).indices.to(torch.int32)
    assert torch.equal(observed[0].sort().values, expected[0].sort().values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm120_multquery_short_context_matches_oracle():
    """Broadcast request lengths must be materialized before the Triton launch.

    A zero-stride ``kv_lens.expand(...).reshape(-1)`` previously made the raw
    pointer kernel read valid storage only for query row zero.  This is the
    exact first-prefill-chunk geometry that turned full-model logits into NaN.
    """

    torch.manual_seed(1234)
    device = torch.device("cuda")
    index_query = torch.randn(
        1, 32, 4, 128, dtype=torch.bfloat16, device=device
    )
    pooled_keys = torch.randn(
        1, 13, 128, dtype=torch.bfloat16, device=device
    )
    positions = torch.arange(32, dtype=torch.int64, device=device).view(1, -1)
    lengths = torch.tensor([55], dtype=torch.int64, device=device)

    logical, valid = qsa_select_token_indices_sm120(
        index_query,
        pooled_keys,
        positions,
        lengths,
        token_budget=2048,
        compress_ratio=4,
        require_native_topk=True,
    )
    oracle_logical, oracle_valid = qsa_select_token_indices_vectorized(
        index_query,
        pooled_keys,
        positions,
        lengths,
        token_budget=2048,
        compress_ratio=4,
    )

    assert torch.equal(valid.sum(dim=-1), oracle_valid.sum(dim=-1))
    for query_idx in range(positions.shape[1]):
        observed = logical[0, query_idx][valid[0, query_idx]].sort().values
        expected = oracle_logical[0, query_idx][oracle_valid[0, query_idx]].sort().values
        assert torch.equal(observed, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm120_batched_variable_lengths_are_finite_and_match_oracle():
    """Every raw-pointer row must use its own request length in a mixed batch."""

    torch.manual_seed(4321)
    device = torch.device("cuda")
    batch, query_len, index_heads, index_dim = 2, 33, 4, 16
    page_width, kv_heads, query_heads, head_dim = 64, 2, 4, 16
    positions = torch.stack(
        (
            torch.arange(query_len, device=device),
            torch.arange(20, 20 + query_len, device=device),
        )
    ).to(torch.int64)
    lengths = torch.tensor([33, 53], dtype=torch.int64, device=device)
    index_query = torch.randn(
        batch,
        query_len,
        index_heads,
        index_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    pooled_keys = torch.randn(
        batch, 14, index_dim, dtype=torch.bfloat16, device=device
    )

    logical, valid = qsa_select_token_indices_sm120(
        index_query,
        pooled_keys,
        positions,
        lengths,
        token_budget=2048,
        compress_ratio=4,
        require_native_topk=True,
    )
    oracle_logical, oracle_valid = qsa_select_token_indices_vectorized(
        index_query,
        pooled_keys,
        positions,
        lengths,
        token_budget=2048,
        compress_ratio=4,
    )
    assert torch.equal(
        _selection_as_mask(logical, valid, width=page_width),
        _selection_as_mask(oracle_logical, oracle_valid, width=page_width),
    )

    query = torch.randn(
        batch,
        query_len,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    key_rows = torch.randn(
        page_width, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    value_rows = torch.randn_like(key_rows)
    page_rows = torch.arange(page_width, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    observed = qsa_paged_gqa_attention_triton(
        query, key_rows, value_rows, page_rows, logical, valid
    )
    expected = qsa_paged_gqa_attention_vectorized(
        query, key_rows, value_rows, page_rows, oracle_logical, oracle_valid
    )
    assert torch.isfinite(observed).all()
    torch.testing.assert_close(observed, expected, atol=2e-2, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm120_two_full_prefill_chunks_and_decode_3_to_4_match_oracle():
    """Exercise the released Q=512 chunk and the four-token decode boundary."""

    torch.manual_seed(2468)
    device = torch.device("cuda")
    index_heads, index_dim = 4, 16
    pooled_keys = torch.randn(
        1, 256, index_dim, dtype=torch.bfloat16, device=device
    )
    key_rows = torch.randn(1024, 1, 16, dtype=torch.bfloat16, device=device)
    value_rows = torch.randn_like(key_rows)
    page_rows = torch.arange(1024, dtype=torch.int32, device=device).view(1, -1)

    for chunk_start in (0, 512):
        positions = torch.arange(
            chunk_start, chunk_start + 512, dtype=torch.int64, device=device
        ).view(1, -1)
        lengths = torch.tensor([1024], dtype=torch.int64, device=device)
        index_query = torch.randn(
            1, 512, index_heads, index_dim, dtype=torch.bfloat16, device=device
        )
        logical, valid = qsa_select_token_indices_sm120(
            index_query,
            pooled_keys,
            positions,
            lengths,
            token_budget=2048,
            compress_ratio=4,
            require_native_topk=True,
        )
        oracle_logical, oracle_valid = qsa_select_token_indices_vectorized(
            index_query,
            pooled_keys,
            positions,
            lengths,
            token_budget=2048,
            compress_ratio=4,
        )
        assert torch.equal(
            _selection_as_mask(logical, valid, width=1024),
            _selection_as_mask(oracle_logical, oracle_valid, width=1024),
        )
        query = torch.randn(
            1, 512, 2, 16, dtype=torch.bfloat16, device=device
        )
        observed = qsa_paged_gqa_attention_triton(
            query, key_rows, value_rows, page_rows, logical, valid
        )
        expected = qsa_paged_gqa_attention_vectorized(
            query, key_rows, value_rows, page_rows, oracle_logical, oracle_valid
        )
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, expected, atol=2e-2, rtol=0)

    # Decode at position 3 closes the first group; position 4 must then add the
    # one-token causal tail of the next group.  Both calls use the same bounded
    # selector shape as production decode.
    decode_positions = torch.tensor([[3], [4]], dtype=torch.int64, device=device)
    decode_lengths = torch.tensor([4, 5], dtype=torch.int64, device=device)
    decode_query = torch.randn(
        2, 1, index_heads, index_dim, dtype=torch.bfloat16, device=device
    )
    decode_keys = pooled_keys[:, :2].expand(2, -1, -1).contiguous()
    logical, valid = qsa_select_token_indices_sm120(
        decode_query,
        decode_keys,
        decode_positions,
        decode_lengths,
        token_budget=2048,
        compress_ratio=4,
        require_native_topk=True,
    )
    observed_mask = _selection_as_mask(logical, valid, width=5)
    expected_mask = torch.tensor(
        [[[True, True, True, True, False]], [[True, True, True, True, True]]],
        device=device,
    )
    assert torch.equal(observed_mask, expected_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm120_65536_block_selector_matches_oracle_threshold():
    torch.manual_seed(1234)
    device = torch.device("cuda")
    index_query = torch.randn(
        1, 1, 4, 16, dtype=torch.bfloat16, device=device
    )
    pooled_keys = torch.randn(
        1, 65_536, 16, dtype=torch.bfloat16, device=device
    )
    positions = torch.tensor([[262_143]], dtype=torch.int64, device=device)
    lengths = torch.tensor([262_144], dtype=torch.int64, device=device)
    workspace = torch.empty((1, 65_536), dtype=torch.float32, device=device)

    logical, valid = qsa_select_token_indices_sm120(
        index_query,
        pooled_keys,
        positions,
        lengths,
        token_budget=2048,
        compress_ratio=4,
        score_workspace=workspace,
    )
    assert logical.shape == valid.shape == (1, 1, 2051)
    selected_tokens = logical[0, 0][valid[0, 0]]
    selected_blocks = torch.unique(torch.div(selected_tokens, 4, rounding_mode="floor"))
    assert selected_blocks.numel() == QSA_BLOCK_TOPK

    oracle_scores = torch.relu(
        torch.einsum(
            "hd,kd->hk", index_query[0, 0].float(), pooled_keys[0].float()
        )
    ).sum(0) / (index_query.shape[-1] ** 0.5)
    threshold = torch.topk(oracle_scores, QSA_BLOCK_TOPK).values[-1]
    selected_scores = oracle_scores.index_select(0, selected_blocks.long())
    # BF16 tensor-core accumulation can reorder only candidates at the cutoff.
    assert selected_scores.min() >= threshold - 5e-2
