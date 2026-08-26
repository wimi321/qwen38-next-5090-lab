"""CPU oracle tests for Qwen Query-Selective Attention."""

import torch

from freetoken.kernel.qsa_reference import (
    qsa_apply_partial_rope,
    qsa_causal_visibility,
    qsa_reference_attention,
    qsa_select_token_mask,
)


def test_four_token_blocks_top_budget_tail_and_causality():
    # Block 0 points along +x and block 1 along +y; +x queries must select
    # block 0 when only one complete block fits the four-token budget.
    raw_k = torch.tensor(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4 + [[-1.0, 0.0]] * 2
    )
    index_q = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]] * 10)
    visible = qsa_causal_visibility(torch.arange(10), kv_length=10)
    selected = qsa_select_token_mask(
        index_q, raw_k, visible, token_budget=4, compress_ratio=4
    )

    # At position 4 there is one complete block plus a one-token tail: dense.
    assert torch.equal(torch.nonzero(selected[4]).flatten(), torch.arange(5))
    # At position 9 only the best complete block plus the incomplete [8, 9]
    # tail survives.  No future position is ever selected.
    assert torch.equal(
        torch.nonzero(selected[9]).flatten(), torch.tensor([0, 1, 2, 3, 8, 9])
    )
    for query_pos in range(10):
        assert not selected[query_pos, query_pos + 1 :].any()


def test_top_budget_is_counted_in_tokens_not_blocks():
    raw_k = torch.tensor(
        [[1.0, 0.0]] * 4
        + [[0.8, 0.2]] * 4
        + [[0.0, 1.0]] * 4
        + [[-1.0, 0.0]]
    )
    index_q = torch.tensor([[[1.0, 0.0]]])
    visible = torch.ones(1, 13, dtype=torch.bool)
    selected = qsa_select_token_mask(
        index_q, raw_k, visible, token_budget=8, compress_ratio=4
    )[0]
    # 8 tokens -> two complete blocks, plus the one-token tail.
    assert torch.equal(
        torch.nonzero(selected).flatten(), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 12])
    )


def test_context_within_budget_matches_dense_causal_mask():
    torch.manual_seed(1)
    length = 7
    index_q = torch.randn(length, 3, 4)
    raw_k = torch.randn(length, 4)
    causal = qsa_causal_visibility(torch.arange(length), length)
    selected = qsa_select_token_mask(
        index_q, raw_k, causal, token_budget=8, compress_ratio=4
    )
    assert torch.equal(selected, causal)


def test_partial_rope_rotates_prefix_and_preserves_nope_tail():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    cos = torch.zeros(1, 2)
    sin = torch.ones(1, 2)
    assert torch.equal(
        qsa_apply_partial_rope(x, cos, sin),
        torch.tensor([[-2.0, 1.0, 3.0, 4.0]]),
    )


def test_reference_attention_honors_mask_and_gqa():
    query = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])  # [Q=1,Hq=2,D=2]
    key = torch.tensor(
        [[[0.0, 0.0]], [[1000.0, 0.0]], [[0.0, 0.0]]]
    )  # Hkv=1
    value = torch.tensor(
        [[[1.0, 0.0]], [[1000.0, 0.0]], [[3.0, 0.0]]]
    )
    selected = torch.tensor([[True, False, True]])
    output = qsa_reference_attention(query, key, value, selected, sm_scale=1.0)
    assert torch.allclose(output, torch.tensor([[[2.0, 0.0], [2.0, 0.0]]]))
