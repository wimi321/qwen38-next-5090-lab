from __future__ import annotations

import pytest
import torch

from freetoken.layers.rotary import RotaryEmbedding
from freetoken.models.qwen4_exp.attention import Qwen4ExpQSAIndexer


class _IdentityNorm:
    def forward(self, value):
        return value


class _RecordingRotary:
    def __init__(self):
        self.positions = None

    def forward(self, positions, query, key):
        self.positions = positions.clone()
        return query, key


def _indexer_stub(head_dim=4):
    indexer = Qwen4ExpQSAIndexer.__new__(Qwen4ExpQSAIndexer)
    indexer.head_dim = head_dim
    indexer.k_layernorm = _IdentityNorm()
    indexer.rotary = _RecordingRotary()
    return indexer


@pytest.mark.parametrize("batch_size", [1, 3])
def test_qsa_raw_batched_key_starts_are_flattened_as_text(batch_size):
    indexer = _indexer_stub()
    pooled = torch.randn(batch_size, 5, indexer.head_dim)
    starts = torch.arange(5).view(1, 5).expand(batch_size, -1)
    observed = indexer.transform_keys(pooled, starts)
    assert observed.shape == pooled.shape
    assert torch.equal(indexer.rotary.positions, starts.to(torch.int32).reshape(-1))


def test_qsa_compressed_keys_preserve_three_axis_mrope_positions():
    indexer = _indexer_stub()
    pooled = torch.randn(5, indexer.head_dim)
    positions = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 7, 8, 9, 10], [0, 5, 6, 11, 12]]
    )
    observed = indexer.transform_keys(pooled, positions)
    assert observed.shape == pooled.shape
    assert torch.equal(indexer.rotary.positions, positions.to(torch.int32))


def test_interleaved_mrope_matches_transformers():
    pytest.importorskip("transformers")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    section = [3, 3, 2]
    config = Qwen4ExpTextConfig(
        hidden_size=64,
        num_attention_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": section,
            "mrope_interleaved": True,
        },
    )
    reference = Qwen4ExpTextRotaryEmbedding(config)
    actual = RotaryEmbedding(
        head_size=64,
        rotary_dim=16,
        max_position_embeddings=128,
        base=10_000.0,
        mrope_section=tuple(section),
        mrope_interleaved=True,
    )

    positions = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 7, 8, 9],
            [0, 5, 6, 8, 10],
        ],
        dtype=torch.long,
    )
    torch.manual_seed(3)
    query = torch.randn(5, 2, 64)
    key = torch.randn(5, 1, 64)

    cos, sin = reference(query.unsqueeze(0), positions.unsqueeze(1))
    expected_q, expected_k = apply_rotary_pos_emb(
        query.unsqueeze(0),
        key.unsqueeze(0),
        cos=cos,
        sin=sin,
        unsqueeze_dim=2,
    )
    observed_q, observed_k = actual.forward(
        positions,
        query.reshape(5, -1).clone(),
        key.reshape(5, -1).clone(),
    )
    torch.testing.assert_close(observed_q.view_as(query), expected_q.squeeze(0))
    torch.testing.assert_close(observed_k.view_as(key), expected_k.squeeze(0))
