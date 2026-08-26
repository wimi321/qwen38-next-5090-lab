"""CPU tests for subset-layer QSA KV/index storage."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache import create_kvcache_pool, resolve_pool_class
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import (
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    QSAAttentionGroupConfig,
    RotaryConfig,
)


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _rotary():
    return RotaryConfig(
        head_dim=8, rotary_dim=4, max_position=1024, base=1e4, scaling=None
    )


def _model_config():
    rotary = _rotary()
    return ModelConfig(
        num_layers=6,
        num_qo_heads=4,
        num_kv_heads=1,
        head_dim=8,
        hidden_size=32,
        vocab_size=128,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rotary_config=rotary,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="test_qsa",
        architectures=["TestQSA"],
        attention_groups=(
            LinearGatedDeltaGroupConfig(
                name="linear",
                layer_ids=(0, 1, 3, 4),
                num_key_heads=1,
                num_value_heads=1,
                key_head_dim=8,
                value_head_dim=8,
                conv_kernel_dim=4,
                output_gate=True,
            ),
            QSAAttentionGroupConfig(
                name="qsa",
                layer_ids=(2, 5),
                num_kv_heads=1,
                head_dim=8,
                rotary_config=rotary,
                indexer_n_heads=2,
                indexer_kv_heads=1,
                indexer_head_dim=4,
                indexer_budget=8,
                indexer_compress_ratio=4,
            ),
        ),
    )


def test_qsa_config_is_explicit_and_never_bsa():
    config = _model_config()
    assert config.attn_type_for_layer(2) is AttnType.QSA
    assert config.attn_type_for_layer(5) is AttnType.QSA
    (spec,) = config.kv_cache_group_specs()
    assert spec.attn_type is AttnType.QSA
    assert spec.layer_ids == (2, 5)
    assert spec.num_index_layers == 2
    assert resolve_pool_class(config) is QSAKVCache


def test_pool_allocates_main_and_index_slabs_only_for_spec_layers():
    config = _model_config()
    pool = create_kvcache_pool(
        config,
        num_pages=3,
        page_size=2,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert isinstance(pool, QSAKVCache)
    assert pool.layer_ids == (2, 5)
    assert pool.num_storage_layers == 2
    assert pool._kv_buffer.shape == (2, 2, 3, 2, 1, 8)
    assert pool._index_k_buffer.shape == (2, 6, 4)
    with pytest.raises(KeyError, match="no paged KV"):
        pool.k_cache(1)
    with pytest.raises(KeyError, match="no QSA index"):
        pool.index_k_cache(1)


def test_store_rebuild_and_unit_bytes_cpu():
    pool = QSAKVCache(
        num_kv_heads=1,
        num_layers=6,
        head_dim=8,
        num_pages=3,
        page_size=2,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_head_dim=4,
        layer_ids=(2, 5),
    )
    rows = torch.tensor([1, 4], dtype=torch.int32)
    key = torch.arange(16, dtype=torch.bfloat16).view(2, 8)
    value = key + 20
    index_key = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    pool.store_kv(key, value, rows, layer_id=5)
    pool.store_index_k(index_key, rows, layer_id=5)
    assert torch.equal(pool.k_cache(5).view(-1, 1, 8)[rows.long(), 0], key)
    assert torch.equal(pool.index_k_cache(5)[rows.long()], index_key)

    # 2 QSA layers * (K+V) * 1 head * 8 dims * 2 B + 2 index layers * 4 dims * 2 B.
    assert pool.unit_bytes() == (2 * 2 * 1 * 8 * 2 + 2 * 4 * 2, 0)
    pool.rebuild(5)
    assert pool._kv_buffer.shape[2] == 5
    assert pool._index_k_buffer.shape == (2, 10, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_store_kv_cuda_flattens_multihead_rows():
    pool = QSAKVCache(
        num_kv_heads=2,
        num_layers=1,
        head_dim=256,
        num_pages=4,
        page_size=2,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        index_head_dim=4,
        layer_ids=(0,),
    )
    rows = torch.tensor([1, 6], dtype=torch.int32, device="cuda")
    key = torch.arange(1024, dtype=torch.bfloat16, device="cuda").view(2, 2, 256)
    value = key + 40

    pool.store_kv(key, value, rows, layer_id=0)
    torch.cuda.synchronize()

    cached_key = pool.k_cache(0).view(-1, 2, 256).index_select(0, rows.long())
    cached_value = pool.v_cache(0).view(-1, 2, 256).index_select(0, rows.long())
    assert torch.equal(cached_key, key)
    assert torch.equal(cached_value, value)

    with pytest.raises(ValueError, match="rows of width 512"):
        pool.store_kv(key[:, :1], value, rows, layer_id=0)


def test_qsa_group_rejects_unsupported_or_ambiguous_geometry():
    base = dict(
        name="qsa",
        layer_ids=(2,),
        num_kv_heads=1,
        head_dim=8,
        rotary_config=_rotary(),
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=4,
    )
    with pytest.raises(ValueError, match="indexer_kv_heads == 1"):
        QSAAttentionGroupConfig(**{**base, "indexer_kv_heads": 2})
    with pytest.raises(ValueError, match="multiple"):
        QSAAttentionGroupConfig(**{**base, "indexer_budget": 6})
    with pytest.raises(ValueError, match="unique"):
        QSAAttentionGroupConfig(**{**base, "layer_ids": (2, 2)})


def test_qsa_cost_counts_only_owned_layers():
    from freetoken.kvcache.base import spec_kv_bytes_per_token

    config = _model_config()
    (spec,) = config.kv_cache_group_specs()
    runtime = SimpleNamespace(
        tp_info=SimpleNamespace(size=1), dtype=torch.bfloat16
    )
    # two QSA layers only: K+V 8 dims + raw index K 4 dims.
    assert spec_kv_bytes_per_token(spec, runtime) == 2 * 2 * 1 * 8 * 2 + 2 * 4 * 2
