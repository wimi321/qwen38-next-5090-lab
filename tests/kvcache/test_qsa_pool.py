"""CPU tests for subset-layer QSA KV/index storage."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache import create_kvcache_pool, resolve_pool_class
from freetoken.kvcache.qsa_pool import QSAKVCache, QSA_SELECTOR_WORKSPACE_BYTES
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


def test_compressed_pool_uses_one_index_row_per_four_tokens_and_layer_local_ring():
    config = _model_config()
    pool = create_kvcache_pool(
        config,
        num_pages=3,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        qsa_compressed=True,
        qsa_num_request_slots=2,
    )
    assert isinstance(pool, QSAKVCache)
    assert pool.uses_compressed_index is True
    assert pool._kv_buffer.shape == (2, 2, 3, 4, 1, 8)
    assert pool._index_k_buffer is None
    assert pool._compressed_index_k_buffer.shape == (2, 3, 4)
    assert pool._pending_index_k_buffer.shape == (2, 8, 4)
    assert pool._pending_position_buffer.shape == (2, 8, 3)
    # Main K/V is 64 B/token; two 4-wide index layers add 4 B/token.
    assert pool.unit_bytes() == (68, 0)

    ring_locs = torch.tensor([0, 1], dtype=torch.long)
    positions_2 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
    positions_5 = positions_2 + 100
    pool.store_pending_positions(positions_2, ring_locs, layer_id=2)
    pool.store_pending_positions(positions_5, ring_locs, layer_id=5)
    assert torch.equal(pool.pending_index_positions(2)[:2], positions_2.T)
    assert torch.equal(pool.pending_index_positions(5)[:2], positions_5.T)

    # Scheduler mRoPE positions are int32; the persistent ring is int64 so the
    # store path must normalize dtype instead of failing on the first request.
    pool.store_pending_positions(
        positions_2.to(torch.int32), ring_locs, layer_id=2
    )
    assert torch.equal(pool.pending_index_positions(2)[:2], positions_2.T)

    assert pool.selector_score_workspace(2, 4).shape == (2, 4)
    assert pool.selector_workspace_peak_bytes == 32

    assert pool.selector_telemetry() == {
        "workspace_peak_bytes": 32,
        "native_calls": 0,
        "fallback_calls": 0,
        "errors": 0,
    }
    pool.record_selector_dispatch("native")
    pool.record_selector_dispatch("fallback")
    pool.record_selector_dispatch("error")
    assert pool.selector_telemetry() == {
        "workspace_peak_bytes": 32,
        "native_calls": 1,
        "fallback_calls": 1,
        "errors": 1,
    }
    with pytest.raises(ValueError, match="unknown QSA selector"):
        pool.record_selector_dispatch("unknown")
    with pytest.raises(MemoryError, match="bounded"):
        pool.selector_score_workspace(513, 65_536)
    assert pool.selector_workspace_peak_bytes == 32

    pool.rebuild(5)
    assert pool._compressed_index_k_buffer.shape == (2, 5, 4)


def test_qwen_256k_compressed_qsa_memory_accounting_is_exact():
    group = QSAAttentionGroupConfig(
        name="qsa",
        layer_ids=tuple(range(12)),
        num_kv_heads=2,
        head_dim=256,
        rotary_config=RotaryConfig(
            head_dim=256,
            rotary_dim=64,
            max_position=262_144,
            base=10_000_000,
            scaling=None,
        ),
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(attention_groups=(group,)),
        dtype=torch.bfloat16,
        page_size=4,
        max_running_req=1,
        attention_backend="qsa_triton_sm120",
        tp_info=SimpleNamespace(size=1),
    )
    per_page, fixed, page_tokens, minimum = QSAKVCache.kv_cost(runtime)
    assert page_tokens == 4
    assert minimum == 0
    assert per_page == 25_344 * 4
    ring_rows = (runtime.max_running_req + 1) * group.indexer_compress_ratio
    expected_fixed = QSA_SELECTOR_WORKSPACE_BYTES + ring_rows * 12 * (
        128 * torch.bfloat16.itemsize + 3 * torch.int64.itemsize
    )
    assert fixed == expected_fixed == 134_244_608
    assert per_page // page_tokens * 262_144 == int(6.1875 * 2**30)


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
