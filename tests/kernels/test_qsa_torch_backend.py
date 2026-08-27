"""CPU integration smoke for qsa_torch + paged physical rows."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.qsa import QSAReferenceBackend
from freetoken.attention.qsa_triton import QSATritonBackend, QSATritonMetadata
from freetoken.kernel.qsa_reference import qsa_rms_norm
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import ModelConfig, QSAAttentionGroupConfig, RotaryConfig


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _config():
    rotary = RotaryConfig(
        head_dim=2, rotary_dim=2, max_position=64, base=1e4, scaling=None
    )
    return ModelConfig(
        num_layers=1,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=2,
        hidden_size=8,
        vocab_size=16,
        intermediate_size=16,
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
            QSAAttentionGroupConfig(
                name="qsa",
                layer_ids=(0,),
                num_kv_heads=1,
                head_dim=2,
                rotary_config=rotary,
                indexer_n_heads=1,
                indexer_kv_heads=1,
                indexer_head_dim=2,
                indexer_budget=4,
                indexer_compress_ratio=4,
            ),
        ),
    )


def _production_kv_config():
    """Qwen4-Exp QSA head geometry with a single layer for integration tests."""
    rotary = RotaryConfig(
        head_dim=256, rotary_dim=64, max_position=262144, base=1e7, scaling=None
    )
    return ModelConfig(
        num_layers=1,
        num_qo_heads=24,
        num_kv_heads=2,
        head_dim=256,
        hidden_size=2560,
        vocab_size=16,
        intermediate_size=16,
        rms_norm_eps=1e-6,
        rotary_config=rotary,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="test_qsa_production_kv",
        architectures=["TestQSAProductionKV"],
        attention_groups=(
            QSAAttentionGroupConfig(
                name="qsa",
                layer_ids=(0,),
                num_kv_heads=2,
                head_dim=256,
                rotary_config=rotary,
                indexer_n_heads=4,
                indexer_kv_heads=1,
                indexer_head_dim=128,
                indexer_budget=2048,
                indexer_compress_ratio=4,
            ),
        ),
    )


def test_backend_uses_qsa_selection_over_paged_rows(monkeypatch):
    config = _config()
    pool = QSAKVCache(
        num_kv_heads=1,
        num_layers=1,
        head_dim=2,
        num_pages=16,
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_head_dim=2,
        layer_ids=(0,),
    )
    physical = torch.arange(2, 12, dtype=torch.int32)
    page_table = torch.zeros(1, 16, dtype=torch.int32)
    page_table[0, :10] = physical
    ctx = SimpleNamespace(kv_cache=pool, page_table=page_table)
    monkeypatch.setattr("freetoken.attention.qsa.get_global_ctx", lambda: ctx)
    backend = QSAReferenceBackend(config)

    req = SimpleNamespace(extend_len=10, device_len=10, table_idx=0)
    batch = SimpleNamespace(
        reqs=[req],
        padded_reqs=[req],
        positions=torch.arange(10, dtype=torch.int32),
        out_loc=physical,
    )
    backend.prepare_metadata(batch)

    q = torch.zeros(10, 2, 2, dtype=torch.bfloat16)
    k = torch.zeros(10, 2, dtype=torch.bfloat16)
    token_value = torch.arange(10, dtype=torch.bfloat16)
    v = token_value[:, None].expand(-1, 2).contiguous()
    index_q = torch.tensor([[[1.0, 0.0]]] * 10, dtype=torch.bfloat16)
    index_k = torch.tensor(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4 + [[-1.0, 0.0]] * 2,
        dtype=torch.bfloat16,
    )
    output = backend.qsa_forward(
        q,
        k,
        v,
        index_q,
        index_k,
        0,
        batch,
        index_key_transform=lambda pooled, _starts: qsa_rms_norm(pooled),
    )

    # Last query selects block [0..3] and tail [8,9]. Main q/k logits are all
    # equal, so both GQA query heads return the selected values' exact mean.
    expected = torch.tensor((0 + 1 + 2 + 3 + 8 + 9) / 6, dtype=torch.bfloat16)
    assert torch.allclose(output[-1], expected.expand(2, 2), atol=2e-2, rtol=0)
    assert torch.equal(pool.index_k_cache(0).index_select(0, physical.long()), index_k)

    with pytest.raises(RuntimeError, match="must call qsa_forward"):
        backend.forward(q, k, v, 0, batch)


def test_vectorized_backend_prefill_decode_and_fixed_graph_metadata(monkeypatch):
    config = _config()
    pool = QSAKVCache(
        num_kv_heads=1,
        num_layers=1,
        head_dim=2,
        num_pages=16,
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_head_dim=2,
        layer_ids=(0,),
    )
    physical = torch.arange(2, 12, dtype=torch.int32)
    page_table = torch.zeros(1, 16, dtype=torch.int32)
    page_table[0, :10] = physical
    ctx = SimpleNamespace(kv_cache=pool, page_table=page_table)
    monkeypatch.setattr("freetoken.attention.qsa_triton.get_global_ctx", lambda: ctx)
    dispatch_warnings = []
    monkeypatch.setattr(
        "freetoken.attention.qsa_triton.logger.warning_rank0",
        dispatch_warnings.append,
    )
    backend = QSATritonBackend(config)

    req = SimpleNamespace(extend_len=10, device_len=10, table_idx=0)
    batch = SimpleNamespace(
        reqs=[req], padded_reqs=[req], phase="prefill",
        positions=torch.arange(10, dtype=torch.int32), out_loc=physical,
    )
    backend.prepare_metadata(batch)
    q = torch.zeros(10, 2, 2, dtype=torch.bfloat16)
    k = torch.zeros(10, 2, dtype=torch.bfloat16)
    values = torch.arange(10, dtype=torch.bfloat16)[:, None].expand(-1, 2).contiguous()
    index_q = torch.tensor([[[1.0, 0.0]]] * 10, dtype=torch.bfloat16)
    index_k = torch.tensor(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4 + [[-1.0, 0.0]] * 2,
        dtype=torch.bfloat16,
    )
    transform = lambda pooled, _starts: qsa_rms_norm(pooled)
    prefill = backend.qsa_forward(
        q, k, values, index_q, index_k, 0, batch,
        index_key_transform=transform,
    )
    expected = torch.tensor((0 + 1 + 2 + 3 + 8 + 9) / 6, dtype=torch.bfloat16)
    assert torch.allclose(prefill[-1], expected.expand(2, 2), atol=2e-2, rtol=0)
    assert len(dispatch_warnings) == 1
    assert "selection=torch, attention=torch" in dispatch_warnings[0]
    assert "benchmark results include a fallback path" in dispatch_warnings[0]

    # Eager decode takes the same vectorized, fixed-width path (Q=1).
    decode_req = SimpleNamespace(extend_len=1, device_len=10, table_idx=0)
    decode = SimpleNamespace(
        reqs=[decode_req], padded_reqs=[decode_req], phase="decode",
        positions=torch.tensor([9], dtype=torch.int32),
        out_loc=physical[-1:], active_table_idx=None,
    )
    backend.prepare_metadata(decode)
    decoded = backend.qsa_forward(
        q[-1:], k[-1:], values[-1:], index_q[-1:], index_k[-1:], 0, decode,
        index_key_transform=transform,
    )
    assert torch.allclose(decoded[0], expected.expand(2, 2), atol=2e-2, rtol=0)
    assert len(dispatch_warnings) == 1

    # Keep the future capture/replay seam fixed-shape. The registry intentionally
    # disables graphs until QSA capture/replay passes target-hardware validation.
    backend.init_capture_graph(max_seq_len=16, bs_list=[1])
    capture = SimpleNamespace(
        reqs=[decode_req], padded_reqs=[decode_req], phase="decode", size=1,
    )
    backend.prepare_for_capture(capture)
    assert isinstance(capture.attn_metadata, QSATritonMetadata)
    assert capture.attn_metadata.rows.shape == (1, 16)
    replay = SimpleNamespace(
        reqs=[decode_req], padded_reqs=[decode_req], phase="decode",
        active_table_idx=torch.tensor([0]), padded_size=1,
    )
    backend.prepare_metadata(replay)
    backend.prepare_for_replay(replay)
    assert replay.attn_metadata.rows.data_ptr() == backend._rows_buf.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_triton_backend_stores_production_headed_kv_prefill_and_decode(monkeypatch):
    config = _production_kv_config()
    device = torch.device("cuda")
    pool = QSAKVCache(
        num_kv_heads=2,
        num_layers=1,
        head_dim=256,
        num_pages=16,
        page_size=1,
        dtype=torch.bfloat16,
        device=device,
        index_head_dim=128,
        layer_ids=(0,),
    )
    physical = torch.arange(2, 13, dtype=torch.int32, device=device)
    page_table = torch.zeros(1, 16, dtype=torch.int32, device=device)
    page_table[0, : physical.numel()] = physical
    ctx = SimpleNamespace(kv_cache=pool, page_table=page_table)
    monkeypatch.setattr("freetoken.attention.qsa_triton.get_global_ctx", lambda: ctx)
    backend = QSATritonBackend(config)

    tokens = torch.arange(11, dtype=torch.bfloat16, device=device)
    key = (
        tokens[:, None, None]
        .expand(-1, 2, 256)
        .contiguous()
    )
    value = (
        tokens[:, None, None]
        .add(torch.tensor([0, 10], dtype=torch.bfloat16, device=device)[None, :, None])
        .expand(-1, -1, 256)
        .contiguous()
    )
    query = torch.zeros(11, 24, 256, dtype=torch.bfloat16, device=device)
    index_q = torch.zeros(11, 4, 128, dtype=torch.bfloat16, device=device)
    index_k = torch.zeros(11, 128, dtype=torch.bfloat16, device=device)
    transform = lambda pooled, _starts: pooled

    prefill_req = SimpleNamespace(extend_len=10, device_len=10, table_idx=0)
    prefill = SimpleNamespace(
        reqs=[prefill_req],
        padded_reqs=[prefill_req],
        phase="prefill",
        positions=torch.arange(10, dtype=torch.int32, device=device),
        out_loc=physical[:10],
    )
    backend.prepare_metadata(prefill)
    prefill_output = backend.qsa_forward(
        query[:10],
        key[:10],
        value[:10],
        index_q[:10],
        index_k[:10],
        0,
        prefill,
        index_key_transform=transform,
    )

    expected_prefill = torch.cat(
        (
            torch.full((12, 256), 4.5, dtype=torch.bfloat16, device=device),
            torch.full((12, 256), 14.5, dtype=torch.bfloat16, device=device),
        )
    )
    assert torch.allclose(prefill_output[-1], expected_prefill, atol=5e-2, rtol=0)
    cached_key = pool.k_cache(0).view(-1, 2, 256)
    cached_value = pool.v_cache(0).view(-1, 2, 256)
    assert torch.equal(cached_key.index_select(0, physical[:10].long()), key[:10])
    assert torch.equal(cached_value.index_select(0, physical[:10].long()), value[:10])

    decode_req = SimpleNamespace(extend_len=1, device_len=11, table_idx=0)
    decode = SimpleNamespace(
        reqs=[decode_req],
        padded_reqs=[decode_req],
        phase="decode",
        positions=torch.tensor([10], dtype=torch.int32, device=device),
        out_loc=physical[10:],
        active_table_idx=None,
    )
    backend.prepare_metadata(decode)
    decode_output = backend.qsa_forward(
        query[10:],
        key[10:],
        value[10:],
        index_q[10:],
        index_k[10:],
        0,
        decode,
        index_key_transform=transform,
    )

    expected_decode = torch.cat(
        (
            torch.full((12, 256), 5, dtype=torch.bfloat16, device=device),
            torch.full((12, 256), 15, dtype=torch.bfloat16, device=device),
        )
    )
    assert torch.allclose(decode_output[0], expected_decode, atol=5e-2, rtol=0)
    assert torch.equal(cached_key[physical[10].long()], key[10])
    assert torch.equal(cached_value[physical[10].long()], value[10])
