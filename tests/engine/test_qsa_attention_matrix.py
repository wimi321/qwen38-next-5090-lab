"""QSA capability-matrix pins: never fall through to MiniMax BSA/dense."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType, attention_backend_info
from freetoken.models.config import KVCacheGroupSpec


def _model_config():
    spec = KVCacheGroupSpec(
        name="qsa",
        layer_ids=(3, 7),
        num_kv_heads=1,
        head_dim=8,
        sliding_window=None,
        index_head_dim=4,
        num_index_layers=2,
        attn_type=AttnType.QSA,
    )
    config = SimpleNamespace(
        model_type="qwen4_exp",
        single_stream_only=False,
        is_moe=False,
        expert_quant="none",
        has_swa_attention=False,
        has_linear_attention=True,
        num_layers=8,
        rotary_config=SimpleNamespace(max_position=1024),
    )
    config.kv_cache_group_specs = lambda: (spec,)
    return config


def _config(**overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-qsa",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **overrides,
    )
    object.__setattr__(config, "model_config", _model_config())
    return config


def test_qsa_backend_has_an_exclusive_capability():
    fast = attention_backend_info("qsa_triton")
    oracle = attention_backend_info("qsa_torch")
    assert fast.supported_types == oracle.supported_types == frozenset({AttnType.QSA})
    assert AttnType.BSA not in fast.supported_types
    # Fixed-shape metadata exists, but QSA capture/replay is not target-hardware
    # validated yet, so whole-model graph capture must remain disabled.
    assert fast.supports_cuda_graph is False
    assert oracle.supports_cuda_graph is False


def test_auto_selects_triton_qsa_backend_but_disables_model_graphs():
    from freetoken.engine.engine import _adjust_config

    config = _config(attention_backend="auto")
    _adjust_config(config)
    assert config.attention_backend == "qsa_triton"
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0


def test_explicit_oracle_backend_disables_graphs():
    from freetoken.engine.engine import _adjust_config

    config = _config(attention_backend="qsa_torch")
    _adjust_config(config)
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0


def test_qsa_rejects_comma_composed_backends_before_hybrid_wrapping():
    from freetoken.engine.engine import _adjust_config

    config = _config(attention_backend="qsa_triton,qsa_torch")
    with pytest.raises(ValueError, match="requires one dedicated attention backend"):
        _adjust_config(config)


@pytest.mark.parametrize("backend", ["triton", "m3_sparse", "fi"])
def test_qsa_rejects_dense_and_bsa_backends(backend):
    from freetoken.engine.engine import _adjust_config

    config = _config(attention_backend=backend)
    with pytest.raises(ValueError, match="does not support"):
        _adjust_config(config)


def test_qsa_rejects_float32_before_pool_allocation():
    from freetoken.engine.engine import _adjust_config

    config = _config(attention_backend="auto")
    object.__setattr__(config, "dtype", torch.float32)
    with pytest.raises(ValueError, match="16-bit"):
        _adjust_config(config)
