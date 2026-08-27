"""Qwen4-Exp config, registration and complete resident-model structure.

The structure audit derives its expected key set from Transformers' independent
Qwen4-Exp implementation, applies the real FreeToken checkpoint rename/fusion
plan, and compares every resident tensor name and shape.  Routed NVFP4 experts
and the file-backed PLE table are intentionally outside both resident sets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.base import AttnType
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.rotary import set_rope_device
from freetoken.models.config import (
    LinearGatedDeltaGroupConfig,
    QSAAttentionGroupConfig,
)
from freetoken.models.qwen4_exp.config import parse_config


if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


def _target_hf_config() -> SimpleNamespace:
    """Target-revision language geometry without constructing its 135 GB model."""

    layer_types = [
        "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        for layer in range(48)
    ]
    text = SimpleNamespace(
        hidden_size=2560,
        num_hidden_layers=48,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        vocab_size=248320,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        norm_topk_prob=True,
        layer_types=layer_types,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
        hc_count=4,
        hc_lowrank=320,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        seed=1234,
        eos_token_id=248044,
        ple_embedding_dtype="float8_e4m3fn",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    return SimpleNamespace(
        text_config=text,
        vision_config=SimpleNamespace(hidden_size=1152),
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        },
        image_token_id=248056,
    )


def _tiny_kwargs() -> dict:
    """Tiny shape-valid config shared by Transformers and FreeToken structure tests.

    FreeToken's current RoPE kernel supports head sizes starting at 64, so the
    attention and index heads stay at that minimum while all expensive MoE/PLE
    dimensions are reduced.
    """

    return {
        "vocab_size": 32,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "output_gate_type": "sigmoid",
        "hc_count": 4,
        "hc_lowrank": 8,
        "indexer_n_heads": 1,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 64,
        "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "ple_layer_ids": [2],
        "ple_embed_dim": 64,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 64,
        "make_ngram_vocab_size_divisible_by": 4,
        "split_ngram_parts": 4,
        "seed": 1234,
        "eos_token_id": 31,
        "ple_embedding_dtype": "float8_e4m3fn",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "norm_topk_prob": True,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.25,
        },
        "attention_dropout": 0.0,
    }


def _tiny_hf_wrapper() -> SimpleNamespace:
    return SimpleNamespace(
        text_config=SimpleNamespace(**_tiny_kwargs()),
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"},
        image_token_id=30,
    )


def test_parse_target_revision_geometry_and_text_only_policy():
    config = parse_config(_target_hf_config())

    assert (config.num_layers, config.hidden_size, config.vocab_size) == (
        48,
        2560,
        248320,
    )
    assert (config.num_qo_heads, config.num_kv_heads, config.head_dim) == (24, 2, 256)
    assert config.rotary_config.rotary_dim == 64
    assert config.rotary_config.base == 10_000_000.0
    assert config.expert_quant == "nvfp4"
    assert config.moe_auto_runtime_reserve_bytes == 512 * 2**20
    assert (config.attn_quant, config.dense_quant, config.lm_head_quant) == (
        "none",
        "none",
        "none",
    )
    assert config.vision_config is None
    assert not config.is_multimodal

    linear, qsa = config.attention_groups
    assert isinstance(linear, LinearGatedDeltaGroupConfig)
    assert isinstance(qsa, QSAAttentionGroupConfig)
    assert len(linear.layer_ids) == 36
    assert qsa.layer_ids == tuple(range(3, 48, 4))
    assert (qsa.indexer_n_heads, qsa.indexer_kv_heads, qsa.indexer_head_dim) == (
        4,
        1,
        128,
    )
    assert (qsa.indexer_budget, qsa.indexer_compress_ratio) == (2048, 4)
    assert config.attn_type_for_layer(0) == AttnType.LINEAR
    assert config.attn_type_for_layer(3) == AttnType.QSA

    args = config.qwen4_args
    assert (args.hc_count, args.hc_lowrank, args.output_gate_type) == (4, 320, "sigmoid")
    assert args.ple_layer_ids == (2,)
    assert args.has_ple(1) and not args.has_ple(0)
    assert (args.ngram_size, args.heads_per_ngram, args.split_ngram_parts) == (3, 8, 128)

    # Linear recurrent state is sized separately; the paged pool owns only the
    # twelve QSA layers and their twelve index-key slabs.
    (cache_spec,) = config.kv_cache_group_specs()
    assert cache_spec.attn_type == AttnType.QSA
    assert cache_spec.layer_ids == qsa.layer_ids
    assert cache_spec.index_head_dim == 128
    assert cache_spec.num_index_layers == 12


def test_parse_rejects_an_unpinned_expert_format():
    hf_config = _target_hf_config()
    hf_config.quantization_config = None
    with pytest.raises(ValueError, match="pinned to the NVFP4"):
        parse_config(hf_config)


def test_parse_rejects_unsupported_ple_layer_multiplicity():
    hf_config = _target_hf_config()
    hf_config.text_config.ple_layer_ids = [2, 3]
    with pytest.raises(NotImplementedError, match="exactly one PLE layer"):
        parse_config(hf_config)


def test_registry_and_package_exports_are_complete():
    import freetoken.models.qwen4_exp as package
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("Qwen4ExpForConditionalGeneration")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForConditionalGeneration"
    assert package.Qwen4ExpForConditionalGeneration is not None
    assert callable(package.parse_config)
    assert callable(package.iter_weights)
    assert callable(package.setup_offload_expert_banks)


def test_resident_state_dict_exactly_matches_transformers_names_after_loader_plan():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpTextConfig,
    )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    from freetoken.models.qwen4_exp.model import Qwen4ExpForConditionalGeneration
    from freetoken.models.qwen4_exp.weight import _fusion_plan, _rename

    config = replace(parse_config(_tiny_hf_wrapper()), moe_backend="offload")
    # Engine construction supplies a real CUDA device for RoPE while building
    # the remaining tensors on meta.  CPU is the equivalent non-allocating test
    # target; the tables are tiny and are not resident model parameters.
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"):
        actual_model = Qwen4ExpForConditionalGeneration(config)
        hf_model = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(**_tiny_kwargs()))

    actual = {name: tuple(tensor.shape) for name, tensor in actual_model.state_dict().items()}
    expected: dict[str, tuple[int, ...]] = {}
    fusion_parts: dict[str, dict[int, tuple[int, ...]]] = defaultdict(dict)

    for hf_name, tensor in hf_model.state_dict().items():
        # The multimodal release wraps the text model at model.language_model,
        # whereas the independent text-only reference uses model.* directly.
        raw_name = (
            "model.language_model." + hf_name.removeprefix("model.")
            if hf_name.startswith("model.")
            else hf_name
        )
        # The pinned ModelOpt checkpoint stores experts per expert rather than
        # Transformers' stacked tensors.  Both layouts are owned by the offload
        # bank, never by the resident state dict.
        if ".mlp.experts." in raw_name:
            continue
        mapped_name = _rename(raw_name)
        if mapped_name is None:
            continue
        plan = _fusion_plan(mapped_name)
        if plan is None:
            expected[mapped_name] = tuple(tensor.shape)
            continue
        destination, _source_suffixes, slot = plan
        fusion_parts[destination][slot] = tuple(tensor.shape)

    for destination, parts_by_slot in fusion_parts.items():
        parts = [parts_by_slot[slot] for slot in range(len(parts_by_slot))]
        assert all(part[1:] == parts[0][1:] for part in parts)
        expected[destination] = (sum(part[0] for part in parts), *parts[0][1:])

    assert actual == expected
    assert not any("ple_embedding" in name for name in actual)
    assert not any(".experts." in name for name in actual)


def test_transformers_tiny_full_prefill_matches_token_decode_logits():
    """Independent CPU architecture oracle, including GDN, QSA, PLE and MoE.

    This does not pretend to exercise FreeToken's CUDA kernels.  It proves that
    the exact tiny architecture used by the config/structure tests has coherent
    prefill/decode cache semantics in the upstream operator implementation.
    Native FreeToken parity remains a GPU test.
    """

    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpTextConfig,
    )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    oracle_kwargs = {
        **_tiny_kwargs(),
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "qwen_sparse_attention"],
        "linear_num_value_heads": 2,
        "linear_conv_kernel_dim": 3,
        "hc_lowrank": 4,
        "indexer_n_heads": 2,
        "indexer_head_dim": 8,
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
        "ple_layer_ids": [1],
        "ple_embed_dim": 16,
        "ple_conv_kernel_size": 3,
        "ngram_vocab_size_base": 11,
        "vocab_size": 29,
        "eos_token_id": 2,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
        },
    }
    torch.manual_seed(7)
    model = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(**oracle_kwargs)).eval()
    input_ids = torch.tensor([[5, 7, 11, 3, 13, 17]])

    with torch.no_grad():
        full_logits = model(input_ids, use_cache=False).logits
        past_key_values = None
        decode_logits = []
        for token_index in range(input_ids.shape[1]):
            output = model(
                input_ids[:, token_index : token_index + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            decode_logits.append(output.logits)

    torch.testing.assert_close(
        full_logits,
        torch.cat(decode_logits, dim=1),
        atol=1e-6,
        rtol=1e-5,
    )
