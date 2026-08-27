from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.qwen4_exp.config import Qwen4ExpVisionConfig, parse_config
from freetoken.models.qwen4_exp.vision import Qwen4ExpVisionModel
from freetoken.models.qwen4_exp.model import Qwen4ExpModel


if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


def _runtime_config() -> Qwen4ExpVisionConfig:
    return Qwen4ExpVisionConfig(
        depth=2,
        hidden_size=16,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
    )


def test_vision_tower_matches_transformers_tiny():
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpVisionConfig as HFVisionConfig,
    )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel as HFVisionModel

    torch.manual_seed(17)
    runtime = _runtime_config()
    hf_config = HFVisionConfig(
        depth=runtime.depth,
        hidden_size=runtime.hidden_size,
        hidden_act=runtime.hidden_act,
        intermediate_size=runtime.intermediate_size,
        num_heads=runtime.num_heads,
        in_channels=runtime.in_channels,
        patch_size=runtime.patch_size,
        spatial_merge_size=runtime.spatial_merge_size,
        temporal_patch_size=runtime.temporal_patch_size,
        out_hidden_size=runtime.out_hidden_size,
        num_position_embeddings=runtime.num_position_embeddings,
    )
    hf_config._attn_implementation = "sdpa"
    reference = HFVisionModel(hf_config).eval()
    actual = Qwen4ExpVisionModel(runtime)

    expected_state = dict(reference.state_dict())
    assert actual.state_dict().keys() == expected_state.keys()
    actual.load_state_dict(dict(expected_state))

    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixels = torch.randn(
        16,
        runtime.in_channels
        * runtime.temporal_patch_size
        * runtime.patch_size
        * runtime.patch_size,
    )
    with torch.inference_mode():
        expected = reference(pixels, grid_thw=grid, return_dict=True).pooler_output
        observed = actual.forward(pixels, grid)
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-4)


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=64,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        norm_topk_prob=True,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
        hc_count=4,
        hc_lowrank=8,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=8,
        indexer_compress_ratio=4,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=64,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        seed=1234,
        eos_token_id=63,
        ple_embedding_dtype="float8_e4m3fn",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.25,
        },
    )


def test_parse_vision_is_opt_in(monkeypatch):
    wrapper = SimpleNamespace(
        text_config=_text_config(),
        vision_config=SimpleNamespace(
            depth=2,
            hidden_size=16,
            hidden_act="gelu_pytorch_tanh",
            intermediate_size=32,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=16,
        ),
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"},
        image_token_id=60,
    )
    monkeypatch.delenv("FREETOKEN_LOAD_VISION", raising=False)
    assert parse_config(wrapper).vision_config is None
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    config = parse_config(wrapper)
    assert config.is_multimodal
    assert config.vision_config == _runtime_config()


def test_multimodal_merge_prefers_explicit_chunk_indices(monkeypatch):
    embeds = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    batch = SimpleNamespace(
        mm_embeds=embeds,
        mm_embed_indices=torch.tensor([1, 3], dtype=torch.int64),
    )
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(batch=batch),
    )
    model = SimpleNamespace(_image_token_id=60)
    hidden = torch.zeros((4, 2))
    # The explicit plan is authoritative; no placeholder scan or .item() is
    # needed in the per-chunk model path.
    result = Qwen4ExpModel._merge_multimodal(
        model, torch.tensor([1, 1, 1, 1]), hidden
    )
    torch.testing.assert_close(result[1], embeds[0])
    torch.testing.assert_close(result[3], embeds[1])
    torch.testing.assert_close(result[[0, 2]], torch.zeros((2, 2)))
