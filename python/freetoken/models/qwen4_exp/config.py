"""Translate a Hugging Face Qwen4-Exp config into FreeToken's execution config."""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    QSAAttentionGroupConfig,
    RotaryConfig,
    detect_expert_quant,
)

from .args import load_args


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)
    args = load_args(text)

    head_dim = int(getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads)
    num_kv_heads = int(getattr(text, "num_key_value_heads", text.num_attention_heads))
    rope = getattr(text, "rope_parameters", None) or {}
    partial = float(rope.get("partial_rotary_factor", getattr(text, "partial_rotary_factor", 1.0)))
    rotary_dim = int(head_dim * partial)
    rope_type = rope.get("rope_type", "default")
    if rope_type not in (None, "default"):
        raise ValueError(f"Qwen4-Exp text milestone supports default RoPE only, got {rope_type!r}")
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(text.max_position_embeddings),
        base=float(rope.get("rope_theta", getattr(text, "rope_theta", 10_000.0))),
        scaling=None,
    )
    if rotary_dim > args.indexer_head_dim:
        raise ValueError(
            f"QSA index head ({args.indexer_head_dim}) cannot hold rotary dim ({rotary_dim})"
        )

    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=args.linear_layer_ids,
        num_key_heads=int(text.linear_num_key_heads),
        num_value_heads=int(text.linear_num_value_heads),
        key_head_dim=int(text.linear_key_head_dim),
        value_head_dim=int(text.linear_value_head_dim),
        conv_kernel_dim=int(text.linear_conv_kernel_dim),
        output_gate=True,
    )
    qsa_group = QSAAttentionGroupConfig(
        name="qsa",
        layer_ids=args.qsa_layer_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
        indexer_n_heads=args.indexer_n_heads,
        indexer_kv_heads=args.indexer_kv_heads,
        indexer_head_dim=args.indexer_head_dim,
        indexer_budget=args.indexer_budget,
        indexer_compress_ratio=args.indexer_compress_ratio,
    )
    groups = tuple(sorted((linear_group, qsa_group), key=lambda g: g.layer_ids[0]))

    expert_quant = detect_expert_quant(hf_config)
    if expert_quant != "nvfp4":
        raise ValueError(
            "The first Qwen4-Exp milestone is pinned to the NVFP4 routed-expert checkpoint; "
            f"detected expert quantization {expert_quant!r}"
        )

    return ModelConfig(
        num_layers=int(text.num_hidden_layers),
        num_qo_heads=int(text.num_attention_heads),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        intermediate_size=int(getattr(text, "intermediate_size", 0) or 0),
        rms_norm_eps=float(text.rms_norm_eps),
        rotary_config=rotary,
        hidden_act=str(text.hidden_act),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        num_experts=int(text.num_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        shared_expert_intermediate_size=int(text.shared_expert_intermediate_size),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        model_type=str(getattr(hf_config, "model_type", "qwen4_exp")),
        architectures=list(
            getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"])
        ),
        moe_enabled=True,
        expert_quant="nvfp4",
        # The RadixArk revision ignores attention, shared expert, PLE and lm_head
        # in its ModelOpt policy.  Only routed experts stay native NVFP4.
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        use_qk_norm=True,
        vision_config=None,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        qwen4_args=args,
    )


__all__ = ["parse_config"]
