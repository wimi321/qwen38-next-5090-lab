"""Translate a Hugging Face Qwen4-Exp config into FreeToken's execution config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import (
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    QSAAttentionGroupConfig,
    RotaryConfig,
    detect_expert_quant,
    vision_load_enabled,
)

from .args import load_args


# Formal 30-minute RTX 5090/WSL2 evidence showed the lazy MoE LRU approaching
# full residency late in the soak and peaking at 31,940 MiB with the generic
# auto budget.  Keep the public 0.89 profile contract while reserving enough
# physical-commit headroom to remain below the strict 31 GiB release ceiling.
_QWEN4_EXP_MOE_AUTO_RUNTIME_RESERVE_BYTES = 512 * 2**20


@dataclass(frozen=True)
class Qwen4ExpVisionConfig:
    """Runtime geometry of the Qwen3.8/Qwen4-Exp image tower.

    The names deliberately follow the Hugging Face configuration.  Keeping the
    payload immutable lets the model builder create the exact resident vision
    tensors without retaining the much larger top-level Transformers config.
    """

    depth: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    out_hidden_size: int
    num_position_embeddings: int


def _scalar_size(value: Any, name: str) -> int:
    """Normalize HF's scalar-or-sequence patch fields for the pinned model."""

    if isinstance(value, (list, tuple)):
        if not value or len(set(int(item) for item in value)) != 1:
            raise ValueError(f"Qwen4-Exp vision {name} must be a uniform size, got {value!r}")
        value = value[0]
    value = int(value)
    if value <= 0:
        raise ValueError(f"Qwen4-Exp vision {name} must be positive, got {value}")
    return value


def _parse_vision_config(hf_config: Any, text_hidden_size: int) -> Qwen4ExpVisionConfig | None:
    """Return the opt-in image tower config; the legacy 8K profile stays text-only."""

    vision = getattr(hf_config, "vision_config", None)
    if vision is None or not vision_load_enabled():
        return None
    out_hidden_size = int(getattr(vision, "out_hidden_size", text_hidden_size))
    if out_hidden_size != text_hidden_size:
        raise ValueError(
            "Qwen4-Exp vision merger output must equal the text hidden size: "
            f"{out_hidden_size} != {text_hidden_size}"
        )
    depth = int(getattr(vision, "depth", getattr(vision, "num_hidden_layers", 0)))
    num_heads = int(getattr(vision, "num_heads", getattr(vision, "num_attention_heads", 0)))
    if depth <= 0 or num_heads <= 0 or int(vision.hidden_size) % num_heads:
        raise ValueError(
            "Qwen4-Exp vision depth/heads must be positive and hidden_size divisible by heads"
        )
    return Qwen4ExpVisionConfig(
        depth=depth,
        hidden_size=int(vision.hidden_size),
        hidden_act=str(getattr(vision, "hidden_act", "gelu_pytorch_tanh")),
        intermediate_size=int(vision.intermediate_size),
        num_heads=num_heads,
        in_channels=int(getattr(vision, "in_channels", 3)),
        patch_size=_scalar_size(vision.patch_size, "patch_size"),
        spatial_merge_size=_scalar_size(vision.spatial_merge_size, "spatial_merge_size"),
        temporal_patch_size=_scalar_size(
            vision.temporal_patch_size, "temporal_patch_size"
        ),
        out_hidden_size=out_hidden_size,
        num_position_embeddings=int(vision.num_position_embeddings),
    )


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)
    args = load_args(text)

    head_dim = int(getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads)
    num_kv_heads = int(getattr(text, "num_key_value_heads", text.num_attention_heads))
    rope = getattr(text, "rope_parameters", None) or {}
    partial = float(rope.get("partial_rotary_factor", getattr(text, "partial_rotary_factor", 1.0)))
    rotary_dim = int(head_dim * partial)
    mrope_section_raw = rope.get("mrope_section")
    mrope_section = (
        tuple(int(value) for value in mrope_section_raw)
        if mrope_section_raw is not None
        else None
    )
    if mrope_section is not None and sum(mrope_section) != rotary_dim // 2:
        raise ValueError(
            "Qwen4-Exp mrope_section must cover half the rotary dimensions: "
            f"sum({mrope_section}) != {rotary_dim // 2}"
        )
    rope_type = rope.get("rope_type", "default")
    if rope_type not in (None, "default"):
        raise ValueError(f"Qwen4-Exp text milestone supports default RoPE only, got {rope_type!r}")
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(text.max_position_embeddings),
        base=float(rope.get("rope_theta", getattr(text, "rope_theta", 10_000.0))),
        scaling=None,
        mrope_section=mrope_section,
        mrope_interleaved=bool(rope.get("mrope_interleaved", mrope_section is not None)),
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
        vision_config=_parse_vision_config(hf_config, int(text.hidden_size)),
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        qwen4_args=args,
        moe_auto_runtime_reserve_bytes=_QWEN4_EXP_MOE_AUTO_RUNTIME_RESERVE_BYTES,
    )


__all__ = ["Qwen4ExpVisionConfig", "parse_config"]
