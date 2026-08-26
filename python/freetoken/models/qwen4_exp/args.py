"""Qwen4-Exp text-tower hyperparameters.

The public checkpoint calls the architecture ``Qwen4Exp`` while the released
Qwen3.8-Flash-Next model is one concrete instance.  Keeping the new geometry in
one immutable payload prevents model code from growing checkpoint-name branches
and gives tiny parity tests the same path as the 135 GB checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class Qwen4ExpArgs:
    layer_types: Tuple[str, ...]
    hc_count: int
    hc_lowrank: int
    output_gate_type: str

    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int

    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    seed: int
    eos_token_id: int
    ple_embedding_dtype: str

    @property
    def qsa_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "qwen_sparse_attention")

    @property
    def linear_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "linear_attention")

    def has_ple(self, zero_based_layer_id: int) -> bool:
        """PLE ids in the HF config are deliberately one-indexed."""
        return zero_based_layer_id + 1 in self.ple_layer_ids

    def ple_index(self, zero_based_layer_id: int) -> int:
        return self.ple_layer_ids.index(zero_based_layer_id + 1)


def _layer_types(text: Any) -> Tuple[str, ...]:
    raw = getattr(text, "layer_types", None)
    if raw is None:
        interval = int(getattr(text, "full_attention_interval", 4))
        raw = [
            "qwen_sparse_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(int(text.num_hidden_layers))
        ]
    # The released config predates the native Transformers rename and says
    # full_attention even though every such layer owns a QSA indexer.
    return tuple("qwen_sparse_attention" if kind == "full_attention" else str(kind) for kind in raw)


def load_args(text: Any) -> Qwen4ExpArgs:
    layer_types = _layer_types(text)
    allowed = {"linear_attention", "qwen_sparse_attention"}
    unsupported = sorted(set(layer_types) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}")
    if len(layer_types) != int(text.num_hidden_layers):
        raise ValueError(
            "Qwen4-Exp layer_types length must equal num_hidden_layers, got "
            f"{len(layer_types)} != {text.num_hidden_layers}"
        )

    hc_count = int(getattr(text, "hc_count", 4))
    hc_lowrank = int(getattr(text, "hc_lowrank", 320))
    if hc_count <= 1 or hc_lowrank <= 0:
        raise ValueError(f"Invalid Qwen4-Exp hyper-connection geometry: {hc_count=}, {hc_lowrank=}")

    output_gate_type = str(getattr(text, "output_gate_type", None) or text.hidden_act)
    if output_gate_type not in {"sigmoid", "silu"}:
        raise ValueError(f"Unsupported Qwen4-Exp GDN output gate: {output_gate_type!r}")

    qsa_values = {
        "indexer_n_heads": int(getattr(text, "indexer_n_heads")),
        "indexer_kv_heads": int(getattr(text, "indexer_kv_heads")),
        "indexer_head_dim": int(getattr(text, "indexer_head_dim")),
        "indexer_budget": int(getattr(text, "indexer_budget")),
        "indexer_compress_ratio": int(getattr(text, "indexer_compress_ratio")),
    }
    if any(v <= 0 for v in qsa_values.values()) or qsa_values["indexer_kv_heads"] != 1:
        raise ValueError(f"Invalid Qwen4-Exp QSA geometry: {qsa_values}")
    if qsa_values["indexer_budget"] % qsa_values["indexer_compress_ratio"]:
        raise ValueError("QSA token budget must be divisible by the compression ratio")

    ple_layer_ids = tuple(sorted(set(int(x) for x in getattr(text, "ple_layer_ids", ()))))
    if len(ple_layer_ids) != 1:
        raise NotImplementedError(
            "The first Qwen4-Exp text milestone supports exactly one PLE layer "
            f"(the pinned checkpoint uses layer id 2), got {ple_layer_ids}"
        )
    for layer_id in ple_layer_ids:
        if not 1 <= layer_id <= len(layer_types):
            raise ValueError(f"PLE layer id {layer_id} is outside [1, {len(layer_types)}]")
        if layer_types[layer_id - 1] != "linear_attention":
            raise ValueError(f"PLE layer {layer_id} must be a linear-attention layer")

    eos = getattr(text, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    if ple_layer_ids and eos is None:
        raise ValueError("Qwen4-Exp PLE requires eos_token_id")

    ple_embed_dim = int(getattr(text, "ple_embed_dim", None) or text.hidden_size)
    ngram_size = int(getattr(text, "ngram_size", 3))
    heads_per_ngram = int(getattr(text, "heads_per_ngram", 8))
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if ngram_heads <= 0 or ple_embed_dim % ngram_heads:
        raise ValueError(
            f"ple_embed_dim ({ple_embed_dim}) must be divisible by n-gram heads ({ngram_heads})"
        )

    return Qwen4ExpArgs(
        layer_types=layer_types,
        hc_count=hc_count,
        hc_lowrank=hc_lowrank,
        output_gate_type=output_gate_type,
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=int(getattr(text, "ple_conv_kernel_size", 4)),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=int(getattr(text, "ngram_vocab_size_base", 20_000_000)),
        make_ngram_vocab_size_divisible_by=int(
            getattr(text, "make_ngram_vocab_size_divisible_by", 128)
        ),
        split_ngram_parts=int(getattr(text, "split_ngram_parts", 128)),
        seed=int(getattr(text, "seed", 1234)),
        eos_token_id=int(eos if eos is not None else 0),
        ple_embedding_dtype=str(getattr(text, "ple_embedding_dtype", "float8_e4m3fn")),
        **qsa_values,
    )


__all__ = ["Qwen4ExpArgs", "load_args"]
