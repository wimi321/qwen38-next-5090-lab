"""Qwen4-Exp / Qwen3.8-Flash-Next text and opt-in image model."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Sequence

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.utils import download_hf_weight, nvtx_annotate

from .attention import Qwen4ExpAttention
from .hyperconnection import Qwen4ExpGatedResidual
from .moe import Qwen4ExpMoE
from .ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpNGramHasher,
    Qwen4ExpPLELayer,
    SafetensorsRowShard,
    ShardedSafetensorsMmapRowBank,
)
from .vision import Qwen4ExpVisionModel

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


_DEBUG_FINITE = os.getenv("Q38LAB_DEBUG_FINITE", "0") == "1"


def _debug_require_finite(label: str, value: torch.Tensor) -> None:
    """Fail at the first corrupt Qwen4-Exp activation during opt-in bring-up.

    The check deliberately stays behind an import-time environment flag: each
    ``isfinite().all().item()`` synchronizes CUDA and would otherwise distort
    serving latency.  Keeping the diagnostic next to the model makes a full
    checkpoint failure identify the first PLE/attention/MoE boundary instead of
    surfacing later as greedy token id 0 from all-NaN logits.
    """

    if not _DEBUG_FINITE:
        return
    finite = torch.isfinite(value)
    if bool(finite.all().item()):
        return
    invalid = int((~finite).sum().item())
    raise RuntimeError(
        f"Qwen4-Exp non-finite activation at {label}: "
        f"shape={tuple(value.shape)}, dtype={value.dtype}, invalid={invalid}"
    )


class _DeferredRowBank:
    """Shape-only placeholder; never allocates the ~51 GB PLE tensor."""

    def __init__(self, row_count: int, row_width: int):
        self._row_count = int(row_count)
        self._row_width = int(row_width)

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def row_width(self) -> int:
        return self._row_width

    def read_rows(self, *args, **kwargs):
        raise RuntimeError("Qwen4-Exp PLE auxiliary bank has not been bound")


class _ZeroRowBank(_DeferredRowBank):
    """No-allocation dummy-weight bank for structural/kernel smoke tests."""

    def read_rows(self, indices, *, dtype=None, device=None):
        ids = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        return torch.zeros(
            *ids.shape,
            self.row_width,
            dtype=dtype or torch.bfloat16,
            device=device or "cpu",
        )


def _make_ple(config: ModelConfig, layer_id: int) -> Qwen4ExpPLELayer:
    args = config.qwen4_args
    ple_index = args.ple_index(layer_id)
    # Model construction happens under torch.device("meta").  Hash constants are
    # tiny CPU-side auxiliary metadata, not state_dict weights, so build them
    # explicitly on CPU and never let the generic materializer touch them.
    with torch.device("cpu"):
        hasher = Qwen4ExpNGramHasher(
            unigram_vocab_size=config.vocab_size,
            eos_token_id=args.eos_token_id,
            ngram_vocab_size_base=args.ngram_vocab_size_base,
            ngram_size=args.ngram_size,
            heads_per_ngram=args.heads_per_ngram,
            ple_layer_index=ple_index,
            seed=args.seed,
            make_vocab_size_divisible_by=args.make_ngram_vocab_size_divisible_by,
        )
    row_width = args.ple_embed_dim // hasher.num_heads
    embedding = Qwen4ExpNGramEmbedding(
        hasher, _DeferredRowBank(hasher.layout.padded_vocab_size, row_width)
    )
    ple = Qwen4ExpPLELayer(
        embedding,
        hidden_size=config.hidden_size,
        hc_count=args.hc_count,
        conv_kernel_size=args.ple_conv_kernel_size,
        rms_norm_eps=config.rms_norm_eps,
    )
    ple._checkpoint_layer_id = layer_id
    ple._ple_index = ple_index
    return ple


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        args = config.qwen4_args
        if self._is_linear:
            group = config.linear_attention_group()
            assert group is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=group.num_key_heads,
                num_v_heads=group.num_value_heads,
                head_k_dim=group.key_head_dim,
                head_v_dim=group.value_head_dim,
                conv_kernel_size=group.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
                output_gate_type=args.output_gate_type,
            )
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.ple = _make_ple(config, layer_id) if args.has_ple(layer_id) else None
        self.attn_hyper_connection = Qwen4ExpGatedResidual(
            config.hidden_size,
            hc_count=args.hc_count,
            hc_lowrank=args.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(
            config.hidden_size,
            hc_count=args.hc_count,
            hc_lowrank=args.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
        )

    @staticmethod
    def _inject(
        block_output: torch.Tensor,
        hyper_input: torch.Tensor,
        injection_weights: torch.Tensor,
    ) -> torch.Tensor:
        injection = block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        return hyper_input + injection.flatten(-2)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        _debug_require_finite(f"layer.{self._layer_id}.input", hidden_states)
        if self.ple is not None:
            if ctx.ple_state_pool is None:
                raise RuntimeError("Qwen4-Exp PLE state pool is not initialized")
            ple_output = self.ple.forward_flat(
                hidden_states, ctx.batch, ctx.ple_state_pool
            )
            _debug_require_finite(f"layer.{self._layer_id}.ple", ple_output)
            hidden_states = hidden_states + ple_output
            _debug_require_finite(f"layer.{self._layer_id}.post_ple", hidden_states)

        mixed, hyper_input, write = self.attn_hyper_connection.forward(hidden_states)
        _debug_require_finite(f"layer.{self._layer_id}.attn_read", mixed)
        block = (
            self.linear_attn.forward(mixed)
            if self._is_linear
            else self.self_attn.forward(mixed)
        )
        _debug_require_finite(f"layer.{self._layer_id}.attn_block", block)
        hidden_states = self._inject(block, hyper_input, write)
        _debug_require_finite(f"layer.{self._layer_id}.post_attn", hidden_states)

        mixed, hyper_input, write = self.mlp_hyper_connection.forward(hidden_states)
        _debug_require_finite(f"layer.{self._layer_id}.mlp_read", mixed)
        block = self.mlp.forward(mixed)
        _debug_require_finite(f"layer.{self._layer_id}.mlp_block", block)
        hidden_states = self._inject(block, hyper_input, write)
        _debug_require_finite(f"layer.{self._layer_id}.output", hidden_states)
        return hidden_states


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig):
        args = config.qwen4_args
        self._args = args
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            config.hidden_size,
            hc_count=args.hc_count,
            hc_lowrank=args.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            use_combine=False,
        )
        self._image_token_id = config.image_token_id

    def _merge_multimodal(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Replace image placeholders before the four residual streams are copied."""

        mm_embeds = getattr(get_global_ctx().batch, "mm_embeds", None)
        if mm_embeds is None or self._image_token_id is None:
            return hidden_states
        indices = getattr(get_global_ctx().batch, "mm_embed_indices", None)
        if indices is not None:
            slots = int(indices.numel())
            if slots != int(mm_embeds.shape[0]):
                raise ValueError(
                    "Qwen4-Exp image embedding/index mismatch in the current prefill chunk: "
                    f"{slots} destinations != {mm_embeds.shape[0]} feature rows"
                )
            # The scheduler already computed exact destinations while slicing
            # the CPU-resident image plan.  Reusing them avoids mask.sum().item(),
            # which would synchronize the 5090 once per 512-token chunk.
            merged = hidden_states.clone()
            merged.index_copy_(
                0,
                indices.to(device=hidden_states.device, dtype=torch.long),
                mm_embeds.to(hidden_states.device, hidden_states.dtype),
            )
            return merged
        mask = input_ids == self._image_token_id
        slots = int(mask.sum().item())
        if slots != int(mm_embeds.shape[0]):
            raise ValueError(
                "Qwen4-Exp image embedding/token mismatch in the current prefill chunk: "
                f"{slots} placeholders != {mm_embeds.shape[0]} feature rows"
            )
        return hidden_states.masked_scatter(
            mask.unsqueeze(-1), mm_embeds.to(hidden_states.device, hidden_states.dtype)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens.forward(input_ids)
        _debug_require_finite("embedding", hidden_states)
        hidden_states = self._merge_multimodal(input_ids, hidden_states)
        _debug_require_finite("multimodal_merge", hidden_states)
        hidden_states = hidden_states.repeat(1, self._args.hc_count)
        for layer in self.layers.op_list:
            hidden_states = layer.forward(hidden_states)
        hidden_states = self.hyper_connection_mixer.forward(hidden_states)
        _debug_require_finite("final_mixer", hidden_states)
        return hidden_states

    def ple_layers(self) -> Sequence[Qwen4ExpPLELayer]:
        return tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)


def _open_ple_bank(
    folder: str,
    ple: Qwen4ExpPLELayer,
    *,
    split_parts: int,
    dtype: torch.dtype,
    pin_memory: bool,
):
    index_path = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            "Qwen4-Exp PLE streaming requires model.safetensors.index.json"
        )
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    layer_id = ple._checkpoint_layer_id
    base = (
        f"model.language_model.layers.{layer_id}.ple.ple_embedding."
        "ngram_embedding"
    )
    shard_specs = []
    for shard_id in range(split_parts):
        name = f"{base}.shard_{shard_id}.weight"
        try:
            filename = weight_map[name]
        except KeyError:
            raise KeyError(f"checkpoint is missing PLE shard {name!r}") from None
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        shard_specs.append(SafetensorsRowShard(path, name))

    scale_name = f"{base}.weight_scale"
    try:
        scale_path = os.path.join(folder, weight_map[scale_name])
    except KeyError:
        raise KeyError(f"checkpoint is missing PLE scale {scale_name!r}") from None
    return ShardedSafetensorsMmapRowBank(
        shard_specs,
        weight_scale_path=scale_path,
        weight_scale_name=scale_name,
        default_dtype=dtype,
        pin_memory=pin_memory,
    )


class Qwen4ExpForConditionalGeneration(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self._config = config
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        if config.is_multimodal:
            self.visual = Qwen4ExpVisionModel(config.vision_config)
        super().__init__()

    @torch.inference_mode()
    def encode_images(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        """Encode processor patches once; scheduler slices the returned rows per chunk."""

        visual = getattr(self, "visual", None)
        if visual is None:
            raise RuntimeError(
                "Qwen4-Exp vision is disabled; use the rtx5090-wsl2-256k-image profile "
                "or set FREETOKEN_LOAD_VISION=1 before model construction"
            )
        return visual.forward(pixel_values, image_grid_thw)

    def setup_auxiliary_banks(
        self,
        *,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        dummy: bool = False,
    ):
        banks = {}
        folder = download_hf_weight(model_path) if not dummy else model_path
        args = self._config.qwen4_args
        for ple in self.model.ple_layers():
            if dummy:
                hasher = ple.ple_embedding.hasher
                bank = _ZeroRowBank(
                    hasher.layout.padded_vocab_size,
                    args.ple_embed_dim // hasher.num_heads,
                )
            else:
                bank = _open_ple_bank(
                    folder,
                    ple,
                    split_parts=args.split_ngram_parts,
                    dtype=dtype,
                    pin_memory=device.type == "cuda",
                )
            ple.ple_embedding.bind_row_bank(bank)
            banks[f"ple_layer_{ple._checkpoint_layer_id}"] = bank
        return banks

    def ple_graph_input_spec(self) -> tuple[int, torch.dtype]:
        """Width/dtype of the persistent decode embedding graph input."""

        layers = self.model.ple_layers()
        if len(layers) != 1:
            raise RuntimeError(
                "Qwen4-Exp graph staging currently requires exactly one PLE layer, "
                f"got {len(layers)}"
            )
        ple = layers[0]
        return ple.ple_embedding.embedding_dim, ple.key_proj.weight.dtype

    def prepare_batch_auxiliary(self, batch) -> None:
        """Stage PLE hash/mmap inputs before an eager decode or graph replay."""

        if not batch.is_decode:
            return
        pool = get_global_ctx().ple_state_pool
        if pool is None:
            raise RuntimeError("Qwen4-Exp PLE state pool is not initialized")
        ple_layers = self.model.ple_layers()
        if len(ple_layers) != 1:
            raise RuntimeError(
                "Qwen4-Exp decode staging currently supports exactly one PLE layer"
            )
        ple = ple_layers[0]
        ple.stage_decode_batch(
            batch,
            pool,
            device=pool.device,
            dtype=ple.key_proj.weight.dtype,
        )

    def commit_batch_auxiliary(self, batch) -> None:
        """Publish staged PLE token and convolution state after model success."""

        pool = get_global_ctx().ple_state_pool
        if pool is None:
            return
        ple_layers = self.model.ple_layers()
        if len(ple_layers) == 1:
            ple_layers[0].commit_batch(batch, pool)

    def forward(self) -> torch.Tensor:
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden)
        _debug_require_finite("lm_head", logits)
        return logits


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpModel",
]
