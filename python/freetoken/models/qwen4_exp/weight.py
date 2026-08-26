"""Resident-weight loader for the pinned Qwen3.8-Flash-Next NVFP4 checkpoint.

The released RadixArk checkpoint is a multimodal ModelOpt checkpoint.  Only its
routed experts are NVFP4; the text tower, shared experts, and language-model
head are ordinary resident tensors.  The routed experts and the file-backed PLE
table have independent lifecycles, so neither belongs in the model state dict.

This loader intentionally targets that one layout.  In particular, it does not
try to reinterpret another Qwen4-Exp quantization policy:

* ``model.language_model.*`` becomes FreeToken's ``model.*`` namespace;
* the vision tower, MTP head, routed experts, and all ``ple_embedding`` tensors
  are skipped before their safetensors shards are opened;
* q/k/v, the four GDN input projections, and shared-expert gate/up are fused on
  their output dimension; and
* every norm tensor is passed through verbatim.  Qwen4-Exp modules implement
  their checkpoint's ``(1 + weight)`` convention at execution time.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache
from freetoken.utils import cached_load_hf_config, download_hf_weight, init_logger
from tqdm import tqdm

from .config import parse_config

logger = init_logger(__name__)


# Only per-expert ModelOpt tensors match this expression.  The router and the
# always-resident shared expert deliberately do not.
_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")

# Destination suffix -> checkpoint suffixes in their exact LinearColParallelMerged
# output order.  The released q_proj already contains query | attention-gate, so
# its complete matrix is the first qkv part (it must not be split a second time).
_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    ),
    ".linear_attn.in_proj.weight": (
        ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight",
        ".linear_attn.in_proj_a.weight",
    ),
    ".mlp.shared_expert.gate_up_proj.weight": (
        ".mlp.shared_expert.gate_proj.weight",
        ".mlp.shared_expert.up_proj.weight",
    ),
}


def _uses_w4a4_activation_recipe(hf_config) -> bool:
    """Whether ModelOpt metadata asks for block-16 four-bit activations.

    FreeToken's existing NVFP4 expert banks preserve the checkpoint's packed
    weights/scales but all current execution backends consume BF16 activations
    (W4A16). Detect the semantic gap explicitly so experimental bring-up cannot
    be mistaken for W4A4-equivalent serving.
    """

    quant = getattr(hf_config, "quantization_config", None)
    if not isinstance(quant, Mapping):
        return False
    groups = quant.get("config_groups", {})
    if not isinstance(groups, Mapping):
        return False
    for group in groups.values():
        if not isinstance(group, Mapping):
            continue
        activation = group.get("input_activations")
        if not isinstance(activation, Mapping):
            continue
        if (
            int(activation.get("num_bits", 0)) == 4
            and int(activation.get("group_size", 0)) == 16
            and str(activation.get("type", "")).lower() == "float"
        ):
            return True
    return False


def _rename(raw_name: str) -> str | None:
    """Map one HF tensor name to a resident FreeToken key, or skip it.

    Skips are decided from the index name.  Consequently the 51 GB PLE table,
    routed-expert shards, vision tower, and MTP tensors are never materialized by
    the normal state-dict pass.
    """

    if raw_name.startswith(
        (
            "mtp.",
            "model.mtp.",
            "model.visual.",
            "visual.",
            # Be explicit about wrapper variants even though the pinned revision
            # currently stores visual/MTP outside model.language_model.
            "model.language_model.mtp.",
            "model.language_model.visual.",
        )
    ):
        return None
    if _ROUTED_EXPERT_RE.search(raw_name):
        return None
    # The auxiliary bank owns the 128 FP8 shards, their scale, and all hash
    # metadata.  Hash metadata is derived from Qwen4ExpArgs, not loaded as model
    # state, so the whole subtree is excluded rather than only ``*.weight``.
    if ".ple.ple_embedding." in raw_name:
        return None

    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _fusion_plan(
    name: str,
) -> tuple[str, tuple[str, ...], int] | None:
    """Return ``(destination, ordered source suffixes, current slot)`` if fused."""

    for destination_suffix, source_suffixes in _FUSIONS.items():
        for slot, source_suffix in enumerate(source_suffixes):
            if name.endswith(source_suffix):
                prefix = name[: -len(source_suffix)]
                return prefix + destination_suffix, source_suffixes, slot
    return None


class _IndexedShardReader:
    """Open indexed safetensors shards lazily and only when a kept key is read."""

    def __init__(
        self,
        folder: str,
        weight_map: dict[str, str],
        device: torch.device,
    ) -> None:
        self.folder = folder
        self.weight_map = weight_map
        self.device = str(device)
        self._handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self.folder, shard),
                framework="pt",
                device=self.device,
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        opened = tuple(self._handles.items())
        self._handles.clear()
        for _shard, handle in opened:
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best-effort handle cleanup
                pass
        # Windows has no posix_fadvise.  FreeToken serving is Linux-only, while
        # this guard keeps the synthetic CPU loader tests platform-independent.
        if hasattr(os, "posix_fadvise"):
            for shard, _handle in opened:
                drop_page_cache(os.path.join(self.folder, shard))


def _read_fusion(
    reader: _IndexedShardReader,
    raw_name: str,
    mapped_name: str,
    plan: tuple[str, tuple[str, ...], int],
) -> tuple[str, torch.Tensor, tuple[str, ...]]:
    """Read and concatenate one complete fusion group, including cross-shard groups."""

    destination, source_suffixes, current_slot = plan
    raw_prefix = raw_name[: -len(source_suffixes[current_slot])]
    raw_parts = tuple(raw_prefix + suffix for suffix in source_suffixes)
    missing = [part for part in raw_parts if part not in reader.weight_map]
    if missing:
        raise ValueError(
            f"Qwen4-Exp checkpoint has an incomplete fusion for {destination}: "
            f"missing {missing}"
        )

    # Verify that leading-prefix rewriting did not accidentally group unrelated
    # tensors.  This is cheap and makes a future checkpoint layout change fail
    # loudly instead of silently permuting a projection.
    mapped_parts = tuple(_rename(part) for part in raw_parts)
    expected_prefix = mapped_name[: -len(source_suffixes[current_slot])]
    expected_parts = tuple(expected_prefix + suffix for suffix in source_suffixes)
    if mapped_parts != expected_parts:
        raise ValueError(
            f"Qwen4-Exp fusion namespace mismatch for {destination}: "
            f"{mapped_parts} != {expected_parts}"
        )

    tensors = [reader.get(part) for part in raw_parts]
    return destination, torch.cat(tensors, dim=0), raw_parts


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield resident text-tower weights for the pinned RadixArk NVFP4 layout.

    Routed experts are available only through the expert-bank hooks below.  PLE
    table rows are read by the auxiliary mmap bank and are therefore omitted in
    both resident and expert passes.
    """

    if include_moe_experts:
        raise NotImplementedError(
            "Qwen4-Exp routed experts are ModelOpt NVFP4 and must be loaded through "
            "the offload expert-bank hooks (use --moe-backend offload)"
        )
    if not include_non_moe:
        return

    tp_info = get_tp_info()
    if tp_info.size != 1:
        raise NotImplementedError("Qwen4-Exp weight loading currently supports TP=1 only")

    # Besides producing the execution config, parse_config is the checkpoint-policy
    # guard: it rejects anything whose routed experts are not the pinned ModelOpt
    # NVFP4 format.  No alternate dense quantization path is intentionally present.
    hf_config = cached_load_hf_config(model_path)
    parse_config(hf_config)
    if _uses_w4a4_activation_recipe(hf_config):
        logger.warning_rank0(
            "Qwen3.8 checkpoint declares ModelOpt NVFP4 W4A4 activations, but "
            "FreeToken's current NVFP4 expert kernels execute the preserved "
            "weights with BF16 activations (W4A16 compatibility). This experimental "
            "path is not checkpoint-numerically equivalent; do not claim W4A4 "
            "quality or parity from this run."
        )

    folder = download_hf_weight(model_path)
    index_path = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            "Qwen4-Exp expects the indexed RadixArk checkpoint; missing "
            f"{index_path}"
        )
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Qwen4-Exp checkpoint index has no weight_map entries: {index_path}")

    reader = _IndexedShardReader(folder, weight_map, device)
    consumed_fusion_parts: set[str] = set()
    emitted_fusions: set[str] = set()
    try:
        for raw_name in tqdm(
            weight_map,
            desc="Loading Qwen4-Exp resident weights",
            disable=not tp_info.is_primary(),
        ):
            if raw_name in consumed_fusion_parts:
                continue
            name = _rename(raw_name)
            if name is None:
                continue

            plan = _fusion_plan(name)
            if plan is not None:
                destination, tensor, raw_parts = _read_fusion(
                    reader, raw_name, name, plan
                )
                if destination in emitted_fusions:
                    # Every part was added to consumed_fusion_parts on first emit;
                    # reaching this branch means the index contains an alias/duplicate.
                    raise ValueError(f"duplicate Qwen4-Exp fusion destination {destination}")
                consumed_fusion_parts.update(raw_parts)
                emitted_fusions.add(destination)
                yield destination, tensor
                continue

            # Deliberately no ``+ 1`` norm conversion here.  Hyper-connection,
            # PLE, q/k, indexer and decoder norms all retain checkpoint values.
            yield name, reader.get(raw_name)
    finally:
        reader.close()


# The expert tensor schema is byte-for-byte identical to Qwen3.5 ModelOpt NVFP4:
# model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,
# weight_scale,weight_scale_2}.  Keep one implementation of allocation, placement,
# parallel I/O, and pin-after-fill; imports stay lazy so CPU rename/fusion tests do
# not load optional CUDA dequantization modules.
def setup_offload_expert_banks(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    decode_target: str = "gpu",
    layer_sink=None,
):
    from freetoken.models.qwen3_5_moe.weight import (
        setup_offload_expert_banks as qwen35_setup_offload_expert_banks,
    )

    return qwen35_setup_offload_expert_banks(
        model_path,
        model_config,
        device=device,
        dtype=dtype,
        dummy=dummy,
        parallel=parallel,
        workers=workers,
        chunk=chunk,
        decode_target=decode_target,
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    from freetoken.models.qwen3_5_moe.weight import (
        load_nvfp4_expert_sources as qwen35_load_nvfp4_expert_sources,
    )

    return qwen35_load_nvfp4_expert_sources(
        model_path, config, layer_sink=layer_sink
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str,
    config,
    *,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
):
    from freetoken.models.qwen3_5_moe.weight import (
        load_nvfp4_expert_sources_parallel as qwen35_load_nvfp4_expert_sources_parallel,
    )

    return qwen35_load_nvfp4_expert_sources_parallel(
        model_path,
        config,
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "iter_weights",
    "setup_offload_expert_banks",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
