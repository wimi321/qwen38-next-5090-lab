#!/usr/bin/env python3
# Copyright 2026 Qwen3.8 Next 5090 Lab contributors.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, sparse parity probe for the pinned checkpoint's PLE row bank.

This module lives inside ``q38lab`` so the 256K preflight is available from
both a source checkout and the Linux wheel.

The release harness already verifies every checkpoint file against the pinned
manifest.  This probe answers a different question: does the runtime auxiliary
loader map the indexed PLE tensors, shard boundaries, hash hits, FP8 scale and
native sparse reads to the same values as safetensors itself?

The default CLI is intentionally release-strict: Linux under WSL2, an RTX 5090
(SM120), the native io_uring/O_DIRECT backend, and CUDA FP8 decoding are all
required.  ``--debug-nonrelease`` exposes the mmap/CPU seam used by unit tests;
its report is explicitly marked ``release_qualified=false``.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import platform
import struct
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from safetensors import safe_open

from freetoken.models.qwen4_exp.ple import (
    Qwen4ExpNGramHasher,
    SafetensorsRowShard,
    ShardedSafetensorsMmapRowBank,
)
from freetoken.models.qwen4_exp.weight import _rename


SCHEMA_VERSION = "1.0"
RELEASE_BACKEND = "io_uring_odirect"
PROBE_CACHE_BYTES = 64 * 1024**2
PROBE_QUEUE_DEPTH = 512
PROBE_MAX_BATCH_PAGES = 4096
_FP8_DTYPES = frozenset({"F8_E4M3", "F8_E4M3FN", "F8_E5M2"})


class ProbeError(RuntimeError):
    """A release-significant layout, runtime or parity failure."""


@dataclass(frozen=True)
class PLEShard:
    index: int
    tensor_name: str
    file_name: str
    path: Path
    shape: tuple[int, int]
    dtype: str
    global_start: int
    global_end: int

    @property
    def rows(self) -> int:
        return self.shape[0]


@dataclass(frozen=True)
class PLELayout:
    model_dir: Path
    config_sha256: str
    index_sha256: str
    checkpoint_layer_id: int
    ple_layer_index: int
    shards: tuple[PLEShard, ...]
    scale_name: str
    scale_file_name: str
    scale_path: Path
    scale_shape: tuple[int, ...]
    scale_dtype: str
    vocab_size: int
    eos_token_id: int
    ple_embed_dim: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_vocab_size_divisible_by: int
    seed: int
    expected_rows: int

    @property
    def row_count(self) -> int:
        return self.shards[-1].global_end

    @property
    def row_width(self) -> int:
        return self.shards[0].shape[1]

    @property
    def num_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram


@dataclass(frozen=True)
class SamplePoint:
    kind: str
    label: str
    global_row: int
    shard_index: int
    shard_row: int
    hash_token_position: int | None = None
    hash_head: int | None = None


BankFactory = Callable[[PLELayout, str, torch.device], Any]
GroundTruthReader = Callable[[PLELayout, Sequence[int]], dict[int, torch.Tensor]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read {description} {path.name!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"{description} {path.name!r} must contain a JSON object")
    return value


def _required_int(source: dict[str, Any], name: str, *, minimum: int = 0) -> int:
    value = source.get(name)
    if isinstance(value, bool):
        raise ProbeError(f"config field {name!r} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ProbeError(f"config field {name!r} must be an integer") from exc
    if result < minimum:
        raise ProbeError(f"config field {name!r} must be >= {minimum}, got {result}")
    return result


def _defaulted_int(
    source: dict[str, Any], name: str, default: int, *, minimum: int = 0
) -> int:
    """Mirror a loader default while retaining the same strict integer checks."""

    return _required_int(
        {name: source.get(name, default)}, name, minimum=minimum
    )


def _checkpoint_path(model_dir: Path, file_name: Any, *, field: str) -> Path:
    if not isinstance(file_name, str) or not file_name:
        raise ProbeError(f"checkpoint index entry {field!r} has no file name")
    if Path(file_name).is_absolute():
        raise ProbeError(f"checkpoint index entry {field!r} uses an absolute path")
    root = model_dir.resolve()
    path = (root / file_name).resolve()
    if not path.is_relative_to(root):
        raise ProbeError(f"checkpoint index entry {field!r} escapes model_dir")
    if not path.is_file():
        raise ProbeError(f"checkpoint file for {field!r} is missing: {file_name!r}")
    return path


def _tensor_metadata(path: Path, tensor_name: str) -> tuple[tuple[int, ...], str]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if tensor_name not in handle.keys():
                raise ProbeError(
                    f"index maps {tensor_name!r} to {path.name!r}, but the tensor is absent"
                )
            view = handle.get_slice(tensor_name)
            return tuple(int(value) for value in view.get_shape()), str(view.get_dtype())
    except ProbeError:
        raise
    except Exception as exc:
        raise ProbeError(
            f"cannot inspect safetensors metadata for {tensor_name!r}: {exc}"
        ) from exc


def _normalized_eos(text_config: dict[str, Any], top_config: dict[str, Any]) -> int:
    value = text_config.get("eos_token_id", top_config.get("eos_token_id"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, bool) or value is None:
        raise ProbeError("config does not expose a usable eos_token_id")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ProbeError("config eos_token_id must be an integer") from exc
    if result < 0:
        raise ProbeError("config eos_token_id must be non-negative")
    return result


def _build_hasher(layout: PLELayout) -> Qwen4ExpNGramHasher:
    with torch.device("cpu"):
        return Qwen4ExpNGramHasher(
            unigram_vocab_size=layout.vocab_size,
            eos_token_id=layout.eos_token_id,
            ngram_vocab_size_base=layout.ngram_vocab_size_base,
            ngram_size=layout.ngram_size,
            heads_per_ngram=layout.heads_per_ngram,
            ple_layer_index=layout.ple_layer_index,
            seed=layout.seed,
            make_vocab_size_divisible_by=layout.make_vocab_size_divisible_by,
        )


def discover_layout(model_dir: str | os.PathLike[str]) -> PLELayout:
    """Resolve the exact runtime PLE tensor sequence from config + HF index."""

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ProbeError(f"model_dir is not an existing directory: {root}")
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ProbeError("model_dir must contain config.json and model.safetensors.index.json")

    config = _read_json_object(config_path, "model config")
    index = _read_json_object(index_path, "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ProbeError("model.safetensors.index.json has no non-empty weight_map")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ProbeError("config text_config must be a JSON object")

    raw_layer_ids = text.get("ple_layer_ids")
    if not isinstance(raw_layer_ids, list) or len(raw_layer_ids) != 1:
        raise ProbeError("the pinned checkpoint must declare exactly one ple_layer_id")
    try:
        one_based_layer_id = int(raw_layer_ids[0])
    except (TypeError, ValueError) as exc:
        raise ProbeError("config ple_layer_ids must contain one integer") from exc
    if one_based_layer_id <= 0:
        raise ProbeError("config ple_layer_id must be one-based and positive")
    checkpoint_layer_id = one_based_layer_id - 1
    split_parts = _required_int(text, "split_ngram_parts", minimum=1)
    vocab_size = _required_int(text, "vocab_size", minimum=1)
    ple_embed_dim = (
        _required_int(text, "ple_embed_dim", minimum=1)
        if "ple_embed_dim" in text
        else _required_int(text, "hidden_size", minimum=1)
    )
    ngram_size = _defaulted_int(text, "ngram_size", 3, minimum=2)
    heads_per_ngram = _defaulted_int(text, "heads_per_ngram", 8, minimum=1)
    ngram_vocab_size_base = _defaulted_int(
        text, "ngram_vocab_size_base", 20_000_000, minimum=2
    )
    divisible_by = _defaulted_int(
        text, "make_ngram_vocab_size_divisible_by", 128, minimum=1
    )
    seed = _defaulted_int(text, "seed", 1234, minimum=0)
    eos_token_id = _normalized_eos(text, config)
    num_heads = (ngram_size - 1) * heads_per_ngram
    if ple_embed_dim % num_heads:
        raise ProbeError(
            f"ple_embed_dim {ple_embed_dim} is not divisible by {num_heads} hash heads"
        )
    expected_width = ple_embed_dim // num_heads

    base = (
        f"model.language_model.layers.{checkpoint_layer_id}.ple.ple_embedding."
        "ngram_embedding"
    )
    shards: list[PLEShard] = []
    global_start = 0
    for shard_index in range(split_parts):
        tensor_name = f"{base}.shard_{shard_index}.weight"
        if _rename(tensor_name, include_vision=False) is not None:
            raise ProbeError(
                f"resident loader no longer skips auxiliary PLE tensor {tensor_name!r}"
            )
        if tensor_name not in weight_map:
            raise ProbeError(f"checkpoint index is missing PLE tensor {tensor_name!r}")
        file_name = weight_map[tensor_name]
        path = _checkpoint_path(root, file_name, field=tensor_name)
        shape, dtype = _tensor_metadata(path, tensor_name)
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != expected_width:
            raise ProbeError(
                f"PLE tensor {tensor_name!r} has shape {shape}; expected [rows, {expected_width}]"
            )
        shard = PLEShard(
            index=shard_index,
            tensor_name=tensor_name,
            file_name=str(file_name),
            path=path,
            shape=(shape[0], shape[1]),
            dtype=dtype,
            global_start=global_start,
            global_end=global_start + shape[0],
        )
        shards.append(shard)
        global_start = shard.global_end

    dtypes = {shard.dtype for shard in shards}
    if len(dtypes) != 1:
        raise ProbeError(f"PLE shards do not share one dtype: {sorted(dtypes)}")
    declared_dtype = str(text.get("ple_embedding_dtype", "")).lower()
    if declared_dtype in {"float8_e4m3fn", "fp8_e4m3fn"} and not dtypes <= {
        "F8_E4M3",
        "F8_E4M3FN",
    }:
        raise ProbeError(
            f"config declares {declared_dtype!r}, but PLE safetensors use {sorted(dtypes)}"
        )

    scale_name = f"{base}.weight_scale"
    if _rename(scale_name, include_vision=False) is not None:
        raise ProbeError(f"resident loader no longer skips PLE scale {scale_name!r}")
    if scale_name not in weight_map:
        raise ProbeError(f"checkpoint index is missing PLE scale {scale_name!r}")
    scale_file_name = weight_map[scale_name]
    scale_path = _checkpoint_path(root, scale_file_name, field=scale_name)
    scale_shape, scale_dtype = _tensor_metadata(scale_path, scale_name)
    scale_numel = 1
    for dim in scale_shape:
        scale_numel *= dim
    if scale_numel not in {1, split_parts}:
        raise ProbeError(
            f"PLE scale has {scale_numel} values; expected one or {split_parts}"
        )

    provisional = PLELayout(
        model_dir=root,
        config_sha256=_sha256_file(config_path),
        index_sha256=_sha256_file(index_path),
        checkpoint_layer_id=checkpoint_layer_id,
        ple_layer_index=0,
        shards=tuple(shards),
        scale_name=scale_name,
        scale_file_name=str(scale_file_name),
        scale_path=scale_path,
        scale_shape=scale_shape,
        scale_dtype=scale_dtype,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        ple_embed_dim=ple_embed_dim,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=ngram_vocab_size_base,
        make_vocab_size_divisible_by=divisible_by,
        seed=seed,
        expected_rows=0,
    )
    expected_rows = _build_hasher(provisional).layout.padded_vocab_size
    if provisional.row_count != expected_rows:
        raise ProbeError(
            "PLE shard rows do not match the hash layout: "
            f"checkpoint={provisional.row_count}, expected={expected_rows}"
        )
    return PLELayout(**{**provisional.__dict__, "expected_rows": expected_rows})


def _locate_global_row(layout: PLELayout, global_row: int) -> tuple[int, int]:
    if not 0 <= global_row < layout.row_count:
        raise ProbeError(f"sample row {global_row} is outside [0, {layout.row_count})")
    starts = [shard.global_start for shard in layout.shards]
    shard_index = bisect.bisect_right(starts, global_row) - 1
    shard = layout.shards[shard_index]
    return shard_index, global_row - shard.global_start


def _fixture_tokens(layout: PLELayout) -> tuple[int, ...]:
    candidates = (
        layout.eos_token_id,
        0,
        1,
        2,
        7,
        42,
        127,
        1024,
        layout.vocab_size // 2,
        max(0, layout.vocab_size - 2),
    )
    result: list[int] = []
    for value in candidates:
        value = min(max(0, int(value)), layout.vocab_size - 1)
        if value not in result:
            result.append(value)
    if len(result) < 4:
        raise ProbeError("vocabulary is too small to construct deterministic hash fixtures")
    return tuple(result)


def build_samples(layout: PLELayout) -> tuple[tuple[SamplePoint, ...], tuple[int, ...]]:
    """Cover both sides of every shard plus deterministic bigram/trigram hits."""

    samples: list[SamplePoint] = []
    for shard in layout.shards:
        samples.extend(
            (
                SamplePoint(
                    "boundary",
                    f"shard-{shard.index}-first",
                    shard.global_start,
                    shard.index,
                    0,
                ),
                SamplePoint(
                    "boundary",
                    f"shard-{shard.index}-last",
                    shard.global_end - 1,
                    shard.index,
                    shard.rows - 1,
                ),
            )
        )

    tokens = _fixture_tokens(layout)
    hasher = _build_hasher(layout)
    row_ids, _ = hasher(torch.tensor([tokens], dtype=torch.long))
    last = len(tokens) - 1
    middle = len(tokens) // 2
    first_trigram = layout.heads_per_ngram
    coordinates = (
        (0, 0),
        (0, first_trigram),
        (1, layout.heads_per_ngram - 1),
        (1, layout.num_heads - 1),
        (middle, 0),
        (middle, first_trigram),
        (last, layout.heads_per_ngram - 1),
        (last, layout.num_heads - 1),
    )
    for position, head in coordinates:
        global_row = int(row_ids[0, position, head])
        shard_index, shard_row = _locate_global_row(layout, global_row)
        samples.append(
            SamplePoint(
                "hash",
                f"hash-token-{position}-head-{head}",
                global_row,
                shard_index,
                shard_row,
                hash_token_position=position,
                hash_head=head,
            )
        )

    boundary_rows = {sample.global_row for sample in samples if sample.kind == "boundary"}
    expected_boundaries = {0, layout.row_count - 1}
    for shard in layout.shards[1:]:
        expected_boundaries.update((shard.global_start - 1, shard.global_start))
    if not expected_boundaries <= boundary_rows:
        raise ProbeError("internal sample construction did not cover every shard boundary")
    hash_heads = {sample.hash_head for sample in samples if sample.kind == "hash"}
    if not any(head is not None and head < first_trigram for head in hash_heads) or not any(
        head is not None and head >= first_trigram for head in hash_heads
    ):
        raise ProbeError("hash fixtures do not cover both bigram and trigram heads")
    return tuple(samples), tokens


def read_safetensors_ground_truth(
    layout: PLELayout, global_rows: Sequence[int]
) -> dict[int, torch.Tensor]:
    """Read only selected rows through safetensors' independent slice API."""

    requested = sorted(set(int(row) for row in global_rows))
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for global_row in requested:
        shard_index, local_row = _locate_global_row(layout, global_row)
        grouped[shard_index].append((global_row, local_row))
    try:
        with safe_open(str(layout.scale_path), framework="pt", device="cpu") as handle:
            scales = handle.get_tensor(layout.scale_name).reshape(-1).float().clone()
    except Exception as exc:
        raise ProbeError(f"cannot read the PLE scale through safetensors: {exc}") from exc

    result: dict[int, torch.Tensor] = {}
    for shard_index, rows in grouped.items():
        shard = layout.shards[shard_index]
        scale = scales[0 if scales.numel() == 1 else shard_index]
        try:
            with safe_open(str(shard.path), framework="pt", device="cpu") as handle:
                view = handle.get_slice(shard.tensor_name)
                for global_row, local_row in rows:
                    value = view[local_row : local_row + 1].reshape(-1).float().clone()
                    result[global_row] = value.mul(scale)
        except Exception as exc:
            raise ProbeError(
                f"cannot read ground-truth row from {shard.tensor_name!r}: {exc}"
            ) from exc
    if set(result) != set(requested):
        raise ProbeError("safetensors ground truth did not return every requested row")
    return result


def open_auxiliary_bank(
    layout: PLELayout, backend: str, device: torch.device
) -> ShardedSafetensorsMmapRowBank:
    """Construct the same ordered auxiliary-bank mapping used by model startup."""

    return ShardedSafetensorsMmapRowBank(
        [
            SafetensorsRowShard(shard.path, shard.tensor_name)
            for shard in layout.shards
        ],
        weight_scale_path=layout.scale_path,
        weight_scale_name=layout.scale_name,
        default_dtype=torch.float32,
        pin_memory=device.type == "cuda",
        io_backend=backend,
        cache_capacity_bytes=PROBE_CACHE_BYTES,
        queue_depth=PROBE_QUEUE_DEPTH,
        max_batch_pages=PROBE_MAX_BATCH_PAGES,
    )


def _is_wsl2() -> bool:
    values = [platform.release()]
    try:
        values.append(Path("/proc/version").read_text(encoding="utf-8"))
    except OSError:
        pass
    return sys.platform == "linux" and any("microsoft" in value.lower() for value in values)


def _require_release_runtime(backend: str, device: torch.device) -> dict[str, Any]:
    if backend != RELEASE_BACKEND:
        raise ProbeError(f"release probe requires backend={RELEASE_BACKEND!r}")
    if device.type != "cuda":
        raise ProbeError("release probe requires a CUDA device")
    if not _is_wsl2():
        raise ProbeError("release probe requires Linux under WSL2")
    if not torch.cuda.is_available():
        raise ProbeError("release probe requires torch CUDA availability")
    try:
        name = torch.cuda.get_device_name(device)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    except Exception as exc:
        raise ProbeError(f"cannot attest the CUDA device: {exc}") from exc
    if capability != (12, 0) or "5090" not in name or "rtx" not in name.lower():
        raise ProbeError(
            f"release probe requires RTX 5090/SM120, got {name!r} capability={capability}"
        )
    return {"gpu_name": name, "compute_capability": "12.0", "wsl2": True}


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ProbeError("sampled PLE row contains a non-finite value")
    payload = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": list(tensor.shape),
        "dtype": "float32",
        "numel": int(tensor.numel()),
    }


def _aggregate_io(telemetry: dict[str, Any]) -> dict[str, int]:
    children = telemetry.get("shards")
    if not isinstance(children, list):
        children = []
    keys = (
        "storage_bytes",
        "cache_hit_pages",
        "cache_miss_pages",
        "submission_batches",
        "submitted_sqes",
    )
    result = {
        key: sum(int(child.get(key, 0)) for child in children if isinstance(child, dict))
        for key in keys
    }
    result["gpu_decoded_rows"] = int(telemetry.get("gpu_decoded_rows", 0))
    result["mapped_bytes"] = int(telemetry.get("mapped_bytes", 0))
    result["payload_bytes_read"] = int(telemetry.get("payload_bytes_read", 0))
    return result


def run_probe(
    model_dir: str | os.PathLike[str],
    *,
    backend: str = RELEASE_BACKEND,
    device: str | torch.device = "cuda",
    release_mode: bool = True,
    bank_factory: BankFactory = open_auxiliary_bank,
    ground_truth_reader: GroundTruthReader = read_safetensors_ground_truth,
) -> dict[str, Any]:
    """Run sparse loader parity and return a JSON-serializable pass report."""

    target_device = torch.device(device)
    runtime = (
        _require_release_runtime(backend, target_device)
        if release_mode
        else {
            "gpu_name": None,
            "compute_capability": None,
            "wsl2": _is_wsl2(),
        }
    )
    layout = discover_layout(model_dir)
    if release_mode and layout.shards[0].dtype not in _FP8_DTYPES:
        raise ProbeError(
            f"release probe requires FP8 PLE shards, got {layout.shards[0].dtype}"
        )
    samples, fixture_tokens = build_samples(layout)
    rows = [sample.global_row for sample in samples]
    ground_truth = ground_truth_reader(layout, rows)

    bank = bank_factory(layout, backend, target_device)
    telemetry: dict[str, Any] = {}
    try:
        if int(bank.row_count) != layout.row_count or int(bank.row_width) != layout.row_width:
            raise ProbeError("auxiliary bank geometry does not match the indexed PLE layout")
        observed = bank.read_rows(
            torch.tensor(rows, dtype=torch.long),
            dtype=torch.float32,
            device=target_device,
        ).detach().to(device="cpu", dtype=torch.float32)
        telemetry_callback = getattr(bank, "telemetry", None)
        if not callable(telemetry_callback):
            raise ProbeError("auxiliary bank does not expose telemetry")
        telemetry = telemetry_callback()
    except ProbeError:
        raise
    except Exception as exc:
        raise ProbeError(f"auxiliary-bank sparse row read failed: {exc}") from exc
    finally:
        close = getattr(bank, "close", None)
        if callable(close):
            close()

    if observed.shape != (len(samples), layout.row_width):
        raise ProbeError(
            f"auxiliary bank returned shape {tuple(observed.shape)}, expected "
            f"{(len(samples), layout.row_width)}"
        )
    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for index, sample in enumerate(samples):
        shard = layout.shards[sample.shard_index]
        expected = ground_truth[sample.global_row]
        auxiliary_summary = _tensor_summary(observed[index])
        ground_summary = _tensor_summary(expected)
        match = bool(
            auxiliary_summary["sha256"] == ground_summary["sha256"]
            and torch.equal(observed[index], expected)
        )
        if not match:
            mismatches.append(sample.label)
        record: dict[str, Any] = {
            "kind": sample.kind,
            "label": sample.label,
            "tensor_name": shard.tensor_name,
            "tensor_shape": list(shard.shape),
            "global_row": sample.global_row,
            "shard_index": sample.shard_index,
            "shard_row": sample.shard_row,
            "storage_dtype": shard.dtype,
            "ground_truth": ground_summary,
            "auxiliary_bank": auxiliary_summary,
            "match": match,
        }
        if sample.kind == "hash":
            record["hash_token_position"] = sample.hash_token_position
            record["hash_head"] = sample.hash_head
        records.append(record)
    if mismatches:
        preview = ", ".join(mismatches[:8])
        raise ProbeError(
            f"PLE auxiliary-bank parity failed for {len(mismatches)} samples: {preview}"
        )

    io_summary = _aggregate_io(telemetry)
    unique_rows = len(set(rows))
    if release_mode:
        if telemetry.get("backend") != RELEASE_BACKEND:
            raise ProbeError("auxiliary bank telemetry does not attest the native backend")
        if io_summary["mapped_bytes"] != 0:
            raise ProbeError("release PLE probe unexpectedly used mmap payloads")
        if io_summary["storage_bytes"] <= 0 or io_summary["submitted_sqes"] <= 0:
            raise ProbeError("native PLE telemetry does not prove io_uring/O_DIRECT reads")
        if io_summary["gpu_decoded_rows"] < unique_rows:
            raise ProbeError(
                "native PLE telemetry does not prove GPU FP8 decoding of every sampled row"
            )

    token_bytes = b"".join(struct.pack("<q", token) for token in fixture_tokens)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "release_qualified": bool(release_mode),
        "checkpoint": {
            "model_dir_basename": layout.model_dir.name,
            "config_sha256": layout.config_sha256,
            "index_sha256": layout.index_sha256,
        },
        "runtime": {
            **runtime,
            "backend": backend,
            "device": str(target_device),
            "gpu_fp8_decode_attested": bool(
                release_mode and io_summary["gpu_decoded_rows"] >= unique_rows
            ),
        },
        "loader_mapping": {
            "checkpoint_layer_id": layout.checkpoint_layer_id,
            "normal_state_dict_action": "skip",
            "normal_state_dict_mapped_name": None,
            "auxiliary_bank": "ShardedSafetensorsMmapRowBank",
            "scale_tensor_name": layout.scale_name,
            "scale_shape": list(layout.scale_shape),
            "scale_dtype": layout.scale_dtype,
            "shards": [
                {
                    "index": shard.index,
                    "tensor_name": shard.tensor_name,
                    "file_name": shard.file_name,
                    "shape": list(shard.shape),
                    "dtype": shard.dtype,
                    "global_start": shard.global_start,
                    "global_end": shard.global_end,
                }
                for shard in layout.shards
            ],
        },
        "coverage": {
            "sample_count": len(samples),
            "unique_row_count": unique_rows,
            "shard_count": len(layout.shards),
            "all_shard_first_rows": True,
            "all_shard_last_rows": True,
            "global_first_row": True,
            "global_last_row": True,
            "hash_sample_count": sum(sample.kind == "hash" for sample in samples),
            "hash_fixture_token_count": len(fixture_tokens),
            "hash_fixture_tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "bigram_and_trigram_heads": True,
        },
        "io": io_summary,
        "records": records,
    }


def _write_report(report: dict[str, Any], destination: str) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination == "-":
        sys.stdout.write(payload)
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="fixed local checkpoint directory")
    parser.add_argument("--out", default="-", help="JSON destination, or '-' for stdout")
    parser.add_argument(
        "--backend",
        choices=("io_uring_odirect", "mmap", "direct_debug"),
        default=RELEASE_BACKEND,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--debug-nonrelease",
        action="store_true",
        help="allow CPU/mmap seams; output is never release-qualified",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_probe(
            args.model_dir,
            backend=args.backend,
            device=args.device,
            release_mode=not args.debug_nonrelease,
        )
        _write_report(report, args.out)
    except (ProbeError, OSError, ValueError, RuntimeError) as exc:
        message = str(exc).replace(str(args.model_dir), "<MODEL_DIR>")
        print(
            f"PLE checkpoint probe failed ({type(exc).__name__}): {message}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
