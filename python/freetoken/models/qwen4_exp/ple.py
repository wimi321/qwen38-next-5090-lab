"""Qwen4-Exp Per-Layer Embedding (PLE) primitives.

The production Qwen4-Exp table is too large to materialize as a regular
``nn.Embedding``.  This module therefore keeps three concerns separate:

* :class:`Qwen4ExpNGramHasher` implements the checkpoint-defined, EOS-aware
  64-bit bigram/trigram hashing;
* row banks expose only indexed rows, with a small in-memory implementation
  for tests and a read-only safetensors mmap implementation for serving; and
* :class:`Qwen4ExpPLELayer` implements the projections, gates and dilated
  depthwise convolution while accepting and returning explicit request state.

Nothing here owns scheduler/cache slots.  The engine integration can store the
returned :class:`PLEState` in whichever request pool it chooses.
"""

from __future__ import annotations

import functools
import itertools
import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.layers import BaseOP, LinearReplicated

from .hyperconnection import Qwen4ExpGroupedRMSNorm

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int,
) -> torch.Tensor:
    """Build the exact odd int64 hash multipliers used by Qwen4-Exp.

    Multiplication is intentionally performed later in ``torch.int64``: its
    two's-complement wraparound is part of the model's hash definition.
    """

    if unigram_vocab_size <= 0:
        raise ValueError(f"unigram_vocab_size must be positive, got {unigram_vocab_size}")
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")
    if ple_layer_index < 0:
        raise ValueError(f"ple_layer_index must be non-negative, got {ple_layer_index}")

    max_long = (1 << 63) - 1
    multiplier_max = max_long // unigram_vocab_size
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass(frozen=True)
class NGramHashLayout:
    """Concatenated-table layout for all independently hashed n-gram heads."""

    head_vocab_sizes: tuple[int, ...]
    head_offsets: tuple[int, ...]
    total_vocab_size: int
    padded_vocab_size: int


def build_ngram_hash_layout(
    ngram_vocab_size_base: int,
    num_heads: int,
    *,
    ple_layer_index: int = 0,
    make_vocab_size_divisible_by: int = 128,
) -> NGramHashLayout:
    """Derive the distinct prime table size and offset for each hash head."""

    if ngram_vocab_size_base < 2:
        raise ValueError(
            f"ngram_vocab_size_base must be at least 2, got {ngram_vocab_size_base}"
        )
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if ple_layer_index < 0:
        raise ValueError(f"ple_layer_index must be non-negative, got {ple_layer_index}")
    if make_vocab_size_divisible_by <= 0:
        raise ValueError(
            "make_vocab_size_divisible_by must be positive, got "
            f"{make_vocab_size_divisible_by}"
        )

    sizes: list[int] = []
    offsets: list[int] = []
    total = 0
    for head_idx in range(num_heads):
        global_head_idx = ple_layer_index * num_heads + head_idx
        size = _find_nth_prime_after(
            ngram_vocab_size_base - 1, global_head_idx + 1
        )
        sizes.append(size)
        offsets.append(total)
        total += size
    padded = math.ceil(total / make_vocab_size_divisible_by) * make_vocab_size_divisible_by
    return NGramHashLayout(tuple(sizes), tuple(offsets), total, padded)


class Qwen4ExpNGramHasher(nn.Module):
    """EOS-aware Qwen4-Exp n-gram row-id generator.

    Qwen4-Exp defaults to eight bigram and eight trigram heads (16 total), but
    both the maximum n-gram size and ``heads_per_ngram`` are configurable.
    ``forward`` returns global row ids into the concatenated table together
    with the last ``ngram_size - 1`` raw tokens for the next call.
    """

    def __init__(
        self,
        *,
        unigram_vocab_size: int,
        eos_token_id: int,
        ngram_vocab_size_base: int = 20_000_000,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ple_layer_index: int = 0,
        seed: int = 1234,
        make_vocab_size_divisible_by: int = 128,
    ) -> None:
        super().__init__()
        if ngram_size < 2:
            raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")
        if heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be positive, got {heads_per_ngram}")

        self.unigram_vocab_size = int(unigram_vocab_size)
        self.eos_token_id = int(eos_token_id)
        self.ngram_size = int(ngram_size)
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = int(heads_per_ngram)
        self.num_heads = self.context_len * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.seed = int(seed)
        self.layout = build_ngram_hash_layout(
            ngram_vocab_size_base,
            self.num_heads,
            ple_layer_index=self.ple_layer_index,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        )
        self.register_buffer(
            "layer_multipliers",
            build_layer_multipliers(
                self.unigram_vocab_size,
                self.ngram_size,
                self.ple_layer_index,
                self.seed,
            ),
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(self.layout.head_vocab_sizes, dtype=torch.long),
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(self.layout.head_offsets, dtype=torch.long),
        )

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward(
        self,
        input_ids: torch.Tensor,
        token_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids sequence length must be positive")
        input_ids = input_ids.long()
        batch_size = input_ids.shape[0]
        if token_history is None:
            previous_context = input_ids.new_full(
                (batch_size, self.context_len), self.eos_token_id
            )
        else:
            if token_history.shape != (batch_size, self.context_len):
                raise ValueError(
                    "token_history must have shape "
                    f"{(batch_size, self.context_len)}, got {tuple(token_history.shape)}"
                )
            previous_context = token_history.to(device=input_ids.device, dtype=torch.long)

        token_history_full = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [
            self._shift_right_ignore_eos(token_history_full, shift)
            for shift in range(self.ngram_size)
        ]
        multipliers = self.layer_multipliers.to(input_ids.device)
        vocab_sizes = self.ngram_heads_vocab_sizes.to(input_ids.device)
        offsets = self.ngram_heads_offsets.to(input_ids.device)

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            # torch.int64 overflow is the official two's-complement hash behavior.
            mixed_ids = shifted_tokens[0] * multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * multipliers[position],
                )
            head_sizes = vocab_sizes[start_idx:end_idx]
            head_offsets = offsets[start_idx:end_idx]
            local_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_sizes.view(1, 1, -1))
            blocks.append(local_ids + head_offsets.view(1, 1, -1))

        row_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
        next_history = token_history_full[:, -self.context_len :].clone()
        return row_ids, next_history


@runtime_checkable
class ReadonlyRowBank(Protocol):
    """Minimal random-row interface used by the n-gram embedding primitive."""

    @property
    def row_count(self) -> int: ...

    @property
    def row_width(self) -> int: ...

    def read_rows(
        self,
        indices: torch.Tensor | Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor: ...


class TensorRowBank:
    """Small in-memory row bank, useful for tiny models and operator tests."""

    def __init__(self, weight: torch.Tensor) -> None:
        if weight.ndim != 2:
            raise ValueError(f"weight must have shape [rows, width], got {tuple(weight.shape)}")
        self.weight = weight

    @property
    def row_count(self) -> int:
        return self.weight.shape[0]

    @property
    def row_width(self) -> int:
        return self.weight.shape[1]

    def read_rows(
        self,
        indices: torch.Tensor | Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        indices_tensor = torch.as_tensor(indices, dtype=torch.long)
        original_shape = tuple(indices_tensor.shape)
        flat = indices_tensor.to(self.weight.device).reshape(-1)
        if flat.numel() and (flat.min().item() < 0 or flat.max().item() >= self.row_count):
            raise IndexError(f"row index outside [0, {self.row_count})")
        rows = self.weight.index_select(0, flat).reshape(*original_shape, self.row_width)
        if dtype is not None or device is not None:
            rows = rows.to(dtype=dtype or rows.dtype, device=device or rows.device)
        return rows


@dataclass(frozen=True)
class _SafeTensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int
    item_size: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


_SAFE_DTYPE_INFO: dict[str, tuple[torch.dtype | None, int]] = {
    "BOOL": (torch.bool, 1),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "I16": (torch.int16, 2),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    # FP8 values are decoded through a bit-exact lookup table, so no runtime
    # dependency on CPU float8 kernels is required.
    "F8_E4M3": (None, 1),
    "F8_E4M3FN": (None, 1),
    "F8_E5M2": (None, 1),
}

_SAFE_NUMPY_DTYPES = {
    "BOOL": "?",
    "U8": "u1",
    "I8": "i1",
    "I16": "<i2",
    "I32": "<i4",
    "I64": "<i8",
    "F16": "<f2",
    "F32": "<f4",
    "F64": "<f8",
    "F8_E4M3": "u1",
    "F8_E4M3FN": "u1",
    "F8_E5M2": "u1",
    # NumPy has no stable bfloat16 dtype; BF16 uses the byte-copy fallback.
}


@functools.lru_cache(maxsize=3)
def _fp8_decode_lut(dtype: str) -> torch.Tensor:
    canonical = "F8_E4M3" if dtype == "F8_E4M3FN" else dtype
    if canonical not in {"F8_E4M3", "F8_E5M2"}:
        raise ValueError(f"unsupported FP8 dtype {dtype!r}")
    exponent_bits, mantissa_bits, bias = (
        (4, 3, 7) if canonical == "F8_E4M3" else (5, 2, 15)
    )
    exponent_mask = (1 << exponent_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    values: list[float] = []
    for raw in range(256):
        sign = -1.0 if raw & 0x80 else 1.0
        exponent = (raw >> mantissa_bits) & exponent_mask
        mantissa = raw & mantissa_mask
        if exponent == 0:
            value = math.ldexp(mantissa / (1 << mantissa_bits), 1 - bias)
        elif canonical == "F8_E4M3" and exponent == exponent_mask:
            # E4M3FN uses the top exponent for finite values except the two
            # signed mantissa-all-ones encodings, which are NaN.
            value = (
                math.nan
                if mantissa == mantissa_mask
                else math.ldexp(1.0 + mantissa / (1 << mantissa_bits), exponent - bias)
            )
        elif canonical == "F8_E5M2" and exponent == exponent_mask:
            value = math.inf if mantissa == 0 else math.nan
        else:
            value = math.ldexp(
                1.0 + mantissa / (1 << mantissa_bits), exponent - bias
            )
        values.append(sign * value)
    return torch.tensor(values, dtype=torch.float32)


class SafetensorsMmapRowBank:
    """Read selected rows from one safetensors tensor through a read-only mmap.

    Only the requested unique rows are copied out of the mapping.  FP8 weights
    are decoded in FP32 and optionally multiplied by a scalar, per-row,
    per-block or per-element scale tensor before conversion to ``dtype``.

    The scale tensor may be scalar, ``[rows]``, or ``[rows, scale_width]``.
    ``scale_width`` must be 1, equal to the weight width, or divide it evenly;
    the last case applies one scale to each contiguous block.
    """

    _MAX_HEADER_BYTES = 256 << 20

    def __init__(
        self,
        path: str | os.PathLike[str],
        tensor_name: str,
        *,
        scale_name: str | None = None,
        default_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.path = str(Path(path))
        self.tensor_name = tensor_name
        self.scale_name = scale_name
        self.default_dtype = default_dtype
        self._payload_bytes_read = 0
        self._file = open(self.path, "rb")
        try:
            self._mapping = mmap.mmap(self._file.fileno(), length=0, access=mmap.ACCESS_READ)
            # PLE lookups are sparse hashes over a ~51 GB table.  On Linux,
            # advertise that access pattern so the kernel does not spend the
            # no-swap WSL memory budget on sequential readahead around every
            # 160-byte row fault.  Windows lacks mmap.madvise; correctness is
            # unchanged there and in older Python builds.
            madvise = getattr(self._mapping, "madvise", None)
            madv_random = getattr(mmap, "MADV_RANDOM", None)
            if madvise is not None and madv_random is not None:
                madvise(madv_random)
            self._infos = self._parse_header()
            self._weight_info = self._require_tensor(tensor_name)
            if len(self._weight_info.shape) != 2:
                raise ValueError(
                    f"{tensor_name!r} must have shape [rows, width], "
                    f"got {self._weight_info.shape}"
                )
            self._scale_info = self._require_tensor(scale_name) if scale_name else None
            self._validate_scale_shape()
        except Exception:
            mapping = getattr(self, "_mapping", None)
            if mapping is not None:
                mapping.close()
            self._file.close()
            raise

    @property
    def readonly(self) -> bool:
        return True

    @property
    def row_count(self) -> int:
        return self._weight_info.shape[0]

    @property
    def row_width(self) -> int:
        return self._weight_info.shape[1]

    @property
    def mapped_bytes(self) -> int:
        """Virtual mapping size (not resident or materialized tensor bytes)."""

        return self._mapping.size()

    @property
    def payload_bytes_read(self) -> int:
        """Cumulative payload bytes copied for requested weight/scale rows."""

        return self._payload_bytes_read

    @property
    def closed(self) -> bool:
        return self._mapping.closed

    def _parse_header(self) -> dict[str, _SafeTensorInfo]:
        file_size = self._mapping.size()
        if file_size < 8:
            raise ValueError(f"{self.path!r} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", self._mapping[:8])[0]
        if header_len <= 0 or header_len > self._MAX_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header length {header_len}")
        data_start = 8 + header_len
        if data_start > file_size:
            raise ValueError("safetensors header extends beyond the file")
        try:
            header = json.loads(self._mapping[8:data_start].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid safetensors JSON header: {exc}") from exc

        infos: dict[str, _SafeTensorInfo] = {}
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            try:
                dtype = str(metadata["dtype"])
                shape = tuple(int(dim) for dim in metadata["shape"])
                relative_start, relative_end = (
                    int(offset) for offset in metadata["data_offsets"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid metadata for tensor {name!r}") from exc
            if dtype not in _SAFE_DTYPE_INFO:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} for {name!r}")
            if any(dim < 0 for dim in shape):
                raise ValueError(f"negative dimension in tensor {name!r}: {shape}")
            if relative_start < 0 or relative_end < relative_start:
                raise ValueError(f"invalid data offsets for tensor {name!r}")
            item_size = _SAFE_DTYPE_INFO[dtype][1]
            expected = math.prod(shape) * item_size
            if relative_end - relative_start != expected:
                raise ValueError(
                    f"tensor {name!r} payload has {relative_end - relative_start} bytes, "
                    f"expected {expected}"
                )
            start, end = data_start + relative_start, data_start + relative_end
            if end > file_size:
                raise ValueError(f"tensor {name!r} extends beyond the file")
            infos[name] = _SafeTensorInfo(dtype, shape, start, end, item_size)
        return infos

    def _require_tensor(self, name: str | None) -> _SafeTensorInfo:
        if name is None or name not in self._infos:
            raise KeyError(f"tensor {name!r} is not present in {self.path!r}")
        return self._infos[name]

    def _validate_scale_shape(self) -> None:
        info = self._scale_info
        if info is None:
            return
        if info.numel == 1:
            return
        if not info.shape or info.shape[0] != self.row_count:
            raise ValueError(
                f"scale tensor {self.scale_name!r} must be scalar or row-aligned; "
                f"got {info.shape} for {self.row_count} rows"
            )
        scale_width = math.prod(info.shape[1:]) if len(info.shape) > 1 else 1
        if scale_width <= 0 or self.row_width % scale_width != 0:
            raise ValueError(
                f"scale width {scale_width} must divide weight width {self.row_width}"
            )

    @staticmethod
    def _contiguous_runs(sorted_rows: list[int]):
        if not sorted_rows:
            return
        first_position = 0
        first_row = sorted_rows[0]
        previous = first_row
        for position, row in enumerate(sorted_rows[1:], 1):
            if row != previous + 1:
                yield first_position, position, first_row, previous + 1
                first_position, first_row = position, row
            previous = row
        yield first_position, len(sorted_rows), first_row, previous + 1

    def _decode_bytes(self, payload: bytearray, dtype: str) -> torch.Tensor:
        if dtype in {"F8_E4M3", "F8_E4M3FN", "F8_E5M2"}:
            raw = torch.frombuffer(payload, dtype=torch.uint8)
            return _fp8_decode_lut(dtype).index_select(0, raw.long())
        torch_dtype = _SAFE_DTYPE_INFO[dtype][0]
        assert torch_dtype is not None
        return torch.frombuffer(payload, dtype=torch_dtype).to(torch.float32)

    def _read_unique_rows(
        self,
        info: _SafeTensorInfo,
        sorted_rows: list[int],
        row_width: int,
    ) -> torch.Tensor:
        numpy_dtype = _SAFE_NUMPY_DTYPES.get(info.dtype)
        if numpy_dtype is not None and sorted_rows:
            # np.take performs one C-level gather directly from the read-only
            # mapping.  It allocates only [requested_rows, row_width], unlike
            # safe_open().get_tensor(), and avoids one Python mmap slice per
            # random n-gram row during an 8K prefill.
            import numpy as np

            mapped = np.ndarray(
                (info.shape[0], row_width),
                dtype=np.dtype(numpy_dtype),
                buffer=self._mapping,
                offset=info.start,
            )
            selected = np.take(
                mapped,
                np.asarray(sorted_rows, dtype=np.int64),
                axis=0,
            )
            self._payload_bytes_read += len(sorted_rows) * row_width * info.item_size
            selected_tensor = torch.from_numpy(selected)
            if info.dtype in {"F8_E4M3", "F8_E4M3FN", "F8_E5M2"}:
                return _fp8_decode_lut(info.dtype).index_select(
                    0, selected_tensor.reshape(-1).long()
                ).reshape(len(sorted_rows), row_width)
            return selected_tensor.to(torch.float32)

        output = torch.empty((len(sorted_rows), row_width), dtype=torch.float32)
        row_bytes = row_width * info.item_size
        for out_start, out_end, row_start, row_end in self._contiguous_runs(sorted_rows):
            byte_start = info.start + row_start * row_bytes
            byte_end = info.start + row_end * row_bytes
            payload = bytearray(self._mapping[byte_start:byte_end])
            self._payload_bytes_read += len(payload)
            decoded = self._decode_bytes(payload, info.dtype).reshape(row_end - row_start, row_width)
            output[out_start:out_end].copy_(decoded)
        return output

    def _read_scale_rows(self, sorted_rows: list[int]) -> torch.Tensor | None:
        info = self._scale_info
        if info is None:
            return None
        if info.numel == 1:
            payload = bytearray(self._mapping[info.start:info.end])
            self._payload_bytes_read += len(payload)
            return self._decode_bytes(payload, info.dtype).reshape(1, 1)
        scale_width = math.prod(info.shape[1:]) if len(info.shape) > 1 else 1
        return self._read_unique_rows(info, sorted_rows, scale_width)

    def read_rows(
        self,
        indices: torch.Tensor | Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("cannot read from a closed safetensors row bank")
        target_dtype = dtype or self.default_dtype
        if target_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise TypeError(f"row bank target dtype must be floating point, got {target_dtype}")

        indices_tensor = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        original_shape = tuple(indices_tensor.shape)
        flat = indices_tensor.reshape(-1)
        if flat.numel() == 0:
            return torch.empty(
                (*original_shape, self.row_width),
                dtype=target_dtype,
                device=device or "cpu",
            )
        minimum, maximum = flat.min().item(), flat.max().item()
        if minimum < 0 or maximum >= self.row_count:
            raise IndexError(
                f"row index range [{minimum}, {maximum}] is outside [0, {self.row_count})"
            )

        unique_rows, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        sorted_rows = unique_rows.tolist()
        values = self._read_unique_rows(self._weight_info, sorted_rows, self.row_width)
        scales = self._read_scale_rows(sorted_rows)
        if scales is not None:
            scale_width = scales.shape[1]
            if scale_width == 1:
                values.mul_(scales)
            elif scale_width == self.row_width:
                values.mul_(scales)
            else:
                values.mul_(scales.repeat_interleave(self.row_width // scale_width, dim=1))
        values = values.index_select(0, inverse).reshape(*original_shape, self.row_width)
        return values.to(dtype=target_dtype, device=device or "cpu")

    def close(self) -> None:
        if not self._mapping.closed:
            self._mapping.close()
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "SafetensorsMmapRowBank":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        mapping = getattr(self, "_mapping", None)
        file = getattr(self, "_file", None)
        if mapping is not None and not mapping.closed:
            mapping.close()
        if file is not None and not file.closed:
            file.close()


class ConcatenatedRowBank:
    """Present row-sharded banks as one logical table without concatenating data.

    Qwen4-Exp checkpoints split the PLE table along its row dimension.  A future
    checkpoint loader can open one :class:`SafetensorsMmapRowBank` per part and
    bind this inexpensive router to the embedding primitive.
    """

    def __init__(
        self,
        banks: Sequence[ReadonlyRowBank],
        *,
        default_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not banks:
            raise ValueError("ConcatenatedRowBank requires at least one bank")
        widths = {bank.row_width for bank in banks}
        if len(widths) != 1:
            raise ValueError(f"all row-bank widths must match, got {sorted(widths)}")
        self.banks = tuple(banks)
        self.default_dtype = default_dtype
        self._row_offsets = tuple(
            itertools.accumulate((bank.row_count for bank in self.banks), initial=0)
        )
        self._closed = False

    @property
    def row_count(self) -> int:
        return self._row_offsets[-1]

    @property
    def row_width(self) -> int:
        return self.banks[0].row_width

    @property
    def payload_bytes_read(self) -> int:
        return sum(int(getattr(bank, "payload_bytes_read", 0)) for bank in self.banks)

    @property
    def closed(self) -> bool:
        return self._closed

    def read_rows(
        self,
        indices: torch.Tensor | Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("cannot read from a closed concatenated row bank")
        target_dtype = dtype or self.default_dtype
        target_device = torch.device(device or "cpu")
        indices_tensor = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        original_shape = tuple(indices_tensor.shape)
        flat = indices_tensor.reshape(-1)
        if flat.numel() == 0:
            return torch.empty(
                (*original_shape, self.row_width),
                dtype=target_dtype,
                device=target_device,
            )
        minimum, maximum = flat.min().item(), flat.max().item()
        if minimum < 0 or maximum >= self.row_count:
            raise IndexError(
                f"row index range [{minimum}, {maximum}] is outside [0, {self.row_count})"
            )

        output = torch.empty(
            (flat.numel(), self.row_width),
            dtype=target_dtype,
            device=target_device,
        )
        for bank_idx, bank in enumerate(self.banks):
            start, end = self._row_offsets[bank_idx : bank_idx + 2]
            positions = torch.nonzero((flat >= start) & (flat < end), as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            local_rows = flat.index_select(0, positions) - start
            values = bank.read_rows(
                local_rows,
                dtype=target_dtype,
                device=target_device,
            ).reshape(-1, self.row_width)
            output.index_copy_(0, positions.to(target_device), values)
        return output.reshape(*original_shape, self.row_width)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for bank in reversed(self.banks):
            close = getattr(bank, "close", None)
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "ConcatenatedRowBank":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class SafetensorsRowShard:
    """Location of one dim-0 PLE table part in a safetensors file."""

    path: str | os.PathLike[str]
    tensor_name: str
    scale_name: str | None = None


def _load_small_safetensors_tensor(
    path: str | os.PathLike[str], tensor_name: str
) -> torch.Tensor:
    """Materialize a tiny metadata/scale tensor, never a PLE weight shard."""

    import safetensors

    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name).float()


class ShardedSafetensorsMmapRowBank:
    """One logical table backed by row-sharded safetensors FP8 tensors.

    ``shards`` are ordered along dimension zero.  The released checkpoint uses
    128 ``shard_i.weight`` tensors (roughly 2.5M rows by 160 columns each), but
    no fixed count or part size is assumed.  Global ids are uniqued before
    routing, so repeated n-gram hits copy and decode a row only once per call.

    ``weight_scale`` may be one scalar or one scalar per shard.  Alternatively
    ``weight_scale_path``/``weight_scale_name`` may point at the checkpoint's
    separate tiny scale tensor.  Per-row or per-block scales can still be bound
    through each :class:`SafetensorsRowShard.scale_name`.

    When ``pin_memory`` is true, the final requested-row staging tensor is
    pinned before an optional non-blocking device transfer.  The mmap payload
    itself always remains file-backed and read-only.
    """

    def __init__(
        self,
        shards: Sequence[
            SafetensorsRowShard | tuple[str | os.PathLike[str], str]
        ],
        *,
        weight_scale: float | Sequence[float] | torch.Tensor | None = None,
        weight_scale_path: str | os.PathLike[str] | None = None,
        weight_scale_name: str | None = None,
        default_dtype: torch.dtype = torch.bfloat16,
        pin_memory: bool = False,
    ) -> None:
        if not shards:
            raise ValueError("ShardedSafetensorsMmapRowBank requires at least one shard")
        if (weight_scale_path is None) != (weight_scale_name is None):
            raise ValueError(
                "weight_scale_path and weight_scale_name must be provided together"
            )
        if weight_scale is not None and weight_scale_path is not None:
            raise ValueError(
                "pass either weight_scale or weight_scale_path/weight_scale_name, not both"
            )

        specs = tuple(
            shard
            if isinstance(shard, SafetensorsRowShard)
            else SafetensorsRowShard(shard[0], shard[1])
            for shard in shards
        )
        opened: list[SafetensorsMmapRowBank] = []
        try:
            for spec in specs:
                opened.append(
                    SafetensorsMmapRowBank(
                        spec.path,
                        spec.tensor_name,
                        scale_name=spec.scale_name,
                        default_dtype=torch.float32,
                    )
                )
            widths = {bank.row_width for bank in opened}
            if len(widths) != 1:
                raise ValueError(
                    f"all PLE shard widths must match, got {sorted(widths)}"
                )
        except Exception:
            for bank in reversed(opened):
                bank.close()
            raise

        self.shards = specs
        self.banks = tuple(opened)
        self.default_dtype = default_dtype
        self.pin_memory = pin_memory
        self._row_offsets = tuple(
            itertools.accumulate((bank.row_count for bank in self.banks), initial=0)
        )
        self._closed = False
        try:
            if weight_scale_path is not None:
                assert weight_scale_name is not None
                scales = _load_small_safetensors_tensor(
                    weight_scale_path, weight_scale_name
                ).reshape(-1)
            elif weight_scale is None:
                scales = torch.ones(1, dtype=torch.float32)
            else:
                scales = torch.as_tensor(weight_scale, dtype=torch.float32).reshape(-1)
            if scales.numel() not in {1, len(self.banks)}:
                raise ValueError(
                    "weight_scale must contain one value or one per shard, got "
                    f"{scales.numel()} values for {len(self.banks)} shards"
                )
        except Exception:
            self.close()
            raise
        self._weight_scales = scales.clone()

    @property
    def row_count(self) -> int:
        return self._row_offsets[-1]

    @property
    def row_width(self) -> int:
        return self.banks[0].row_width

    @property
    def payload_bytes_read(self) -> int:
        return sum(bank.payload_bytes_read for bank in self.banks)

    @property
    def mapped_bytes(self) -> int:
        return sum(bank.mapped_bytes for bank in self.banks)

    @property
    def closed(self) -> bool:
        return self._closed

    def read_rows(
        self,
        indices: torch.Tensor | Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("cannot read from a closed sharded row bank")
        target_dtype = dtype or self.default_dtype
        if target_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise TypeError(f"row bank target dtype must be floating point, got {target_dtype}")
        target_device = torch.device(device or "cpu")
        indices_tensor = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        original_shape = tuple(indices_tensor.shape)
        flat = indices_tensor.reshape(-1)
        if flat.numel() == 0:
            return torch.empty(
                (*original_shape, self.row_width),
                dtype=target_dtype,
                device=target_device,
            )
        minimum, maximum = flat.min().item(), flat.max().item()
        if minimum < 0 or maximum >= self.row_count:
            raise IndexError(
                f"row index range [{minimum}, {maximum}] is outside [0, {self.row_count})"
            )

        unique_rows, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        unique_values = torch.empty(
            (unique_rows.numel(), self.row_width), dtype=torch.float32
        )
        for shard_idx, bank in enumerate(self.banks):
            start, end = self._row_offsets[shard_idx : shard_idx + 2]
            positions = torch.nonzero(
                (unique_rows >= start) & (unique_rows < end), as_tuple=False
            ).flatten()
            if positions.numel() == 0:
                continue
            local_rows = unique_rows.index_select(0, positions) - start
            values = bank.read_rows(local_rows, dtype=torch.float32, device="cpu")
            scale = self._weight_scales[
                0 if self._weight_scales.numel() == 1 else shard_idx
            ]
            values.mul_(scale)
            unique_values.index_copy_(0, positions, values)

        staging = unique_values.index_select(0, inverse).reshape(
            *original_shape, self.row_width
        )
        staging = staging.to(dtype=target_dtype)
        if self.pin_memory and staging.numel():
            staging = staging.pin_memory()
        if target_device.type != "cpu":
            staging = staging.to(
                device=target_device,
                non_blocking=self.pin_memory,
            )
        return staging

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        first_error: BaseException | None = None
        for bank in reversed(getattr(self, "banks", ())):
            try:
                bank.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "ShardedSafetensorsMmapRowBank":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hash input ids, fetch the selected rows, and concatenate all heads."""

    def __init__(self, hasher: Qwen4ExpNGramHasher, row_bank: ReadonlyRowBank) -> None:
        super().__init__()
        self.hasher = hasher
        self.embedding_dim = hasher.num_heads * row_bank.row_width
        self.row_bank = row_bank
        self._validate_row_bank(row_bank)

    def _validate_row_bank(self, row_bank: ReadonlyRowBank) -> None:
        if row_bank.row_count < self.hasher.layout.padded_vocab_size:
            raise ValueError(
                f"row bank has {row_bank.row_count} rows, but hash layout needs "
                f"{self.hasher.layout.padded_vocab_size}"
            )
        expected_width = self.embedding_dim // self.hasher.num_heads
        if row_bank.row_width != expected_width:
            raise ValueError(
                f"row bank width {row_bank.row_width} does not match the configured "
                f"head width {expected_width}"
            )

    def bind_row_bank(self, row_bank: ReadonlyRowBank) -> None:
        """Validate and bind a deferred production mmap bank."""

        self._validate_row_bank(row_bank)
        self.row_bank = row_bank

    def forward(
        self,
        input_ids: torch.Tensor,
        token_history: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_ids, next_history = self.hasher(input_ids, token_history)
        output_device = device or input_ids.device
        rows = self.row_bank.read_rows(row_ids, dtype=dtype, device=output_device)
        embeddings = rows.flatten(-2)
        return embeddings, next_history


@dataclass(frozen=True)
class PLEState:
    """Request-local PLE state, intentionally independent of engine cache types."""

    token_history: torch.Tensor | None = None
    conv_state: torch.Tensor | None = None


class _DepthwiseDilatedConv1d(BaseOP):
    """Explicit-weight conv wrapper preserving the HF ``conv1d.weight`` key."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = torch.empty(channels, 1, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            hidden_states.to(self.weight.dtype),
            self.weight,
            bias=None,
            dilation=self.dilation,
            groups=self.channels,
        )


class Qwen4ExpPLELayer(BaseOP):
    """Reference-correct Qwen4-Exp PLE projection, gate and short convolution."""

    def __init__(
        self,
        embedding: Qwen4ExpNGramEmbedding,
        *,
        hidden_size: int,
        hc_count: int = 4,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if hc_count <= 1:
            raise ValueError(f"hc_count must be greater than one, got {hc_count}")
        if conv_kernel_size <= 0:
            raise ValueError(
                f"conv_kernel_size must be positive, got {conv_kernel_size}"
            )
        self.ple_embedding = embedding
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_hidden_size = hidden_size * hc_count
        self.conv_dilation = embedding.hasher.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * self.conv_dilation

        self.key_proj = LinearReplicated(
            embedding.embedding_dim, self.hc_hidden_size, has_bias=False
        )
        self.value_proj = LinearReplicated(
            embedding.embedding_dim, hidden_size, has_bias=False
        )
        self.norm_key = Qwen4ExpGroupedRMSNorm(
            self.hc_hidden_size, group_size=hidden_size, eps=rms_norm_eps
        )
        self.norm_query = Qwen4ExpGroupedRMSNorm(
            self.hc_hidden_size, group_size=hidden_size, eps=rms_norm_eps
        )
        self.norm_conv = Qwen4ExpGroupedRMSNorm(
            self.hc_hidden_size, group_size=hidden_size, eps=rms_norm_eps
        )
        self.conv1d = _DepthwiseDilatedConv1d(
            self.hc_hidden_size, conv_kernel_size, self.conv_dilation
        )

    def _short_conv(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = hidden_states.shape
        states = hidden_states.transpose(1, 2)
        expected_shape = (batch_size, channels, self.short_conv_state_len)
        if conv_state is None:
            previous = states.new_zeros(expected_shape)
        else:
            if conv_state.shape != expected_shape:
                raise ValueError(
                    f"conv_state must have shape {expected_shape}, got {tuple(conv_state.shape)}"
                )
            previous = conv_state.to(device=states.device, dtype=states.dtype)
        combined = torch.cat([previous, states], dim=-1)
        if self.short_conv_state_len:
            next_state = combined[..., -self.short_conv_state_len :].clone()
        else:
            next_state = combined[..., :0].clone()
        output = F.silu(self.conv1d.forward(combined))
        if output.shape[-1] != seq_len:
            raise RuntimeError(
                f"PLE convolution returned {output.shape[-1]} tokens for input length {seq_len}"
            )
        return output.transpose(1, 2).to(hidden_states.dtype), next_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        state: PLEState | None = None,
        *,
        conv_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PLEState]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hc_hidden_size:
            raise ValueError(
                "hidden_states must have shape [batch, seq, "
                f"{self.hc_hidden_size}], got {tuple(hidden_states.shape)}"
            )
        if input_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} does not match "
                f"hidden states {tuple(hidden_states.shape[:2])}"
            )
        if state is None:
            state = PLEState()
        embeddings, next_history = self.ple_embedding(
            input_ids,
            state.token_history,
            dtype=self.key_proj.weight.dtype,
            device=hidden_states.device,
        )
        output, next_conv_state = self._forward_embeddings(
            hidden_states,
            embeddings,
            state.conv_state,
            conv_mask=conv_mask,
        )
        return output, PLEState(next_history, next_conv_state)

    def _forward_embeddings(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        conv_state: torch.Tensor | None,
        *,
        conv_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU-only PLE math after graph-external row staging."""

        if embeddings.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"PLE embeddings shape {tuple(embeddings.shape[:2])} does not match "
                f"hidden states {tuple(hidden_states.shape[:2])}"
            )
        if embeddings.shape[-1] != self.ple_embedding.embedding_dim:
            raise ValueError(
                "PLE embeddings must have width "
                f"{self.ple_embedding.embedding_dim}, got {embeddings.shape[-1]}"
            )
        key_normed = self.norm_key.forward(self.key_proj.forward(embeddings)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        value = self.value_proj.forward(embeddings)
        query_normed = self.norm_query.forward(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(
            self.hidden_size
        )
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_flat = gated_value.flatten(-2)
        gated_value_normed = self.norm_conv.forward(gated_value_flat)

        if conv_mask is not None:
            if conv_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"conv_mask must have shape {tuple(hidden_states.shape[:2])}, "
                    f"got {tuple(conv_mask.shape)}"
                )
            mask = conv_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
            gated_value_flat = gated_value_flat * mask.unsqueeze(-1)
            gated_value_normed = gated_value_normed * mask.unsqueeze(-1)

        conv_output, next_conv_state = self._short_conv(
            gated_value_normed, conv_state
        )
        output = gated_value_flat + conv_output
        return output, next_conv_state

    def stage_decode_batch(
        self,
        batch,
        pool,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Hash and mmap-fetch one decode row per padded request outside graphs.

        The resulting GPU tensors are ordinary batch inputs.  GraphRunner copies
        them into persistent capture buffers before replay; token history stays
        pending on CPU until ``commit_batch`` is called after a successful
        forward.
        """

        if not batch.is_decode:
            return
        reqs = batch.padded_reqs
        if not reqs:
            raise ValueError("cannot stage an empty PLE decode batch")

        row_ids: list[torch.Tensor] = []
        next_histories: list[torch.Tensor] = []
        reset_flags: list[bool] = []
        slots: list[int] = []
        for req in reqs:
            length = int(req.extend_len)
            if length != 1:
                raise ValueError(
                    "PLE decode staging requires exactly one token per padded "
                    f"request, got extend_len={length} for slot {req.table_idx}"
                )
            token_ids = req.input_ids[req.cached_len : req.device_len]
            if token_ids.numel() != 1:
                raise ValueError(
                    f"request {req.table_idx} exposes {token_ids.numel()} decode token ids"
                )
            history, reset = pool.stage_token_history(
                req.table_idx, fresh=int(req.cached_len) == 0
            )
            ids, next_history = self.ple_embedding.hasher(
                token_ids.reshape(1, 1), history.reshape(1, -1)
            )
            row_ids.append(ids)
            next_histories.append(next_history)
            reset_flags.append(reset)
            slots.append(int(req.table_idx))

        # Read all hashed heads in one bank call.  The sharded mmap bank
        # deduplicates global row ids across the whole padded batch and performs
        # a single pinned H2D staging transfer.
        all_row_ids = torch.cat(row_ids, dim=0)
        rows = self.ple_embedding.row_bank.read_rows(
            all_row_ids, dtype=dtype, device=device
        )
        embeddings = rows.flatten(-2).squeeze(1)
        pin = device.type == "cuda" and torch.cuda.is_available()
        table_idx_cpu = torch.tensor(slots, dtype=torch.long, pin_memory=pin)
        reset_cpu = torch.tensor(reset_flags, dtype=torch.bool, pin_memory=pin)

        batch.ple_embeddings = embeddings
        batch.ple_table_idx = table_idx_cpu.to(device, non_blocking=pin)
        batch.ple_reset_mask = reset_cpu.to(device, non_blocking=pin)
        batch.ple_next_token_history = torch.cat(next_histories[: batch.size], dim=0)
        batch.ple_commit_slots = tuple(slots[: batch.size])

    @staticmethod
    def commit_batch(batch, pool) -> None:
        """Publish staged prefill/decode state after complete model success."""

        histories = getattr(batch, "ple_next_token_history", None)
        slots = getattr(batch, "ple_commit_slots", ())
        if histories is None:
            return
        if histories.shape[0] != len(slots):
            raise RuntimeError(
                "PLE staged history/slot count mismatch: "
                f"{histories.shape[0]} histories for {len(slots)} slots"
            )
        table_idx = getattr(batch, "ple_table_idx", None)
        if table_idx is None or table_idx.ndim != 1 or table_idx.numel() < len(slots):
            shape = None if table_idx is None else tuple(table_idx.shape)
            raise RuntimeError(
                "PLE staged table_idx cannot commit the active requests: "
                f"shape={shape}, active={len(slots)}"
            )

        # The pending convolution rows were produced inside model.forward;
        # publishing both halves happens only here, after Engine.forward_batch
        # has received logits successfully.
        pool.commit_pending_decode(
            table_idx[: len(slots)], tuple(slots), histories
        )
        batch.ple_next_token_history = None
        batch.ple_commit_slots = ()

    # Compatibility seam for callers/tests written before prefill joined the
    # same transaction.  New code should use the mode-neutral name above.
    commit_decode_batch = commit_batch

    def forward_staged_decode(self, hidden_states: torch.Tensor, batch, pool) -> torch.Tensor:
        """Run only capturable tensor operations for a staged decode batch."""

        embeddings = getattr(batch, "ple_embeddings", None)
        table_idx = getattr(batch, "ple_table_idx", None)
        reset_mask = getattr(batch, "ple_reset_mask", None)
        if embeddings is None or table_idx is None or reset_mask is None:
            raise RuntimeError("PLE decode batch was not staged before model.forward")
        batch_size = table_idx.numel()
        if hidden_states.shape != (batch_size, self.hc_hidden_size):
            raise ValueError(
                "staged PLE decode hidden states must have shape "
                f"{(batch_size, self.hc_hidden_size)}, got {tuple(hidden_states.shape)}"
            )
        if embeddings.shape != (batch_size, self.ple_embedding.embedding_dim):
            raise ValueError(
                "staged PLE embeddings must have shape "
                f"{(batch_size, self.ple_embedding.embedding_dim)}, got "
                f"{tuple(embeddings.shape)}"
            )
        previous = pool.gather_decode_conv_states(table_idx, reset_mask)
        output, next_conv_state = self._forward_embeddings(
            hidden_states.unsqueeze(1),
            embeddings.unsqueeze(1),
            previous,
        )
        pool.write_pending_decode_conv_states(table_idx, next_conv_state)
        return output.squeeze(1)

    def forward_flat(self, hidden_states: torch.Tensor, batch, pool) -> torch.Tensor:
        """Apply PLE to FreeToken's flattened ragged batch and stage request state.

        Hashing uses each request's CPU token buffer, while projections and the
        convolution run on ``hidden_states.device``.  ``pool`` is intentionally
        duck-typed to the small ``PLEStatePool`` staging interface so this
        operator remains independent of engine imports.  Neither prefill nor
        decode publishes committed state from inside model.forward.

        Decode is staged before forward so this method contains only capturable
        GPU tensor work. Ragged prefill intentionally keeps the eager request
        loop because its shapes vary and it is never replayed as a CUDA graph.
        """

        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.hc_hidden_size:
            raise ValueError(
                "flattened hidden_states must have shape [tokens, "
                f"{self.hc_hidden_size}], got {tuple(hidden_states.shape)}"
            )
        if getattr(batch, "is_decode", False):
            return self.forward_staged_decode(hidden_states, batch, pool)
        reqs = getattr(batch, "padded_reqs", None) or batch.reqs
        expected_tokens = sum(int(req.extend_len) for req in reqs)
        if expected_tokens != hidden_states.shape[0]:
            raise ValueError(
                f"ragged request lengths sum to {expected_tokens}, but hidden_states "
                f"contains {hidden_states.shape[0]} tokens"
            )

        outputs: list[torch.Tensor] = []
        next_histories: list[torch.Tensor] = []
        next_conv_states: list[torch.Tensor] = []
        slots: list[int] = []
        offset = 0
        for req in reqs:
            length = int(req.extend_len)
            end = offset + length
            token_ids = req.input_ids[req.cached_len : req.device_len]
            if token_ids.numel() != length:
                raise ValueError(
                    f"request {req.table_idx} exposes {token_ids.numel()} new token ids, "
                    f"expected extend_len={length}"
                )
            history, conv_state = pool.begin(
                req.table_idx, fresh=int(req.cached_len) == 0
            )
            output, next_state = self.forward(
                hidden_states[offset:end].unsqueeze(0),
                token_ids.reshape(1, length),
                PLEState(history.unsqueeze(0), conv_state.unsqueeze(0)),
            )
            assert next_state.token_history is not None
            assert next_state.conv_state is not None
            outputs.append(output.squeeze(0))
            next_histories.append(next_state.token_history[0])
            next_conv_states.append(next_state.conv_state[0])
            slots.append(int(req.table_idx))
            offset = end

        # Stage only after every ragged request has completed PLE.  Pending rows
        # may be overwritten by retries, while committed request state remains
        # unchanged until Qwen4ExpForConditionalGeneration's engine hook runs.
        slot_tuple = tuple(slots)
        batch.ple_table_idx = pool.write_pending_prefill_conv_states(
            slot_tuple, torch.stack(next_conv_states, dim=0)
        )
        batch.ple_next_token_history = torch.stack(next_histories, dim=0)
        batch.ple_commit_slots = slot_tuple
        return torch.cat(outputs, dim=0)


__all__ = [
    "ConcatenatedRowBank",
    "NGramHashLayout",
    "PLEState",
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpNGramHasher",
    "Qwen4ExpPLELayer",
    "ReadonlyRowBank",
    "SafetensorsMmapRowBank",
    "SafetensorsRowShard",
    "ShardedSafetensorsMmapRowBank",
    "TensorRowBank",
    "build_layer_multipliers",
    "build_ngram_hash_layout",
]
