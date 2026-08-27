"""Qwen Query-Selective Attention KV pools.

The bring-up backends store one raw index key per token.  The SM120 long-context
backend instead stores one transformed key per complete four-token block and a
small per-request ring for the incomplete block.  Its full KV pages are aligned
to the compression ratio, making ``physical_token_slot // ratio`` a stable,
ownership-free compressed-cache address.

Unlike :mod:`freetoken.kvcache.bsa_pool`, both slabs are allocated only for the
explicit QSA ``layer_ids``.  Qwen interleaves QSA with recurrent GDN layers, and
allocating storage for all model layers would waste 4x the intended KV memory.
"""

from __future__ import annotations

from typing import Sequence

import torch
from freetoken.utils import div_even

from .mha_pool import MHAKVCache


QSA_SELECTOR_WORKSPACE_BYTES = 128 * 2**20
QSA_COMPRESSED_BACKEND = "qsa_triton_sm120"


def _uses_compressed_qsa(config) -> bool:
    backend = str(getattr(config, "attention_backend", ""))
    return QSA_COMPRESSED_BACKEND in {part.strip() for part in backend.split(",")}


class QSAKVCache(MHAKVCache):
    """Subset-layer GQA K/V plus raw or four-token-compressed index keys."""

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        layer_ids: Sequence[int],
        index_compress_ratio: int | None = None,
        num_request_slots: int = 0,
        selector_workspace_bytes: int = QSA_SELECTOR_WORKSPACE_BYTES,
    ) -> None:
        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if not layer_ids:
            raise ValueError("QSAKVCache requires at least one layer id")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"QSAKVCache layer_ids must be unique, got {layer_ids}")
        if index_head_dim <= 0:
            raise ValueError(f"QSA index_head_dim must be positive, got {index_head_dim}")
        if dtype.itemsize != 2:
            raise ValueError(
                "QSA index keys are stored in the 16-bit compute dtype; "
                f"got {dtype} ({dtype.itemsize} bytes)"
            )

        self.layer_ids = layer_ids
        self._index_head_dim = int(index_head_dim)
        self._page_size = int(page_size)
        self._index_layer_map = {
            global_id: dense for dense, global_id in enumerate(layer_ids)
        }
        self._index_dtype = dtype
        self._index_compress_ratio = (
            None if index_compress_ratio is None else int(index_compress_ratio)
        )
        self._num_request_slots = int(num_request_slots)
        self._selector_workspace_bytes = int(selector_workspace_bytes)
        self._selector_workspace_peak_bytes = 0
        self._selector_native_calls = 0
        self._selector_fallback_calls = 0
        self._selector_errors = 0
        if self._index_compress_ratio is not None:
            if self._index_compress_ratio <= 0:
                raise ValueError("QSA index compression ratio must be positive")
            if self._page_size % self._index_compress_ratio:
                raise ValueError(
                    "compressed QSA requires page_size divisible by the index "
                    f"compression ratio, got page_size={self._page_size}, "
                    f"ratio={self._index_compress_ratio}"
                )
            if self._num_request_slots <= 0:
                raise ValueError("compressed QSA requires at least one request slot")
            if self._selector_workspace_bytes < 0 or self._selector_workspace_bytes % 4:
                raise ValueError(
                    "QSA selector workspace must be a non-negative multiple of four bytes"
                )
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
        )
        self._alloc_index_slab(num_pages)
        self._alloc_pending_ring()
        # Only the production CUDA path needs a persistent score arena.  CPU
        # cache tests allocate exact-size temporary views on demand instead of
        # reserving 128 MiB of host RAM per fixture.
        self._selector_score_workspace = (
            torch.empty(
                self._selector_workspace_bytes // 4,
                dtype=torch.float32,
                device=device,
            )
            if self.uses_compressed_index
            and device.type == "cuda"
            and self._selector_workspace_bytes
            else None
        )

    @property
    def num_storage_layers(self) -> int:
        return len(self.layer_ids)

    def _alloc_index_slab(self, num_pages: int) -> None:
        # Zero-init is defensive: valid selection never reads unwritten rows, but
        # finite zeros turn an addressing regression into a wrong-test result
        # instead of NaN poisoning the whole model.
        if self.uses_compressed_index:
            assert self._index_compress_ratio is not None
            compressed_rows = (
                num_pages * self._page_size // self._index_compress_ratio
            )
            self._compressed_index_k_buffer = torch.zeros(
                len(self.layer_ids),
                compressed_rows,
                self._index_head_dim,
                dtype=self._index_dtype,
                device=self._device,
            )
            self._index_k_buffer = None
        else:
            self._index_k_buffer = torch.zeros(
                len(self.layer_ids),
                num_pages * self._page_size,
                self._index_head_dim,
                dtype=self._index_dtype,
                device=self._device,
            )
            self._compressed_index_k_buffer = None

    def _alloc_pending_ring(self) -> None:
        if not self.uses_compressed_index:
            self._pending_index_k_buffer = None
            self._pending_position_buffer = None
            return
        assert self._index_compress_ratio is not None
        ring_rows = self._num_request_slots * self._index_compress_ratio
        self._pending_index_k_buffer = torch.zeros(
            len(self.layer_ids),
            ring_rows,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        # Keep positions per QSA layer, just like the raw-key ring.  Attention
        # layers execute one after another, so a shared position ring would let
        # an earlier layer overwrite the prior chunk before a later layer had
        # transformed its completed block.  Three axes also cover plain text:
        # the backend stores the scalar coordinate in all axes.
        self._pending_position_buffer = torch.zeros(
            len(self.layer_ids), ring_rows, 3, dtype=torch.int64, device=self._device
        )

    @property
    def uses_compressed_index(self) -> bool:
        return self._index_compress_ratio is not None

    @property
    def index_compress_ratio(self) -> int:
        return int(self._index_compress_ratio or 1)

    def _index_dense(self, layer_id: int) -> int:
        try:
            return self._index_layer_map[layer_id]
        except KeyError:
            raise KeyError(f"layer {layer_id} has no QSA index-key storage") from None

    def index_k_cache(self, layer_id: int) -> torch.Tensor:
        """Raw index keys ``[physical_token_rows, index_head_dim]``."""
        if self.uses_compressed_index:
            raise RuntimeError(
                "compressed QSA has no per-token index slab; use "
                "compressed_index_k_cache/pending_index_k_cache"
            )
        assert self._index_k_buffer is not None
        return self._index_k_buffer[self._index_dense(layer_id)]

    def compressed_index_k_cache(self, layer_id: int) -> torch.Tensor:
        if not self.uses_compressed_index:
            raise RuntimeError("raw QSA cache has no compressed index-key slab")
        assert self._compressed_index_k_buffer is not None
        return self._compressed_index_k_buffer[self._index_dense(layer_id)]

    def pending_index_k_cache(self, layer_id: int) -> torch.Tensor:
        if not self.uses_compressed_index:
            raise RuntimeError("raw QSA cache has no pending index-key ring")
        assert self._pending_index_k_buffer is not None
        return self._pending_index_k_buffer[self._index_dense(layer_id)]

    def pending_index_positions(self, layer_id: int) -> torch.Tensor:
        if not self.uses_compressed_index:
            raise RuntimeError("raw QSA cache has no pending position ring")
        assert self._pending_position_buffer is not None
        return self._pending_position_buffer[self._index_dense(layer_id)]

    def store_index_k(
        self, key: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        if key.ndim == 3 and key.shape[-2] == 1:
            key = key.squeeze(-2)
        if key.ndim != 2 or key.shape[-1] != self._index_head_dim:
            raise ValueError(
                "QSA index key must have shape [tokens, index_head_dim] (or a "
                f"singleton KV-head axis), got {tuple(key.shape)}"
            )
        self.index_k_cache(layer_id)[out_loc.to(torch.long)] = key

    def store_pending_index_k(
        self, key: torch.Tensor, ring_loc: torch.Tensor, layer_id: int
    ) -> None:
        if key.ndim == 3 and key.shape[-2] == 1:
            key = key.squeeze(-2)
        if key.ndim != 2 or key.shape[-1] != self._index_head_dim:
            raise ValueError(
                f"QSA pending keys must be [tokens, {self._index_head_dim}], "
                f"got {tuple(key.shape)}"
            )
        self.pending_index_k_cache(layer_id)[ring_loc.to(torch.long)] = key

    def store_pending_positions(
        self, positions: torch.Tensor, ring_loc: torch.Tensor, layer_id: int
    ) -> None:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                "QSA pending positions must be [tokens] or [3,tokens], got "
                f"{tuple(positions.shape)}"
            )
        if positions.shape[1] != ring_loc.numel():
            raise ValueError("QSA pending positions and ring locations do not agree")
        destination = self.pending_index_positions(layer_id)
        source = positions.transpose(0, 1).to(
            device=destination.device,
            dtype=destination.dtype,
        )
        destination[ring_loc.to(device=destination.device, dtype=torch.long)] = source

    def store_compressed_index_k(
        self, key: torch.Tensor, compressed_loc: torch.Tensor, layer_id: int
    ) -> None:
        if key.ndim == 3 and key.shape[-2] == 1:
            key = key.squeeze(-2)
        if key.ndim != 2 or key.shape[-1] != self._index_head_dim:
            raise ValueError(
                f"QSA compressed keys must be [blocks, {self._index_head_dim}], "
                f"got {tuple(key.shape)}"
            )
        self.compressed_index_k_cache(layer_id)[compressed_loc.to(torch.long)] = key

    def selector_score_workspace(self, rows: int, num_blocks: int) -> torch.Tensor:
        elements = int(rows) * int(num_blocks)
        need = elements * 4
        if need > self._selector_workspace_bytes:
            raise MemoryError(
                f"QSA score workspace needs {need} bytes, exceeds the bounded "
                f"{self._selector_workspace_bytes}-byte arena"
            )
        self._selector_workspace_peak_bytes = max(
            self._selector_workspace_peak_bytes, need
        )
        if self._selector_score_workspace is None:
            return torch.empty(
                (rows, num_blocks), dtype=torch.float32, device=self._device
            )
        return self._selector_score_workspace[:elements].view(rows, num_blocks)

    @property
    def selector_workspace_peak_bytes(self) -> int:
        """Largest score-workspace view requested since engine startup."""

        return self._selector_workspace_peak_bytes

    def record_selector_dispatch(self, outcome: str) -> None:
        """Record one native selector launch, visible fallback, or launch error."""

        if outcome == "native":
            self._selector_native_calls += 1
        elif outcome == "fallback":
            self._selector_fallback_calls += 1
        elif outcome == "error":
            self._selector_errors += 1
        else:
            raise ValueError(f"unknown QSA selector dispatch outcome: {outcome!r}")

    def selector_telemetry(self) -> dict[str, int]:
        """Cumulative counters exported in release runtime evidence."""

        return {
            "workspace_peak_bytes": self._selector_workspace_peak_bytes,
            "native_calls": self._selector_native_calls,
            "fallback_calls": self._selector_fallback_calls,
            "errors": self._selector_errors,
        }

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        num_rows = int(out_loc.numel())
        row_shape = self._storage_shape[1:]
        row_width = int(row_shape[0]) * int(row_shape[1])

        def _flatten(name: str, value: torch.Tensor) -> torch.Tensor:
            if value.shape[0] != num_rows or value.numel() != num_rows * row_width:
                raise ValueError(
                    f"QSA {name} must contain {num_rows} rows of width {row_width}, "
                    f"got {tuple(value.shape)}"
                )
            return value.reshape(num_rows, row_width).contiguous()

        k_flat = _flatten("key", k)
        v_flat = _flatten("value", v)
        # The production path keeps using the fused scatter.  A tiny CPU fallback
        # makes allocation/addressing tests independent of CUDA/Triton.
        if self._device.type != "cpu":
            return super().store_kv(k_flat, v_flat, out_loc, layer_id)
        dense = self._dense(layer_id)
        rows = out_loc.to(torch.long)
        k_rows = self._k_buffer[dense].view(self._storage_shape)
        v_rows = self._v_buffer[dense].view(self._storage_shape)
        k_rows[rows] = k_flat.view(-1, *row_shape)
        v_rows[rows] = v_flat.view(-1, *row_shape)

    def rebuild(self, num_pages: int) -> None:
        self._index_k_buffer = None
        self._compressed_index_k_buffer = None
        super().rebuild(num_pages)
        try:
            self._alloc_index_slab(num_pages)
        except Exception:
            # Do not leave a half-rebuilt pool available to the scheduler.
            self._kv_buffer = None
            self._k_buffer = None
            self._v_buffer = None
            raise

    def unit_bytes(self) -> tuple[int, int]:
        kv, _ = super().unit_bytes()
        index_buffer = (
            self._compressed_index_k_buffer
            if self.uses_compressed_index
            else self._index_k_buffer
        )
        assert index_buffer is not None
        raw_rows = int(self._kv_buffer.shape[2]) * int(self._kv_buffer.shape[3])
        index = int(index_buffer.numel() * index_buffer.element_size()) // raw_rows
        return kv + index, 0

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        """Exact QSA page and fixed-arena cost before pool allocation."""
        from freetoken.models.config import QSAAttentionGroupConfig

        groups = [
            group
            for group in config.model_config.attention_groups
            if isinstance(group, QSAAttentionGroupConfig)
        ]
        if len(groups) != 1:
            raise ValueError(f"QSA cache pricing requires one QSA group, got {len(groups)}")
        group = groups[0]
        layers = len(group.layer_ids)
        local_kv_heads = div_even(
            group.num_kv_heads, config.tp_info.size, allow_replicate=True
        )
        main_per_token = (
            2
            * layers
            * local_kv_heads
            * group.head_dim
            * config.dtype.itemsize
        )
        fixed = 0
        if _uses_compressed_qsa(config):
            ratio = group.indexer_compress_ratio
            if config.page_size % ratio:
                raise ValueError(
                    f"compressed QSA page_size {config.page_size} must be divisible by {ratio}"
                )
            index_numerator = (
                layers * group.indexer_head_dim * config.dtype.itemsize
            )
            if index_numerator % ratio:
                raise ValueError("QSA compressed index bytes are not integral per token")
            index_per_token = index_numerator // ratio
            ring_rows = (config.max_running_req + 1) * ratio
            ring_bytes = (
                ring_rows
                * layers
                * group.indexer_head_dim
                * config.dtype.itemsize
            )
            position_bytes = ring_rows * layers * 3 * 8
            fixed = QSA_SELECTOR_WORKSPACE_BYTES + ring_bytes + position_bytes
        else:
            index_per_token = (
                layers * group.indexer_head_dim * config.dtype.itemsize
            )
        return (
            (main_per_token + index_per_token) * config.page_size,
            fixed,
            config.page_size,
            0,
        )

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        num_pages = super().solve_num_pages(config, available_memory)
        per_page, fixed, _, _ = cls.kv_cost(config)
        need = num_pages * per_page + fixed
        if need > available_memory:
            raise MemoryError(
                "QSA cache geometry exceeds its startup budget: "
                f"pages={num_pages}, page_bytes={per_page}, fixed_bytes={fixed}, "
                f"need={need}, available={available_memory}. Reduce --num-tokens "
                "or the MoE cache before loading the pool."
            )
        return num_pages


__all__ = [
    "QSAKVCache",
    "QSA_COMPRESSED_BACKEND",
    "QSA_SELECTOR_WORKSPACE_BYTES",
]
