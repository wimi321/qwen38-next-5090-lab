"""Qwen Query-Selective Attention KV pool.

QSA stores ordinary paged GQA K/V plus one *raw* index key per token and QSA
layer.  The raw keys are intentionally not pooled in the cache: a query can end
inside a four-token microblock, so the set of complete blocks is causal and
query-dependent.  The correctness backend pools visible groups when selecting.

Unlike :mod:`freetoken.kvcache.bsa_pool`, both slabs are allocated only for the
explicit QSA ``layer_ids``.  Qwen interleaves QSA with recurrent GDN layers, and
allocating storage for all model layers would waste 4x the intended KV memory.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .mha_pool import MHAKVCache


class QSAKVCache(MHAKVCache):
    """Subset-layer GQA K/V plus a subset-layer raw index-key slab."""

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

    @property
    def num_storage_layers(self) -> int:
        return len(self.layer_ids)

    def _alloc_index_slab(self, num_pages: int) -> None:
        # Zero-init is defensive: valid selection never reads unwritten rows, but
        # finite zeros turn an addressing regression into a wrong-test result
        # instead of NaN poisoning the whole model.
        self._index_k_buffer = torch.zeros(
            len(self.layer_ids),
            num_pages * self._page_size,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )

    def _index_dense(self, layer_id: int) -> int:
        try:
            return self._index_layer_map[layer_id]
        except KeyError:
            raise KeyError(f"layer {layer_id} has no QSA index-key storage") from None

    def index_k_cache(self, layer_id: int) -> torch.Tensor:
        """Raw index keys ``[physical_token_rows, index_head_dim]``."""
        return self._index_k_buffer[self._index_dense(layer_id)]

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

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        # The production path keeps using the fused scatter.  A tiny CPU fallback
        # makes allocation/addressing tests independent of CUDA/Triton.
        if self._device.type != "cpu":
            return super().store_kv(k, v, out_loc, layer_id)
        dense = self._dense(layer_id)
        rows = out_loc.to(torch.long)
        k_rows = self._k_buffer[dense].view(self._storage_shape)
        v_rows = self._v_buffer[dense].view(self._storage_shape)
        k_rows[rows] = k.view(-1, *self._storage_shape[1:])
        v_rows[rows] = v.view(-1, *self._storage_shape[1:])

    def rebuild(self, num_pages: int) -> None:
        self._index_k_buffer = None
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
        rows = int(self._index_k_buffer.shape[1])
        index = int(self._index_k_buffer.numel() * self._index_k_buffer.element_size()) // rows
        return kv + index, 0


__all__ = ["QSAKVCache"]
