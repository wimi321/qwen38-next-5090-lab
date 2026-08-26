"""Triton CUDA QSA backend with a vectorized Torch fallback.

Supported 8K shapes dispatch score/top-k/token expansion and selected paged GQA
to real Triton kernels. Unsupported shapes use the no-per-query-loop Torch CUDA
implementation. Prefill memory stays bounded with fixed query chunks.

Decode metadata is fixed-shape, but model-level CUDA graph capture remains
disabled until the new QSA kernels have passed capture/replay on target hardware.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.kernel.qsa_vectorized import (
    qsa_paged_gqa_attention_vectorized,
    qsa_pool_index_keys_vectorized,
    qsa_select_token_indices_vectorized,
)
from freetoken.kernel.qsa_triton import (
    can_use_qsa_attention_triton,
    can_use_qsa_selection_triton,
    qsa_paged_gqa_attention_triton,
    qsa_select_token_indices_triton,
)
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


_PREFILL_QUERY_CHUNK = 32
logger = init_logger(__name__)


@dataclass
class QSATritonMetadata(BaseAttnMetadata):
    is_decode: bool
    qo_indptr_cpu: torch.Tensor
    kv_len_cpu: torch.Tensor
    last_indices: torch.Tensor
    # Fixed-shape decode addressing, eagerly snapshotted or staged into capture
    # buffers. Prefill derives per-request rows directly from the page table.
    rows: torch.Tensor | None = None
    kv_lens: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSATritonBackend(BaseAttnBackend):
    """Triton QSA with a vectorized Torch fallback for unsupported geometry."""

    # Metadata and tensors are fixed-shape, but the new kernels have not yet
    # passed capture/replay on the target GPU. Do not advertise an unverified path.
    supports_cuda_graph = False

    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.qsa_pool import QSAKVCache
        from freetoken.models.config import QSAAttentionGroupConfig

        groups = [
            group
            for group in getattr(config, "attention_groups", ())
            if isinstance(group, QSAAttentionGroupConfig)
        ]
        if len(groups) != 1:
            raise ValueError(f"qsa_triton requires exactly one QSA group, got {len(groups)}")
        self.group = groups[0]
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        if not isinstance(self.kvcache, QSAKVCache):
            raise TypeError(
                "qsa_triton requires QSAKVCache, got "
                f"{type(self.kvcache).__name__}; QSA must not route through BSA"
            )
        self.device = self.kvcache.device
        self.sm_scale = (
            config.attn_sm_scale
            if config.attn_sm_scale is not None
            else self.group.head_dim**-0.5
        )
        self.capture_bs: List[int] = []
        self._rows_buf: torch.Tensor | None = None
        self._kv_lens_buf: torch.Tensor | None = None
        self._logged_dispatch: set[tuple[bool, bool]] = set()

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        q_lens = [int(req.extend_len) for req in reqs]
        kv_lens = [int(req.device_len) for req in reqs]
        qo = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum_(0)
        batch.attn_metadata = QSATritonMetadata(
            is_decode=getattr(batch, "phase", None) == "decode",
            qo_indptr_cpu=qo,
            kv_len_cpu=torch.tensor(kv_lens, dtype=torch.int32),
            last_indices=(qo[1:] - 1).to(self.device),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "QSA layers must call qsa_forward with index_q, raw index_k, and "
            "index_key_transform; generic forward would change model semantics"
        )

    def _flat_kv(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (-1, self.group.num_kv_heads, self.group.head_dim)
        return (
            self.kvcache.k_cache(layer_id).view(shape),
            self.kvcache.v_cache(layer_id).view(shape),
        )

    def _run_vectorized(
        self,
        q: torch.Tensor,
        index_q: torch.Tensor,
        page_rows: torch.Tensor,
        query_positions: torch.Tensor,
        kv_lens: torch.Tensor,
        layer_id: int,
        index_key_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        pooled_index_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q/index_q arrive batched: [B,Q,H,D]. page_rows has a fixed padded
        # logical width; length masks make every trailing row inert.
        index_rows = self.kvcache.index_k_cache(layer_id)
        if pooled_index_key is None:
            raw = index_rows[page_rows.to(torch.long)]
            pooled_index_key = qsa_pool_index_keys_vectorized(
                raw,
                compress_ratio=self.group.indexer_compress_ratio,
                key_transform=index_key_transform,
            )
        selection_kwargs = {
            "token_budget": self.group.indexer_budget,
            "compress_ratio": self.group.indexer_compress_ratio,
        }
        use_triton_selection = can_use_qsa_selection_triton(
            index_q, pooled_index_key, **selection_kwargs
        )
        if use_triton_selection:
            logical, valid = qsa_select_token_indices_triton(
                index_q,
                pooled_index_key,
                query_positions,
                kv_lens,
                **selection_kwargs,
            )
        else:
            logical, valid = qsa_select_token_indices_vectorized(
                index_q,
                pooled_index_key,
                query_positions,
                kv_lens,
                **selection_kwargs,
            )
        key_rows, value_rows = self._flat_kv(layer_id)
        attention_args = (q, key_rows, value_rows, page_rows, logical, valid)
        use_triton_attention = can_use_qsa_attention_triton(q, key_rows, logical)
        dispatch = (use_triton_selection, use_triton_attention)
        if dispatch not in self._logged_dispatch:
            self._logged_dispatch.add(dispatch)
            message = (
                "QSA dispatch: selection="
                f"{'triton' if use_triton_selection else 'torch'}, attention="
                f"{'triton' if use_triton_attention else 'torch'}, "
                f"dtype={q.dtype}, q_shape={tuple(q.shape)}, "
                f"index_shape={tuple(index_q.shape)}"
            )
            if all(dispatch):
                logger.info_rank0(message)
            else:
                logger.warning_rank0(message + "; benchmark results include a fallback path")
        if use_triton_attention:
            return qsa_paged_gqa_attention_triton(
                *attention_args, sm_scale=self.sm_scale
            )
        return qsa_paged_gqa_attention_vectorized(
            *attention_args, sm_scale=self.sm_scale
        )

    def qsa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        layer_id: int,
        batch: Batch,
        *,
        index_key_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Store current rows and run vectorized causal QSA.

        The stable transform interface is batched:
        ``index_key_transform(pooled [B,N,Di], block_starts [B,N])``.  It must
        apply key RMSNorm + partial RoPE and preserve the pooled shape.
        """
        if layer_id not in self.group.layer_ids:
            raise KeyError(f"layer {layer_id} is not owned by QSA group {self.group.name!r}")
        md = batch.attn_metadata
        if not isinstance(md, QSATritonMetadata):
            raise TypeError("qsa_triton metadata was not prepared for this batch")
        if index_q.ndim != 3 or index_q.shape[-2:] != (
            self.group.indexer_n_heads,
            self.group.indexer_head_dim,
        ):
            raise ValueError(
                "QSA index_q shape mismatch: expected [T, "
                f"{self.group.indexer_n_heads}, {self.group.indexer_head_dim}], "
                f"got {tuple(index_q.shape)}"
            )

        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        self.kvcache.store_index_k(index_k, batch.out_loc, layer_id)

        if md.is_decode:
            if md.rows is None or md.kv_lens is None:
                self._snapshot_eager_decode(batch, md)
            batch_size = md.rows.shape[0]
            return self._run_vectorized(
                q.view(batch_size, 1, q.shape[-2], q.shape[-1]),
                index_q.view(batch_size, 1, index_q.shape[-2], index_q.shape[-1]),
                md.rows,
                batch.positions[:batch_size].view(batch_size, 1).to(torch.long),
                md.kv_lens,
                layer_id,
                index_key_transform,
            ).view_as(q)

        # Prefill/extend is eager. One loop per request and bounded query chunk,
        # never per query. Pool/transform index keys once per request, not once
        # per chunk.
        output = torch.empty_like(q)
        page_table = get_global_ctx().page_table
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        qo = md.qo_indptr_cpu.tolist()
        ratio = self.group.indexer_compress_ratio
        index_rows = self.kvcache.index_k_cache(layer_id)
        for req_idx, req in enumerate(reqs):
            begin, end = qo[req_idx], qo[req_idx + 1]
            if begin == end:
                continue
            padded_width = ((int(req.device_len) + ratio - 1) // ratio) * ratio
            rows = page_table[req.table_idx, :padded_width].view(1, -1)
            raw = index_rows[rows.to(torch.long)]
            pooled = qsa_pool_index_keys_vectorized(
                raw,
                compress_ratio=ratio,
                key_transform=index_key_transform,
            )
            kv_lens = torch.tensor([req.device_len], dtype=torch.long, device=self.device)
            for chunk_start in range(begin, end, _PREFILL_QUERY_CHUNK):
                chunk_end = min(chunk_start + _PREFILL_QUERY_CHUNK, end)
                sl = slice(chunk_start, chunk_end)
                output[sl] = self._run_vectorized(
                    q[sl].unsqueeze(0),
                    index_q[sl].unsqueeze(0),
                    rows,
                    batch.positions[sl].view(1, -1).to(torch.long),
                    kv_lens,
                    layer_id,
                    index_key_transform,
                    pooled_index_key=pooled,
                ).squeeze(0)
        return output

    def _snapshot_eager_decode(self, batch: Batch, md: QSATritonMetadata) -> None:
        if getattr(batch, "active_table_idx", None) is not None:
            table_idx = batch.active_table_idx.to(torch.long)
        else:
            reqs = getattr(batch, "padded_reqs", batch.reqs)
            table_idx = torch.tensor(
                [req.table_idx for req in reqs], dtype=torch.long, device=self.device
            )
        md.rows = get_global_ctx().page_table.index_select(0, table_idx)
        md.kv_lens = md.kv_len_cpu.to(self.device, non_blocking=True).to(torch.long)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        if not bs_list:
            return
        width = get_global_ctx().page_table.shape[1]
        max_bs = max(bs_list)
        self._rows_buf = torch.zeros(
            max_bs, width, dtype=torch.int32, device=self.device
        )
        self._kv_lens_buf = torch.zeros(max_bs, dtype=torch.long, device=self.device)

    def _stage_decode(
        self, batch: Batch, table_idx: torch.Tensor, bs: int
    ) -> None:
        md = batch.attn_metadata
        if not isinstance(md, QSATritonMetadata):
            raise TypeError("qsa_triton metadata was not prepared for graph staging")
        assert self._rows_buf is not None and self._kv_lens_buf is not None
        self._rows_buf[:bs].copy_(
            get_global_ctx().page_table.index_select(0, table_idx.to(torch.long))
        )
        self._kv_lens_buf[:bs].copy_(
            md.kv_len_cpu[:bs].to(self.device, non_blocking=True)
        )
        md.rows = self._rows_buf[:bs]
        md.kv_lens = self._kv_lens_buf[:bs]

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
        bs = batch.size
        dummy = torch.full(
            (bs,),
            batch.padded_reqs[0].table_idx,
            dtype=torch.long,
            device=self.device,
        )
        self._stage_decode(batch, dummy, bs)

    def prepare_for_replay(self, batch: Batch) -> None:
        if batch.active_table_idx is None:
            raise RuntimeError("qsa_triton decode replay is missing active_table_idx")
        self._stage_decode(
            batch,
            batch.active_table_idx,
            batch.padded_size,
        )

    def reset_capture(self) -> None:
        super().reset_capture()
        self._rows_buf = None
        self._kv_lens_buf = None


__all__ = ["QSATritonBackend", "QSATritonMetadata"]
