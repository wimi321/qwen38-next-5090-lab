"""Slow, exact PyTorch backend for Qwen Query-Selective Attention.

This is the architecture bring-up path, not the final performance kernel.  It
uses the regular paged scheduler for physical rows, stores QSA raw index keys in
``QSAKVCache``, materializes each request's logical history, then invokes the
independent oracle in :mod:`freetoken.kernel.qsa_reference`.

QSA model layers must call :meth:`QSAReferenceBackend.qsa_forward`, supplying
their normalized/RoPE'd index queries, raw index keys, and an
``index_key_transform`` that applies key RMSNorm + partial RoPE *after* each
four-token mean pool.  The ordinary ``forward`` entry point raises rather than
silently serving QSA as dense or MiniMax BSA attention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.kernel.qsa_reference import (
    qsa_causal_visibility,
    qsa_reference_forward,
)

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class QSAReferenceMetadata(BaseAttnMetadata):
    qo_indptr_cpu: torch.Tensor
    kv_len_cpu: torch.Tensor
    last_indices: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSAReferenceBackend(BaseAttnBackend):
    """Correctness-first QSA backend (works on CPU or CUDA, deliberately slow)."""

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
            raise ValueError(f"qsa_torch requires exactly one QSA group, got {len(groups)}")
        self.group = groups[0]
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        if not isinstance(self.kvcache, QSAKVCache):
            raise TypeError(
                "qsa_torch requires QSAKVCache, got "
                f"{type(self.kvcache).__name__}; QSA must not route through BSA"
            )
        self.device = self.kvcache.device
        self.sm_scale = (
            config.attn_sm_scale
            if config.attn_sm_scale is not None
            else self.group.head_dim**-0.5
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        q_lens = [int(req.extend_len) for req in reqs]
        kv_lens = [int(req.device_len) for req in reqs]
        qo = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum_(0)
        last = (qo[1:] - 1).to(self.device)
        batch.attn_metadata = QSAReferenceMetadata(
            qo_indptr_cpu=qo,
            kv_len_cpu=torch.tensor(kv_lens, dtype=torch.int32),
            last_indices=last,
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
            "index_key_transform; generic forward would silently change model semantics"
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
        """Store current rows and run exact causal QSA over each request.

        Shapes are FreeToken's flattened convention: ``q [T,Hq,D]``, flattened
        or headed ``k/v``, ``index_q [T,Hi,Di]``, and raw ``index_k [T,Di]``.
        The stable transform contract is batched, matching ``qsa_triton``:
        ``index_key_transform(pooled [B,N,Di], starts [B,N])``. It must apply
        Qwen key norm + partial RoPE and preserve the pooled shape.
        """
        if layer_id not in self.group.layer_ids:
            raise KeyError(f"layer {layer_id} is not owned by QSA group {self.group.name!r}")
        md = batch.attn_metadata
        if not isinstance(md, QSAReferenceMetadata):
            raise TypeError("qsa_torch metadata was not prepared for this batch")
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

        q_out = torch.empty_like(q)
        page_table = get_global_ctx().page_table
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        qo = md.qo_indptr_cpu.tolist()
        k_rows = self.kvcache.k_cache(layer_id).view(
            -1, self.group.num_kv_heads, self.group.head_dim
        )
        v_rows = self.kvcache.v_cache(layer_id).view(
            -1, self.group.num_kv_heads, self.group.head_dim
        )
        index_rows = self.kvcache.index_k_cache(layer_id)

        for req_idx, req in enumerate(reqs):
            start, end = qo[req_idx], qo[req_idx + 1]
            if start == end:
                continue
            logical_rows = page_table[req.table_idx, : req.device_len].to(torch.long)
            hist_k = k_rows.index_select(0, logical_rows)
            hist_v = v_rows.index_select(0, logical_rows)
            hist_index_k = index_rows.index_select(0, logical_rows)
            query_positions = batch.positions[start:end].to(torch.long)
            visible = qsa_causal_visibility(query_positions, req.device_len)

            def transform(
                pooled: torch.Tensor, block_starts: torch.Tensor, _batch_idx: int
            ) -> torch.Tensor:
                return index_key_transform(
                    pooled.unsqueeze(0), block_starts.unsqueeze(0)
                ).squeeze(0)

            q_out[start:end], _ = qsa_reference_forward(
                q[start:end],
                hist_k,
                hist_v,
                index_q[start:end],
                hist_index_k,
                visible,
                token_budget=self.group.indexer_budget,
                compress_ratio=self.group.indexer_compress_ratio,
                key_transform=transform,
                sm_scale=self.sm_scale,
            )
        return q_out

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        if bs_list:
            raise RuntimeError(
                "qsa_torch is a correctness backend and cannot be CUDA-graph captured; "
                "configuration must disable graph batch sizes"
            )

    def prepare_for_capture(self, batch: Batch) -> None:
        raise RuntimeError("qsa_torch does not support CUDA graph capture")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise RuntimeError("qsa_torch does not support CUDA graph replay")


__all__ = ["QSAReferenceBackend", "QSAReferenceMetadata"]
