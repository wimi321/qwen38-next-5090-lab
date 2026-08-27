"""Triton CUDA QSA backend with a vectorized Torch fallback.

Supported 8K shapes dispatch score/top-k/token expansion and selected paged GQA
to real Triton kernels. Unsupported shapes use the no-per-query-loop Torch CUDA
implementation. Prefill memory stays bounded with fixed query chunks.

Decode metadata is fixed-shape, but model-level CUDA graph capture remains
disabled until the new QSA kernels have passed capture/replay on target hardware.
"""

from __future__ import annotations

import os
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
    can_use_qsa_selection_sm120,
    can_use_qsa_selection_triton,
    qsa_paged_gqa_attention_triton,
    qsa_select_token_indices_sm120,
    qsa_select_token_indices_triton,
)
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


_PREFILL_QUERY_CHUNK = 32
logger = init_logger(__name__)


def _assert_qsa_selection_has_visible_tokens(valid: torch.Tensor) -> None:
    """Fail closed if any query row has no selected causal token.

    An empty row makes the Triton softmax subtract ``-inf`` from ``-inf`` and
    poisons every head with NaNs.  ``torch._assert_async`` keeps the invariant
    check on the CUDA stream: the healthy path does not synchronize or alter
    attention values, while an internal selector/addressing regression aborts
    the worker instead of returning corrupt token logits.  On CPU it raises
    synchronously, which gives focused tests a cheap seam for the same guard.
    """

    if valid.ndim < 2:
        raise ValueError(
            f"QSA valid mask must include query and selection axes, got {valid.shape}"
        )
    torch._assert_async(
        valid.any(dim=-1).all(),
        "QSA selector returned an empty visible-token row; aborting before attention",
    )


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
    requires_compressed_index = False
    prefill_query_chunk = _PREFILL_QUERY_CHUNK

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
        if self.kvcache.uses_compressed_index != self.requires_compressed_index:
            expected = "compressed" if self.requires_compressed_index else "raw"
            raise TypeError(
                f"{type(self).__name__} requires the {expected} QSA index layout"
            )
        require_native_raw = os.environ.get(
            "FREETOKEN_QSA_REQUIRE_NATIVE_TOPK", "0"
        ).strip().lower()
        if require_native_raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError(
                "FREETOKEN_QSA_REQUIRE_NATIVE_TOPK must be a boolean value"
            )
        self.require_native_topk = require_native_raw in {"1", "true", "yes", "on"}
        if self.require_native_topk and not self.requires_compressed_index:
            raise ValueError(
                "native SM120 QSA fast-topk can only be required by the "
                "qsa_triton_sm120 backend"
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
        self._logged_dispatch: set[tuple[str, str]] = set()

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
        if pooled_index_key is None:
            index_rows = self.kvcache.index_k_cache(layer_id)
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
        use_sm120_selection = (
            self.requires_compressed_index
            and can_use_qsa_selection_sm120(
                index_q, pooled_index_key, **selection_kwargs
            )
        )
        if self.require_native_topk and not use_sm120_selection:
            self.kvcache.record_selector_dispatch("error")
            raise RuntimeError(
                "the active release profile requires native SM120 QSA fast-topk, "
                "but this selector geometry cannot use qsa_triton_sm120"
            )
        if use_sm120_selection:
            selection_impl = "sm120-radix"
            score_workspace = self.kvcache.selector_score_workspace(
                index_q.shape[0] * index_q.shape[1], pooled_index_key.shape[1]
            )
            logical, valid = qsa_select_token_indices_sm120(
                index_q,
                pooled_index_key,
                query_positions,
                kv_lens,
                score_workspace=score_workspace,
                require_native_topk=self.require_native_topk,
                selector_telemetry=self.kvcache.record_selector_dispatch,
                **selection_kwargs,
            )
        elif use_triton_selection:
            selection_impl = "triton"
            logical, valid = qsa_select_token_indices_triton(
                index_q,
                pooled_index_key,
                query_positions,
                kv_lens,
                **selection_kwargs,
            )
        else:
            selection_impl = "torch"
            logical, valid = qsa_select_token_indices_vectorized(
                index_q,
                pooled_index_key,
                query_positions,
                kv_lens,
                **selection_kwargs,
            )
        _assert_qsa_selection_has_visible_tokens(valid)
        key_rows, value_rows = self._flat_kv(layer_id)
        attention_args = (q, key_rows, value_rows, page_rows, logical, valid)
        use_triton_attention = can_use_qsa_attention_triton(q, key_rows, logical)
        attention_impl = "triton" if use_triton_attention else "torch"
        dispatch = (selection_impl, attention_impl)
        if dispatch not in self._logged_dispatch:
            self._logged_dispatch.add(dispatch)
            message = (
                "QSA dispatch: selection="
                f"{selection_impl}, attention={attention_impl}, "
                f"dtype={q.dtype}, q_shape={tuple(q.shape)}, "
                f"index_shape={tuple(index_q.shape)}"
            )
            if selection_impl != "torch" and use_triton_attention:
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
        if self.requires_compressed_index:
            self._update_compressed_index(
                index_k,
                layer_id,
                batch,
                md,
                index_key_transform,
            )
        else:
            self.kvcache.store_index_k(index_k, batch.out_loc, layer_id)

        if md.is_decode:
            if md.rows is None or md.kv_lens is None:
                self._snapshot_eager_decode(batch, md)
            batch_size = md.rows.shape[0]
            pooled = (
                self._gather_decode_compressed_keys(md, layer_id)
                if self.requires_compressed_index
                else None
            )
            return self._run_vectorized(
                q.view(batch_size, 1, q.shape[-2], q.shape[-1]),
                index_q.view(batch_size, 1, index_q.shape[-2], index_q.shape[-1]),
                md.rows,
                batch.positions[:batch_size].view(batch_size, 1).to(torch.long),
                md.kv_lens,
                layer_id,
                index_key_transform,
                pooled_index_key=pooled,
            ).view_as(q)

        # Prefill/extend is eager. One loop per request and bounded query chunk,
        # never per query. Pool/transform index keys once per request, not once
        # per chunk.
        output = torch.empty_like(q)
        page_table = get_global_ctx().page_table
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        qo = md.qo_indptr_cpu.tolist()
        ratio = self.group.indexer_compress_ratio
        index_rows = (
            None
            if self.requires_compressed_index
            else self.kvcache.index_k_cache(layer_id)
        )
        for req_idx, req in enumerate(reqs):
            begin, end = qo[req_idx], qo[req_idx + 1]
            if begin == end:
                continue
            padded_width = ((int(req.device_len) + ratio - 1) // ratio) * ratio
            rows = page_table[req.table_idx, :padded_width].view(1, -1)
            if self.requires_compressed_index:
                complete_blocks = int(req.device_len) // ratio
                compressed_locs = (
                    rows[:, : complete_blocks * ratio : ratio].to(torch.long)
                    // ratio
                )
                pooled = self.kvcache.compressed_index_k_cache(layer_id)[
                    compressed_locs
                ]
            else:
                assert index_rows is not None
                raw = index_rows[rows.to(torch.long)]
                pooled = qsa_pool_index_keys_vectorized(
                    raw,
                    compress_ratio=ratio,
                    key_transform=index_key_transform,
                )
            kv_lens = torch.tensor([req.device_len], dtype=torch.long, device=self.device)
            for chunk_start in range(begin, end, self.prefill_query_chunk):
                chunk_end = min(chunk_start + self.prefill_query_chunk, end)
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

    @staticmethod
    def _batch_rope_positions(batch: Batch) -> torch.Tensor:
        positions = getattr(batch, "mrope_positions", None)
        if positions is None:
            positions = batch.positions
        if positions.ndim not in (1, 2):
            raise ValueError(
                f"QSA positions must be [tokens] or [3,tokens], got {tuple(positions.shape)}"
            )
        if positions.ndim == 2 and positions.shape[0] != 3:
            raise ValueError(
                f"QSA mRoPE positions must be [3,tokens], got {tuple(positions.shape)}"
            )
        return positions

    def _update_compressed_index(
        self,
        index_k: torch.Tensor,
        layer_id: int,
        batch: Batch,
        md: QSATritonMetadata,
        index_key_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Write completed groups and retain only the request's incomplete tail."""
        ratio = self.group.indexer_compress_ratio
        if ratio != self.kvcache.index_compress_ratio:
            raise RuntimeError("QSA backend/pool compression ratios do not agree")
        if index_k.ndim == 3 and index_k.shape[-2] == 1:
            index_k = index_k.squeeze(-2)
        if index_k.ndim != 2:
            raise ValueError(f"QSA raw index keys must be rank 2, got {index_k.shape}")

        page_table = get_global_ctx().page_table
        rope_positions = self._batch_rope_positions(batch)
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        qo = md.qo_indptr_cpu.tolist()
        pending_keys = self.kvcache.pending_index_k_cache(layer_id)
        pending_positions = self.kvcache.pending_index_positions(layer_id)
        offsets = torch.arange(ratio, dtype=torch.long, device=self.device)

        for req_idx, req in enumerate(reqs):
            begin, end = qo[req_idx], qo[req_idx + 1]
            if begin == end:
                continue
            start = int(req.device_len) - int(req.extend_len)
            stop = int(req.device_len)
            if stop - start != end - begin:
                raise ValueError(
                    "QSA packed query span does not match request extend length: "
                    f"packed={end - begin}, logical={stop - start}"
                )
            table_idx = int(req.table_idx)
            current = index_k[begin:end]
            current_rope = (
                rope_positions[:, begin:end]
                if rope_positions.ndim == 2
                else rope_positions[begin:end]
            )

            first_group = (start // ratio) * ratio
            last_group = stop - ratio
            if first_group <= last_group:
                group_starts = torch.arange(
                    first_group,
                    last_group + 1,
                    ratio,
                    dtype=torch.long,
                    device=self.device,
                )
                group_tokens = group_starts[:, None] + offsets[None, :]
                current_mask = group_tokens >= start
                current_locs = (group_tokens - start).clamp_min(0)
                current_group_keys = current.index_select(
                    0, current_locs.reshape(-1)
                ).view(group_tokens.shape[0], ratio, -1)
                ring_locs = table_idx * ratio + torch.remainder(group_tokens, ratio)
                prior_group_keys = pending_keys[ring_locs]
                groups = torch.where(
                    current_mask[..., None], current_group_keys, prior_group_keys
                )
                pooled = groups.float().mean(dim=1).to(index_k.dtype)

                start_current_mask = group_starts >= start
                start_current_locs = (group_starts - start).clamp_min(0)
                prior_start_locs = table_idx * ratio + torch.remainder(
                    group_starts, ratio
                )
                if rope_positions.ndim == 2:
                    current_starts = current_rope.index_select(
                        1, start_current_locs
                    )
                    prior_starts = pending_positions[
                        prior_start_locs
                    ].transpose(0, 1)
                    transform_positions = torch.where(
                        start_current_mask.unsqueeze(0),
                        current_starts,
                        prior_starts,
                    )
                else:
                    current_starts = current_rope.index_select(
                        0, start_current_locs
                    )
                    prior_starts = pending_positions[prior_start_locs, 0]
                    transform_positions = torch.where(
                        start_current_mask, current_starts, prior_starts
                    )

                transformed = index_key_transform(pooled, transform_positions)
                if transformed.shape != pooled.shape:
                    raise ValueError(
                        "QSA compressed key transform must preserve shape, got "
                        f"{tuple(transformed.shape)} != {tuple(pooled.shape)}"
                    )
                full_locs = page_table[table_idx].index_select(0, group_starts)
                compressed_locs = full_locs.to(torch.long) // ratio
                self.kvcache.store_compressed_index_k(
                    transformed, compressed_locs, layer_id
                )

            pending_count = stop % ratio
            if pending_count:
                pending_start = max(start, stop - pending_count)
                local_start = pending_start - start
                logical = torch.arange(
                    pending_start,
                    stop,
                    dtype=torch.long,
                    device=self.device,
                )
                ring_locs = table_idx * ratio + torch.remainder(logical, ratio)
                self.kvcache.store_pending_index_k(
                    current[local_start:], ring_locs, layer_id
                )
                pending_rope = (
                    current_rope[:, local_start:]
                    if current_rope.ndim == 2
                    else current_rope[local_start:]
                )
                self.kvcache.store_pending_positions(
                    pending_rope, ring_locs, layer_id
                )

    def _gather_decode_compressed_keys(
        self, md: QSATritonMetadata, layer_id: int
    ) -> torch.Tensor:
        assert md.rows is not None and md.kv_lens is not None
        ratio = self.group.indexer_compress_ratio
        max_blocks = max(
            (int(length) // ratio for length in md.kv_len_cpu.tolist()),
            default=0,
        )
        compressed_locs = (
            md.rows[:, : max_blocks * ratio : ratio].to(torch.long) // ratio
        )
        return self.kvcache.compressed_index_k_cache(layer_id)[compressed_locs]

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


class QSASM120Backend(QSATritonBackend):
    """Native-256K QSA: compressed index cache plus bounded SM120 top-512."""

    requires_compressed_index = True
    # 512 query rows x 65,536 compressed blocks x fp32 = exactly
    # the pool-owned 128 MiB selector arena.
    prefill_query_chunk = 512


__all__ = ["QSASM120Backend", "QSATritonBackend", "QSATritonMetadata"]
