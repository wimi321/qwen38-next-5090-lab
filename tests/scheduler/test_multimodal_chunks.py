from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.multimodal import ImageTokenSpan, MMEmbeddingPlan
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.scheduler import Scheduler, _make_mrope_positions
from freetoken.scheduler.utils import PendingReq


def _req(*, cached_len: int, device_len: int, plan: MMEmbeddingPlan | None = None) -> Req:
    request = Req(
        input_ids=torch.arange(device_len, dtype=torch.int32),
        table_idx=0,
        cached_len=cached_len,
        output_len=2,
        uid=1,
        sampling_params=SamplingParams(max_tokens=2),
        cache_handle=SimpleNamespace(cached_len=0),
        mm_plan=plan,
    )
    return request


def test_embedding_plan_selects_only_intersections_across_chunks() -> None:
    features = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    plan = MMEmbeddingPlan.from_image_spans(
        features,
        [ImageTokenSpan(0, 3, 6), ImageTokenSpan(1, 9, 12)],
    )

    chunk = plan.select(4, 10)

    assert chunk.token_indices.tolist() == [0, 1, 5]
    assert chunk.placeholder_mask.tolist() == [True, True, False, False, False, True]
    assert torch.equal(chunk.features, features[[1, 2, 3]])


def test_scheduler_gathers_chunk_features_with_explicit_indices() -> None:
    features = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    plan = MMEmbeddingPlan.from_image_spans(features, [ImageTokenSpan(0, 2, 6)])
    req = _req(cached_len=4, device_len=8, plan=plan)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.device = torch.device("cpu")

    Scheduler._gather_multimodal(scheduler, batch)

    assert torch.equal(batch.mm_embeds, features[2:4])
    assert batch.mm_embed_indices.tolist() == [0, 1]
    assert batch.mm_placeholder_mask.tolist() == [True, True, False, False]


def test_mrope_prompt_slice_and_decode_delta() -> None:
    positions = torch.arange(18, dtype=torch.int64).reshape(3, 6)
    req = _req(cached_len=2, device_len=5)
    req.mrope_positions = positions
    req.mrope_delta = -2
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    sliced = _make_mrope_positions(batch, torch.device("cpu"))
    assert torch.equal(sliced, positions[:, 2:5])

    decode = _req(cached_len=6, device_len=7)
    decode.mrope_positions = positions
    decode.mrope_delta = -2
    batch = Batch(reqs=[decode], phase="decode")
    batch.padded_reqs = [decode]
    generated = _make_mrope_positions(batch, torch.device("cpu"))
    assert generated.tolist() == [[4], [4], [4]]


def test_finished_request_drops_all_multimodal_references_after_cache_decision() -> None:
    events: list[tuple[str, object]] = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = SimpleNamespace(
        cache_req=lambda req, *, finished: events.append(
            ("cache", (finished, req.is_multimodal))
        )
    )
    scheduler.table_manager = SimpleNamespace(
        free=lambda index: events.append(("free", index))
    )
    req = _req(cached_len=0, device_len=2)
    req.table_idx = 7
    req.mm_embeds = torch.ones(1, 2)
    req.image_inputs = object()
    req.mm_plan = object()
    req.mrope_positions = torch.ones(3, 2, dtype=torch.int64)

    Scheduler._free_req_resources(scheduler, req)

    assert events == [("cache", (True, True)), ("free", 7)]
    assert req.table_idx == -1
    assert req.mm_embeds is None
    assert req.image_inputs is None
    assert req.mm_plan is None
    assert req.mrope_positions is None


def test_first_chunk_transfers_multimodal_ownership_out_of_pending_request(
    monkeypatch,
) -> None:
    """A 256K continuation must not retain the consumed processor pixel tensor."""

    image_inputs = object()
    mrope_positions = torch.ones(3, 8, dtype=torch.int64)
    pending = PendingReq(
        uid=9,
        input_ids=torch.arange(8, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=2),
        image_inputs=image_inputs,
        mrope_positions=mrope_positions,
    )
    chunk = ChunkedReq(
        input_ids=pending.input_ids[:4],
        table_idx=3,
        cached_len=0,
        output_len=2,
        uid=9,
        cache_handle=SimpleNamespace(cached_len=0),
        sampling_params=pending.sampling_params,
        image_inputs=image_inputs,
        mrope_positions=mrope_positions,
    )
    manager = PrefillManager(
        cache_manager=object(),
        table_manager=object(),
        decode_manager=SimpleNamespace(inflight_tokens=0),
        pending_list=[pending],
    )
    monkeypatch.setattr(
        "freetoken.scheduler.prefill.PrefillAdder.try_add_one",
        lambda _self, candidate: chunk if candidate is pending else None,
    )

    batch = manager.schedule_next_batch(prefill_budget=4)

    assert batch is not None and batch.reqs == [chunk]
    assert pending.chunked_req is chunk
    assert chunk.image_inputs is image_inputs
    assert chunk.mrope_positions is mrope_positions
    assert pending.image_inputs is None
    assert pending.mm_plan is None
    assert pending.mm_embeds is None
    assert pending.mrope_positions is None
