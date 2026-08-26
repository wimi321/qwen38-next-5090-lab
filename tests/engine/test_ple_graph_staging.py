from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.graph import GraphCaptureBuffer
from freetoken.kvcache.ple_state_pool import PLEStatePool


def test_graph_capture_buffer_copies_ple_inputs_then_exposes_stable_slices():
    buffer = GraphCaptureBuffer.init(
        4,
        11,
        torch.device("cpu"),
        ple_embedding_dim=3,
        ple_dtype=torch.float32,
    )
    expected_embeddings = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected_slots = torch.tensor([7, 2], dtype=torch.long)
    expected_reset = torch.tensor([False, True])
    batch = SimpleNamespace(
        padded_size=2,
        ple_embeddings=expected_embeddings.clone(),
        ple_table_idx=expected_slots.clone(),
        ple_reset_mask=expected_reset.clone(),
    )

    buffer.copy_auxiliary_from(batch)
    buffer.set_batch(batch)

    torch.testing.assert_close(batch.ple_embeddings, expected_embeddings)
    torch.testing.assert_close(batch.ple_table_idx, expected_slots)
    torch.testing.assert_close(batch.ple_reset_mask, expected_reset)
    assert batch.ple_embeddings.data_ptr() == buffer.ple_embeddings.data_ptr()
    assert batch.ple_table_idx.data_ptr() == buffer.ple_table_idx.data_ptr()
    assert batch.ple_reset_mask.data_ptr() == buffer.ple_reset_mask.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ple_state_graph_replay_writes_pending_then_commits_dynamic_slots_outside_graph():
    device = torch.device("cuda")
    pool = PLEStatePool(
        num_slots=4,
        context_len=2,
        channels=2,
        conv_state_len=3,
        eos_token_id=2,
        dtype=torch.float32,
        device=device,
    )
    pool.conv_states.copy_(
        torch.arange(24, dtype=torch.float32, device=device).reshape(4, 2, 3)
    )
    table_idx = torch.tensor([0, 1], dtype=torch.long, device=device)
    reset_mask = torch.zeros(2, dtype=torch.bool, device=device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gathered = pool.gather_decode_conv_states(table_idx, reset_mask)
        pool.write_pending_decode_conv_states(table_idx, gathered + 100.0)

    # Replay against different slot/reset values at the same input addresses.
    baseline = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 2, 3)
    pool.conv_states.copy_(baseline)
    table_idx.copy_(torch.tensor([2, 3], dtype=torch.long, device=device))
    reset_mask.copy_(torch.tensor([False, True], dtype=torch.bool, device=device))
    graph.replay()
    torch.cuda.synchronize()

    # Replay is transactional: the captured graph can update a dynamic pending
    # slot without publishing committed request state.
    torch.testing.assert_close(pool.conv_states, baseline)
    torch.testing.assert_close(pool.pending_conv_states[2], baseline[2] + 100.0)
    torch.testing.assert_close(
        pool.pending_conv_states[3], torch.full_like(baseline[3], 100.0)
    )

    pool.commit_pending_decode_conv_states(table_idx)
    torch.testing.assert_close(pool.conv_states[2], baseline[2] + 100.0)
    torch.testing.assert_close(pool.conv_states[3], torch.full_like(baseline[3], 100.0))
    torch.testing.assert_close(pool.conv_states[:2], baseline[:2])
