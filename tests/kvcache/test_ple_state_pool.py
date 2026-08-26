from __future__ import annotations

import torch

from freetoken.kvcache.ple_state_pool import PLEStatePool


def test_ple_state_pool_clear_commit_and_slot_isolation():
    pool = PLEStatePool(
        num_slots=3,
        context_len=2,
        channels=4,
        conv_state_len=6,
        eos_token_id=9,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    history, conv = pool.begin(1, fresh=True)
    assert history.tolist() == [9, 9]
    assert torch.count_nonzero(conv) == 0

    pool.commit(1, torch.tensor([3, 5]), torch.ones(4, 6))
    history, conv = pool.begin(1, fresh=False)
    assert history.tolist() == [3, 5]
    assert torch.all(conv == 1)
    assert not pool.initialized[0]

    # A fresh request observes logical reset state without destroying the
    # previous request's committed slot before the model transaction succeeds.
    fresh_history, fresh_conv = pool.begin(1, fresh=True)
    assert fresh_history.tolist() == [9, 9]
    assert torch.count_nonzero(fresh_conv) == 0
    assert pool.token_history[1].tolist() == [3, 5]
    assert torch.all(pool.conv_states[1] == 1)
    assert pool.initialized[1]

    pool.clear(1)
    assert pool.token_history[1].tolist() == [9, 9]
    assert torch.count_nonzero(pool.conv_states[1]) == 0
    assert not pool.initialized[1]


def test_ple_state_pool_staging_defers_history_and_conv_state_until_commit():
    pool = PLEStatePool(
        num_slots=4,
        context_len=2,
        channels=2,
        conv_state_len=3,
        eos_token_id=9,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool.commit(1, torch.tensor([3, 5]), torch.full((2, 3), 11.0))
    pool.commit(2, torch.tensor([7, 8]), torch.full((2, 3), 22.0))

    history, reset = pool.stage_token_history(1, fresh=False)
    assert history.tolist() == [3, 5]
    assert not reset
    fresh_history, fresh_reset = pool.stage_token_history(2, fresh=True)
    assert fresh_history.tolist() == [9, 9]
    assert fresh_reset
    # Staging is transactional: it does not publish the pending history or
    # eagerly destroy the recurrent GPU state.
    assert pool.token_history[2].tolist() == [7, 8]
    assert torch.all(pool.conv_states[2] == 22)

    table_idx = torch.tensor([1, 2], dtype=torch.long)
    reset_mask = torch.tensor([False, True])
    gathered = pool.gather_decode_conv_states(table_idx, reset_mask)
    assert torch.all(gathered[0] == 11)
    assert torch.count_nonzero(gathered[1]) == 0

    next_state = torch.stack(
        [torch.full((2, 3), 31.0), torch.full((2, 3), 32.0)]
    )
    pool.write_pending_decode_conv_states(table_idx, next_state)
    assert torch.all(pool.conv_states[1] == 11)
    assert torch.all(pool.conv_states[2] == 22)
    assert torch.all(pool.pending_conv_states[1] == 31)
    assert torch.all(pool.pending_conv_states[2] == 32)

    histories = torch.tensor([[5, 6], [9, 4]])
    pool.commit_pending_decode(table_idx, (1, 2), histories)
    assert torch.all(pool.conv_states[1] == 31)
    assert torch.all(pool.conv_states[2] == 32)
    assert pool.token_history[1].tolist() == [5, 6]
    assert pool.token_history[2].tolist() == [9, 4]
    assert pool.initialized[1:3].tolist() == [True, True]
