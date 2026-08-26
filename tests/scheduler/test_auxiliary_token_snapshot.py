from types import SimpleNamespace

import torch

from freetoken.core import Batch
from freetoken.scheduler.scheduler import Scheduler


def test_prepare_auxiliary_and_forward_share_token_pool_snapshot(monkeypatch):
    import freetoken.scheduler.scheduler as scheduler_module

    req = SimpleNamespace(
        table_idx=0,
        cached_len=2,
        device_len=3,
        extend_len=1,
        decode_batch_idx=0,
    )
    batch = Batch(reqs=[req], phase="decode")
    token_pool = torch.zeros((1, 5), dtype=torch.int32)
    token_pool[0, 2] = 11
    observed = {}

    def pad_batch(value):
        value.padded_reqs = value.reqs

    def prepare_auxiliary(value):
        observed["staged"] = value.input_ids.clone()

    def forward_batch(value, _sample_args):
        observed["forward"] = value.input_ids.clone()
        return SimpleNamespace(next_tokens_gpu=torch.tensor([42], dtype=torch.int32))

    engine = SimpleNamespace(
        graph_runner=SimpleNamespace(pad_batch=pad_batch),
        model=SimpleNamespace(prepare_batch_auxiliary=prepare_auxiliary),
        page_table=torch.zeros((1, 5), dtype=torch.int32),
        linear_state_pool=None,
        attn_backend=SimpleNamespace(prepare_metadata=lambda _batch: None),
        sampler=SimpleNamespace(prepare=lambda _batch: object()),
        forward_batch=forward_batch,
    )
    scheduler = SimpleNamespace(
        engine=engine,
        device=torch.device("cpu"),
        token_pool=token_pool,
        cache_manager=SimpleNamespace(
            maybe_free_swa_out_of_window=lambda *_args, **_kwargs: None,
            allocate_paged=lambda _reqs: None,
        ),
        decode_manager=SimpleNamespace(filter_reqs=lambda _reqs: None),
        _forward_iter=0,
        toolcall_anchor_id=None,
    )
    monkeypatch.setattr(
        scheduler_module,
        "_make_positions",
        lambda _batch, _device: torch.tensor([2], dtype=torch.int32),
    )
    monkeypatch.setattr(
        scheduler_module,
        "_make_input_tuple",
        lambda _batch, _device: (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        scheduler_module,
        "_make_write_tuple",
        lambda _batch, _device: (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        ),
    )

    forward_input = Scheduler._prepare_batch(scheduler, batch)
    assert torch.equal(observed["staged"], torch.tensor([11], dtype=torch.int32))

    # Mutating the token pool after preparation proves _forward does not gather a
    # second, potentially different token after PLE has already hashed the first.
    token_pool[0, 2] = 99
    Scheduler._forward(scheduler, forward_input)
    assert torch.equal(observed["forward"], observed["staged"])
    assert token_pool[0, 3].item() == 42
