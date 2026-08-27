"""CPU parity and streaming-state tests for Qwen4-Exp PLE primitives."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.ple import (
    ConcatenatedRowBank,
    PLEState,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpNGramHasher,
    Qwen4ExpPLELayer,
    SafetensorsMmapRowBank,
    SafetensorsRowShard,
    ShardedSafetensorsMmapRowBank,
    TensorRowBank,
)


def _tiny_hasher(*, heads_per_ngram: int = 2) -> Qwen4ExpNGramHasher:
    return Qwen4ExpNGramHasher(
        unigram_vocab_size=32,
        eos_token_id=2,
        ngram_vocab_size_base=17,
        ngram_size=3,
        heads_per_ngram=heads_per_ngram,
        seed=1234,
        make_vocab_size_divisible_by=1,
    )


def test_ngram_hash_matches_official_known_vector_and_streaming_state():
    hasher = _tiny_hasher()
    assert hasher.layer_multipliers.tolist() == [
        256496738022509279,
        251627398771002343,
        55314113489879221,
    ]
    assert hasher.layout.head_vocab_sizes == (17, 19, 23, 29)
    assert hasher.layout.head_offsets == (0, 17, 36, 59)

    tokens = torch.tensor([[5, 7, 11, 2, 13, 17]])
    expected = torch.tensor(
        [[
            [15, 27, 45, 77],
            [2, 25, 47, 73],
            [14, 32, 57, 63],
            [5, 20, 37, 62],
            [3, 21, 52, 71],
            [8, 19, 55, 71],
        ]]
    )
    full, full_history = hasher(tokens)
    torch.testing.assert_close(full, expected, rtol=0, atol=0)

    history = None
    chunks = []
    for chunk in (tokens[:, :2], tokens[:, 2:4], tokens[:, 4:]):
        ids, history = hasher(chunk, history)
        chunks.append(ids)
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=0, atol=0)
    torch.testing.assert_close(history, full_history, rtol=0, atol=0)
    assert history.tolist() == [[13, 17]]


def test_default_configuration_has_sixteen_heads():
    hasher = Qwen4ExpNGramHasher(
        unigram_vocab_size=32,
        eos_token_id=2,
        ngram_vocab_size_base=17,
        make_vocab_size_divisible_by=128,
    )
    assert hasher.num_heads == 16
    assert len(hasher.layout.head_vocab_sizes) == 16
    assert hasher.layout.padded_vocab_size % 128 == 0


def test_small_tensor_row_bank_embedding_lookup():
    hasher = _tiny_hasher()
    row_width = 3
    weight = torch.arange(
        hasher.layout.padded_vocab_size * row_width, dtype=torch.float32
    ).reshape(hasher.layout.padded_vocab_size, row_width)
    embedding = Qwen4ExpNGramEmbedding(hasher, TensorRowBank(weight))
    input_ids = torch.tensor([[5, 7, 11]])

    row_ids, expected_history = hasher(input_ids)
    expected = weight[row_ids].flatten(-2)
    actual, history = embedding(input_ids)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(history, expected_history)
    assert actual.shape == (1, 3, hasher.num_heads * row_width)


def test_deferred_row_bank_rebind_validates_shape():
    import pytest

    hasher = _tiny_hasher()
    initial = TensorRowBank(torch.zeros(hasher.layout.padded_vocab_size, 2))
    embedding = Qwen4ExpNGramEmbedding(hasher, initial)
    replacement = TensorRowBank(torch.ones(hasher.layout.padded_vocab_size, 2))
    embedding.bind_row_bank(replacement)
    assert embedding.row_bank is replacement
    with pytest.raises(ValueError, match="row bank width"):
        embedding.bind_row_bank(
            TensorRowBank(torch.zeros(hasher.layout.padded_vocab_size, 3))
        )


def test_concatenated_row_bank_routes_across_part_boundaries():
    first = TensorRowBank(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    second = TensorRowBank(torch.tensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]))
    bank = ConcatenatedRowBank([first, second])

    actual = bank.read_rows(torch.tensor([[4, 1, 2, 4]]))
    expected = torch.tensor([[[8.0, 9.0], [2.0, 3.0], [4.0, 5.0], [8.0, 9.0]]])
    torch.testing.assert_close(actual, expected)
    assert bank.row_count == 5
    assert bank.row_width == 2


def _make_tiny_ple() -> Qwen4ExpPLELayer:
    hasher = _tiny_hasher()
    row_width = 2
    torch.manual_seed(4)
    table = torch.randn(hasher.layout.padded_vocab_size, row_width)
    embedding = Qwen4ExpNGramEmbedding(hasher, TensorRowBank(table))
    layer = Qwen4ExpPLELayer(
        embedding,
        hidden_size=4,
        hc_count=2,
        conv_kernel_size=3,
        rms_norm_eps=1e-6,
    )
    with torch.no_grad():
        for index, tensor in enumerate(layer.state_dict().values(), start=1):
            tensor.copy_(
                torch.linspace(-0.1 * index, 0.1 * index, tensor.numel()).reshape_as(tensor)
            )
    return layer


def test_ple_explicit_token_and_conv_state_matches_full_sequence():
    layer = _make_tiny_ple()
    torch.manual_seed(9)
    hidden = torch.randn(2, 7, 8)
    tokens = torch.tensor(
        [[5, 7, 11, 13, 17, 19, 23], [3, 2, 5, 7, 2, 11, 13]]
    )

    full, full_state = layer.forward(hidden, tokens)
    chunks = []
    state: PLEState | None = None
    for start, end in ((0, 2), (2, 3), (3, 7)):
        output, state = layer.forward(hidden[:, start:end], tokens[:, start:end], state)
        chunks.append(output)

    torch.testing.assert_close(torch.cat(chunks, dim=1), full, rtol=1e-5, atol=1e-6)
    assert state is not None
    torch.testing.assert_close(state.token_history, full_state.token_history)
    torch.testing.assert_close(state.conv_state, full_state.conv_state)
    assert state.conv_state.shape == (2, 8, 6)  # (kernel - 1) * dilation


def test_ple_full_sequence_matches_transformers_reference():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    config = Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        hc_count=2,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=1,
        seed=1234,
        eos_token_id=2,
        layer_types=["linear_attention", "linear_attention"],
    )
    torch.manual_seed(12)
    reference = Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0)
    ref_embedding = reference.ple_embedding.ngram_embedding.weight.detach().clone()
    hasher = _tiny_hasher()
    ours = Qwen4ExpPLELayer(
        Qwen4ExpNGramEmbedding(hasher, TensorRowBank(ref_embedding)),
        hidden_size=4,
        hc_count=2,
        conv_kernel_size=3,
        rms_norm_eps=config.rms_norm_eps,
    )
    with torch.no_grad():
        ours.key_proj.weight.copy_(reference.key_proj.weight)
        ours.value_proj.weight.copy_(reference.value_proj.weight)
        ours.norm_key.weight.copy_(reference.norm_key.weight)
        ours.norm_query.weight.copy_(reference.norm_query.weight)
        ours.norm_conv.weight.copy_(reference.norm_conv.weight)
        ours.conv1d.weight.copy_(reference.conv1d.weight)

    hidden = torch.randn(2, 6, 8)
    tokens = torch.tensor([[5, 7, 11, 2, 13, 17], [3, 5, 7, 11, 13, 17]])
    expected = reference(hidden, tokens, past_key_values=None)
    actual, _ = ours.forward(hidden, tokens)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_baseop_parameter_keys_align_with_hf_ple_names():
    assert set(_make_tiny_ple().state_dict()) == {
        "key_proj.weight",
        "value_proj.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
        "conv1d.weight",
    }


def test_forward_flat_matches_explicit_per_request_calls_and_stages_pool():
    from freetoken.kvcache.ple_state_pool import PLEStatePool

    layer = _make_tiny_ple()
    reqs = [
        SimpleNamespace(
            input_ids=torch.tensor([5, 7, 11]),
            table_idx=0,
            cached_len=0,
            device_len=3,
            extend_len=3,
        ),
        SimpleNamespace(
            input_ids=torch.tensor([3, 2]),
            table_idx=1,
            cached_len=0,
            device_len=2,
            extend_len=2,
        ),
    ]
    batch = SimpleNamespace(reqs=reqs, padded_reqs=reqs)
    hidden = torch.randn(5, 8)
    expected = [
        layer.forward(hidden[:3].unsqueeze(0), reqs[0].input_ids.unsqueeze(0)),
        layer.forward(hidden[3:].unsqueeze(0), reqs[1].input_ids.unsqueeze(0)),
    ]
    pool = PLEStatePool(
        num_slots=3,
        context_len=2,
        channels=8,
        conv_state_len=6,
        eos_token_id=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    actual = layer.forward_flat(hidden, batch, pool)

    torch.testing.assert_close(
        actual, torch.cat([item[0][0] for item in expected]), rtol=1e-5, atol=1e-6
    )
    assert pool.initialized[:2].tolist() == [False, False]
    assert pool.token_history[:2].tolist() == [[2, 2], [2, 2]]
    assert torch.count_nonzero(pool.conv_states[:2]) == 0
    for slot, (_, state) in enumerate(expected):
        assert state.conv_state is not None
        torch.testing.assert_close(pool.pending_conv_states[slot], state.conv_state[0])

    layer.commit_batch(batch, pool)
    assert pool.initialized[:2].tolist() == [True, True]
    assert pool.token_history[0].tolist() == [7, 11]
    assert pool.token_history[1].tolist() == [3, 2]


def test_forward_flat_prefetches_next_chunk_with_streaming_history():
    from freetoken.kvcache.ple_state_pool import PLEStatePool

    layer = _make_tiny_ple()
    captured: list[torch.Tensor] = []
    handle = object()

    class TrackingBank(TensorRowBank):
        def prefetch_rows(self, indices):
            captured.append(torch.as_tensor(indices).clone())
            return (handle,)

    original_bank = layer.ple_embedding.row_bank
    layer.ple_embedding.bind_row_bank(TrackingBank(original_bank.weight.clone()))
    full_prompt = torch.tensor([5, 7, 11, 13, 17, 19])
    req = SimpleNamespace(
        input_ids=full_prompt[:3],
        ple_prefetch_input_ids=full_prompt,
        ple_prefetch_handles=(),
        table_idx=0,
        cached_len=0,
        device_len=3,
        extend_len=3,
    )
    batch = SimpleNamespace(reqs=[req], padded_reqs=[req], is_decode=False)
    pool = PLEStatePool(
        num_slots=2,
        context_len=2,
        channels=8,
        conv_state_len=6,
        eos_token_id=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    layer.forward_flat(torch.randn(3, 8), batch, pool)

    _, history = layer.ple_embedding.hasher(full_prompt[:3].reshape(1, -1))
    expected, _ = layer.ple_embedding.hasher(
        full_prompt[3:].reshape(1, -1), history
    )
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected)
    assert req.ple_prefetch_handles == (handle,)


def test_prefill_downstream_failure_is_retryable_for_ragged_fresh_and_continued_slots():
    from freetoken.kvcache.ple_state_pool import PLEStatePool

    layer = _make_tiny_ple()
    pool = PLEStatePool(
        num_slots=3,
        context_len=2,
        channels=8,
        conv_state_len=6,
        eos_token_id=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # Slot 0 continues a real prefix. Slot 1 deliberately contains stale state
    # from an old request, but cached_len=0 must make the new request see reset
    # state without clearing the committed slot on a failed forward.
    prefix_hidden = torch.randn(1, 2, 8)
    prefix_tokens = torch.tensor([[5, 7]])
    _, continued_state = layer.forward(prefix_hidden, prefix_tokens)
    assert continued_state.token_history is not None
    assert continued_state.conv_state is not None
    pool.commit(0, continued_state.token_history[0], continued_state.conv_state[0])
    stale_history = torch.tensor([17, 19])
    stale_conv = torch.full((8, 6), 23.0)
    pool.commit(1, stale_history, stale_conv)

    reqs = [
        SimpleNamespace(
            input_ids=torch.tensor([5, 7, 11, 13]),
            table_idx=0,
            cached_len=2,
            device_len=4,
            extend_len=2,
        ),
        SimpleNamespace(
            input_ids=torch.tensor([3, 2, 29]),
            table_idx=1,
            cached_len=0,
            device_len=3,
            extend_len=3,
        ),
    ]
    batch = SimpleNamespace(reqs=reqs, padded_reqs=reqs, is_decode=False)
    hidden = torch.randn(5, 8)
    expected_continued = layer.forward(
        hidden[:2].unsqueeze(0),
        reqs[0].input_ids[2:].unsqueeze(0),
        continued_state,
    )
    expected_fresh = layer.forward(
        hidden[2:].unsqueeze(0), reqs[1].input_ids.unsqueeze(0)
    )
    expected_output = torch.cat([expected_continued[0][0], expected_fresh[0][0]])
    committed_history = pool.token_history.clone()
    committed_conv = pool.conv_states.clone()
    committed_initialized = pool.initialized.clone()

    failed_outputs = []

    def forward_then_fail_downstream():
        failed_outputs.append(layer.forward_flat(hidden, batch, pool))
        raise RuntimeError("synthetic downstream layer failure")

    with pytest.raises(RuntimeError, match="downstream layer failure"):
        forward_then_fail_downstream()

    first_output = failed_outputs[0]
    torch.testing.assert_close(first_output, expected_output, rtol=1e-5, atol=1e-6)
    # The later layer failed, so no engine success hook was invoked.
    torch.testing.assert_close(pool.token_history, committed_history)
    torch.testing.assert_close(pool.conv_states, committed_conv)
    torch.testing.assert_close(pool.initialized, committed_initialized)

    retry_output = layer.forward_flat(hidden, batch, pool)
    torch.testing.assert_close(retry_output, first_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pool.token_history, committed_history)
    torch.testing.assert_close(pool.conv_states, committed_conv)

    layer.commit_batch(batch, pool)
    for slot, (_, state) in enumerate((expected_continued, expected_fresh)):
        assert state.token_history is not None and state.conv_state is not None
        torch.testing.assert_close(pool.token_history[slot], state.token_history[0])
        torch.testing.assert_close(pool.conv_states[slot], state.conv_state[0])
    assert pool.initialized[:2].tolist() == [True, True]


def test_decode_staging_matches_explicit_state_and_defers_all_pool_commit():
    from freetoken.kvcache.ple_state_pool import PLEStatePool

    layer = _make_tiny_ple()
    pool = PLEStatePool(
        num_slots=3,
        context_len=2,
        channels=8,
        conv_state_len=6,
        eos_token_id=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    prefixes = [torch.tensor([[5, 7]]), torch.tensor([[3, 2]])]
    prefix_hidden = [torch.randn(1, 2, 8), torch.randn(1, 2, 8)]
    states = []
    for slot, (hidden, tokens) in enumerate(zip(prefix_hidden, prefixes)):
        _, state = layer.forward(hidden, tokens)
        assert state.token_history is not None and state.conv_state is not None
        pool.commit(slot, state.token_history[0], state.conv_state[0])
        states.append(state)

    reqs = [
        SimpleNamespace(
            input_ids=torch.tensor([5, 7, 11]),
            table_idx=0,
            cached_len=2,
            device_len=3,
            extend_len=1,
        ),
        SimpleNamespace(
            input_ids=torch.tensor([3, 2, 13]),
            table_idx=1,
            cached_len=2,
            device_len=3,
            extend_len=1,
        ),
    ]
    batch = SimpleNamespace(
        reqs=reqs,
        padded_reqs=reqs,
        is_decode=True,
        size=2,
    )
    decode_hidden = torch.randn(2, 8)
    expected_parts = []
    expected_states = []
    for index, req in enumerate(reqs):
        output, state = layer.forward(
            decode_hidden[index].reshape(1, 1, 8),
            req.input_ids[-1:].reshape(1, 1),
            states[index],
        )
        expected_parts.append(output[0, 0])
        expected_states.append(state)

    histories_before = pool.token_history.clone()
    conv_before = pool.conv_states.clone()
    layer.stage_decode_batch(
        batch,
        pool,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(pool.token_history, histories_before)

    actual = layer.forward_flat(decode_hidden, batch, pool)
    torch.testing.assert_close(actual, torch.stack(expected_parts), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pool.conv_states, conv_before)
    for slot, state in enumerate(expected_states):
        assert state.conv_state is not None
        torch.testing.assert_close(pool.pending_conv_states[slot], state.conv_state[0])

    layer.commit_decode_batch(batch, pool)
    for slot, state in enumerate(expected_states):
        assert state.token_history is not None and state.conv_state is not None
        torch.testing.assert_close(pool.token_history[slot], state.token_history[0])
        torch.testing.assert_close(pool.conv_states[slot], state.conv_state[0])
    assert batch.ple_next_token_history is None


def test_decode_downstream_failure_leaves_committed_pool_unchanged_until_retry_succeeds():
    from freetoken.kvcache.ple_state_pool import PLEStatePool

    layer = _make_tiny_ple()
    pool = PLEStatePool(
        num_slots=2,
        context_len=2,
        channels=8,
        conv_state_len=6,
        eos_token_id=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    prefix_tokens = torch.tensor([[5, 7]])
    prefix_hidden = torch.randn(1, 2, 8)
    _, prefix_state = layer.forward(prefix_hidden, prefix_tokens)
    assert prefix_state.token_history is not None and prefix_state.conv_state is not None
    pool.commit(0, prefix_state.token_history[0], prefix_state.conv_state[0])

    req = SimpleNamespace(
        input_ids=torch.tensor([5, 7, 11]),
        table_idx=0,
        cached_len=2,
        device_len=3,
        extend_len=1,
    )
    batch = SimpleNamespace(
        reqs=[req],
        padded_reqs=[req],
        is_decode=True,
        size=1,
        input_ids=torch.tensor([11]),
    )
    decode_hidden = torch.randn(1, 8)
    expected_output, expected_state = layer.forward(
        decode_hidden.reshape(1, 1, 8),
        req.input_ids[-1:].reshape(1, 1),
        prefix_state,
    )
    committed_history = pool.token_history.clone()
    committed_conv = pool.conv_states.clone()

    layer.stage_decode_batch(
        batch, pool, device=torch.device("cpu"), dtype=torch.float32
    )

    def forward_then_fail_downstream():
        layer.forward_flat(decode_hidden, batch, pool)
        raise RuntimeError("synthetic downstream layer failure")

    with pytest.raises(RuntimeError, match="downstream layer failure"):
        forward_then_fail_downstream()

    # No Engine/model success hook ran: both committed state components remain
    # at the prefix even though the device-side pending bank has a decode result.
    torch.testing.assert_close(pool.token_history, committed_history)
    torch.testing.assert_close(pool.conv_states, committed_conv)
    assert expected_state.conv_state is not None
    torch.testing.assert_close(pool.pending_conv_states[0], expected_state.conv_state[0])

    # A retry stages from committed state, produces the same result, and publishes
    # token history plus convolution state together on the success hook.
    layer.stage_decode_batch(
        batch, pool, device=torch.device("cpu"), dtype=torch.float32
    )
    retry_output = layer.forward_flat(decode_hidden, batch, pool)
    torch.testing.assert_close(retry_output, expected_output[0], rtol=1e-5, atol=1e-6)
    layer.commit_decode_batch(batch, pool)

    assert expected_state.token_history is not None
    torch.testing.assert_close(pool.token_history[0], expected_state.token_history[0])
    torch.testing.assert_close(pool.conv_states[0], expected_state.conv_state[0])


def test_decode_staging_uses_batch_snapshot_while_req_host_view_lags():
    from freetoken.kvcache.ple_state_pool import PLEStatePool

    layer = _make_tiny_ple()
    pool = PLEStatePool(
        num_slots=2,
        context_len=2,
        channels=8,
        conv_state_len=6,
        eos_token_id=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool.commit(
        0,
        torch.tensor([5, 7]),
        torch.zeros(8, 6, dtype=torch.float32),
    )
    # Engine.complete_one() has advanced the logical device length, but overlap
    # scheduling has not yet appended sampled token 11 to the host Req view.
    req = SimpleNamespace(
        input_ids=torch.tensor([5, 7]),
        table_idx=0,
        cached_len=2,
        device_len=3,
        extend_len=1,
    )
    batch = SimpleNamespace(
        reqs=[req],
        padded_reqs=[req],
        is_decode=True,
        size=1,
        input_ids=torch.tensor([11]),
    )

    layer.stage_decode_batch(
        batch, pool, device=torch.device("cpu"), dtype=torch.float32
    )
    assert torch.equal(batch.ple_next_token_history, torch.tensor([[7, 11]]))

    del batch.input_ids
    with pytest.raises(ValueError, match="host token view is not drained"):
        layer.stage_decode_batch(
            batch, pool, device=torch.device("cpu"), dtype=torch.float32
        )


def _write_safetensors(path, tensors):
    """Write the tiny subset of the safetensors format needed by mmap tests."""

    header = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_sharded_mmap_bank_maps_global_rows_deduplicates_and_uses_external_scale(tmp_path):
    part0 = tmp_path / "part0.safetensors"
    part1 = tmp_path / "part1.safetensors"
    scale_path = tmp_path / "scale.safetensors"
    _write_safetensors(
        part0,
        [("shard_0.weight", "F8_E4M3", (2, 2), bytes([0x38, 0x40, 0x44, 0x48]))],
    )
    _write_safetensors(
        part1,
        [
            (
                "shard_1.weight",
                "F8_E4M3",
                (3, 2),
                bytes([0x30, 0x38, 0x40, 0x44, 0x48, 0x4A]),
            )
        ],
    )
    _write_safetensors(
        scale_path,
        [("weight_scale", "F32", (1,), struct.pack("<f", 0.5))],
    )
    specs = [
        SafetensorsRowShard(part0, "shard_0.weight"),
        SafetensorsRowShard(part1, "shard_1.weight"),
    ]

    with ShardedSafetensorsMmapRowBank(
        specs,
        weight_scale_path=scale_path,
        weight_scale_name="weight_scale",
        default_dtype=torch.float32,
    ) as bank:
        actual = bank.read_rows(torch.tensor([[4, 1, 2, 4]]))
        expected = torch.tensor([[[2.0, 2.5], [1.5, 2.0], [0.25, 0.5], [2.0, 2.5]]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert bank.row_count == 5
        assert bank.row_width == 2
        # Three unique global rows, two FP8 bytes per row. Row 4 occurs twice.
        assert bank.payload_bytes_read == 6
    assert bank.closed


def test_safetensors_mmap_reads_only_unique_fp8_rows_and_applies_scale(tmp_path):
    # E4M3FN encodings for exact values:
    # [1, 2, 3, 4], [-1, -2, -3, -4], [0.5, 1.5, 0, 448], ...
    raw_weight = bytes([
        0x38, 0x40, 0x44, 0x48,
        0xB8, 0xC0, 0xC4, 0xC8,
        0x30, 0x3C, 0x00, 0x7E,
        0x40, 0x40, 0x40, 0x40,
        0x44, 0x44, 0x44, 0x44,
    ])
    scales = struct.pack("<5f", 0.5, 2.0, 0.25, 4.0, 0.125)
    path = tmp_path / "rows.safetensors"
    _write_safetensors(
        path,
        [
            ("weight", "F8_E4M3", (5, 4), raw_weight),
            ("scale", "F32", (5,), scales),
        ],
    )

    with SafetensorsMmapRowBank(path, "weight", scale_name="scale") as bank:
        assert bank.readonly
        actual = bank.read_rows(torch.tensor([[4, 1, 4]]), dtype=torch.bfloat16)
        expected = torch.tensor(
            [[[0.375, 0.375, 0.375, 0.375], [-2, -4, -6, -8], [0.375] * 4]],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Duplicate row 4 is fetched once: 2 * (4 FP8 bytes + 1 FP32 scale).
        assert bank.payload_bytes_read == 16
        assert bank.payload_bytes_read < 40
    assert bank.closed


def test_safetensors_mmap_supports_per_block_scales(tmp_path):
    raw_weight = bytes([0x38, 0x40, 0x44, 0x48, 0x30, 0x38, 0x40, 0x44])
    scales = struct.pack("<4f", 2.0, 0.5, 4.0, 0.25)
    path = tmp_path / "blocks.safetensors"
    _write_safetensors(
        path,
        [
            ("weight", "F8_E4M3", (2, 4), raw_weight),
            ("scale", "F32", (2, 2), scales),
        ],
    )

    with SafetensorsMmapRowBank(path, "weight", scale_name="scale") as bank:
        actual = bank.read_rows([1, 0])
    expected = torch.tensor([[2.0, 4.0, 0.5, 0.75], [2.0, 4.0, 1.5, 2.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
