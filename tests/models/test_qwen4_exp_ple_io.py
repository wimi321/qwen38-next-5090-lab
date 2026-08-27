from __future__ import annotations

import os
import importlib.util
import json
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from freetoken.models.qwen4_exp.ple import (
    SafetensorsRowShard,
    ShardedSafetensorsMmapRowBank,
)
from freetoken.models.qwen4_exp.ple_io import (
    DirectIOPageReader,
    IoUringPageReader,
    PLEStreamingUnavailable,
    probe_ple_streaming_capability,
    require_production_ple_streaming,
)


def test_native_prefetch_splits_more_sparse_spans_than_ring_queue_depth():
    """One pathological chunk must batch instead of failing at 512 spans."""

    class FakeNative:
        closed = False

        @staticmethod
        def read_many(spans):
            return tuple(f"{offset}:{length}".encode() for offset, length in spans)

    reader = IoUringPageReader.__new__(IoUringPageReader)
    reader.queue_depth = 2
    reader._native = FakeNative()
    reader._executor = ThreadPoolExecutor(max_workers=1)
    reader._lock = threading.RLock()
    reader._outstanding = set()
    reader._outstanding_spans = {}
    reader._prefetch_submitted = 0
    reader._prefetch_completed = 0
    reader._prefetch_cancelled = 0
    reader._prefetch_errors = 0
    spans = tuple((index * 4096, 64) for index in range(5))
    try:
        handle = reader.prefetch(spans)
        assert handle.wait() == tuple(
            f"{offset}:{length}".encode() for offset, length in spans
        )
        assert reader._prefetch_submitted == 5
    finally:
        reader._executor.shutdown(wait=True, cancel_futures=True)


def test_capability_never_labels_missing_native_extension_ready(tmp_path, monkeypatch):
    payload = tmp_path / "model.safetensors"
    payload.write_bytes(b"x" * 8192)
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.ple_io.importlib.util.find_spec",
        lambda name: None,
    )
    capability = probe_ple_streaming_capability(payload)
    assert not capability.native_extension
    assert not capability.production_ready
    assert "native extension" in capability.detail
    with pytest.raises(PLEStreamingUnavailable, match="native io_uring"):
        require_production_ple_streaming(payload)


def test_sharded_direct_bank_splits_one_global_lru_budget(monkeypatch, tmp_path):
    """The 4 GiB profile value must not be multiplied by 128 PLE shards."""

    capacities: list[int] = []

    class FakeDirectBank:
        def __init__(
            self,
            path,
            tensor_name,
            *,
            scale_name=None,
            default_dtype=None,
            cache_capacity_bytes,
            queue_depth,
            max_batch_pages,
        ):
            del path, tensor_name, scale_name, default_dtype, queue_depth, max_batch_pages
            capacities.append(cache_capacity_bytes)
            self.row_width = 4
            self.row_count = 8
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.ple.SafetensorsDirectRowBank",
        FakeDirectBank,
    )
    specs = [
        SafetensorsRowShard(tmp_path / f"part-{index}.safetensors", f"shard_{index}")
        for index in range(3)
    ]
    with ShardedSafetensorsMmapRowBank(
        specs,
        weight_scale=1.0,
        io_backend="direct_debug",
        cache_capacity_bytes=10,
    ) as bank:
        assert bank.cache_capacity_bytes == 10
        assert capacities == [4, 3, 3]
        assert sum(capacities) == 10


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv"),
    reason="requires Linux O_DIRECT",
)
def test_direct_reader_unaligned_lru_prefetch_and_telemetry(tmp_path: Path):
    path = tmp_path / "payload.bin"
    content = bytes(range(256)) * 64
    path.write_bytes(content)
    try:
        reader = DirectIOPageReader(
            path,
            cache_capacity_bytes=4096,
            queue_depth=2,
            max_batch_pages=2,
        )
    except PLEStreamingUnavailable as exc:
        pytest.skip(str(exc))
    except OSError as exc:
        pytest.skip(f"test filesystem does not support O_DIRECT: {exc}")
    with reader:
        assert reader.read(37, 5000) == content[37:5037]
        before = reader.telemetry()
        # With a one-page LRU, the first page was evicted and the second remains hot.
        assert reader.read(4096, 128) == content[4096:4224]
        after = reader.telemetry()
        assert after.cache_hit_pages > before.cache_hit_pages
        handle = reader.prefetch(((8192, 256),))
        assert handle.wait() == (content[8192:8448],)
        telemetry = reader.telemetry()
        assert telemetry.prefetch_submitted == 1
        assert telemetry.prefetch_completed == 1
        assert telemetry.cache_bytes <= telemetry.cache_capacity_bytes
        assert telemetry.storage_bytes > 0


def test_direct_reader_validates_production_geometry_before_open(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"x" * 4096)
    if sys.platform != "linux":
        with pytest.raises(PLEStreamingUnavailable):
            DirectIOPageReader(path)
        return
    with pytest.raises(ValueError, match="power of two"):
        DirectIOPageReader(path, alignment=1000)
    with pytest.raises(ValueError, match="max_batch_pages"):
        DirectIOPageReader(path, max_batch_pages=4097)


@pytest.mark.skipif(
    importlib.util.find_spec(
        "freetoken.models.qwen4_exp._ple_io_uring"
    ) is None,
    reason="bundled native io_uring extension is not built",
)
def test_native_io_uring_real_submit_completion_odirect_and_lru(tmp_path: Path):
    path = tmp_path / "native-payload.bin"
    content = bytes(range(251)) * 100
    path.write_bytes(content)
    capability = probe_ple_streaming_capability(path)
    assert capability.production_ready, capability.detail
    with IoUringPageReader(
        path,
        cache_capacity_bytes=8192,
        queue_depth=8,
        max_batch_pages=2,
    ) as reader:
        assert reader.read(31, 9000) == content[31:9031]
        cold = reader.telemetry()
        assert cold.storage_bytes >= 3 * 4096
        assert cold.cache_miss_pages == 3
        assert reader.read(8192, 128) == content[8192:8320]
        warm = reader.telemetry()
        assert warm.cache_hit_pages > cold.cache_hit_pages
        handle = reader.prefetch(((12_288, 512),))
        assert handle.wait() == (content[12_288:12_800],)
        final = reader.telemetry()
        assert final.prefetch_submitted == 1
        assert final.prefetch_completed == 1
        assert final.cache_bytes <= final.cache_capacity_bytes


@pytest.mark.skipif(
    importlib.util.find_spec(
        "freetoken.models.qwen4_exp._ple_io_uring"
    ) is None,
    reason="bundled native io_uring extension is not built",
)
def test_native_io_uring_batches_sparse_pages_into_one_ring_submission(tmp_path: Path):
    path = tmp_path / "native-batch.bin"
    content = bytes(range(253)) * 400
    path.write_bytes(content)
    spans = tuple((page * 8192 + 17, 200) for page in range(8))
    with IoUringPageReader(
        path,
        cache_capacity_bytes=0,
        queue_depth=16,
        max_batch_pages=1,
    ) as reader:
        actual = reader.read_many(spans)
        assert actual == tuple(content[offset : offset + length] for offset, length in spans)
        telemetry = reader.telemetry()
        assert telemetry.submission_batches == 1
        assert telemetry.submitted_sqes == len(spans)
        assert telemetry.cache_miss_pages == len(spans)


@pytest.mark.skipif(
    importlib.util.find_spec(
        "freetoken.models.qwen4_exp._ple_io_uring"
    ) is None,
    reason="bundled native io_uring extension is not built",
)
def test_native_io_uring_safetensors_fp8_row_bank(tmp_path: Path):
    path = tmp_path / "rows.safetensors"
    raw = bytes([0x38, 0x40, 0x44, 0x48, 0x30, 0x38, 0x40, 0x44])
    header = {
        "weight": {
            "dtype": "F8_E4M3",
            "shape": [2, 4],
            "data_offsets": [0, len(raw)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)

    with ShardedSafetensorsMmapRowBank(
        [SafetensorsRowShard(path, "weight")],
        weight_scale=0.5,
        default_dtype=torch.float32,
        io_backend="io_uring_odirect",
        cache_capacity_bytes=8192,
        queue_depth=8,
        max_batch_pages=2,
    ) as bank:
        actual = bank.read_rows([1, 0, 1])
        expected = torch.tensor(
            [[0.25, 0.5, 1.0, 1.5], [0.5, 1.0, 1.5, 2.0], [0.25, 0.5, 1.0, 1.5]]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        telemetry = bank.telemetry()
        assert telemetry["backend"] == "io_uring_odirect"
        assert telemetry["shards"][0]["storage_bytes"] > 0
