from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.scheduler import Scheduler, _sum_nested_counter


class _Bank:
    def telemetry(self):
        return {
            "shards": [
                {
                    "storage_bytes": 4096,
                    "cache_hit_pages": 3,
                    "cache_miss_pages": 2,
                    "wait_ns": 250_000,
                },
                {
                    "storage_bytes": 8192,
                    "cache_hit_pages": 4,
                    "cache_miss_pages": 1,
                    "wait_ns": 750_000,
                },
            ]
        }


def test_nested_counter_prefers_parent_rollup_over_children():
    value = {"storage_bytes": 12, "children": [{"storage_bytes": 5}]}
    assert _sum_nested_counter(value, "storage_bytes") == 12


def test_scheduler_runtime_snapshot_aggregates_native_ple_readers():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(
        auxiliary_banks={"ple": _Bank()},
        moe_offload_cache=SimpleNamespace(
            prefill_sparse_stats=lambda: {
                "active_rows": 128,
                "possible_rows": 512,
                "bytes_copied": 400,
                "full_bytes": 1000,
                "row_fraction": 0.25,
                "byte_fraction": 0.4,
            }
        ),
        kv_cache=SimpleNamespace(
            selector_telemetry=lambda: {
                "workspace_peak_bytes": 128 * 2**20,
                "native_calls": 11,
                "fallback_calls": 0,
                "errors": 0,
            }
        ),
    )
    scheduler._runtime_telemetry = {
        "vision": {"image_tokens": 64, "latency_ms": 12.5},
        "prefill_chunks": {"count": 3, "total_ms": 25.0},
    }

    snapshot = scheduler._runtime_telemetry_snapshot()

    assert snapshot["selector"]["workspace_peak_bytes"] == 128 * 2**20
    assert snapshot["selector"]["native_calls"] == 11
    assert snapshot["selector"]["fallback_calls"] == 0
    assert snapshot["selector"]["errors"] == 0
    assert snapshot["ple"]["bytes_read"] == 12_288
    assert snapshot["ple"]["cache_hits"] == 7
    assert snapshot["ple"]["cache_misses"] == 3
    assert snapshot["ple"]["wait_ms"] == 1.0
    assert snapshot["ple"]["page_faults"] >= 0
    assert snapshot["vision"] == {"image_tokens": 64, "latency_ms": 12.5}
    assert snapshot["prefill_chunks"] == {"count": 3, "total_ms": 25.0}
    assert snapshot["moe_prefill"] == {
        "active_rows": 128,
        "possible_rows": 512,
        "bytes_copied": 400,
        "full_bytes": 1000,
        "row_fraction": 0.25,
        "byte_fraction": 0.4,
    }
