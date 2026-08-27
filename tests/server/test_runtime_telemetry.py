from __future__ import annotations

from types import SimpleNamespace

from freetoken.message import UserReply
from freetoken.server.stats import StatsTracker, build_stats


def _state(stats: StatsTracker):
    model_config = SimpleNamespace(
        has_linear_attention=False,
        has_swa_attention=False,
        is_moe=False,
    )
    config = SimpleNamespace(
        served_model_name="test-model",
        max_seq_len=262_144,
        page_size=4,
        model_config=model_config,
    )
    return SimpleNamespace(
        stats=stats,
        config=config,
        ready_at=None,
        gpus=[],
        instance_id="test-instance",
    )


def test_stats_exposes_schema_complete_zero_runtime_telemetry():
    stats = StatsTracker()
    doc = build_stats(_state(stats), p95_ms=0, ttft_mean_ms=0)

    assert doc["q38lab"]["selector"]["workspace_peak_bytes"] == 0
    assert doc["q38lab"]["selector"] == {
        "workspace_peak_bytes": 0,
        "native_calls": 0,
        "fallback_calls": 0,
        "errors": 0,
    }
    assert doc["q38lab"]["ple"] == {
        "bytes_read": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "wait_ms": 0.0,
        "page_faults": 0,
    }
    assert doc["q38lab"]["vision"]["image_tokens"] == 0
    assert doc["q38lab"]["prefill_chunks"]["count"] == 0
    assert doc["q38lab"]["moe_prefill"]["active_rows"] == 0
    assert doc["q38lab"]["moe_prefill"]["byte_fraction"] == 0.0


def test_terminal_reply_updates_latest_runtime_telemetry_snapshot():
    stats = StatsTracker()
    telemetry = {
        "selector": {
            "workspace_peak_bytes": 128 * 2**20,
            "native_calls": 17,
            "fallback_calls": 0,
            "errors": 0,
        },
        "ple": {
            "bytes_read": 8192,
            "cache_hits": 4,
            "cache_misses": 2,
            "wait_ms": 1.5,
            "page_faults": 7,
        },
        "vision": {"image_tokens": 64, "latency_ms": 12.25},
        "prefill_chunks": {"count": 2, "total_ms": 20.0},
    }
    stats.observe(
        UserReply(
            uid=1,
            incremental_output="done",
            finished=True,
            runtime_telemetry=telemetry,
        )
    )
    telemetry["ple"]["bytes_read"] = -1

    doc = build_stats(_state(stats), p95_ms=0, ttft_mean_ms=0)
    assert doc["q38lab"]["ple"]["bytes_read"] == 8192
    assert doc["q38lab"]["vision"]["image_tokens"] == 64
    assert doc["q38lab"]["selector"]["native_calls"] == 17


def test_stats_uses_effective_engine_page_size_from_readiness_metadata():
    stats = StatsTracker()
    stats.observe(
        UserReply(
            uid=2,
            incremental_output="x",
            finished=False,
            kv_used_pages=1,
            kv_total_pages=65_536,
        )
    )
    state = _state(stats)
    # The API/frontend config retains the requested value, while the QSA SM120
    # backend resolves the live engine to four-token pages.
    state.config.page_size = 1
    state.cache_pools = {"num_pages": 65_536, "page_size": 4}

    doc = build_stats(state, p95_ms=0, ttft_mean_ms=0)

    assert doc["kv"] == {"used_pages": 1, "total_pages": 65_536, "page_size": 4}
    assert doc["kv"]["total_pages"] * doc["kv"]["page_size"] == 262_144


def test_stats_page_size_falls_back_to_frontend_config_without_engine_metadata():
    stats = StatsTracker()
    stats.observe(
        UserReply(
            uid=3,
            incremental_output="x",
            finished=False,
            kv_used_pages=2,
            kv_total_pages=128,
        )
    )
    state = _state(stats)
    state.config.page_size = 16

    assert build_stats(state, p95_ms=0, ttft_mean_ms=0)["kv"]["page_size"] == 16
