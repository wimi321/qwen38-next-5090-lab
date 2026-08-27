from __future__ import annotations

from dataclasses import replace

from q38lab.doctor import DoctorSnapshot, evaluate_doctor


def _envelope_budget() -> dict[str, int]:
    total = 32_607 * 1024**2
    free = 32_000 * 1024**2
    planned = total - free + int(0.89 * free)
    envelope = 31 * 1024**3
    return {
        "gpu_memory_envelope_bytes": envelope,
        "gpu_runtime_reserve_bytes": 512 * 1024**2,
        "gpu_current_used_bytes": total - free,
        "gpu_ratio_managed_bytes": int(0.89 * free),
        "gpu_planned_peak_bytes": planned,
        "gpu_envelope_headroom_bytes": envelope - planned,
    }


def _host_budget() -> dict[str, int]:
    bank_bytes = 68_136_468_480
    locked = bank_bytes * 15 // 48
    staging = 96 * 1024**2
    return {
        "host_ple_pinned_staging_bytes": 32 * 1024**2,
        "host_moe_bounce_staging_bytes": 64 * 1024**2,
        "host_pinned_staging_bytes": staging,
        "host_moe_registered_banks_bytes": bank_bytes - locked,
        "host_wddm_pinned_planned_bytes": bank_bytes - locked + staging,
        "host_wddm_pin_budget_bytes": int(0.4 * 112 * 1024**3),
        "host_moe_os_locked_bytes": locked,
    }


def _ready_snapshot() -> DoctorSnapshot:
    return DoctorSnapshot(
        is_wsl2=True,
        kernel_release="6.6.87.2-microsoft-standard-WSL2",
        os_id="ubuntu",
        os_version_id="24.04",
        source_dir="/home/user/src/FreeToken",
        source_filesystem="ext4",
        model_dir="/home/user/models/checkpoint",
        model_filesystem="ext4",
        gpu_name="NVIDIA GeForce RTX 5090",
        compute_capability="12.0",
        driver_version="591.86",
        gpu_memory_mib=32607,
        gpu_memory_free_mib=32000,
        nvcc_version="13.0",
        torch_version="2.11.0",
        torch_cuda_version="13.0",
        torch_cuda_available=True,
        torch_cuda_probe="NVIDIA GeForce RTX 5090; SM 12.0; tensor probe passed",
        triton_version="3.6.0",
        memory_total_bytes=112 * 1024**3,
        memory_available_bytes=100 * 1024**3,
        swap_total_bytes=0,
        disk_free_bytes=200 * 1024**3,
        model_exists=True,
        checkpoint_file_count=419,
        checkpoint_total_bytes=135_253_622_894,
        checkpoint_safetensors_count=206,
        checkpoint_error=None,
        checkpoint_verification_mode="full-sha256",
        q38lab_distribution_version="0.1.0a1",
        upstream_freetoken_distribution_version=None,
        port=1919,
        port_available=True,
    )


def test_ready_doctor_report():
    report = evaluate_doctor(_ready_snapshot())
    assert report.ready
    assert all(check.status != "fail" for check in report.checks)
    assert report.as_dict()["schema_version"] == 1


def test_doctor_reports_swap_checkpoint_and_busy_port_failures():
    snapshot = replace(
        _ready_snapshot(),
        swap_total_bytes=4 * 1024**3,
        checkpoint_error="checkpoint shape mismatch",
        checkpoint_file_count=None,
        port_available=False,
    )
    report = evaluate_doctor(snapshot)
    statuses = {check.name: check.status for check in report.checks}
    assert not report.ready
    assert statuses["swap"] == "fail"
    assert statuses["checkpoint"] == "fail"
    assert statuses["port"] == "fail"


def test_doctor_gpu_free_floor_allows_normal_wddm_reservation():
    report = evaluate_doctor(replace(_ready_snapshot(), gpu_memory_free_mib=30_000))
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["gpu_memory_available"] == "pass"

    report = evaluate_doctor(replace(_ready_snapshot(), gpu_memory_free_mib=29_999))
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["gpu_memory_available"] == "fail"


def test_256k_doctor_fails_closed_without_native_ple_streaming():
    snapshot = replace(
        _ready_snapshot(),
        requested_profile="rtx5090-wsl2-256k-image",
        ple_linux=True,
        ple_odirect=True,
        ple_kernel_io_uring=True,
        ple_native_extension=False,
        ple_production_ready=False,
        ple_capability_detail="native extension absent",
        memlock_soft_bytes=-1,
        memory_budget={
            "gpu_qsa_cache_bytes": 6_643_777_536,
            "gpu_selector_workspace_bytes": 134_217_728,
            "gpu_vision_weights_bytes": 897_862_112,
            "gpu_fixed_profile_bytes": 7_675_857_376,
            "host_ple_lru_bytes": 4 * 1024**3,
            **_host_budget(),
            **_envelope_budget(),
        },
    )
    report = evaluate_doctor(snapshot)
    statuses = {check.name: check.status for check in report.checks}
    assert not report.ready
    assert statuses["ple_streaming"] == "fail"
    assert statuses["locked_memory"] == "pass"
    assert statuses["profile_memory_budget"] == "pass"


def test_256k_doctor_accepts_complete_ple_capability_contract():
    snapshot = replace(
        _ready_snapshot(),
        requested_profile="rtx5090-wsl2-256k-image",
        ple_linux=True,
        ple_odirect=True,
        ple_kernel_io_uring=True,
        ple_native_extension=True,
        ple_production_ready=True,
        ple_capability_detail="native probe passed",
        qsa_native_topk_ready=True,
        qsa_native_topk_detail="native JIT/launch/parity passed",
        memlock_soft_bytes=-1,
        memory_budget={
            **_host_budget(),
            **_envelope_budget(),
        },
    )
    report = evaluate_doctor(snapshot)
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["ple_streaming"] == "pass"
    assert statuses["qsa_native_fast_topk"] == "pass"
    assert statuses["locked_memory"] == "pass"
    assert statuses["wddm_pin_budget"] == "pass"
    assert statuses["media_dns_policy"] == "warn"
    snapshot_json = report.as_dict()["snapshot"]
    assert snapshot_json["media_doh_fallback_enabled"] is False
    assert snapshot_json["media_system_dns_hard_cancel_supported"] is False
    assert report.ready


def test_256k_doctor_rejects_memlock_below_locked_moe_banks():
    host = _host_budget()
    snapshot = replace(
        _ready_snapshot(),
        requested_profile="rtx5090-wsl2-256k-image",
        ple_linux=True,
        ple_odirect=True,
        ple_kernel_io_uring=True,
        ple_native_extension=True,
        ple_production_ready=True,
        qsa_native_topk_ready=True,
        memlock_soft_bytes=host["host_moe_os_locked_bytes"] - 1,
        memory_budget={**host, **_envelope_budget()},
    )
    report = evaluate_doctor(snapshot)
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["locked_memory"] == "fail"
    assert statuses["wddm_pin_budget"] == "pass"
    assert not report.ready


def test_256k_doctor_records_explicit_doh_opt_in_without_claiming_hard_cancel():
    snapshot = replace(
        _ready_snapshot(),
        requested_profile="rtx5090-wsl2-256k-image",
        media_doh_fallback_enabled=True,
        media_system_dns_hard_cancel_supported=False,
    )
    report = evaluate_doctor(snapshot)
    check = next(item for item in report.checks if item.name == "media_dns_policy")
    assert check.status == "warn"
    assert "enabled" in check.detail and "soft cancellation" in check.detail


def test_256k_doctor_rejects_planned_peak_at_release_envelope():
    budget = _envelope_budget()
    budget["gpu_planned_peak_bytes"] = budget["gpu_memory_envelope_bytes"]
    budget["gpu_envelope_headroom_bytes"] = 0
    snapshot = replace(
        _ready_snapshot(),
        requested_profile="rtx5090-wsl2-256k-image",
        ple_linux=True,
        ple_odirect=True,
        ple_kernel_io_uring=True,
        ple_native_extension=True,
        ple_production_ready=True,
        qsa_native_topk_ready=True,
        qsa_native_topk_detail="native JIT/launch/parity passed",
        memlock_soft_bytes=-1,
        memory_budget={
            **_host_budget(),
            **budget,
        },
    )
    report = evaluate_doctor(snapshot)
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["profile_memory_budget"] == "fail"
    assert not report.ready
