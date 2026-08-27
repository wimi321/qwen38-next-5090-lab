from __future__ import annotations

from dataclasses import replace

from q38lab.doctor import DoctorSnapshot, evaluate_doctor


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
