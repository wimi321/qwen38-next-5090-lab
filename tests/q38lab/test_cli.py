from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from q38lab.cli import build_parser, main
from q38lab.doctor import DoctorSnapshot
from q38lab.runtime import Dependencies


def test_download_requires_explicit_license_acceptance():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["download"])
    assert exc.value.code == 2


def test_smoke_image_flags_are_parameterized() -> None:
    args = build_parser().parse_args(
        [
            "smoke",
            "--images",
            "--https-image-url",
            "https://images.example/public.png",
        ]
    )
    assert args.images is True
    assert args.https_image_url == "https://images.example/public.png"


def test_serve_validates_checkpoint_and_launches_exact_profile(tmp_path):
    calls = []

    def verify(path, *, full):
        calls.append(("verify", path, full))

    def launch(argv, prog):
        calls.append(("launch", argv, prog))

    def attest(config, argv):
        calls.append(("attest", config, argv))
        return tmp_path / "attestation.json"

    def remove(path):
        calls.append(("remove", path))

    deps = Dependencies(
        env={}, verify_checkpoint_receipt=lambda path, *, require_full: verify(path, full=require_full), launch_server=launch,
        attestation_writer=attest, attestation_remover=remove,
    )
    code = main(
        ["serve", "--profile", "rtx5090-wsl2", "--model-dir", str(tmp_path)],
        deps=deps,
    )
    assert code == 0
    assert calls[0] == ("verify", tmp_path, False)
    argv = calls[1][2]
    assert calls[2][2] == "q38lab serve"
    assert argv[argv.index("--memory-ratio") + 1] == "0.89"
    assert argv[argv.index("--num-tokens") + 1] == "8192"
    assert "--moe-cache-auto" in argv
    assert not any("moe-cpu-layers" in arg for arg in argv)
    assert calls[3] == ("remove", tmp_path / "attestation.json")


def test_doctor_json_uses_injected_collector(capsys):
    snapshot = DoctorSnapshot(
        is_wsl2=False,
        kernel_release="test",
        os_id="windows",
        os_version_id="test",
        source_dir="C:/src",
        source_filesystem="ntfs",
        model_dir="C:/model",
        model_filesystem="ntfs",
        gpu_name=None,
        compute_capability=None,
        driver_version=None,
        gpu_memory_mib=None,
        gpu_memory_free_mib=None,
        nvcc_version=None,
        torch_version=None,
        torch_cuda_version=None,
        torch_cuda_available=False,
        torch_cuda_probe="not available",
        triton_version=None,
        memory_total_bytes=None,
        memory_available_bytes=None,
        swap_total_bytes=None,
        disk_free_bytes=None,
        model_exists=False,
        checkpoint_file_count=None,
        checkpoint_total_bytes=None,
        checkpoint_safetensors_count=None,
        checkpoint_error=None,
        checkpoint_verification_mode=None,
        q38lab_distribution_version=None,
        upstream_freetoken_distribution_version=None,
        port=1919,
        port_available=True,
    )
    deps = Dependencies(env={}, doctor_collector=lambda **kwargs: snapshot)
    assert main(["doctor", "--json"], deps=deps) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == 1
    assert document["ready"] is False


def test_bench_wraps_the_single_authoritative_release_harness(tmp_path):
    model = tmp_path / "model"
    attestation = tmp_path / "serve.json"
    attestation.write_text(
        json.dumps({
            "pid": 4321,
            "runtime_commit": "a" * 40,
            "model_realpath": str(model.resolve()),
            "resolved_config": {
                "profile": "rtx5090-wsl2",
                "profile_contract_verified": True,
                "host": "127.0.0.1",
                "port": 1919,
            },
        }) + "\n",
        encoding="utf-8",
    )
    calls = []

    def harness(argv):
        calls.append(argv)
        return 0

    deps = Dependencies(env={}, release_harness=harness)
    out = tmp_path / "results"
    code = main(
        [
            "bench", "--out", str(out), "--model-dir", str(model),
            "--attestation", str(attestation),
        ],
        deps=deps,
    )
    assert code == 0
    argv = calls[0]
    assert argv[argv.index("--model-dir") + 1] == str(model)
    assert argv[argv.index("--server-pid") + 1] == "4321"
    assert argv[argv.index("--expected-commit") + 1] == "a" * 40
    assert argv[argv.index("--profile") + 1] == "rtx5090-wsl2"
    assert argv[argv.index("--duration-seconds") + 1] == "1800"
    assert argv[argv.index("--sequential-requests") + 1] == "100"
    assert argv[argv.index("--decode-tokens") + 1] == "256"


def test_bench_rejects_weakened_release_counts_before_running_harness(
    tmp_path, capsys
):
    model = tmp_path / "model"
    attestation = tmp_path / "serve.json"
    attestation.write_text(
        json.dumps({
            "pid": 4321,
            "runtime_commit": "a" * 40,
            "model_realpath": str(model.resolve()),
            "resolved_config": {
                "profile": "rtx5090-wsl2",
                "profile_contract_verified": True,
                "host": "127.0.0.1",
                "port": 1919,
            },
        }) + "\n",
        encoding="utf-8",
    )
    calls = []
    deps = Dependencies(env={}, release_harness=lambda argv: calls.append(argv) or 0)
    code = main(
        [
            "bench", "--out", str(tmp_path / "results"),
            "--model-dir", str(model), "--attestation", str(attestation),
            "--warmups", "2",
        ],
        deps=deps,
    )
    assert code == 1
    assert calls == []
    assert "must be at least 3" in capsys.readouterr().err


def test_256k_bench_forwards_profile_image_fixtures_and_1024_decode(tmp_path):
    model = tmp_path / "model"
    image = tmp_path / "fixture.png"
    image.write_bytes(b"not-decoded-by-cli")
    attestation = tmp_path / "serve.json"
    attestation.write_text(
        json.dumps({
            "pid": 4321,
            "runtime_commit": "b" * 40,
            "model_realpath": str(model.resolve()),
            "resolved_config": {
                "profile": "rtx5090-wsl2-256k-image",
                "profile_contract_verified": True,
                "host": "127.0.0.1",
                "port": 1919,
            },
        }) + "\n",
        encoding="utf-8",
    )
    calls = []
    deps = Dependencies(env={}, release_harness=lambda argv: calls.append(argv) or 0)

    assert main([
        "bench",
        "--profile", "rtx5090-wsl2-256k-image",
        "--out", str(tmp_path / "results"),
        "--model-dir", str(model),
        "--attestation", str(attestation),
        "--decode-tokens", "1024",
        "--image-file", str(image),
        "--https-image-url", "https://example.com/fixture.png",
    ], deps=deps) == 0

    argv = calls[0]
    assert argv[argv.index("--profile") + 1] == "rtx5090-wsl2-256k-image"
    assert argv[argv.index("--decode-tokens") + 1] == "1024"
    assert argv[argv.index("--image-file") + 1] == str(image)
    assert argv[argv.index("--https-image-url") + 1] == "https://example.com/fixture.png"


def test_256k_serve_requires_native_ple_before_checkpoint_or_launch(tmp_path, capsys):
    calls = []
    deps = Dependencies(
        env={},
        ple_capability_probe=lambda path: SimpleNamespace(
            production_ready=False,
            detail="native extension absent",
        ),
        verify_checkpoint_receipt=lambda *args, **kwargs: calls.append("verify"),
        launch_server=lambda *args, **kwargs: calls.append("launch"),
    )
    code = main(
        [
            "serve",
            "--profile",
            "rtx5090-wsl2-256k-image",
            "--model-dir",
            str(tmp_path),
        ],
        deps=deps,
    )
    assert code == 1
    assert calls == []
    assert "native io_uring + O_DIRECT" in capsys.readouterr().err


def test_256k_serve_sets_vision_and_ple_environment_only_during_launch(tmp_path):
    observed = {}

    def launch(argv, prog):
        observed["argv"] = argv
        observed["vision"] = os.environ.get("FREETOKEN_LOAD_VISION")
        observed["ple"] = os.environ.get("FREETOKEN_PLE_IO_BACKEND")
        observed["native_topk"] = os.environ.get("FREETOKEN_QSA_REQUIRE_NATIVE_TOPK")

    previous_vision = os.environ.get("FREETOKEN_LOAD_VISION")
    previous_ple = os.environ.get("FREETOKEN_PLE_IO_BACKEND")
    previous_native_topk = os.environ.get("FREETOKEN_QSA_REQUIRE_NATIVE_TOPK")
    deps = Dependencies(
        env={},
        ple_capability_probe=lambda path: SimpleNamespace(
            production_ready=True,
            detail="ok",
        ),
        qsa_native_topk_probe=lambda: SimpleNamespace(
            production_ready=True,
            detail="native JIT/launch/parity passed",
        ),
        ple_checkpoint_probe=lambda path: {
            "status": "pass",
            "release_qualified": True,
        },
        verify_checkpoint_receipt=lambda *args, **kwargs: None,
        launch_server=launch,
        attestation_writer=lambda config, argv, **kwargs: tmp_path / "attestation.json",
        attestation_remover=lambda path: None,
    )
    assert main(
        [
            "serve",
            "--profile",
            "rtx5090-wsl2-256k-image",
            "--model-dir",
            str(tmp_path),
        ],
        deps=deps,
    ) == 0
    assert observed["vision"] == "1"
    assert observed["ple"] == "io_uring_odirect"
    assert observed["native_topk"] == "1"
    argv = observed["argv"]
    assert argv[argv.index("--max-seq-len-override") + 1] == "262144"
    assert argv[argv.index("--max-prefill-length") + 1] == "512"
    assert os.environ.get("FREETOKEN_LOAD_VISION") == previous_vision
    assert os.environ.get("FREETOKEN_PLE_IO_BACKEND") == previous_ple
    assert os.environ.get("FREETOKEN_QSA_REQUIRE_NATIVE_TOPK") == previous_native_topk


def test_256k_serve_requires_native_fast_topk_before_checkpoint_or_launch(
    tmp_path, capsys
):
    calls = []
    deps = Dependencies(
        env={},
        ple_capability_probe=lambda path: SimpleNamespace(
            production_ready=True,
            detail="ok",
        ),
        qsa_native_topk_probe=lambda: SimpleNamespace(
            production_ready=False,
            detail="JIT launch failed",
        ),
        verify_checkpoint_receipt=lambda *args, **kwargs: calls.append("verify"),
        launch_server=lambda *args, **kwargs: calls.append("launch"),
    )
    code = main(
        [
            "serve",
            "--profile",
            "rtx5090-wsl2-256k-image",
            "--model-dir",
            str(tmp_path),
        ],
        deps=deps,
    )
    assert code == 1
    assert calls == []
    assert "native SM120 QSA fast-topk" in capsys.readouterr().err


def test_256k_serve_requires_release_qualified_ple_row_probe_before_launch(
    tmp_path, capsys
):
    calls = []
    deps = Dependencies(
        env={},
        ple_capability_probe=lambda path: SimpleNamespace(
            production_ready=True,
            detail="ok",
        ),
        qsa_native_topk_probe=lambda: SimpleNamespace(
            production_ready=True,
            detail="native JIT/launch/parity passed",
        ),
        ple_checkpoint_probe=lambda path: {
            "status": "pass",
            "release_qualified": False,
        },
        verify_checkpoint_receipt=lambda *args, **kwargs: calls.append("verify"),
        launch_server=lambda *args, **kwargs: calls.append("launch"),
    )
    code = main(
        [
            "serve",
            "--profile",
            "rtx5090-wsl2-256k-image",
            "--model-dir",
            str(tmp_path),
        ],
        deps=deps,
    )
    assert code == 1
    assert calls == []
    assert "PLE checkpoint row/loader parity" in capsys.readouterr().err
