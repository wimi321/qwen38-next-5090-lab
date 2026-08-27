from __future__ import annotations

import json
from pathlib import Path

import pytest

from q38lab.config import ConfigurationError, is_loopback_host, resolve_serve_config
from q38lab.constants import (
    RTX5090_WSL2_256K_IMAGE_PROFILE,
    RTX5090_WSL2_PROFILE,
)


def test_profile_file_matches_runtime_profile():
    root = Path(__file__).parents[2]
    document = json.loads((root / "profiles" / "rtx5090-wsl2.json").read_text())
    profile = RTX5090_WSL2_PROFILE
    for key in (
        "profile",
        "served_model_name",
        "gpu",
        "tp_size",
        "max_running_requests",
        "max_seq_len",
        "max_prefill_length",
        "num_tokens",
        "memory_ratio",
        "cache_type",
        "attention_backend",
        "graph",
        "moe_backend",
        "moe_cache_auto",
        "nvfp4_backend",
        "host",
        "port",
    ):
        attr = "name" if key == "profile" else key
        assert document[key] == getattr(profile, attr)


def test_serve_precedence_and_exact_profile_arguments(tmp_path):
    config = resolve_serve_config(
        profile_name="rtx5090-wsl2",
        cli={
            "model_dir": tmp_path,
            "memory_ratio": 0.88,
            "host": None,
            "port": None,
            "served_model_name": None,
            "gpu": None,
            "num_tokens": None,
            "max_seq_len": None,
            "max_prefill_length": None,
            "unsafe_non_loopback": False,
        },
        env={
            "Q38LAB_MEMORY_RATIO": "0.87",
            "Q38LAB_MAX_SEQ_LEN": "4096",
            "Q38LAB_MAX_PREFILL_LENGTH": "4096",
            "Q38LAB_PORT": "2020",
        },
    )
    assert config.memory_ratio == 0.88  # CLI beats environment
    assert config.max_seq_len == 4096  # environment beats profile
    assert config.num_tokens == 8192  # profile fallback
    assert config.port == 2020
    assert config.profile_contract_verified is False
    argv = config.to_ft_argv()
    assert "--cache-type" in argv and argv[argv.index("--cache-type") + 1] == "naive"
    assert "--attention-backend" in argv
    assert argv[argv.index("--attention-backend") + 1] == "qsa_triton"
    assert argv[argv.index("--graph") + 1] == "0"
    assert argv[argv.index("--moe-backend") + 1] == "offload"
    assert "--moe-cache-auto" in argv
    assert "--nvfp4-backend" in argv
    assert not any("moe-cpu-layers" in arg for arg in argv)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.23.4.5", "::1", "[::1]", "localhost"])
def test_loopback_hosts(host):
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.com"])
def test_non_loopback_requires_explicit_acknowledgement(tmp_path, host):
    cli = {
        "model_dir": tmp_path,
        "host": host,
        "unsafe_non_loopback": False,
    }
    with pytest.raises(ConfigurationError, match="unsafe-allow-non-loopback"):
        resolve_serve_config(profile_name="rtx5090-wsl2", cli=cli, env={})
    cli["unsafe_non_loopback"] = True
    assert resolve_serve_config(
        profile_name="rtx5090-wsl2", cli=cli, env={}
    ).host == host


def test_profile_rejects_inconsistent_or_out_of_contract_token_geometry(tmp_path):
    base = {"model_dir": tmp_path, "unsafe_non_loopback": False}
    with pytest.raises(ConfigurationError, match="capped"):
        resolve_serve_config(
            profile_name="rtx5090-wsl2",
            cli={**base, "max_seq_len": 8193}, env={},
        )
    with pytest.raises(ConfigurationError, match="prefill"):
        resolve_serve_config(
            profile_name="rtx5090-wsl2",
            cli={**base, "max_seq_len": 4096, "max_prefill_length": 8192}, env={},
        )
    with pytest.raises(ConfigurationError, match="KV token"):
        resolve_serve_config(
            profile_name="rtx5090-wsl2",
            cli={**base, "num_tokens": 4096}, env={},
        )


def test_profile_contract_requires_the_canonical_loopback_spelling(tmp_path):
    config = resolve_serve_config(
        profile_name="rtx5090-wsl2",
        cli={
            "model_dir": tmp_path,
            "host": "localhost",
            "unsafe_non_loopback": False,
        },
        env={},
    )
    assert config.host == "localhost"
    assert config.profile_contract_verified is False


def test_256k_image_profile_file_runtime_and_environment_contract(tmp_path):
    root = Path(__file__).parents[2]
    document = json.loads(
        (root / "profiles" / "rtx5090-wsl2-256k-image.json").read_text()
    )
    profile = RTX5090_WSL2_256K_IMAGE_PROFILE
    assert document["profile"] == profile.name
    assert document["max_seq_len"] == profile.max_seq_len == 262_144
    assert document["max_prefill_length"] == profile.max_prefill_length == 512
    assert document["num_tokens"] == profile.num_tokens == 262_144
    assert document["load_vision"] is profile.load_vision is True
    assert document["ple_io_backend"] == "io_uring_odirect"
    assert document["ple_cache_bytes"] == profile.ple_cache_bytes == 4 * 1024**3
    assert (
        document["qsa_require_native_topk"]
        is profile.qsa_require_native_topk
        is True
    )
    assert (
        document["gpu_memory_envelope_bytes"]
        == profile.gpu_memory_envelope_bytes
        == 31 * 1024**3
    )
    assert (
        document["gpu_runtime_reserve_bytes"]
        == profile.gpu_runtime_reserve_bytes
        == 512 * 1024**2
    )

    config = resolve_serve_config(
        profile_name=profile.name,
        cli={"model_dir": tmp_path, "unsafe_non_loopback": False},
        env={},
    )
    assert config.profile_contract_verified
    assert config.moe_prefill_sparse is True
    assert config.runtime_environment()["FREETOKEN_MOE_PREFILL_SPARSE"] == "1"
    assert config.attention_backend == "qsa_triton_sm120"
    environment = config.runtime_environment()
    assert environment["FREETOKEN_LOAD_VISION"] == "1"
    assert environment["FREETOKEN_PLE_IO_BACKEND"] == "io_uring_odirect"
    assert environment["FREETOKEN_PLE_CACHE_BYTES"] == str(4 * 1024**3)
    assert environment["FREETOKEN_QSA_REQUIRE_NATIVE_TOPK"] == "1"
    assert environment["FREETOKEN_GPU_MEMORY_ENVELOPE_BYTES"] == str(31 * 1024**3)


def test_8k_profile_explicitly_keeps_vision_and_native_streaming_off(tmp_path):
    config = resolve_serve_config(
        profile_name=RTX5090_WSL2_PROFILE.name,
        cli={"model_dir": tmp_path, "unsafe_non_loopback": False},
        env={},
    )
    environment = config.runtime_environment()
    assert environment["FREETOKEN_LOAD_VISION"] == "0"
    assert environment["FREETOKEN_PLE_IO_BACKEND"] == "mmap"
    assert environment["FREETOKEN_QSA_REQUIRE_NATIVE_TOPK"] == "0"
