from __future__ import annotations

import json
from pathlib import Path

import pytest

from q38lab.config import ConfigurationError, is_loopback_host, resolve_serve_config
from q38lab.constants import RTX5090_WSL2_PROFILE


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
