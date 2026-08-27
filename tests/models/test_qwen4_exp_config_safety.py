# Modified by Qwen3.8 Next 5090 Lab contributors in 2026; see MODIFICATIONS.md.
from __future__ import annotations

import json

import pytest

from freetoken.utils import hf


@pytest.mark.parametrize(
    "identity",
    [
        {"model_type": "qwen4_exp"},
        {"model_type": "qwen4_exp_moe"},
        {"architectures": ["Qwen4ExpForCausalLM"]},
        {"text_config": {"model_type": "qwen4_exp"}},
        {
            "text_config": {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
            },
        },
    ],
)
def test_native_qwen4_exp_config_family_never_executes_remote_code(
    tmp_path, monkeypatch, identity
):
    (tmp_path / "config.json").write_text(
        json.dumps({
            **identity,
            "auto_map": {"AutoConfig": "configuration_qwen4_exp.CustomConfig"},
        }),
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("AutoConfig must not run for the native Qwen4-Exp model")

    monkeypatch.setattr(hf.AutoConfig, "from_pretrained", forbidden)
    hf._load_hf_config.cache_clear()
    try:
        config = hf._load_hf_config(str(tmp_path))
    finally:
        hf._load_hf_config.cache_clear()
    assert isinstance(config, hf.RawConfigShim)
    assert config.to_dict()["auto_map"]["AutoConfig"].endswith(".CustomConfig")
