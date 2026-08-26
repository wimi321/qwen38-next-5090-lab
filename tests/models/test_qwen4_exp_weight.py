from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import safetensors.torch
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.qwen4_exp import weight as qwen4_weight


if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        (
            "model.language_model.layers.11.self_attn.q_proj.weight",
            "model.layers.11.self_attn.q_proj.weight",
        ),
        (
            "language_model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.A_log",
        ),
        (
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            "model.hyper_connection_mixer.hc_norm.weight",
        ),
        (
            "model.language_model.layers.1.ple.norm_key.weight",
            "model.layers.1.ple.norm_key.weight",
        ),
        ("lm_head.weight", "lm_head.weight"),
        ("model.visual.blocks.0.attn.qkv.weight", None),
        ("visual.blocks.0.attn.qkv.weight", None),
        ("mtp.layers.0.self_attn.q_proj.weight", None),
        ("model.mtp.layers.0.self_attn.q_proj.weight", None),
        ("model.language_model.mtp.layers.0.self_attn.q_proj.weight", None),
        (
            "model.language_model.layers.0.mlp.experts.3.gate_proj.weight",
            None,
        ),
        (
            "model.language_model.layers.0.mlp.experts.3.gate_proj.input_scale",
            None,
        ),
        (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_127.weight",
            None,
        ),
        (
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
            None,
        ),
        (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_heads_vocab_sizes",
            None,
        ),
        (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.weight_scale",
            None,
        ),
    ],
)
def test_rename_maps_only_resident_text_state(raw_name, expected):
    assert qwen4_weight._rename(raw_name) == expected


def test_detects_target_w4a4_activation_recipe():
    config = types.SimpleNamespace(
        quantization_config={
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "dynamic": False,
                        "group_size": 16,
                        "num_bits": 4,
                        "type": "float",
                    }
                }
            }
        }
    )
    assert qwen4_weight._uses_w4a4_activation_recipe(config)
    config.quantization_config["config_groups"]["group_0"][
        "input_activations"
    ]["num_bits"] = 8
    assert not qwen4_weight._uses_w4a4_activation_recipe(config)


def test_fusion_plan_uses_checkpoint_output_order():
    q = "model.layers.11.self_attn.q_proj.weight"
    q_plan = qwen4_weight._fusion_plan(q)
    assert q_plan == (
        "model.layers.11.self_attn.qkv_proj.weight",
        (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
        ),
        0,
    )

    a = "model.layers.0.linear_attn.in_proj_a.weight"
    a_plan = qwen4_weight._fusion_plan(a)
    assert a_plan == (
        "model.layers.0.linear_attn.in_proj.weight",
        (
            ".linear_attn.in_proj_qkv.weight",
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_b.weight",
            ".linear_attn.in_proj_a.weight",
        ),
        3,
    )

    shared = "model.layers.0.mlp.shared_expert.up_proj.weight"
    shared_plan = qwen4_weight._fusion_plan(shared)
    assert shared_plan == (
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
        (
            ".mlp.shared_expert.gate_proj.weight",
            ".mlp.shared_expert.up_proj.weight",
        ),
        1,
    )
    assert qwen4_weight._fusion_plan("model.layers.0.mlp.gate.weight") is None


def _write_indexed_shards(tmp_path: Path, shards: dict[str, dict[str, torch.Tensor]]) -> None:
    weight_map: dict[str, str] = {}
    for shard_name, tensors in shards.items():
        safetensors.torch.save_file(tensors, str(tmp_path / shard_name))
        for name in tensors:
            weight_map[name] = shard_name
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


def _rows(value: float, count: int, width: int = 3) -> torch.Tensor:
    return torch.full((count, width), value, dtype=torch.float32)


def test_synthetic_cross_shard_fusion_skip_and_raw_norms(tmp_path, monkeypatch):
    p = "model.language_model."
    q_norm = torch.tensor([-0.90, -0.25, 0.10], dtype=torch.float32)
    hc_norm = torch.tensor([-0.75, 0.00, 0.35, 0.80], dtype=torch.float32)
    ple_norm = torch.tensor([-0.60, 0.20, 0.70], dtype=torch.float32)

    shards = {
        "resident-a.safetensors": {
            p + "embed_tokens.weight": _rows(0.25, 5),
            p + "layers.11.self_attn.q_proj.weight": _rows(1.0, 4),
            p + "layers.0.linear_attn.in_proj_z.weight": _rows(5.0, 2),
            p + "layers.0.mlp.shared_expert.gate_proj.weight": _rows(9.0, 2),
            p + "layers.11.self_attn.q_norm.weight": q_norm,
            p + "layers.0.attn_hyper_connection.hc_norm.weight": hc_norm,
        },
        "resident-b.safetensors": {
            p + "layers.11.self_attn.k_proj.weight": _rows(2.0, 1),
            p + "layers.0.linear_attn.in_proj_qkv.weight": _rows(4.0, 3),
            p + "layers.0.linear_attn.in_proj_a.weight": _rows(7.0, 1),
            p + "layers.0.mlp.shared_expert.up_proj.weight": _rows(10.0, 2),
            p + "layers.0.mlp.shared_expert.down_proj.weight": _rows(11.0, 3, 2),
        },
        "resident-c.safetensors": {
            p + "layers.11.self_attn.v_proj.weight": _rows(3.0, 1),
            p + "layers.0.linear_attn.in_proj_b.weight": _rows(6.0, 1),
            p + "layers.1.ple.norm_key.weight": ple_norm,
            p + "layers.0.mlp.gate.weight": _rows(12.0, 3),
            "lm_head.weight": _rows(13.0, 5),
        },
        # Every key in this shard is auxiliary or disabled.  The loader must
        # filter from the index and never mmap this file.
        "excluded-only.safetensors": {
            "model.visual.blocks.0.attn.qkv.weight": _rows(20.0, 1),
            "mtp.layers.0.self_attn.q_proj.weight": _rows(21.0, 1),
            p + "layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(
                1, 1, dtype=torch.uint8
            ),
            p + "layers.0.mlp.experts.0.gate_proj.weight_scale": torch.ones(1, 1),
            p + "layers.0.mlp.experts.0.gate_proj.weight_scale_2": torch.ones(1),
            p + "layers.0.mlp.experts.0.gate_proj.input_scale": torch.ones(1),
            p
            + "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": torch.zeros(
                2, 2, dtype=torch.uint8
            ),
            p + "layers.1.ple.ple_embedding.ngram_embedding.weight_scale": torch.ones(1),
            p + "layers.1.ple.ple_embedding.layer_multipliers": torch.ones(
                3, dtype=torch.int64
            ),
        },
    }
    _write_indexed_shards(tmp_path, shards)

    # Config parsing is independently tested in test_qwen4_exp_config; here it is
    # replaced so this remains a synthetic loader/oracle test.
    monkeypatch.setattr(qwen4_weight, "cached_load_hf_config", lambda _path: object())
    monkeypatch.setattr(qwen4_weight, "parse_config", lambda _config: object())

    opened: list[str] = []
    real_safe_open = qwen4_weight.safetensors.safe_open

    def tracking_safe_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(qwen4_weight.safetensors, "safe_open", tracking_safe_open)

    loaded = dict(
        qwen4_weight.iter_weights(
            str(tmp_path),
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )

    assert set(opened) == {
        "resident-a.safetensors",
        "resident-b.safetensors",
        "resident-c.safetensors",
    }
    assert "excluded-only.safetensors" not in opened

    qkv = loaded["model.layers.11.self_attn.qkv_proj.weight"]
    assert qkv.shape == (6, 3)
    assert torch.equal(qkv[:, 0], torch.tensor([1, 1, 1, 1, 2, 3], dtype=qkv.dtype))

    gdn = loaded["model.layers.0.linear_attn.in_proj.weight"]
    assert gdn.shape == (7, 3)
    assert torch.equal(
        gdn[:, 0], torch.tensor([4, 4, 4, 5, 5, 6, 7], dtype=gdn.dtype)
    )

    shared = loaded["model.layers.0.mlp.shared_expert.gate_up_proj.weight"]
    assert torch.equal(shared[:, 0], torch.tensor([9, 9, 10, 10], dtype=shared.dtype))

    # The grouped RMSNorm implementations apply (1 + weight) themselves.  A
    # loader-side +1 would make all three comparisons fail observably.
    assert torch.equal(loaded["model.layers.11.self_attn.q_norm.weight"], q_norm)
    assert torch.equal(
        loaded["model.layers.0.attn_hyper_connection.hc_norm.weight"], hc_norm
    )
    assert torch.equal(loaded["model.layers.1.ple.norm_key.weight"], ple_norm)

    assert "model.embed_tokens.weight" in loaded
    assert "model.layers.0.mlp.shared_expert.down_proj.weight" in loaded
    assert "model.layers.0.mlp.gate.weight" in loaded
    assert "lm_head.weight" in loaded
    assert not any("visual" in key or "mtp" in key for key in loaded)
    assert not any(".experts." in key or "ple_embedding" in key for key in loaded)
    assert not any(
        any(f".self_attn.{part}_proj." in key for part in ("q", "k", "v"))
        for key in loaded
    )


def test_incomplete_fusion_fails_before_opening_a_shard(tmp_path, monkeypatch):
    raw_q = "model.language_model.layers.11.self_attn.q_proj.weight"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {raw_q: "not-present.safetensors"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(qwen4_weight, "cached_load_hf_config", lambda _path: object())
    monkeypatch.setattr(qwen4_weight, "parse_config", lambda _config: object())

    with pytest.raises(ValueError, match="incomplete fusion.*k_proj.*v_proj"):
        list(
            qwen4_weight.iter_weights(
                str(tmp_path),
                torch.device("cpu"),
                include_moe_experts=False,
                include_non_moe=True,
            )
        )


def test_resident_loader_rejects_direct_expert_materialization():
    with pytest.raises(NotImplementedError, match="offload expert-bank"):
        list(
            qwen4_weight.iter_weights(
                "unused",
                torch.device("cpu"),
                include_moe_experts=True,
                include_non_moe=True,
            )
        )
    assert list(
        qwen4_weight.iter_weights(
            "unused",
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=False,
        )
    ) == []


def test_expert_hooks_delegate_to_qwen35_nvfp4_implementation(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []
    fake = types.ModuleType("freetoken.models.qwen3_5_moe.weight")

    def record(name):
        def implementation(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return implementation

    fake.setup_offload_expert_banks = record("setup")
    fake.load_nvfp4_expert_sources = record("serial")
    fake.load_nvfp4_expert_sources_parallel = record("parallel")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)

    config = object()
    sink = object()
    assert qwen4_weight.setup_offload_expert_banks(
        "checkpoint",
        config,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dummy=True,
        parallel=True,
        workers=3,
        chunk=4096,
        decode_target="cpu",
        layer_sink=sink,
    ) == "setup"
    assert qwen4_weight.load_nvfp4_expert_sources(
        "checkpoint", config, layer_sink=sink
    ) == "serial"
    assert qwen4_weight.load_nvfp4_expert_sources_parallel(
        "checkpoint", config, workers=5, chunk=8192, layer_sink=sink
    ) == "parallel"

    assert calls[0] == (
        "setup",
        ("checkpoint", config),
        {
            "device": torch.device("cpu"),
            "dtype": torch.bfloat16,
            "dummy": True,
            "parallel": True,
            "workers": 3,
            "chunk": 4096,
            "decode_target": "cpu",
            "layer_sink": sink,
        },
    )
    assert calls[1] == ("serial", ("checkpoint", config), {"layer_sink": sink})
    assert calls[2] == (
        "parallel",
        ("checkpoint", config),
        {"workers": 5, "chunk": 8192, "layer_sink": sink},
    )
