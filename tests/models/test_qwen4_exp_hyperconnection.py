from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.qwen4_exp.hyperconnection import (
    Qwen4ExpGatedResidual,
    Qwen4ExpGroupedRMSNorm,
)


if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


def _group_norm_oracle(
    x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    variance = (grouped * grouped).sum(dim=-1, keepdim=True) / group_size
    normalized = grouped / torch.sqrt(variance + eps)
    return (normalized.flatten(-2) * (weight.float() + 1.0)).to(x.dtype)


def _gated_residual_oracle(
    x: torch.Tensor,
    *,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    hc_count: int,
    hidden_size: int,
    eps: float,
):
    normed = _group_norm_oracle(x, norm_weight, hidden_size, eps)
    lowrank = F.linear(normed, down_weight) / hc_count
    read_weights = torch.sigmoid(F.linear(F.silu(lowrank), up_weight))
    read_weights = read_weights.reshape(*x.shape[:-1], hc_count, hidden_size)
    mixed = (read_weights * normed.reshape(*x.shape[:-1], hc_count, hidden_size)).sum(dim=-2)
    mixed = mixed / hc_count
    if inject_weight is None:
        return mixed
    write_weights = 2.0 * torch.sigmoid(F.linear(normed, inject_weight) / hc_count)
    return mixed, x, write_weights


def _set_deterministic_weights(op: Qwen4ExpGatedResidual) -> None:
    with torch.no_grad():
        for index, tensor in enumerate(op.state_dict().values(), start=1):
            values = torch.linspace(
                -0.25 * index,
                0.25 * index,
                tensor.numel(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor.copy_(values.reshape_as(tensor))


def test_grouped_rmsnorm_matches_independent_torch_oracle():
    norm = Qwen4ExpGroupedRMSNorm(dim=24, group_size=6, eps=3e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.2, 0.3, 24))
    x = torch.linspace(-2.0, 1.0, 2 * 3 * 24).reshape(2, 3, 24)

    got = norm.forward(x)
    expected = _group_norm_oracle(x, norm.weight, group_size=6, eps=3e-5)

    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    # A global 24-wide norm is observably different; this guards the four group boundary.
    global_norm = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 3e-5)
    global_norm = global_norm * (1.0 + norm.weight)
    assert not torch.allclose(got, global_norm)


def test_gated_residual_matches_independent_torch_oracle():
    op = Qwen4ExpGatedResidual(
        hidden_size=6, hc_count=4, hc_lowrank=5, rms_norm_eps=2e-5
    )
    _set_deterministic_weights(op)
    x = torch.linspace(-1.5, 2.0, 2 * 3 * 24).reshape(2, 3, 24)

    got = op.forward(x)
    expected = _gated_residual_oracle(
        x,
        norm_weight=op.hc_norm.weight,
        down_weight=op.input_mix_weight_down.weight,
        up_weight=op.input_mix_weight_up.weight,
        inject_weight=op.block_inject_weight.weight,
        hc_count=4,
        hidden_size=6,
        eps=2e-5,
    )

    for actual, reference in zip(got, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    assert got[0].shape == (2, 3, 6)
    assert got[1] is x
    assert got[2].shape == (2, 3, 4)
    assert torch.all((got[2] > 0) & (got[2] < 2))


def test_final_mixer_uses_same_read_path_without_write_parameters():
    mixer = Qwen4ExpGatedResidual(
        hidden_size=3, hc_count=4, hc_lowrank=7, use_combine=False
    )
    _set_deterministic_weights(mixer)
    x = torch.linspace(-0.7, 1.1, 5 * 12).reshape(5, 12)

    got = mixer.forward(x)
    expected = _gated_residual_oracle(
        x,
        norm_weight=mixer.hc_norm.weight,
        down_weight=mixer.input_mix_weight_down.weight,
        up_weight=mixer.input_mix_weight_up.weight,
        inject_weight=None,
        hc_count=4,
        hidden_size=3,
        eps=1e-6,
    )

    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    assert got.shape == (5, 3)
    assert mixer.block_inject_weight is None
    assert not any(key.startswith("block_inject_weight") for key in mixer.state_dict())


def test_released_defaults_and_checkpoint_parameter_names():
    op = Qwen4ExpGatedResidual(hidden_size=8)
    state = op.state_dict()

    assert op.hc_count == 4
    assert op.hc_lowrank == 320
    assert state["hc_norm.weight"].shape == (32,)
    assert state["input_mix_weight_down.weight"].shape == (320, 32)
    assert state["input_mix_weight_up.weight"].shape == (32, 320)
    assert state["block_inject_weight.weight"].shape == (4, 32)


def test_gated_residual_rejects_wrong_feature_width():
    op = Qwen4ExpGatedResidual(hidden_size=8, hc_lowrank=4)
    with pytest.raises(ValueError, match="Expected 32 hyper-connection features, got 31"):
        op.forward(torch.zeros(2, 31))


def test_qwen35_gdn_output_gate_is_configurable_and_legacy_default_is_silu():
    # Import lazily so the pure hyper-connection tests remain usable in a CPU-only
    # environment that does not install FreeToken's optional Triton dependencies.
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet

    signature = inspect.signature(Qwen3_5GatedDeltaNet.__init__)
    assert signature.parameters["output_gate_type"].default == "silu"

    common = dict(
        hidden_size=16,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=4,
        head_v_dim=4,
        conv_kernel_size=3,
        rms_norm_eps=1e-6,
        layer_id=0,
    )
    legacy = Qwen3_5GatedDeltaNet(**common)
    qwen4 = Qwen3_5GatedDeltaNet(**common, output_gate_type="sigmoid")

    assert legacy.norm.activation == "silu"
    assert qwen4.norm.activation == "sigmoid"
    with pytest.raises(ValueError, match="must be 'silu' or 'sigmoid'"):
        Qwen3_5GatedDeltaNet(**common, output_gate_type="gelu")


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_qwen35_fused_output_gate_matches_torch_oracle(monkeypatch, activation):
    from freetoken.models.qwen3_5_moe.gdn import _GatedRMSNorm
    import freetoken.kernel.fla as fla

    calls = []

    def fake_rms_norm_gated(
        *, x, weight, bias, z, eps, is_rms_norm, norm_before_gate, activation
    ):
        calls.append((bias, is_rms_norm, norm_before_gate, activation))
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + eps
        )
        gate = F.silu(z.float()) if activation == "silu" else torch.sigmoid(z.float())
        return (normalized * weight.float() * gate).to(x.dtype)

    monkeypatch.setattr(fla, "rms_norm_gated", fake_rms_norm_gated)
    norm = _GatedRMSNorm(dim=4, eps=2e-5, activation=activation)
    norm.weight.copy_(torch.tensor([0.75, 1.0, 1.25, 1.5]))
    x = torch.tensor([[0.25, -0.5, 1.25, -2.0], [1.0, 0.5, -0.25, 0.75]])
    z = torch.tensor([[-1.0, -0.1, 0.4, 1.5], [2.0, -2.0, 0.3, -0.7]])

    got = norm.forward(x, z)
    normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 2e-5)
    gate = F.silu(z) if activation == "silu" else torch.sigmoid(z)
    expected = normalized * norm.weight * gate

    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    assert calls == [(None, True, True, activation)]
