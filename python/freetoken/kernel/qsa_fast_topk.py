"""Bounded-width QSA block top-k with an SM120 CUDA radix-select path."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from freetoken.utils import init_logger

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module


QSA_BLOCK_TOPK = 512
logger = init_logger(__name__)
_native_failure_reported = False
SelectorDispatch = Literal["native", "fallback", "error"]
SelectorTelemetry = Callable[[SelectorDispatch], None]


@dataclass(frozen=True)
class QSAFastTopKCapability:
    """Result of compiling and synchronously exercising the native selector."""

    production_ready: bool
    detail: str


@functools.cache
def _jit_qsa_fast_topk_module() -> Module:
    args = make_cpp_args(QSA_BLOCK_TOPK, False)
    return load_jit(
        "qsa_fast_topk",
        *args,
        cuda_files=["qsa_fast_topk.cuh"],
        cuda_wrappers=[
            ("launch", f"QSAFastTopKKernel<{args}>::run"),
        ],
    )


def _torch_topk(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    rows, width = scores.shape
    columns = torch.arange(width, device=scores.device).view(1, -1)
    eligible = columns < lengths.to(device=scores.device, dtype=torch.long).view(-1, 1)
    masked = scores.masked_fill(~eligible, -torch.inf)
    if width < QSA_BLOCK_TOPK:
        masked = torch.nn.functional.pad(
            masked, (0, QSA_BLOCK_TOPK - width), value=-torch.inf
        )
    values, indices = torch.topk(masked, k=QSA_BLOCK_TOPK, dim=1)
    return torch.where(
        torch.isfinite(values),
        indices.to(torch.int32),
        torch.full_like(indices, -1, dtype=torch.int32),
    )


def _dispatch_native_or_fallback(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    *,
    require_native: bool,
    telemetry: SelectorTelemetry | None,
) -> torch.Tensor:
    """Launch native top-k and apply the explicitly configured failure policy."""

    try:
        _jit_qsa_fast_topk_module().launch(scores, lengths, output)
    except (ImportError, OSError, RuntimeError) as exc:
        if telemetry is not None:
            telemetry("error")
        if require_native:
            detail = next(
                (line.strip() for line in str(exc).splitlines() if line.strip()),
                type(exc).__name__,
            )
            raise RuntimeError(
                "native SM120 QSA fast-topk is required but its JIT/launch "
                f"failed: {detail}"
            ) from exc

        global _native_failure_reported
        if not _native_failure_reported:
            _native_failure_reported = True
            detail = next(
                (line.strip() for line in str(exc).splitlines() if line.strip()),
                type(exc).__name__,
            )
            logger.warning_rank0(
                "SM120 QSA radix-select JIT is unavailable; using torch.topk "
                f"over the bounded score workspace ({detail})"
            )
        if telemetry is not None:
            telemetry("fallback")
        return _torch_topk(scores, lengths)

    if telemetry is not None:
        telemetry("native")
    return output


def qsa_fast_topk(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    *,
    require_native: bool = False,
    telemetry: SelectorTelemetry | None = None,
) -> torch.Tensor:
    """Select 512 block ids per fp32 score row.

    ``scores`` is the caller-owned bounded workspace; no secondary score matrix
    is allocated.  The CUDA fast path is the adapted radix-select kernel.  A
    vectorized Torch path remains available for CPU tests and as a visible
    bring-up fallback when runtime JIT compilation is unavailable.
    """
    if scores.ndim != 2 or scores.dtype != torch.float32:
        raise ValueError("QSA fast top-k expects a rank-2 float32 score matrix")
    if lengths.ndim != 1 or lengths.numel() != scores.shape[0]:
        raise ValueError("QSA fast top-k lengths must contain one value per score row")
    # Avoid synchronizing every CUDA selector call merely to validate values
    # produced by the backend.  CPU/debug callers still get the eager range
    # check; the native kernel defensively clamps lengths below.
    if not lengths.is_cuda and (
        torch.any(lengths < 0) or torch.any(lengths > scores.shape[1])
    ):
        raise ValueError("QSA fast top-k lengths must be within the score width")
    if not scores.is_cuda:
        if require_native:
            if telemetry is not None:
                telemetry("error")
            raise RuntimeError(
                "native SM120 QSA fast-topk is required but scores are not on CUDA"
            )
        return _torch_topk(scores, lengths)

    lengths = lengths.to(device=scores.device, dtype=torch.int32).contiguous()
    output = torch.empty(
        (scores.shape[0], QSA_BLOCK_TOPK),
        dtype=torch.int32,
        device=scores.device,
    )
    return _dispatch_native_or_fallback(
        scores,
        lengths,
        output,
        require_native=require_native,
        telemetry=telemetry,
    )


def probe_qsa_fast_topk_native(device: int | str | torch.device = 0) -> QSAFastTopKCapability:
    """Compile, launch, synchronize and verify the native SM120 selector."""

    try:
        if not torch.cuda.is_available():
            return QSAFastTopKCapability(False, "torch.cuda.is_available() is false")
        if isinstance(device, torch.device):
            cuda_device = device
        elif isinstance(device, str):
            cuda_device = torch.device(device)
        else:
            cuda_device = torch.device(f"cuda:{device}")
        major, minor = torch.cuda.get_device_capability(cuda_device)
        if major != 12:
            return QSAFastTopKCapability(
                False, f"SM {major}.{minor} is not the required SM120 family"
            )
        scores = torch.arange(1024, dtype=torch.float32, device=cuda_device).view(1, -1)
        lengths = torch.tensor([1024], dtype=torch.int32, device=cuda_device)
        selected = qsa_fast_topk(scores, lengths, require_native=True)
        torch.cuda.synchronize(cuda_device)
        observed = torch.sort(selected[0]).values
        expected = torch.arange(512, 1024, dtype=torch.int32, device=cuda_device)
        if not torch.equal(observed, expected):
            return QSAFastTopKCapability(
                False, "native selector returned an incorrect deterministic probe result"
            )
        return QSAFastTopKCapability(
            True,
            "native SM120 fast-topk JIT, launch, synchronize and parity passed "
            f"on SM {major}.{minor}",
        )
    except Exception as exc:
        detail = next(
            (line.strip() for line in str(exc).splitlines() if line.strip()),
            type(exc).__name__,
        )
        return QSAFastTopKCapability(False, f"{type(exc).__name__}: {detail[:240]}")


__all__ = [
    "QSA_BLOCK_TOPK",
    "QSAFastTopKCapability",
    "probe_qsa_fast_topk_native",
    "qsa_fast_topk",
]
