"""Lifecycle for very large, non-expert checkpoint banks.

Some model weights are lookup stores rather than ordinary accelerator-resident
parameters.  Treating them as a ``state_dict`` entry makes the generic loader
materialize the whole tensor and then copy it to the GPU.  Qwen4-Exp's FP8 PLE
table is the motivating case (~51 GB), but this interface is intentionally
model-neutral.

A model may expose ``setup_auxiliary_banks(model_path, device, dtype)`` and
return closeable bank objects.  The model is responsible for binding those
banks to its operators; the engine owns their lifetime and closes them during
shutdown.  With no hook this module is a no-op, preserving every existing model.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class CloseableAuxiliaryBank(Protocol):
    def close(self) -> None: ...


class AuxiliaryBankSet(Mapping[str, Any]):
    """Named banks with deterministic, idempotent reverse-order teardown."""

    def __init__(self, banks: Mapping[str, Any] | None = None):
        self._banks = dict(banks or {})
        self._closed = False

    def __getitem__(self, name: str) -> Any:
        return self._banks[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._banks)

    def __len__(self) -> int:
        return len(self._banks)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for bank in reversed(tuple(self._banks.values())):
            close = getattr(bank, "close", None)
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:  # close every bank before surfacing one error
                if first_error is None:
                    first_error = exc
        self._banks.clear()
        if first_error is not None:
            raise first_error


def setup_auxiliary_banks(
    model: Any,
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
) -> AuxiliaryBankSet:
    hook = getattr(model, "setup_auxiliary_banks", None)
    if hook is None:
        return AuxiliaryBankSet()
    result = hook(model_path=model_path, device=device, dtype=dtype, dummy=dummy)
    if result is None:
        return AuxiliaryBankSet()
    if isinstance(result, AuxiliaryBankSet):
        return result
    if not isinstance(result, Mapping):
        raise TypeError(
            "setup_auxiliary_banks must return a mapping, AuxiliaryBankSet or None, "
            f"got {type(result).__name__}"
        )
    return AuxiliaryBankSet(result)


__all__ = ["AuxiliaryBankSet", "CloseableAuxiliaryBank", "setup_auxiliary_banks"]
