from __future__ import annotations

import torch

from freetoken.models.auxiliary import AuxiliaryBankSet, setup_auxiliary_banks


class _Bank:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def close(self):
        self.events.append(self.name)


def test_auxiliary_bank_lifecycle_is_optional_and_reverse_order():
    empty = setup_auxiliary_banks(
        object(), "unused", device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert not empty

    events = []

    class Owner:
        def setup_auxiliary_banks(self, **kwargs):
            assert kwargs["model_path"] == "checkpoint"
            return {"first": _Bank(events, "first"), "second": _Bank(events, "second")}

    banks = setup_auxiliary_banks(
        Owner(), "checkpoint", device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert isinstance(banks, AuxiliaryBankSet)
    assert tuple(banks) == ("first", "second")
    banks.close()
    banks.close()
    assert events == ["second", "first"]
