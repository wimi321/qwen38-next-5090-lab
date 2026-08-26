"""Per-request token-history and dilated-convolution state for Qwen4 PLE."""

from __future__ import annotations

import torch


class PLEStatePool:
    """Fixed request-table-indexed PLE state.

    PLE is currently restricted to the naive prefix cache, so the scheduler's
    stable ``Req.table_idx`` is also the state slot.  Keeping this as a separate
    pool makes that limitation explicit and leaves a clear future seam for
    radix snapshot/copy-on-write semantics.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        context_len: int,
        channels: int,
        conv_state_len: int,
        eos_token_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if num_slots <= 0 or context_len <= 0 or channels <= 0 or conv_state_len <= 0:
            raise ValueError(
                "PLEStatePool dimensions must be positive, got "
                f"{num_slots=}, {context_len=}, {channels=}, {conv_state_len=}"
            )
        self.num_slots = int(num_slots)
        self.context_len = int(context_len)
        self.channels = int(channels)
        self.conv_state_len = int(conv_state_len)
        self.eos_token_id = int(eos_token_id)
        self.device = device
        self.dtype = dtype
        # Hashing and mmap lookup are CPU work.  The tiny history stays on CPU;
        # pin it when CUDA is available so batched diagnostics/copies remain cheap.
        self.token_history = torch.full(
            (num_slots, context_len),
            eos_token_id,
            dtype=torch.long,
            device="cpu",
            pin_memory=device.type == "cuda" and torch.cuda.is_available(),
        )
        self.conv_states = torch.zeros(
            num_slots,
            channels,
            conv_state_len,
            dtype=dtype,
            device=device,
        )
        # Decode runs may be captured in a CUDA graph, so the convolution result
        # has to be written by device-side ops during replay.  Keep that result in
        # a separate slot-indexed bank: ``conv_states`` is the committed request
        # state and must remain unchanged until the *whole* model forward has
        # succeeded.  A failed forward may leave arbitrary values here; the next
        # replay overwrites its live slots before they can be committed.
        self.pending_conv_states = torch.zeros_like(self.conv_states)
        self.initialized = torch.zeros(num_slots, dtype=torch.bool, device="cpu")

    def _check_slot(self, slot: int) -> int:
        slot = int(slot)
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"PLE state slot {slot} outside [0, {self.num_slots})")
        return slot

    def clear(self, slot: int) -> None:
        slot = self._check_slot(slot)
        self.token_history[slot].fill_(self.eos_token_id)
        self.conv_states[slot].zero_()
        self.pending_conv_states[slot].zero_()
        self.initialized[slot] = False

    def begin(self, slot: int, *, fresh: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the committed state without changing it.

        A fresh request may reuse a scheduler slot whose previous request state
        is still committed.  Treat that state as logically reset for this
        forward, but do not clear it until the new request has completed the
        whole model successfully.  This makes eager prefill retryable under the
        same transaction boundary as graph/eager decode.
        """

        slot = self._check_slot(slot)
        if fresh or not bool(self.initialized[slot]):
            return (
                torch.full_like(self.token_history[slot], self.eos_token_id),
                torch.zeros_like(self.conv_states[slot]),
            )
        return self.token_history[slot], self.conv_states[slot]

    def stage_token_history(
        self, slot: int, *, fresh: bool
    ) -> tuple[torch.Tensor, bool]:
        """Return hash history for graph-external PLE staging.

        Decode row lookup happens before CUDA-graph replay.  It must not clear
        ``conv_states`` on the host: the slot id and reset decision are dynamic
        graph inputs, and clearing a captured dummy slot would otherwise be
        replayed for real requests.  The returned bool is therefore consumed by
        the captured GPU path to mask the gathered convolution state.

        This method deliberately does not mark the slot initialized.  The CPU
        history is committed only after the corresponding model forward
        succeeds.
        """

        slot = self._check_slot(slot)
        reset = bool(fresh or not bool(self.initialized[slot]))
        if reset:
            history = torch.full_like(self.token_history[slot], self.eos_token_id)
        else:
            history = self.token_history[slot]
        return history, reset

    def commit_token_history(self, slot: int, token_history: torch.Tensor) -> None:
        """Commit the graph-external half of a successful decode step."""

        slot = self._check_slot(slot)
        if token_history.shape != (self.context_len,):
            raise ValueError(
                f"token history must have shape {(self.context_len,)}, got "
                f"{tuple(token_history.shape)}"
            )
        self.token_history[slot].copy_(
            token_history.to(device="cpu", dtype=torch.long)
        )
        self.initialized[slot] = True

    def gather_decode_conv_states(
        self,
        table_idx: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized, CUDA-graph-safe decode state gather and reset.

        ``table_idx`` and ``reset_mask`` are expected to be persistent graph
        input tensors on CUDA.  Their values may change between replays while
        their addresses stay fixed.
        """

        if table_idx.ndim != 1 or reset_mask.shape != table_idx.shape:
            raise ValueError(
                "PLE decode table_idx/reset_mask must both have shape [batch], got "
                f"{tuple(table_idx.shape)} and {tuple(reset_mask.shape)}"
            )
        indices = table_idx.to(device=self.device, dtype=torch.long)
        if indices.numel() and self.device.type == "cpu":
            minimum, maximum = indices.min().item(), indices.max().item()
            if minimum < 0 or maximum >= self.num_slots:
                raise IndexError(
                    f"PLE state slot range [{minimum}, {maximum}] outside "
                    f"[0, {self.num_slots})"
                )
        states = self.conv_states.index_select(0, indices)
        reset = reset_mask.to(device=self.device, dtype=torch.bool).view(-1, 1, 1)
        return torch.where(reset, torch.zeros_like(states), states)

    def write_pending_decode_conv_states(
        self,
        table_idx: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> None:
        """Stage a decode result without mutating committed request state.

        This method is part of the captured model forward.  Publishing to
        :attr:`conv_states` here would advance PLE even when a later model layer
        raises, so graph/eager forwards write only the pending bank.
        """

        if table_idx.ndim != 1:
            raise ValueError(
                f"PLE decode table_idx must have shape [batch], got {tuple(table_idx.shape)}"
            )
        expected = (table_idx.numel(), self.channels, self.conv_state_len)
        if conv_state.shape != expected:
            raise ValueError(
                f"PLE pending decode conv state must have shape {expected}, got "
                f"{tuple(conv_state.shape)}"
            )
        indices = table_idx.to(device=self.device, dtype=torch.long)
        self.pending_conv_states.index_copy_(
            0, indices, conv_state.to(device=self.device, dtype=self.dtype)
        )

    def write_pending_prefill_conv_states(
        self,
        slots: tuple[int, ...],
        conv_states: torch.Tensor,
    ) -> torch.Tensor:
        """Stage ragged-prefill convolution states and return their table ids.

        Prefill is eager, but it has the same retry contract as decode: a later
        decoder layer or expert watchdog may still fail.  The returned tensor is
        retained on the batch and passed to :meth:`commit_pending_decode` only
        from the engine's successful-forward hook.
        """

        checked_slots = tuple(self._check_slot(slot) for slot in slots)
        expected = (len(checked_slots), self.channels, self.conv_state_len)
        if conv_states.shape != expected:
            raise ValueError(
                f"PLE pending prefill conv states must have shape {expected}, got "
                f"{tuple(conv_states.shape)}"
            )
        table_idx = torch.tensor(
            checked_slots,
            dtype=torch.long,
            device=self.device,
        )
        self.write_pending_decode_conv_states(table_idx, conv_states)
        return table_idx

    def commit_pending_decode_conv_states(self, table_idx: torch.Tensor) -> None:
        """Publish staged decode convolution states after model success.

        ``table_idx`` contains only real requests, never graph-padding slots.
        The copy intentionally lives outside CUDA graph replay so an exception
        anywhere after PLE leaves :attr:`conv_states` untouched.
        """

        if table_idx.ndim != 1:
            raise ValueError(
                f"PLE decode table_idx must have shape [batch], got {tuple(table_idx.shape)}"
            )
        indices = table_idx.to(device=self.device, dtype=torch.long)
        if indices.numel() and self.device.type == "cpu":
            minimum, maximum = indices.min().item(), indices.max().item()
            if minimum < 0 or maximum >= self.num_slots:
                raise IndexError(
                    f"PLE state slot range [{minimum}, {maximum}] outside "
                    f"[0, {self.num_slots})"
                )
        staged = self.pending_conv_states.index_select(0, indices)
        self.conv_states.index_copy_(0, indices, staged)

    def commit_pending_decode(
        self,
        table_idx: torch.Tensor,
        slots: tuple[int, ...],
        token_histories: torch.Tensor,
    ) -> None:
        """Atomically validate, then publish both halves of decode state."""

        if token_histories.ndim != 2 or token_histories.shape != (
            len(slots),
            self.context_len,
        ):
            raise ValueError(
                "PLE pending token histories must have shape "
                f"{(len(slots), self.context_len)}, got {tuple(token_histories.shape)}"
            )
        if table_idx.ndim != 1 or table_idx.numel() != len(slots):
            raise ValueError(
                "PLE pending table_idx must contain one active slot per history, got "
                f"{tuple(table_idx.shape)} for {len(slots)} histories"
            )
        checked_slots = tuple(self._check_slot(slot) for slot in slots)
        if self.device.type == "cpu" and table_idx.tolist() != list(checked_slots):
            raise ValueError(
                "PLE pending GPU/CPU slot order differs: "
                f"{table_idx.tolist()} != {list(checked_slots)}"
            )

        # All shape/range/order checks precede mutation.  Once the convolution
        # copy succeeds, the prevalidated history copies cannot fail on shape.
        self.commit_pending_decode_conv_states(table_idx)
        for index, slot in enumerate(checked_slots):
            self.commit_token_history(slot, token_histories[index])

    def commit(
        self,
        slot: int,
        token_history: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> None:
        slot = self._check_slot(slot)
        if token_history.shape != (self.context_len,):
            raise ValueError(
                f"token history must have shape {(self.context_len,)}, got {tuple(token_history.shape)}"
            )
        if conv_state.shape != (self.channels, self.conv_state_len):
            raise ValueError(
                "conv state must have shape "
                f"{(self.channels, self.conv_state_len)}, got {tuple(conv_state.shape)}"
            )
        self.token_history[slot].copy_(token_history.to(device="cpu", dtype=torch.long))
        self.conv_states[slot].copy_(conv_state.to(device=self.device, dtype=self.dtype))
        self.initialized[slot] = True


__all__ = ["PLEStatePool"]
