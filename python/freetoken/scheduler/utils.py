from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams
    from freetoken.multimodal import ImageInputs, MMEmbeddingPlan

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    image_inputs: ImageInputs | None = None
    mm_plan: MMEmbeddingPlan | None = None
    mrope_positions: torch.Tensor | None = None
    mrope_delta: int = 0

    @property
    def is_multimodal(self) -> bool:
        return self.mm_embeds is not None or self.image_inputs is not None or self.mm_plan is not None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
