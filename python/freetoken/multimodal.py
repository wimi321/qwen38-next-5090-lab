# Modified by Qwen3.8 Next 5090 Lab contributors in 2026; see MODIFICATIONS.md.
"""Transport-neutral image inputs and chunkable multimodal embedding plans.

The online frontend transports *validated bytes* to the tokenizer worker.  The
worker turns those bytes into :class:`ImageInputs`; a model-owned vision encoder
then produces :class:`MMEmbeddingPlan`.  Keeping these stages separate prevents
the scheduler from knowing about PIL/Transformers while still allowing a long
prompt's image features to be scattered one prefill chunk at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# Request-level resource ceilings.  Byte-size limits on the encoded payload live
# in ``server.media``; these bounds cover the much larger decoded/processed
# representation before it can become a CPU- or GPU-memory denial of service.
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 16_777_216
MAX_TOTAL_IMAGE_PIXELS = 4 * MAX_IMAGE_PIXELS
MAX_VISION_PATCHES = 65_536
MAX_IMAGE_TOKENS = 16_384
MAX_VISION_TENSOR_BYTES = 512 * 1024**2


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return dense tensor storage required by a processor output."""

    return int(tensor.numel()) * int(tensor.element_size())


def image_grid_totals(image_grid_thw: torch.Tensor, merge_size: int) -> tuple[int, int]:
    """Validate image-grid geometry and return ``(patches, merged tokens)``."""

    if merge_size <= 0:
        raise ValueError("spatial merge size must be positive")
    if image_grid_thw.dim() != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [N, 3]")
    patches = 0
    tokens = 0
    for raw in image_grid_thw.detach().cpu().to(torch.int64).tolist():
        grid_t, grid_h, grid_w = (int(value) for value in raw)
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"invalid image grid {(grid_t, grid_h, grid_w)}")
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError(
                f"image grid {(grid_t, grid_h, grid_w)} is not divisible by "
                f"merge size {merge_size}"
            )
        patches += grid_t * grid_h * grid_w
        tokens += grid_t * (grid_h // merge_size) * (grid_w // merge_size)
        if patches > MAX_VISION_PATCHES:
            raise ValueError(
                f"image processor produced {patches} vision patches; "
                f"limit is {MAX_VISION_PATCHES}"
            )
        if tokens > MAX_IMAGE_TOKENS:
            raise ValueError(
                f"image processor produced {tokens} expanded image tokens; "
                f"limit is {MAX_IMAGE_TOKENS}"
            )
    return patches, tokens


@dataclass(frozen=True)
class MediaPayload:
    """One image fetched and validated by the API frontend.

    ``source`` is deliberately only ``"data"`` or ``"https"``.  The original
    URL is not sent to worker processes or retained with the request.
    """

    mime_type: str
    data: bytes
    source: str


@dataclass(frozen=True)
class ImageTokenSpan:
    """Absolute half-open token interval occupied by one image."""

    image_index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.image_index < 0 or self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"invalid image token span: image={self.image_index}, [{self.start}, {self.end})"
            )


@dataclass
class ImageInputs:
    """CPU processor output for one request.

    ``input_ids`` are already expanded by the processor, so image soft tokens
    count toward the same context limit as text and generated tokens.
    ``mrope_positions`` always has shape ``[3, L]`` and ``rope_delta`` contains
    one scalar.  Video tensors are intentionally absent from the v0.2 contract.
    """

    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    mm_token_type_ids: torch.Tensor
    mrope_positions: torch.Tensor
    rope_delta: torch.Tensor
    image_spans: list[ImageTokenSpan] = field(default_factory=list)

    def __post_init__(self) -> None:
        tensors = {
            "input_ids": self.input_ids,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
            "mm_token_type_ids": self.mm_token_type_ids,
            "mrope_positions": self.mrope_positions,
            "rope_delta": self.rope_delta,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cpu:
                raise ValueError(f"ImageInputs.{name} must be a CPU tensor")
        if self.input_ids.dim() != 1:
            raise ValueError("ImageInputs.input_ids must have shape [L]")
        length = int(self.input_ids.numel())
        if self.mm_token_type_ids.shape != (length,):
            raise ValueError("ImageInputs.mm_token_type_ids must have shape [L]")
        if self.mrope_positions.shape != (3, length):
            raise ValueError("ImageInputs.mrope_positions must have shape [3, L]")
        if self.image_grid_thw.dim() != 2 or self.image_grid_thw.shape[1] != 3:
            raise ValueError("ImageInputs.image_grid_thw must have shape [N, 3]")
        if self.rope_delta.numel() != 1:
            raise ValueError("ImageInputs.rope_delta must contain one scalar")
        if len(self.image_spans) != int(self.image_grid_thw.shape[0]):
            raise ValueError("one ImageTokenSpan is required for every image_grid_thw row")
        total_bytes = sum(tensor_nbytes(tensor) for tensor in tensors.values())
        if total_bytes > MAX_VISION_TENSOR_BYTES:
            raise ValueError(
                f"ImageInputs tensors require {total_bytes} bytes; "
                f"limit is {MAX_VISION_TENSOR_BYTES}"
            )
        # Qwen4-Exp uses a fixed spatial merge of two.  The tokenizer performs
        # the authoritative processor-specific check; this second boundary
        # prevents a deserialized worker message from bypassing the patch cap.
        _patches, grid_tokens = image_grid_totals(self.image_grid_thw, 2)
        if self.image_tokens > MAX_IMAGE_TOKENS:
            raise ValueError(
                f"ImageInputs contain {self.image_tokens} image tokens; "
                f"limit is {MAX_IMAGE_TOKENS}"
            )
        if self.image_tokens != grid_tokens:
            raise ValueError(
                f"ImageInputs spans contain {self.image_tokens} image tokens, but grids "
                f"describe {grid_tokens}"
            )

    @property
    def image_tokens(self) -> int:
        return sum(span.end - span.start for span in self.image_spans)

    @property
    def text_tokens(self) -> int:
        return int(self.input_ids.numel()) - self.image_tokens


@dataclass(frozen=True)
class MMEmbeddingSpan:
    """Map an absolute request token interval to rows of a feature tensor."""

    image_index: int
    start: int
    end: int
    feature_start: int
    feature_end: int

    def __post_init__(self) -> None:
        token_len = self.end - self.start
        feature_len = self.feature_end - self.feature_start
        if (
            self.image_index < 0
            or self.start < 0
            or token_len <= 0
            or self.feature_start < 0
            or feature_len != token_len
        ):
            raise ValueError(
                "MMEmbeddingSpan must map equal non-empty token/feature intervals, got "
                f"tokens [{self.start}, {self.end}) and features "
                f"[{self.feature_start}, {self.feature_end})"
            )


@dataclass(frozen=True)
class MMChunk:
    """Features and token offsets selected for one request-local prefill chunk."""

    features: torch.Tensor
    token_indices: torch.Tensor
    placeholder_mask: torch.Tensor


@dataclass
class MMEmbeddingPlan:
    """Pageable-CPU vision features plus absolute request token mappings."""

    features: torch.Tensor
    spans: list[MMEmbeddingSpan]

    def __post_init__(self) -> None:
        if not isinstance(self.features, torch.Tensor) or not self.features.is_cpu:
            raise ValueError("MMEmbeddingPlan.features must be a CPU tensor")
        if self.features.dim() != 2:
            raise ValueError("MMEmbeddingPlan.features must have shape [tokens, hidden]")
        previous_end = 0
        previous_feature_end = 0
        for span in self.spans:
            if span.start < previous_end or span.feature_start < previous_feature_end:
                raise ValueError("MMEmbeddingPlan spans must be sorted and non-overlapping")
            if span.feature_end > self.features.shape[0]:
                raise ValueError("MMEmbeddingPlan span exceeds available feature rows")
            previous_end = span.end
            previous_feature_end = span.feature_end

    @classmethod
    def from_image_spans(
        cls, features: torch.Tensor, image_spans: list[ImageTokenSpan]
    ) -> "MMEmbeddingPlan":
        feature_offset = 0
        spans: list[MMEmbeddingSpan] = []
        for image_span in image_spans:
            length = image_span.end - image_span.start
            spans.append(
                MMEmbeddingSpan(
                    image_index=image_span.image_index,
                    start=image_span.start,
                    end=image_span.end,
                    feature_start=feature_offset,
                    feature_end=feature_offset + length,
                )
            )
            feature_offset += length
        if feature_offset != int(features.shape[0]):
            raise ValueError(
                f"vision feature rows ({features.shape[0]}) do not match image token slots "
                f"({feature_offset})"
            )
        return cls(features=features, spans=spans)

    def select(self, start: int, end: int) -> MMChunk:
        """Return rows intersecting absolute request interval ``[start, end)``.

        Returned ``token_indices`` are local to that interval.  They are suitable
        for adding the request's flattened-batch offset before an ``index_copy_``.
        """

        if start < 0 or end < start:
            raise ValueError(f"invalid prefill interval [{start}, {end})")
        rows: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for span in self.spans:
            overlap_start = max(start, span.start)
            overlap_end = min(end, span.end)
            if overlap_start >= overlap_end:
                continue
            row_start = span.feature_start + overlap_start - span.start
            row_end = row_start + overlap_end - overlap_start
            rows.append(self.features[row_start:row_end])
            indices.append(torch.arange(overlap_start - start, overlap_end - start, dtype=torch.int64))

        chunk_len = end - start
        if not rows:
            return MMChunk(
                features=self.features[:0],
                token_indices=torch.empty(0, dtype=torch.int64),
                placeholder_mask=torch.zeros(chunk_len, dtype=torch.bool),
            )
        token_indices = torch.cat(indices)
        mask = torch.zeros(chunk_len, dtype=torch.bool)
        mask[token_indices] = True
        return MMChunk(torch.cat(rows, dim=0), token_indices, mask)
