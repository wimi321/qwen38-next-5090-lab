from __future__ import annotations

import base64
import struct
import zlib
from types import SimpleNamespace

import torch
import pytest

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.multimodal import (
    MAX_IMAGE_DIMENSION,
    MAX_VISION_PATCHES,
    ImageInputs,
    MediaPayload,
)
from freetoken.tokenizer.tokenize import TokenizeManager, _decode_image, build_image_mrope


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _png_with_dimensions(width: int, height: int) -> bytes:
    """Change only IHDR geometry; safety checks run before invalid IDAT is loaded."""

    encoded = bytearray(_PNG)
    encoded[16:20] = struct.pack(">I", width)
    encoded[20:24] = struct.pack(">I", height)
    encoded[29:33] = struct.pack(">I", zlib.crc32(encoded[12:29]) & 0xFFFFFFFF)
    return bytes(encoded)


class _Tokenizer:
    name_or_path = "fake"
    chat_template = "present"


class _Processor:
    image_processor = SimpleNamespace(merge_size=2)

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["content"][0] == {"type": "image"}
        assert kwargs["tokenize"] is False
        return "<vision> describe"

    def __call__(self, **kwargs):
        assert kwargs["text"] == ["<vision> describe"]
        assert len(kwargs["images"]) == 1
        assert kwargs["add_special_tokens"] is False
        return {
            "input_ids": torch.tensor([[10, 20, 20, 20, 20, 11]], dtype=torch.int64),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.int64),
            "pixel_values": torch.arange(24, dtype=torch.float32).reshape(2, 12),
            "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.int64),
        }


def test_processor_bridge_returns_self_contained_image_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freetoken.tokenizer.tokenize as tokenize_module

    monkeypatch.setattr(
        tokenize_module,
        "_decode_image",
        lambda _payload: SimpleNamespace(width=1, height=1),
    )
    manager = TokenizeManager(_Tokenizer(), processor=_Processor())
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]}],
        sampling_params=SamplingParams(),
        media=[MediaPayload("image/png", _PNG, "data")],
    )

    encoded = manager.encode(msg)

    assert isinstance(encoded, ImageInputs)
    assert encoded.input_ids.tolist() == [10, 20, 20, 20, 20, 11]
    assert encoded.mm_token_type_ids.tolist() == [0, 1, 1, 1, 1, 0]
    assert [(span.image_index, span.start, span.end) for span in encoded.image_spans] == [
        (0, 1, 5)
    ]
    assert encoded.image_tokens == 4 and encoded.text_tokens == 2
    assert encoded.mrope_positions.shape == (3, 6)
    assert encoded.rope_delta.shape == (1,)


def test_decode_rejects_oversized_dimensions_before_pixel_materialization() -> None:
    payload = MediaPayload(
        "image/png",
        _png_with_dimensions(MAX_IMAGE_DIMENSION + 1, 1),
        "data",
    )

    with pytest.raises(ValueError, match="dimensions"):
        _decode_image(payload)


def test_decode_converts_pillow_decompression_bomb_to_request_error() -> None:
    payload = MediaPayload("image/png", _png_with_dimensions(10_000, 10_000), "data")

    with pytest.raises(ValueError, match="pixel"):
        _decode_image(payload)


def test_decode_rejects_declared_mime_mismatch() -> None:
    payload = MediaPayload("image/jpeg", _PNG, "data")

    with pytest.raises(ValueError, match="does not match"):
        _decode_image(payload)


class _OversizedGridProcessor(_Processor):
    def __call__(self, **kwargs):
        batch = super().__call__(**kwargs)
        # Divisible by merge-size two, but 258^2 exceeds 65,536 patches.
        batch["image_grid_thw"] = torch.tensor([[1, 258, 258]], dtype=torch.int64)
        return batch


def _image_message() -> TokenizeMsg:
    return TokenizeMsg(
        uid=2,
        text=[{"role": "user", "content": [{"type": "image"}]}],
        sampling_params=SamplingParams(),
        media=[MediaPayload("image/png", _PNG, "data")],
    )


def test_processor_rejects_patch_count_above_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freetoken.tokenizer.tokenize as tokenize_module

    monkeypatch.setattr(
        tokenize_module,
        "_decode_image",
        lambda _payload: SimpleNamespace(width=1, height=1),
    )
    manager = TokenizeManager(_Tokenizer(), processor=_OversizedGridProcessor())

    with pytest.raises(ValueError, match=f"limit is {MAX_VISION_PATCHES}"):
        manager.encode(_image_message())


def test_grid_validator_enforces_expanded_image_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freetoken.multimodal as multimodal_module

    monkeypatch.setattr(multimodal_module, "MAX_IMAGE_TOKENS", 3)
    with pytest.raises(ValueError, match="expanded image tokens"):
        multimodal_module.image_grid_totals(torch.tensor([[1, 4, 4]]), 2)


@pytest.mark.parametrize(("limit", "accepted"), [(216, True), (215, False)])
def test_processor_tensor_byte_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch, limit: int, accepted: bool
) -> None:
    import freetoken.tokenizer.tokenize as tokenize_module

    monkeypatch.setattr(
        tokenize_module,
        "_decode_image",
        lambda _payload: SimpleNamespace(width=1, height=1),
    )
    monkeypatch.setattr(tokenize_module, "MAX_VISION_TENSOR_BYTES", limit)
    manager = TokenizeManager(_Tokenizer(), processor=_Processor())

    if accepted:
        assert isinstance(manager.encode(_image_message()), ImageInputs)
    else:
        with pytest.raises(ValueError, match="tensor bytes"):
            manager.encode(_image_message())


def test_decoded_request_pixel_budget_is_checked_before_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freetoken.tokenizer.tokenize as tokenize_module

    monkeypatch.setattr(
        tokenize_module,
        "_decode_image",
        lambda _payload: SimpleNamespace(width=4096, height=4096),
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(5)],
        }
    ]
    msg = TokenizeMsg(
        uid=3,
        text=messages,
        sampling_params=SamplingParams(),
        media=[MediaPayload("image/png", _PNG, "data") for _ in range(5)],
    )
    manager = TokenizeManager(_Tokenizer(), processor=_Processor())

    with pytest.raises(ValueError, match="pixels in total"):
        manager.encode(msg)


def test_image_mrope_matches_grid_and_rejects_video_token_types() -> None:
    types = torch.tensor([0, 0, 1, 1, 1, 1, 0], dtype=torch.int32)
    positions, delta, spans = build_image_mrope(
        types, torch.tensor([[1, 4, 4]]), spatial_merge_size=2
    )
    assert positions[:, :2].tolist() == [[0, 1], [0, 1], [0, 1]]
    assert positions[:, 2:6].tolist() == [[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]]
    assert spans[0].start == 2 and spans[0].end == 6
    assert delta.item() == positions.max().item() + 1 - len(types)

    try:
        build_image_mrope(torch.tensor([2]), torch.tensor([[1, 1, 1]]), spatial_merge_size=1)
    except ValueError as exc:
        assert "images only" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("video token type was accepted")


def test_image_mrope_positions_match_transformers_qwen4_exp_reference() -> None:
    pytest.importorskip("transformers")
    from transformers import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

    # Instantiate a genuinely tiny HF model (zero language/vision blocks), so
    # parity calls the public implementation with its real composite config but
    # allocates only a few KiB rather than the checkpoint architecture.
    config = Qwen4ExpConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": [],
            "hc_count": 4,
            "hc_lowrank": 8,
            "ple_layer_ids": [],
        },
        vision_config={
            "depth": 0,
            "hidden_size": 32,
            "out_hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "num_position_embeddings": 64,
        },
    )
    reference = Qwen4ExpModel(config)
    input_ids = torch.arange(13, dtype=torch.int64)
    mm_types = torch.tensor([0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0])
    grids = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.int64)

    expected_positions, expected_delta = Qwen4ExpModel.get_rope_index(
        reference,
        input_ids.unsqueeze(0),
        mm_types.unsqueeze(0),
        image_grid_thw=grids.clone(),
    )
    positions, delta, spans = build_image_mrope(
        mm_types, grids, spatial_merge_size=2
    )

    torch.testing.assert_close(positions, expected_positions[:, 0])
    torch.testing.assert_close(delta.reshape(1, 1), expected_delta)
    assert [(span.start, span.end) for span in spans] == [(2, 6), (7, 11)]
