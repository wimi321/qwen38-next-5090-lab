from __future__ import annotations

import importlib.util
import io
import itertools
import json
import os
import threading
import warnings
from types import ModuleType
from typing import Any, List

import torch
from freetoken.message import TokenizeMsg
from freetoken.multimodal import (
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    MAX_TOTAL_IMAGE_PIXELS,
    MAX_VISION_TENSOR_BYTES,
    ImageInputs,
    ImageTokenSpan,
    image_grid_totals,
    tensor_nbytes,
)
from freetoken.utils import init_logger
from transformers import PreTrainedTokenizerBase

from .effort import (
    EffortProfile,
    ThinkingProfile,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)

logger = init_logger(__name__)


def resolve_thinking_mode(chat_template_kwargs: dict[str, Any] | None, tools: Any | None) -> str:
    """Resolve the thinking mode (``"thinking"`` or ``"chat"``) for a chat request.

    The single source of truth for this decision: the encode side
    (``_apply_dsv4_chat_encoder`` below) uses it to pick the prompt the model
    sees, and the frontend parse side (``server/openai_api.py``) imports it to
    decide whether the model's output begins inside a reasoning block. Keeping
    one implementation prevents the two sides from disagreeing. Thinking is on
    when tools are offered (dsv4 only emits well-formed tool calls in thinking
    mode) or when the caller requests it via ``chat_template_kwargs``.
    """
    ctk = chat_template_kwargs or {}
    mode = str(ctk.get("thinking_mode") or "chat")
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        mode = "thinking"
    if mode not in ("chat", "thinking"):
        mode = "chat"
    return mode


_EFFORT_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, processor: Any | None = None) -> None:
        self.tokenizer = tokenizer
        self._dsv4_encoder = _load_dsv4_encoder_if_needed(tokenizer)
        self._effort_profile: EffortProfile | None = None
        self._thinking_profile: ThinkingProfile | None = None
        self._effort_lock = threading.Lock()
        self._logged_effort_maps: set[tuple[Any, str | None]] = set()
        self._processor = processor
        self._processor_lock = threading.Lock()

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            encoded = self.encode(msg)
            results.append(encoded.input_ids if isinstance(encoded, ImageInputs) else encoded)
        return results

    def encode(self, msg: TokenizeMsg) -> torch.Tensor | ImageInputs:
        """Encode one request, returning rich processor output for image chats."""

        if msg.media:
            return self._encode_images(msg)
        prompt = self.render_prompt(msg)
        # A jinja chat template owns every special token (HF's apply_chat_template
        # tokenizes with add_special_tokens=False for the same reason): tokenizers
        # that auto-add bos (muse-glimmer's, llama's) would otherwise double it --
        # the template already rendered one. Raw-string prompts and the dsv4
        # encoder path keep the default.
        templated = isinstance(msg.text, list) and self._dsv4_encoder is None
        input_ids: torch.Tensor = (  # type: ignore
            self.tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=not templated
            )
        )
        return input_ids.view(-1).to(torch.int32)

    def render_prompt(self, msg: TokenizeMsg) -> str:
        """The template/encoder half of ``tokenize``, exposed so the frontend can
        validate a request before committing an SSE stream. Sanitizes
        ``reasoning_effort`` first: every render path (worker, frontend
        validation, count_tokens) must quantize identically."""
        if not isinstance(msg.text, list):
            return msg.text
        if msg.media:
            if self._dsv4_encoder is not None:
                raise ValueError("image messages are not supported by the custom DSV4 encoder")
            kwargs = self._sanitize_effort(msg.chat_template_kwargs or {})
            if "reasoning_effort" in kwargs:
                kwargs = dict(kwargs)
                kwargs.setdefault("reasoning_strength", kwargs["reasoning_effort"])
            if msg.tools is not None:
                kwargs = {**kwargs, "tools": msg.tools}
            processor = self._get_processor()
            prompt = processor.apply_chat_template(
                msg.text,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            if not isinstance(prompt, str):
                raise TypeError("multimodal processor chat template did not return text")
            return prompt
        return self._render(
            msg.text, msg.tools, self._sanitize_effort(msg.chat_template_kwargs or {})
        )

    def _get_processor(self) -> Any:
        with self._processor_lock:
            if self._processor is None:
                model_path = getattr(self.tokenizer, "name_or_path", None) or getattr(
                    self.tokenizer, "_name_or_path", ""
                )
                if not model_path:
                    raise RuntimeError("cannot load AutoProcessor: tokenizer has no model path")
                from transformers import AutoProcessor

                self._processor = AutoProcessor.from_pretrained(str(model_path))
            return self._processor

    def _encode_images(self, msg: TokenizeMsg) -> ImageInputs:
        if not isinstance(msg.text, list):
            raise ValueError("image inputs require structured chat messages")
        processor = self._get_processor()
        images = [_decode_image(payload) for payload in msg.media or []]
        decoded_pixels = sum(int(image.width) * int(image.height) for image in images)
        if decoded_pixels > MAX_TOTAL_IMAGE_PIXELS:
            raise ValueError(
                f"decoded images contain {decoded_pixels} pixels in total; "
                f"limit is {MAX_TOTAL_IMAGE_PIXELS}"
            )
        marker_count = sum(
            1
            for message in msg.text
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
            if isinstance(part, dict) and part.get("type") == "image"
        )
        if marker_count != len(images):
            raise ValueError(
                f"image marker count ({marker_count}) does not match image payload count ({len(images)})"
            )
        prompt = self.render_prompt(msg)
        batch = processor(
            images=images,
            text=[prompt],
            return_tensors="pt",
            return_mm_token_type_ids=True,
            add_special_tokens=False,
        )
        required = ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"multimodal processor did not return: {', '.join(missing)}")
        processor_tensor_bytes = sum(
            tensor_nbytes(value) for value in batch.values()
            if isinstance(value, torch.Tensor)
        )
        if processor_tensor_bytes > MAX_VISION_TENSOR_BYTES:
            raise ValueError(
                f"multimodal processor returned {processor_tensor_bytes} tensor bytes; "
                f"limit is {MAX_VISION_TENSOR_BYTES}"
            )
        merge_size = int(
            getattr(processor.image_processor, "merge_size", None)
            or getattr(processor.image_processor, "spatial_merge_size", 0)
        )
        if merge_size <= 0:
            raise ValueError("multimodal processor does not expose a positive spatial merge size")
        grid = batch["image_grid_thw"].detach().cpu().to(torch.int64).reshape(-1, 3).contiguous()
        if int(grid.shape[0]) != len(images):
            raise ValueError(
                f"multimodal processor returned {grid.shape[0]} image grids for "
                f"{len(images)} images"
            )
        image_grid_totals(grid, merge_size)
        input_ids = _single_row(batch["input_ids"], "input_ids").to(torch.int32).contiguous()
        mm_types = _single_row(batch["mm_token_type_ids"], "mm_token_type_ids").to(
            torch.int32
        ).contiguous()
        pixel_values = batch["pixel_values"].detach().cpu().contiguous()
        positions, rope_delta, spans = build_image_mrope(
            mm_types, grid, spatial_merge_size=merge_size
        )
        return ImageInputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            mm_token_type_ids=mm_types,
            mrope_positions=positions,
            rope_delta=rope_delta,
            image_spans=spans,
        )

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any],
    ) -> str:
        """Raw render, no effort sanitation — the probe needs unsupported values
        to actually reach the template so rejection is observable."""
        if self._dsv4_encoder is not None:
            return _apply_dsv4_chat_encoder(
                self._dsv4_encoder, messages, tools, chat_template_kwargs
            )
        # Broadcast the effort in every spelling the ecosystem's templates read
        # (muse-glimmer grades ``reasoning_strength``; Jinja ignores undeclared
        # variables) -- the same rule the thinking toggles use. An explicit
        # caller-provided spelling wins over the broadcast.
        if "reasoning_effort" in chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs.setdefault(
                "reasoning_strength", chat_template_kwargs["reasoning_effort"]
            )
        if tools is not None:
            chat_template_kwargs = {**chat_template_kwargs, "tools": tools}
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        assert isinstance(prompt, str)
        return prompt

    def effort_profile(self) -> EffortProfile:
        """The checkpoint's effort vocabulary, probed on first use and cached
        for the process lifetime."""
        with self._effort_lock:
            if self._effort_profile is None:
                self._effort_profile = probe_effort_profile(self._probe_render)
                logger.info(
                    "reasoning-effort profile: supported=%s default=%s",
                    sorted(self._effort_profile.supported) or "(none)",
                    self._effort_profile.default,
                )
            return self._effort_profile

    def thinking_profile(self) -> ThinkingProfile:
        """The checkpoint's thinking controls (toggle behavior + effort
        vocabulary), probed on first use and cached for the process lifetime.
        Feeds the /v1/cache/status gear derivation."""
        efforts = self.effort_profile()
        with self._effort_lock:
            if self._thinking_profile is None:
                self._thinking_profile = probe_thinking_profile(self._probe_render, efforts)
            return self._thinking_profile

    def _probe_render(
        self, kwargs: dict[str, Any], tools: list[dict[str, Any]] | None
    ) -> str:
        return self._render(_EFFORT_PROBE_MESSAGES, tools, kwargs)

    def _sanitize_effort(self, chat_template_kwargs: dict[str, Any]) -> dict[str, Any]:
        if "reasoning_effort" not in chat_template_kwargs:
            return chat_template_kwargs
        raw = chat_template_kwargs.get("reasoning_effort")
        mapped = quantize_effort(raw, self.effort_profile())
        if mapped == raw:
            return chat_template_kwargs
        # raw is client-controlled and may be unhashable (a JSON list/dict).
        key = (raw if isinstance(raw, str) else repr(raw), mapped)
        if key not in self._logged_effort_maps:
            self._logged_effort_maps.add(key)
            logger.info(
                "reasoning_effort %r is not supported by this checkpoint; using %s",
                raw,
                mapped if mapped is not None else "the template default",
            )
        sanitized = dict(chat_template_kwargs)
        if mapped is None:
            del sanitized["reasoning_effort"]
        else:
            sanitized["reasoning_effort"] = mapped
        return sanitized


def _single_row(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().cpu()
    if value.dim() == 2 and value.shape[0] == 1:
        return value[0]
    if value.dim() == 1:
        return value
    raise ValueError(f"multimodal processor {name} must have shape [1, L]")


def _decode_image(payload: Any) -> Any:
    """Decode a validated payload without allowing lazy file/network reads."""

    from PIL import Image, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            # Pillow's built-in threshold is intentionally high and process
            # global.  Convert its warning to an exception without mutating the
            # global, then enforce the tighter per-request bound below.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload.data)) as image:
                width, height = (int(image.width), int(image.height))
                pixels = width * height
                if (
                    width <= 0
                    or height <= 0
                    or width > MAX_IMAGE_DIMENSION
                    or height > MAX_IMAGE_DIMENSION
                    or pixels > MAX_IMAGE_PIXELS
                ):
                    raise ValueError(
                        f"decoded image dimensions {width}x{height} exceed limits: "
                        f"max side {MAX_IMAGE_DIMENSION}, max pixels {MAX_IMAGE_PIXELS}"
                    )
                detected_mime = Image.MIME.get(image.format or "", "").lower()
                if detected_mime == "image/jpg":
                    detected_mime = "image/jpeg"
                if not detected_mime or detected_mime != payload.mime_type:
                    raise ValueError(
                        f"declared image MIME {payload.mime_type!r} does not match decoded "
                        f"format {detected_mime or '(unknown)'!r}"
                    )
                # Materialize while the in-memory source is alive; select the
                # first frame of an animated format because images, not video,
                # are the v0.2 contract.
                image.seek(0)
                image.load()
                return image.convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError("image exceeds the safe decoded-pixel limit") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"could not decode image: {exc}") from exc


def build_image_mrope(
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[ImageTokenSpan]]:
    """Qwen4-Exp image-only 3-axis position construction.

    This is the CPU equivalent of Transformers' ``Qwen4ExpModel.get_rope_index``
    for a single unpadded request.  Keeping it beside processor expansion makes
    the request self-contained before it crosses into the scheduler.
    """

    if mm_token_type_ids.dim() != 1:
        raise ValueError("mm_token_type_ids must have shape [L]")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    types = mm_token_type_ids.to(torch.int64).cpu()
    grids = image_grid_thw.to(torch.int64).cpu().reshape(-1, 3)
    grid_index = 0
    current_pos = 0
    chunks: list[torch.Tensor] = []
    spans: list[ImageTokenSpan] = []

    for modality, grouped in itertools.groupby(enumerate(types.tolist()), lambda item: item[1]):
        group = list(grouped)
        start = group[0][0]
        end = group[-1][0] + 1
        length = end - start
        if modality == 0:
            chunks.append(torch.arange(length, dtype=torch.int64).view(1, -1).expand(3, -1) + current_pos)
            current_pos += length
            continue
        if modality != 1:
            raise ValueError(
                f"unsupported multimodal token type {modality}; this release accepts images only"
            )
        if grid_index >= grids.shape[0]:
            raise ValueError("more image token spans than image_grid_thw rows")
        grid_t, grid_h, grid_w = (int(v) for v in grids[grid_index].tolist())
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"invalid image grid {(grid_t, grid_h, grid_w)}")
        if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
            raise ValueError(
                f"image grid {(grid_t, grid_h, grid_w)} is not divisible by merge size "
                f"{spatial_merge_size}"
            )
        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        expected = grid_t * llm_h * llm_w
        if expected != length:
            raise ValueError(
                f"image token span length {length} does not match grid-derived length {expected}"
            )
        temporal = torch.arange(grid_t, dtype=torch.int64)
        height = torch.arange(llm_h, dtype=torch.int64) + current_pos
        width = torch.arange(llm_w, dtype=torch.int64) + current_pos
        t_grid, h_grid, w_grid = torch.meshgrid(temporal, height, width, indexing="ij")
        positions = torch.stack((t_grid, h_grid, w_grid), dim=0).reshape(3, -1)
        positions[0] += current_pos
        chunks.append(positions)
        spans.append(ImageTokenSpan(grid_index, start, end))
        current_pos += max(grid_h, grid_w) // spatial_merge_size
        grid_index += 1

    if grid_index != grids.shape[0]:
        raise ValueError("fewer image token spans than image_grid_thw rows")
    if not chunks:
        raise ValueError("multimodal processor returned an empty prompt")
    mrope_positions = torch.cat(chunks, dim=1).contiguous()
    if mrope_positions.shape[1] != types.numel():
        raise ValueError("mRoPE positions do not cover the expanded prompt")
    rope_delta = torch.tensor(
        [int(mrope_positions.max().item()) + 1 - int(types.numel())], dtype=torch.int64
    )
    return mrope_positions, rope_delta, spans


def _load_dsv4_encoder_if_needed(tokenizer: PreTrainedTokenizerBase) -> ModuleType | None:
    if getattr(tokenizer, "chat_template", None):
        return None
    model_path = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", "")
    if not model_path:
        return None
    encoder_path = os.path.join(str(model_path), "encoding", "encoding_dsv4.py")
    if not os.path.isfile(encoder_path):
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "encode_messages"):
        return None
    return module


def _apply_dsv4_chat_encoder(
    encoder: ModuleType,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict,
) -> str:
    rendered_messages = [dict(message) for message in messages]
    for message in rendered_messages:
        if message.get("tool_calls"):
            message["tool_calls"] = _dsv4_tool_calls(message["tool_calls"])
    if tools:
        _attach_tools_to_dsv4_messages(rendered_messages, tools)

    # No effort filtering here: the caller sanitized already, and the probe
    # needs raw values to reach the encoder's own validation.
    return encoder.encode_messages(
        rendered_messages,
        thinking_mode=resolve_thinking_mode(chat_template_kwargs, tools),
        reasoning_effort=chat_template_kwargs.get("reasoning_effort"),
    )


def _dsv4_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """The dsv4 encoder's contract is ``function.arguments`` = JSON-object STRING
    (it json.loads then iterates .items()); a dict (what ``render_messages``
    produces for Jinja templates) trips its bare-except fallback, which wraps the
    whole payload in a bogus parameter literally named ``arguments``. Re-serialize
    here. Copies each tool-call dict: the outer message copy is shallow, so these
    are shared with the caller."""
    rendered = []
    for tc in tool_calls:
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        fn["arguments"] = _dsv4_arguments_str(fn.get("arguments"))
        tc["function"] = fn
        rendered.append(tc)
    return rendered


def _dsv4_arguments_str(arguments: Any) -> str:
    """Missing/empty means no arguments (vLLM parity); anything else that is not
    a JSON object is rejected -- ValueError becomes a per-request "could not
    encode request" error, never a worker crash -- matching sglang's
    validate-then-400. A non-object would otherwise raise uncaught in the
    encoder's .items() or be wrapped as garbage."""
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    shown = f"{arguments!r:.200}"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"tool call function.arguments must be valid JSON, got {shown}"
            ) from err
        if isinstance(parsed, dict):
            return arguments
    raise ValueError(f"tool call function.arguments must be a JSON object, got {shown}")


def _attach_tools_to_dsv4_messages(messages: list[dict], tools: list[dict]) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["tools"] = tools
            return
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
