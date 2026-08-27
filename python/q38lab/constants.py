"""Immutable public inputs for the first RTX 5090 developer preview."""

from __future__ import annotations

from dataclasses import dataclass

PROFILE_NAME = "rtx5090-wsl2"

MODEL_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MODEL_DIRECTORY_NAME = "qwen38-flash-next-nvfp4-7b71922"
SERVED_MODEL_NAME = "qwen3.8-flash-next-nvfp4"
QWEN_LICENSE_URL = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE"

EXPECTED_FILE_COUNT = 419
EXPECTED_TOTAL_BYTES = 135_253_622_894
EXPECTED_SAFETENSORS_COUNT = 206
EXPECTED_MANIFEST_SHA256 = "6cc22b628ca575785e5dfdcab3c7056e79a7eac798969a145341ed1530c2a3a8"


@dataclass(frozen=True)
class ServeProfile:
    """Parser-backed FreeToken settings verified on the reference machine."""

    name: str = PROFILE_NAME
    served_model_name: str = SERVED_MODEL_NAME
    gpu: str = "0"
    tp_size: int = 1
    max_running_requests: int = 1
    max_seq_len: int = 8192
    max_prefill_length: int = 8192
    num_tokens: int = 8192
    memory_ratio: float = 0.89
    cache_type: str = "naive"
    attention_backend: str = "qsa_triton"
    graph: int = 0
    moe_backend: str = "offload"
    moe_cache_auto: bool = True
    nvfp4_backend: str = "auto"
    host: str = "127.0.0.1"
    port: int = 1919


RTX5090_WSL2_PROFILE = ServeProfile()

