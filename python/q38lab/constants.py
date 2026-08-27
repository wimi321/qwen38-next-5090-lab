"""Immutable public inputs for the first RTX 5090 developer preview."""

from __future__ import annotations

from dataclasses import dataclass

PROFILE_NAME = "rtx5090-wsl2"
PROFILE_256K_IMAGE_NAME = "rtx5090-wsl2-256k-image"

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
    moe_prefill_sparse: bool = False
    nvfp4_backend: str = "auto"
    host: str = "127.0.0.1"
    port: int = 1919
    load_vision: bool = False
    ple_io_backend: str = "mmap"
    ple_require_native_io_uring: bool = False
    ple_cache_bytes: int = 0
    ple_queue_depth: int = 0
    ple_max_batch_pages: int = 0
    ple_staging_buffers: int = 1
    selector_workspace_bytes: int = 0
    qsa_require_native_topk: bool = False
    qsa_cache_bytes: int = 0
    vision_weights_bytes: int = 0
    gpu_memory_envelope_bytes: int = 0
    gpu_runtime_reserve_bytes: int = 0
    moe_bank_bytes: int = 0
    moe_total_layers: int = 0
    moe_locked_layers: int = 0
    moe_bounce_staging_bytes: int = 0


RTX5090_WSL2_PROFILE = ServeProfile()

RTX5090_WSL2_256K_IMAGE_PROFILE = ServeProfile(
    name=PROFILE_256K_IMAGE_NAME,
    max_seq_len=262_144,
    max_prefill_length=512,
    num_tokens=262_144,
    attention_backend="qsa_triton_sm120",
    moe_prefill_sparse=True,
    load_vision=True,
    ple_io_backend="io_uring_odirect",
    ple_require_native_io_uring=True,
    ple_cache_bytes=4 * 1024**3,
    ple_queue_depth=512,
    ple_max_batch_pages=4096,
    ple_staging_buffers=2,
    selector_workspace_bytes=128 * 1024**2,
    qsa_require_native_topk=True,
    # 12 QSA layers: BF16 K/V plus one 128-wide index key per four tokens.
    qsa_cache_bytes=6 * 1024**3 + 192 * 1024**2,
    vision_weights_bytes=897_862_112,
    gpu_memory_envelope_bytes=31 * 1024**3,
    gpu_runtime_reserve_bytes=512 * 1024**2,
    # Fixed-revision native NVFP4 routed-expert banks. Auto residency leaves 33
    # layers CUDA-registered and OS-locks 15 head/tail layers for CPU decode.
    moe_bank_bytes=68_136_468_480,
    moe_total_layers=48,
    moe_locked_layers=15,
    moe_bounce_staging_bytes=64 * 1024**2,
)

SERVE_PROFILES = {
    RTX5090_WSL2_PROFILE.name: RTX5090_WSL2_PROFILE,
    RTX5090_WSL2_256K_IMAGE_PROFILE.name: RTX5090_WSL2_256K_IMAGE_PROFILE,
}


def serve_profile(name: str) -> ServeProfile:
    try:
        return SERVE_PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile: {name!r}") from None
