"""Configuration resolution and launch-argument construction."""

from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .constants import (
    MODEL_DIRECTORY_NAME,
    PROFILE_256K_IMAGE_NAME,
    RTX5090_WSL2_PROFILE,
    SERVE_PROFILES,
    ServeProfile,
)


class ConfigurationError(ValueError):
    pass


def default_model_dir() -> Path:
    return Path.home() / "models" / MODEL_DIRECTORY_NAME


def _pick(
    cli_value: object | None,
    env: Mapping[str, str],
    env_name: str,
    profile_value: object,
    converter,
):
    if cli_value is not None:
        return cli_value
    raw = env.get(env_name)
    if raw is not None and raw.strip() != "":
        try:
            return converter(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{env_name} has an invalid value: {raw!r}") from exc
    return profile_value


def _positive_int(value: str | int) -> int:
    number = int(value)
    if number < 1:
        raise ValueError("must be at least 1")
    return number


def _port(value: str | int) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise ValueError("must be in [1, 65535]")
    return number


def _ratio(value: str | float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number < 1:
        raise ValueError("must be between 0 and 1")
    return number


def is_loopback_host(host: str) -> bool:
    """Return true only for an unambiguous local bind address."""

    candidate = host.strip().lower()
    if candidate == "localhost":
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # Do not resolve names: DNS/rebinding must not turn an apparently remote
        # unauthenticated bind into an allowed one.
        return False


@dataclass(frozen=True)
class ResolvedServeConfig:
    profile: str
    model_dir: Path
    served_model_name: str
    gpu: str
    host: str
    port: int
    memory_ratio: float
    num_tokens: int
    max_seq_len: int
    max_prefill_length: int
    unsafe_non_loopback: bool
    tp_size: int
    max_running_requests: int
    cache_type: str
    attention_backend: str
    graph: int
    moe_backend: str
    moe_cache_auto: bool
    moe_prefill_sparse: bool
    nvfp4_backend: str
    load_vision: bool
    ple_io_backend: str
    ple_require_native_io_uring: bool
    ple_cache_bytes: int
    ple_queue_depth: int
    ple_max_batch_pages: int
    ple_staging_buffers: int
    selector_workspace_bytes: int
    qsa_require_native_topk: bool
    qsa_cache_bytes: int
    vision_weights_bytes: int
    gpu_memory_envelope_bytes: int
    gpu_runtime_reserve_bytes: int
    profile_contract_verified: bool

    def to_ft_argv(self) -> list[str]:
        argv = [
            "--model", str(self.model_dir),
            "--served-model-name", self.served_model_name,
            "--gpu", self.gpu,
            "--tp-size", str(self.tp_size),
            "--max-running-requests", str(self.max_running_requests),
            "--max-seq-len-override", str(self.max_seq_len),
            "--max-prefill-length", str(self.max_prefill_length),
            "--num-tokens", str(self.num_tokens),
            "--memory-ratio", str(self.memory_ratio),
            "--cache-type", self.cache_type,
            "--attention-backend", self.attention_backend,
            "--graph", str(self.graph),
            "--moe-backend", self.moe_backend,
        ]
        if self.moe_cache_auto:
            argv.append("--moe-cache-auto")
        argv.extend([
            "--nvfp4-backend", self.nvfp4_backend,
            "--host", self.host,
            "--port", str(self.port),
        ])
        return argv

    def public_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["model_dir"] = str(self.model_dir)
        return data

    def runtime_environment(self) -> dict[str, str]:
        """Environment consumed by model construction and PLE bank setup."""

        return {
            "FREETOKEN_LOAD_VISION": "1" if self.load_vision else "0",
            "FREETOKEN_PLE_IO_BACKEND": self.ple_io_backend,
            "FREETOKEN_PLE_CACHE_BYTES": str(self.ple_cache_bytes),
            "FREETOKEN_PLE_QUEUE_DEPTH": str(self.ple_queue_depth),
            "FREETOKEN_PLE_MAX_BATCH_PAGES": str(self.ple_max_batch_pages),
            "FREETOKEN_PLE_STAGING_BUFFERS": str(self.ple_staging_buffers),
            "FREETOKEN_QSA_REQUIRE_NATIVE_TOPK": (
                "1" if self.qsa_require_native_topk else "0"
            ),
            "FREETOKEN_MOE_PREFILL_SPARSE": (
                "1" if self.moe_prefill_sparse else "0"
            ),
            "FREETOKEN_GPU_MEMORY_ENVELOPE_BYTES": str(
                self.gpu_memory_envelope_bytes
            ),
            # The 256K profile promises swap-free resident CPU expert banks.
            # Fail startup if an mlock request silently settles as PAGEABLE.
            "Q38LAB_REQUIRE_LOCKED_MOE": (
                "1" if self.profile == PROFILE_256K_IMAGE_NAME else "0"
            ),
        }


def resolve_serve_config(
    *,
    profile_name: str,
    cli: Mapping[str, object | None],
    env: Mapping[str, str] | None = None,
    profile: ServeProfile | None = None,
) -> ResolvedServeConfig:
    """Resolve CLI > ``Q38LAB_*`` > profile and enforce safe binding.

    Architecture-defining switches (TP, cache, QSA, graph and MoE policy) are
    intentionally profile-only for this first reproducibility release.
    """

    if profile is None:
        try:
            profile = SERVE_PROFILES[profile_name]
        except KeyError:
            raise ConfigurationError(f"unknown profile: {profile_name!r}") from None
    elif profile_name != profile.name:
        raise ConfigurationError(f"unknown profile: {profile_name!r}")
    environ = os.environ if env is None else env

    model_dir = Path(_pick(
        cli.get("model_dir"), environ, "Q38LAB_MODEL_DIR", default_model_dir(), Path
    )).expanduser()
    host = str(_pick(cli.get("host"), environ, "Q38LAB_HOST", profile.host, str))
    unsafe = bool(cli.get("unsafe_non_loopback"))
    if not is_loopback_host(host) and not unsafe:
        raise ConfigurationError(
            f"refusing unauthenticated non-loopback bind {host!r}; "
            "pass --unsafe-allow-non-loopback to acknowledge the exposure"
        )

    served_model_name = str(_pick(
        cli.get("served_model_name"), environ, "Q38LAB_SERVED_MODEL_NAME",
        profile.served_model_name, str,
    ))
    gpu = str(_pick(cli.get("gpu"), environ, "Q38LAB_GPU", profile.gpu, str))
    port = _port(_pick(cli.get("port"), environ, "Q38LAB_PORT", profile.port, _port))
    memory_ratio = _ratio(_pick(
        cli.get("memory_ratio"), environ, "Q38LAB_MEMORY_RATIO",
        profile.memory_ratio, _ratio,
    ))
    num_tokens = _positive_int(_pick(
        cli.get("num_tokens"), environ, "Q38LAB_NUM_TOKENS",
        profile.num_tokens, _positive_int,
    ))
    max_seq_len = _positive_int(_pick(
        cli.get("max_seq_len"), environ, "Q38LAB_MAX_SEQ_LEN",
        profile.max_seq_len, _positive_int,
    ))
    max_prefill_length = _positive_int(_pick(
        cli.get("max_prefill_length"), environ, "Q38LAB_MAX_PREFILL_LENGTH",
        profile.max_prefill_length, _positive_int,
    ))
    if not served_model_name.strip() or not gpu.strip():
        raise ConfigurationError("served model name and GPU selector must not be empty")
    if max_seq_len > profile.max_seq_len:
        raise ConfigurationError(
            f"the {profile.name} alpha is capped at {profile.max_seq_len} tokens"
        )
    if max_prefill_length > max_seq_len:
        raise ConfigurationError("max prefill length must not exceed max sequence length")
    if num_tokens < max_seq_len:
        raise ConfigurationError("KV token capacity must be at least the max sequence length")
    contract_verified = (
        served_model_name == profile.served_model_name
        and gpu == profile.gpu
        and host == profile.host
        and port == profile.port
        and memory_ratio == profile.memory_ratio
        and num_tokens == profile.num_tokens
        and max_seq_len == profile.max_seq_len
        and max_prefill_length == profile.max_prefill_length
        and not unsafe
    )

    return ResolvedServeConfig(
        profile=profile.name,
        model_dir=model_dir,
        served_model_name=served_model_name,
        gpu=gpu,
        host=host,
        port=port,
        memory_ratio=memory_ratio,
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        max_prefill_length=max_prefill_length,
        unsafe_non_loopback=unsafe,
        tp_size=profile.tp_size,
        max_running_requests=profile.max_running_requests,
        cache_type=profile.cache_type,
        attention_backend=profile.attention_backend,
        graph=profile.graph,
        moe_backend=profile.moe_backend,
        moe_cache_auto=profile.moe_cache_auto,
        moe_prefill_sparse=profile.moe_prefill_sparse,
        nvfp4_backend=profile.nvfp4_backend,
        load_vision=profile.load_vision,
        ple_io_backend=profile.ple_io_backend,
        ple_require_native_io_uring=profile.ple_require_native_io_uring,
        ple_cache_bytes=profile.ple_cache_bytes,
        ple_queue_depth=profile.ple_queue_depth,
        ple_max_batch_pages=profile.ple_max_batch_pages,
        ple_staging_buffers=profile.ple_staging_buffers,
        selector_workspace_bytes=profile.selector_workspace_bytes,
        qsa_require_native_topk=profile.qsa_require_native_topk,
        qsa_cache_bytes=profile.qsa_cache_bytes,
        vision_weights_bytes=profile.vision_weights_bytes,
        gpu_memory_envelope_bytes=profile.gpu_memory_envelope_bytes,
        gpu_runtime_reserve_bytes=profile.gpu_runtime_reserve_bytes,
        profile_contract_verified=contract_verified,
    )
