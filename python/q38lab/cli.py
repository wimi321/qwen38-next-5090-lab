"""The ``q38lab`` reproducibility command-line interface."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Mapping

from .attestation import default_attestation_path
from .checkpoint import (
    CheckpointVerificationError,
    download_checkpoint,
    verify_checkpoint_receipt,
)
from .config import (
    ConfigurationError,
    default_model_dir,
    resolve_serve_config,
)
from .constants import (
    MODEL_REPO,
    MODEL_REVISION,
    PROFILE_NAME,
    QWEN_LICENSE_URL,
    RTX5090_WSL2_256K_IMAGE_PROFILE,
    SERVE_PROFILES,
)
from .doctor import evaluate_doctor, format_doctor_report
from .http import Q38HTTPError
from .runtime import Dependencies, cuda_toolkit_environment, temporary_environment
from .smoke import format_smoke_report, run_smoke


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _port(raw: str) -> int:
    value = _positive_int(raw)
    if value > 65535:
        raise argparse.ArgumentTypeError("must be no greater than 65535")
    return value


def _ratio(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _release_decode_tokens(raw: str) -> int:
    value = _positive_int(raw)
    if value > 1024:
        raise argparse.ArgumentTypeError("must be no greater than 1024")
    if value < 256:
        raise argparse.ArgumentTypeError("must be at least 256")
    return value


def _boolean(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("must be one of true/false, yes/no, on/off, or 1/0")


def _env_value(
    cli_value,
    env: Mapping[str, str],
    name: str,
    default,
    converter: Callable[[str], object] = str,
):
    if cli_value is not None:
        return cli_value
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise ConfigurationError(f"{name} has an invalid value: {raw!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="q38lab",
        description="Run Qwen3.8 Next text and image profiles on RTX 5090/WSL2.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Run read-only host and checkpoint checks")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    doctor.add_argument("--source-dir", type=Path, default=None)
    doctor.add_argument("--model-dir", type=Path, default=None)
    doctor.add_argument("--port", type=_port, default=None)
    doctor.add_argument("--profile", default=None, choices=sorted(SERVE_PROFILES))

    download = subparsers.add_parser(
        "download",
        help=f"Download the pinned {MODEL_REPO} checkpoint",
    )
    download.add_argument(
        "--accept-qwen-license",
        action="store_true",
        required=True,
        help=f"Confirm that you reviewed and accept {QWEN_LICENSE_URL}",
    )
    download.add_argument("--model-dir", type=Path, default=None)
    download.add_argument(
        "--full-verify",
        action="store_true",
        help="Read all 135GB and verify the audited SHA-256 manifest",
    )

    serve = subparsers.add_parser("serve", help="Launch the verified FreeToken profile")
    serve.add_argument("--profile", default=None, choices=sorted(SERVE_PROFILES))
    serve.add_argument("--model-dir", type=Path, default=None)
    serve.add_argument("--served-model-name", default=None)
    serve.add_argument("--gpu", default=None)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=_port, default=None)
    serve.add_argument("--memory-ratio", type=_ratio, default=None)
    serve.add_argument("--num-tokens", type=_positive_int, default=None)
    serve.add_argument("--max-seq-len", type=_positive_int, default=None)
    serve.add_argument("--max-prefill-length", type=_positive_int, default=None)
    serve.add_argument(
        "--unsafe-allow-non-loopback",
        action="store_true",
        dest="unsafe_non_loopback",
        help="Acknowledge that this unauthenticated server will be exposed beyond localhost",
    )

    smoke = subparsers.add_parser("smoke", help="Exercise lifecycle and OpenAI API behavior")
    smoke.add_argument("--base-url", default=None)
    smoke.add_argument("--model", default=None)
    smoke.add_argument("--timeout", type=_positive_float, default=None)
    smoke.add_argument(
        "--images",
        action="store_true",
        default=None,
        help="Run data-URL, four-image, streaming and media-security smoke checks",
    )
    smoke.add_argument(
        "--https-image-url",
        default=None,
        help="Also fetch one caller-selected public HTTPS image (implies --images)",
    )
    smoke.add_argument("--json", action="store_true")

    bench = subparsers.add_parser(
        "bench", help="Run the authoritative RTX 5090 release evidence harness",
    )
    bench.add_argument("--profile", default=None, choices=sorted(SERVE_PROFILES))
    bench.add_argument("--out", type=Path, required=True)
    bench.add_argument("--base-url", default=None)
    bench.add_argument("--model-dir", type=Path, default=None)
    bench.add_argument("--server-pid", type=_positive_int, default=None)
    bench.add_argument("--attestation", type=Path, default=None)
    bench.add_argument("--duration-seconds", type=_positive_int, default=None)
    bench.add_argument("--soak-interval-seconds", type=_positive_float, default=None)
    bench.add_argument("--sequential-requests", type=_positive_int, default=None)
    bench.add_argument("--warmups", type=_positive_int, default=None)
    bench.add_argument("--measurements", type=_positive_int, default=None)
    bench.add_argument(
        "--decode-tokens", type=_release_decode_tokens, default=None,
        metavar="256..1024",
    )
    bench.add_argument(
        "--image-file",
        type=Path,
        default=None,
        help="Local release-owned image fixture required by the 256K image profile",
    )
    bench.add_argument(
        "--https-image-url",
        default=None,
        help="Public HTTPS image fixture required by the 256K image profile",
    )
    bench.add_argument("--timeout", type=_positive_float, default=None)
    return parser


def _model_dir(cli_value: Path | None, deps: Dependencies) -> Path:
    return Path(_env_value(
        cli_value,
        deps.env,
        "Q38LAB_MODEL_DIR",
        default_model_dir(),
        Path,
    )).expanduser()


def _run_doctor(args: argparse.Namespace, deps: Dependencies) -> int:
    profile_name = str(_env_value(
        args.profile,
        deps.env,
        "Q38LAB_PROFILE",
        PROFILE_NAME,
        str,
    ))
    try:
        profile = SERVE_PROFILES[profile_name]
    except KeyError:
        raise ConfigurationError(f"unknown profile: {profile_name!r}") from None
    source_dir = Path(_env_value(
        args.source_dir,
        deps.env,
        "Q38LAB_SOURCE_DIR",
        Path.cwd(),
        Path,
    )).expanduser()
    model_dir = _model_dir(args.model_dir, deps)
    port = int(_env_value(
        args.port,
        deps.env,
        "Q38LAB_PORT",
        profile.port,
        int,
    ))
    if not 1 <= port <= 65535:
        raise ConfigurationError(f"Q38LAB_PORT is outside [1, 65535]: {port}")
    snapshot = deps.doctor_collector(
        source_dir=source_dir,
        model_dir=model_dir,
        port=port,
        profile_name=profile_name,
    )
    report = evaluate_doctor(snapshot)
    print(report.to_json() if args.json else format_doctor_report(report))
    return 0 if report.ready else 1


def _run_download(args: argparse.Namespace, deps: Dependencies) -> int:
    # argparse's required flag ensures acceptance is explicit before network or disk writes.
    assert args.accept_qwen_license
    verification = download_checkpoint(
        _model_dir(args.model_dir, deps),
        full_verify=args.full_verify,
        snapshot_download=deps.snapshot_download,
    )
    print(json.dumps(verification.as_dict(), indent=2, sort_keys=True))
    return 0


def _run_serve(args: argparse.Namespace, deps: Dependencies) -> int:
    profile_name = str(_env_value(
        args.profile,
        deps.env,
        "Q38LAB_PROFILE",
        PROFILE_NAME,
        str,
    ))
    config = resolve_serve_config(
        profile_name=profile_name,
        cli=vars(args),
        env=deps.env,
    )
    toolkit_environment = cuda_toolkit_environment(deps.env)
    preflight: dict[str, object] = {}
    if config.ple_require_native_io_uring:
        capability = deps.ple_capability_probe(config.model_dir)
        if not bool(getattr(capability, "production_ready", False)):
            detail = str(getattr(capability, "detail", "capability probe failed"))
            raise ConfigurationError(
                "the 256K profile requires native io_uring + O_DIRECT PLE "
                f"streaming: {detail}"
            )
    if config.qsa_require_native_topk:
        with temporary_environment(toolkit_environment):
            capability = deps.qsa_native_topk_probe()
        if not bool(getattr(capability, "production_ready", False)):
            detail = str(getattr(capability, "detail", "capability probe failed"))
            raise ConfigurationError(
                "the 256K profile requires a verified native SM120 QSA "
                f"fast-topk JIT and launch: {detail}"
            )
        with temporary_environment(toolkit_environment):
            ple_report = deps.ple_checkpoint_probe(config.model_dir)
        if (
            ple_report.get("status") != "pass"
            or ple_report.get("release_qualified") is not True
        ):
            raise ConfigurationError(
                "the 256K profile requires a release-qualified PLE checkpoint "
                "row/loader parity probe"
            )
        preflight["ple_checkpoint_probe"] = ple_report
    # A quick shape check prevents a moved tag or partial local download from ever
    # entering the 60+ second model loader. Full hashing remains opt-in.
    deps.verify_checkpoint_receipt(config.model_dir, require_full=False)
    argv = config.to_ft_argv()
    if any(item == "--moe-cpu-layers" or item.startswith("--moe-cpu-layers=") for item in argv):
        raise AssertionError("the reproducibility profile must leave CPU-layer selection on auto")
    attestation = (
        deps.attestation_writer(config, argv, preflight=preflight)
        if preflight else deps.attestation_writer(config, argv)
    )
    try:
        runtime_environment = config.runtime_environment()
        runtime_environment.update(toolkit_environment)
        with temporary_environment(runtime_environment):
            deps.launch_server(argv, "q38lab serve")
    finally:
        deps.attestation_remover(attestation)
    return 0


def _client_settings(
    args: argparse.Namespace,
    deps: Dependencies,
    *,
    default_timeout: float = 120.0,
) -> tuple[str, float]:
    base_url = str(_env_value(
        args.base_url,
        deps.env,
        "Q38LAB_BASE_URL",
        "http://127.0.0.1:1919",
        str,
    ))
    timeout = float(_env_value(
        args.timeout,
        deps.env,
        "Q38LAB_TIMEOUT",
        default_timeout,
        float,
    ))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigurationError("Q38LAB_TIMEOUT must be greater than 0")
    return base_url, timeout


def _run_smoke(args: argparse.Namespace, deps: Dependencies) -> int:
    base_url, timeout = _client_settings(args, deps)
    model = _env_value(
        args.model,
        deps.env,
        "Q38LAB_SERVED_MODEL_NAME",
        None,
        str,
    )
    client = deps.http_client_factory(base_url, timeout=timeout)
    include_images = bool(_env_value(
        args.images,
        deps.env,
        "Q38LAB_SMOKE_IMAGES",
        False,
        _boolean,
    ))
    https_image_url = _env_value(
        args.https_image_url,
        deps.env,
        "Q38LAB_SMOKE_HTTPS_IMAGE_URL",
        None,
        str,
    )
    report = run_smoke(
        client,
        requested_model=model,
        include_images=include_images or https_image_url is not None,
        https_image_url=https_image_url,
    )
    print(
        json.dumps(report.as_dict(), indent=2, sort_keys=True)
        if args.json
        else format_smoke_report(report)
    )
    return 0 if report.passed else 1


def _run_bench(args: argparse.Namespace, deps: Dependencies) -> int:
    model_dir = _model_dir(args.model_dir, deps)
    profile_name = str(_env_value(
        args.profile, deps.env, "Q38LAB_PROFILE", PROFILE_NAME, str,
    ))
    try:
        profile = SERVE_PROFILES[profile_name]
    except KeyError:
        raise ConfigurationError(f"unknown profile: {profile_name!r}") from None
    default_timeout = (
        1200.0
        if profile.name == RTX5090_WSL2_256K_IMAGE_PROFILE.name
        else 120.0
    )
    base_url, timeout = _client_settings(
        args, deps, default_timeout=default_timeout,
    )

    def positive_env(cli_value, name: str, default: int | float, converter=int):
        value = converter(_env_value(cli_value, deps.env, name, default, converter))
        if not math.isfinite(float(value)) or value <= 0:
            raise ConfigurationError(f"{name} must be finite and greater than 0")
        return value

    duration = int(positive_env(
        args.duration_seconds, "Q38LAB_BENCH_DURATION_SECONDS", 1800,
    ))
    soak_interval = float(positive_env(
        args.soak_interval_seconds, "Q38LAB_BENCH_SOAK_INTERVAL_SECONDS", 15.0, float,
    ))
    sequential = int(positive_env(
        args.sequential_requests, "Q38LAB_BENCH_SEQUENTIAL_REQUESTS", 100,
    ))
    warmups = int(positive_env(args.warmups, "Q38LAB_BENCH_WARMUPS", 3))
    measurements = int(positive_env(
        args.measurements, "Q38LAB_BENCH_MEASUREMENTS", 10,
    ))
    decode_tokens = int(positive_env(
        args.decode_tokens, "Q38LAB_BENCH_DECODE_TOKENS", 256,
    ))
    max_decode_tokens = (
        1024
        if profile.name == RTX5090_WSL2_256K_IMAGE_PROFILE.name
        else 512
    )
    if not 256 <= decode_tokens <= max_decode_tokens:
        raise ConfigurationError(
            "Q38LAB_BENCH_DECODE_TOKENS must be in "
            f"[256, {max_decode_tokens}] for {profile.name}"
        )
    if duration < 1800:
        raise ConfigurationError("Q38LAB_BENCH_DURATION_SECONDS must be at least 1800")
    if not 0 < soak_interval <= 25:
        raise ConfigurationError(
            "Q38LAB_BENCH_SOAK_INTERVAL_SECONDS must be in (0, 25]"
        )
    if sequential < 100:
        raise ConfigurationError("Q38LAB_BENCH_SEQUENTIAL_REQUESTS must be at least 100")
    if warmups < 3:
        raise ConfigurationError("Q38LAB_BENCH_WARMUPS must be at least 3")
    if measurements < 10:
        raise ConfigurationError("Q38LAB_BENCH_MEASUREMENTS must be at least 10")

    from urllib.parse import urlsplit

    parsed = urlsplit(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not (
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and port == profile.port
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
        and not parsed.username
        and not parsed.password
    ):
        raise ConfigurationError(
            "release bench requires exactly http://127.0.0.1:1919"
        )
    attestation_path = Path(_env_value(
        args.attestation,
        deps.env,
        "Q38LAB_ATTESTATION",
        default_attestation_path(port),
        Path,
    )).expanduser()
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"cannot read live q38lab serve attestation {attestation_path}: {exc}"
        ) from exc
    if not isinstance(attestation, dict):
        raise ConfigurationError("serve attestation must be a JSON object")
    attested_pid = attestation.get("pid")
    attested_commit = attestation.get("runtime_commit")
    attested_config = attestation.get("resolved_config")
    if not isinstance(attested_config, dict):
        raise ConfigurationError("serve attestation has no resolved profile")
    if (
        attested_config.get("profile_contract_verified") is not True
        or attested_config.get("profile") != profile.name
        or attested_config.get("host") != "127.0.0.1"
        or attested_config.get("port") != profile.port
    ):
        raise ConfigurationError(
            f"release bench requires q38lab serve's unmodified {profile.name} profile"
        )
    try:
        attested_model = Path(str(attestation.get("model_realpath"))).resolve()
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("serve attestation has an invalid model path") from exc
    if attested_model != model_dir.resolve():
        raise ConfigurationError("serve attestation model does not match --model-dir")
    server_pid_raw = _env_value(
        args.server_pid, deps.env, "Q38LAB_SERVER_PID", attested_pid, int,
    )
    if not isinstance(server_pid_raw, int) or server_pid_raw < 1:
        raise ConfigurationError("serve attestation has no valid server pid")
    if server_pid_raw != attested_pid:
        raise ConfigurationError("--server-pid does not match the live serve attestation")
    if (
        not isinstance(attested_commit, str)
        or len(attested_commit) != 40
        or any(char not in "0123456789abcdef" for char in attested_commit)
    ):
        raise ConfigurationError(
            "release bench requires a clean Git checkout attested by q38lab serve"
        )

    harness_argv = [
        "--profile", profile.name,
        "--model-dir", str(model_dir),
        "--server-pid", str(server_pid_raw),
        "--base-url", base_url,
        "--expected-commit", attested_commit,
        "--attestation", str(attestation_path),
        "--out", str(args.out),
        "--duration-seconds", str(duration),
        "--soak-interval-seconds", str(soak_interval),
        "--sequential-requests", str(sequential),
        "--warmups", str(warmups),
        "--measurements", str(measurements),
        "--decode-tokens", str(decode_tokens),
        "--request-timeout", str(timeout),
    ]
    image_file = _env_value(
        args.image_file, deps.env, "Q38LAB_BENCH_IMAGE_FILE", None, Path,
    )
    https_image_url = _env_value(
        args.https_image_url,
        deps.env,
        "Q38LAB_BENCH_HTTPS_IMAGE_URL",
        None,
        str,
    )
    if profile.name == RTX5090_WSL2_256K_IMAGE_PROFILE.name:
        if image_file is None:
            raise ConfigurationError(
                "the 256K image release bench requires --image-file"
            )
        image_path = Path(image_file).expanduser()
        if not image_path.is_file():
            raise ConfigurationError(f"--image-file is not a file: {image_path}")
        if not isinstance(https_image_url, str) or not https_image_url.startswith("https://"):
            raise ConfigurationError(
                "the 256K image release bench requires --https-image-url with HTTPS"
            )
        harness_argv.extend([
            "--image-file", str(image_path),
            "--https-image-url", https_image_url,
        ])
    return int(deps.release_harness(harness_argv))


_RUNNERS = {
    "doctor": _run_doctor,
    "download": _run_download,
    "serve": _run_serve,
    "smoke": _run_smoke,
    "bench": _run_bench,
}


def main(argv: Sequence[str] | None = None, *, deps: Dependencies | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    dependencies = deps or Dependencies()
    try:
        return _RUNNERS[args.command](args, dependencies)
    except (
        CheckpointVerificationError,
        ConfigurationError,
        FileExistsError,
        Q38HTTPError,
        ValueError,
    ) as exc:
        print(f"q38lab {args.command}: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
