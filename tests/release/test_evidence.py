from __future__ import annotations

import json
import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "release"))

import evidence  # noqa: E402


RUNTIME_COMMIT = "1" * 40


def ple_probe_report() -> dict[str, object]:
    digest = "a" * 64
    shards = []
    records = []
    for index in range(128):
        start = index * 10
        end = start + 10
        tensor_name = (
            "model.language_model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{index}.weight"
        )
        shards.append({
            "index": index,
            "tensor_name": tensor_name,
            "file_name": f"model-{index + 1:05d}-of-00206.safetensors",
            "shape": [10, 2],
            "dtype": "F8_E4M3",
            "global_start": start,
            "global_end": end,
        })
        for label, row in (("first", start), ("last", end - 1)):
            summary = {"sha256": digest, "shape": [2], "dtype": "float32", "numel": 2}
            records.append({
                "kind": "boundary",
                "label": f"shard-{index}-{label}",
                "tensor_name": tensor_name,
                "tensor_shape": [10, 2],
                "global_row": row,
                "shard_index": index,
                "shard_row": row - start,
                "storage_dtype": "F8_E4M3",
                "ground_truth": dict(summary),
                "auxiliary_bank": dict(summary),
                "match": True,
            })
    for index in range(8):
        shard_index = index
        tensor_name = shards[shard_index]["tensor_name"]
        summary = {"sha256": digest, "shape": [2], "dtype": "float32", "numel": 2}
        records.append({
            "kind": "hash",
            "label": f"hash-{index}",
            "tensor_name": tensor_name,
            "tensor_shape": [10, 2],
            "global_row": shard_index * 10 + 3,
            "shard_index": shard_index,
            "shard_row": 3,
            "storage_dtype": "F8_E4M3",
            "ground_truth": dict(summary),
            "auxiliary_bank": dict(summary),
            "match": True,
            "hash_token_position": index,
            "hash_head": 0 if index < 4 else 8,
        })
    unique_rows = {int(record["global_row"]) for record in records}
    return {
        "schema_version": "1.0",
        "status": "pass",
        "release_qualified": True,
        "checkpoint": {
            "model_dir_basename": "qwen38-flash-next-nvfp4-7b71922",
            "config_sha256": "b" * 64,
            "index_sha256": "c" * 64,
        },
        "runtime": {
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "compute_capability": "12.0",
            "wsl2": True,
            "backend": "io_uring_odirect",
            "device": "cuda",
            "gpu_fp8_decode_attested": True,
        },
        "loader_mapping": {
            "checkpoint_layer_id": 1,
            "normal_state_dict_action": "skip",
            "normal_state_dict_mapped_name": None,
            "auxiliary_bank": "ShardedSafetensorsMmapRowBank",
            "scale_tensor_name": "scale",
            "scale_shape": [1],
            "scale_dtype": "F32",
            "shards": shards,
        },
        "coverage": {
            "sample_count": len(records),
            "unique_row_count": len(unique_rows),
            "shard_count": 128,
            "all_shard_first_rows": True,
            "all_shard_last_rows": True,
            "global_first_row": True,
            "global_last_row": True,
            "hash_sample_count": 8,
            "hash_fixture_token_count": 8,
            "hash_fixture_tokens_sha256": "d" * 64,
            "bigram_and_trigram_heads": True,
        },
        "io": {
            "storage_bytes": 4096,
            "cache_hit_pages": 0,
            "cache_miss_pages": 1,
            "submission_batches": 1,
            "submitted_sqes": 1,
            "gpu_decoded_rows": len(unique_rows),
            "mapped_bytes": 0,
            "payload_bytes_read": len(unique_rows) * 2,
        },
        "records": records,
    }


def write_release_bundle(directory: Path) -> None:
    environment = evidence.read_json(directory / "environment.json")
    environment["synthetic"] = False
    environment["hardware"].update({
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_memory_mib": 32768,
        "gpu_index": 0,
        "compute_capability": "12.0",
        "wsl_memory_gib": 112,
        "wsl_processors": 32,
    })
    environment["software"].update({
        "os": "Linux under WSL2",
        "kernel": "6.6.87.2-microsoft-standard-WSL2",
        "os_id": "ubuntu",
        "os_version_id": "24.04",
        "nvidia_driver": "591.86",
        "cuda_toolkit": "13.0",
        "python": "3.12.0",
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "triton": "3.6.0",
        "cuda_runtime_probe": True,
        "media_doh_fallback_enabled": False,
        "media_system_dns_hard_cancel_supported": False,
    })
    evidence.write_json(directory / "environment.json", environment)

    requests: list[dict[str, object]] = []

    def add(
        case: str,
        iteration: int,
        *,
        stream: bool,
        started: float,
        warmup: bool = False,
        prompt_tokens: int = 10,
        completion_tokens: int = 1,
        content_tokens: int | None = None,
        success: bool = True,
        total_ms: float = 10.0,
        ttft_ms: float | None = None,
        proof: dict[str, object] | None = None,
    ) -> None:
        if proof is None:
            if case == "stream-parity":
                proof = {"text_present": True, "text_sha256": "a" * 64}
            elif case == "thinking-none":
                proof = {"reasoning_present": False, "visible_text_present": True}
            elif case == "thinking-high":
                proof = {"reasoning_present": True, "visible_text_present": True}
            elif case == "tool-call":
                proof = {
                    "arguments_sha256": "b" * 64,
                    "tool_city": "shanghai",
                    "tool_name": "get_weather",
                }
            elif case == "steady-decode":
                proof = {"finish_reason": "length", "requested_tokens": 256}
            else:
                proof = {}
        requests.append({
            "case": case,
            "iteration": iteration,
            "warmup": warmup,
            "stream": stream,
            "success": success,
            "http_status": 200,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "content_tokens": content_tokens,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "started_elapsed_s": round(started, 3),
            "finished_elapsed_s": round(started + total_ms / 1000, 3),
            "recorded_at": "2026-08-27T00:00:00Z",
            "proof": proof,
        })

    clock = 1.0
    for case in ("stream-parity", "thinking-none", "thinking-high", "tool-call"):
        add(case, 0, stream=False, started=clock)
        clock += 0.02
        add(case, 1, stream=True, started=clock, ttft_ms=1.0)
        clock += 0.02
    for target in (13, 128, 2048, 8176):
        for iteration in range(13):
            add(
                f"prompt-{target}", iteration, stream=True, started=clock,
                warmup=iteration < 3, prompt_tokens=target, completion_tokens=7,
                content_tokens=1 if target == 13 else target - 12, ttft_ms=1.0,
            )
            clock += 0.02
    add("steady-decode", 0, stream=True, started=clock, completion_tokens=256, ttft_ms=1.0)
    clock += 0.02
    for iteration in range(100):
        add("stability", iteration, stream=False, started=clock)
        clock += 0.02
    for iteration in range(61):
        add("soak", iteration, stream=False, started=100.0 + iteration * 30.0)

    (directory / "requests.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in requests),
        encoding="utf-8",
    )
    with (directory / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["case", "iteration", "prompt_tokens", "completion_tokens", "ttft_ms", "total_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in requests:
            if not item["warmup"]:
                writer.writerow({key: item[key] for key in fields})
    with (directory / "resource-samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "elapsed_s", "gpu_memory_mib", "wsl_rss_kib", "wsl_rss_source",
            "wsl_swap_kib", "minor_faults", "major_faults", "fault_processes",
            "pcie_rx_mib_s", "pcie_tx_mib_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for elapsed in range(1902):
            writer.writerow({
                "elapsed_s": elapsed,
                "gpu_memory_mib": 100,
                "wsl_rss_kib": 1000,
                "wsl_rss_source": "Windows vmmemWSL working set",
                "wsl_swap_kib": 0,
                "minor_faults": elapsed + 1,
                "major_faults": 0,
                "fault_processes": 1,
                "pcie_rx_mib_s": "",
                "pcie_tx_mib_s": "",
            })
    (directory / "pytest.txt").write_text(
        "1454 passed, 9 skipped, 11 deselected in 1.00s\n", encoding="utf-8"
    )
    summary = evidence.read_json(directory / "summary.json")
    summary.update({"status": "verified", "errors": []})
    summary["source"] = {
        "validated_runtime_commit": RUNTIME_COMMIT,
        "runtime_tree_sha256": "2" * 64,
        "upstream_base": "9ef3651309fe4058672f2cc92069238dea06be1b",
        "release_compatible": True,
    }
    summary["model"] = dict(evidence.EXPECTED_MODEL)
    summary["verification"] = {
        "checkpoint_full_sha256": True,
        "checkpoint_shape": True,
        "server_profile": True,
        "launch_attestation": True,
        "runtime_clean_tree": True,
        "server_port_owner": True,
    }
    summary["gates"] = {
        "pytest": {"passed": 1454, "failed": 0, "skipped": 9, "deselected": 11},
        "prompt_cases": [
            {
                "content_prompt_tokens": 1 if target == 13 else target - 12,
                "rendered_prompt_tokens": target,
                "completion_tokens": 7,
                "ttft_p50_ms": 1.0,
                "ttft_p95_ms": 1.0,
                "total_p50_ms": 10.0,
                "total_p95_ms": 10.0,
            }
            for target in (13, 128, 2048, 8176)
        ],
        "steady_decode": {
            "requested_tokens": 256,
            "observed_tokens": 256,
            "finish_reason": "length",
            "http_status": 200,
            "decode_tokens_per_second": 28333.333,
        },
        "api": {"stream_nonstream_match": True, "thinking_none": True, "thinking_high": True, "tool_call": True},
        "stability": {"attempted": 100, "succeeded": 100},
        "soak": {
            "attempted": 61,
            "succeeded": 61,
            "started_elapsed_s": 100.0,
            "finished_elapsed_s": 1900.01,
            "duration_seconds": 1800.01,
            "max_start_gap_seconds": 30.0,
        },
        "continuous_run_seconds": 1800.01,
        "memory_leak_detected": False,
    }
    summary["resources"] = {
        "acceptance_window_start_elapsed_s": 1.0,
        "acceptance_window_end_elapsed_s": 1900.01,
        "peak_vram_mib": 100,
        "peak_wsl_rss_kib": 1000,
        "wsl_rss_source": "Windows vmmemWSL working set",
        "wsl_swap_kib": 0,
        "page_faults": {"minor_delta": 1899.0, "major_delta": 0},
        "windows_pagefile_note": "Windows pagefile may be active",
    }
    evidence.write_json(directory / "summary.json", summary)
    evidence.write_checksums(directory)


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "bundle"
        shutil.copytree(ROOT / "results" / "example-synthetic", self.directory)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_tracked_synthetic_example_is_valid_but_not_releasable(self) -> None:
        summary = evidence.validate_directory(self.directory)
        self.assertEqual(summary["status"], "synthetic")
        with self.assertRaisesRegex(evidence.EvidenceError, "status=verified"):
            evidence.validate_directory(self.directory, release=True)

    def test_checksum_is_exhaustive_and_detects_mutation(self) -> None:
        (self.directory / "pytest.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(evidence.EvidenceError, "checksum mismatch"):
            evidence.validate_directory(self.directory)
        evidence.write_checksums(self.directory)
        evidence.validate_directory(self.directory)
        (self.directory / "unlisted.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(evidence.EvidenceError, "not exhaustive"):
            evidence.validate_directory(self.directory)

    def test_private_paths_and_request_text_are_rejected(self) -> None:
        environment = evidence.read_json(self.directory / "environment.json")
        environment["software"]["os"] = "/home/alice/private"
        evidence.write_json(self.directory / "environment.json", environment)
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "private-looking"):
            evidence.validate_directory(self.directory)

        shutil.rmtree(self.directory)
        shutil.copytree(ROOT / "results" / "example-synthetic", self.directory)
        (self.directory / "requests.jsonl").write_text(
            json.dumps({"case": "bad", "prompt": "do not persist me"}) + "\n",
            encoding="utf-8",
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "must not be tracked"):
            evidence.validate_directory(self.directory)

    def test_release_thresholds_accept_only_the_exact_preview_contract(self) -> None:
        write_release_bundle(self.directory)
        evidence.validate_directory(self.directory, release=True)
        summary = evidence.read_json(self.directory / "summary.json")
        summary["resources"]["peak_vram_mib"] = 31 * 1024
        evidence.write_json(self.directory / "summary.json", summary)
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "below 31 GiB"):
            evidence.validate_directory(self.directory, release=True)

    def test_v02_summary_contract_requires_256k_images_and_runtime_telemetry(self) -> None:
        write_release_bundle(self.directory)
        summary = evidence.read_json(self.directory / "summary.json")
        summary["execution"] = dict(evidence.EXPECTED_EXECUTION_V02)
        summary["gates"]["prompt_cases"] = [
            {
                "content_prompt_tokens": target - 12,
                "rendered_prompt_tokens": target,
                "completion_tokens": 8,
                "ttft_p50_ms": 10.0,
                "ttft_p95_ms": 10.0,
                "total_p50_ms": 20.0,
                "total_p95_ms": 20.0,
            }
            for target in evidence.V02_PROMPT_TARGETS
        ]
        summary["gates"]["steady_decode"]["decode_tokens_per_second"] = 5.0
        summary["gates"]["api"] = {key: True for key in evidence.V02_REQUIRED_API_GATES}
        summary["gates"]["boundary"] = {
            "input_tokens": 261120, "output_tokens": 1024, "total_tokens": 262144,
            "text_completed": True, "image_completed": True,
            "image_tokens": 64, "text_tokens": 261056,
        }
        summary["gates"]["long_context_ttft_ms"] = 900000
        summary["gates"]["niah"] = {
            "attempted": 4, "succeeded": 4,
            "depths": list(evidence.V02_NIAH_DEPTHS),
        }
        summary["gates"]["vision_quality"] = {
            "ocr": True, "object": True, "chart": True,
        }
        summary["telemetry"] = {
            "selector": {
                "workspace_peak_bytes": 128 * 1024**2,
                "native_calls": 1,
                "fallback_calls": 0,
                "errors": 0,
            },
            "ple": {"cold_runs": 1, "warm_runs": 3, "bytes_read": 1,
                    "cache_hits": 1, "cache_misses": 1, "wait_ms": 1, "page_faults": 0},
            "vision": {"image_tokens": 64, "latency_ms": 1},
            "prefill_chunks": {"count": 512, "total_ms": 1},
            "moe_prefill": {
                "active_rows": 1024,
                "possible_rows": 4096,
                "bytes_copied": 4000,
                "full_bytes": 10000,
                "row_fraction": 0.25,
                "byte_fraction": 0.4,
            },
        }
        evidence._validate_summary(summary, release=True)

        summary["telemetry"]["ple"]["warm_runs"] = 2
        with self.assertRaisesRegex(evidence.EvidenceError, "one cold and three warm"):
            evidence._validate_summary(summary, release=True)
        summary["telemetry"]["ple"]["warm_runs"] = 3
        summary["telemetry"]["selector"]["fallback_calls"] = 1
        with self.assertRaisesRegex(evidence.EvidenceError, "zero fast-topk fallbacks"):
            evidence._validate_summary(summary, release=True)
        summary["telemetry"]["selector"]["fallback_calls"] = 0
        summary["telemetry"]["moe_prefill"]["active_rows"] = 4097
        with self.assertRaisesRegex(evidence.EvidenceError, "cannot exceed possible_rows"):
            evidence._validate_summary(summary, release=True)
        summary["telemetry"]["moe_prefill"]["active_rows"] = 1024
        summary["telemetry"]["moe_prefill"]["row_fraction"] = 0.5
        with self.assertRaisesRegex(evidence.EvidenceError, "does not match"):
            evidence._validate_summary(summary, release=True)

    def test_moe_prefill_telemetry_contract_rejects_invalid_counts_and_ratios(self) -> None:
        telemetry = {
            "active_rows": 128,
            "possible_rows": 512,
            "bytes_copied": 400,
            "full_bytes": 1000,
            "row_fraction": 0.25,
            "byte_fraction": 0.4,
        }
        self.assertIs(
            evidence.validate_moe_prefill_telemetry(telemetry), telemetry
        )

        mutations = (
            ("active_rows", 128.0, "non-negative integer"),
            ("possible_rows", True, "non-negative integer"),
            ("bytes_copied", -1, "non-negative integer"),
            ("active_rows", 513, "cannot exceed possible_rows"),
            ("bytes_copied", 1001, "cannot exceed full_bytes"),
            ("row_fraction", 1.01, r"\[0, 1\]"),
            ("byte_fraction", float("nan"), r"\[0, 1\]"),
            ("row_fraction", 0.3, "does not match"),
            ("byte_fraction", 0.5, "does not match"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key, value=value):
                candidate = dict(telemetry)
                candidate[key] = value
                with self.assertRaisesRegex(evidence.EvidenceError, message):
                    evidence.validate_moe_prefill_telemetry(candidate)

        zero = {
            "active_rows": 0,
            "possible_rows": 0,
            "bytes_copied": 0,
            "full_bytes": 0,
            "row_fraction": 0.0,
            "byte_fraction": 0.0,
        }
        evidence.validate_moe_prefill_telemetry(zero)
        zero["row_fraction"] = 0.000001
        with self.assertRaisesRegex(evidence.EvidenceError, "does not match"):
            evidence.validate_moe_prefill_telemetry(zero)

    def test_v02_summary_rejects_slow_decode_and_long_ttft(self) -> None:
        write_release_bundle(self.directory)
        summary = evidence.read_json(self.directory / "summary.json")
        summary["execution"] = dict(evidence.EXPECTED_EXECUTION_V02)
        summary["gates"]["prompt_cases"] = [
            {"rendered_prompt_tokens": target, "content_prompt_tokens": target - 12}
            for target in evidence.V02_PROMPT_TARGETS
        ]
        summary["gates"]["api"] = {key: True for key in evidence.V02_REQUIRED_API_GATES}
        summary["gates"]["boundary"] = {
            "input_tokens": 261120, "output_tokens": 1024, "total_tokens": 262144,
            "text_completed": True, "image_completed": True,
            "image_tokens": 1, "text_tokens": 261119,
        }
        summary["gates"]["long_context_ttft_ms"] = 1
        summary["gates"]["niah"] = {
            "attempted": 4, "succeeded": 4,
            "depths": list(evidence.V02_NIAH_DEPTHS),
        }
        summary["gates"]["vision_quality"] = {
            "ocr": True, "object": True, "chart": True,
        }
        summary["gates"]["steady_decode"]["decode_tokens_per_second"] = 4.999
        summary["telemetry"] = {
            "selector": {
                "workspace_peak_bytes": 128 * 1024**2,
                "native_calls": 1,
                "fallback_calls": 0,
                "errors": 0,
            },
            "ple": {"cold_runs": 1, "warm_runs": 3, "bytes_read": 1,
                    "cache_hits": 1, "cache_misses": 1, "wait_ms": 1, "page_faults": 0},
            "vision": {"image_tokens": 1, "latency_ms": 1},
            "prefill_chunks": {"count": 512, "total_ms": 1},
            "moe_prefill": {
                "active_rows": 1024,
                "possible_rows": 4096,
                "bytes_copied": 4000,
                "full_bytes": 10000,
                "row_fraction": 0.25,
                "byte_fraction": 0.4,
            },
        }
        with self.assertRaisesRegex(evidence.EvidenceError, "at least 5 tok/s"):
            evidence._validate_summary(summary, release=True)

    def test_v02_release_crosschecks_raw_image_boundary_and_telemetry(self) -> None:
        write_release_bundle(self.directory)
        config = evidence.read_json(self.directory / "resolved-config.json")
        config["profile"] = evidence.PROFILE_V02
        config["settings"] = dict(evidence.EXPECTED_SETTINGS_V02)
        evidence.write_json(self.directory / "resolved-config.json", config)

        rows = [
            json.loads(line)
            for line in (self.directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        rows = [row for row in rows if not row["case"].startswith("prompt-")]
        for index, row in enumerate(item for item in rows if item["case"] == "stability"):
            row["proof"] = {"image": bool(index % 2)}
        for index, row in enumerate(item for item in rows if item["case"] == "soak"):
            row["proof"] = {"image": bool(index % 2)}

        template = dict(rows[0])
        def add(case: str, iteration: int = 0, *, stream: bool = False,
                status: int = 200, prompt: int = 10, completion: int = 1,
                proof: dict | None = None, warmup: bool = False,
                started: float = 1.0) -> None:
            item = dict(template)
            item.update({
                "case": case, "iteration": iteration, "warmup": warmup, "stream": stream,
                "success": True, "http_status": status, "prompt_tokens": prompt,
                "completion_tokens": completion, "content_tokens": prompt - 12,
                "ttft_ms": 1.0 if stream else None, "total_ms": 10.0,
                "started_elapsed_s": started, "finished_elapsed_s": started + 0.01,
                "proof": dict(proof or {}),
            })
            rows.append(item)

        digest = "d" * 64
        add("image-data-url", proof={"image_count": 1})
        add("image-https", proof={"https": True})
        add("image-four", proof={"image_count": 4})
        add("image-stream-parity", 0, proof={"text_sha256": digest})
        add("image-stream-parity", 1, stream=True, proof={"text_sha256": digest})
        add("image-security-rejections", status=400, proof={"attempted": 4, "passed": 4})
        add("context-length-rejection", status=400, prompt=261120,
            proof={"error_code": "context_length_exceeded"})
        for kind in ("ocr", "object", "chart"):
            add(f"image-{kind}-quality", proof={
                "answer_match": True,
                "fixture_sha256": "c" * 64,
                "answer_sha256": "d" * 64,
            })
        add("image-thinking", proof={
            "reasoning_present": True,
            "visible_text_present": True,
            "answer_match": True,
            "fixture_sha256": "c" * 64,
        })
        add("image-tool-call", proof={
            "tool_name": "report_access_code",
            "answer_match": True,
            "arguments_sha256": "e" * 64,
            "fixture_sha256": "c" * 64,
        })
        add("ple-cold", proof={"phase": "cold"}, started=0.5)
        for iteration in (1, 2, 3):
            add(
                "ple-warm", iteration, proof={"phase": "warm"},
                warmup=True, started=0.5 + iteration * 0.02,
            )
        for target, depth in zip(evidence.V02_PROMPT_TARGETS, evidence.V02_NIAH_DEPTHS):
            add(f"prompt-{target}", stream=True, prompt=target, proof={
                "needle_depth": depth,
                "needle_found": True,
                "expected_code_sha256": "f" * 64,
                "answer_sha256": "a" * 64,
            })
        add("boundary-text", stream=True, prompt=261120, completion=1024,
            proof={"finish_reason": "length"})
        add("boundary-image", stream=True, prompt=261120, completion=1024,
            proof={
                "finish_reason": "length", "image_count": 1,
                "image_tokens": 64, "text_tokens": 261056,
                "runtime_image_tokens_delta": 64,
            })
        (self.directory / "requests.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        with (self.directory / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["case", "iteration", "prompt_tokens", "completion_tokens", "ttft_ms", "total_ms"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows if not row["warmup"])

        samples = []
        for index, phase in enumerate(
            ("baseline", "cold", "warm-1", "warm-2", "warm-3", "final")
        ):
            samples.append({
                "phase": phase,
                "selector": {
                    "workspace_peak_bytes": 128 * 1024**2,
                    "native_calls": index,
                    "fallback_calls": 0,
                    "errors": 0,
                },
                "ple": {
                    "bytes_read": 0 if index == 0 else (100 if index < 5 else 500),
                    "cache_hits": max(0, min(index - 1, 3)),
                    "cache_misses": 0 if index == 0 else (1 if index < 5 else 5),
                    "wait_ms": index, "page_faults": 0,
                },
                "vision": {"image_tokens": 64 if index == 5 else max(1, index),
                           "latency_ms": max(1, index)},
                "prefill_chunks": {"count": 512 if phase == "final" else index, "total_ms": max(1, index)},
                "moe_prefill": {
                    "active_rows": index * 100,
                    "possible_rows": index * 512,
                    "bytes_copied": index * 400,
                    "full_bytes": index * 1000,
                    "row_fraction": 0.0 if index == 0 else 100 / 512,
                    "byte_fraction": 0.0 if index == 0 else 0.4,
                },
            })
        evidence.write_json(self.directory / evidence.V02_RUNTIME_TELEMETRY_FILE,
                            {"schema_version": "1.0", "samples": samples})

        summary = evidence.read_json(self.directory / "summary.json")
        summary["execution"] = dict(evidence.EXPECTED_EXECUTION_V02)
        summary["gates"]["api"] = {key: True for key in evidence.V02_REQUIRED_API_GATES}
        summary["gates"]["prompt_cases"] = [{
            "content_prompt_tokens": target - 12, "rendered_prompt_tokens": target,
            "completion_tokens": 1, "ttft_p50_ms": 1.0, "ttft_p95_ms": 1.0,
            "total_p50_ms": 10.0, "total_p95_ms": 10.0,
        } for target in evidence.V02_PROMPT_TARGETS]
        summary["gates"]["boundary"] = {
            "input_tokens": 261120, "output_tokens": 1024, "total_tokens": 262144,
            "text_completed": True, "image_completed": True,
            "image_tokens": 64, "text_tokens": 261056,
        }
        summary["gates"]["long_context_ttft_ms"] = 1.0
        summary["gates"]["niah"] = {
            "attempted": 4, "succeeded": 4,
            "depths": list(evidence.V02_NIAH_DEPTHS),
        }
        summary["gates"]["vision_quality"] = {
            "ocr": True, "object": True, "chart": True,
        }
        summary["telemetry"] = {
            "selector": {
                "workspace_peak_bytes": 128 * 1024**2,
                "native_calls": 5,
                "fallback_calls": 0,
                "errors": 0,
            },
            "ple": {"cold_runs": 1, "warm_runs": 3, "bytes_read": 500.0,
                    "cache_hits": 3.0, "cache_misses": 5.0, "wait_ms": 5.0, "page_faults": 0.0},
            "vision": {"image_tokens": 64, "latency_ms": 5.0},
            "prefill_chunks": {"count": 512, "total_ms": 5.0},
            "moe_prefill": {
                "active_rows": 500,
                "possible_rows": 2560,
                "bytes_copied": 2000,
                "full_bytes": 5000,
                "row_fraction": round(100 / 512, 6),
                "byte_fraction": 0.4,
            },
        }
        summary["verification"]["ple_checkpoint_rows"] = True
        summary["resources"]["acceptance_window_start_elapsed_s"] = 0.5
        evidence.write_json(
            self.directory / evidence.V02_PLE_CHECKPOINT_PROBE_FILE,
            ple_probe_report(),
        )
        evidence.write_json(self.directory / "summary.json", summary)
        evidence.write_checksums(self.directory)
        evidence.validate_directory(self.directory, release=True)

        boundary_image = next(
            row for row in rows if row["case"] == "boundary-image"
        )
        boundary_image["proof"]["runtime_image_tokens_delta"] = 63
        (self.directory / "requests.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "token accounting"):
            evidence.validate_directory(self.directory, release=True)
        boundary_image["proof"]["runtime_image_tokens_delta"] = 64

        samples[2]["ple"]["bytes_read"] = 101
        evidence.write_json(
            self.directory / evidence.V02_RUNTIME_TELEMETRY_FILE,
            {"schema_version": "1.0", "samples": samples},
        )
        (self.directory / "requests.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "each raw PLE warm"):
            evidence.validate_directory(self.directory, release=True)
        samples[2]["ple"]["bytes_read"] = 100
        evidence.write_json(
            self.directory / evidence.V02_RUNTIME_TELEMETRY_FILE,
            {"schema_version": "1.0", "samples": samples},
        )

        samples[2]["moe_prefill"]["full_bytes"] = 1000.0
        evidence.write_json(
            self.directory / evidence.V02_RUNTIME_TELEMETRY_FILE,
            {"schema_version": "1.0", "samples": samples},
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "non-negative integer"):
            evidence.validate_directory(self.directory, release=True)
        samples[2]["moe_prefill"]["full_bytes"] = 2000
        evidence.write_json(
            self.directory / evidence.V02_RUNTIME_TELEMETRY_FILE,
            {"schema_version": "1.0", "samples": samples},
        )

        for row in rows:
            if row["case"] == "soak":
                row["proof"] = {"image": False}
        (self.directory / "requests.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "soak must continuously mix"):
            evidence.validate_directory(self.directory, release=True)

    def test_release_commit_is_exact_and_expected_commit_is_bound(self) -> None:
        write_release_bundle(self.directory)
        evidence.validate_directory(
            self.directory, release=True, expected_commit=RUNTIME_COMMIT
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "!= expected"):
            evidence.validate_directory(
                self.directory, release=True, expected_commit="3" * 40
            )
        summary = evidence.read_json(self.directory / "summary.json")
        summary["source"]["validated_runtime_commit"] = "short"
        evidence.write_json(self.directory / "summary.json", summary)
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "exact 40-hex"):
            evidence.validate_directory(self.directory, release=True)

    def test_release_environment_requires_wsl2_cuda13_and_verified_stack(self) -> None:
        mutations = (
            ("kernel", "6.8.0-generic", "WSL2"),
            ("cuda_toolkit", "not found", "CUDA toolkit"),
            ("torch", None, "Torch"),
            ("triton", None, "Triton"),
            ("cuda_runtime_probe", False, "runtime probe"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                write_release_bundle(self.directory)
                environment = evidence.read_json(self.directory / "environment.json")
                environment["software"][field] = value
                evidence.write_json(self.directory / "environment.json", environment)
                evidence.write_checksums(self.directory)
                with self.assertRaisesRegex(evidence.EvidenceError, message):
                    evidence.validate_directory(self.directory, release=True)

    def test_release_recomputes_pytest_requests_and_summary(self) -> None:
        write_release_bundle(self.directory)
        summary = evidence.read_json(self.directory / "summary.json")
        summary["gates"]["stability"]["succeeded"] = 99
        evidence.write_json(self.directory / "summary.json", summary)
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "stability gate"):
            evidence.validate_directory(self.directory, release=True)

        write_release_bundle(self.directory)
        (self.directory / "pytest.txt").write_text("1453 passed\n", encoding="utf-8")
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "pytest counts"):
            evidence.validate_directory(self.directory, release=True)

        write_release_bundle(self.directory)
        request_rows = [json.loads(line) for line in (self.directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        next(row for row in request_rows if row["case"] == "stream-parity")["success"] = False
        (self.directory / "requests.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in request_rows), encoding="utf-8"
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "API gates"):
            evidence.validate_directory(self.directory, release=True)

    def test_release_steady_decode_requires_exact_length_termination(self) -> None:
        mutations = (
            (
                lambda rows, summary: next(
                    row for row in rows if row["case"] == "steady-decode"
                )["proof"].update({"finish_reason": "stop"}),
                "finish_reason=length",
            ),
            (
                lambda rows, summary: summary["gates"]["steady_decode"].update(
                    {"requested_tokens": 512}
                ),
                "exact requested token budget",
            ),
            (
                lambda rows, summary: next(
                    row for row in rows if row["case"] == "steady-decode"
                )["proof"].update({"requested_tokens": 512}),
                "raw requested_tokens do not match",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                write_release_bundle(self.directory)
                rows = [
                    json.loads(line)
                    for line in (self.directory / "requests.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                summary = evidence.read_json(self.directory / "summary.json")
                mutate(rows, summary)
                (self.directory / "requests.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )
                evidence.write_json(self.directory / "summary.json", summary)
                evidence.write_checksums(self.directory)
                with self.assertRaisesRegex(evidence.EvidenceError, message):
                    evidence.validate_directory(self.directory, release=True)

    def test_one_content_token_and_rendered_length_are_both_proven(self) -> None:
        write_release_bundle(self.directory)
        request_rows = [json.loads(line) for line in (self.directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in request_rows:
            if row["case"] == "prompt-13":
                row["content_tokens"] = 2
        (self.directory / "requests.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in request_rows), encoding="utf-8"
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "one content token"):
            evidence.validate_directory(self.directory, release=True)

    def test_release_api_gates_require_semantic_proofs_not_success_flags(self) -> None:
        mutations = (
            (
                "stream-parity",
                lambda row: row["proof"].update({"text_sha256": "c" * 64}),
            ),
            (
                "thinking-none",
                lambda row: row["proof"].update({"reasoning_present": True}),
            ),
            (
                "tool-call",
                lambda row: row["proof"].update({"tool_city": "beijing"}),
            ),
        )
        for case, mutate in mutations:
            with self.subTest(case=case):
                write_release_bundle(self.directory)
                rows = [
                    json.loads(line)
                    for line in (self.directory / "requests.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                mutate(next(row for row in rows if row["case"] == case and row["stream"]))
                (self.directory / "requests.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )
                evidence.write_checksums(self.directory)
                with self.assertRaisesRegex(evidence.EvidenceError, "API gates"):
                    evidence.validate_directory(self.directory, release=True)

    def test_release_soak_and_telemetry_fail_closed(self) -> None:
        write_release_bundle(self.directory)
        rows = (self.directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        kept = [line for line in rows if json.loads(line)["case"] != "soak" or json.loads(line)["iteration"] < 10]
        (self.directory / "requests.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "latency.csv row count"):
            evidence.validate_directory(self.directory, release=True)

        write_release_bundle(self.directory)
        (self.directory / "resource-samples.csv").write_text(
            "elapsed_s,gpu_memory_mib,wsl_rss_kib,wsl_rss_source,wsl_swap_kib,"
            "minor_faults,major_faults,fault_processes,pcie_rx_mib_s,pcie_tx_mib_s\n",
            encoding="utf-8",
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "telemetry is empty"):
            evidence.validate_directory(self.directory, release=True)

        write_release_bundle(self.directory)
        resource_lines = (self.directory / "resource-samples.csv").read_text(encoding="utf-8").splitlines()
        sparse = [resource_lines[0], *resource_lines[1::10], resource_lines[-1]]
        (self.directory / "resource-samples.csv").write_text("\n".join(sparse) + "\n", encoding="utf-8")
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "sampling gap"):
            evidence.validate_directory(self.directory, release=True)

        write_release_bundle(self.directory)
        with (self.directory / "resource-samples.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for row in rows:
            row.update({
                "gpu_memory_mib": "0",
                "wsl_rss_kib": "0",
                "minor_faults": "0",
                "fault_processes": "0",
            })
        with (self.directory / "resource-samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "must be positive|live process"):
            evidence.validate_directory(self.directory, release=True)

    def test_release_scans_every_text_artifact_for_privacy(self) -> None:
        write_release_bundle(self.directory)
        (self.directory / "pytest.txt").write_text(
            "1454 passed from /home/alice/private\n", encoding="utf-8"
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "private-looking text"):
            evidence.validate_directory(self.directory, release=True)

        write_release_bundle(self.directory)
        (self.directory / "pytest.txt").write_text(
            "1454 passed via 192.168.1.25\n", encoding="utf-8"
        )
        evidence.write_checksums(self.directory)
        with self.assertRaisesRegex(evidence.EvidenceError, "non-loopback IP"):
            evidence.validate_directory(self.directory, release=True)

    def test_readme_markers_are_exact_and_checkable(self) -> None:
        original = f"before\n{evidence.BEGIN_MARKER}\nstale\n{evidence.END_MARKER}\nafter\n"
        updated = evidence.update_marked_text(original, "generated")
        self.assertIn(f"{evidence.BEGIN_MARKER}\ngenerated\n{evidence.END_MARKER}", updated)
        with self.assertRaisesRegex(evidence.EvidenceError, "exactly one"):
            evidence.update_marked_text("no markers", "generated")

    def test_release_input_is_bound_to_requested_profile(self) -> None:
        root = Path(self.temporary.name) / "selection"
        old = root / "rtx5090-2026-01-01"
        new = root / "rtx5090-2026-08-27"
        old.mkdir(parents=True)
        new.mkdir()

        def validate(directory, **_kwargs):
            profile = evidence.PROFILE_V01 if directory == old else evidence.PROFILE_V02
            return {"execution": {"profile": profile}}

        with mock.patch.object(evidence, "validate_directory", side_effect=validate):
            assert evidence.find_release_input(
                root, expected_profile=evidence.PROFILE_V01
            ) == old
            assert evidence.find_release_input(
                root, expected_profile=evidence.PROFILE_V02
            ) == new
            with self.assertRaisesRegex(evidence.EvidenceError, "unsupported requested"):
                evidence.find_release_input(root, expected_profile="future-profile")

    def test_runtime_to_tag_binding_allows_only_evidence_and_docs(self) -> None:
        repo = Path(self.temporary.name) / "git-binding"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "python").mkdir()
        (repo / "python" / "runtime.py").write_text("value = 1\n", encoding="utf-8")
        (repo / "profiles").mkdir()
        (repo / "profiles" / "profile.json").write_text("{}\n", encoding="utf-8")
        (repo / "scripts" / "release").mkdir(parents=True)
        (repo / "scripts" / "release" / "gate.py").write_text("pass\n", encoding="utf-8")
        (repo / "pyproject.toml").write_text("[project]\nname='x'\nversion='1'\n", encoding="utf-8")
        (repo / "docs" / "assets").mkdir(parents=True)
        (repo / "MODIFICATIONS.md").write_text("candidate\n", encoding="utf-8")
        (repo / "SECURITY.md").write_text("candidate\n", encoding="utf-8")
        (repo / "docs" / "assets" / "q38lab-architecture.svg").write_text(
            "<svg/>\n", encoding="utf-8",
        )
        (repo / "docs" / "cli.md").write_text("candidate\n", encoding="utf-8")
        (repo / "docs" / "models.md").write_text("candidate\n", encoding="utf-8")
        (repo / "docs" / "qwen4-exp.md").write_text("candidate\n", encoding="utf-8")
        (repo / "THIRD_PARTY_NOTICES.md").write_text("notices\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "runtime"], cwd=repo, check=True)
        runtime = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        digest = evidence.runtime_tree_sha256(repo, runtime)
        (repo / "results" / "rtx5090-test").mkdir(parents=True)
        (repo / "results" / "rtx5090-test" / "summary.json").write_text("{}\n", encoding="utf-8")
        for relative in sorted(evidence.ALLOWED_POST_RUNTIME_FILES):
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("reviewed release metadata\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "evidence"], cwd=repo, check=True)
        tag = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        summary = {"source": {"validated_runtime_commit": runtime, "runtime_tree_sha256": digest}}
        evidence.validate_tag_binding(summary, tag_commit=tag, repo_root=repo)
        (repo / "THIRD_PARTY_NOTICES.md").write_text("changed notices\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "forbidden notice change"], cwd=repo, check=True)
        notice_tag = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        with self.assertRaisesRegex(evidence.EvidenceError, "THIRD_PARTY_NOTICES"):
            evidence.validate_tag_binding(summary, tag_commit=notice_tag, repo_root=repo)
        (repo / "python" / "runtime.py").write_text("value = 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "bad runtime change"], cwd=repo, check=True)
        bad_tag = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        with self.assertRaisesRegex(evidence.EvidenceError, r"python/runtime\.py"):
            evidence.validate_tag_binding(summary, tag_commit=bad_tag, repo_root=repo)


if __name__ == "__main__":
    unittest.main()
