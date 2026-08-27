from __future__ import annotations

import json
import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "release"))

import evidence  # noqa: E402


RUNTIME_COMMIT = "1" * 40


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
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "runtime"], cwd=repo, check=True)
        runtime = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        digest = evidence.runtime_tree_sha256(repo, runtime)
        (repo / "results" / "rtx5090-test").mkdir(parents=True)
        (repo / "results" / "rtx5090-test" / "summary.json").write_text("{}\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "evidence"], cwd=repo, check=True)
        tag = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        summary = {"source": {"validated_runtime_commit": runtime, "runtime_tree_sha256": digest}}
        evidence.validate_tag_binding(summary, tag_commit=tag, repo_root=repo)
        (repo / "python" / "runtime.py").write_text("value = 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "bad runtime change"], cwd=repo, check=True)
        bad_tag = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        with self.assertRaisesRegex(evidence.EvidenceError, "runtime files"):
            evidence.validate_tag_binding(summary, tag_commit=bad_tag, repo_root=repo)


if __name__ == "__main__":
    unittest.main()
