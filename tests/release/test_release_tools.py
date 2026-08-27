from __future__ import annotations

import json
import base64
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "release"))

import repository_audit  # noqa: E402
import rtx5090_harness  # noqa: E402
import source_sbom  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        del tokenize, add_generation_prompt
        assert kwargs == {
            "enable_thinking": False,
            "return_dict": False,
            "thinking_mode": "disabled",
        }
        content = messages[0]["content"]
        return list(range(13 + content.count(" x")))


class DefaultBatchEncodingTokenizer:
    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, return_dict=True, **kwargs
    ):
        del tokenize, add_generation_prompt
        assert kwargs == {
            "enable_thinking": False,
            "thinking_mode": "disabled",
        }
        content = messages[0]["content"]
        input_ids = list(range(13 + content.count(" x")))
        if return_dict:
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        return input_ids

    def encode(self, content, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(1 + content.count(" x")))


class ReleaseToolTests(unittest.TestCase):
    def test_exact_prompt_verifies_rendered_length(self) -> None:
        tokenizer = FakeTokenizer()
        for target in (13, 128, 2048, 8176):
            prompt = rtx5090_harness.exact_prompt(tokenizer, target)
            self.assertEqual(rtx5090_harness.rendered_length(tokenizer, prompt), target)

    def test_exact_needle_prompt_preserves_target_and_one_code(self) -> None:
        tokenizer = FakeTokenizer()
        prompt = rtx5090_harness.exact_needle_prompt(
            tokenizer, 2048, depth=0.65, code="Q38-TEST-C",
        )
        self.assertEqual(rtx5090_harness.rendered_length(tokenizer, prompt), 2048)
        self.assertEqual(prompt.count("Q38-TEST-C"), 1)
        self.assertIn("secret code", prompt.casefold())

    def test_deterministic_vision_fixtures_are_distinct_real_png_files(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is installed by the vision/release profile")

        digests = set()
        for kind in ("ocr", "object", "chart"):
            data_url, digest = rtx5090_harness.synthetic_vision_fixture(kind)
            prefix, encoded = data_url.split(",", 1)
            self.assertEqual(prefix, "data:image/png;base64")
            payload = base64.b64decode(encoded, validate=True)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
            image = Image.open(io.BytesIO(payload))
            image.load()
            self.assertEqual(image.size, (1024, 512) if kind == "ocr" else (768, 512))
            digests.add(digest)
        self.assertEqual(len(digests), 3)

    def test_image_accounting_matches_processor_types_and_grid(self) -> None:
        try:
            import torch
            from PIL import Image
        except ImportError:
            self.skipTest("Torch and Pillow are installed by the release profile")

        class Processor:
            image_processor = SimpleNamespace(merge_size=2)

            def apply_chat_template(self, *_args, **_kwargs):
                return "rendered"

            def __call__(self, **_kwargs):
                return {
                    "input_ids": torch.arange(6).view(1, 6),
                    "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0]]),
                    "image_grid_thw": torch.tensor([[1, 4, 4]]),
                }

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "fixture.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            accounting = rtx5090_harness.rendered_image_accounting(
                Processor(), image_path, "x"
            )
        self.assertEqual(accounting.total_tokens, 6)
        self.assertEqual(accounting.image_tokens, 4)
        self.assertEqual(accounting.text_tokens, 2)

    def test_256k_default_timeout_covers_ttft_and_boundary_decode(self) -> None:
        parser = rtx5090_harness.build_parser()
        args = parser.parse_args([
            "--profile", rtx5090_harness.RTX5090_WSL2_256K_IMAGE_PROFILE.name,
            "--model-dir", str(ROOT),
            "--server-pid", "1",
            "--expected-commit", "1" * 40,
            "--out", str(ROOT / "results" / "test"),
        ])
        self.assertGreaterEqual(args.request_timeout, 1200)

    def test_256k_rejects_request_timeout_below_release_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "fixture.png"
            image.write_bytes(b"fixture")
            code = rtx5090_harness.main([
                "--profile", rtx5090_harness.RTX5090_WSL2_256K_IMAGE_PROFILE.name,
                "--model-dir", str(ROOT),
                "--server-pid", "1",
                "--expected-commit", "1" * 40,
                "--out", str(root / "rtx5090-test"),
                "--image-file", str(image),
                "--https-image-url", "https://example.com/fixture.png",
                "--request-timeout", "1199",
            ])
        self.assertEqual(code, 2)

    def test_rendered_length_disables_batch_encoding_return(self) -> None:
        tokenizer = DefaultBatchEncodingTokenizer()
        prompt = rtx5090_harness.exact_prompt(tokenizer, 13)
        self.assertEqual(prompt, "x")
        self.assertEqual(rtx5090_harness.rendered_length(tokenizer, prompt), 13)
        self.assertEqual(len(tokenizer.encode(prompt, add_special_tokens=False)), 1)

    def test_rendered_length_rejects_non_flat_results(self) -> None:
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = {"attention_mask": [1]}
        with self.assertRaisesRegex(rtx5090_harness.EvidenceError, "token sequence"):
            rtx5090_harness.rendered_length(tokenizer, "x")

        tokenizer.apply_chat_template.return_value = [[1], [2]]
        with self.assertRaisesRegex(rtx5090_harness.EvidenceError, "more than one"):
            rtx5090_harness.rendered_length(tokenizer, "x")

    def test_recorder_returns_the_exact_rounded_timestamps_it_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorder = rtx5090_harness.Recorder(directory, origin=100.0)
            try:
                item = recorder.record(
                    case="soak",
                    iteration=0,
                    warmup=False,
                    stream=False,
                    success=True,
                    prompt_tokens=1,
                    completion_tokens=1,
                    ttft_ms=None,
                    total_ms=999.0,
                    started_elapsed_s=1.2344,
                    finished_elapsed_s=1.2451,
                )
            finally:
                recorder.close()
            persisted = json.loads(
                (directory / "requests.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(item["started_elapsed_s"], persisted["started_elapsed_s"])
            self.assertEqual(item["finished_elapsed_s"], persisted["finished_elapsed_s"])
            self.assertEqual(item["total_ms"], persisted["total_ms"])
            self.assertEqual(recorder.first_started_elapsed_s, 1.2344)
            self.assertNotIn(b"\r\n", (directory / "latency.csv").read_bytes())

            sampler = rtx5090_harness.ResourceSampler.__new__(
                rtx5090_harness.ResourceSampler
            )
            sampler.target = directory / "resource-samples.csv"
            sampler.samples = [{
                "elapsed_s": 1.0,
                "gpu_memory_mib": 1.0,
                "wsl_rss_kib": 1,
                "wsl_rss_source": "test",
                "wsl_swap_kib": 0,
                "minor_faults": 1,
                "major_faults": 0,
                "fault_processes": 1,
                "pcie_rx_mib_s": None,
                "pcie_tx_mib_s": None,
            }]
            sampler._write()
            self.assertNotIn(
                b"\r\n", (directory / "resource-samples.csv").read_bytes()
            )

    def test_server_profile_rejects_manual_cpu_layer_override(self) -> None:
        model_dir = ROOT / "model"
        command = [
            "ft", "serve", "--model", str(model_dir), "--gpu", "0",
            "--served-model-name", "qwen3.8-flash-next-nvfp4", "--host", "127.0.0.1",
            "--port", "1919", "--tp-size", "1", "--max-running-requests", "1",
            "--max-seq-len-override", "8192", "--max-prefill-length", "8192",
            "--num-tokens", "8192", "--memory-ratio", "0.89", "--cache-type", "naive",
            "--attention-backend", "qsa_triton", "--graph", "0", "--moe-backend", "offload",
            "--moe-cache-auto", "--nvfp4-backend", "auto",
        ]
        rtx5090_harness.verify_launch_argv(
            command, model_dir=model_dir, host="127.0.0.1", port=1919
        )
        command.extend(["--moe-cpu-layers", "15"])
        with self.assertRaisesRegex(rtx5090_harness.EvidenceError, "must be omitted"):
            rtx5090_harness.verify_launch_argv(
                command, model_dir=model_dir, host="127.0.0.1", port=1919
            )

    def test_final_clean_check_allows_only_generated_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=repo, check=True
            )
            (repo / "python").mkdir()
            (repo / "python" / "runtime.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "runtime"], cwd=repo, check=True)
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()

            output = repo / "results" / "rtx5090-test"
            output.mkdir(parents=True)
            (output / "summary.json").write_text("{}\n", encoding="utf-8")
            rtx5090_harness.verify_clean_runtime(
                repo, commit, allowed_untracked_root=output
            )

            (repo / "python" / "unexpected.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(
                rtx5090_harness.EvidenceError, "completely clean"
            ):
                rtx5090_harness.verify_clean_runtime(
                    repo, commit, allowed_untracked_root=output
                )

    def test_256k_image_profile_launch_and_resolved_config_are_exact(self) -> None:
        model_dir = ROOT / "model"
        profile = rtx5090_harness.RTX5090_WSL2_256K_IMAGE_PROFILE
        command = [
            "ft", "serve", "--model", str(model_dir), "--gpu", "0",
            "--served-model-name", "qwen3.8-flash-next-nvfp4", "--host", "127.0.0.1",
            "--port", "1919", "--tp-size", "1", "--max-running-requests", "1",
            "--max-seq-len-override", "262144", "--max-prefill-length", "512",
            "--num-tokens", "262144", "--memory-ratio", "0.89", "--cache-type", "naive",
            "--attention-backend", "qsa_triton_sm120", "--graph", "0",
            "--moe-backend", "offload", "--moe-cache-auto", "--nvfp4-backend", "auto",
        ]
        rtx5090_harness.verify_launch_argv(
            command, model_dir=model_dir, host="127.0.0.1", port=1919, profile=profile
        )
        config = rtx5090_harness.resolved_config(model_dir, profile)
        self.assertEqual(config["profile"], "rtx5090-wsl2-256k-image")
        self.assertEqual(config["settings"]["max_seq_len"], 262144)
        self.assertEqual(config["settings"]["ple_cache_bytes"], 4 * 1024**3)
        self.assertTrue(config["settings"]["qsa_require_native_topk"])
        self.assertTrue(config["settings"]["vision_enabled"])

    def test_runtime_telemetry_fails_closed_when_server_does_not_export_it(self) -> None:
        with self.assertRaisesRegex(rtx5090_harness.EvidenceError, "does not expose"):
            rtx5090_harness._runtime_telemetry({"model": {"ctx": 262144}})

    def test_runtime_telemetry_requires_native_selector_and_valid_moe_counters(self) -> None:
        telemetry = {
            "selector": {
                "workspace_peak_bytes": 1024,
                "native_calls": 3,
                "fallback_calls": 0,
                "errors": 0,
            },
            "ple": {
                "bytes_read": 4096,
                "cache_hits": 1,
                "cache_misses": 1,
                "wait_ms": 0.5,
                "page_faults": 0,
            },
            "vision": {"image_tokens": 16, "latency_ms": 1.0},
            "prefill_chunks": {"count": 1, "total_ms": 2.0},
            "moe_prefill": {
                "active_rows": 128,
                "possible_rows": 512,
                "bytes_copied": 400,
                "full_bytes": 1000,
                "row_fraction": 0.25,
                "byte_fraction": 0.4,
            },
        }
        snapshot = rtx5090_harness._sanitized_runtime_snapshot(
            telemetry, "cold"
        )
        self.assertEqual(snapshot["selector"]["native_calls"], 3)

        moe = telemetry["moe_prefill"]
        for key, value, message in (
            ("active_rows", 128.0, "non-negative integer"),
            ("active_rows", 513, "cannot exceed possible_rows"),
            ("bytes_copied", 1001, "cannot exceed full_bytes"),
            ("row_fraction", 0.5, "does not match"),
        ):
            original = moe[key]
            moe[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(
                    rtx5090_harness.EvidenceError, message
                ):
                    rtx5090_harness._sanitized_runtime_snapshot(telemetry, "cold")
            moe[key] = original

        del telemetry["selector"]["native_calls"]
        with self.assertRaisesRegex(
            rtx5090_harness.EvidenceError, "selector.native_calls"
        ):
            rtx5090_harness._sanitized_runtime_snapshot(telemetry, "cold")

    def test_published_moe_prefill_schemas_constrain_types_and_ranges(self) -> None:
        counters = {"active_rows", "possible_rows", "bytes_copied", "full_bytes"}
        fractions = {"row_fraction", "byte_fraction"}
        for schema_name in (
            "runtime-telemetry.schema.json", "summary.schema.json"
        ):
            schema = json.loads(
                (ROOT / "results" / "schema" / schema_name).read_text(
                    encoding="utf-8"
                )
            )
            definition = schema["$defs"]["moePrefillGroup"]
            self.assertEqual(definition["type"], "object")
            self.assertFalse(definition["additionalProperties"])
            self.assertEqual(set(definition["required"]), counters | fractions)
            self.assertEqual(set(definition["properties"]), counters | fractions)
            for key in counters:
                self.assertEqual(
                    definition["properties"][key],
                    {"type": "integer", "minimum": 0},
                )
            for key in fractions:
                self.assertEqual(
                    definition["properties"][key],
                    {"type": "number", "minimum": 0, "maximum": 1},
                )

    def test_refresh_runtime_telemetry_preserves_six_digit_moe_ratios(self) -> None:
        telemetry = {
            "selector": {
                "workspace_peak_bytes": 1024,
                "native_calls": 3,
                "fallback_calls": 0,
                "errors": 0,
            },
            "ple": {
                "bytes_read": 4096,
                "cache_hits": 1,
                "cache_misses": 1,
                "wait_ms": 0.5,
                "page_faults": 0,
            },
            "vision": {"image_tokens": 16, "latency_ms": 1.0},
            "prefill_chunks": {"count": 1, "total_ms": 2.0},
            "moe_prefill": {
                "active_rows": 100,
                "possible_rows": 512,
                "bytes_copied": 400,
                "full_bytes": 1000,
                "row_fraction": 100 / 512,
                "byte_fraction": 0.4,
            },
        }
        current = {key: {} for key in telemetry}
        refreshed = rtx5090_harness.refresh_runtime_telemetry(
            current, {"q38lab": telemetry}
        )
        self.assertEqual(refreshed["moe_prefill"]["row_fraction"], 0.195312)
        self.assertIsInstance(refreshed["moe_prefill"]["active_rows"], int)

    def test_sse_rejects_invalid_json_and_missing_done(self) -> None:
        class Response:
            status = 200

            def __init__(self, lines):
                self.lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(self.lines)

        invalid = Response([b"data: {bad}\n", b"data: [DONE]\n"])
        with mock.patch.object(rtx5090_harness.LOCAL_OPENER, "open", return_value=invalid):
            with self.assertRaisesRegex(rtx5090_harness.EvidenceError, "invalid JSON"):
                rtx5090_harness.post_sse("http://127.0.0.1:1919/v1/chat/completions", {}, 1)
        missing = Response([
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
        ])
        with mock.patch.object(rtx5090_harness.LOCAL_OPENER, "open", return_value=missing):
            with self.assertRaisesRegex(rtx5090_harness.EvidenceError, "without the required"):
                rtx5090_harness.post_sse("http://127.0.0.1:1919/v1/chat/completions", {}, 1)

    def test_sse_captures_final_finish_reason(self) -> None:
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n',
                    b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n',
                    b'data: [DONE]\n',
                ])

        with mock.patch.object(rtx5090_harness.LOCAL_OPENER, "open", return_value=Response()):
            result = rtx5090_harness.post_sse(
                "http://127.0.0.1:1919/v1/chat/completions", {}, 1
            )
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.completion_tokens, 2)

        class InvalidResponse(Response):
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":7}]}\n',
                    b'data: [DONE]\n',
                ])

        with mock.patch.object(
            rtx5090_harness.LOCAL_OPENER, "open", return_value=InvalidResponse()
        ):
            with self.assertRaisesRegex(
                rtx5090_harness.EvidenceError, "finish_reason"
            ):
                rtx5090_harness.post_sse(
                    "http://127.0.0.1:1919/v1/chat/completions", {}, 1
                )

    def test_tool_arguments_require_a_valid_shanghai_city(self) -> None:
        self.assertIsNotNone(rtx5090_harness.valid_city_arguments('{"city":"Shanghai"}'))
        self.assertIsNone(rtx5090_harness.valid_city_arguments('{"city":""}'))
        self.assertIsNone(rtx5090_harness.valid_city_arguments('{"city":123}'))
        self.assertIsNone(rtx5090_harness.valid_city_arguments('{"city":"Beijing"}'))

    def test_thinking_contract_is_strict_for_none_and_high(self) -> None:
        self.assertTrue(rtx5090_harness.thinking_response_valid("none", "answer", ""))
        self.assertFalse(rtx5090_harness.thinking_response_valid("none", "answer", "hidden"))
        self.assertFalse(rtx5090_harness.thinking_response_valid("none", "", ""))
        self.assertTrue(rtx5090_harness.thinking_response_valid("high", "answer", "reason"))
        self.assertFalse(rtx5090_harness.thinking_response_valid("high", "answer", ""))

    def test_image_thinking_requires_visible_answer_and_untruncated_finish(self) -> None:
        self.assertEqual(rtx5090_harness.IMAGE_THINKING_MAX_TOKENS, 512)
        self.assertTrue(
            rtx5090_harness.image_thinking_response_valid(
                "The image is a red square.", "visual reasoning", "stop", 511
            )
        )
        for content, finish_reason, completion_tokens in (
            ("", "length", 511),
            ("The image is a red square.", "length", 511),
            ("The image is a red square.", "stop", 0),
            ("The image is a red square.", "stop", 512),
        ):
            with self.subTest(
                content=content,
                finish_reason=finish_reason,
                completion_tokens=completion_tokens,
            ):
                self.assertFalse(
                    rtx5090_harness.image_thinking_response_valid(
                        content,
                        "visual reasoning",
                        finish_reason,
                        completion_tokens,
                    )
                )

    def test_access_code_tool_arguments_are_exact(self) -> None:
        self.assertIsNotNone(
            rtx5090_harness.valid_access_code_arguments('{"code":"382741"}')
        )
        for arguments in (
            '{"code":"Q382741"}',
            '{"code":"382742"}',
            '{"code":382741}',
            '{"code":"382741","extra":true}',
            '{}',
            'not-json',
        ):
            with self.subTest(arguments=arguments):
                self.assertIsNone(
                    rtx5090_harness.valid_access_code_arguments(arguments)
                )

    def test_tokenizer_never_enables_remote_code(self) -> None:
        tokenizer = mock.Mock()
        auto = mock.Mock()
        auto.from_pretrained.return_value = tokenizer
        module = mock.Mock(AutoTokenizer=auto)
        with mock.patch.dict(sys.modules, {"transformers": module}):
            self.assertIs(rtx5090_harness.load_tokenizer(ROOT), tokenizer)
        auto.from_pretrained.assert_called_once_with(
            str(ROOT), trust_remote_code=False, local_files_only=True
        )

    def test_source_sbom_is_spdx_and_repeatable(self) -> None:
        first = source_sbom.build_document("qwen38-next-5090-lab", "test", ROOT)
        second = source_sbom.build_document("qwen38-next-5090-lab", "test", ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first["spdxVersion"], "SPDX-2.3")
        self.assertTrue(first["files"])
        json.dumps(first)

    def test_repository_audit_parses_local_links(self) -> None:
        self.assertIsNone(repository_audit._link_target("https://example.com/a"))
        self.assertEqual(repository_audit._link_target("docs/guide.md#install"), "docs/guide.md")

    def test_repository_audit_blocks_upstream_branding_and_publishers(self) -> None:
        candidates = [
            "README.md",
            "install.sh",
            "assets/freetoken-logo-dark.svg",
            "assets/freetoken-icon.svg",
            "assets/desktop-console.png",
            "assets/freetoken-wechatgroup.png",
            "scripts/build-release-wheels.sh",
            "scripts/publish-wheels.sh",
            "scripts/ci/manylinux-build.sh",
            "scripts/ci/retag-manylinux.py",
            "freetoken-kernel-cache/cache.py",
        ]
        self.assertEqual(
            repository_audit.forbidden_rebrand_paths(candidates),
            sorted(candidates[1:]),
        )

    def test_release_workflow_binds_tag_version_runtime_and_source_only_assets(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "source-release.yml").read_text(encoding="utf-8")
        self.assertIn('git rev-parse "${RELEASE_TAG}^{commit}"', workflow)
        self.assertIn("python/freetoken/version.py", workflow)
        self.assertIn('--tag-commit "$RELEASE_COMMIT"', workflow)
        self.assertNotIn("twine upload", workflow)
        self.assertNotIn("docker push", workflow)
        self.assertNotIn("bdist_wheel", workflow)


if __name__ == "__main__":
    unittest.main()
