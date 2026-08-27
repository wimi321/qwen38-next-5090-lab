from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
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
