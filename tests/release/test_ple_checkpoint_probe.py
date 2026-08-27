from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts" / "release"))

try:
    import pytest
    import torch
    from safetensors.torch import save_file
except ModuleNotFoundError as exc:  # repository-safety's dependency-light unittest pass
    raise unittest.SkipTest(f"PLE probe tests require the CPU oracle dependencies: {exc}")

import ple_checkpoint_probe as probe  # noqa: E402
from freetoken.models.qwen4_exp.ple import Qwen4ExpNGramHasher  # noqa: E402


def _synthetic_checkpoint(root: Path) -> Path:
    model_dir = root / "synthetic-model"
    model_dir.mkdir()
    text = {
        "vocab_size": 128,
        "hidden_size": 8,
        "eos_token_id": 127,
        "ple_layer_ids": [2],
        "ple_embed_dim": 8,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 11,
        "make_ngram_vocab_size_divisible_by": 4,
        "split_ngram_parts": 4,
        "ple_embedding_dtype": "float32",
    }
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": text}), encoding="utf-8"
    )
    hasher = Qwen4ExpNGramHasher(
        unigram_vocab_size=text["vocab_size"],
        eos_token_id=text["eos_token_id"],
        ngram_vocab_size_base=text["ngram_vocab_size_base"],
        ngram_size=text["ngram_size"],
        heads_per_ngram=text["heads_per_ngram"],
        ple_layer_index=0,
        seed=text.get("seed", 1234),
        make_vocab_size_divisible_by=text["make_ngram_vocab_size_divisible_by"],
    )
    total_rows = hasher.layout.padded_vocab_size
    assert total_rows % text["split_ngram_parts"] == 0
    rows_per_shard = total_rows // text["split_ngram_parts"]
    row_width = text["ple_embed_dim"] // hasher.num_heads
    base = (
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_embedding"
    )
    weight_map: dict[str, str] = {}
    for shard_index in range(text["split_ngram_parts"]):
        name = f"{base}.shard_{shard_index}.weight"
        file_name = f"ple-{shard_index}.safetensors"
        start = shard_index * rows_per_shard * row_width
        values = (
            torch.arange(
                start,
                start + rows_per_shard * row_width,
                dtype=torch.float32,
            ).reshape(rows_per_shard, row_width)
            / 100
        )
        save_file({name: values}, model_dir / file_name)
        weight_map[name] = file_name
    scale_name = f"{base}.weight_scale"
    save_file(
        {scale_name: torch.tensor([0.5], dtype=torch.float32)},
        model_dir / "scale.safetensors",
    )
    weight_map[scale_name] = "scale.safetensors"
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return model_dir


def test_debug_probe_matches_safetensors_and_covers_every_boundary(tmp_path: Path) -> None:
    model_dir = _synthetic_checkpoint(tmp_path)

    report = probe.run_probe(
        model_dir,
        backend="mmap",
        device="cpu",
        release_mode=False,
    )

    assert report["status"] == "pass"
    assert report["release_qualified"] is False
    assert report["coverage"]["shard_count"] == 4
    assert report["coverage"]["sample_count"] == 16
    assert report["coverage"]["hash_sample_count"] == 8
    assert report["coverage"]["all_shard_first_rows"] is True
    assert report["coverage"]["all_shard_last_rows"] is True
    assert all(record["match"] is True for record in report["records"])
    assert all(
        record["ground_truth"]["sha256"]
        == record["auxiliary_bank"]["sha256"]
        for record in report["records"]
    )
    serialized = json.dumps(report)
    assert '"values"' not in serialized
    assert str(model_dir.resolve()) not in serialized
    assert report["loader_mapping"]["normal_state_dict_action"] == "skip"


def test_probe_fails_closed_on_one_corrupted_auxiliary_row(tmp_path: Path) -> None:
    model_dir = _synthetic_checkpoint(tmp_path)

    class CorruptingBank:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.row_count = wrapped.row_count
            self.row_width = wrapped.row_width

        def read_rows(self, *args, **kwargs):
            value = self._wrapped.read_rows(*args, **kwargs)
            value[0, 0] += 1
            return value

        def telemetry(self):
            return self._wrapped.telemetry()

        def close(self):
            self._wrapped.close()

    def corrupting_factory(layout, backend, device):
        return CorruptingBank(probe.open_auxiliary_bank(layout, backend, device))

    with pytest.raises(probe.ProbeError, match="parity failed"):
        probe.run_probe(
            model_dir,
            backend="mmap",
            device="cpu",
            release_mode=False,
            bank_factory=corrupting_factory,
        )


def test_discovery_fails_when_one_indexed_shard_mapping_is_missing(tmp_path: Path) -> None:
    model_dir = _synthetic_checkpoint(tmp_path)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing = next(name for name in index["weight_map"] if ".shard_2.weight" in name)
    del index["weight_map"][missing]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(probe.ProbeError, match="missing PLE tensor"):
        probe.discover_layout(model_dir)


def test_release_mode_rejects_cpu_mmap_before_checkpoint_access(tmp_path: Path) -> None:
    with pytest.raises(probe.ProbeError, match="requires backend"):
        probe.run_probe(
            tmp_path / "does-not-exist",
            backend="mmap",
            device="cpu",
            release_mode=True,
        )


def test_debug_cli_writes_an_atomic_json_report(tmp_path: Path) -> None:
    model_dir = _synthetic_checkpoint(tmp_path)
    output = tmp_path / "evidence" / "ple-checkpoint-probe.json"

    status = probe.main(
        [
            str(model_dir),
            "--backend",
            "mmap",
            "--device",
            "cpu",
            "--debug-nonrelease",
            "--out",
            str(output),
        ]
    )

    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["release_qualified"] is False
    assert list(output.parent.glob("*.tmp")) == []
