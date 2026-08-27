from __future__ import annotations

import json

from q38lab import attestation
from q38lab.config import resolve_serve_config


def _config(tmp_path):
    return resolve_serve_config(
        profile_name="rtx5090-wsl2",
        cli={"model_dir": tmp_path, "unsafe_non_loopback": False},
        env={},
    )


def test_atomic_attestation_has_release_binding_fields(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(attestation, "_proc_start_ticks", lambda pid: 12345)
    monkeypatch.setattr(
        attestation, "_git_identity", lambda root: ("a" * 40, True),
    )
    monkeypatch.setattr(attestation, "_source_root", lambda: tmp_path)
    target = tmp_path / "state" / "serve.json"
    path = attestation.write_launch_attestation(
        config,
        config.to_ft_argv(),
        target=target,
        preflight={"ple_checkpoint_probe": {"status": "pass"}},
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == "1.0"
    assert document["runtime_commit"] == "a" * 40
    assert document["clean_tree"] is True
    assert document["model_realpath"] == str(tmp_path.resolve())
    assert document["resolved_config"] == config.public_dict()
    assert document["argv"] == config.to_ft_argv()
    assert document["proc_start_ticks"] == 12345
    assert document["preflight"] == {
        "ple_checkpoint_probe": {"status": "pass"}
    }


def test_cleanup_never_removes_another_process_attestation(tmp_path):
    path = tmp_path / "serve.json"
    path.write_text('{"pid": 99}\n', encoding="utf-8")
    attestation.remove_launch_attestation(path, pid=100)
    assert path.exists()
    attestation.remove_launch_attestation(path, pid=99)
    assert not path.exists()
