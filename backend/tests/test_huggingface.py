from pathlib import Path

import pytest
from app.tasks import huggingface
from app.tasks.huggingface import (
    cache_repository_path,
    serialize_card_data,
    validate_repository_id,
)
from huggingface_hub import ModelCardData


@pytest.mark.parametrize(
    "value",
    ["../model", "org/../../etc", "org/model/extra", "/absolute/model", "org\\model"],
)
def test_repository_validation_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        validate_repository_id(value)


def test_cache_repository_path_stays_inside_cache(tmp_path):
    result = cache_repository_path(tmp_path, "nvidia/Nemotron")

    assert result == Path(tmp_path, "models--nvidia--Nemotron")
    assert result.is_relative_to(tmp_path)


def test_model_card_data_uses_hub_serialization_contract():
    card = ModelCardData(license="apache-2.0", pipeline_tag="text-generation")

    assert serialize_card_data(card)["license"] == "apache-2.0"
    assert serialize_card_data({"license": "mit"}) == {"license": "mit"}
    assert serialize_card_data(None) == {}


def test_download_size_only_counts_selected_files():
    files = [
        {"name": "config.json", "size": 100},
        {"name": "model.safetensors", "size": 1_000},
        {"name": "README.md", "size": 50},
    ]
    calculate = getattr(huggingface, "selected_download_size", lambda *_args, **_kwargs: None)

    assert calculate(files, include=["*.json", "*.safetensors"], exclude=["README*"]) == 1_100


def test_disk_preflight_rejects_download_larger_than_free_space():
    validate = getattr(huggingface, "validate_disk_capacity", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="free disk space"):
        validate(total_bytes=10_000, existing_bytes=1_000, free_bytes=8_999)

    assert validate(total_bytes=10_000, existing_bytes=2_000, free_bytes=8_000) == 8_000


def test_download_handler_runs_disk_preflight_before_starting_cli(tmp_path, monkeypatch):
    service = huggingface.HuggingFaceService(tmp_path / "hub")
    monkeypatch.setattr(
        service,
        "info",
        lambda *_args: {
            "sha": "abc",
            "siblings": [{"name": "model.safetensors", "size": 10_000}],
            "total_size": 10_000,
        },
    )
    monkeypatch.setattr(huggingface.shutil, "which", lambda _name: "/usr/bin/hf")
    monkeypatch.setattr(
        huggingface.shutil,
        "disk_usage",
        lambda _path: huggingface.shutil._ntuple_diskusage(20_000, 11_001, 8_999),
    )
    monkeypatch.setattr(
        huggingface.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("CLI started before disk preflight"),
    )

    class Context:
        def update(self, **_kwargs):
            return None

    with pytest.raises(RuntimeError, match="free disk space"):
        service.download_handler(
            Context(),
            {"repository_id": "org/model", "revision": "main", "include": [], "exclude": []},
        )


def test_snapshot_integrity_check_requires_every_selected_file(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    files = [
        {"name": "config.json", "size": 2},
        {"name": "model.safetensors", "size": 100},
    ]
    verify = getattr(huggingface, "verify_snapshot_files", lambda *_args, **_kwargs: None)

    assert verify(snapshot, files, include=["config.json"], exclude=[]) == ["config.json"]
    with pytest.raises(RuntimeError, match="model.safetensors"):
        verify(snapshot, files, include=[], exclude=[])


def test_snapshot_integrity_allows_hub_links_to_repository_blobs(tmp_path):
    repository = tmp_path / "models--org--model"
    snapshot = repository / "snapshots" / "abc"
    blob = repository / "blobs" / "hash"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    is_safe = getattr(huggingface, "is_safe_snapshot_file", lambda *_args: False)

    assert is_safe(
        snapshot,
        snapshot / "model.safetensors",
        blob,
    ) is True
    assert is_safe(snapshot, snapshot / "../escape", tmp_path / "outside") is False


def test_huggingface_cli_uses_writable_cache_home(tmp_path):
    cache = tmp_path / "hf-cache" / "hub"
    build_environment = getattr(
        huggingface,
        "huggingface_environment",
        lambda *_args, **_kwargs: {},
    )

    environment = build_environment(cache, {"HOME": "/"})

    assert environment["HOME"] == str(cache.parent)
    assert environment["HF_HOME"] == str(cache.parent)
    assert environment["HF_HUB_CACHE"] == str(cache)
    assert environment["HF_XET_CACHE"] == str(cache.parent / "xet")
    assert environment["XDG_CACHE_HOME"] == str(cache.parent / ".cache")


def test_huggingface_cli_errors_are_sanitized():
    sanitize = getattr(huggingface, "sanitize_cli_output", lambda _value: "")

    result = sanitize("\x1b[31mPermission denied hf_abcdefghijklmnopqrstuvwxyz\x1b[0m")

    assert result == "Permission denied [REDACTED_HF_TOKEN]"
