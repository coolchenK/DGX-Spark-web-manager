import json
import time
from pathlib import Path

import pytest
from app.services import model_evidence
from app.services.model_evidence import ModelEvidenceLoader, tokenizer_fingerprint


def _symlink_or_skip(link: Path, target: Path) -> None:
    resolved_target = target if target.is_absolute() else link.parent / target
    try:
        link.symlink_to(target, target_is_directory=resolved_target.is_dir())
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform")


def _hub_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "hub" / "models--org--model"
    snapshot = repository / "snapshots" / "revision"
    (repository / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    return repository, snapshot


def _linked_hub_blob(
    repository: Path,
    snapshot: Path,
    filename: str,
    digest: str,
    data: bytes,
) -> None:
    (repository / "blobs" / digest).write_bytes(data)
    _symlink_or_skip(
        snapshot / filename,
        Path("..") / ".." / "blobs" / digest,
    )


def test_model_evidence_loads_structured_files_and_allowlisted_card_values(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"max_position_embeddings":65536,"hidden_size":4096,'
        '"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8}',
        encoding="utf-8",
    )
    (model / "generation_config.json").write_text(
        '{"temperature":0.7,"top_p":0.8}', encoding="utf-8"
    )
    (model / "README.md").write_text(
        "```bash\nvllm serve org/model --max-model-len 32768 --max-num-seqs 4\n```\n"
        '```json\n{"temperature": 0.6, "top_p": 0.95}\n```',
        encoding="utf-8",
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.config == {
        "max_position_embeddings": 65536,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }
    assert evidence.card_deployment_values == {
        "context_length": 32768,
        "max_concurrency": 4,
    }
    assert evidence.card_generation_values == {"temperature": 0.6, "top_p": 0.95}
    assert evidence.local_generation_values == {"temperature": 0.7, "top_p": 0.8}
    assert len(evidence.evidence_hash) == 64


def test_model_card_commands_never_return_unknown_flags_or_values(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "README.md").write_text(
        "```bash\nvllm serve x --max-model-len 8192 --evil-command rm-all\n```",
        encoding="utf-8",
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)
    dumped = json.dumps(evidence.model_dump())

    assert evidence.card_deployment_values == {"context_length": 8192}
    assert "evil" not in dumped
    assert "rm-all" not in dumped


def test_sanitized_shell_fence_preserves_following_newline(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "README.md").write_text(
        "before\n```bash\nvllm serve x --max-model-len 4096\n```\nend\n",
        encoding="utf-8",
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.card_text.endswith('```\nend\n')


def test_posix_model_root_open_rejects_root_symlink(tmp_path):
    if model_evidence.os.name != "posix":
        pytest.skip("POSIX O_NOFOLLOW is unavailable on this platform")
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "config.json").write_text("{}", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this platform")

    with pytest.raises((OSError, ValueError)):
        with model_evidence._open_model_file(link, "config.json"):
            pass


def test_huggingface_snapshot_blob_links_are_read_as_model_evidence(tmp_path):
    repository, snapshot = _hub_snapshot(tmp_path)
    _linked_hub_blob(
        repository,
        snapshot,
        "config.json",
        "a" * 40,
        b'{"max_position_embeddings":32768,"hidden_size":4096}',
    )
    _linked_hub_blob(
        repository,
        snapshot,
        "generation_config.json",
        "b" * 64,
        b'{"temperature":0.4,"top_p":0.9}',
    )
    _linked_hub_blob(
        repository,
        snapshot,
        "README.md",
        "c" * 40,
        b"```bash\nvllm serve org/model --max-model-len 16384\n```\n",
    )
    _linked_hub_blob(
        repository,
        snapshot,
        "tokenizer.json",
        "d" * 64,
        b'{"version":"1.0"}',
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(snapshot)

    assert evidence.config == {
        "max_position_embeddings": 32768,
        "hidden_size": 4096,
    }
    assert evidence.local_generation_values == {"temperature": 0.4, "top_p": 0.9}
    assert evidence.card_deployment_values == {"context_length": 16384}
    assert evidence.tokenizer_fingerprint is not None
    assert evidence.warnings == []


def test_huggingface_blob_link_parser_accepts_only_canonical_posix_targets():
    sha1 = "a" * 40
    sha256 = "A1" * 32

    assert model_evidence._parse_huggingface_blob_link_target(f"../../blobs/{sha1}") == sha1
    assert model_evidence._parse_huggingface_blob_link_target(f"../../blobs/{sha256}") == sha256
    assert model_evidence._parse_huggingface_blob_link_target(f"/blobs/{sha1}") is None
    assert model_evidence._parse_huggingface_blob_link_target(f"../../blobs/{sha1}/file") is None
    assert model_evidence._parse_huggingface_blob_link_target("../../blobs/not-a-hash") is None
    assert model_evidence._parse_huggingface_blob_link_target(f"../blobs/{sha1}") is None


@pytest.mark.parametrize(
    "target_factory",
    [
        pytest.param(lambda repository, digest: repository / "blobs" / digest, id="absolute"),
        pytest.param(
            lambda _repository, _digest: Path("..") / ".." / "blobs" / "not-a-hash",
            id="non-hash",
        ),
        pytest.param(lambda _repository, digest: Path("..") / "blobs" / digest, id="one-parent"),
        pytest.param(
            lambda _repository, digest: Path("..") / ".." / "other" / digest,
            id="wrong-directory",
        ),
        pytest.param(
            lambda _repository, digest: Path("..") / ".." / "blobs" / digest / "config.json",
            id="extra-level",
        ),
    ],
)
def test_huggingface_snapshot_rejects_noncanonical_blob_link_targets(
    tmp_path, target_factory
):
    repository, snapshot = _hub_snapshot(tmp_path)
    digest = "a" * 40
    (repository / "blobs" / digest).write_text('{"hidden_size":4096}', encoding="utf-8")
    _symlink_or_skip(snapshot / "config.json", target_factory(repository, digest))

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(snapshot)

    assert evidence.config == {}
    assert evidence.warnings == ["config.json is not a safe regular file"]


@pytest.mark.parametrize(
    ("repository_name", "snapshot_directory"),
    [
        ("datasets--org--model", "snapshots"),
        ("models--org--model", "revisions"),
    ],
)
def test_huggingface_blob_links_require_the_exact_model_snapshot_hierarchy(
    tmp_path, repository_name, snapshot_directory
):
    repository = tmp_path / "hub" / repository_name
    snapshot = repository / snapshot_directory / "revision"
    digest = "a" * 40
    (repository / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repository / "blobs" / digest).write_text('{"hidden_size":4096}', encoding="utf-8")
    _symlink_or_skip(
        snapshot / "config.json",
        Path("..") / ".." / "blobs" / digest,
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(snapshot)

    assert evidence.config == {}
    assert evidence.warnings == ["config.json is not a safe regular file"]


def test_huggingface_snapshot_rejects_a_symlinked_blob_directory(tmp_path):
    repository, snapshot = _hub_snapshot(tmp_path)
    (repository / "blobs").rmdir()
    outside_blobs = tmp_path / "outside-blobs"
    outside_blobs.mkdir()
    digest = "a" * 40
    (outside_blobs / digest).write_text('{"hidden_size":4096}', encoding="utf-8")
    _symlink_or_skip(repository / "blobs", outside_blobs)
    _symlink_or_skip(
        snapshot / "config.json",
        Path("..") / ".." / "blobs" / digest,
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(snapshot)

    assert evidence.config == {}
    assert evidence.warnings == ["config.json is not a safe regular file"]


def test_huggingface_snapshot_rejects_a_symlinked_blob_file(tmp_path):
    repository, snapshot = _hub_snapshot(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"hidden_size":4096}', encoding="utf-8")
    digest = "a" * 40
    _symlink_or_skip(repository / "blobs" / digest, outside)
    _symlink_or_skip(
        snapshot / "config.json",
        Path("..") / ".." / "blobs" / digest,
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(snapshot)

    assert evidence.config == {}
    assert evidence.warnings == ["config.json is not a safe regular file"]


def test_huggingface_snapshot_rejects_a_blob_directory_as_file_content(tmp_path):
    repository, snapshot = _hub_snapshot(tmp_path)
    digest = "a" * 40
    (repository / "blobs" / digest).mkdir()
    _symlink_or_skip(
        snapshot / "config.json",
        Path("..") / ".." / "blobs" / digest,
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(snapshot)

    assert evidence.config == {}
    assert evidence.warnings == ["config.json is not a safe regular file"]


def test_missing_shell_value_does_not_consume_following_option_or_leak_it(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "README.md").write_text(
        "```bash\nvllm serve x --quantization --evil-command "
        "--max-model-len --max-num-seqs 4\n```",
        encoding="utf-8",
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)
    dumped = json.dumps(evidence.model_dump())

    assert evidence.card_deployment_values == {"max_concurrency": 4}
    assert "evil" not in dumped


def test_inline_quantization_is_accepted_but_option_like_values_are_rejected(tmp_path):
    valid = tmp_path / "valid"
    invalid = tmp_path / "invalid"
    valid.mkdir()
    invalid.mkdir()
    (valid / "README.md").write_text(
        "```sh\nvllm serve x --quantization=awq\n```", encoding="utf-8"
    )
    (invalid / "README.md").write_text(
        "```sh\nvllm serve x --quantization=-awq\n```", encoding="utf-8"
    )

    valid_evidence = ModelEvidenceLoader(card_max_chars=100_000).load(valid)
    invalid_evidence = ModelEvidenceLoader(card_max_chars=100_000).load(invalid)

    assert valid_evidence.card_deployment_values == {"quantization": "awq"}
    assert invalid_evidence.card_deployment_values == {}


def test_tokenizer_fingerprint_changes_when_special_tokens_change(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "tokenizer.json").write_text('{"v":1}', encoding="utf-8")
    (second / "tokenizer.json").write_text('{"v":1}', encoding="utf-8")
    (first / "special_tokens_map.json").write_text(
        '{"eos_token":"a"}', encoding="utf-8"
    )
    (second / "special_tokens_map.json").write_text(
        '{"eos_token":"b"}', encoding="utf-8"
    )

    assert tokenizer_fingerprint(first) != tokenizer_fingerprint(second)


def test_unclosed_and_repeated_fences_are_scanned_linearly(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    body = "\n".join(
        ["```bash", "vllm serve x --max-model-len 4096"] * 5_000
    )
    (model / "README.md").write_text(body, encoding="utf-8")

    started = time.perf_counter()
    evidence = ModelEvidenceLoader(card_max_chars=500_000).load(model)
    elapsed = time.perf_counter() - started

    assert elapsed < 2
    assert evidence.card_deployment_values == {}


def test_oversized_tokenizer_file_is_reported_without_fingerprint(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_bytes(b"12345")
    monkeypatch.setattr(model_evidence, "MAX_TOKENIZER_FILE_BYTES", 4)

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.tokenizer_fingerprint is None
    assert evidence.warnings == ["tokenizer.json is too large for tokenizer fingerprint"]


def test_invalid_oversized_and_non_dictionary_json_is_bounded(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"{" + b"x" * (1024**2))
    (model / "generation_config.json").write_text("[]", encoding="utf-8")

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.config == {}
    assert evidence.generation_config == {}
    assert evidence.local_generation_values == {}
    assert len(evidence.warnings) == 2
    assert len(json.dumps(evidence.warnings)) < 500


def test_malformed_json_does_not_abort_evidence_loading(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"broken":', encoding="utf-8")

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.config == {}
    assert evidence.warnings == ["config.json is not valid JSON"]


def test_card_is_truncated_and_only_allowlisted_json_keys_are_extracted(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    prefix = '```json\n{"temperature": 0.2, "unknown_secret": "discard"}\n```\n'
    (model / "README.md").write_text(prefix + "x" * 500, encoding="utf-8")

    evidence = ModelEvidenceLoader(card_max_chars=len(prefix)).load(model)

    assert evidence.card_text == prefix
    assert evidence.card_generation_values == {"temperature": 0.2}
    assert "unknown_secret" not in json.dumps(
        {
            "card_data": evidence.card_data,
            "card_generation_values": evidence.card_generation_values,
            "warnings": evidence.warnings,
        }
    )


def test_remote_card_overlay_reuses_local_evidence_and_has_a_stable_hash(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"max_position_embeddings":16384,"hidden_size":4096}', encoding="utf-8"
    )
    (model / "generation_config.json").write_text(
        '{"temperature":0.7}', encoding="utf-8"
    )
    (model / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
    (model / "README.md").write_text(
        "```bash\nvllm serve org/model --max-model-len 4096\n```",
        encoding="utf-8",
    )
    remote_card = (
        "```bash\nvllm serve org/model --max-model-len 8192 "
        "--gpu-memory-utilization 0.75 --unknown secret\n```\n"
        '```json\n{"temperature":0.6,"unknown":"discard"}\n```'
    )
    loader = ModelEvidenceLoader(card_max_chars=100_000)

    local = loader.load(model)
    first = loader.load_with_card(model, remote_card)
    second = loader.load_with_card(model, remote_card)

    assert first.config == local.config
    assert first.local_generation_values == {"temperature": 0.7}
    assert first.tokenizer_fingerprint == local.tokenizer_fingerprint
    assert first.card_deployment_values == {
        "context_length": 8192,
        "memory_fraction": 0.75,
    }
    assert first.card_generation_values == {"temperature": 0.6}
    assert "unknown" not in json.dumps(
        {
            "card_data": first.card_data,
            "deployment": first.card_deployment_values,
            "generation": first.card_generation_values,
        }
    )
    assert first.evidence_hash == second.evidence_hash
    assert first.evidence_hash != local.evidence_hash


def test_remote_card_overlay_is_bounded_without_temporary_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    loader = ModelEvidenceLoader(card_max_chars=64)
    remote_card = "# Remote\n" + "x" * 500

    before = set(model.iterdir())
    result = loader.load_with_card(model, remote_card)

    assert len(result.card_text) == 64
    assert result.warnings == ["Remote model card was truncated to configured limit"]
    assert set(model.iterdir()) == before


def test_malformed_shell_and_later_explicit_values_are_handled_deterministically(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "README.md").write_text(
        "```bash\nvllm serve x --max-model-len 'unterminated\n```\n"
        "```sh\nvllm serve x --max-model-len 4096\n```\n"
        "```shell\npython -m vllm.entrypoints.openai.api_server --max-model-len=8192 "
        "--gpu-memory-utilization 0.7 --quantization modelopt_fp4\n```",
        encoding="utf-8",
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.card_deployment_values == {
        "context_length": 8192,
        "memory_fraction": 0.7,
        "quantization": "modelopt_fp4",
    }
    assert evidence.warnings == ["README.md contains a malformed shell fence"]


def test_front_matter_uses_model_card_metadata_allowlist(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "README.md").write_text(
        "---\n"
        "base_model:\n  - org/Target-8B\n"
        "speculative_method: eagle3\n"
        "unknown_nested:\n  secret: do-not-copy\n"
        "---\n# Draft\n",
        encoding="utf-8",
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.target_model_ids == ["org/Target-8B"]
    assert evidence.speculative_method == "eagle3"
    assert evidence.card_data["base_model"] == ["org/Target-8B"]
    assert "unknown_nested" not in evidence.card_data


def test_target_model_ids_are_trimmed_deduplicated_and_strictly_validated(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "README.md").write_text(
        "---\n"
        "base_model:\n"
        "  - org/model\n"
        '  - " org/model "\n'
        "  - ../etc\n"
        "  - /absolute\n"
        "  - 'org\\model'\n"
        "  - https://host/org/model\n"
        "  - org/model/extra\n"
        "  - org/..\n"
        '  - "org/mo del"\n'
        "target_model: second/valid-model\n"
        "---\n# Draft\n",
        encoding="utf-8",
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.target_model_ids == ["org/model", "second/valid-model"]


def test_malformed_model_card_metadata_is_a_bounded_warning(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "README.md").write_text("---\nbase_model: [\n---\n# Card", encoding="utf-8")

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.card_data == {}
    assert evidence.target_model_ids == []
    assert evidence.warnings == ["README.md contains invalid model card metadata"]


def test_evidence_hash_is_stable_across_directories_and_dictionary_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_text('{"a":1,"b":2}', encoding="utf-8")
    (second / "config.json").write_text('{"b":2,"a":1}', encoding="utf-8")
    for model in (first, second):
        (model / "README.md").write_text("# Same card", encoding="utf-8")

    first_evidence = ModelEvidenceLoader(card_max_chars=100_000).load(first)
    second_evidence = ModelEvidenceLoader(card_max_chars=100_000).load(second)

    assert first_evidence.evidence_hash == second_evidence.evidence_hash


def test_model_directory_must_exist_and_must_not_be_a_symlink(tmp_path):
    loader = ModelEvidenceLoader(card_max_chars=100_000)

    with pytest.raises(ValueError, match="model directory"):
        loader.load(tmp_path / "missing")

    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symlink"):
        loader.load(link)


def test_symlinked_evidence_files_outside_model_are_never_read(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"max_position_embeddings": 1234}', encoding="utf-8")
    try:
        (model / "config.json").symlink_to(outside)
    except OSError:
        pytest.skip("File symlinks are unavailable on this platform")

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.config == {}
    assert evidence.warnings == ["config.json is not a safe regular file"]


def test_unreadable_tokenizer_file_is_reported_without_fingerprint(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    tokenizer = model / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")

    original = model_evidence._read_model_file

    def fail_tokenizer(root, filename, max_bytes):
        if filename == "tokenizer.json":
            return None, "unreadable"
        return original(root, filename, max_bytes)

    monkeypatch.setattr(model_evidence, "_read_model_file", fail_tokenizer)
    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.tokenizer_fingerprint is None
    assert evidence.warnings == ["tokenizer.json could not be read for tokenizer fingerprint"]


def test_generation_config_keeps_dict_but_only_exposes_generation_allowlist(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "generation_config.json").write_text(
        '{"temperature":0.4,"do_sample":true,"nested":{"untrusted":1}}',
        encoding="utf-8",
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.generation_config == {
        "temperature": 0.4,
        "do_sample": True,
        "nested": {"untrusted": 1},
    }
    assert evidence.local_generation_values == {"temperature": 0.4}


def test_missing_tokenizer_files_returns_none(tmp_path):
    assert tokenizer_fingerprint(tmp_path) is None


@pytest.mark.parametrize("card_max_chars", [0, -1, 500_001])
def test_loader_rejects_unbounded_card_limits(card_max_chars):
    with pytest.raises(ValueError, match="card_max_chars"):
        ModelEvidenceLoader(card_max_chars=card_max_chars)
