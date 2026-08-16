import json
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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


def _fake_directory_stat(inode: int, *, reparse: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        st_mode=stat.S_IFDIR,
        st_dev=1,
        st_ino=inode,
        st_file_attributes=0x400 if reparse else 0,
    )


def _posix_snapshot_with_linked_ancestor(
    tmp_path: Path, linked_level: str
) -> tuple[Path, Path]:
    if os.name != "posix":
        pytest.skip("POSIX directory-fd semantics are unavailable")
    visible_hub = tmp_path / "visible-hub"
    visible_hub.mkdir()
    visible_repository = visible_hub / "models--org--model"
    real_repository = tmp_path / "real-hub" / "models--org--model"
    real_snapshot = real_repository / "snapshots" / "revision"
    (real_repository / "blobs").mkdir(parents=True)
    real_snapshot.mkdir(parents=True)
    if linked_level == "repository":
        _symlink_or_skip(visible_repository, real_repository)
    else:
        visible_repository.mkdir()
        _symlink_or_skip(
            visible_repository / "snapshots",
            real_repository / "snapshots",
        )
    return visible_repository / "snapshots" / "revision", real_repository


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


@pytest.mark.parametrize("linked_level", ["repository", "snapshots"])
@pytest.mark.parametrize("filename", ["config.json", "README.md"])
def test_posix_huggingface_snapshot_rejects_linked_ancestors_for_regular_evidence(
    tmp_path, linked_level, filename
):
    snapshot, _repository = _posix_snapshot_with_linked_ancestor(tmp_path, linked_level)
    resolved_snapshot = snapshot.resolve(strict=True)
    payload = (
        '{"hidden_size":4096}'
        if filename == "config.json"
        else "```bash\nvllm serve org/model --max-model-len 8192\n```\n"
    )
    (resolved_snapshot / filename).write_text(payload, encoding="utf-8")

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(snapshot)

    if filename == "config.json":
        assert evidence.config == {}
        assert "config.json is not a safe regular file" in evidence.warnings
    else:
        assert evidence.card_text == ""
        assert "README.md is not a safe regular file" in evidence.warnings


@pytest.mark.parametrize("linked_level", ["repository", "snapshots"])
def test_posix_huggingface_snapshot_rejects_linked_ancestors_for_canonical_blobs(
    tmp_path, linked_level
):
    snapshot, repository = _posix_snapshot_with_linked_ancestor(tmp_path, linked_level)
    digest = "a" * 40
    (repository / "blobs" / digest).write_text('{"hidden_size":4096}', encoding="utf-8")
    _symlink_or_skip(
        snapshot.resolve(strict=True) / "config.json",
        Path("..") / ".." / "blobs" / digest,
    )

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(snapshot)

    assert evidence.config == {}
    assert "config.json is not a safe regular file" in evidence.warnings


def test_huggingface_blob_link_parser_accepts_only_canonical_posix_targets():
    sha1 = "a" * 40
    sha256 = "A1" * 32

    assert model_evidence._parse_huggingface_blob_link_target(f"../../blobs/{sha1}") == sha1
    assert model_evidence._parse_huggingface_blob_link_target(f"../../blobs/{sha256}") == sha256
    assert model_evidence._parse_huggingface_blob_link_target(f"/blobs/{sha1}") is None
    assert model_evidence._parse_huggingface_blob_link_target(f"../../blobs/{sha1}/file") is None
    assert model_evidence._parse_huggingface_blob_link_target("../../blobs/not-a-hash") is None
    assert model_evidence._parse_huggingface_blob_link_target(f"../blobs/{sha1}") is None


def test_huggingface_snapshot_repository_requires_owner_and_model_segments(tmp_path):
    snapshot = tmp_path / "models--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)

    assert model_evidence._huggingface_snapshot_repository(snapshot) is None


@pytest.mark.parametrize("reparse_level", ["repository", "snapshots", "revision"])
def test_windows_snapshot_layout_rejects_every_reparse_level_before_resolve(
    tmp_path, monkeypatch, reparse_level
):
    repository = tmp_path / "models--org--model"
    snapshots = repository / "snapshots"
    revision = snapshots / "revision"
    paths = {
        repository: _fake_directory_stat(1, reparse=reparse_level == "repository"),
        snapshots: _fake_directory_stat(2, reparse=reparse_level == "snapshots"),
        revision: _fake_directory_stat(3, reparse=reparse_level == "revision"),
    }
    resolve_calls: list[Path] = []
    monkeypatch.setattr(model_evidence, "_path_lstat", paths.__getitem__, raising=False)
    monkeypatch.setattr(
        model_evidence,
        "_resolve_strict",
        lambda path: resolve_calls.append(path) or path,
        raising=False,
    )

    layout = model_evidence._windows_huggingface_snapshot_layout(revision)

    assert layout is None
    assert resolve_calls == []


@pytest.mark.parametrize("replaced_level", ["repository", "snapshots", "revision"])
def test_windows_snapshot_layout_detects_path_identity_replacement(
    tmp_path, monkeypatch, replaced_level
):
    repository = tmp_path / "models--org--model"
    snapshots = repository / "snapshots"
    revision = snapshots / "revision"
    paths = {
        "repository": repository,
        "snapshots": snapshots,
        "revision": revision,
    }
    current = {
        repository: _fake_directory_stat(1),
        snapshots: _fake_directory_stat(2),
        revision: _fake_directory_stat(3),
    }
    monkeypatch.setattr(model_evidence, "_path_lstat", current.__getitem__, raising=False)
    monkeypatch.setattr(model_evidence, "_resolve_strict", lambda path: path, raising=False)
    layout = model_evidence._windows_huggingface_snapshot_layout(revision)
    assert layout is not None
    current[paths[replaced_level]] = _fake_directory_stat(99)

    assert model_evidence._windows_snapshot_hierarchy_unchanged(layout) is False


def test_windows_snapshot_layout_converts_resolve_runtime_error_to_rejection(
    tmp_path, monkeypatch
):
    repository = tmp_path / "models--org--model"
    snapshots = repository / "snapshots"
    revision = snapshots / "revision"
    current = {
        repository: _fake_directory_stat(1),
        snapshots: _fake_directory_stat(2),
        revision: _fake_directory_stat(3),
    }
    monkeypatch.setattr(model_evidence, "_path_lstat", current.__getitem__, raising=False)

    def fail_resolve(_path):
        raise RuntimeError("simulated symlink loop")

    monkeypatch.setattr(model_evidence, "_resolve_strict", fail_resolve, raising=False)

    assert model_evidence._windows_huggingface_snapshot_layout(revision) is None


def test_snapshot_repository_converts_resolve_runtime_error_to_rejection(monkeypatch):
    root = Path("models--org--model") / "snapshots" / "revision"

    def fail_resolve(_self, *, strict=False):
        raise RuntimeError("simulated replacement race")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    assert model_evidence._huggingface_snapshot_repository(root) is None


@pytest.mark.skipif(os.name == "posix", reason="Exercises the Windows fallback")
def test_windows_candidate_resolve_runtime_error_becomes_an_unreadable_warning(
    tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size":4096}', encoding="utf-8")
    original_resolve = Path.resolve

    def fail_candidate_resolve(self, *, strict=False):
        if self.name == "config.json":
            raise RuntimeError("simulated replacement race")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_candidate_resolve)

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.config == {}
    assert evidence.warnings == ["config.json is not a safe regular file"]


def test_loader_keeps_the_unresolved_absolute_root_for_evidence_boundary_checks(
    tmp_path, monkeypatch
):
    original = tmp_path / "original" / "models--org--model" / "snapshots" / "revision"
    replacement = (
        tmp_path / "replacement" / "models--org--model" / "snapshots" / "revision"
    )
    original.mkdir(parents=True)
    replacement.mkdir(parents=True)
    original_resolve = Path.resolve

    def replace_model_root(self, *, strict=False):
        if self == original:
            return replacement
        return original_resolve(self, strict=strict)

    evidence_roots: list[Path] = []

    def read_json(root, _filename, _warnings):
        evidence_roots.append(root)
        return {}

    def read_card(root, *_args):
        evidence_roots.append(root)
        return ""

    def fingerprint(root):
        evidence_roots.append(root)
        return None, []

    monkeypatch.setattr(Path, "resolve", replace_model_root)
    monkeypatch.setattr(model_evidence, "_read_json_dict", read_json)
    monkeypatch.setattr(model_evidence, "_read_card", read_card)
    monkeypatch.setattr(model_evidence, "_tokenizer_fingerprint", fingerprint)

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(original)

    assert evidence.model_path == str(replacement)
    assert evidence_roots == [original.absolute()] * 4


def test_windows_blob_resolve_runtime_error_is_converted_to_value_error(
    tmp_path, monkeypatch
):
    repository, snapshot = _hub_snapshot(tmp_path)
    digest = "a" * 40
    blob = repository / "blobs" / digest
    blob.write_text('{"hidden_size":4096}', encoding="utf-8")
    candidate = snapshot / "config.json"
    candidate.write_text("placeholder", encoding="utf-8")
    candidate_stat = os.stat(candidate, follow_symlinks=False)
    fake_link_stat = SimpleNamespace(
        st_mode=stat.S_IFLNK,
        st_dev=candidate_stat.st_dev,
        st_ino=candidate_stat.st_ino,
        st_file_attributes=0,
    )
    original_stat = os.stat
    original_readlink = os.readlink
    original_resolve = Path.resolve

    def fake_stat(path, *args, **kwargs):
        if Path(path) == candidate:
            return fake_link_stat
        return original_stat(path, *args, **kwargs)

    def fake_readlink(path, *args, **kwargs):
        if Path(path) == candidate:
            return f"../../blobs/{digest}"
        return original_readlink(path, *args, **kwargs)

    def fail_blob_resolve(self, *, strict=False):
        if self == blob:
            raise RuntimeError("simulated blob replacement race")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(os, "stat", fake_stat)
    monkeypatch.setattr(os, "readlink", fake_readlink)
    monkeypatch.setattr(Path, "resolve", fail_blob_resolve)

    with pytest.raises(ValueError, match="resolved safely"):
        with model_evidence._open_windows_huggingface_blob(
            snapshot, candidate, fake_link_stat
        ):
            pass


def test_read_model_file_converts_value_error_to_unreadable(tmp_path, monkeypatch):
    @contextmanager
    def fail_open(*_args, **_kwargs):
        raise ValueError("simulated safe path rejection")
        yield

    monkeypatch.setattr(model_evidence, "_open_model_file", fail_open)

    assert model_evidence._read_model_file(tmp_path, "config.json", 1024) == (
        None,
        "unreadable",
    )


def test_safe_evidence_resolve_does_not_swallow_programming_errors(tmp_path, monkeypatch):
    target = tmp_path / "config.json"

    def fail_runtime(_path):
        raise RuntimeError("simulated symlink loop")

    monkeypatch.setattr(model_evidence, "_resolve_strict", fail_runtime)
    with pytest.raises(ValueError, match="resolved safely"):
        model_evidence._resolve_evidence_path(target)

    def fail_type(_path):
        raise TypeError("programming error")

    monkeypatch.setattr(model_evidence, "_resolve_strict", fail_type)
    with pytest.raises(TypeError, match="programming error"):
        model_evidence._resolve_evidence_path(target)


def test_loader_converts_model_root_resolve_runtime_error_to_value_error(
    tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()

    def fail_resolve(_self, *, strict=False):
        raise RuntimeError("simulated model root replacement race")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    with pytest.raises(ValueError, match="model directory could not be resolved safely"):
        ModelEvidenceLoader(card_max_chars=100_000).load(model)


@pytest.mark.skipif(os.name == "posix", reason="Exercises the Windows fallback")
@pytest.mark.parametrize("reparse_level", ["repository", "snapshots", "revision"])
def test_windows_model_file_open_applies_snapshot_reparse_checks_to_regular_files(
    tmp_path, monkeypatch, reparse_level
):
    repository, snapshot = _hub_snapshot(tmp_path)
    snapshots = repository / "snapshots"
    candidate = snapshot / "config.json"
    candidate.write_text('{"hidden_size":4096}', encoding="utf-8")
    paths = {
        "repository": repository,
        "snapshots": snapshots,
        "revision": snapshot,
    }
    original_lstat = model_evidence._path_lstat

    def fake_lstat(path):
        value = original_lstat(path)
        if path == paths[reparse_level]:
            return SimpleNamespace(
                st_mode=value.st_mode,
                st_dev=value.st_dev,
                st_ino=value.st_ino,
                st_file_attributes=0x400,
            )
        return value

    monkeypatch.setattr(model_evidence, "_path_lstat", fake_lstat)

    with pytest.raises(ValueError, match="snapshot hierarchy|model root"):
        with model_evidence._open_model_file(snapshot, "config.json"):
            pass


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


def test_invalid_oversized_and_non_dictionary_json_is_bounded(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(model_evidence, "MAX_JSON_BYTES", 4)
    (model / "config.json").write_bytes(b"12345")
    (model / "generation_config.json").write_text("[]", encoding="utf-8")

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.config == {}
    assert evidence.generation_config == {}
    assert evidence.local_generation_values == {}
    assert evidence.warnings == [
        "config.json exceeds the size limit",
        "generation_config.json must contain a JSON object",
    ]
    assert len(json.dumps(evidence.warnings)) < 500


def test_large_nemotron_config_within_four_mib_is_read(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    payload = json.dumps(
        {
            "max_position_embeddings": 1_048_576,
            "padding": "x" * 1_337_600,
        }
    )
    assert 1_300_000 < len(payload.encode("utf-8")) < 1_400_000
    (model / "config.json").write_text(payload, encoding="utf-8")

    evidence = ModelEvidenceLoader(card_max_chars=100_000).load(model)

    assert evidence.config["max_position_embeddings"] == 1_048_576
    assert evidence.warnings == []


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
