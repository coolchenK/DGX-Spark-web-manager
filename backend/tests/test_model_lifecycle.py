from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.db import Database
from app.main import create_app
from app.models import AuditEvent, Deployment, ModelAsset, TaskRecord
from app.services.model_lifecycle import (
    ModelInUseError,
    ModelLifecycleService,
    ModelReference,
    _resolved,
)
from app.tasks.engine import TaskCancelled, TaskPaused
from fastapi.testclient import TestClient
from sqlalchemy import select


@pytest.fixture
def database(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'lifecycle.db'}")
    database.create_schema()
    yield database
    database.dispose()


@pytest.fixture
def service(tmp_path):
    return ModelLifecycleService(
        model_roots=(tmp_path / "models",),
        hf_cache_dir=tmp_path / "hf-cache",
    )


def _asset(*, model_id: str = "target", local_path: str) -> ModelAsset:
    return ModelAsset(
        id=model_id,
        name=model_id,
        source="local",
        local_path=local_path,
        status="available",
    )


class HandlerContext:
    def __init__(self, error: Exception | None = None, task_id: str = "task-1"):
        self.error = error
        self.task_id = task_id
        self.checks = 0

    def check_control(self):
        self.checks += 1
        if self.error is not None:
            raise self.error


def _test_local_remover(_root: Path, target: Path) -> None:
    shutil.rmtree(target)


def _deletion_service(
    database,
    tmp_path,
    *,
    command_runner=None,
    discovery_service=None,
    local_remover=None,
) -> ModelLifecycleService:
    if local_remover is None:
        local_remover = _test_local_remover
    return ModelLifecycleService(
        model_roots=(tmp_path / "models",),
        hf_cache_dir=tmp_path / "hf-cache",
        session_factory=database.session_factory,
        command_runner=command_runner,
        discovery_service=discovery_service,
        local_remover=local_remover,
    )


def _add_asset(
    database,
    path: Path,
    *,
    model_id: str = "target",
    name: str = "target",
    source: str = "local",
    repository_id: str | None = None,
    status: str = "available",
    metadata_json: dict | None = None,
):
    with database.session_factory() as db:
        db.add(
            ModelAsset(
                id=model_id,
                name=name,
                source=source,
                repository_id=repository_id,
                local_path=str(path),
                status=status,
                size_bytes=123,
                metadata_json={} if metadata_json is None else metadata_json,
            )
        )
        db.commit()


@pytest.fixture
def stopped_task_client(authenticated_client):
    authenticated_client.app.state.task_engine.stop()
    return authenticated_client


def _delete_model(client, model_id: str, confirmation: str):
    return client.request(
        "DELETE",
        f"/api/models/{model_id}",
        json={"confirmation": confirmation},
    )


def _model_status(database, model_id: str = "target") -> str | None:
    with database.session_factory() as db:
        asset = db.get(ModelAsset, model_id)
        return None if asset is None else asset.status


def _model_metadata(database, model_id: str = "target") -> dict | None:
    with database.session_factory() as db:
        asset = db.get(ModelAsset, model_id)
        return None if asset is None else asset.metadata_json


def _deployment(
    *,
    deployment_id: str,
    name: str,
    model_id: str | None = None,
    config: object = None,
    status: str = "running",
    health: str = "healthy",
) -> Deployment:
    return Deployment(
        id=deployment_id,
        name=name,
        model_id=model_id,
        runtime="vllm",
        endpoint_url=f"http://127.0.0.1/{deployment_id}",
        api_model_name=f"api-{deployment_id}",
        status=status,
        health=health,
        config={} if config is None else config,
    )


def test_model_reference_is_frozen():
    reference = ModelReference(
        deployment_id="deployment-1",
        deployment_name="deployment",
        usage="base",
    )

    with pytest.raises(FrozenInstanceError):
        reference.usage = "draft"  # type: ignore[misc]


def test_references_include_base_and_draft_in_name_order_regardless_of_state(
    database, service, tmp_path
):
    with database.session_factory() as db:
        db.add(_asset(local_path=str(tmp_path / "target")))
        db.add_all(
            [
                _deployment(
                    deployment_id="base-id",
                    name="z-base",
                    model_id="target",
                    status="stopped",
                    health="unhealthy",
                ),
                _deployment(
                    deployment_id="draft-id",
                    name="a-draft",
                    config={"speculative": {"draft_model_id": "target"}},
                    status="error",
                    health="unhealthy",
                ),
            ]
        )
        db.commit()

        references = service.references(db, "target")

    assert references == [
        ModelReference("draft-id", "a-draft", "draft"),
        ModelReference("base-id", "z-base", "base"),
    ]


def test_references_match_legacy_base_and_draft_paths(database, service, tmp_path):
    target_path = tmp_path / "models" / "target"
    equivalent_path = target_path.parent / "other" / ".." / target_path.name

    with database.session_factory() as db:
        db.add(_asset(local_path=str(target_path)))
        db.add_all(
            [
                _deployment(
                    deployment_id="legacy-base",
                    name="legacy-base",
                    config={"model_path": str(equivalent_path)},
                ),
                _deployment(
                    deployment_id="legacy-draft",
                    name="legacy-draft",
                    config={
                        "speculative": {"draft_model_path": str(equivalent_path)}
                    },
                ),
            ]
        )
        db.commit()

        references = service.references(db, "target")

    assert references == [
        ModelReference("legacy-base", "legacy-base", "legacy_path"),
        ModelReference("legacy-draft", "legacy-draft", "legacy_path"),
    ]


def test_references_raise_when_model_does_not_exist(database, service):
    with database.session_factory() as db:
        with pytest.raises(LookupError, match="^Model was not found$"):
            service.references(db, "missing")


def test_references_ignore_non_dict_malformed_and_invalid_path_configs(
    database, service, tmp_path
):
    with database.session_factory() as db:
        db.add(_asset(local_path=str(tmp_path / "target")))
        db.add_all(
            [
                _deployment(deployment_id="list", name="list", config=[]),
                _deployment(
                    deployment_id="bad-speculative",
                    name="bad-speculative",
                    config={"speculative": "not-a-dict"},
                ),
                _deployment(
                    deployment_id="bad-draft-path",
                    name="bad-draft-path",
                    config={"speculative": {"draft_model_path": 42}},
                ),
                _deployment(
                    deployment_id="invalid-path",
                    name="invalid-path",
                    config={"model_path": "\0invalid"},
                ),
            ]
        )
        db.commit()

        assert service.references(db, "target") == []


def test_references_return_each_deployment_once_using_highest_priority(
    database, service, tmp_path
):
    target_path = tmp_path / "target"
    with database.session_factory() as db:
        db.add(_asset(local_path=str(target_path)))
        db.add_all(
            [
                _deployment(
                    deployment_id="all-usages",
                    name="all-usages",
                    model_id="target",
                    config={
                        "model_path": str(target_path),
                        "speculative": {
                            "draft_model_id": "target",
                            "draft_model_path": str(target_path),
                        },
                    },
                ),
                _deployment(
                    deployment_id="draft-and-path",
                    name="draft-and-path",
                    config={
                        "model_path": str(target_path),
                        "speculative": {"draft_model_id": "target"},
                    },
                ),
            ]
        )
        db.commit()

        references = service.references(db, "target")

    assert references == [
        ModelReference("all-usages", "all-usages", "base"),
        ModelReference("draft-and-path", "draft-and-path", "draft"),
    ]


def test_references_do_not_treat_path_prefix_as_a_match(database, service, tmp_path):
    target_path = tmp_path / "models" / "target"
    with database.session_factory() as db:
        db.add(_asset(local_path=str(target_path)))
        db.add(
            _deployment(
                deployment_id="prefix",
                name="prefix",
                config={"model_path": f"{target_path}-copy"},
            )
        )
        db.commit()

        assert service.references(db, "target") == []


def test_references_match_persisted_draft_mount_when_draft_id_is_stale(
    database, service, tmp_path
):
    target_path = tmp_path / "models" / "draft"
    with database.session_factory() as db:
        db.add(_asset(local_path=str(target_path)))
        db.add(
            _deployment(
                deployment_id="persisted-draft",
                name="persisted-draft",
                config={
                    "speculative": {"draft_model_id": "stale-id"},
                    "mounts": {"draft": {"model_path": str(target_path)}},
                },
            )
        )
        db.commit()

        references = service.references(db, "target")

    assert references == [
        ModelReference("persisted-draft", "persisted-draft", "legacy_path")
    ]


@pytest.mark.parametrize("path_location", ["spec", "base_mount"])
def test_references_match_persisted_base_paths(
    database, service, tmp_path, path_location
):
    target_path = tmp_path / "models" / "base"
    config = (
        {"spec": {"model_path": str(target_path)}}
        if path_location == "spec"
        else {"mounts": {"base": {"model_path": str(target_path)}}}
    )

    with database.session_factory() as db:
        db.add(_asset(local_path=str(target_path)))
        db.add(
            _deployment(
                deployment_id=f"persisted-{path_location}",
                name=f"persisted-{path_location}",
                config=config,
            )
        )
        db.commit()

        references = service.references(db, "target")

    assert references == [
        ModelReference(
            f"persisted-{path_location}",
            f"persisted-{path_location}",
            "legacy_path",
        )
    ]


@pytest.mark.parametrize("value", ["", "   ", "\t", None, 42])
def test_resolved_rejects_empty_blank_and_non_string_values(value):
    assert _resolved(value) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        {"spec": []},
        {"mounts": "not-a-dict"},
        {"mounts": {"base": [], "draft": "not-a-dict"}},
        {
            "spec": {"model_path": []},
            "mounts": {
                "base": {"model_path": 42},
                "draft": {"model_path": {}},
            },
        },
    ],
)
def test_references_ignore_malformed_persisted_path_structures(
    database, service, tmp_path, config
):
    with database.session_factory() as db:
        db.add(_asset(local_path=str(tmp_path / "target")))
        db.add(
            _deployment(
                deployment_id="malformed",
                name="malformed",
                config=config,
            )
        )
        db.commit()

        assert service.references(db, "target") == []


def test_validate_local_target_accepts_only_a_child_directory(tmp_path):
    root = tmp_path / "models"
    target = root / "target"
    target.mkdir(parents=True)
    service = ModelLifecycleService((root,), tmp_path / "hf-cache")

    assert service.validate_local_target(target) == target.resolve()

    with pytest.raises(ValueError, match="configured model root"):
        service.validate_local_target(root)


def test_validate_local_target_rejects_path_prefix_sibling(tmp_path):
    root = tmp_path / "models"
    sibling = tmp_path / "models-copy" / "target"
    root.mkdir()
    sibling.mkdir(parents=True)
    service = ModelLifecycleService((root,), tmp_path / "hf-cache")

    with pytest.raises(ValueError, match="configured model root"):
        service.validate_local_target(sibling)


@pytest.mark.parametrize("reverse_roots", [False, True])
def test_validate_local_target_rejects_target_equal_to_any_nested_root(
    tmp_path, reverse_roots
):
    parent = tmp_path / "models"
    child = parent / "nested"
    child.mkdir(parents=True)
    roots = (parent, child)
    if reverse_roots:
        roots = tuple(reversed(roots))
    service = ModelLifecycleService(roots, tmp_path / "hf-cache")

    with pytest.raises(ValueError, match="configured model root"):
        service.validate_local_target(child)


@pytest.mark.parametrize("nested_exists", [False, True])
@pytest.mark.parametrize("reverse_roots", [False, True])
def test_validate_local_target_rejects_parent_of_nested_root(
    tmp_path, nested_exists, reverse_roots
):
    outer_root = tmp_path / "models"
    target = outer_root / "target"
    nested_root = target / "nested-root"
    target.mkdir(parents=True)
    if nested_exists:
        nested_root.mkdir()
    roots = (outer_root, nested_root)
    if reverse_roots:
        roots = tuple(reversed(roots))
    service = ModelLifecycleService(roots, tmp_path / "hf-cache")

    with pytest.raises(ValueError, match="configured model root"):
        service.validate_local_target(target)


@pytest.mark.parametrize("resolved_relation", ["equal", "inside"])
def test_validate_local_target_rejects_unrelated_reparse_root_resolved_inside_target(
    tmp_path, monkeypatch, resolved_relation
):
    outer_root = tmp_path / "models"
    target = outer_root / "target"
    target.mkdir(parents=True)
    resolved_alias = target
    if resolved_relation == "inside":
        resolved_alias = target / "nested-root"
        resolved_alias.mkdir()
    alias_root = tmp_path / "alias-root"
    alias_root.mkdir()
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == alias_root or original_is_symlink(self),
    )
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, strict=False: (
            resolved_alias
            if self == alias_root
            else original_resolve(self, strict=strict)
        ),
    )
    service = ModelLifecycleService(
        (outer_root, alias_root), tmp_path / "hf-cache"
    )

    with pytest.raises(ValueError, match="configured model root"):
        service.validate_local_target(target)


def test_local_delete_ignores_unrelated_missing_model_root(database, tmp_path):
    valid_root = tmp_path / "models"
    target = valid_root / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)
    service = ModelLifecycleService(
        model_roots=(tmp_path / "missing-root", valid_root),
        hf_cache_dir=tmp_path / "hf-cache",
        session_factory=database.session_factory,
        local_remover=_test_local_remover,
    )

    assert service.validate_local_target(target) == target.resolve()
    service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert not target.exists()
    assert _model_status(database) is None


def test_local_delete_ignores_unrelated_reparse_root(database, tmp_path, monkeypatch):
    valid_root = tmp_path / "models"
    target = valid_root / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)
    unrelated_root = tmp_path / "unrelated"
    unrelated_root.mkdir()
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == unrelated_root or original_is_symlink(self),
    )
    service = ModelLifecycleService(
        (unrelated_root, valid_root),
        tmp_path / "hf-cache",
        session_factory=database.session_factory,
        local_remover=_test_local_remover,
    )

    assert service.validate_local_target(target) == target.resolve()
    service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert not target.exists()
    assert _model_status(database) is None


@pytest.mark.parametrize("destination", ["inside", "outside"])
def test_validate_local_target_rejects_symlinks(tmp_path, destination, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    real_target = (root / "real") if destination == "inside" else (tmp_path / "outside")
    real_target.mkdir()
    link = root / "link"
    try:
        link.symlink_to(real_target, target_is_directory=True)
    except OSError:
        link.mkdir()
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda self: self == link or original_is_symlink(self),
        )
    service = ModelLifecycleService((root,), tmp_path / "hf-cache")

    with pytest.raises(ValueError, match="symbolic link"):
        service.validate_local_target(link)


@pytest.mark.parametrize("destination", ["inside", "outside"])
def test_validate_local_target_rejects_intermediate_symlink(
    tmp_path, destination, monkeypatch
):
    root = tmp_path / "models"
    real_parent = (root / "real") if destination == "inside" else (tmp_path / "outside")
    target = real_parent / "target"
    target.mkdir(parents=True)
    link = root / "link"
    try:
        link.symlink_to(real_parent, target_is_directory=True)
        linked_target = link / "target"
    except OSError:
        linked_target = root / "link" / "target"
        linked_target.mkdir(parents=True)
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda self: self == link or original_is_symlink(self),
        )
    service = ModelLifecycleService((root,), tmp_path / "hf-cache")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        service.validate_local_target(linked_target)


def test_validate_local_target_rejects_symlink_model_root(tmp_path, monkeypatch):
    real_root = tmp_path / "real-models"
    target = real_root / "target"
    target.mkdir(parents=True)
    root = tmp_path / "models"
    try:
        root.symlink_to(real_root, target_is_directory=True)
        linked_target = root / "target"
    except OSError:
        linked_target = root / "target"
        linked_target.mkdir(parents=True)
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda self: self == root or original_is_symlink(self),
        )
    service = ModelLifecycleService((root,), tmp_path / "hf-cache")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        service.validate_local_target(linked_target)


def test_validate_local_target_rejects_junction_component_when_supported(
    tmp_path, monkeypatch
):
    if not hasattr(Path, "is_junction"):
        pytest.skip("Path.is_junction is unavailable")
    root = tmp_path / "models"
    junction = root / "junction"
    target = junction / "target"
    target.mkdir(parents=True)
    original_is_junction = Path.is_junction
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == junction or original_is_junction(self),
    )
    service = ModelLifecycleService((root,), tmp_path / "hf-cache")

    with pytest.raises(ValueError, match="reparse point"):
        service.validate_local_target(target)


@pytest.mark.skipif(os.name != "posix", reason="fd-safe deletion is POSIX-specific")
def test_secure_rmtree_does_not_follow_replaced_intermediate_component(
    tmp_path, monkeypatch
):
    root = tmp_path / "models"
    original_parent = root / "parent"
    target = original_parent / "target"
    target.mkdir(parents=True)
    (target / "weights").write_bytes(b"model")
    outside = tmp_path / "outside"
    outside_target = outside / "target"
    outside_target.mkdir(parents=True)
    (outside_target / "keep").write_bytes(b"outside")
    moved_parent = root / "opened-parent"
    original_open = os.open
    replaced = False

    def replacing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "parent" and dir_fd is not None and not replaced:
            replaced = True
            original_parent.rename(moved_parent)
            original_parent.symlink_to(outside, target_is_directory=True)
        return fd

    monkeypatch.setattr(os, "open", replacing_open)

    ModelLifecycleService._secure_rmtree(root, target)

    assert not (moved_parent / "target").exists()
    assert (outside_target / "keep").read_bytes() == b"outside"


def test_delete_local_directory_and_database_record(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    (target / "weights.bin").write_bytes(b"weights")
    _add_asset(database, target)
    service = _deletion_service(database, tmp_path)

    result = service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert not target.exists()
    assert _model_status(database) is None
    assert result["model_id"] == "target"
    assert result["source"] == "local"
    assert result["estimated_bytes"] == len(b"weights")
    assert result["released_bytes"] >= 0
    assert result["inventory_models"] is None


def test_delete_huggingface_runs_validated_dry_run_before_yes(database, tmp_path):
    cache_dir = tmp_path / "hf-cache"
    repository_root = cache_dir / "models--org--model"
    snapshot = repository_root / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (repository_root / "blob").write_bytes(b"cached")
    _add_asset(
        database,
        snapshot,
        source="huggingface",
        repository_id="org/model",
    )
    calls = []

    def runner(argv):
        calls.append(argv)
        if "--yes" in argv:
            for child in sorted(repository_root.rglob("*"), reverse=True):
                child.unlink() if child.is_file() else child.rmdir()
            repository_root.rmdir()
            payload = {
                "repos_deleted": 1,
                "revisions_deleted": 1,
                "freed": "7.0",
            }
        else:
            payload = {
                "dry_run": True,
                "repos": 1,
                "revisions": 1,
                "size": "7.0",
            }
        return SimpleNamespace(stdout=json.dumps(payload))

    service = _deletion_service(database, tmp_path, command_runner=runner)

    result = service.delete_handler(HandlerContext(), {"model_id": "target"})

    base = [
        "hf",
        "cache",
        "rm",
        "model/org/model",
        "--cache-dir",
        str(cache_dir),
    ]
    assert calls == [
        [*base, "--dry-run", "--json"],
        [*base, "--yes", "--json"],
    ]
    assert result["estimated_bytes"] == len(b"cached")
    assert _model_status(database) is None


@pytest.mark.parametrize(
    ("validator", "payload"),
    [
        (
            lambda payload: ModelLifecycleService._validate_hf_preview(
                payload, "model/org/model"
            ),
            {"dry_run": True, "repos": 1, "revisions": 1.5},
        ),
        (
            ModelLifecycleService._validate_hf_result,
            {"repos_deleted": 1, "revisions_deleted": 1.5},
        ),
    ],
)
def test_huggingface_revision_counts_must_be_nonnegative_integers(
    validator, payload
):
    with pytest.raises(RuntimeError, match="invalid"):
        validator(payload)


def test_huggingface_mismatched_dry_run_stops_before_delete(database, tmp_path):
    repository_root = tmp_path / "hf-cache" / "models--org--model"
    snapshot = repository_root / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    _add_asset(
        database,
        snapshot,
        source="huggingface",
        repository_id="org/model",
    )
    calls = []

    def runner(argv):
        calls.append(argv)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "dry_run": True,
                    "repos": 1,
                    "targets": ["model/org/other"],
                }
            )
        )

    service = _deletion_service(database, tmp_path, command_runner=runner)

    with pytest.raises(RuntimeError, match="preview did not identify"):
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert len(calls) == 1
    assert "--dry-run" in calls[0]
    assert repository_root.exists()
    assert _model_status(database) == "delete_failed"


def test_huggingface_dry_run_rejects_target_plus_extra_target(database, tmp_path):
    repository_root = tmp_path / "hf-cache" / "models--org--model"
    snapshot = repository_root / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    _add_asset(
        database,
        snapshot,
        source="huggingface",
        repository_id="org/model",
    )
    calls = []

    def runner(argv):
        calls.append(argv)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "dry_run": True,
                    "repos": 1,
                    "targets": ["model/org/model", "model/org/other"],
                }
            )
        )

    service = _deletion_service(database, tmp_path, command_runner=runner)

    with pytest.raises(RuntimeError, match="preview did not identify only the target"):
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert len(calls) == 1
    assert _model_status(database) == "delete_failed"


def test_huggingface_malformed_yes_json_reconciles_deleted_repository(
    database, tmp_path, monkeypatch
):
    repository_root = tmp_path / "hf-cache" / "models--org--model"
    snapshot = repository_root / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    _add_asset(
        database,
        snapshot,
        source="huggingface",
        repository_id="org/model",
    )
    calls = []
    disk_free = iter((100, 125))
    monkeypatch.setattr(
        "app.services.model_lifecycle.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=next(disk_free)),
    )

    def runner(argv):
        calls.append(argv)
        if "--yes" in argv:
            shutil.rmtree(repository_root)
            stdout = "not-json"
        else:
            stdout = json.dumps({"dry_run": True, "repos": 1})
        return SimpleNamespace(stdout=stdout)

    service = _deletion_service(database, tmp_path, command_runner=runner)

    result = service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert len(calls) == 2
    assert _model_status(database) is None
    assert result["released_bytes"] == 25
    assert result["warnings"] == ["Model files were removed before completion"]


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (
            lambda _argv: (_ for _ in ()).throw(
                subprocess.CalledProcessError(1, ["hf"], output="secret", stderr="token")
            ),
            "Hugging Face cache command failed",
        ),
        (
            lambda _argv: SimpleNamespace(
                returncode=1,
                stdout='{"repo_id":"org/model"}',
                stderr="token",
            ),
            "Hugging Face cache command failed",
        ),
        (
            lambda _argv: (_ for _ in ()).throw(FileNotFoundError("hf")),
            "Hugging Face cache command failed",
        ),
        (
            lambda argv: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(argv, timeout=120)
            ),
            "Hugging Face cache command failed",
        ),
        (lambda _argv: SimpleNamespace(stdout="not-json"), "invalid JSON"),
    ],
)
def test_huggingface_command_failure_preserves_record_and_marks_failed(
    database, tmp_path, runner, message
):
    repository_root = tmp_path / "hf-cache" / "models--org--model"
    snapshot = repository_root / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    _add_asset(
        database,
        snapshot,
        source="huggingface",
        repository_id="org/model",
    )
    service = _deletion_service(database, tmp_path, command_runner=runner)

    with pytest.raises(RuntimeError, match=message) as caught:
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert "secret" not in str(caught.value)
    assert "token" not in str(caught.value)
    assert _model_status(database) == "delete_failed"


@pytest.mark.parametrize("source", ["local", "huggingface"])
def test_unavailable_missing_asset_only_removes_database_record(
    database, tmp_path, source
):
    repository_id = "org/model" if source == "huggingface" else None
    missing = tmp_path / "missing"
    _add_asset(
        database,
        missing,
        source=source,
        repository_id=repository_id,
        status="unavailable",
    )
    calls = []
    service = _deletion_service(
        database,
        tmp_path,
        command_runner=lambda argv: calls.append(argv),
    )

    result = service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert calls == []
    assert _model_status(database) is None
    assert result["estimated_bytes"] == 0


def test_unavailable_negative_only_hf_cache_is_removed_without_cli(database, tmp_path):
    repository_root = tmp_path / "hf-cache" / "models--org--model"
    markers = repository_root / ".no_exist" / "commit123"
    markers.mkdir(parents=True)
    (markers / "missing.json").write_bytes(b"")
    _add_asset(
        database,
        repository_root,
        source="huggingface",
        repository_id="org/model",
        status="unavailable",
    )
    calls = []
    service = _deletion_service(
        database,
        tmp_path,
        command_runner=lambda argv: calls.append(argv),
    )

    result = service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert calls == []
    assert not repository_root.exists()
    assert _model_status(database) is None
    assert result["source"] == "huggingface"
    assert result["estimated_bytes"] == 0


@pytest.mark.parametrize(
    "unsafe_entry",
    ["nonzero_marker", "cached_content", "linked_marker"],
)
def test_unavailable_hf_cache_with_content_does_not_bypass_cli(
    database, tmp_path, unsafe_entry
):
    repository_root = tmp_path / "hf-cache" / "models--org--model"
    markers = repository_root / ".no_exist" / "commit123"
    markers.mkdir(parents=True)
    if unsafe_entry == "nonzero_marker":
        (markers / "missing.json").write_bytes(b"unexpected")
    elif unsafe_entry == "cached_content":
        (repository_root / "blobs").mkdir()
    else:
        outside = tmp_path / "outside"
        outside.write_bytes(b"")
        try:
            (markers / "missing.json").symlink_to(outside)
        except OSError:
            pytest.skip("Symlink creation is unavailable")
    _add_asset(
        database,
        repository_root,
        source="huggingface",
        repository_id="org/model",
        status="unavailable",
    )
    calls = []

    def runner(argv):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    service = _deletion_service(database, tmp_path, command_runner=runner)

    with pytest.raises(RuntimeError, match="cache command failed"):
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert len(calls) == 1
    assert repository_root.exists()
    assert _model_status(database) == "delete_failed"


def test_delete_rechecks_references_at_execution_time(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)
    with database.session_factory() as db:
        db.add(_deployment(deployment_id="live", name="live", model_id="target"))
        db.commit()
    service = _deletion_service(database, tmp_path)

    with pytest.raises(ModelInUseError) as caught:
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert caught.value.references == [ModelReference("live", "live", "base")]
    assert target.exists()
    assert _model_status(database) == "available"


@pytest.mark.parametrize("usage", ["base", "draft", "legacy_path"])
def test_same_owner_reentry_with_new_reference_restores_original_state(
    database, tmp_path, usage
):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(
        database,
        target,
        status="deleting",
        metadata_json={
            "keep": "value",
            "_delete_task_id": "task-1",
            "_delete_original_status": "available",
        },
    )
    deployment_values = {
        "model_id": "target" if usage == "base" else None,
        "config": (
            {"speculative": {"draft_model_id": "target"}}
            if usage == "draft"
            else ({"model_path": str(target)} if usage == "legacy_path" else {})
        ),
    }
    with database.session_factory() as db:
        db.add(
            _deployment(
                deployment_id=f"ref-{usage}",
                name=f"ref-{usage}",
                **deployment_values,
            )
        )
        db.commit()
    service = _deletion_service(database, tmp_path)

    with pytest.raises(ModelInUseError):
        service.delete_handler(HandlerContext(task_id="task-1"), {"model_id": "target"})

    assert _model_status(database) == "available"
    assert _model_metadata(database) == {"keep": "value"}
    with pytest.raises(ModelInUseError):
        service.delete_handler(HandlerContext(task_id="task-2"), {"model_id": "target"})


@pytest.mark.parametrize("control_error", [TaskCancelled(), TaskPaused()])
def test_control_stop_restores_original_state_before_destructive_action(
    database, tmp_path, control_error
):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target, metadata_json={"keep": "value"})
    service = _deletion_service(database, tmp_path)

    with pytest.raises(type(control_error)):
        service.delete_handler(HandlerContext(control_error), {"model_id": "target"})

    assert target.exists()
    assert _model_status(database) == "available"
    assert _model_metadata(database) == {"keep": "value"}


def test_local_delete_revalidates_target_after_control_check(
    database, tmp_path, monkeypatch
):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)
    service = _deletion_service(database, tmp_path)
    replaced = False
    original_is_symlink = Path.is_symlink

    class ReplacingContext(HandlerContext):
        def check_control(self):
            nonlocal replaced
            replaced = True

    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: (self == target and replaced) or original_is_symlink(self),
    )
    deleted = []
    monkeypatch.setattr(
        "app.services.model_lifecycle.shutil.rmtree", lambda path: deleted.append(path)
    )

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        service.delete_handler(ReplacingContext(), {"model_id": "target"})

    assert deleted == []
    assert _model_status(database) == "delete_failed"


def test_local_delete_failure_marks_record_delete_failed(database, tmp_path, monkeypatch):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)
    def fail_delete(_root, _path):
        raise OSError("private filesystem detail")

    service = _deletion_service(database, tmp_path, local_remover=fail_delete)
    with pytest.raises(RuntimeError, match="Local model deletion failed") as caught:
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert "private filesystem detail" not in str(caught.value)
    assert _model_status(database) == "delete_failed"


def test_successful_delete_scans_inventory_in_new_session(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)

    class Discovery:
        def __init__(self):
            self.sessions = []

        def scan_models(self, db):
            self.sessions.append(db)
            return [object(), object()]

    discovery = Discovery()
    service = _deletion_service(database, tmp_path, discovery_service=discovery)

    result = service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert result["inventory_models"] == 2
    assert len(discovery.sessions) == 1
    assert _model_status(database) is None


def test_discovery_failure_does_not_reverse_successful_deletion(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)

    class Discovery:
        def scan_models(self, _db):
            raise RuntimeError("scan failed")

    service = _deletion_service(database, tmp_path, discovery_service=Discovery())

    result = service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert _model_status(database) is None
    assert result["inventory_models"] is None
    assert result["inventory_warning"] == "Inventory refresh failed"


def test_same_task_restarts_deletion_before_physical_action(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(
        database,
        target,
        status="deleting",
        metadata_json={
            "keep": "value",
            "_delete_task_id": "task-1",
            "_delete_original_status": "available",
        },
    )
    service = _deletion_service(database, tmp_path)

    result = service.delete_handler(HandlerContext(task_id="task-1"), {"model_id": "target"})

    assert not target.exists()
    assert _model_status(database) is None
    assert result["model_id"] == "target"


def test_same_task_reconciles_restart_after_physical_deletion(database, tmp_path):
    target = tmp_path / "models" / "target"
    _add_asset(
        database,
        target,
        status="deleting",
        metadata_json={
            "_delete_task_id": "task-1",
            "_delete_original_status": "available",
        },
    )
    calls = []
    service = _deletion_service(
        database,
        tmp_path,
        local_remover=lambda root, path: calls.append((root, path)),
    )

    result = service.delete_handler(HandlerContext(task_id="task-1"), {"model_id": "target"})

    assert calls == []
    assert _model_status(database) is None
    assert result["warnings"] == []


def test_different_task_cannot_resume_owned_deletion(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(
        database,
        target,
        status="deleting",
        metadata_json={
            "_delete_task_id": "old-task",
            "_delete_original_status": "available",
        },
    )
    service = _deletion_service(database, tmp_path)

    with pytest.raises(ValueError, match="another task"):
        service.delete_handler(HandlerContext(task_id="new-task"), {"model_id": "target"})

    assert target.exists()
    assert _model_status(database) == "deleting"


def test_delete_failed_missing_target_is_reconciled(database, tmp_path):
    target = tmp_path / "models" / "target"
    _add_asset(
        database,
        target,
        status="delete_failed",
        metadata_json={"_delete_original_status": "available"},
    )
    service = _deletion_service(database, tmp_path)

    service.delete_handler(HandlerContext(task_id="retry"), {"model_id": "target"})

    assert _model_status(database) is None


def test_post_delete_disk_usage_failure_reconciles_success(
    database, tmp_path, monkeypatch
):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)
    calls = 0

    def disk_usage(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(free=100)
        raise OSError("secret disk detail")

    monkeypatch.setattr("app.services.model_lifecycle.shutil.disk_usage", disk_usage)
    service = _deletion_service(database, tmp_path)

    result = service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert _model_status(database) is None
    assert result["released_bytes"] == 0
    assert result["warnings"] == ["Disk usage could not be measured"]
    assert "secret" not in str(result)


@pytest.mark.skipif(os.name == "posix", reason="unsupported-platform behavior")
def test_default_local_remover_rejects_unsupported_platform(tmp_path):
    root = tmp_path / "models"
    target = root / "target"
    target.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Secure local deletion is unsupported"):
        ModelLifecycleService._secure_rmtree(root, target)


def test_delete_handler_requires_session_factory(service):
    with pytest.raises(RuntimeError, match="session_factory is required"):
        service.delete_handler(HandlerContext(), {"model_id": "target"})


def test_delete_handler_missing_model_uses_bounded_error(database, tmp_path):
    service = _deletion_service(database, tmp_path)

    with pytest.raises(ValueError, match="^Model was not found$"):
        service.delete_handler(HandlerContext(), {"model_id": "missing"})


def test_delete_handler_rejects_already_deleting_asset(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target, status="deleting")
    service = _deletion_service(database, tmp_path)

    with pytest.raises(ValueError, match="another task"):
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert target.exists()
    assert _model_status(database) == "deleting"


def test_delete_model_requires_login_and_csrf(client, tmp_path):
    client.app.state.task_engine.stop()
    _add_asset(client.app.state.database, tmp_path / "models" / "target")

    anonymous = _delete_model(client, "target", "target")
    assert anonymous.status_code == 401
    assert anonymous.json()["detail"] == "Not authenticated"

    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    assert login.status_code == 200

    missing_csrf = _delete_model(client, "target", "target")
    assert missing_csrf.status_code == 403
    assert missing_csrf.json()["detail"] == "CSRF token is missing or invalid"


@pytest.mark.parametrize("confirmation", ["", "x" * 256])
def test_delete_model_validates_confirmation_length(
    stopped_task_client, tmp_path, confirmation
):
    _add_asset(
        stopped_task_client.app.state.database,
        tmp_path / "models" / "target",
    )

    response = _delete_model(stopped_task_client, "target", confirmation)

    assert response.status_code == 422


def test_delete_model_returns_not_found_without_task_or_audit(stopped_task_client):
    response = _delete_model(stopped_task_client, "missing", "missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"
    with stopped_task_client.app.state.database.session_factory() as db:
        assert list(db.scalars(select(TaskRecord))) == []
        assert list(
            db.scalars(
                select(AuditEvent).where(AuditEvent.action == "model.delete.create")
            )
        ) == []


def test_delete_model_rejects_confirmation_mismatch_without_task_or_audit(
    stopped_task_client, tmp_path
):
    _add_asset(
        stopped_task_client.app.state.database,
        tmp_path / "models" / "target",
        name="Exact Model Name",
    )

    response = _delete_model(stopped_task_client, "target", "exact model name")

    assert response.status_code == 422
    assert response.json()["detail"] == "Confirmation does not match model name"
    with stopped_task_client.app.state.database.session_factory() as db:
        assert list(db.scalars(select(TaskRecord))) == []
        assert list(
            db.scalars(
                select(AuditEvent).where(AuditEvent.action == "model.delete.create")
            )
        ) == []


def test_delete_model_reports_all_references_without_task_or_audit(
    stopped_task_client, tmp_path
):
    database = stopped_task_client.app.state.database
    target_path = tmp_path / "models" / "target"
    _add_asset(database, target_path, name="Referenced Model")
    with database.session_factory() as db:
        db.add_all(
            [
                _deployment(
                    deployment_id="base-id",
                    name="a-base",
                    model_id="target",
                ),
                _deployment(
                    deployment_id="draft-id",
                    name="b-draft",
                    config={"speculative": {"draft_model_id": "target"}},
                ),
                _deployment(
                    deployment_id="legacy-id",
                    name="c-legacy",
                    config={"model_path": str(target_path)},
                ),
            ]
        )
        db.commit()

    response = _delete_model(stopped_task_client, "target", "Referenced Model")

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "model_in_use",
        "references": [
            {
                "deployment_id": "base-id",
                "deployment_name": "a-base",
                "usage": "base",
            },
            {
                "deployment_id": "draft-id",
                "deployment_name": "b-draft",
                "usage": "draft",
            },
            {
                "deployment_id": "legacy-id",
                "deployment_name": "c-legacy",
                "usage": "legacy_path",
            },
        ],
    }
    with database.session_factory() as db:
        assert list(db.scalars(select(TaskRecord))) == []
        assert list(
            db.scalars(
                select(AuditEvent).where(AuditEvent.action == "model.delete.create")
            )
        ) == []


def test_delete_model_creates_bounded_audited_task(stopped_task_client, tmp_path):
    database = stopped_task_client.app.state.database
    secret_path = tmp_path / "models" / "private-model-path"
    _add_asset(
        database,
        secret_path,
        model_id="model-1",
        name="Qwen Model",
        source="huggingface",
        repository_id="org/qwen-model",
        status="unavailable",
    )

    response = _delete_model(stopped_task_client, "model-1", "Qwen Model")

    assert response.status_code == 202
    body = response.json()
    assert body["type"] == "model.delete"
    assert body["status"] == "queued"
    assert body["title"] == "删除模型 Qwen Model"
    with database.session_factory() as db:
        task = db.get(TaskRecord, body["id"])
        assert task is not None
        assert task.input_json == {"model_id": "model-1"}
        assert task.idempotency_key == "model:model-1:delete"
        audit = db.scalar(
            select(AuditEvent).where(AuditEvent.action == "model.delete.create")
        )
        assert audit is not None
        assert audit.actor == "admin"
        assert audit.resource_type == "model"
        assert audit.resource_id == "model-1"
        assert audit.details == {
            "task_id": task.id,
            "source": "huggingface",
            "repository_id": "org/qwen-model",
        }
        serialized_audit = json.dumps(audit.details)
        assert str(secret_path) not in serialized_audit
        assert "confirmation" not in serialized_audit
        assert "token" not in serialized_audit


def test_delete_model_audit_failure_rolls_back_task_and_does_not_notify(
    settings, tmp_path, monkeypatch
):
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.task_engine.stop()
        login = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "Test-password-1234"},
        )
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        _add_asset(
            app.state.database,
            tmp_path / "models" / "target",
            name="Target Model",
        )
        notifications = []
        monkeypatch.setattr(
            app.state.task_engine,
            "notify",
            lambda: notifications.append("notified"),
            raising=False,
        )

        def fail_audit(*_args, **_kwargs):
            raise RuntimeError("audit unavailable")

        monkeypatch.setattr("app.api.inventory.record_audit", fail_audit)

        response = _delete_model(client, "target", "Target Model")

        assert response.status_code == 500
        assert notifications == []
        with app.state.database.session_factory() as db:
            assert list(db.scalars(select(TaskRecord))) == []
            assert list(
                db.scalars(
                    select(AuditEvent).where(
                        AuditEvent.action == "model.delete.create"
                    )
                )
            ) == []


def test_delete_model_reuses_queued_task_before_worker_start(
    stopped_task_client, tmp_path
):
    database = stopped_task_client.app.state.database
    _add_asset(database, tmp_path / "models" / "target", name="Target Model")

    first = _delete_model(stopped_task_client, "target", "Target Model")
    second = _delete_model(stopped_task_client, "target", "Target Model")

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["id"] == first.json()["id"]
    with database.session_factory() as db:
        assert len(list(db.scalars(select(TaskRecord)))) == 1


def test_delete_model_rejects_deleting_asset_even_with_running_task(
    stopped_task_client, tmp_path
):
    database = stopped_task_client.app.state.database
    with database.session_factory() as db:
        task = stopped_task_client.app.state.task_engine.create_task(
            db,
            task_type="model.delete",
            title="删除模型 Target Model",
            input_json={"model_id": "target"},
            idempotency_key="model:target:delete",
        )
        task.status = "running"
        db.commit()
        task_id = task.id
    _add_asset(
        database,
        tmp_path / "models" / "target",
        name="Target Model",
        status="deleting",
        metadata_json={"_delete_task_id": task_id},
    )

    response = _delete_model(stopped_task_client, "target", "Target Model")

    assert response.status_code == 409
    assert response.json()["detail"] == "Model deletion is already in progress"
    with database.session_factory() as db:
        task = db.get(TaskRecord, task_id)
        assert task is not None
        assert task.status == "running"
        assert list(
            db.scalars(
                select(AuditEvent).where(AuditEvent.action == "model.delete.create")
            )
        ) == []


def test_delete_model_rejects_asset_already_deleting(stopped_task_client, tmp_path):
    database = stopped_task_client.app.state.database
    _add_asset(
        database,
        tmp_path / "models" / "target",
        name="Target Model",
        status="deleting",
    )

    response = _delete_model(stopped_task_client, "target", "Target Model")

    assert response.status_code == 409
    assert "deletion" in response.json()["detail"].lower()
    with database.session_factory() as db:
        assert list(db.scalars(select(TaskRecord))) == []
        assert list(
            db.scalars(
                select(AuditEvent).where(AuditEvent.action == "model.delete.create")
            )
        ) == []


def test_delete_model_allows_retry_after_delete_failure(stopped_task_client, tmp_path):
    database = stopped_task_client.app.state.database
    _add_asset(
        database,
        tmp_path / "models" / "target",
        name="Target Model",
        status="delete_failed",
    )

    response = _delete_model(stopped_task_client, "target", "Target Model")

    assert response.status_code == 202
    assert response.json()["type"] == "model.delete"


def test_create_app_registers_model_delete_handler(settings):
    app = create_app(settings)

    handler = app.state.task_engine.handlers["model.delete"]
    assert handler.__self__ is app.state.model_lifecycle_service
    assert handler.__func__ is app.state.model_lifecycle_service.delete_handler.__func__
    assert app.state.model_lifecycle_service.session_factory is app.state.database.session_factory
    assert app.state.model_lifecycle_service.discovery_service is app.state.discovery_service
