from __future__ import annotations

import json
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.db import Database
from app.models import Deployment, ModelAsset
from app.services.model_lifecycle import (
    ModelInUseError,
    ModelLifecycleService,
    ModelReference,
    _resolved,
)


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
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.checks = 0

    def check_control(self):
        self.checks += 1
        if self.error is not None:
            raise self.error


def _deletion_service(
    database,
    tmp_path,
    *,
    command_runner=None,
    discovery_service=None,
) -> ModelLifecycleService:
    return ModelLifecycleService(
        model_roots=(tmp_path / "models",),
        hf_cache_dir=tmp_path / "hf-cache",
        session_factory=database.session_factory,
        command_runner=command_runner,
        discovery_service=discovery_service,
    )


def _add_asset(
    database,
    path: Path,
    *,
    source: str = "local",
    repository_id: str | None = None,
    status: str = "available",
):
    with database.session_factory() as db:
        db.add(
            ModelAsset(
                id="target",
                name="target",
                source=source,
                repository_id=repository_id,
                local_path=str(path),
                status=status,
                size_bytes=123,
            )
        )
        db.commit()


def _model_status(database, model_id: str = "target") -> str | None:
    with database.session_factory() as db:
        asset = db.get(ModelAsset, model_id)
        return None if asset is None else asset.status


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
    assert result["inventory_models"] == 0


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
            payload = {"repos_deleted": 1, "revisions_deleted": 1, "freed": 6}
        else:
            payload = {"dry_run": True, "repos": 1, "revisions": 1, "size": 6}
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


def test_huggingface_malformed_yes_json_preserves_database_record(database, tmp_path):
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
        stdout = (
            "not-json"
            if "--yes" in argv
            else json.dumps({"dry_run": True, "repos": 1})
        )
        return SimpleNamespace(stdout=stdout)

    service = _deletion_service(database, tmp_path, command_runner=runner)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert len(calls) == 2
    assert _model_status(database) == "delete_failed"


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


def test_cancellation_is_checked_before_local_destructive_action(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)
    service = _deletion_service(database, tmp_path)

    with pytest.raises(RuntimeError, match="cancelled"):
        service.delete_handler(
            HandlerContext(RuntimeError("cancelled")), {"model_id": "target"}
        )

    assert target.exists()
    assert _model_status(database) == "delete_failed"


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
    service = _deletion_service(database, tmp_path)

    def fail_delete(_path):
        raise OSError("private filesystem detail")

    monkeypatch.setattr("app.services.model_lifecycle.shutil.rmtree", fail_delete)
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


def test_discovery_failure_preserves_record_as_delete_failed(database, tmp_path):
    target = tmp_path / "models" / "target"
    target.mkdir(parents=True)
    _add_asset(database, target)

    class Discovery:
        def scan_models(self, _db):
            raise RuntimeError("scan failed")

    service = _deletion_service(database, tmp_path, discovery_service=Discovery())

    with pytest.raises(RuntimeError, match="scan failed"):
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert _model_status(database) == "delete_failed"


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

    with pytest.raises(ValueError, match="already being deleted"):
        service.delete_handler(HandlerContext(), {"model_id": "target"})

    assert target.exists()
    assert _model_status(database) == "deleting"
