from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from app.db import Database
from app.models import Deployment, ModelAsset
from app.services.model_lifecycle import ModelLifecycleService, ModelReference, _resolved


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
