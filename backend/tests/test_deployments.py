import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import docker
import pytest
from app.config import Settings
from app.db import Database
from app.main import create_app
from app.models import AuditEvent, Deployment, ModelAsset, OperationPlan, TaskRecord
from app.runtime.base import (
    DeploymentSpec,
    GenerationDefaults,
    RecommendationProvenance,
    ResolvedDeploymentSpec,
    ResourceSnapshot,
    SpeculativeConfig,
    deterministic_container_name,
    validate_model_path,
)
from app.runtime.sglang import SGLangAdapter
from app.runtime.vllm import VllmAdapter
from app.services import deployments as deployment_service
from app.services.draft_models import DraftCandidate
from app.services.model_evidence import ModelEvidenceLoader
from app.services.resource_estimator import ResourceEstimate, ResourceEstimator
from app.services.runtime_capabilities import RuntimeCapabilities
from app.tasks.engine import TaskCancelled
from sqlalchemy import event
from sqlalchemy.orm import Session


def valid_spec_payload(tmp_path):
    return {
        "name": "Qwen Local",
        "model_id": "org/qwen-model",
        "model_path": str(tmp_path / "models" / "qwen"),
        "api_model_name": "qwen-local",
        "runtime": "vllm",
        "image": "vllm:test",
        "port": 8100,
        "quantization": "modelopt_fp4",
        "generation_defaults": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": -0.1,
            "max_tokens": 2048,
            "stop": ["</s>", "<|end|>"],
        },
        "speculative": {
            "draft_model_id": "org/qwen-draft",
            "method": "eagle3",
            "num_speculative_tokens": 8,
            "num_steps": 2,
            "eagle_top_k": 4,
            "num_draft_tokens": 16,
            "manual_review_acknowledged": True,
        },
        "recommendation": {
            "generated_at": datetime(2026, 8, 16, tzinfo=UTC),
            "evidence_hash": "a" * 64,
            "provider_id": "provider-1",
            "resource_snapshot": {
                "total_bytes": 1_000,
                "available_bytes": 800,
                "reserved_bytes": 200,
            },
            "modified_fields": ["generation_defaults.temperature", "quantization"],
            "sources": {
                "generation_defaults.temperature": "model_card",
                "quantization": "local_config",
                "context_length": "runtime_default",
                "memory_fraction": "device_rule",
                "generation_defaults.max_tokens": "ai",
            },
        },
    }


def legacy_spec_fingerprint(spec: DeploymentSpec) -> str:
    public = (
        spec.public_dump()
        if isinstance(spec, ResolvedDeploymentSpec)
        else spec.model_dump(mode="json")
    )
    canonical = json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def valid_recommendation_payload():
    return {
        "generated_at": "2026-08-16T00:00:00Z",
        "evidence_hash": "a" * 64,
        "provider_id": "provider-1",
        "resource_snapshot": {
            "total_bytes": 1_000,
            "available_bytes": 800,
            "reserved_bytes": 200,
        },
        "modified_fields": ["generation_defaults.temperature"],
        "sources": {"generation_defaults.temperature": "model_card"},
    }


class StaticRuntimeCapabilities:
    def __init__(self, *, methods=None, mapping=None):
        self.calls = 0
        self.value = RuntimeCapabilities(
            runtime="vllm",
            image="vllm:test",
            image_digest="sha256:test",
            source="probe",
            generation_defaults=[
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "max_tokens",
                "stop",
            ],
            quantization_methods=["auto", "modelopt_fp4"],
            quantization_mapping={"nvfp4": "modelopt_fp4"},
            speculative_methods=(["draft_model", "eagle3"] if methods is None else methods),
            method_mapping=(
                {"draft_model": "draft_model", "eagle3": "eagle3"} if mapping is None else mapping
            ),
            speculative_transport="json",
            warnings=[],
        )

    def get(self, runtime, image):
        self.calls += 1
        return self.value.model_copy(update={"runtime": runtime, "image": image})


class StaticDraftService:
    def __init__(self, candidate):
        self.candidate = candidate

    def list_candidates(self, db, target, capabilities, snapshot):
        return [self.candidate]


class StaticEstimator:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def estimate(self, **kwargs):
        self.calls.append(kwargs)
        return ResourceEstimate(
            total_bytes=64 * 1024**3,
            available_bytes=48 * 1024**3,
            reserved_bytes=8 * 1024**3,
            weight_bytes=8,
            draft_weight_bytes=kwargs.get("draft_size_bytes", 0),
            kv_cache_bytes=1024,
            runtime_overhead_bytes=4 * 1024**3,
            required_bytes=4 * 1024**3 + 1032,
            decision=self.decision,
            confidence="high",
            reasons=[f"resource decision is {self.decision}"],
        )


class HandlerContext:
    def __init__(self, task_id="task-1", *, update_error=None):
        self.task_id = task_id
        self.update_error = update_error
        self.messages = []

    def update(self, **kwargs):
        if self.update_error is not None:
            raise self.update_error
        if kwargs.get("message"):
            self.messages.append(kwargs["message"])

    def check_control(self):
        return None


class ActionContainer:
    def __init__(self, container_id, name, *, status="running"):
        self.id = container_id
        self.name = name
        self.status = status
        self.reloads = 0
        self.stops = 0
        self.starts = 0
        self.restarts = 0
        self.removed = False

    def reload(self):
        self.reloads += 1

    def stop(self, **_kwargs):
        self.stops += 1
        self.status = "exited"

    def start(self):
        self.starts += 1
        self.status = "running"

    def restart(self, **_kwargs):
        self.restarts += 1
        self.status = "running"

    def remove(self):
        self.removed = True


def add_model_asset(db, path, *, name="Base", status="available", **values):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        '{"architectures":["Qwen2ForCausalLM"],"hidden_size":256,'
        '"num_hidden_layers":2,"num_key_value_heads":2,"head_dim":128}',
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"weights")
    asset = ModelAsset(
        name=name,
        local_path=str(path),
        status=status,
        size_bytes=7,
        repository_id=values.pop("repository_id", f"org/{name.lower()}"),
        **values,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def build_preflight_service(
    database,
    model_roots,
    *,
    host_model_roots=None,
    capabilities=None,
    draft_service=None,
    estimator=None,
    snapshot=None,
):
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=model_roots)
    return deployment_service.DeploymentService(
        adapters={"vllm": adapter},
        session_factory=database.session_factory,
        model_roots=model_roots,
        host_model_roots=host_model_roots,
        runtime_capability_service=capabilities or StaticRuntimeCapabilities(),
        evidence_loader=ModelEvidenceLoader(),
        draft_service=draft_service,
        resource_estimator=estimator or ResourceEstimator(),
        system_snapshot=snapshot
        or (
            lambda: {
                "memory": {
                    "total_bytes": 64 * 1024**3,
                    "available_bytes": 64 * 1024**3,
                }
            }
        ),
    )


def build_action_service(database, model_root, target):
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(model_root,))
    get_calls = []

    class Containers:
        def get(self, identifier):
            get_calls.append(identifier)
            if isinstance(target, BaseException):
                raise target
            if callable(target):
                return target(identifier)
            return target

    client = type("Client", (), {"containers": Containers()})()
    service = deployment_service.DeploymentService(
        adapters={"vllm": adapter},
        session_factory=database.session_factory,
        model_roots=(model_root,),
        docker_client=client,
    )
    return service, adapter, get_calls


def add_action_deployment(db, *, managed, name="service", **values):
    deployment = Deployment(
        name=name,
        runtime="vllm",
        container_id=values.pop("container_id", "a" * 64),
        container_name=values.pop("container_name", f"dgx-{name}"),
        endpoint_url="http://127.0.0.1:8100",
        api_model_name=values.pop("api_model_name", name),
        status=values.pop("status", "running"),
        health=values.pop("health", "healthy"),
        managed=managed,
        **values,
    )
    db.add(deployment)
    db.commit()
    db.refresh(deployment)
    return deployment


def action_handler_payload(deployment_id, action, *, container_id, container_name):
    return {
        "deployment_id": deployment_id,
        "action": action,
        "expected_container_id": container_id,
        "expected_container_name": container_name,
    }


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"confirm_container_name": "wrong-container"}],
)
def test_discovered_delete_requires_exact_container_confirmation(authenticated_client, payload):
    authenticated_client.app.state.task_engine.stop()
    with authenticated_client.app.state.database.session_factory() as db:
        deployment = add_action_deployment(db, managed=False, name="discovered")
        deployment_id = deployment.id

    kwargs = {} if payload is None else {"json": payload}
    response = authenticated_client.post(f"/api/deployments/{deployment_id}/delete", **kwargs)

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "To uninstall a discovered service, confirm_container_name must exactly "
        "match its container name"
    )
    with authenticated_client.app.state.database.session_factory() as db:
        assert db.query(TaskRecord).filter_by(type="deployment.action").count() == 0


def test_discovered_delete_confirmation_is_not_persisted(authenticated_client):
    authenticated_client.app.state.task_engine.stop()
    with authenticated_client.app.state.database.session_factory() as db:
        deployment = add_action_deployment(db, managed=False, name="discovered")
        deployment_id = deployment.id
        container_name = deployment.container_name

    response = authenticated_client.post(
        f"/api/deployments/{deployment_id}/delete",
        json={"confirm_container_name": container_name},
    )

    assert response.status_code == 202
    with authenticated_client.app.state.database.session_factory() as db:
        task = db.get(TaskRecord, response.json()["id"])
        assert task.title == "卸载服务 discovered"
        assert task.input_json == {
            "deployment_id": deployment_id,
            "action": "delete",
            "expected_container_id": "a" * 64,
            "expected_container_name": "dgx-discovered",
        }
        audit = db.query(AuditEvent).filter_by(action="deployment.delete").one()
        assert "confirm" not in json.dumps(audit.details).lower()


def test_discovered_delete_never_treats_missing_container_name_as_confirmation(
    authenticated_client,
):
    authenticated_client.app.state.task_engine.stop()
    with authenticated_client.app.state.database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name="unnamed-discovered",
            container_name=None,
        )
        deployment_id = deployment.id

    response = authenticated_client.post(f"/api/deployments/{deployment_id}/delete", json={})

    assert response.status_code == 422
    with authenticated_client.app.state.database.session_factory() as db:
        assert db.query(TaskRecord).filter_by(type="deployment.action").count() == 0


@pytest.mark.parametrize("action", ["start", "stop", "restart", "delete"])
def test_managed_deployment_actions_remain_body_optional(authenticated_client, action):
    authenticated_client.app.state.task_engine.stop()
    with authenticated_client.app.state.database.session_factory() as db:
        deployment = add_action_deployment(db, managed=True, name=f"managed-{action}")
        deployment_id = deployment.id

    response = authenticated_client.post(f"/api/deployments/{deployment_id}/{action}")

    assert response.status_code == 202


def test_deployment_spec_serializes_recommendation_settings(tmp_path):
    spec = DeploymentSpec.model_validate(valid_spec_payload(tmp_path))

    dumped = spec.model_dump(mode="json")

    assert dumped["quantization"] == "modelopt_fp4"
    assert dumped["generation_defaults"]["stop"] == ["</s>", "<|end|>"]
    assert dumped["speculative"]["method"] == "eagle3"
    assert dumped["speculative"]["num_draft_tokens"] == 16
    assert dumped["recommendation"]["generated_at"] == "2026-08-16T00:00:00Z"
    assert dumped["recommendation"]["sources"]["memory_fraction"] == "device_rule"


@pytest.mark.parametrize(
    ("field", "lower", "upper", "below", "above"),
    [
        ("temperature", 0, 2, -0.01, 2.01),
        ("top_p", 0.01, 1, 0, 1.01),
        ("top_k", 0, 1_000_000, -1, 1_000_001),
        ("min_p", 0, 1, -0.01, 1.01),
        ("repetition_penalty", 0.01, 2, 0, 2.01),
        ("presence_penalty", -2, 2, -2.01, 2.01),
        ("frequency_penalty", -2, 2, -2.01, 2.01),
        ("max_tokens", 1, 1_048_576, 0, 1_048_577),
    ],
)
def test_generation_defaults_enforces_numeric_boundaries(field, lower, upper, below, above):
    assert getattr(GenerationDefaults.model_validate({field: lower}), field) == lower
    assert getattr(GenerationDefaults.model_validate({field: upper}), field) == upper

    for invalid in (below, above):
        with pytest.raises(ValueError):
            GenerationDefaults.model_validate({field: invalid})


@pytest.mark.parametrize(
    "stop",
    [[], "x", "x" * 500, ["stop"] * 16],
)
def test_generation_defaults_accepts_valid_stop_boundaries(stop):
    assert GenerationDefaults(stop=stop).stop == stop


@pytest.mark.parametrize(
    "stop",
    ["", "x" * 501, [""], ["stop"] * 17],
)
def test_generation_defaults_rejects_invalid_stop_boundaries(stop):
    with pytest.raises(ValueError, match="stop must contain"):
        GenerationDefaults(stop=stop)


@pytest.mark.parametrize(
    ("field", "lower", "upper", "below", "above"),
    [
        ("num_speculative_tokens", 1, 64, 0, 65),
        ("num_steps", 1, 32, 0, 33),
        ("eagle_top_k", 1, 32, 0, 33),
        ("num_draft_tokens", 1, 256, 0, 257),
    ],
)
def test_speculative_config_enforces_numeric_boundaries(field, lower, upper, below, above):
    def payload(value):
        settings = {
            "draft_model_id": "draft-id",
            "method": "eagle",
            field: value,
        }
        if field in {"num_steps", "eagle_top_k", "num_draft_tokens"}:
            settings.update(
                {
                    "num_steps": 1,
                    "eagle_top_k": 1,
                    "num_draft_tokens": 1,
                    field: value,
                }
            )
        return settings

    assert getattr(SpeculativeConfig.model_validate(payload(lower)), field) == lower
    assert getattr(SpeculativeConfig.model_validate(payload(upper)), field) == upper

    for invalid in (below, above):
        with pytest.raises(ValueError):
            SpeculativeConfig.model_validate(payload(invalid))


@pytest.mark.parametrize("draft_model_id", ["a", "a" * 64])
def test_speculative_config_accepts_draft_model_id_boundaries(draft_model_id):
    spec = SpeculativeConfig(draft_model_id=draft_model_id, method="draft_model")

    assert spec.draft_model_id == draft_model_id


@pytest.mark.parametrize("draft_model_id", ["", "a" * 65])
def test_speculative_config_rejects_invalid_draft_model_id_lengths(draft_model_id):
    with pytest.raises(ValueError):
        SpeculativeConfig(draft_model_id=draft_model_id, method="draft_model")


@pytest.mark.parametrize(
    "method", ["draft_model", "dflash", "dspark", "eagle", "eagle3", "mtp"]
)
def test_speculative_config_accepts_supported_methods(method):
    assert SpeculativeConfig(draft_model_id="draft-id", method=method).method == method


def test_speculative_config_rejects_unknown_method():
    with pytest.raises(ValueError):
        SpeculativeConfig(draft_model_id="draft-id", method="unknown")


def test_dflash_accepts_only_its_block_size_tuning():
    config = SpeculativeConfig(
        draft_model_id="draft-id",
        method="dflash",
        num_draft_tokens=8,
    )

    assert config.num_draft_tokens == 8
    with pytest.raises(ValueError, match="DFlash tuning only accepts num_draft_tokens"):
        SpeculativeConfig(
            draft_model_id="draft-id",
            method="dflash",
            num_steps=2,
            eagle_top_k=4,
            num_draft_tokens=8,
        )


@pytest.mark.parametrize(
    "tuning",
    [
        {"num_steps": 2},
        {"eagle_top_k": 4},
        {"num_draft_tokens": 16},
        {"num_steps": 2, "eagle_top_k": 4},
        {"num_steps": 2, "num_draft_tokens": 16},
        {"eagle_top_k": 4, "num_draft_tokens": 16},
    ],
)
def test_speculative_eagle_tuning_fields_must_be_set_together(tmp_path, tuning):
    payload = valid_spec_payload(tmp_path)
    payload["quantization"] = "fp8"
    payload["speculative"] = {
        "draft_model_id": "org/qwen-draft",
        "method": "eagle",
        **tuning,
    }

    with pytest.raises(
        ValueError,
        match="set num_steps, eagle_top_k and num_draft_tokens together",
    ):
        DeploymentSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resolved_draft_model_path", "/models/qwen-draft"),
        ("speculative_runtime_method", "eagle"),
    ],
)
def test_public_deployment_spec_rejects_internal_resolution_fields(tmp_path, field, value):
    payload = valid_spec_payload(tmp_path)
    payload["quantization"] = "fp8"
    payload[field] = value

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        DeploymentSpec.model_validate(payload)


def test_resolved_deployment_spec_public_dump_is_json_safe_and_excludes_internal_fields(
    tmp_path,
):
    resolved = ResolvedDeploymentSpec.model_validate(
        {
            **valid_spec_payload(tmp_path),
            "resolved_draft_model_path": str(tmp_path / "models" / "qwen-draft"),
            "draft_container_model_path": "/models/qwen-draft",
            "speculative_runtime_method": "eagle",
        }
    )

    public = resolved.public_dump()

    assert json.loads(json.dumps(public)) == public
    assert public["recommendation"]["generated_at"] == "2026-08-16T00:00:00Z"
    assert set(public) == set(DeploymentSpec.model_fields)
    assert "resolved_draft_model_path" not in public
    assert "draft_container_model_path" not in public
    assert "speculative_runtime_method" not in public


def test_deployment_spec_roundtrips_resource_warning_acknowledgement(tmp_path):
    payload = valid_spec_payload(tmp_path)
    payload["resource_warning_acknowledged"] = True

    spec = DeploymentSpec.model_validate(payload)

    assert spec.resource_warning_acknowledged is True
    assert spec.model_dump(mode="json")["resource_warning_acknowledged"] is True


def test_deployment_spec_fingerprint_ignores_browser_model_path(tmp_path):
    first = DeploymentSpec.model_validate(valid_spec_payload(tmp_path))
    second = first.model_copy(update={"model_path": str(tmp_path / "other-browser-path")})

    assert deployment_service.deployment_spec_fingerprint(
        first
    ) == deployment_service.deployment_spec_fingerprint(second)


def test_stored_container_fingerprints_accept_specs_created_before_optional_fields(
    tmp_path,
):
    spec = DeploymentSpec(
        name="Qwen",
        model_id="model-1",
        model_path=str(tmp_path / "models" / "qwen"),
        api_model_name="qwen",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )
    deployment = Deployment(
        name=spec.name,
        model_id=spec.model_id,
        runtime=spec.runtime,
        api_model_name=spec.api_model_name,
        image=spec.image,
        port=spec.port,
        status="stopped",
        health="unknown",
        managed=True,
        config={"spec": spec.model_dump(mode="json")},
    )
    legacy = spec.model_dump(mode="json")
    legacy.pop("chat_template_kwargs")
    legacy_without_path = {key: value for key, value in legacy.items() if key != "model_path"}

    def fingerprint(payload):
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    accepted = deployment_service.DeploymentService._stored_container_fingerprints(
        deployment
    )

    assert fingerprint(legacy) in accepted
    assert fingerprint(legacy_without_path) in accepted


def test_resolve_spec_uses_available_database_asset_instead_of_browser_path(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'resolve.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "org" / "base")
        target_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="Base",
        model_id=target_id,
        model_path=str(tmp_path / "browser-controlled"),
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with database.session_factory() as db:
        resolved = service.resolve_spec(db, spec)

    assert resolved.model_path == str((root / "org" / "base").resolve())
    assert resolved.base_container_model_path == "/models/org/base"
    assert resolved.base_model_root == str(root.resolve())
    assert resolved.public_dump()["model_path"] == str((root / "org" / "base").resolve())


@pytest.mark.parametrize("status", ["missing", "unavailable"])
def test_resolve_spec_rejects_missing_or_unavailable_database_asset(tmp_path, status):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / f'{status}.db'}")
    database.create_schema()
    model_id = "missing"
    if status == "unavailable":
        with database.session_factory() as db:
            target = add_model_asset(db, root / "base", status="failed")
            model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="Base",
        model_id=model_id,
        model_path=str(root / "base"),
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with (
        database.session_factory() as db,
        pytest.raises(ValueError, match="Base model is missing or unavailable"),
    ):
        service.resolve_spec(db, spec)


def test_resolve_spec_maps_cross_root_draft_to_separate_read_only_mount(tmp_path):
    base_root = tmp_path / "base-models"
    draft_root = tmp_path / "draft-models"
    host_base = Path("/srv/base-models")
    host_draft = Path("/srv/draft-models")
    database = Database(f"sqlite:///{tmp_path / 'cross-root.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, base_root / "base", name="Base")
        draft = add_model_asset(db, draft_root / "draft", name="Draft")
        base_id, draft_id = base.id, draft.id
    service = build_preflight_service(
        database,
        (base_root, draft_root),
        host_model_roots=(host_base, host_draft),
    )
    spec = DeploymentSpec(
        name="Base",
        model_id=base_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        speculative={
            "draft_model_id": draft_id,
            "method": "draft_model",
        },
    )
    with database.session_factory() as db:
        resolved = service.resolve_spec(db, spec)

    captured = {}
    containers = type(
        "Containers",
        (),
        {"run": lambda _self, _image, **kwargs: captured.update(kwargs) or object()},
    )()
    service._run_container(
        type("Client", (), {"containers": containers})(),
        resolved,
        service.adapter("vllm"),
        "dgx-base",
    )

    assert resolved.draft_container_model_path == "/draft-models/draft"
    assert captured["volumes"] == {
        str(host_base): {"bind": "/models", "mode": "ro"},
        str(host_draft): {"bind": "/draft-models", "mode": "ro"},
    }


def test_resolve_spec_reuses_base_mount_for_same_root_draft(tmp_path):
    root = tmp_path / "models"
    host_root = Path("/srv/models")
    database = Database(f"sqlite:///{tmp_path / 'same-root.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base", name="Base")
        draft = add_model_asset(db, root / "draft", name="Draft")
        base_id, draft_id = base.id, draft.id
    service = build_preflight_service(database, (root,), host_model_roots=(host_root,))
    spec = DeploymentSpec(
        name="Base",
        model_id=base_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        speculative={"draft_model_id": draft_id, "method": "draft_model"},
    )
    with database.session_factory() as db:
        resolved = service.resolve_spec(db, spec)

    captured = {}
    containers = type(
        "Containers",
        (),
        {"run": lambda _self, _image, **kwargs: captured.update(kwargs) or object()},
    )()
    service._run_container(
        type("Client", (), {"containers": containers})(),
        resolved,
        service.adapter("vllm"),
        "dgx-base",
    )

    assert resolved.draft_container_model_path == "/models/draft"
    assert captured["volumes"] == {str(host_root): {"bind": "/models", "mode": "ro"}}


@pytest.mark.parametrize(
    ("draft_state", "message"),
    [
        ("missing", "Draft Model is missing or unavailable"),
        ("unavailable", "Draft Model is missing or unavailable"),
        ("same", "Base model and Draft Model must be different"),
    ],
)
def test_resolve_spec_rejects_invalid_draft_assets(tmp_path, draft_state, message):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / f'draft-{draft_state}.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base", name="Base")
        if draft_state == "missing":
            draft_id = "missing"
        elif draft_state == "same":
            draft_id = base.id
        else:
            draft_id = add_model_asset(db, root / "draft", name="Draft", status="failed").id
        base_id = base.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="base",
        model_id=base_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        speculative={"draft_model_id": draft_id, "method": "draft_model"},
    )

    with database.session_factory() as db, pytest.raises(ValueError, match=message):
        service.resolve_spec(db, spec)


@pytest.mark.parametrize(
    ("capabilities", "message"),
    [
        (
            StaticRuntimeCapabilities(methods=[]),
            "Speculative method is unsupported by the runtime",
        ),
        (
            StaticRuntimeCapabilities(methods=["draft_model"], mapping={}),
            "Speculative method mapping is unavailable",
        ),
    ],
)
def test_resolve_spec_rejects_unsupported_speculative_capabilities(tmp_path, capabilities, message):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'capabilities.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base", name="Base")
        draft = add_model_asset(db, root / "draft", name="Draft")
        base_id, draft_id = base.id, draft.id
    service = build_preflight_service(database, (root,), capabilities=capabilities)
    spec = DeploymentSpec(
        name="base",
        model_id=base_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        speculative={"draft_model_id": draft_id, "method": "draft_model"},
    )

    with database.session_factory() as db, pytest.raises(ValueError, match=message):
        service.resolve_spec(db, spec)


def test_disallowed_image_is_rejected_without_capability_probe(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'disallowed-image.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    capabilities = StaticRuntimeCapabilities()
    service = build_preflight_service(database, (root,), capabilities=capabilities)
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:not-allowed",
        port=8100,
    )

    with database.session_factory() as db, pytest.raises(ValueError, match="Image is not allowed"):
        service.preview(db, spec)

    assert capabilities.calls == 0


def test_preflight_gets_runtime_capabilities_once(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'single-capability.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    capabilities = StaticRuntimeCapabilities()
    service = build_preflight_service(database, (root,), capabilities=capabilities)
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with database.session_factory() as db:
        service.preview(db, spec)

    assert capabilities.calls == 1


def test_run_container_rejects_public_browser_spec(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'public-run.db'}")
    database.create_schema()
    service = build_preflight_service(database, (tmp_path,))
    spec = DeploymentSpec(
        name="base",
        model_path=str(tmp_path),
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with pytest.raises(ValueError, match="Resolved deployment spec is required"):
        service._run_container(object(), spec, service.adapter("vllm"), "dgx-base")


def test_preview_rejects_shared_route_with_different_generation_defaults(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'route.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base")
        deployment = Deployment(
            name="replica-a",
            model_id=base.id,
            runtime="vllm",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="replica-a",
            status="running",
            health="healthy",
            managed=True,
            image="vllm:test",
            port=8100,
            config={
                "spec": {
                    "route_alias": "shared",
                    "generation_defaults": {"temperature": 0.2},
                }
            },
        )
        db.add(deployment)
        db.commit()
        base_id = base.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="replica-b",
        model_id=base_id,
        model_path="ignored",
        api_model_name="replica-b",
        route_alias="shared",
        runtime="vllm",
        image="vllm:test",
        port=8101,
        generation_defaults={"temperature": 0.7},
    )

    with (
        database.session_factory() as db,
        pytest.raises(ValueError, match="Shared route generation defaults must match"),
    ):
        service.preview(db, spec)


def test_preview_excludes_current_deployment_from_shared_route_check(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'route-edit.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base")
        deployment = Deployment(
            name="replica-a",
            model_id=base.id,
            runtime="vllm",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="replica-a",
            status="running",
            health="healthy",
            managed=True,
            image="vllm:test",
            port=8100,
            config={
                "spec": {
                    "route_alias": "shared",
                    "generation_defaults": {"temperature": 0.2},
                }
            },
        )
        db.add(deployment)
        db.commit()
        base_id, deployment_id = base.id, deployment.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="replica-a",
        model_id=base_id,
        model_path="ignored",
        api_model_name="replica-a",
        route_alias="shared",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        generation_defaults={"temperature": 0.7},
    )

    with database.session_factory() as db:
        preview = service.preview(db, spec, exclude_deployment_id=deployment_id)

    assert preview["spec"]["generation_defaults"]["temperature"] == 0.7


@pytest.mark.parametrize(
    ("existing_api_name", "existing_config", "new_api_name", "new_route_alias"),
    [
        (
            "shared",
            {"spec": {"generation_defaults": {"temperature": 0.2}}},
            "replica-b",
            "shared",
        ),
        (
            "replica-a",
            {
                "route_alias": "shared",
                "spec": {"generation_defaults": {"temperature": 0.2}},
            },
            "shared",
            None,
        ),
        (
            "replica-a",
            {
                "route_alias": "shared",
                "generation_defaults": {"temperature": 0.2},
            },
            "shared",
            None,
        ),
    ],
)
def test_preview_compares_effective_routes_and_legacy_generation_defaults(
    tmp_path,
    existing_api_name,
    existing_config,
    new_api_name,
    new_route_alias,
):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'effective-route.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        db.add(
            Deployment(
                name="replica-a",
                model_id=target.id,
                runtime="vllm",
                endpoint_url="http://127.0.0.1:8100",
                api_model_name=existing_api_name,
                status="running",
                health="healthy",
                managed=True,
                image="vllm:test",
                port=8100,
                config=existing_config,
            )
        )
        db.commit()
        model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="replica-b",
        model_id=model_id,
        model_path="ignored",
        api_model_name=new_api_name,
        route_alias=new_route_alias,
        runtime="vllm",
        image="vllm:test",
        port=8101,
        generation_defaults={"temperature": 0.7},
    )

    with (
        database.session_factory() as db,
        pytest.raises(ValueError, match="Shared route generation defaults must match"),
    ):
        service.preview(db, spec)


def test_preview_allows_effective_route_with_matching_generation_defaults(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'matching-route.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        db.add(
            Deployment(
                name="replica-a",
                model_id=target.id,
                runtime="vllm",
                endpoint_url="http://127.0.0.1:8100",
                api_model_name="shared",
                status="running",
                health="healthy",
                managed=True,
                image="vllm:test",
                port=8100,
                config={
                    "generation_defaults": {
                        "temperature": 0.2,
                        "top_p": None,
                    }
                },
            )
        )
        db.commit()
        model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="replica-b",
        model_id=model_id,
        model_path="ignored",
        api_model_name="replica-b",
        route_alias="shared",
        runtime="vllm",
        image="vllm:test",
        port=8101,
        generation_defaults={"temperature": 0.2},
    )

    with database.session_factory() as db:
        preview = service.preview(db, spec)

    assert preview["route_alias"] == "shared"


@pytest.mark.parametrize("decision", ["blocked", "warning"])
def test_preview_enforces_realtime_resource_decision(tmp_path, decision):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / f'{decision}.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base")
        base_id = base.id
    estimator = StaticEstimator(decision)
    snapshot_calls = 0

    def snapshot():
        nonlocal snapshot_calls
        snapshot_calls += 1
        return {
            "memory": {
                "total_bytes": 64 * 1024**3,
                "available_bytes": 48 * 1024**3,
            }
        }

    service = build_preflight_service(database, (root,), estimator=estimator, snapshot=snapshot)
    spec = DeploymentSpec(
        name="base",
        model_id=base_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        resource_warning_acknowledged=False,
    )

    expected = "Deployment is blocked" if decision == "blocked" else "acknowledgement"
    with database.session_factory() as db, pytest.raises(ValueError, match=expected):
        service.preview(db, spec)
    assert snapshot_calls == 1
    assert estimator.calls[0]["context_length"] == spec.context_length

    if decision == "warning":
        with database.session_factory() as db:
            preview = service.preview(
                db, spec.model_copy(update={"resource_warning_acknowledged": True})
            )
        assert preview["resource_estimate"]["decision"] == "warning"


@pytest.mark.parametrize(
    ("candidate_status", "acknowledged", "message"),
    [
        ("incompatible", True, "Draft Model is incompatible"),
        ("review", False, "Draft Model review acknowledgement is required"),
    ],
)
def test_preview_enforces_selected_draft_compatibility(
    tmp_path, candidate_status, acknowledged, message
):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / f'draft-{candidate_status}.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base", name="Base")
        draft = add_model_asset(db, root / "draft", name="Draft")
        base_id, draft_id = base.id, draft.id
    candidate = DraftCandidate(
        model_id=draft_id,
        name="Draft",
        repository_id="org/draft",
        method="draft_model",
        status=candidate_status,
        reasons=["bounded candidate reason"],
        size_bytes=7,
        estimated_total_bytes=1024,
    )
    service = build_preflight_service(
        database,
        (root,),
        draft_service=StaticDraftService(candidate),
    )
    spec = DeploymentSpec(
        name="base",
        model_id=base_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        speculative={
            "draft_model_id": draft_id,
            "method": "draft_model",
            "manual_review_acknowledged": acknowledged,
        },
    )

    with database.session_factory() as db, pytest.raises(ValueError, match=message):
        service.preview(db, spec)


def test_preview_persists_public_spec_capabilities_resource_and_mounts(tmp_path):
    root = tmp_path / "models"
    host_root = Path("/srv/models")
    database = Database(f"sqlite:///{tmp_path / 'preview.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base")
        base_id = base.id
    service = build_preflight_service(database, (root,), host_model_roots=(host_root,))
    spec = DeploymentSpec(
        name="base",
        model_id=base_id,
        model_path=str(tmp_path / "browser-path"),
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        generation_defaults={"temperature": 0.2},
        recommendation=valid_recommendation_payload(),
    )

    with database.session_factory() as db:
        preview = service.preview(db, spec)

    assert preview["spec"]["model_path"] == str((root / "base").resolve())
    assert "base_model_root" not in preview["spec"]
    assert "resolved_draft_model_path" not in preview["spec"]
    assert preview["runtime_capabilities"]["image_digest"] == "sha256:test"
    assert preview["resource_estimate"]["decision"] == "ok"
    assert preview["mounts"]["base"]["host_root"] == str(host_root)
    assert preview["generation_defaults"] == {"temperature": 0.2}
    assert preview["recommendation"]["evidence_hash"] == "a" * 64
    json.dumps(preview)


def test_preview_rejects_incompatible_base_model_before_resource_estimation(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'incompatible-base.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base")
        (root / "base" / "model.safetensors").unlink()
        base_id = base.id
    estimator = StaticEstimator("ok")
    service = build_preflight_service(database, (root,), estimator=estimator)
    spec = DeploymentSpec(
        name="base",
        model_id=base_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with (
        database.session_factory() as db,
        pytest.raises(ValueError, match="Base model is incompatible"),
    ):
        service.preview(db, spec)

    assert estimator.calls == []


@pytest.mark.parametrize(
    "field",
    ["total_bytes", "available_bytes", "reserved_bytes"],
)
def test_resource_snapshot_enforces_nonnegative_boundaries(field):
    payload = {"total_bytes": 0, "available_bytes": 0, "reserved_bytes": 0}
    assert getattr(ResourceSnapshot.model_validate(payload), field) == 0

    payload[field] = -1
    with pytest.raises(ValueError):
        ResourceSnapshot.model_validate(payload)


@pytest.mark.parametrize(
    "evidence_hash",
    ["a" * 63, "a" * 65, "A" * 64, "g" * 64],
)
def test_recommendation_provenance_rejects_invalid_evidence_hash(evidence_hash):
    payload = valid_recommendation_payload()
    payload["evidence_hash"] = evidence_hash

    with pytest.raises(ValueError):
        RecommendationProvenance.model_validate(payload)


def test_recommendation_provenance_accepts_lowercase_sha256_hash():
    provenance = RecommendationProvenance.model_validate(valid_recommendation_payload())

    assert provenance.evidence_hash == "a" * 64


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (GenerationDefaults, {"temperature": 0.7}),
        (
            SpeculativeConfig,
            {"draft_model_id": "draft-id", "method": "draft_model"},
        ),
        (
            ResourceSnapshot,
            {"total_bytes": 1, "available_bytes": 1, "reserved_bytes": 0},
        ),
        (RecommendationProvenance, valid_recommendation_payload()),
    ],
)
def test_nested_deployment_models_reject_extra_fields(model_type, payload):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        model_type.model_validate({**payload, "unexpected": True})


def test_container_name_is_deterministic_and_safe():
    assert deterministic_container_name("Qwen 3.8 / 27B") == "dgx-qwen-3-8-27b"
    assert deterministic_container_name("Qwen 3.8 / 27B") == "dgx-qwen-3-8-27b"


def test_model_path_must_be_in_allowed_root(tmp_path):
    root = tmp_path / "models"
    model = root / "org" / "model"
    model.mkdir(parents=True)

    assert validate_model_path(model, (root,)) == model.resolve()
    with pytest.raises(ValueError):
        validate_model_path(Path(tmp_path, "outside"), (root,))


def test_runtime_adapters_generate_only_validated_arguments(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    spec = DeploymentSpec(
        name="Qwen Local",
        model_path=str(model_path),
        api_model_name="qwen-local",
        runtime="vllm",
        image="vllm/vllm-openai:v0.27.1",
        port=8100,
        context_length=32768,
        memory_fraction=0.5,
        max_concurrency=8,
    )

    vllm = VllmAdapter(
        allowed_images={"vllm/vllm-openai:v0.27.1"}, model_roots=(tmp_path / "models",)
    )
    command = vllm.command(spec)
    assert command[:4] == ["--model", "/models/qwen", "--served-model-name", "qwen-local"]
    assert "--gpu-memory-utilization" in command

    sglang_spec = spec.model_copy(update={"runtime": "sglang", "image": "sglang:test"})
    sglang = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    assert sglang.command(sglang_spec)[:3] == ["python3", "-m", "sglang.launch_server"]


def test_deployment_preview_preserves_shared_gateway_route(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    adapter = VllmAdapter(
        allowed_images={"vllm/vllm-openai:v0.27.1"}, model_roots=(tmp_path / "models",)
    )
    spec = DeploymentSpec(
        name="Qwen replica A",
        model_path=str(model_path),
        api_model_name="qwen-replica-a",
        route_alias="qwen-shared",
        runtime="vllm",
        image="vllm/vllm-openai:v0.27.1",
        port=8100,
        recommendation=valid_recommendation_payload(),
    )
    (model_path / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    (model_path / "model.safetensors").write_bytes(b"weights")

    preview = adapter.preview(spec)
    assert preview["route_alias"] == "qwen-shared"
    assert preview["compatibility"]["compatible"] is True
    assert preview["compatibility"]["architectures"] == ["Qwen2ForCausalLM"]
    assert preview["estimated_disk_bytes"] > 0
    assert preview["estimated_memory_bytes"] >= preview["estimated_disk_bytes"]
    assert preview["spec"] == spec.model_dump(mode="json")
    json.dumps(preview)
    assert preview["operations"][-1] == "Probe /v1/models and register the gateway route"
    assert "client.chat.completions.create" in preview["api_example"]

    with pytest.raises(ValueError):
        DeploymentSpec(
            name="Invalid route",
            model_path=str(model_path),
            api_model_name="qwen-replica-b",
            route_alias="invalid route with spaces",
            runtime="vllm",
            image="vllm/vllm-openai:v0.27.1",
            port=8101,
        )


def test_runtime_adapter_exposes_lifecycle_contract(tmp_path):
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path,))

    for method in (
        "start",
        "stop",
        "restart",
        "health_check",
        "logs",
        "metrics",
        "uninstall",
    ):
        assert callable(getattr(adapter, method, None)), method


def test_vllm_command_includes_batch_and_quantization_settings(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    spec = DeploymentSpec(
        name="Qwen",
        model_path=str(model_path),
        api_model_name="qwen",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        max_batched_tokens=4096,
        quantization="fp8",
    )

    command = adapter.command(spec)

    assert command[command.index("--max-num-batched-tokens") + 1] == "4096"
    assert command[command.index("--quantization") + 1] == "fp8"


def test_chat_template_kwargs_are_bounded_and_serialized_canonically(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    spec = DeploymentSpec(
        name="Qwen",
        model_path=str(model_path),
        api_model_name="qwen",
        runtime="vllm",
        image="vllm:test",
        chat_template_kwargs={"reasoning_effort": "high", "enable_thinking": True},
    )

    command = adapter.command(spec)

    index = command.index("--default-chat-template-kwargs")
    assert command[index + 1] == '{"enable_thinking":true,"reasoning_effort":"high"}'
    with pytest.raises(ValueError, match="Unsupported chat template kwarg name"):
        DeploymentSpec.model_validate(
            {**spec.model_dump(mode="json"), "chat_template_kwargs": {"bad-key": True}}
        )


def resolved_speculative_spec(
    tmp_path,
    *,
    runtime="vllm",
    method="draft_model",
    runtime_method="draft_model",
    draft_container_path="/draft-models/draft",
    **speculative_settings,
):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True, exist_ok=True)
    image = f"{runtime}:test"
    return ResolvedDeploymentSpec(
        name="Qwen",
        model_path=str(model_path),
        api_model_name="qwen",
        runtime=runtime,
        image=image,
        port=8100,
        speculative={
            "draft_model_id": "org/qwen-draft",
            "method": method,
            **speculative_settings,
        },
        resolved_draft_model_path=str(tmp_path / "models" / "qwen-draft"),
        draft_container_model_path=draft_container_path,
        speculative_runtime_method=runtime_method,
    )


def test_vllm_command_uses_canonical_speculative_json(tmp_path):
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(tmp_path, num_speculative_tokens=5)

    command = adapter.command(spec)

    index = command.index("--speculative-config")
    assert command[index + 1] == (
        '{"method":"draft_model","model":"/draft-models/draft","num_speculative_tokens":5}'
    )


def test_vllm_command_rejects_unsupported_grouped_speculative_tuning(tmp_path):
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(
        tmp_path,
        num_steps=2,
        eagle_top_k=4,
        num_draft_tokens=16,
    )

    with pytest.raises(
        ValueError,
        match="grouped speculative tuning fields are not supported by vLLM",
    ):
        adapter.command(spec)


def test_vllm_speculative_json_keeps_path_content_in_one_argument(tmp_path):
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    draft_path = "/draft models/--trust-remote-code"
    spec = resolved_speculative_spec(tmp_path, draft_container_path=draft_path)

    command = adapter.command(spec)
    payload = command[command.index("--speculative-config") + 1]

    assert json.loads(payload)["model"] == draft_path
    assert "--trust-remote-code" not in command
    assert command.count("--speculative-config") == 1


def test_sglang_command_adds_only_base_speculative_flags_without_tuning(tmp_path):
    adapter = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(
        tmp_path,
        runtime="sglang",
        method="eagle3",
        runtime_method="EAGLE3",
        draft_container_path="/models/draft",
    )

    command = adapter.command(spec)

    assert command[-4:] == [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        "/models/draft",
    ]
    assert "--speculative-num-steps" not in command
    assert "--speculative-eagle-topk" not in command
    assert "--speculative-num-draft-tokens" not in command


def test_sglang_command_adds_complete_grouped_speculative_tuning(tmp_path):
    adapter = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(
        tmp_path,
        runtime="sglang",
        method="eagle3",
        runtime_method="EAGLE3",
        draft_container_path="/models/draft",
        num_steps=2,
        eagle_top_k=4,
        num_draft_tokens=16,
    )

    assert adapter.command(spec)[-10:] == [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        "/models/draft",
        "--speculative-num-steps",
        "2",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "16",
    ]


@pytest.mark.parametrize(
    ("method", "runtime_method"),
    [
        ("draft_model", "STANDALONE"),
        ("dflash", "DFLASH"),
        ("dspark", "DSPARK"),
        ("eagle", "EAGLE"),
        ("eagle3", "EAGLE3"),
        ("mtp", "NEXTN"),
    ],
)
def test_sglang_command_accepts_only_trusted_method_mappings(tmp_path, method, runtime_method):
    adapter = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(
        tmp_path,
        runtime="sglang",
        method=method,
        runtime_method=runtime_method,
    )

    command = adapter.command(spec)

    assert command[command.index("--speculative-algorithm") + 1] == runtime_method


@pytest.mark.parametrize(
    ("runtime", "field", "message"),
    [
        ("vllm", "draft_container_model_path", "resolved draft container path is required"),
        ("vllm", "speculative_runtime_method", "resolved speculative runtime method is required"),
        ("sglang", "draft_container_model_path", "resolved draft container path is required"),
        ("sglang", "speculative_runtime_method", "resolved speculative runtime method is required"),
    ],
)
def test_speculative_commands_require_resolved_internal_fields(tmp_path, runtime, field, message):
    adapter_type = VllmAdapter if runtime == "vllm" else SGLangAdapter
    adapter = adapter_type(allowed_images={f"{runtime}:test"}, model_roots=(tmp_path / "models",))
    runtime_method = "draft_model" if runtime == "vllm" else "STANDALONE"
    spec = resolved_speculative_spec(
        tmp_path, runtime=runtime, runtime_method=runtime_method
    ).model_copy(update={field: None})

    with pytest.raises(ValueError, match=message):
        adapter.command(spec)


@pytest.mark.parametrize("draft_path", ["", "relative/draft", "--draft-model"])
def test_speculative_commands_reject_non_absolute_draft_container_paths(tmp_path, draft_path):
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(tmp_path, draft_container_path=draft_path)

    with pytest.raises(ValueError, match="resolved draft container path is required"):
        adapter.command(spec)


def test_public_speculative_spec_cannot_build_a_runtime_command(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    spec = DeploymentSpec(
        name="Qwen",
        model_path=str(model_path),
        api_model_name="qwen",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        speculative={"draft_model_id": "org/draft", "method": "draft_model"},
    )

    with pytest.raises(ValueError, match="resolved speculative runtime method is required"):
        adapter.command(spec)


@pytest.mark.parametrize(
    ("runtime", "method", "runtime_method"),
    [
        ("vllm", "draft_model", "eagle"),
        ("vllm", "draft_model", "arbitrary_option"),
        ("sglang", "draft_model", "EAGLE"),
        ("sglang", "mtp", "UNTRUSTED"),
    ],
)
def test_speculative_commands_reject_mismatched_or_unsupported_mappings(
    tmp_path, runtime, method, runtime_method
):
    adapter_type = VllmAdapter if runtime == "vllm" else SGLangAdapter
    adapter = adapter_type(allowed_images={f"{runtime}:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(
        tmp_path,
        runtime=runtime,
        method=method,
        runtime_method=runtime_method,
    )

    with pytest.raises(ValueError, match="does not match speculative method"):
        adapter.command(spec)


def test_sglang_command_rejects_unmapped_num_speculative_tokens(tmp_path):
    adapter = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(
        tmp_path,
        runtime="sglang",
        runtime_method="STANDALONE",
        num_speculative_tokens=5,
    )

    with pytest.raises(ValueError, match="num_speculative_tokens is not supported by SGLang"):
        adapter.command(spec)


def test_sglang_dspark_uses_repo_id_cache_root_and_dgx_spark_flags(tmp_path):
    adapter = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(
        tmp_path,
        runtime="sglang",
        method="dspark",
        runtime_method="DSPARK",
        draft_container_path=(
            "/draft-models/models--RadixArk--Qwen3.8-27B-DSpark/snapshots/abc123"
        ),
    )

    command = adapter.command(spec)

    assert command[command.index("--speculative-algorithm") + 1] == "DSPARK"
    assert command[command.index("--speculative-draft-model-path") + 1] == (
        "RadixArk/Qwen3.8-27B-DSpark"
    )
    assert command[command.index("--speculative-draft-model-quantization") + 1] == "unquant"
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert command[command.index("--mamba-ssm-dtype") + 1] == "float32"
    assert adapter.environment(spec)["HF_HUB_CACHE"] == "/draft-models"


def test_sglang_dflash_uses_model_card_block_size(tmp_path):
    adapter = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    spec = resolved_speculative_spec(
        tmp_path,
        runtime="sglang",
        method="dflash",
        runtime_method="DFLASH",
        draft_container_path="/draft-models/qwen38-dflash2",
        num_draft_tokens=8,
    )

    command = adapter.command(spec)

    assert command[command.index("--speculative-algorithm") + 1] == "DFLASH"
    assert command[command.index("--speculative-draft-model-path") + 1] == (
        "/draft-models/qwen38-dflash2"
    )
    assert command[command.index("--speculative-num-draft-tokens") + 1] == "8"
    assert "--speculative-num-steps" not in command
    assert "--speculative-eagle-topk" not in command


def test_runtime_commands_detect_model_specific_parsers_and_memory_flags(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["QwenMoeForCausalLM"],
                "layer_types": ["linear_attention", "full_attention"],
                "num_experts": 128,
            }
        ),
        encoding="utf-8",
    )
    (model_path / "hf_quant_config.json").write_text(
        '{"quantization":{"format":"NVFP4"}}', encoding="utf-8"
    )
    (model_path / "chat_template.jinja").write_text(
        "<think><tool_call><function=name><parameter=key>", encoding="utf-8"
    )
    common = {
        "name": "Qwen",
        "model_path": str(model_path),
        "api_model_name": "qwen",
        "chat_template_kwargs": {"enable_thinking": True},
    }

    sglang = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    sglang_command = sglang.command(
        DeploymentSpec(runtime="sglang", image="sglang:test", **common)
    )
    assert sglang_command[sglang_command.index("--max-mamba-cache-size") + 1] == "40"
    assert sglang_command[sglang_command.index("--moe-runner-backend") + 1] == (
        "flashinfer_cutlass"
    )
    assert sglang_command[sglang_command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert sglang_command[sglang_command.index("--reasoning-parser") + 1] == "qwen3"

    vllm = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    vllm_command = vllm.command(DeploymentSpec(runtime="vllm", image="vllm:test", **common))
    assert "--enable-auto-tool-choice" in vllm_command
    assert vllm_command[vllm_command.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert vllm_command[vllm_command.index("--reasoning-parser") + 1] == "qwen3"


def test_runtime_commands_without_speculative_config_remain_unchanged(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    recommendation = valid_recommendation_payload()
    recommendation["provider_id"] = None
    common = {
        "name": "Qwen",
        "model_path": str(model_path),
        "api_model_name": "qwen",
        "port": 8100,
        "generation_defaults": {"temperature": 0.2, "max_tokens": 256},
        "recommendation": recommendation,
    }

    vllm = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    vllm_spec = DeploymentSpec(runtime="vllm", image="vllm:test", **common)
    assert vllm.command(vllm_spec) == [
        "--model",
        "/models/qwen",
        "--served-model-name",
        "qwen",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.8",
        "--max-num-seqs",
        "8",
    ]

    sglang = SGLangAdapter(allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",))
    sglang_spec = DeploymentSpec(runtime="sglang", image="sglang:test", **common)
    assert sglang.command(sglang_spec) == [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "/models/qwen",
        "--served-model-name",
        "qwen",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--context-length",
        "32768",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "8",
        "--weight-loader-prefetch-num-threads",
        "20",
    ]


def test_create_endpoint_persists_json_safe_recommendation(authenticated_client, tmp_path):
    payload = valid_spec_payload(tmp_path)
    Path(payload["model_path"]).mkdir(parents=True)
    payload["model_id"] = None
    payload["image"] = "vllm/vllm-openai:v0.27.1"
    payload["speculative"] = None
    payload["recommendation"] = valid_recommendation_payload()
    with authenticated_client.app.state.database.session_factory() as db:
        target = add_model_asset(db, Path(payload["model_path"]))
        payload["model_id"] = target.id
    original_preview = authenticated_client.app.state.deployment_service.preview

    def preview(db, spec, **kwargs):
        assert db.get(ModelAsset, spec.model_id) is not None
        return {"spec": spec.model_dump(mode="json")}

    authenticated_client.app.state.deployment_service.preview = preview

    try:
        response = authenticated_client.post("/api/deployments", json=payload)
    finally:
        authenticated_client.app.state.deployment_service.preview = original_preview

    assert response.status_code == 202
    with authenticated_client.app.state.database.session_factory() as db:
        task = db.get(TaskRecord, response.json()["id"])
        assert task.input_json["recommendation"]["generated_at"] == ("2026-08-16T00:00:00Z")
        json.dumps(task.input_json)


def test_preview_endpoint_passes_database_and_deployment_exclusion(authenticated_client, tmp_path):
    payload = valid_spec_payload(tmp_path)
    payload["speculative"] = None
    payload["recommendation"] = None
    captured = {}
    original_preview = authenticated_client.app.state.deployment_service.preview

    def preview(db, spec, **kwargs):
        captured["db"] = db
        captured["spec"] = spec
        captured.update(kwargs)
        return {"spec": spec.model_dump(mode="json")}

    authenticated_client.app.state.deployment_service.preview = preview
    try:
        response = authenticated_client.post(
            "/api/deployments/preview?deployment_id=deployment-1", json=payload
        )
    finally:
        authenticated_client.app.state.deployment_service.preview = original_preview

    assert response.status_code == 200
    assert captured["db"] is not None
    assert captured["exclude_deployment_id"] == "deployment-1"


def test_create_handler_rechecks_resources_before_any_docker_call(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'handler-preflight.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base")
        base_id = base.id
    service = build_preflight_service(
        database,
        (root,),
        estimator=StaticEstimator("blocked"),
    )
    docker_calls = 0

    def docker_client():
        nonlocal docker_calls
        docker_calls += 1
        raise AssertionError("Docker must not be touched after a blocked preflight")

    service.docker_client = docker_client
    spec = DeploymentSpec(
        name="base",
        model_id=base_id,
        model_path=str(root / "base"),
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with pytest.raises(ValueError, match="Deployment is blocked"):
        service.create_handler(type("Context", (), {})(), spec.model_dump(mode="json"))

    assert docker_calls == 0


def test_create_handler_with_auto_port_rechecks_resources_before_docker_call(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'handler-auto-port-preflight.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base")
        base_id = base.id
    service = build_preflight_service(
        database,
        (root,),
        estimator=StaticEstimator("blocked"),
    )
    docker_calls = 0

    def docker_client():
        nonlocal docker_calls
        docker_calls += 1
        raise AssertionError("Docker must not be touched after a blocked preflight")

    service.docker_client = docker_client
    spec = DeploymentSpec(
        name="base-auto-port",
        model_id=base_id,
        model_path=str(root / "base"),
        api_model_name="base-auto-port",
        runtime="vllm",
        image="vllm:test",
    )

    with pytest.raises(ValueError, match="Deployment is blocked"):
        service.create_handler(type("Context", (), {})(), spec.model_dump(mode="json"))

    assert docker_calls == 0


@pytest.mark.parametrize(
    ("existing_name", "existing_api"),
    [("base", "other-api"), ("other-name", "base")],
)
def test_create_handler_rejects_partial_name_or_api_conflicts(
    tmp_path, existing_name, existing_api
):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'partial-conflict.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        db.add(
            Deployment(
                name=existing_name,
                model_id=target.id,
                runtime="vllm",
                container_id="existing-container",
                container_name="dgx-existing",
                endpoint_url="http://127.0.0.1:8100",
                api_model_name=existing_api,
                status="running",
                health="healthy",
                managed=True,
                image="vllm:test",
                port=8100,
                config={},
            )
        )
        db.commit()
        model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with pytest.raises(ValueError, match="conflicts with an existing deployment"):
        service.create_handler(HandlerContext(), spec.model_dump(mode="json"))


def test_create_handler_rejects_same_identity_with_different_spec(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'spec-conflict.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))
    original = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )
    with database.session_factory() as db:
        preview = service.preview(db, original)
        db.add(
            Deployment(
                name="base",
                model_id=model_id,
                runtime="vllm",
                container_id="existing-container",
                container_name="dgx-base",
                endpoint_url="http://127.0.0.1:8100",
                api_model_name="base",
                status="running",
                health="healthy",
                managed=True,
                image="vllm:test",
                port=8100,
                config=preview,
            )
        )
        db.commit()

    changed = original.model_copy(update={"port": 8101})
    with pytest.raises(ValueError, match="different deployment spec"):
        service.create_handler(HandlerContext(), changed.model_dump(mode="json"))


def test_create_handler_recovers_committed_target_before_blocked_live_preflight(
    tmp_path, monkeypatch
):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'create-committed-retry.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))
    original = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path=str(tmp_path / "browser-path-a"),
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )
    with database.session_factory() as db:
        preview = service.preview(db, original)
        deployment = Deployment(
            name="base",
            model_id=model_id,
            runtime="vllm",
            container_id="committed-container",
            container_name="stale-container-name",
            endpoint_url="http://127.0.0.1:9999",
            api_model_name="base",
            status="exited",
            health="unhealthy",
            managed=True,
            image="vllm:test",
            port=8100,
            config=preview,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    stored_spec = DeploymentSpec.model_validate(preview["spec"])
    labels = service._expected_container_labels(
        stored_spec,
        task_id="original-task",
        spec_fingerprint=legacy_spec_fingerprint(stored_spec),
    )

    class Container:
        id = "committed-container"
        name = "dgx-base"
        status = "running"
        attrs = {"Config": {"Labels": labels}}

        def start(self):
            self.status = "running"

    container = Container()
    containers = type("Containers", (), {"get": lambda _self, _identifier: container})()
    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": containers})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)
    blocked = StaticEstimator("blocked")
    service.resource_estimator = blocked
    retry = original.model_copy(update={"model_path": str(tmp_path / "browser-path-b")})

    result = service.create_handler(
        HandlerContext(task_id="retry-task"), retry.model_dump(mode="json")
    )

    assert result["deployment_id"] == deployment_id
    assert result["idempotent"] is True
    assert blocked.calls == []
    with database.session_factory() as db:
        recovered = db.get(Deployment, deployment_id)
        assert recovered.status == "running"
        assert recovered.health == "healthy"
        assert recovered.container_name == "dgx-base"
        assert recovered.endpoint_url == "http://127.0.0.1:8100"


@pytest.mark.parametrize("operation", ["create", "update"])
def test_committed_recovery_rejects_different_full_fingerprint(tmp_path, monkeypatch, operation):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / f'fingerprint-{operation}.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )
    with database.session_factory() as db:
        preview = service.preview(db, spec)
        deployment = Deployment(
            name=spec.name,
            model_id=model_id,
            runtime=spec.runtime,
            container_id="committed-container",
            container_name="dgx-base",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name=spec.api_model_name,
            status="running",
            health="healthy",
            managed=True,
            image=spec.image,
            port=spec.port,
            config=preview,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    different_spec = spec.model_copy(update={"context_length": 65536})
    labels = service._expected_container_labels(
        spec,
        task_id="original-task",
        spec_fingerprint=deployment_service.deployment_spec_fingerprint(different_spec),
        deployment_id=deployment_id if operation == "update" else None,
    )

    class Container:
        id = "committed-container"
        name = "dgx-base"
        status = "running"
        attrs = {"Config": {"Labels": labels}}

    container = Container()
    containers = type("Containers", (), {"get": lambda _self, _identifier: container})()
    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": containers})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)

    with pytest.raises(ValueError, match="container labels do not match"):
        if operation == "create":
            service.create_handler(HandlerContext(), spec.model_dump(mode="json"))
        else:
            service.update_handler(
                HandlerContext(),
                {"deployment_id": deployment_id, "spec": spec.model_dump(mode="json")},
            )


def test_create_handler_rejects_orphan_container_with_incomplete_labels(tmp_path, monkeypatch):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'orphan-labels.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))

    class ExistingContainer:
        id = "orphan"
        name = "dgx-base"
        status = "running"
        attrs = {"Config": {"Labels": {"com.dgx-spark-manager.managed": "true"}}}

        def reload(self):
            return None

    existing_container = ExistingContainer()
    containers = type("Containers", (), {"get": lambda _self, _name: existing_container})()
    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": containers})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with pytest.raises(ValueError, match="container labels do not match"):
        service.create_handler(HandlerContext(), spec.model_dump(mode="json"))

    with database.session_factory() as db:
        resolved, preview = service._preflight(db, spec)
    existing_container.attrs["Config"]["Labels"] = service._expected_container_labels(
        resolved,
        task_id="task-1",
        spec_fingerprint=preview["spec_fingerprint"],
    )
    result = service.create_handler(HandlerContext(), spec.model_dump(mode="json"))

    assert result["container_name"] == "dgx-base"


@pytest.mark.parametrize("failure_stage", ["cancel", "health", "commit"])
def test_create_handler_removes_new_container_for_all_pre_persist_failures(
    tmp_path, monkeypatch, failure_stage
):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / f'cleanup-{failure_stage}.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))

    class NewContainer:
        id = "new-container"
        name = "dgx-base"
        status = "running"
        attrs = {"Config": {"Labels": {}}}
        removed = False
        force = None

        def reload(self):
            return None

        def remove(self, **kwargs):
            self.removed = True
            self.force = kwargs.get("force")

        def logs(self, **_kwargs):
            return b"startup"

    container = NewContainer()

    class Containers:
        def get(self, _name):
            raise docker.errors.NotFound("missing")

        def run(self, _image, **_kwargs):
            return container

    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": Containers()})(),
    )
    context = HandlerContext(update_error=TaskCancelled() if failure_stage == "cancel" else None)
    if failure_stage == "health":
        monkeypatch.setattr(
            service,
            "wait_for_health",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("health probe failed")),
        )
    else:
        monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)
    if failure_stage == "commit":
        original_commit = Session.commit

        def fail_deployment_commit(session):
            if any(isinstance(value, Deployment) for value in session.new):
                raise RuntimeError("deployment commit failed")
            return original_commit(session)

        monkeypatch.setattr(Session, "commit", fail_deployment_commit)
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    expected = TaskCancelled if failure_stage == "cancel" else RuntimeError
    with pytest.raises(expected):
        service.create_handler(context, spec.model_dump(mode="json"))

    assert container.removed is True
    assert container.force is True


def test_update_handler_blocked_preflight_does_not_touch_old_container(tmp_path):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'update-preflight.db'}")
    database.create_schema()
    with database.session_factory() as db:
        base = add_model_asset(db, root / "base")
        deployment = Deployment(
            name="base",
            model_id=base.id,
            runtime="vllm",
            container_id="old-container",
            container_name="dgx-base",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="base",
            status="running",
            health="healthy",
            managed=True,
            image="vllm:test",
            port=8100,
        )
        db.add(deployment)
        db.commit()
        model_id, deployment_id = base.id, deployment.id
    service = build_preflight_service(
        database,
        (root,),
        estimator=StaticEstimator("blocked"),
    )
    docker_calls = 0

    def docker_client():
        nonlocal docker_calls
        docker_calls += 1
        raise AssertionError("Old container must remain untouched")

    service.docker_client = docker_client
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path=str(root / "base"),
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with pytest.raises(ValueError, match="Deployment is blocked"):
        service.update_handler(
            type("Context", (), {})(),
            {"deployment_id": deployment_id, "spec": spec.model_dump(mode="json")},
        )

    assert docker_calls == 0
    with database.session_factory() as db:
        unchanged = db.get(Deployment, deployment_id)
        assert unchanged.container_id == "old-container"
        assert unchanged.status == "running"


def test_update_recovers_committed_target_and_keeps_unmanaged_backup(tmp_path, monkeypatch):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'update-committed-retry.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path=str(tmp_path / "browser-path-a"),
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )
    with database.session_factory() as db:
        preview = service.preview(db, spec)
        deployment = Deployment(
            name=spec.name,
            model_id=model_id,
            runtime=spec.runtime,
            container_id="replacement-container",
            container_name="stale-container-name",
            endpoint_url="http://127.0.0.1:9999",
            api_model_name=spec.api_model_name,
            status="exited",
            health="unhealthy",
            managed=True,
            image=spec.image,
            port=spec.port,
            config=preview,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    stored_spec = DeploymentSpec.model_validate(preview["spec"])
    labels = service._expected_container_labels(
        stored_spec,
        task_id="original-task",
        spec_fingerprint=deployment_service.deployment_spec_fingerprint(stored_spec),
        deployment_id=deployment_id,
    )
    labels["com.dgx-spark-manager.replaces-container-id"] = "old-container"

    class Replacement:
        id = "replacement-container"
        name = "actual-container-name"
        status = "exited"
        attrs = {"Config": {"Labels": labels}}

        def start(self):
            self.status = "running"

    class UnmanagedBackup:
        id = "attacker-container"
        name = service._backup_container_name(deployment_id)
        attrs = {"Config": {"Labels": {}}}
        removed = False

        def remove(self, **_kwargs):
            self.removed = True

    replacement = Replacement()
    backup = UnmanagedBackup()

    class Containers:
        def get(self, identifier):
            if identifier == "replacement-container":
                return replacement
            if identifier == backup.name:
                return backup
            raise docker.errors.NotFound("missing")

    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": Containers()})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)
    blocked = StaticEstimator("blocked")
    service.resource_estimator = blocked
    context = HandlerContext(task_id="retry-task")
    retry = spec.model_copy(update={"model_path": str(tmp_path / "browser-path-b")})

    result = service.update_handler(
        context,
        {"deployment_id": deployment_id, "spec": retry.model_dump(mode="json")},
    )

    assert result["idempotent"] is True
    assert blocked.calls == []
    assert backup.removed is False
    assert any("backup cleanup conflict" in message for message in context.messages)
    assert replacement.status == "running"
    with database.session_factory() as db:
        recovered = db.get(Deployment, deployment_id)
        assert recovered.status == "running"
        assert recovered.health == "healthy"
        assert recovered.container_name == "actual-container-name"
        assert recovered.endpoint_url == "http://127.0.0.1:8100"


@pytest.mark.parametrize("operation", ["create", "update"])
def test_committed_recovery_rejects_wrong_deployment_label(tmp_path, monkeypatch, operation):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'wrong-deployment-label.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )
    with database.session_factory() as db:
        preview = service.preview(db, spec)
        deployment = Deployment(
            name=spec.name,
            model_id=model_id,
            runtime=spec.runtime,
            container_id="replacement-container",
            container_name="dgx-base",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name=spec.api_model_name,
            status="running",
            health="healthy",
            managed=True,
            image=spec.image,
            port=spec.port,
            config=preview,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    labels = service._expected_container_labels(
        spec,
        task_id="original-task",
        spec_fingerprint=deployment_service.deployment_spec_fingerprint(spec),
        deployment_id="another-deployment",
    )

    class Container:
        id = "replacement-container"
        name = "dgx-base"
        status = "running"
        attrs = {"Config": {"Labels": labels}}

    container = Container()
    containers = type("Containers", (), {"get": lambda _self, _identifier: container})()
    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": containers})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)

    with pytest.raises(ValueError, match="container labels do not match"):
        if operation == "create":
            service.create_handler(HandlerContext(), spec.model_dump(mode="json"))
        else:
            service.update_handler(
                HandlerContext(),
                {"deployment_id": deployment_id, "spec": spec.model_dump(mode="json")},
            )


@pytest.mark.parametrize(
    ("db_status", "db_health", "expected_stops"),
    [("exited", "unhealthy", 1), ("running", "healthy", 0)],
)
def test_committed_recovery_cas_rejects_race_and_coordinates_started_container(
    tmp_path, monkeypatch, db_status, db_health, expected_stops
):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'concurrent-recovery.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )
    with database.session_factory() as db:
        preview = service.preview(db, spec)
        deployment = Deployment(
            name=spec.name,
            model_id=model_id,
            runtime=spec.runtime,
            container_id="replacement-container",
            container_name="dgx-base",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name=spec.api_model_name,
            status=db_status,
            health=db_health,
            managed=True,
            image=spec.image,
            port=spec.port,
            config=preview,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    labels = service._expected_container_labels(
        spec,
        task_id="original-task",
        spec_fingerprint=deployment_service.deployment_spec_fingerprint(spec),
        deployment_id=deployment_id,
    )

    class Container:
        id = "replacement-container"
        name = "dgx-base"
        status = "exited"
        attrs = {"Config": {"Labels": labels}}
        starts = 0
        stops = 0

        def start(self):
            self.starts += 1
            self.status = "running"

        def stop(self, **_kwargs):
            self.stops += 1
            self.status = "exited"

    container = Container()

    class Containers:
        def get(self, identifier):
            if identifier == container.id:
                return container
            raise docker.errors.NotFound("missing")

    recovery_updates = []

    def capture_update(_connection, _cursor, statement, _params, _context, _many):
        if statement.lstrip().upper().startswith("UPDATE DEPLOYMENTS"):
            recovery_updates.append(statement)

    original_session_factory = service.session_factory
    session_calls = 0

    class RacingSessionFactory:
        def __call__(self):
            nonlocal session_calls
            session_calls += 1
            if session_calls == 3:
                with original_session_factory() as concurrent_db:
                    changed = concurrent_db.get(Deployment, deployment_id)
                    changed.status = "concurrent"
                    concurrent_db.commit()
            return original_session_factory()

    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": Containers()})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "session_factory", RacingSessionFactory())
    event.listen(database.engine, "before_cursor_execute", capture_update)

    try:
        with pytest.raises(ValueError, match="changed while recovery was running"):
            service.update_handler(
                HandlerContext(),
                {
                    "deployment_id": deployment_id,
                    "spec": spec.model_dump(mode="json"),
                },
            )
    finally:
        event.remove(database.engine, "before_cursor_execute", capture_update)

    assert container.starts == 1
    assert container.stops == expected_stops
    assert len(recovery_updates) == 2
    normalized_update = " ".join(recovery_updates[-1].lower().split())
    assert " where " in normalized_update
    where_clause = normalized_update.split(" where ", 1)[1]
    for field in (
        "updated_at",
        "container_id",
        "status",
        "health",
        "container_name",
        "endpoint_url",
    ):
        assert field in where_clause
    with database.session_factory() as db:
        changed = db.get(Deployment, deployment_id)
        assert changed.container_id == "replacement-container"
        assert changed.status == "concurrent"
        assert changed.health == db_health


def test_committed_recovery_health_failure_stops_container_started_by_retry(tmp_path, monkeypatch):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'recovery-health-failure.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        model_id = target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )
    with database.session_factory() as db:
        preview = service.preview(db, spec)
        deployment = Deployment(
            name=spec.name,
            model_id=model_id,
            runtime=spec.runtime,
            container_id="replacement-container",
            container_name="dgx-base",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name=spec.api_model_name,
            status="exited",
            health="unhealthy",
            managed=True,
            image=spec.image,
            port=spec.port,
            config=preview,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    labels = service._expected_container_labels(
        spec,
        task_id="original-task",
        spec_fingerprint=deployment_service.deployment_spec_fingerprint(spec),
        deployment_id=deployment_id,
    )

    class Container:
        id = "replacement-container"
        name = "dgx-base"
        status = "exited"
        attrs = {"Config": {"Labels": labels}}
        starts = 0
        stops = 0

        def start(self):
            self.starts += 1
            self.status = "running"

        def stop(self, **_kwargs):
            self.stops += 1
            self.status = "exited"

    container = Container()
    containers = type("Containers", (), {"get": lambda _self, _identifier: container})()
    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": containers})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="replacement container is unhealthy"):
        service.update_handler(
            HandlerContext(),
            {"deployment_id": deployment_id, "spec": spec.model_dump(mode="json")},
        )

    assert container.starts == 1
    assert container.stops == 1
    assert container.status == "exited"


def test_create_app_shares_preflight_dependencies(settings):
    app = create_app(settings)
    deployments = app.state.deployment_service
    recommendations = app.state.deployment_recommendation_service

    assert deployments.runtime_capability_service is recommendations.runtime_capability_service
    assert deployments.evidence_loader is recommendations.evidence_loader
    assert deployments.resource_estimator is recommendations.resource_estimator
    assert deployments.draft_service is recommendations.draft_service
    assert deployments._docker_client is recommendations.runtime_capability_service.docker_client


def test_update_endpoint_queues_managed_deployment_change(authenticated_client, tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"architectures":["Qwen2ForCausalLM"]}')
    (model_path / "model.safetensors").write_bytes(b"weights")
    with authenticated_client.app.state.database.session_factory() as db:
        target = add_model_asset(db, model_path)
        deployment = Deployment(
            name="managed",
            model_id=target.id,
            runtime="vllm",
            container_id="container-id",
            container_name="dgx-managed",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="managed",
            status="running",
            health="healthy",
            managed=True,
            image="vllm/vllm-openai:v0.27.1",
            port=8100,
        )
        db.add(deployment)
        db.commit()
        deployment_id, model_id = deployment.id, target.id

    original_preview = authenticated_client.app.state.deployment_service.preview
    authenticated_client.app.state.deployment_service.preview = lambda _db, spec, **_kwargs: {
        "spec": spec.model_dump(mode="json")
    }

    try:
        response = authenticated_client.patch(
            f"/api/deployments/{deployment_id}",
            json={
                "name": "managed",
                "model_id": model_id,
                "model_path": str(model_path),
                "api_model_name": "managed",
                "runtime": "vllm",
                "image": "vllm/vllm-openai:v0.27.1",
                "port": 8100,
                "recommendation": valid_recommendation_payload(),
            },
        )
    finally:
        authenticated_client.app.state.deployment_service.preview = original_preview

    assert response.status_code == 202
    assert response.json()["type"] == "deployment.update"
    with authenticated_client.app.state.database.session_factory() as db:
        task = db.get(TaskRecord, response.json()["id"])
        assert task.input_json["spec"]["recommendation"]["generated_at"] == ("2026-08-16T00:00:00Z")
        json.dumps(task.input_json)


def test_container_model_path_maps_to_the_configured_host_root(tmp_path):
    container_root = tmp_path / "manager" / "hf-cache" / "hub"
    model_path = container_root / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / "abc"
    model_path.mkdir(parents=True)
    host_root = Path("/home/operator/.cache/huggingface/hub")

    resolver = getattr(deployment_service, "resolve_host_model_mount", lambda *_: None)

    assert resolver(model_path, (container_root,), (host_root,)) == host_root


def test_settings_preserve_container_to_host_model_root_order(tmp_path):
    container_models = tmp_path / "container-models"
    container_hf = tmp_path / "container-hf"
    host_models = Path("/srv/models")
    host_hf = Path("/home/operator/.cache/huggingface/hub")
    settings = Settings(
        secret_key="test-secret-key-with-at-least-32-characters",
        admin_password="Test-password-1234",
        model_roots=f"{container_models},{container_hf}",
        model_root_mappings=(f"{container_models}={host_models};{container_hf}={host_hf}"),
    )

    assert getattr(settings, "host_model_root_paths", ()) == (host_models, host_hf)
    assert getattr(settings, "deployment_startup_timeout_seconds", None) == 1200


def test_port_allocator_uses_db_and_docker_ports_and_reuses_lowest_gap(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'ports.db'}")
    database.create_schema()

    class Container:
        id = "external-container"
        attrs = {"HostConfig": {"PortBindings": {"8000/tcp": [{"HostPort": "8003"}]}}}

    class Containers:
        def list(self, all=True):
            return [Container()]

    docker_client = type("Client", (), {"containers": Containers()})()
    service = deployment_service.DeploymentService(
        adapters={},
        session_factory=database.session_factory,
        model_roots=(tmp_path,),
        docker_client=docker_client,
    )
    with database.session_factory() as db:
        db.add_all(
            [
                Deployment(
                    name="one",
                    runtime="vllm",
                    endpoint_url="http://one",
                    api_model_name="one",
                    port=8000,
                ),
                Deployment(
                    name="three",
                    runtime="vllm",
                    endpoint_url="http://three",
                    api_model_name="three",
                    port=8002,
                ),
            ]
        )
        db.commit()
        assert service._allocate_deployment_port(db) == 8001
        db.query(Deployment).filter(Deployment.name == "one").delete()
        db.commit()
        assert service._allocate_deployment_port(db) == 8000


def test_deployment_spec_can_omit_port_for_service_allocation():
    spec = DeploymentSpec(
        name="automatic-port",
        model_path="/models/automatic-port",
        api_model_name="automatic-port",
        runtime="vllm",
        image="vllm:test",
    )

    assert spec.port is None


def test_concurrent_port_allocations_are_serialized_and_unique(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'concurrent-ports.db'}")
    database.create_schema()
    service = deployment_service.DeploymentService(
        adapters={},
        session_factory=database.session_factory,
        model_roots=(tmp_path,),
        docker_client=type(
            "Client",
            (),
            {"containers": type("Containers", (), {"list": lambda self, all=True: []})()},
        )(),
    )
    results: list[int] = []

    def create(index: int):
        with service._port_allocation_lock:
            with database.session_factory() as db:
                port = service._allocate_deployment_port(db)
                db.add(
                    Deployment(
                        name=f"parallel-{index}",
                        runtime="vllm",
                        endpoint_url=f"http://parallel-{index}",
                        api_model_name=f"parallel-{index}",
                        port=port,
                    )
                )
                db.commit()
                results.append(port)

    threads = [threading.Thread(target=create, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [8000, 8001]


def test_deployment_service_mounts_the_host_model_root(tmp_path, monkeypatch):
    container_root = tmp_path / "manager-hf"
    model_path = container_root / "models--org--model" / "snapshots" / "abc"
    model_path.mkdir(parents=True)
    host_root = Path("/home/operator/.cache/huggingface/hub")
    database = Database(f"sqlite:///{tmp_path / 'manager.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, model_path)
        model_id = target.id
    adapter = VllmAdapter(
        allowed_images={"vllm/vllm-openai:v0.27.1"}, model_roots=(container_root,)
    )
    service = deployment_service.DeploymentService(
        adapters={"vllm": adapter},
        session_factory=database.session_factory,
        model_roots=(container_root,),
        host_model_roots=(host_root,),
        runtime_capability_service=StaticRuntimeCapabilities(),
        evidence_loader=ModelEvidenceLoader(),
        resource_estimator=ResourceEstimator(),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 64 * 1024**3, "available_bytes": 64 * 1024**3}
        },
    )
    captured: dict = {}

    class FakeContainer:
        id = "container-id"
        name = "dgx-test-model"
        status = "running"
        attrs = {"Config": {"Labels": {"com.dgx-spark-manager.managed": "true"}}}

        def reload(self):
            return None

        def stop(self, **_kwargs):
            return None

        def remove(self):
            return None

    class FakeContainers:
        def get(self, _name):
            raise docker.errors.NotFound("missing")

        def run(self, _image, **kwargs):
            captured.update(kwargs)
            return FakeContainer()

    class Context:
        def update(self, **_kwargs):
            return None

        def check_control(self):
            return None

    client_type = type("Client", (), {"containers": FakeContainers()})
    monkeypatch.setattr(service, "docker_client", lambda: client_type())
    monkeypatch.setattr(
        deployment_service.httpx,
        "get",
        lambda *_args, **_kwargs: type("Response", (), {"is_success": True})(),
    )

    service.create_handler(
        Context(),
        DeploymentSpec(
            name="Test Model",
            model_id=model_id,
            model_path=str(model_path),
            api_model_name="test-model",
            runtime="vllm",
            image="vllm/vllm-openai:v0.27.1",
            port=8100,
        ).model_dump(),
    )

    assert captured["volumes"] == {str(host_root): {"bind": "/models", "mode": "ro"}}
    assert captured["log_config"] == {
        "Type": "json-file",
        "Config": {"max-size": "10m", "max-file": "5"},
    }
    labels = captured["labels"]
    assert labels["com.dgx-spark-manager.task-id"] == "manual"
    assert labels["com.dgx-spark-manager.runtime"] == "vllm"
    assert labels["com.dgx-spark-manager.model-id"] == model_id
    assert labels["com.dgx-spark-manager.route"] == "test-model"
    assert labels["com.dgx-spark-manager.image"] == "vllm/vllm-openai:v0.27.1"
    assert labels["com.dgx-spark-manager.port"] == "8100"
    assert len(labels["com.dgx-spark-manager.spec-fingerprint"]) == 64
    assert all(str(model_path) not in value for value in labels.values())


def test_deployment_timeout_captures_logs_and_rolls_back_new_container(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_path = model_root / "model"
    model_path.mkdir(parents=True)
    database = Database(f"sqlite:///{tmp_path / 'timeout.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, model_path)
        model_id = target.id
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(model_root,))
    service = deployment_service.DeploymentService(
        adapters={"vllm": adapter},
        session_factory=database.session_factory,
        model_roots=(model_root,),
        startup_timeout_seconds=2,
        runtime_capability_service=StaticRuntimeCapabilities(),
        evidence_loader=ModelEvidenceLoader(),
        resource_estimator=ResourceEstimator(),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 64 * 1024**3, "available_bytes": 64 * 1024**3}
        },
    )

    class FakeContainer:
        id = "new-container"
        name = "dgx-timeout"
        status = "running"
        stopped = False
        removed = False
        remove_force = None

        def logs(self, **_kwargs):
            return b"engine failed while allocating memory"

        def stop(self, **_kwargs):
            self.stopped = True

        def remove(self, **kwargs):
            self.removed = True
            self.remove_force = kwargs.get("force")

    container = FakeContainer()

    class FakeContainers:
        def get(self, _name):
            raise docker.errors.NotFound("missing")

        def run(self, _image, **_kwargs):
            return container

    class Context:
        def update(self, **_kwargs):
            return None

        def check_control(self):
            return None

    client_type = type("Client", (), {"containers": FakeContainers()})
    monkeypatch.setattr(service, "docker_client", lambda: client_type())
    response_type = type("Response", (), {"is_success": False})
    monkeypatch.setattr(
        deployment_service.httpx,
        "get",
        lambda *_args, **_kwargs: response_type(),
    )
    monkeypatch.setattr(deployment_service.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="engine failed while allocating memory"):
        service.create_handler(
            Context(),
            DeploymentSpec(
                name="Timeout",
                model_id=model_id,
                model_path=str(model_path),
                api_model_name="timeout",
                runtime="vllm",
                image="vllm:test",
                port=8100,
            ).model_dump(),
        )

    assert container.removed is True
    assert container.remove_force is True


def test_queued_delete_rejects_deployment_rebound_to_same_named_container(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'queued-rebound.db'}")
    database.create_schema()
    container_a_id = "3a" * 32
    container_b = ActionContainer("3b" * 32, "shared-name")
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name="discovered",
            container_id=container_a_id,
            container_name=container_b.name,
        )
        deployment_id = deployment.id
        deployment.container_id = container_b.id
        db.commit()
    service, _, get_calls = build_action_service(database, tmp_path, container_b)

    with pytest.raises(ValueError, match="identity changed concurrently"):
        service.action_handler(
            HandlerContext(),
            {
                "deployment_id": deployment_id,
                "action": "delete",
                "expected_container_id": container_a_id,
                "expected_container_name": container_b.name,
            },
        )

    assert get_calls == []
    assert container_b.removed is False
    with database.session_factory() as db:
        current = db.get(Deployment, deployment_id)
        assert current.container_id == container_b.id


def test_legacy_delete_without_snapshot_fails_closed_after_rebind(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'legacy-delete-rebound.db'}")
    database.create_schema()
    container_b = ActionContainer("3c" * 32, "shared-name")
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name="discovered",
            container_id=container_b.id,
            container_name=container_b.name,
        )
        deployment_id = deployment.id
    service, _, get_calls = build_action_service(database, tmp_path, container_b)

    with pytest.raises(ValueError, match="missing its container snapshot; retry"):
        service.action_handler(
            HandlerContext(),
            {"deployment_id": deployment_id, "action": "delete"},
        )

    assert get_calls == []
    assert container_b.removed is False
    with database.session_factory() as db:
        current = db.get(Deployment, deployment_id)
        assert current is not None
        assert current.container_id == container_b.id
        assert current.container_name == container_b.name
        assert current.status == "running"
        assert current.health == "healthy"


def test_not_found_cleanup_cas_preserves_concurrently_rebound_deployment(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'not-found-rebound.db'}")
    database.create_schema()
    container_a_id = "4a" * 32
    container_b_id = "4b" * 32
    container_name = "shared-name"
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name="discovered",
            container_id=container_a_id,
            container_name=container_name,
        )
        deployment_id = deployment.id

    def rebind_then_report_missing(_identifier):
        with database.session_factory() as db:
            current = db.get(Deployment, deployment_id)
            current.container_id = container_b_id
            current.status = "running"
            current.health = "healthy"
            db.commit()
        raise docker.errors.NotFound("container A disappeared")

    service, _, _ = build_action_service(database, tmp_path, rebind_then_report_missing)

    with pytest.raises(ValueError, match="identity changed concurrently"):
        service.action_handler(
            HandlerContext(),
            {
                "deployment_id": deployment_id,
                "action": "delete",
                "expected_container_id": container_a_id,
                "expected_container_name": container_name,
            },
        )

    with database.session_factory() as db:
        current = db.get(Deployment, deployment_id)
        assert current is not None
        assert current.container_id == container_b_id
        assert current.status == "running"
        assert current.health == "healthy"


def test_failed_action_recovery_cas_does_not_pollute_rebound_deployment(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'failed-rebound.db'}")
    database.create_schema()
    container_a = ActionContainer("5a" * 32, "shared-name")
    container_b_id = "5b" * 32
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=container_a.id,
            container_name=container_a.name,
        )
        deployment_id = deployment.id
    service, adapter, _ = build_action_service(database, tmp_path, container_a)

    def rebind_then_fail(_container, **_kwargs):
        with database.session_factory() as db:
            current = db.get(Deployment, deployment_id)
            current.container_id = container_b_id
            current.status = "running"
            current.health = "healthy"
            db.commit()
        raise RuntimeError("container A stop failed")

    monkeypatch.setattr(adapter, "stop", rebind_then_fail)
    context = HandlerContext()

    with pytest.raises(RuntimeError, match="container A stop failed"):
        service.action_handler(
            context,
            {
                "deployment_id": deployment_id,
                "action": "stop",
                "expected_container_id": container_a.id,
                "expected_container_name": container_a.name,
            },
        )

    with database.session_factory() as db:
        current = db.get(Deployment, deployment_id)
        assert current.container_id == container_b_id
        assert current.status == "running"
        assert current.health == "healthy"
    assert "identity changed concurrently" in "\n".join(context.messages)


def test_successful_delete_cas_does_not_delete_rebound_deployment(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'success-rebound.db'}")
    database.create_schema()
    container_a = ActionContainer("6a" * 32, "shared-name")
    container_b_id = "6b" * 32
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=container_a.id,
            container_name=container_a.name,
        )
        deployment_id = deployment.id
    service, adapter, _ = build_action_service(database, tmp_path, container_a)

    def uninstall_a_then_rebind(_container):
        container_a.removed = True
        with database.session_factory() as db:
            current = db.get(Deployment, deployment_id)
            current.container_id = container_b_id
            current.status = "running"
            current.health = "healthy"
            db.commit()

    monkeypatch.setattr(adapter, "uninstall", uninstall_a_then_rebind)

    with pytest.raises(ValueError, match="identity changed concurrently"):
        service.action_handler(
            HandlerContext(),
            {
                "deployment_id": deployment_id,
                "action": "delete",
                "expected_container_id": container_a.id,
                "expected_container_name": container_a.name,
            },
        )

    assert container_a.removed is True
    with database.session_factory() as db:
        current = db.get(Deployment, deployment_id)
        assert current is not None
        assert current.container_id == container_b_id
        assert current.status == "running"
        assert current.health == "healthy"


@pytest.mark.parametrize(
    ("action", "transition_status", "adapter_method"),
    [("stop", "stopping", "stop"), ("delete", "deleting", "uninstall")],
)
def test_destructive_action_commits_transition_before_adapter_call(
    tmp_path, monkeypatch, action, transition_status, adapter_method
):
    database = Database(f"sqlite:///{tmp_path / f'{action}-transition.db'}")
    database.create_schema()
    container = ActionContainer("a" * 64, "dgx-managed")
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=container.id,
            container_name=container.name,
        )
        deployment_id = deployment.id
    service, adapter, _ = build_action_service(database, tmp_path, container)
    observed = {}

    def operation(_container, **_kwargs):
        with database.session_factory() as db:
            current = db.get(Deployment, deployment_id)
            observed.update(status=current.status, health=current.health)
        if action == "delete":
            container.removed = True
        else:
            container.status = "exited"

    monkeypatch.setattr(adapter, adapter_method, operation)

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            action,
            container_id=container.id,
            container_name=container.name,
        ),
    )

    assert observed == {"status": transition_status, "health": "unknown"}
    assert result["container_missing"] is False


def test_discovered_uninstall_removes_container_record_but_keeps_model(
    tmp_path,
):
    database = Database(f"sqlite:///{tmp_path / 'discovered-uninstall.db'}")
    database.create_schema()
    model_path = tmp_path / "models" / "base"
    container = ActionContainer("b" * 64, "external-inference")
    with database.session_factory() as db:
        model = add_model_asset(db, model_path)
        deployment = add_action_deployment(
            db,
            managed=False,
            name="discovered",
            model_id=model.id,
            container_id=container.id,
            container_name=container.name,
        )
        plan = OperationPlan(
            deployment_id=deployment.id,
            summary="Referenced deployment",
            diagnosis="Uninstall should preserve plan",
        )
        db.add(plan)
        db.commit()
        deployment_id, model_id = deployment.id, model.id
        plan_id = plan.id
    service, _, _ = build_action_service(database, tmp_path, container)

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "delete",
            container_id=container.id,
            container_name=container.name,
        ),
    )

    assert result == {
        "deployment_id": deployment_id,
        "action": "delete",
        "status": "deleted",
        "health": "unknown",
        "container_missing": False,
    }
    assert container.stops == 1
    assert container.removed is True
    with database.session_factory() as db:
        assert db.get(Deployment, deployment_id) is None
        assert db.get(ModelAsset, model_id) is not None
        assert db.get(OperationPlan, plan_id).deployment_id is None
    assert model_path.is_dir()
    assert (model_path / "model.safetensors").is_file()


@pytest.mark.parametrize(
    ("actual_id", "actual_name"),
    [("b" * 64, "dgx-managed"), ("a" * 64, "renamed-container")],
)
def test_action_rejects_changed_container_identity(tmp_path, actual_id, actual_name):
    database = Database(f"sqlite:///{tmp_path / f'identity-{actual_name}.db'}")
    database.create_schema()
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id="a" * 64,
            container_name="dgx-managed",
        )
        deployment_id = deployment.id
    container = ActionContainer(actual_id, actual_name)
    service, _, _ = build_action_service(database, tmp_path, container)

    with pytest.raises(ValueError, match="identity does not match"):
        service.action_handler(
            HandlerContext(),
            action_handler_payload(
                deployment_id,
                "delete",
                container_id="a" * 64,
                container_name="dgx-managed",
            ),
        )

    assert container.removed is False
    with database.session_factory() as db:
        current = db.get(Deployment, deployment_id)
        assert current is not None
        assert current.status != "deleting"


@pytest.mark.parametrize("container_name", ["dgx-spark-web-manager", "dgx-spark-ops-agent"])
def test_action_rejects_reserved_manager_containers(tmp_path, container_name):
    database = Database(f"sqlite:///{tmp_path / f'{container_name}.db'}")
    database.create_schema()
    container = ActionContainer("c" * 64, container_name)
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name=f"record-{container_name}",
            container_id=container.id,
            container_name=container_name,
        )
        deployment_id = deployment.id
    service, _, _ = build_action_service(database, tmp_path, container)

    with pytest.raises(ValueError, match="protected manager container"):
        service.action_handler(
            HandlerContext(),
            action_handler_payload(
                deployment_id,
                "delete",
                container_id=container.id,
                container_name=container_name,
            ),
        )

    assert container.removed is False


def test_action_rejects_current_manager_container_id(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'manager-self.db'}")
    database.create_schema()
    container = ActionContainer("d" * 64, "tampered-record")
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name="tampered",
            container_id=container.id,
            container_name=container.name,
        )
        deployment_id = deployment.id
    service, _, _ = build_action_service(database, tmp_path, container)
    monkeypatch.setenv("HOSTNAME", container.id[:12])

    with pytest.raises(ValueError, match="protected manager container"):
        service.action_handler(
            HandlerContext(),
            action_handler_payload(
                deployment_id,
                "delete",
                container_id=container.id,
                container_name=container.name,
            ),
        )

    assert container.removed is False


@pytest.mark.parametrize("missing_kind", ["no_id", "not_found"])
def test_delete_clears_stale_deployment_record(tmp_path, missing_kind):
    database = Database(f"sqlite:///{tmp_path / f'stale-{missing_kind}.db'}")
    database.create_schema()
    target = docker.errors.NotFound("missing")
    container_id = None if missing_kind == "no_id" else "e" * 64
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name=f"stale-{missing_kind}",
            container_id=container_id,
            container_name=f"stale-{missing_kind}",
        )
        deployment_id = deployment.id
    service, _, get_calls = build_action_service(database, tmp_path, target)

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "delete",
            container_id=container_id,
            container_name=f"stale-{missing_kind}",
        ),
    )

    assert result["status"] == "deleted"
    assert result["container_missing"] is True
    assert get_calls == ([] if missing_kind == "no_id" else [container_id])
    with database.session_factory() as db:
        assert db.get(Deployment, deployment_id) is None


@pytest.mark.parametrize("container_id", [None, "2b" * 32])
def test_stale_delete_does_not_require_a_supported_runtime(tmp_path, container_id):
    database = Database(f"sqlite:///{tmp_path / f'stale-runtime-{container_id}.db'}")
    database.create_schema()
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name=f"stale-runtime-{container_id}",
            container_id=container_id,
            container_name="stale-runtime",
        )
        deployment.runtime = "removed-runtime"
        db.commit()
        deployment_id = deployment.id
    service, _, _ = build_action_service(
        database,
        tmp_path,
        docker.errors.NotFound("missing"),
    )

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "delete",
            container_id=container_id,
            container_name="stale-runtime",
        ),
    )

    assert result["container_missing"] is True
    with database.session_factory() as db:
        assert db.get(Deployment, deployment_id) is None


def test_non_delete_action_with_no_container_id_reports_missing(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'missing-stop.db'}")
    database.create_schema()
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=False,
            name="missing-stop",
            container_id=None,
            container_name="missing-stop",
        )
        deployment_id = deployment.id
    service, _, get_calls = build_action_service(
        database,
        tmp_path,
        ActionContainer("f" * 64, "missing-stop"),
    )

    with pytest.raises(ValueError, match="Deployment or container was not found"):
        service.action_handler(
            HandlerContext(),
            action_handler_payload(
                deployment_id,
                "stop",
                container_id=None,
                container_name="missing-stop",
            ),
        )

    assert get_calls == []
    with database.session_factory() as db:
        current = db.get(Deployment, deployment_id)
        assert current.status == "running"
        assert current.health == "healthy"


@pytest.mark.parametrize(
    ("action", "container_status", "expected_status", "adapter_method"),
    [
        ("stop", "running", "running", "stop"),
        ("delete", "exited", "exited", "uninstall"),
        ("delete", "paused", "unknown", "uninstall"),
    ],
)
def test_failed_destructive_action_restores_non_transitional_state(
    tmp_path,
    monkeypatch,
    action,
    container_status,
    expected_status,
    adapter_method,
):
    database = Database(f"sqlite:///{tmp_path / f'failed-{action}.db'}")
    database.create_schema()
    container = ActionContainer("f" * 64, "dgx-managed")
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=container.id,
            container_name=container.name,
        )
        deployment_id = deployment.id
    service, adapter, _ = build_action_service(database, tmp_path, container)

    def fail(_container, **_kwargs):
        container.status = container_status
        raise RuntimeError("docker operation failed")

    monkeypatch.setattr(adapter, adapter_method, fail)

    with pytest.raises(RuntimeError, match="docker operation failed"):
        service.action_handler(
            HandlerContext(),
            action_handler_payload(
                deployment_id,
                action,
                container_id=container.id,
                container_name=container.name,
            ),
        )

    with database.session_factory() as db:
        current = db.get(Deployment, deployment_id)
        assert current is not None
        assert current.status == expected_status
        assert current.health == "unhealthy"


def test_recovery_failure_does_not_replace_docker_error(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'recovery-error.db'}")
    database.create_schema()
    container = ActionContainer("2a" * 32, "dgx-managed")
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=container.id,
            container_name=container.name,
        )
        deployment_id = deployment.id
    service, adapter, _ = build_action_service(database, tmp_path, container)

    def fail_stop(_container, **_kwargs):
        raise RuntimeError("original docker error")

    def fail_recovery(*_args, **_kwargs):
        raise RuntimeError("recovery error")

    monkeypatch.setattr(adapter, "stop", fail_stop)
    monkeypatch.setattr(service, "_recover_action_failure", fail_recovery)

    with pytest.raises(RuntimeError, match="original docker error"):
        service.action_handler(
            HandlerContext(),
            action_handler_payload(
                deployment_id,
                "stop",
                container_id=container.id,
                container_name=container.name,
            ),
        )


def test_action_accepts_docker_short_id_and_normalized_name(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'short-id.db'}")
    database.create_schema()
    full_id = "1a" * 32
    container = ActionContainer(full_id, "/dgx-managed", status="exited")
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=full_id[:12],
            container_name="dgx-managed",
            status="exited",
        )
        deployment_id = deployment.id
    service, _, _ = build_action_service(database, tmp_path, container)

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "delete",
            container_id=full_id[:12],
            container_name="dgx-managed",
        ),
    )

    assert result["status"] == "deleted"
    assert container.removed is True


def test_restart_action_preserves_identity_and_reports_container_present(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'restart.db'}")
    database.create_schema()
    container = ActionContainer("1b" * 32, "dgx-managed", status="running")
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=container.id,
            container_name=container.name,
        )
        deployment_id = deployment.id
    service, _, _ = build_action_service(database, tmp_path, container)
    service.wait_for_health = lambda *_args, **_kwargs: True

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "restart",
            container_id=container.id,
            container_name=container.name,
        ),
    )

    assert container.restarts == 1
    assert result["status"] == "running"
    assert result["health"] == "healthy"
    assert result["container_missing"] is False


def test_start_action_waits_for_real_runtime_health(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'action.db'}")
    database.create_schema()
    with database.session_factory() as db:
        deployment = Deployment(
            name="managed",
            runtime="vllm",
            container_id="container-id",
            container_name="dgx-managed",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="managed",
            status="exited",
            health="unknown",
            managed=True,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    service = deployment_service.DeploymentService(
        adapters={"vllm": VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path,))},
        session_factory=database.session_factory,
        model_roots=(tmp_path,),
        startup_timeout_seconds=4,
    )

    class FakeContainer:
        id = "container-id"
        name = "dgx-managed"
        starts = 0

        def reload(self):
            return None

        def start(self):
            self.starts += 1

    container = FakeContainer()
    containers = type("Containers", (), {"get": lambda _self, _id: container})()
    client_type = type("Client", (), {"containers": containers})
    monkeypatch.setattr(service, "docker_client", lambda: client_type())
    health_results = iter([False, True])
    response_type = type(
        "Response",
        (),
        {"is_success": property(lambda _self: next(health_results))},
    )
    monkeypatch.setattr(deployment_service.httpx, "get", lambda *_args, **_kwargs: response_type())
    monkeypatch.setattr(deployment_service.time, "sleep", lambda _seconds: None)

    class Context:
        def update(self, **_kwargs):
            return None

        def check_control(self):
            return None

    result = service.action_handler(
        Context(),
        action_handler_payload(
            deployment_id,
            "start",
            container_id="container-id",
            container_name="dgx-managed",
        ),
    )

    assert container.starts == 1
    assert result["health"] == "healthy"
    with database.session_factory() as db:
        updated = db.get(Deployment, deployment_id)
        assert updated.status == "running"
        assert updated.health == "healthy"


def test_update_handler_replaces_container_and_keeps_deployment_id(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_path = model_root / "qwen"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"architectures":["Qwen2ForCausalLM"]}')
    (model_path / "model.safetensors").write_bytes(b"weights")
    database = Database(f"sqlite:///{tmp_path / 'update.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, model_path)
        deployment = Deployment(
            name="managed",
            model_id=target.id,
            runtime="vllm",
            container_id="old-container",
            container_name="dgx-managed",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="managed",
            status="running",
            health="healthy",
            managed=True,
            image="vllm:test",
            port=8100,
        )
        db.add(deployment)
        db.commit()
        deployment_id, model_id = deployment.id, target.id

    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(model_root,))
    service = deployment_service.DeploymentService(
        adapters={"vllm": adapter},
        session_factory=database.session_factory,
        model_roots=(model_root,),
        runtime_capability_service=StaticRuntimeCapabilities(),
        evidence_loader=ModelEvidenceLoader(),
        resource_estimator=ResourceEstimator(),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 64 * 1024**3, "available_bytes": 64 * 1024**3}
        },
    )

    class FakeContainer:
        def __init__(self, container_id, name):
            self.id = container_id
            self.name = name
            self.status = "running"
            self.removed = False

        def reload(self):
            return None

        def stop(self, **_kwargs):
            self.status = "exited"

        def start(self):
            self.status = "running"

        def rename(self, name):
            self.name = name

        def remove(self, **_kwargs):
            self.removed = True

        def logs(self, **_kwargs):
            return b"ready"

    old = FakeContainer("old-container", "dgx-managed")
    new = FakeContainer("new-container", "dgx-managed")

    class FakeContainers:
        def get(self, identifier):
            if identifier in {"old-container", "dgx-managed"}:
                return old
            raise docker.errors.NotFound("missing")

        def run(self, _image, **_kwargs):
            return new

    client_type = type("Client", (), {"containers": FakeContainers()})
    monkeypatch.setattr(service, "docker_client", lambda: client_type())
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)

    class Context:
        def update(self, **_kwargs):
            return None

        def check_control(self):
            return None

    result = service.update_handler(
        Context(),
        {
            "deployment_id": deployment_id,
            "spec": DeploymentSpec(
                name="managed",
                model_id=model_id,
                model_path=str(model_path),
                api_model_name="managed",
                route_alias="managed-route",
                runtime="vllm",
                image="vllm:test",
                port=8100,
                context_length=8192,
            ).model_dump(),
        },
    )

    assert result["deployment_id"] == deployment_id
    assert old.removed is True
    with database.session_factory() as db:
        updated = db.get(Deployment, deployment_id)
        assert updated.container_id == "new-container"
        assert updated.config["spec"]["context_length"] == 8192
        assert updated.config["route_alias"] == "managed-route"


def test_update_handler_restores_old_container_when_replacement_is_unhealthy(tmp_path, monkeypatch):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / 'update-rollback.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        deployment = Deployment(
            name="base",
            model_id=target.id,
            runtime="vllm",
            container_id="old-container",
            container_name="dgx-base",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="base",
            status="running",
            health="healthy",
            managed=True,
            image="vllm:test",
            port=8100,
            config={"before": True},
        )
        db.add(deployment)
        db.commit()
        deployment_id, model_id = deployment.id, target.id
    service = build_preflight_service(database, (root,))

    class Container:
        def __init__(self, container_id, name):
            self.id = container_id
            self.name = name
            self.status = "running"
            self.removed = False
            self.started = 0

        def reload(self):
            return None

        def stop(self, **_kwargs):
            self.status = "exited"

        def start(self):
            self.started += 1
            self.status = "running"

        def rename(self, name):
            self.name = name

        def remove(self, **_kwargs):
            self.removed = True

        def logs(self, **_kwargs):
            return b"replacement failed"

    old = Container("old-container", "dgx-base")
    new = Container("new-container", "dgx-base")

    class Containers:
        def get(self, identifier):
            if identifier == "old-container":
                return old
            raise docker.errors.NotFound("missing")

        def run(self, _image, **_kwargs):
            return new

    containers = Containers()
    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": containers})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: False)
    context = type(
        "Context",
        (),
        {"update": lambda _self, **_kwargs: None, "check_control": lambda _self: None},
    )()
    spec = DeploymentSpec(
        name="base",
        model_id=model_id,
        model_path="ignored",
        api_model_name="base",
        runtime="vllm",
        image="vllm:test",
        port=8100,
    )

    with pytest.raises(RuntimeError, match="replacement failed"):
        service.update_handler(
            context,
            {"deployment_id": deployment_id, "spec": spec.model_dump(mode="json")},
        )

    assert new.removed is True
    assert old.name == "dgx-base"
    assert old.started == 1
    with database.session_factory() as db:
        unchanged = db.get(Deployment, deployment_id)
        assert unchanged.container_id == "old-container"
        assert unchanged.config == {"before": True}


@pytest.mark.parametrize("crash_stage", ["renamed", "replacement", "committed", "mismatched"])
def test_update_handler_reconciles_interrupted_replacement_stages(
    tmp_path, monkeypatch, crash_stage
):
    root = tmp_path / "models"
    database = Database(f"sqlite:///{tmp_path / f'reconcile-{crash_stage}.db'}")
    database.create_schema()
    with database.session_factory() as db:
        target = add_model_asset(db, root / "base")
        deployment = Deployment(
            name="old-name",
            model_id=target.id,
            runtime="vllm",
            container_id="old-container",
            container_name="dgx-old-name",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="old-api",
            status="running",
            health="healthy",
            managed=True,
            image="vllm:test",
            port=8100,
            config={"before": True},
        )
        db.add(deployment)
        db.commit()
        deployment_id, model_id = deployment.id, target.id
    service = build_preflight_service(database, (root,))
    spec = DeploymentSpec(
        name="new-name",
        model_id=model_id,
        model_path="ignored",
        api_model_name="new-api",
        runtime="vllm",
        image="vllm:test",
        port=8101,
    )
    with database.session_factory() as db:
        resolved, preview = service._preflight(db, spec, exclude_deployment_id=deployment_id)
    fingerprint = preview["spec_fingerprint"]
    target_name = deterministic_container_name(spec.name)
    backup_name = f"dgx-backup-{deployment_id}"[:63]
    labels = service._expected_container_labels(
        resolved,
        task_id="task-1",
        spec_fingerprint=fingerprint,
        deployment_id=deployment_id,
        replaces_container_id="old-container",
    )

    class Container:
        def __init__(self, container_id, name, *, labels=None):
            self.id = container_id
            self.name = name
            self.status = "running"
            self.attrs = {"Config": {"Labels": labels or {}}}
            self.removed = False
            self.renames = []

        def reload(self):
            return None

        def stop(self, **_kwargs):
            self.status = "exited"

        def start(self):
            self.status = "running"

        def rename(self, name):
            self.renames.append(name)
            self.name = name

        def remove(self, **_kwargs):
            self.removed = True

        def logs(self, **_kwargs):
            return b"ready"

    old = Container(
        "old-container",
        backup_name,
        labels={"com.dgx-spark-manager.managed": "true"},
    )
    replacement = (
        None
        if crash_stage == "renamed"
        else Container(
            "new-container",
            target_name,
            labels=(
                {"com.dgx-spark-manager.managed": "true"} if crash_stage == "mismatched" else labels
            ),
        )
    )
    if crash_stage == "committed":
        with database.session_factory() as db:
            deployment = db.get(Deployment, deployment_id)
            deployment.name = spec.name
            deployment.api_model_name = spec.api_model_name
            deployment.container_id = replacement.id
            deployment.container_name = target_name
            deployment.endpoint_url = "http://127.0.0.1:8101"
            deployment.port = spec.port
            deployment.config = preview
            db.commit()

    class Containers:
        run_calls = 0

        def get(self, identifier):
            if identifier in {"old-container", backup_name}:
                return old
            if replacement is not None and identifier in {
                "new-container",
                target_name,
            }:
                return replacement
            raise docker.errors.NotFound("missing")

        def run(self, _image, **kwargs):
            self.run_calls += 1
            if replacement is not None:
                raise AssertionError("replacement must not be created twice")
            created = Container(
                "new-container",
                kwargs["name"],
                labels=kwargs["labels"],
            )
            nonlocal_replacement[0] = created
            return created

    nonlocal_replacement = [replacement]
    containers = Containers()

    def get_with_created(identifier):
        current = nonlocal_replacement[0]
        if identifier in {"old-container", backup_name}:
            return old
        if current is not None and identifier in {"new-container", target_name}:
            return current
        raise docker.errors.NotFound("missing")

    monkeypatch.setattr(containers, "get", get_with_created)
    monkeypatch.setattr(
        service,
        "docker_client",
        lambda: type("Client", (), {"containers": containers})(),
    )
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)

    if crash_stage == "mismatched":
        with pytest.raises(ValueError, match="container labels do not match"):
            service.update_handler(
                HandlerContext(),
                {
                    "deployment_id": deployment_id,
                    "spec": spec.model_dump(mode="json"),
                },
            )
        assert old.name == "dgx-old-name"
        assert old.status == "running"
        assert replacement.removed is False
        return

    result = service.update_handler(
        HandlerContext(),
        {"deployment_id": deployment_id, "spec": spec.model_dump(mode="json")},
    )

    assert result["deployment_id"] == deployment_id
    assert containers.run_calls == (1 if crash_stage == "renamed" else 0)
    assert old.renames == []
    assert old.removed is True
    with database.session_factory() as db:
        updated = db.get(Deployment, deployment_id)
        assert updated.container_id == "new-container"
        assert updated.config["spec_fingerprint"] == fingerprint


class LaunchContractContainer(ActionContainer):
    def __init__(self, container_id, name, attrs, *, status="exited"):
        super().__init__(container_id, name, status=status)
        self.attrs = attrs
        self.renames = []
        self.remove_error = None

    def rename(self, name):
        self.renames.append(name)
        self.name = name

    def remove(self, **_kwargs):
        if self.remove_error is not None:
            raise self.remove_error
        self.removed = True

    def logs(self, **_kwargs):
        return b"launch failed"


def launch_contract():
    return {
        "image": "runtime:test",
        "entrypoint": ["/opt/runtime/server"],
        "command": ["--model", "/models/base", "--port", "8000"],
        "environment": {"HF_HUB_OFFLINE": "1", "MODE": "serve"},
        "mounts": [
            {
                "type": "bind",
                "source": "/host/models",
                "target": "/models",
                "read_only": True,
            }
        ],
        "network_mode": "bridge",
        "ipc_mode": "host",
        "restart_policy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
        "device_requests": [
            {
                "Driver": "",
                "Count": -1,
                "DeviceIDs": [],
                "Capabilities": [["gpu"]],
                "Options": {},
            }
        ],
        "port": {
            "container_port": 8000,
            "host_port": 8100,
            "protocol": "tcp",
        },
    }


def launch_contract_attrs(contract=None):
    contract = contract or launch_contract()
    return {
        "Config": {
            "Image": contract["image"],
            "Entrypoint": contract["entrypoint"],
            "Cmd": contract["command"],
            "Env": [f"{key}={value}" for key, value in contract["environment"].items()],
        },
        "Mounts": [
            {
                "Type": mount["type"],
                "Source": mount["source"],
                "Destination": mount["target"],
                "RW": not mount["read_only"],
            }
            for mount in contract["mounts"]
        ],
        "HostConfig": {
            "NetworkMode": contract["network_mode"],
            "IpcMode": contract["ipc_mode"],
            "RestartPolicy": contract["restart_policy"],
            "DeviceRequests": contract["device_requests"],
            "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8100"}]},
        },
    }


def test_start_with_drifted_launch_contract_atomically_rebuilds_and_updates_db(
    tmp_path,
):
    database = Database(f"sqlite:///{tmp_path / 'contract-drift.db'}")
    database.create_schema()
    contract = launch_contract()
    drifted = launch_contract_attrs(contract)
    drifted["Config"]["Image"] = "wrong:snapshot"
    old = LaunchContractContainer("a" * 64, "dgx-managed", drifted)
    replacement = LaunchContractContainer(
        "b" * 64,
        "dgx-managed",
        launch_contract_attrs(contract),
        status="running",
    )
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=old.id,
            container_name=old.name,
            status="exited",
            config={"launch_contract": contract},
        )
        deployment_id = deployment.id

    class Containers:
        def __init__(self):
            self.run_calls = []

        def get(self, identifier):
            if identifier == old.id:
                return old
            raise docker.errors.NotFound("missing")

        def run(self, image, **kwargs):
            self.run_calls.append((image, kwargs))
            return replacement

    containers = Containers()
    service, _, _ = build_action_service(database, tmp_path, old)
    service._docker_client = type("Client", (), {"containers": containers})()
    service.wait_for_health = lambda *_args, **_kwargs: True

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "start",
            container_id=old.id,
            container_name="dgx-managed",
        ),
    )

    assert old.starts == 0 and old.restarts == 0
    assert old.name != "dgx-managed" and old.removed is True
    image, kwargs = containers.run_calls[0]
    assert image == contract["image"]
    assert kwargs["entrypoint"] == contract["entrypoint"]
    assert kwargs["command"] == contract["command"]
    assert kwargs["environment"] == contract["environment"]
    assert kwargs["volumes"] == {"/host/models": {"bind": "/models", "mode": "ro"}}
    assert kwargs["network_mode"] == "bridge" and kwargs["ipc_mode"] == "host"
    assert kwargs["restart_policy"] == contract["restart_policy"]
    assert kwargs["ports"] == {"8000/tcp": ("", 8100)}
    assert kwargs["device_requests"][0]["Count"] == -1
    assert result["status"] == "running" and result["health"] == "healthy"
    with database.session_factory() as db:
        updated = db.get(Deployment, deployment_id)
        assert (updated.container_id, updated.container_name) == (
            replacement.id,
            "dgx-managed",
        )
        assert updated.status == "running"
        assert updated.health == "healthy"
        assert updated.managed is True


def test_restart_with_matching_launch_contract_uses_existing_container(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'contract-match.db'}")
    database.create_schema()
    contract = launch_contract()
    container = LaunchContractContainer(
        "c" * 64,
        "dgx-managed",
        launch_contract_attrs(contract),
        status="running",
    )
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=container.id,
            container_name=container.name,
            config={"launch_contract": contract},
        )
        deployment_id = deployment.id
    service, _, _ = build_action_service(database, tmp_path, container)
    service.wait_for_health = lambda *_args, **_kwargs: True

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "restart",
            container_id=container.id,
            container_name=container.name,
        ),
    )

    assert container.restarts == 1 and container.renames == []
    assert result["health"] == "healthy"


def test_start_with_missing_launch_contract_container_rebuilds(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'contract-missing.db'}")
    database.create_schema()
    contract = launch_contract()
    replacement = LaunchContractContainer(
        "d" * 64,
        "dgx-managed",
        launch_contract_attrs(contract),
        status="running",
    )
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id="e" * 64,
            container_name="dgx-managed",
            status="exited",
            config={"launch_contract": contract},
        )
        deployment_id = deployment.id

    class Containers:
        def __init__(self):
            self.runs = 0

        def get(self, _identifier):
            raise docker.errors.NotFound("missing")

        def run(self, _image, **_kwargs):
            self.runs += 1
            return replacement

    containers = Containers()
    service, _, _ = build_action_service(database, tmp_path, docker.errors.NotFound("missing"))
    service._docker_client = type("Client", (), {"containers": containers})()
    service.wait_for_health = lambda *_args, **_kwargs: True

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "start",
            container_id="e" * 64,
            container_name="dgx-managed",
        ),
    )

    assert containers.runs == 1 and result["container_missing"] is True
    with database.session_factory() as db:
        assert db.get(Deployment, deployment_id).container_id == replacement.id


def test_unhealthy_launch_contract_replacement_rolls_back_and_marks_unhealthy(
    tmp_path,
):
    database = Database(f"sqlite:///{tmp_path / 'contract-rollback.db'}")
    database.create_schema()
    contract = launch_contract()
    drifted = launch_contract_attrs(contract)
    drifted["Config"]["Cmd"] = ["wrong"]
    old = LaunchContractContainer("f" * 64, "dgx-managed", drifted, status="running")
    replacement = LaunchContractContainer(
        "1" * 64,
        "dgx-managed",
        launch_contract_attrs(contract),
        status="running",
    )
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=old.id,
            container_name=old.name,
            config={"launch_contract": contract},
        )
        deployment_id = deployment.id

    class Containers:
        def get(self, identifier):
            if identifier == old.id:
                return old
            raise docker.errors.NotFound("missing")

        def run(self, _image, **_kwargs):
            return replacement

    service, _, _ = build_action_service(database, tmp_path, old)
    service._docker_client = type("Client", (), {"containers": Containers()})()
    service.wait_for_health = lambda *_args, **_kwargs: False

    with pytest.raises(RuntimeError, match="did not become healthy"):
        service.action_handler(
            HandlerContext(),
            action_handler_payload(
                deployment_id,
                "restart",
                container_id=old.id,
                container_name="dgx-managed",
            ),
        )

    assert replacement.removed is True
    assert old.name == "dgx-managed" and old.starts == 1 and old.restarts == 0
    with database.session_factory() as db:
        updated = db.get(Deployment, deployment_id)
        assert updated.container_id == old.id
        assert updated.status == "running" and updated.health == "unhealthy"


def test_launch_contract_cleanup_failure_keeps_committed_replacement(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'contract-cleanup.db'}")
    database.create_schema()
    contract = launch_contract()
    drifted = launch_contract_attrs(contract)
    drifted["Config"]["Image"] = "wrong:snapshot"
    old = LaunchContractContainer("2" * 64, "dgx-managed", drifted)
    old.remove_error = RuntimeError("cleanup failed")
    replacement = LaunchContractContainer(
        "3" * 64,
        "dgx-managed",
        launch_contract_attrs(contract),
        status="running",
    )
    with database.session_factory() as db:
        deployment = add_action_deployment(
            db,
            managed=True,
            name="managed",
            container_id=old.id,
            container_name=old.name,
            status="exited",
            config={"launch_contract": contract},
        )
        deployment_id = deployment.id

    class Containers:
        def get(self, _identifier):
            return old

        def run(self, _image, **_kwargs):
            return replacement

    service, _, _ = build_action_service(database, tmp_path, old)
    service._docker_client = type("Client", (), {"containers": Containers()})()
    service.wait_for_health = lambda *_args, **_kwargs: True

    result = service.action_handler(
        HandlerContext(),
        action_handler_payload(
            deployment_id,
            "start",
            container_id=old.id,
            container_name="dgx-managed",
        ),
    )

    assert result["status"] == "running"
    assert replacement.removed is False
    with database.session_factory() as db:
        updated = db.get(Deployment, deployment_id)
        assert updated.container_id == replacement.id
        assert updated.status == "running" and updated.health == "healthy"
