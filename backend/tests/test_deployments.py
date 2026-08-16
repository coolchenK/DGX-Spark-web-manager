import json
from datetime import UTC, datetime
from pathlib import Path

import docker
import pytest
from app.config import Settings
from app.db import Database
from app.main import create_app
from app.models import Deployment, ModelAsset, TaskRecord
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
            speculative_methods=(
                ["draft_model", "eagle3"] if methods is None else methods
            ),
            method_mapping=(
                {"draft_model": "draft_model", "eagle3": "eagle3"}
                if mapping is None
                else mapping
            ),
            speculative_transport="json",
            warnings=[],
        )

    def get(self, runtime, image):
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
def test_generation_defaults_enforces_numeric_boundaries(
    field, lower, upper, below, above
):
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
def test_speculative_config_enforces_numeric_boundaries(
    field, lower, upper, below, above
):
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


@pytest.mark.parametrize("method", ["draft_model", "eagle", "eagle3", "mtp"])
def test_speculative_config_accepts_supported_methods(method):
    assert SpeculativeConfig(draft_model_id="draft-id", method=method).method == method


def test_speculative_config_rejects_unknown_method():
    with pytest.raises(ValueError):
        SpeculativeConfig(draft_model_id="draft-id", method="unknown")


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
def test_public_deployment_spec_rejects_internal_resolution_fields(
    tmp_path, field, value
):
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

    with database.session_factory() as db, pytest.raises(
        ValueError, match="Base model is missing or unavailable"
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
    assert captured["volumes"] == {
        str(host_root): {"bind": "/models", "mode": "ro"}
    }


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
            draft_id = add_model_asset(
                db, root / "draft", name="Draft", status="failed"
            ).id
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
def test_resolve_spec_rejects_unsupported_speculative_capabilities(
    tmp_path, capabilities, message
):
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

    with database.session_factory() as db, pytest.raises(
        ValueError, match="Shared route generation defaults must match"
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

    service = build_preflight_service(
        database, (root,), estimator=estimator, snapshot=snapshot
    )
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
    service = build_preflight_service(
        database, (root,), host_model_roots=(host_root,)
    )
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

    with database.session_factory() as db, pytest.raises(
        ValueError, match="Base model is incompatible"
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
    sglang = SGLangAdapter(
        allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",)
    )
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
        '{"method":"draft_model","model":"/draft-models/draft",'
        '"num_speculative_tokens":5}'
    )


def test_vllm_command_rejects_unsupported_grouped_speculative_tuning(tmp_path):
    adapter = VllmAdapter(
        allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",)
    )
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
    adapter = SGLangAdapter(
        allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",)
    )
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
    adapter = SGLangAdapter(
        allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",)
    )
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
        ("eagle", "EAGLE"),
        ("eagle3", "EAGLE3"),
        ("mtp", "NEXTN"),
    ],
)
def test_sglang_command_accepts_only_trusted_method_mappings(
    tmp_path, method, runtime_method
):
    adapter = SGLangAdapter(
        allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",)
    )
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
def test_speculative_commands_require_resolved_internal_fields(
    tmp_path, runtime, field, message
):
    adapter_type = VllmAdapter if runtime == "vllm" else SGLangAdapter
    adapter = adapter_type(
        allowed_images={f"{runtime}:test"}, model_roots=(tmp_path / "models",)
    )
    runtime_method = "draft_model" if runtime == "vllm" else "STANDALONE"
    spec = resolved_speculative_spec(
        tmp_path, runtime=runtime, runtime_method=runtime_method
    ).model_copy(update={field: None})

    with pytest.raises(ValueError, match=message):
        adapter.command(spec)


@pytest.mark.parametrize("draft_path", ["", "relative/draft", "--draft-model"])
def test_speculative_commands_reject_non_absolute_draft_container_paths(
    tmp_path, draft_path
):
    adapter = VllmAdapter(
        allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",)
    )
    spec = resolved_speculative_spec(tmp_path, draft_container_path=draft_path)

    with pytest.raises(ValueError, match="resolved draft container path is required"):
        adapter.command(spec)


def test_public_speculative_spec_cannot_build_a_runtime_command(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    adapter = VllmAdapter(
        allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",)
    )
    spec = DeploymentSpec(
        name="Qwen",
        model_path=str(model_path),
        api_model_name="qwen",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        speculative={"draft_model_id": "org/draft", "method": "draft_model"},
    )

    with pytest.raises(
        ValueError, match="resolved speculative runtime method is required"
    ):
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
    adapter = adapter_type(
        allowed_images={f"{runtime}:test"}, model_roots=(tmp_path / "models",)
    )
    spec = resolved_speculative_spec(
        tmp_path,
        runtime=runtime,
        method=method,
        runtime_method=runtime_method,
    )

    with pytest.raises(ValueError, match="does not match speculative method"):
        adapter.command(spec)


def test_sglang_command_rejects_unmapped_num_speculative_tokens(tmp_path):
    adapter = SGLangAdapter(
        allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",)
    )
    spec = resolved_speculative_spec(
        tmp_path,
        runtime="sglang",
        runtime_method="STANDALONE",
        num_speculative_tokens=5,
    )

    with pytest.raises(ValueError, match="num_speculative_tokens is not supported by SGLang"):
        adapter.command(spec)


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

    sglang = SGLangAdapter(
        allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",)
    )
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
        assert task.input_json["recommendation"]["generated_at"] == (
            "2026-08-16T00:00:00Z"
        )
        json.dumps(task.input_json)


def test_preview_endpoint_passes_database_and_deployment_exclusion(
    authenticated_client, tmp_path
):
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


def test_create_app_shares_preflight_dependencies(settings):
    app = create_app(settings)
    deployments = app.state.deployment_service
    recommendations = app.state.deployment_recommendation_service

    assert (
        deployments.runtime_capability_service
        is recommendations.runtime_capability_service
    )
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
    authenticated_client.app.state.deployment_service.preview = (
        lambda _db, spec, **_kwargs: {"spec": spec.model_dump(mode="json")}
    )

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
        assert task.input_json["spec"]["recommendation"]["generated_at"] == (
            "2026-08-16T00:00:00Z"
        )
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
        model_root_mappings=(
            f"{container_models}={host_models};{container_hf}={host_hf}"
        ),
    )

    assert getattr(settings, "host_model_root_paths", ()) == (host_models, host_hf)
    assert getattr(settings, "deployment_startup_timeout_seconds", None) == 300


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

        def logs(self, **_kwargs):
            return b"engine failed while allocating memory"

        def stop(self, **_kwargs):
            self.stopped = True

        def remove(self):
            self.removed = True

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

    assert container.stopped is True
    assert container.removed is True


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
        adapters={
            "vllm": VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path,))
        },
        session_factory=database.session_factory,
        model_roots=(tmp_path,),
        startup_timeout_seconds=4,
    )

    class FakeContainer:
        name = "dgx-managed"
        starts = 0

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
        {"deployment_id": deployment_id, "action": "start"},
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
            assert identifier == "old-container"
            return old

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


def test_update_handler_restores_old_container_when_replacement_is_unhealthy(
    tmp_path, monkeypatch
):
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
    containers = type(
        "Containers",
        (),
        {
            "get": lambda _self, _identifier: old,
            "run": lambda _self, _image, **_kwargs: new,
        },
    )()
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

