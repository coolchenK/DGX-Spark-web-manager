from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from app.db import Database
from app.models import ModelAsset
from app.services.deployment_recommendations import (
    DeploymentRecommendation,
    DeploymentRecommendationService,
    RecommendationRequest,
    RecommendedValue,
    select_deployment_values,
    select_generation_defaults,
)
from app.services.draft_models import DraftCandidate
from app.services.model_evidence import ModelEvidence
from app.services.resource_estimator import ResourceEstimate
from app.services.runtime_capabilities import RuntimeCapabilities
from pydantic import ValidationError

GiB = 1024**3


def evidence(
    path: str,
    *,
    config: dict[str, Any] | None = None,
    card_deployment: dict[str, Any] | None = None,
    card_generation: dict[str, Any] | None = None,
    local_generation: dict[str, Any] | None = None,
    card_text: str = "card",
    evidence_hash: str = "a" * 64,
) -> ModelEvidence:
    return ModelEvidence(
        model_path=path,
        config=config or {},
        generation_config=local_generation or {},
        tokenizer_fingerprint="tokenizer",
        card_text=card_text,
        card_data={},
        card_deployment_values=card_deployment or {},
        card_generation_values=card_generation or {},
        local_generation_values=local_generation or {},
        target_model_ids=[],
        speculative_method=None,
        evidence_hash=evidence_hash,
        warnings=[],
    )


def capabilities(
    *,
    runtime: str = "vllm",
    generation_defaults: list[str] | None = None,
    quantization_methods: list[str] | None = None,
    quantization_mapping: dict[str, str] | None = None,
) -> RuntimeCapabilities:
    return RuntimeCapabilities(
        runtime=runtime,
        image=f"{runtime}:test",
        image_digest="sha256:tested",
        source="probe",
        generation_defaults=(
            generation_defaults
            if generation_defaults is not None
            else ["temperature", "top_p", "max_tokens"]
        ),
        quantization_methods=(
            quantization_methods
            if quantization_methods is not None
            else ["auto", "modelopt_fp4", "awq"]
        ),
        quantization_mapping=quantization_mapping or {"nvfp4": "modelopt_fp4"},
        speculative_methods=["draft_model"],
        method_mapping={"draft_model": "draft_model"},
        speculative_transport="json" if runtime == "vllm" else "flags",
        warnings=[],
    )


def resource_estimate(
    context: int,
    *,
    decision: str = "ok",
    total: int = 128 * GiB,
    available: int = 112 * GiB,
) -> ResourceEstimate:
    return ResourceEstimate(
        total_bytes=total,
        available_bytes=available,
        reserved_bytes=13 * GiB,
        weight_bytes=8 * GiB,
        draft_weight_bytes=0,
        kv_cache_bytes=context * 1024,
        runtime_overhead_bytes=2 * GiB,
        required_bytes=10 * GiB + context * 1024,
        decision=decision,
        confidence="high",
        reasons=[],
    )


def test_contracts_are_strict_bounded_and_json_serializable() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            model_id="model",
            runtime="vllm",
            image="vllm:test",
            unknown=True,
        )
    with pytest.raises(ValidationError):
        RecommendedValue(
            value=1,
            source="device_rule",
            confidence="high",
            reason="x" * 501,
        )
    with pytest.raises(ValidationError):
        RecommendedValue(
            value=object(),
            source="device_rule",
            confidence="high",
            reason="bounded",
        )

    response = DeploymentRecommendation(
        status="partial",
        generated_at=datetime.now().astimezone(),
        model_id="model",
        runtime="vllm",
        image_digest="sha256:test",
        evidence_hash="a" * 64,
        fields={},
        generation_defaults={},
        resource_snapshot={},
        resource_estimate={},
        runtime_capabilities={},
        draft_candidates=[],
        warnings=["AI analysis is required"],
    )
    json.dumps(response.model_dump(mode="json"))


def test_pure_selectors_apply_precedence_limits_and_per_field_validation() -> None:
    model_evidence = evidence(
        "/models/target",
        config={"max_position_embeddings": 16_384},
        card_deployment={
            "context_length": 32_768,
            "memory_fraction": 0.75,
            "max_concurrency": "invalid",
        },
        card_generation={"temperature": 0.6, "top_p": 2, "top_k": 20},
        local_generation={"temperature": 0.7, "top_p": 0.9},
    )

    fields, field_warnings = select_deployment_values(
        model_evidence,
        capabilities(),
        runtime_defaults={"memory_fraction": 0.8, "max_concurrency": 8},
    )
    generation, generation_warnings = select_generation_defaults(
        model_evidence,
        capabilities(),
        runtime_defaults={"temperature": 1.0, "max_tokens": 1024},
    )

    assert fields["context_length"].value == 16_384
    assert fields["context_length"].source == "device_rule"
    assert "32768" in fields["context_length"].reason
    assert "16384" in fields["context_length"].reason
    assert fields["memory_fraction"].source == "model_card"
    assert fields["max_concurrency"].value == 8
    assert fields["max_concurrency"].source == "runtime_default"
    assert any("max_concurrency" in warning for warning in field_warnings)
    assert generation["temperature"].value == 0.6
    assert generation["temperature"].source == "model_card"
    assert generation["max_tokens"].source == "runtime_default"
    assert generation["top_p"].value == 0.9
    assert generation["top_p"].source == "local_config"
    assert "top_k" not in generation
    assert any("top_p" in warning for warning in generation_warnings)
    assert any("top_k" in warning and "unsupported" in warning for warning in generation_warnings)


@pytest.mark.parametrize(
    ("methods", "mapping", "expected", "has_warning"),
    [
        (["auto", "modelopt_fp4"], {"nvfp4": "modelopt_fp4"}, "modelopt_fp4", False),
        (["auto"], {"nvfp4": "modelopt_fp4"}, "auto", True),
        (["auto", "awq"], {}, "awq", False),
    ],
)
def test_quantization_is_a_bounded_hard_fact(
    methods: list[str], mapping: dict[str, str], expected: str, has_warning: bool
) -> None:
    quantization = "awq" if expected == "awq" else "nvfp4"
    fields, warnings = select_deployment_values(
        evidence(
            "/models/target",
            config={"quantization_config": {"quant_method": quantization}},
        ),
        capabilities(
            quantization_methods=methods,
            quantization_mapping=mapping,
        ),
        runtime_defaults={"quantization": "auto"},
    )

    assert fields["quantization"].value == expected
    assert fields["quantization"].source == (
        "device_rule" if has_warning else "local_config"
    )
    assert bool(warnings) is has_warning


class FakeEvidenceLoader:
    def __init__(self, result: ModelEvidence | Exception):
        self.result = result
        self.calls: list[str] = []

    def load(self, path: Path | str) -> ModelEvidence:
        self.calls.append(str(path))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeCapabilities:
    def __init__(self, result: RuntimeCapabilities | Exception):
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def get(self, runtime: str, image: str) -> RuntimeCapabilities:
        self.calls.append((runtime, image))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeEstimator:
    def __init__(self, blocked_above: int | None = None, *, fail: bool = False):
        self.blocked_above = blocked_above
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def estimate(self, **kwargs: Any) -> ResourceEstimate:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("private estimator failure")
        context = int(kwargs["context_length"])
        decision = (
            "blocked"
            if self.blocked_above is not None and context > self.blocked_above
            else "ok"
        )
        memory = kwargs["system_memory"]
        return resource_estimate(
            context,
            decision=decision,
            total=int(memory["total_bytes"]),
            available=int(memory["available_bytes"]),
        )


class FakeDraftService:
    def __init__(self, result: list[DraftCandidate] | Exception):
        self.result = result
        self.calls = 0

    def list_candidates(self, *_args: Any) -> list[DraftCandidate]:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeHuggingFace:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[str] = []

    def model_card_text(self, repository_id: str) -> str:
        self.calls.append(repository_id)
        return self.text


def add_asset(database: Database, path: Path, **overrides: Any) -> ModelAsset:
    payload = {
        "id": "target",
        "name": "Target",
        "source": "huggingface",
        "repository_id": "org/target",
        "local_path": str(path),
        "status": "available",
        "size_bytes": 8 * GiB,
        "quantization": "nvfp4",
    }
    payload.update(overrides)
    asset = ModelAsset(**payload)
    with database.session_factory() as db:
        db.add(asset)
        db.commit()
        db.refresh(asset)
        db.expunge(asset)
    return asset


def test_service_clamps_context_and_returns_complete_deterministic_result(
    settings, tmp_path: Path
) -> None:
    database = Database(settings.database_url)
    database.create_schema()
    model_path = tmp_path / "target"
    model_path.mkdir()
    add_asset(database, model_path)
    target_evidence = evidence(
        str(model_path),
        config={"max_position_embeddings": 32_768},
        card_deployment={"context_length": 32_768, "memory_fraction": 0.8},
        card_generation={"temperature": 0.6},
    )
    candidate = DraftCandidate(
        model_id="draft",
        name="Draft",
        repository_id="org/draft",
        method="draft_model",
        status="compatible",
        reasons=["Explicit target match"],
        size_bytes=GiB,
        estimated_total_bytes=20 * GiB,
    )
    snapshot_calls = 0

    def snapshot() -> dict[str, Any]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return {
            "memory": {"total_bytes": 128 * GiB, "available_bytes": 112 * GiB},
            "gpus": [{"memory_used_bytes": 9 * GiB}],
        }

    estimator = FakeEstimator(blocked_above=16_384)
    service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(target_evidence),
        runtime_capability_service=FakeCapabilities(capabilities()),
        resource_estimator=estimator,
        draft_service=FakeDraftService([candidate]),
        system_snapshot=snapshot,
    )

    with database.session_factory() as db:
        result = service.recommend(db, "target", "vllm", "vllm:test")

    assert result.status == "complete"
    assert result.fields["context_length"].value == 16_384
    assert result.fields["context_length"].source == "device_rule"
    assert "32768" in result.fields["context_length"].reason
    assert "16384" in result.fields["context_length"].reason
    assert result.generation_defaults["temperature"].value == 0.6
    assert result.generation_defaults["temperature"].source == "model_card"
    assert result.fields["quantization"].value == "modelopt_fp4"
    assert result.draft_candidates[0].status == "compatible"
    assert result.resource_snapshot["total_bytes"] == 128 * GiB
    assert result.resource_snapshot["available_bytes"] == 112 * GiB
    assert snapshot_calls == 1
    assert all(
        call["system_memory"]
        == {"total_bytes": 128 * GiB, "available_bytes": 112 * GiB}
        for call in estimator.calls
    )
    assert all("gpus" not in call["system_memory"] for call in estimator.calls)
    json.dumps(result.model_dump(mode="json"))
    database.dispose()


def test_resource_clamp_preserves_the_original_card_context_in_reason(
    settings, tmp_path: Path
) -> None:
    database = Database(settings.database_url)
    database.create_schema()
    model_path = tmp_path / "target"
    model_path.mkdir()
    add_asset(database, model_path)
    target_evidence = evidence(
        str(model_path),
        config={"max_position_embeddings": 16_384},
        card_deployment={"context_length": 32_768},
    )
    service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(target_evidence),
        runtime_capability_service=FakeCapabilities(capabilities()),
        resource_estimator=FakeEstimator(blocked_above=8192),
        draft_service=FakeDraftService([]),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 128 * GiB, "available_bytes": 100 * GiB}
        },
    )

    with database.session_factory() as db:
        result = service.recommend(db, "target", "vllm", "vllm:test")

    assert result.fields["context_length"].value == 8192
    assert "32768" in result.fields["context_length"].reason
    assert "16384" in result.fields["context_length"].reason
    assert "8192" in result.fields["context_length"].reason
    database.dispose()


def test_minimal_model_is_partial_and_remote_card_fallback_only_runs_when_needed(
    settings, tmp_path: Path
) -> None:
    database = Database(settings.database_url)
    database.create_schema()
    model_path = tmp_path / "minimal"
    model_path.mkdir()
    add_asset(database, model_path, quantization=None)
    local = evidence(str(model_path), card_text="")
    remote = evidence(str(model_path), card_text="remote", evidence_hash="b" * 64)
    huggingface = FakeHuggingFace("remote card")
    remote_calls: list[tuple[str, str]] = []

    def remote_loader(path: Path | str, card_text: str) -> ModelEvidence:
        remote_calls.append((str(path), card_text))
        return remote

    service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(local),
        runtime_capability_service=FakeCapabilities(capabilities(runtime="sglang")),
        resource_estimator=FakeEstimator(),
        draft_service=FakeDraftService([]),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 128 * GiB, "available_bytes": 100 * GiB},
            "deployments": [{"id": "running", "memory_bytes": 4 * GiB}],
        },
        huggingface_service=huggingface,
        remote_evidence_loader=remote_loader,
    )

    with database.session_factory() as db:
        result = service.recommend(db, "target", "sglang", "sglang:test")

    assert result.status == "partial"
    assert result.evidence_hash == "b" * 64
    assert any("AI" in warning for warning in result.warnings)
    assert result.resource_snapshot["reserved_bytes"] > 0
    assert result.resource_snapshot["deployments"] == [
        {"id": "running", "memory_bytes": 4 * GiB}
    ]
    assert huggingface.calls == ["org/target"]
    assert remote_calls == [(str(model_path), "remote card")]

    local_with_card = evidence(str(model_path), card_text="already local")
    service.evidence_loader = FakeEvidenceLoader(local_with_card)
    with database.session_factory() as db:
        service.recommend(db, "target", "sglang", "sglang:test")
    assert huggingface.calls == ["org/target"]
    database.dispose()


@pytest.mark.parametrize(
    ("asset_overrides", "capability_result", "expected_warning"),
    [
        ({"status": "downloading"}, capabilities(), "unavailable"),
        ({"local_path": ""}, capabilities(), "path"),
        ({}, RuntimeError("secret image error"), "capabilities"),
    ],
)
def test_unavailable_model_and_capability_failures_are_structured_and_redacted(
    settings,
    tmp_path: Path,
    asset_overrides: dict[str, Any],
    capability_result: RuntimeCapabilities | Exception,
    expected_warning: str,
) -> None:
    database = Database(settings.database_url)
    database.create_schema()
    model_path = tmp_path / "target"
    model_path.mkdir()
    add_asset(database, model_path, **asset_overrides)
    service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(evidence(str(model_path))),
        runtime_capability_service=FakeCapabilities(capability_result),
        resource_estimator=FakeEstimator(),
        draft_service=FakeDraftService([]),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 128 * GiB, "available_bytes": 100 * GiB}
        },
    )

    with database.session_factory() as db:
        result = service.recommend(db, "target", "vllm", "vllm:test")

    assert result.status == "unavailable"
    assert result.evidence_hash is None
    assert any(expected_warning in warning.lower() for warning in result.warnings)
    assert all("secret" not in warning for warning in result.warnings)
    json.dumps(result.model_dump(mode="json"))
    database.dispose()


def test_dependency_failures_are_isolated_with_explicit_status(settings, tmp_path: Path) -> None:
    database = Database(settings.database_url)
    database.create_schema()
    model_path = tmp_path / "target"
    model_path.mkdir()
    add_asset(database, model_path)
    target_evidence = evidence(
        str(model_path),
        config={"max_position_embeddings": 8192},
    )

    draft_failure_service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(target_evidence),
        runtime_capability_service=FakeCapabilities(capabilities()),
        resource_estimator=FakeEstimator(),
        draft_service=FakeDraftService(RuntimeError("private draft failure")),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 128 * GiB, "available_bytes": 100 * GiB}
        },
    )
    resource_failure_service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(target_evidence),
        runtime_capability_service=FakeCapabilities(capabilities()),
        resource_estimator=FakeEstimator(fail=True),
        draft_service=FakeDraftService([]),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 128 * GiB, "available_bytes": 100 * GiB}
        },
    )
    evidence_failure_service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(RuntimeError("private evidence path")),
        runtime_capability_service=FakeCapabilities(capabilities()),
        resource_estimator=FakeEstimator(),
        draft_service=FakeDraftService([]),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 128 * GiB, "available_bytes": 100 * GiB}
        },
    )

    with database.session_factory() as db:
        draft_result = draft_failure_service.recommend(
            db, "target", "vllm", "vllm:test"
        )
        resource_result = resource_failure_service.recommend(
            db, "target", "vllm", "vllm:test"
        )
        evidence_result = evidence_failure_service.recommend(
            db, "target", "vllm", "vllm:test"
        )

    assert draft_result.status == "complete"
    assert draft_result.draft_candidates == []
    assert any("Draft" in warning for warning in draft_result.warnings)
    assert resource_result.status == "unavailable"
    assert any("resource" in warning.lower() for warning in resource_result.warnings)
    assert evidence_result.status == "unavailable"
    assert any("evidence" in warning.lower() for warning in evidence_result.warnings)
    assert all(
        "private" not in warning
        for result in (draft_result, resource_result, evidence_result)
        for warning in result.warnings
    )
    database.dispose()


def test_context_that_remains_blocked_at_minimum_is_unavailable(
    settings, tmp_path: Path
) -> None:
    database = Database(settings.database_url)
    database.create_schema()
    model_path = tmp_path / "target"
    model_path.mkdir()
    add_asset(database, model_path)
    target_evidence = evidence(
        str(model_path),
        config={"max_position_embeddings": 4096},
    )
    service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(target_evidence),
        runtime_capability_service=FakeCapabilities(capabilities()),
        resource_estimator=FakeEstimator(blocked_above=0),
        draft_service=FakeDraftService([]),
        system_snapshot=lambda: {
            "memory": {"total_bytes": 128 * GiB, "available_bytes": 100 * GiB}
        },
    )

    with database.session_factory() as db:
        result = service.recommend(db, "target", "vllm", "vllm:test")

    assert result.status == "unavailable"
    assert result.fields["context_length"].value == 1024
    assert result.fields["context_length"].source == "device_rule"
    assert any("blocked" in warning.lower() for warning in result.warnings)
    database.dispose()


def test_missing_model_returns_json_safe_unavailable_response(settings) -> None:
    database = Database(settings.database_url)
    database.create_schema()
    service = DeploymentRecommendationService(
        evidence_loader=FakeEvidenceLoader(RuntimeError()),
        runtime_capability_service=FakeCapabilities(capabilities()),
        resource_estimator=FakeEstimator(),
        draft_service=FakeDraftService([]),
        system_snapshot=lambda: {},
    )
    with database.session_factory() as db:
        result = service.recommend(db, "missing", "vllm", "vllm:test")
    assert result.status == "unavailable"
    assert result.runtime_capabilities == {}
    assert any("not found" in warning.lower() for warning in result.warnings)
    json.dumps(result.model_dump(mode="json"))
    database.dispose()
