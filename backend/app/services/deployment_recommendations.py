from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)
from sqlalchemy.orm import Session

from app.models import ModelAsset
from app.runtime.base import GenerationDefaults, RecommendationSource
from app.services.draft_models import DraftCandidate
from app.services.model_evidence import ModelEvidence, ModelEvidenceLoader
from app.services.resource_estimator import (
    MAX_CONTEXT_LENGTH,
    GiB,
    ResourceEstimate,
    ResourceEstimator,
    clamp_context_length,
    reserve_bytes,
)
from app.services.runtime_capabilities import RuntimeCapabilities, RuntimeName

BoundedText = Annotated[str, StringConstraints(min_length=1, max_length=500)]
MAX_WARNINGS = 32

DEFAULT_DEPLOYMENT_VALUES: dict[str, Any] = {
    "memory_fraction": 0.8,
    "max_concurrency": 8,
    "max_batched_tokens": 8192,
    "quantization": "auto",
}

CRITICAL_FIELDS = {
    "context_length",
    "memory_fraction",
    "max_concurrency",
    "quantization",
}


class RecommendedValue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: Any
    source: RecommendationSource
    confidence: Literal["high", "medium", "low"]
    reason: BoundedText
    warning: BoundedText | None = None

    @field_validator("value")
    @classmethod
    def validate_json_value(cls, value: Any) -> Any:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("value must be JSON serializable") from exc
        return value


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str = Field(min_length=1, max_length=64)
    runtime: RuntimeName
    image: str = Field(min_length=1, max_length=500)
    provider_id: str | None = Field(default=None, min_length=1, max_length=64)


class DeploymentRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["complete", "partial", "unavailable"]
    generated_at: datetime
    model_id: str = Field(min_length=1, max_length=64)
    runtime: RuntimeName
    image_digest: str | None = Field(default=None, max_length=500)
    evidence_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    fields: dict[str, RecommendedValue]
    generation_defaults: dict[str, RecommendedValue]
    resource_snapshot: dict[str, Any]
    resource_estimate: dict[str, Any]
    runtime_capabilities: dict[str, Any]
    draft_candidates: list[DraftCandidate]
    warnings: list[BoundedText] = Field(default_factory=list, max_length=MAX_WARNINGS)

    @field_validator("generated_at")
    @classmethod
    def normalize_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("resource_snapshot", "resource_estimate", "runtime_capabilities")
    @classmethod
    def validate_json_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("mapping must be JSON serializable") from exc
        return value


class _RuntimeCapabilityService(Protocol):
    def get(self, runtime: RuntimeName, image: str) -> RuntimeCapabilities: ...


class _DraftService(Protocol):
    def list_candidates(
        self,
        db: Session,
        target: ModelAsset,
        runtime_capabilities: RuntimeCapabilities,
        system_snapshot: Mapping[str, Any] | BaseModel,
    ) -> list[DraftCandidate]: ...


class _HuggingFaceService(Protocol):
    def model_card_text(self, repository_id: str) -> str: ...


RemoteEvidenceLoader = Callable[[Path | str, str], ModelEvidence]
SystemSnapshot = Callable[[], Mapping[str, Any] | BaseModel]


def _warning(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        return "Recommendation warning"
    return normalized[:500]


def _merge_warnings(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            bounded = _warning(item)
            if bounded in seen:
                continue
            seen.add(bounded)
            merged.append(bounded)
            if len(merged) == MAX_WARNINGS:
                return merged
    return merged


def _recommended(
    value: Any,
    source: RecommendationSource,
    reason: str,
    *,
    confidence: Literal["high", "medium", "low"] = "high",
    warning: str | None = None,
) -> RecommendedValue:
    return RecommendedValue(
        value=value,
        source=source,
        confidence=confidence,
        reason=_warning(reason),
        warning=_warning(warning) if warning else None,
    )


def _valid_int(value: Any, minimum: int, maximum: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if minimum <= value <= maximum else None


def _valid_float(value: Any, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        return None
    return result


DEPLOYMENT_VALIDATORS: dict[str, Callable[[Any], Any | None]] = {
    "context_length": lambda value: _valid_int(value, 1024, MAX_CONTEXT_LENGTH),
    "memory_fraction": lambda value: _valid_float(value, 0.05, 0.98),
    "max_concurrency": lambda value: _valid_int(value, 1, 1024),
    "max_batched_tokens": lambda value: _valid_int(value, 1024, MAX_CONTEXT_LENGTH),
}


def _nested_quantization(config: Mapping[str, Any]) -> str | None:
    direct = config.get("quantization")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    quantization_config = config.get("quantization_config")
    if isinstance(quantization_config, Mapping):
        for key in ("quant_method", "quantization", "method"):
            value = quantization_config.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _select_quantization(
    evidence: ModelEvidence,
    capabilities: RuntimeCapabilities,
    runtime_defaults: Mapping[str, Any],
    detected_quantization: str | None,
) -> tuple[RecommendedValue, list[str]]:
    raw: str | None = None
    source: RecommendationSource = "runtime_default"
    reason = "Runtime default quantization"
    card_value = evidence.card_deployment_values.get("quantization")
    if isinstance(card_value, str) and card_value.strip():
        raw = card_value.strip()
        source = "model_card"
        reason = "Model card explicitly declares quantization"
    else:
        raw = _nested_quantization(evidence.config)
        if raw is not None:
            source = "local_config"
            reason = "Local config declares quantization"
        elif isinstance(detected_quantization, str) and detected_quantization.strip():
            raw = detected_quantization.strip()
            source = "local_config"
            reason = "Local model asset metadata declares quantization"
        else:
            default = runtime_defaults.get("quantization")
            raw = default if isinstance(default, str) else "auto"

    normalized = raw.casefold()
    mapped = capabilities.quantization_mapping.get(normalized, normalized)
    if mapped in capabilities.quantization_methods:
        return _recommended(mapped, source, reason), []

    warning = (
        f"Quantization {raw} maps to unsupported runtime method {mapped}; using auto"
    )
    return (
        _recommended(
            "auto",
            "device_rule",
            f"{reason}; unsupported value {raw} was clamped to auto",
            confidence="medium",
            warning=warning,
        ),
        [warning],
    )


def select_deployment_values(
    evidence: ModelEvidence,
    runtime_capabilities: RuntimeCapabilities,
    runtime_defaults: Mapping[str, Any] | None = None,
    *,
    detected_quantization: str | None = None,
) -> tuple[dict[str, RecommendedValue], list[str]]:
    """Select deployment settings from structured evidence without I/O."""
    defaults = runtime_defaults or {}
    values: dict[str, RecommendedValue] = {}
    warnings: list[str] = []

    for field, validator in DEPLOYMENT_VALIDATORS.items():
        candidates: list[tuple[Any, RecommendationSource, str, str]] = []
        if field in evidence.card_deployment_values:
            candidates.append(
                (
                    evidence.card_deployment_values[field],
                    "model_card",
                    f"Model card explicitly sets {field}",
                    "model card",
                )
            )
        if field == "context_length" and "max_position_embeddings" in evidence.config:
            candidates.append(
                (
                    evidence.config["max_position_embeddings"],
                    "local_config",
                    "Local config defines the maximum position embeddings",
                    "local config",
                )
            )
        if field in defaults:
            candidates.append(
                (
                    defaults[field],
                    "runtime_default",
                    f"Conservative runtime default for {field}",
                    "runtime default",
                )
            )

        for raw, source, reason, label in candidates:
            validated = validator(raw)
            if validated is None:
                warnings.append(f"Ignored invalid {field} from {label}")
                continue
            values[field] = _recommended(
                validated,
                source,
                reason,
                confidence="low" if source == "runtime_default" else "high",
            )
            break

    quantization, quantization_warnings = _select_quantization(
        evidence,
        runtime_capabilities,
        defaults,
        detected_quantization,
    )
    values["quantization"] = quantization
    warnings.extend(quantization_warnings)

    context = values.get("context_length")
    config_limit = _valid_int(
        evidence.config.get("max_position_embeddings"), 1024, MAX_CONTEXT_LENGTH
    )
    if context is not None and config_limit is not None and context.value > config_limit:
        original = context.value
        values["context_length"] = _recommended(
            config_limit,
            "device_rule",
            f"Selected context {original} was capped to local config hard limit {config_limit}",
            confidence="high",
        )

    return values, _merge_warnings(warnings)


def _generation_value(field: str, value: Any) -> Any | None:
    try:
        validated = GenerationDefaults.model_validate({field: value}, strict=True)
    except Exception:
        return None
    return getattr(validated, field)


def select_generation_defaults(
    evidence: ModelEvidence,
    runtime_capabilities: RuntimeCapabilities,
    runtime_defaults: Mapping[str, Any] | None = None,
) -> tuple[dict[str, RecommendedValue], list[str]]:
    """Select supported generation settings from structured evidence without I/O."""
    defaults = runtime_defaults or {}
    supported = set(runtime_capabilities.generation_defaults)
    values: dict[str, RecommendedValue] = {}
    warnings: list[str] = []
    observed = set(evidence.card_generation_values) | set(
        evidence.local_generation_values
    ) | set(defaults)

    for field in sorted(observed):
        if field not in supported:
            warnings.append(f"Generation field {field} is unsupported by this runtime")
            continue
        candidates = (
            (
                evidence.card_generation_values,
                "model_card",
                "Model card",
                "high",
            ),
            (
                evidence.local_generation_values,
                "local_config",
                "Local generation config",
                "high",
            ),
            (defaults, "runtime_default", "Runtime default", "low"),
        )
        for source_values, source, label, confidence in candidates:
            if field not in source_values:
                continue
            validated = _generation_value(field, source_values[field])
            if validated is None:
                warnings.append(f"Ignored invalid generation field {field} from {label.lower()}")
                continue
            values[field] = _recommended(
                validated,
                source,  # type: ignore[arg-type]
                f"{label} sets {field}",
                confidence=confidence,  # type: ignore[arg-type]
            )
            break
    return values, _merge_warnings(warnings)


def _snapshot_mapping(value: Mapping[str, Any] | BaseModel) -> Mapping[str, Any]:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _host_memory(snapshot: Mapping[str, Any]) -> dict[str, int]:
    memory = snapshot.get("memory")
    if not isinstance(memory, Mapping):
        raise ValueError("system memory is unavailable")
    total = _valid_int(memory.get("total_bytes"), 1, 1 << 60)
    available = _valid_int(memory.get("available_bytes"), 0, 1 << 60)
    if total is None or available is None:
        raise ValueError("system memory is invalid")
    return {"total_bytes": total, "available_bytes": min(available, total)}


class DeploymentRecommendationService:
    def __init__(
        self,
        *,
        evidence_loader: ModelEvidenceLoader,
        runtime_capability_service: _RuntimeCapabilityService,
        resource_estimator: ResourceEstimator,
        draft_service: _DraftService,
        system_snapshot: SystemSnapshot,
        huggingface_service: _HuggingFaceService | None = None,
        remote_evidence_loader: RemoteEvidenceLoader | None = None,
        runtime_defaults: Mapping[str, Any] | None = None,
        generation_runtime_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.evidence_loader = evidence_loader
        self.runtime_capability_service = runtime_capability_service
        self.resource_estimator = resource_estimator
        self.draft_service = draft_service
        self.system_snapshot = system_snapshot
        self.huggingface_service = huggingface_service
        self.remote_evidence_loader = remote_evidence_loader
        self.runtime_defaults = dict(runtime_defaults or DEFAULT_DEPLOYMENT_VALUES)
        self.generation_runtime_defaults = dict(generation_runtime_defaults or {})

    @staticmethod
    def _unavailable(
        request: RecommendationRequest,
        warning: str,
        *,
        image_digest: str | None = None,
        evidence_hash: str | None = None,
        capabilities: RuntimeCapabilities | None = None,
        fields: dict[str, RecommendedValue] | None = None,
        generation_defaults: dict[str, RecommendedValue] | None = None,
        resource_snapshot: dict[str, Any] | None = None,
        resource_estimate: ResourceEstimate | None = None,
        warnings: list[str] | None = None,
    ) -> DeploymentRecommendation:
        return DeploymentRecommendation(
            status="unavailable",
            generated_at=datetime.now(UTC),
            model_id=request.model_id,
            runtime=request.runtime,
            image_digest=image_digest,
            evidence_hash=evidence_hash,
            fields=fields or {},
            generation_defaults=generation_defaults or {},
            resource_snapshot=resource_snapshot or {},
            resource_estimate=(
                resource_estimate.model_dump(mode="json") if resource_estimate else {}
            ),
            runtime_capabilities=(
                capabilities.model_dump(mode="json") if capabilities else {}
            ),
            draft_candidates=[],
            warnings=_merge_warnings(warnings or [], [warning]),
        )

    def _load_evidence(
        self, target: ModelAsset
    ) -> tuple[ModelEvidence, list[str]]:
        evidence = self.evidence_loader.load(target.local_path)
        warnings: list[str] = []
        if (
            not evidence.card_text
            and target.repository_id
            and self.huggingface_service is not None
        ):
            try:
                remote_card = self.huggingface_service.model_card_text(
                    target.repository_id
                )
                if remote_card and self.remote_evidence_loader is not None:
                    evidence = self.remote_evidence_loader(
                        target.local_path, remote_card
                    )
                elif remote_card:
                    warnings.append(
                        "Remote model card was available but no safe evidence adapter is configured"
                    )
            except Exception:
                warnings.append("Remote model card fallback failed")
        return evidence, warnings

    def recommend(
        self,
        db: Session,
        model_id: str,
        runtime: RuntimeName,
        image: str,
        provider: str | None = None,
    ) -> DeploymentRecommendation:
        request = RecommendationRequest(
            model_id=model_id,
            runtime=runtime,
            image=image,
            provider_id=provider,
        )
        target = db.get(ModelAsset, request.model_id)
        if target is None:
            return self._unavailable(request, "Model asset was not found")
        if target.status != "available":
            return self._unavailable(request, "Model asset is unavailable")
        if not target.local_path:
            return self._unavailable(request, "Model asset path is missing")

        try:
            capabilities = self.runtime_capability_service.get(runtime, image)
        except Exception:
            return self._unavailable(
                request, "Runtime capabilities could not be verified"
            )

        try:
            evidence, remote_warnings = self._load_evidence(target)
        except Exception:
            return self._unavailable(
                request,
                "Model evidence could not be verified",
                image_digest=capabilities.image_digest,
                capabilities=capabilities,
            )

        fields, field_warnings = select_deployment_values(
            evidence,
            capabilities,
            self.runtime_defaults,
            detected_quantization=target.quantization,
        )
        generation, generation_warnings = select_generation_defaults(
            evidence,
            capabilities,
            self.generation_runtime_defaults,
        )
        warnings = _merge_warnings(
            evidence.warnings,
            capabilities.warnings,
            remote_warnings,
            field_warnings,
            generation_warnings,
        )

        try:
            raw_snapshot = self.system_snapshot()
            snapshot = _snapshot_mapping(raw_snapshot)
            memory = _host_memory(snapshot)
        except Exception:
            return self._unavailable(
                request,
                "Unified memory resource snapshot could not be verified",
                image_digest=capabilities.image_digest,
                evidence_hash=evidence.evidence_hash,
                capabilities=capabilities,
                fields=fields,
                generation_defaults=generation,
                warnings=warnings,
            )

        estimate: ResourceEstimate | None = None
        reserve_fraction = getattr(self.resource_estimator, "reserve_fraction", 0.10)
        reserve_minimum = getattr(
            self.resource_estimator, "reserve_min_bytes", 8 * GiB
        )
        try:
            reserved = reserve_bytes(
                memory["total_bytes"], float(reserve_fraction), int(reserve_minimum)
            )
        except (TypeError, ValueError, OverflowError):
            reserved = reserve_bytes(memory["total_bytes"], 0.10, 8 * GiB)
        resource_snapshot: dict[str, Any] = {**memory, "reserved_bytes": reserved}
        deployments = snapshot.get("deployments")
        if isinstance(deployments, (list, dict)):
            try:
                json.dumps(deployments, allow_nan=False)
            except (TypeError, ValueError):
                pass
            else:
                resource_snapshot["deployments"] = deployments
        context_value = fields.get("context_length")
        concurrency_value = fields.get("max_concurrency")
        if context_value is not None and concurrency_value is not None:
            estimates: dict[int, ResourceEstimate] = {}

            def estimate_context(context: int) -> ResourceEstimate:
                estimate_result = self.resource_estimator.estimate(
                    model_size_bytes=target.size_bytes,
                    config=evidence.config,
                    context_length=context,
                    max_concurrency=concurrency_value.value,
                    system_memory=memory,
                    draft_size_bytes=0,
                )
                estimates[context] = estimate_result
                return estimate_result

            hard_limit = _valid_int(
                evidence.config.get("max_position_embeddings"),
                1024,
                MAX_CONTEXT_LENGTH,
            ) or context_value.value
            try:
                clamp = clamp_context_length(
                    context_value.value,
                    hard_limit,
                    estimate_context,
                )
                estimate = estimates[clamp.final_context_length]
            except Exception:
                return self._unavailable(
                    request,
                    "Deployment resource requirements could not be verified",
                    image_digest=capabilities.image_digest,
                    evidence_hash=evidence.evidence_hash,
                    capabilities=capabilities,
                    fields=fields,
                    generation_defaults=generation,
                    resource_snapshot=resource_snapshot,
                    warnings=warnings,
                )
            resource_snapshot["reserved_bytes"] = estimate.reserved_bytes
            if clamp.final_context_length != context_value.value:
                original = context_value.value
                fields["context_length"] = _recommended(
                    clamp.final_context_length,
                    "device_rule",
                    f"{context_value.reason}; resource rule reduced {original} to "
                    f"{clamp.final_context_length}: {clamp.explanation}",
                    confidence="high" if clamp.fits else "low",
                )
            if not clamp.fits or estimate.decision == "blocked":
                return self._unavailable(
                    request,
                    "Deployment is blocked by unified memory requirements",
                    image_digest=capabilities.image_digest,
                    evidence_hash=evidence.evidence_hash,
                    capabilities=capabilities,
                    fields=fields,
                    generation_defaults=generation,
                    resource_snapshot=resource_snapshot,
                    resource_estimate=estimate,
                    warnings=warnings,
                )
            if estimate.decision == "warning":
                warnings = _merge_warnings(
                    warnings,
                    ["Current available unified memory requires deployment review"],
                )

        try:
            candidates = self.draft_service.list_candidates(
                db, target, capabilities, raw_snapshot
            )
        except Exception:
            candidates = []
            warnings = _merge_warnings(
                warnings, ["Draft Model candidates could not be evaluated"]
            )

        unresolved = sorted(CRITICAL_FIELDS - set(fields))
        if unresolved:
            status: Literal["complete", "partial"] = "partial"
            warnings = _merge_warnings(
                warnings,
                [
                    "AI analysis may complete unresolved deployment fields: "
                    + ", ".join(unresolved)
                ],
            )
        else:
            status = "complete"

        return DeploymentRecommendation(
            status=status,
            generated_at=datetime.now(UTC),
            model_id=request.model_id,
            runtime=request.runtime,
            image_digest=capabilities.image_digest,
            evidence_hash=evidence.evidence_hash,
            fields=fields,
            generation_defaults=generation,
            resource_snapshot=resource_snapshot,
            resource_estimate=(estimate.model_dump(mode="json") if estimate else {}),
            runtime_capabilities=capabilities.model_dump(mode="json"),
            draft_candidates=candidates,
            warnings=warnings,
        )
