from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ModelAsset
from app.services.model_evidence import ModelEvidence, ModelEvidenceLoader
from app.services.resource_estimator import (
    MAX_CONTEXT_LENGTH,
    MAX_SIZE_BYTES,
    ResourceEstimate,
    ResourceEstimator,
)
from app.services.runtime_capabilities import RuntimeCapabilities

SpeculativeMethod = Literal["draft_model", "eagle", "eagle3", "mtp"]
CandidateStatus = Literal["compatible", "review", "incompatible"]
BoundedReason = Annotated[str, StringConstraints(min_length=1, max_length=240)]

KNOWN_METHODS = {"draft_model", "eagle", "eagle3", "mtp"}
AUXILIARY_METHODS = {"eagle", "eagle3", "mtp"}
STATUS_ORDER: dict[CandidateStatus, int] = {
    "compatible": 0,
    "review": 1,
    "incompatible": 2,
}


class DraftCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str
    name: str
    repository_id: str | None
    method: SpeculativeMethod | None
    status: CandidateStatus
    reasons: list[BoundedReason] = Field(max_length=12)
    size_bytes: int = Field(ge=0, le=MAX_SIZE_BYTES)
    estimated_total_bytes: int | None = Field(default=None, ge=0)


def _candidate(
    draft: ModelAsset,
    *,
    method: SpeculativeMethod | None,
    status: CandidateStatus,
    reasons: list[str],
    resource_estimate: ResourceEstimate | None,
) -> DraftCandidate:
    return DraftCandidate(
        model_id=draft.id,
        name=draft.name,
        repository_id=draft.repository_id,
        method=method,
        status=status,
        reasons=reasons,
        size_bytes=draft.size_bytes,
        estimated_total_bytes=(
            resource_estimate.required_bytes if resource_estimate is not None else None
        ),
    )


def classify_draft_candidate(
    target: ModelAsset,
    target_evidence: ModelEvidence | None,
    draft: ModelAsset,
    draft_evidence: ModelEvidence | None,
    supported_methods: Iterable[str],
    resource_estimate: ResourceEstimate | None = None,
) -> DraftCandidate:
    """Classify a local Draft Model from bounded, structured evidence only."""
    supported = frozenset(supported_methods)
    raw_method = draft_evidence.speculative_method if draft_evidence is not None else None
    method = cast(SpeculativeMethod, raw_method) if raw_method in KNOWN_METHODS else None
    hard_reasons: list[str] = []

    if target.id == draft.id:
        hard_reasons.append("Target and Draft Model are the same model")
    if target.status != "available":
        hard_reasons.append("Target model is unavailable")
    if draft.status != "available":
        hard_reasons.append("Draft Model is unavailable")
    if not target.local_path:
        hard_reasons.append("Target model path is missing")
    if not draft.local_path:
        hard_reasons.append("Draft Model path is missing")
    if target_evidence is None:
        hard_reasons.append("Target model evidence is unreadable")
    if draft_evidence is None:
        hard_reasons.append("Draft Model evidence is unreadable")

    if draft_evidence is not None:
        if raw_method is None:
            hard_reasons.append("Draft Model method is missing")
        elif raw_method not in KNOWN_METHODS:
            hard_reasons.append("Draft Model method is unsupported")
        elif raw_method not in supported:
            hard_reasons.append("Draft Model method is unsupported by this runtime")

    explicit_match = False
    target_repository_missing = False
    if draft_evidence is not None and draft_evidence.target_model_ids:
        if target.repository_id is None:
            target_repository_missing = True
        elif target.repository_id in draft_evidence.target_model_ids:
            explicit_match = True
        else:
            hard_reasons.append("Draft Model declares a different target model")

    if resource_estimate is not None and resource_estimate.decision == "blocked":
        hard_reasons.append("Combined model memory requirement is blocked")

    if hard_reasons:
        return _candidate(
            draft,
            method=method,
            status="incompatible",
            reasons=hard_reasons,
            resource_estimate=resource_estimate,
        )

    assert draft_evidence is not None
    assert target_evidence is not None
    assert method is not None

    if method in AUXILIARY_METHODS and explicit_match:
        return _candidate(
            draft,
            method=method,
            status="compatible",
            reasons=["Auxiliary Draft Model explicitly matches the target repository"],
            resource_estimate=resource_estimate,
        )

    if method == "draft_model":
        target_fingerprint = target_evidence.tokenizer_fingerprint
        draft_fingerprint = draft_evidence.tokenizer_fingerprint
        if (
            target_fingerprint is not None
            and draft_fingerprint is not None
            and target_fingerprint != draft_fingerprint
        ):
            return _candidate(
                draft,
                method=method,
                status="incompatible",
                reasons=["Target and Draft Model tokenizer fingerprints conflict"],
                resource_estimate=resource_estimate,
            )
        if (
            explicit_match
            and target_fingerprint is not None
            and target_fingerprint == draft_fingerprint
        ):
            return _candidate(
                draft,
                method=method,
                status="compatible",
                reasons=["Draft Model explicitly matches the target and tokenizer"],
                resource_estimate=resource_estimate,
            )
        if target_fingerprint is None or draft_fingerprint is None:
            review_reason = "Tokenizer fingerprint evidence is incomplete"
        elif not draft_evidence.target_model_ids:
            review_reason = "Draft Model target pairing evidence is missing"
        else:
            review_reason = "Draft Model pairing requires manual review"
    elif target_repository_missing:
        review_reason = "Target repository ID is missing; explicit pairing cannot be verified"
    else:
        review_reason = "Auxiliary Draft Model target pairing evidence is missing"

    return _candidate(
        draft,
        method=method,
        status="review",
        reasons=[review_reason],
        resource_estimate=resource_estimate,
    )


def _system_memory(system_snapshot: Mapping[str, Any] | BaseModel) -> Mapping[str, Any]:
    if isinstance(system_snapshot, BaseModel):
        value = system_snapshot.model_dump()
    else:
        value = system_snapshot
    memory = value.get("memory")
    return memory if isinstance(memory, Mapping) else value


def _context_length(evidence: ModelEvidence) -> int:
    value = evidence.config.get("max_position_embeddings")
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_CONTEXT_LENGTH:
        return value
    return 4096


class DraftCompatibilityService:
    def __init__(
        self,
        *,
        evidence_loader: ModelEvidenceLoader,
        resource_estimator: ResourceEstimator | None = None,
    ) -> None:
        self.evidence_loader = evidence_loader
        self.resource_estimator = resource_estimator or ResourceEstimator()

    def _estimate(
        self,
        target: ModelAsset,
        target_evidence: ModelEvidence,
        draft: ModelAsset,
        system_snapshot: Mapping[str, Any] | BaseModel,
    ) -> ResourceEstimate | None:
        try:
            return self.resource_estimator.estimate(
                model_size_bytes=target.size_bytes,
                config=target_evidence.config,
                context_length=_context_length(target_evidence),
                max_concurrency=1,
                system_memory=_system_memory(system_snapshot),
                draft_size_bytes=draft.size_bytes,
            )
        except Exception:
            return None

    def list_candidates(
        self,
        db: Session,
        target: ModelAsset,
        runtime_capabilities: RuntimeCapabilities,
        system_snapshot: Mapping[str, Any] | BaseModel,
    ) -> list[DraftCandidate]:
        assets = list(db.scalars(select(ModelAsset)).all())
        try:
            target_evidence = self.evidence_loader.load(target.local_path)
        except Exception:
            target_evidence = None

        candidates: list[DraftCandidate] = []
        for draft in assets:
            if draft.id == target.id:
                draft_evidence = target_evidence
            else:
                try:
                    draft_evidence = self.evidence_loader.load(draft.local_path)
                except Exception:
                    draft_evidence = None
            resource_estimate = (
                self._estimate(target, target_evidence, draft, system_snapshot)
                if target_evidence is not None and draft_evidence is not None
                else None
            )
            candidates.append(
                classify_draft_candidate(
                    target,
                    target_evidence,
                    draft,
                    draft_evidence,
                    supported_methods=runtime_capabilities.speculative_methods,
                    resource_estimate=resource_estimate,
                )
            )

        return sorted(
            candidates,
            key=lambda candidate: (
                STATUS_ORDER[candidate.status],
                candidate.size_bytes,
                candidate.name.casefold(),
                candidate.name,
                candidate.model_id,
            ),
        )
