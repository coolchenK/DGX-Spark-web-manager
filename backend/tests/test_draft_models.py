from __future__ import annotations

from pathlib import Path

import pytest
from app.db import Database
from app.models import ModelAsset
from app.services.draft_models import (
    DraftCandidate,
    DraftCompatibilityService,
    classify_draft_candidate,
)
from app.services.model_evidence import ModelEvidence
from app.services.resource_estimator import ResourceEstimate
from app.services.runtime_capabilities import RuntimeCapabilities
from pydantic import ValidationError

GiB = 1024**3


def asset(
    repository_id: str | None,
    *,
    model_id: str,
    name: str | None = None,
    local_path: str | None = None,
    status: str = "available",
    size_bytes: int = GiB,
) -> ModelAsset:
    return ModelAsset(
        id=model_id,
        name=name or repository_id or model_id,
        source="huggingface",
        repository_id=repository_id,
        local_path=local_path if local_path is not None else f"/models/{model_id}",
        status=status,
        size_bytes=size_bytes,
    )


def evidence(
    model: ModelAsset,
    *,
    tokenizer: str | None = "tokenizer",
    targets: list[str] | None = None,
    method: str | None = None,
) -> ModelEvidence:
    return ModelEvidence(
        model_path=model.local_path,
        config={"max_position_embeddings": 8192},
        generation_config={},
        tokenizer_fingerprint=tokenizer,
        card_text="",
        card_data={},
        card_deployment_values={},
        card_generation_values={},
        local_generation_values={},
        target_model_ids=targets or [],
        speculative_method=method,
        evidence_hash="a" * 64,
        warnings=[],
    )


def estimate(decision: str = "ok", required_bytes: int = 4 * GiB) -> ResourceEstimate:
    return ResourceEstimate(
        total_bytes=128 * GiB,
        available_bytes=120 * GiB,
        reserved_bytes=8 * GiB,
        weight_bytes=2 * GiB,
        draft_weight_bytes=GiB,
        kv_cache_bytes=0,
        runtime_overhead_bytes=GiB,
        required_bytes=required_bytes,
        decision=decision,
        confidence="high",
        reasons=[],
    )


def classify(
    target: ModelAsset,
    draft: ModelAsset,
    *,
    target_evidence: ModelEvidence | None = None,
    draft_evidence: ModelEvidence | None = None,
    supported: set[str] | None = None,
    resource: ResourceEstimate | None = None,
) -> DraftCandidate:
    return classify_draft_candidate(
        target,
        target_evidence if target_evidence is not None else evidence(target),
        draft,
        draft_evidence,
        supported_methods=supported or {"draft_model", "eagle", "eagle3", "mtp"},
        resource_estimate=resource,
    )


def test_explicit_eagle3_target_match_is_compatible_despite_tokenizer_difference() -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/Target-8B-EAGLE3", model_id="draft")

    result = classify(
        target,
        draft,
        target_evidence=evidence(target, tokenizer="target"),
        draft_evidence=evidence(
            draft,
            tokenizer="draft",
            targets=["org/Target-8B"],
            method="eagle3",
        ),
        supported={"eagle3"},
    )

    assert result.status == "compatible"
    assert result.method == "eagle3"


def test_explicit_target_mismatch_is_incompatible() -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/Other-EAGLE3", model_id="draft")

    result = classify(
        target,
        draft,
        draft_evidence=evidence(
            draft,
            targets=["org/Other-8B"],
            method="eagle3",
        ),
        supported={"eagle3"},
    )

    assert result.status == "incompatible"
    assert any("target" in reason.lower() for reason in result.reasons)


def test_repository_target_matching_is_case_sensitive() -> None:
    target = asset("Org/Target-8B", model_id="target")
    draft = asset("org/eagle", model_id="draft")

    result = classify(
        target,
        draft,
        draft_evidence=evidence(
            draft,
            targets=["org/target-8b"],
            method="eagle",
        ),
        supported={"eagle"},
    )

    assert result.status == "incompatible"


def test_target_without_repository_id_cannot_be_implicitly_matched() -> None:
    target = asset(None, model_id="target", name="Target-8B")
    draft = asset("org/eagle", model_id="draft")

    result = classify(
        target,
        draft,
        draft_evidence=evidence(
            draft,
            targets=["org/Target-8B"],
            method="eagle",
        ),
        supported={"eagle"},
    )

    assert result.status == "review"
    assert any("repository" in reason.lower() for reason in result.reasons)


def test_same_tokenizer_standalone_draft_without_explicit_pair_is_review() -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/Target-0.5B", model_id="draft")

    result = classify(
        target,
        draft,
        target_evidence=evidence(target, tokenizer="same"),
        draft_evidence=evidence(draft, tokenizer="same", method="draft_model"),
        supported={"draft_model"},
    )

    assert result.status == "review"


def test_standalone_explicit_pair_with_same_tokenizer_is_compatible() -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/Target-0.5B", model_id="draft")

    result = classify(
        target,
        draft,
        target_evidence=evidence(target, tokenizer="same"),
        draft_evidence=evidence(
            draft,
            tokenizer="same",
            targets=["org/Target-8B"],
            method="draft_model",
        ),
        supported={"draft_model"},
    )

    assert result.status == "compatible"


def test_standalone_tokenizer_conflict_is_incompatible_even_with_explicit_pair() -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/Target-0.5B", model_id="draft")

    result = classify(
        target,
        draft,
        target_evidence=evidence(target, tokenizer="target"),
        draft_evidence=evidence(
            draft,
            tokenizer="draft",
            targets=["org/Target-8B"],
            method="draft_model",
        ),
        supported={"draft_model"},
    )

    assert result.status == "incompatible"
    assert any("tokenizer" in reason.lower() for reason in result.reasons)


def test_missing_tokenizer_fingerprint_requires_review() -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/Target-0.5B", model_id="draft")

    result = classify(
        target,
        draft,
        target_evidence=evidence(target, tokenizer=None),
        draft_evidence=evidence(
            draft,
            tokenizer=None,
            targets=["org/Target-8B"],
            method="draft_model",
        ),
        supported={"draft_model"},
    )

    assert result.status == "review"
    assert any("tokenizer" in reason.lower() for reason in result.reasons)


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        ("same_model", "same model"),
        ("target_unavailable", "target model is unavailable"),
        ("draft_unavailable", "draft model is unavailable"),
        ("target_path", "target model path"),
        ("draft_path", "draft model path"),
        ("missing_method", "method"),
        ("unsupported_method", "runtime"),
        ("unreadable_evidence", "evidence"),
    ],
)
def test_hard_incompatibilities_are_reported_deterministically(
    mutation: str, reason_fragment: str
) -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/draft", model_id="draft")
    draft_ev: ModelEvidence | None = evidence(
        draft, targets=["org/Target-8B"], method="eagle3"
    )
    supported = {"eagle3"}
    if mutation == "same_model":
        draft.id = target.id
    elif mutation == "target_unavailable":
        target.status = "unavailable"
    elif mutation == "draft_unavailable":
        draft.status = "unavailable"
    elif mutation == "target_path":
        target.local_path = ""
    elif mutation == "draft_path":
        draft.local_path = ""
    elif mutation == "missing_method":
        assert draft_ev is not None
        draft_ev = draft_ev.model_copy(update={"speculative_method": None})
    elif mutation == "unsupported_method":
        supported = {"draft_model"}
    elif mutation == "unreadable_evidence":
        draft_ev = None

    result = classify(
        target,
        draft,
        draft_evidence=draft_ev,
        supported=supported,
    )

    assert result.status == "incompatible"
    assert any(reason_fragment in reason.lower() for reason in result.reasons)


def test_blocked_resource_estimate_overrides_explicit_compatibility() -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/eagle", model_id="draft")

    result = classify(
        target,
        draft,
        draft_evidence=evidence(
            draft,
            targets=["org/Target-8B"],
            method="eagle",
        ),
        supported={"eagle"},
        resource=estimate("blocked", 140 * GiB),
    )

    assert result.status == "incompatible"
    assert result.estimated_total_bytes == 140 * GiB
    assert any("memory" in reason.lower() for reason in result.reasons)


def test_warning_resource_estimate_does_not_block_candidate() -> None:
    target = asset("org/Target-8B", model_id="target")
    draft = asset("org/eagle", model_id="draft")

    result = classify(
        target,
        draft,
        draft_evidence=evidence(
            draft,
            targets=["org/Target-8B"],
            method="eagle",
        ),
        supported={"eagle"},
        resource=estimate("warning", 100 * GiB),
    )

    assert result.status == "compatible"
    assert result.estimated_total_bytes == 100 * GiB


def test_draft_candidate_is_strict_and_bounds_reasons() -> None:
    payload = {
        "model_id": "draft",
        "name": "Draft",
        "repository_id": None,
        "method": "eagle3",
        "status": "compatible",
        "reasons": ["explicit target match"],
        "size_bytes": 1,
        "estimated_total_bytes": None,
    }
    with pytest.raises(ValidationError):
        DraftCandidate.model_validate({**payload, "unknown": True})
    with pytest.raises(ValidationError):
        DraftCandidate.model_validate({**payload, "size_bytes": "1"})
    with pytest.raises(ValidationError):
        DraftCandidate.model_validate({**payload, "reasons": ["x" * 241]})


class FakeEvidenceLoader:
    def __init__(self, evidence_by_path: dict[str, ModelEvidence | Exception]):
        self.evidence_by_path = evidence_by_path
        self.calls: list[str] = []

    def load(self, model_path: Path | str) -> ModelEvidence:
        path = str(model_path)
        self.calls.append(path)
        value = self.evidence_by_path[path]
        if isinstance(value, Exception):
            raise value
        return value


class FakeResourceEstimator:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def estimate(self, **kwargs: object) -> ResourceEstimate:
        self.calls.append(kwargs)
        draft_size = int(kwargs["draft_size_bytes"])
        if draft_size == 3 * GiB:
            raise RuntimeError("estimator failed")
        decision = "blocked" if draft_size >= 12 * GiB else "ok"
        return estimate(decision, required_bytes=20 * GiB + draft_size)


def capabilities() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        runtime="vllm",
        image="vllm:test",
        image_digest="sha256:test",
        source="probe",
        generation_defaults=[],
        quantization_methods=[],
        quantization_mapping={},
        speculative_methods=["draft_model", "eagle3"],
        method_mapping={"draft_model": "draft_model", "eagle3": "eagle3"},
        speculative_transport="json",
        warnings=[],
    )


def test_service_lists_all_assets_isolates_evidence_failures_and_sorts(settings) -> None:
    database = Database(settings.database_url)
    database.create_schema()
    target = asset(
        "org/Target-8B", model_id="target", local_path="/models/target", size_bytes=8 * GiB
    )
    compatible = asset(
        "org/EAGLE3", model_id="compatible", local_path="/models/eagle", size_bytes=2 * GiB
    )
    review_large = asset(
        "org/Draft-Z", model_id="review-z", local_path="/models/review-z", size_bytes=3 * GiB
    )
    review_small_b = asset(
        "org/beta", model_id="review-b", local_path="/models/review-b", size_bytes=GiB
    )
    review_small_a = asset(
        "org/Alpha", model_id="review-a", local_path="/models/review-a", size_bytes=GiB
    )
    blocked = asset(
        "org/Huge", model_id="blocked", local_path="/models/huge", size_bytes=12 * GiB
    )
    unreadable = asset(
        "org/Broken", model_id="broken", local_path="/secret/private/model", size_bytes=4 * GiB
    )
    unavailable = asset(
        "org/Unavailable",
        model_id="unavailable",
        local_path="/models/unavailable",
        status="unavailable",
        size_bytes=5 * GiB,
    )
    assets = [
        target,
        compatible,
        review_large,
        review_small_b,
        review_small_a,
        blocked,
        unreadable,
        unavailable,
    ]
    target_ev = evidence(target, tokenizer="same")
    loader = FakeEvidenceLoader(
        {
            target.local_path: target_ev,
            compatible.local_path: evidence(
                compatible,
                tokenizer="different",
                targets=["org/Target-8B"],
                method="eagle3",
            ),
            review_large.local_path: evidence(
                review_large, tokenizer="same", method="draft_model"
            ),
            review_small_b.local_path: evidence(
                review_small_b, tokenizer="same", method="draft_model"
            ),
            review_small_a.local_path: evidence(
                review_small_a, tokenizer="same", method="draft_model"
            ),
            blocked.local_path: evidence(
                blocked,
                tokenizer="same",
                targets=["org/Target-8B"],
                method="draft_model",
            ),
            unreadable.local_path: RuntimeError("leaked /secret/private/model card"),
            unavailable.local_path: evidence(
                unavailable,
                targets=["org/Target-8B"],
                method="draft_model",
            ),
        }
    )
    estimator = FakeResourceEstimator()
    service = DraftCompatibilityService(
        evidence_loader=loader,
        resource_estimator=estimator,
    )

    with database.session_factory() as db:
        db.add_all(assets)
        db.commit()
        candidates = service.list_candidates(
            db,
            target,
            capabilities(),
            {"memory": {"total_bytes": 128 * GiB, "available_bytes": 120 * GiB}},
        )

    assert [candidate.model_id for candidate in candidates] == [
        "compatible",
        "review-a",
        "review-b",
        "review-z",
        "broken",
        "unavailable",
        "target",
        "blocked",
    ]
    by_id = {candidate.model_id: candidate for candidate in candidates}
    assert by_id["blocked"].estimated_total_bytes == 32 * GiB
    assert by_id["broken"].status == "incompatible"
    assert all("secret" not in reason for reason in by_id["broken"].reasons)
    assert by_id["unavailable"].status == "incompatible"
    assert len(loader.calls) == len(assets)
    assert any(
        call["model_size_bytes"] == 8 * GiB
        and call["draft_size_bytes"] == 2 * GiB
        for call in estimator.calls
    )
    database.dispose()
