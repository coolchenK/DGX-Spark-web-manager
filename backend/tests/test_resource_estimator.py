from __future__ import annotations

import math

import pytest
from app.services.resource_estimator import (
    MAX_KV_CACHE_BYTES,
    MAX_RESERVE_MIN_BYTES,
    ContextClampResult,
    ResourceEstimate,
    ResourceEstimator,
    clamp_context_length,
    kv_cache_bytes,
    reserve_bytes,
)

GiB = 1024**3


def test_kv_cache_formula_uses_grouped_query_attention() -> None:
    assert kv_cache_bytes(
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
        context_length=8192,
        max_concurrency=2,
    ) == 2 * 32 * 8 * (4096 // 32) * (8192 * 2) * 2


def test_kv_cache_missing_architecture_is_zero() -> None:
    assert kv_cache_bytes({}, context_length=8192, max_concurrency=2) == 0
    assert kv_cache_bytes(
        {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32},
        context_length=8192,
        max_concurrency=2,
    ) > 0
    assert kv_cache_bytes(
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": None,
        },
        context_length=8192,
        max_concurrency=2,
    ) > 0


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, math.inf, "4"])
def test_kv_cache_rejects_invalid_architecture_values(value: object) -> None:
    assert kv_cache_bytes(
        {
            "hidden_size": value,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
        },
        context_length=8192,
        max_concurrency=2,
    ) == 0


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, math.inf, "8", 10**1000],
)
def test_kv_cache_rejects_explicit_invalid_kv_heads(value: object) -> None:
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": value,
    }
    assert kv_cache_bytes(config, context_length=8192, max_concurrency=2) == 0


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, math.inf, "8"])
def test_invalid_kv_heads_lower_estimate_confidence(value: object) -> None:
    estimate = ResourceEstimator().estimate(
        model_size_bytes=1 * GiB,
        config={
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": value,
        },
        context_length=8192,
        max_concurrency=2,
        system_memory={"total_bytes": 128 * GiB},
    )
    assert estimate.kv_cache_bytes == 0
    assert estimate.confidence == "low"
    assert any("invalid" in reason.lower() for reason in estimate.reasons)


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ],
)
def test_kv_cache_rejects_extreme_architecture_values(field: str) -> None:
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }
    config[field] = 10**1000
    assert kv_cache_bytes(config, context_length=8192, max_concurrency=2) == 0


def test_kv_cache_rejects_fractional_head_dimension() -> None:
    assert kv_cache_bytes(
        {
            "hidden_size": 4097,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
        context_length=8192,
        max_concurrency=2,
    ) == 0


def test_kv_cache_checked_arithmetic_rejects_int64_overflow() -> None:
    assert kv_cache_bytes(
        {
            "hidden_size": 1_048_576,
            "num_hidden_layers": 4096,
            "num_attention_heads": 4096,
            "num_key_value_heads": 4096,
        },
        context_length=1_048_576,
        max_concurrency=1_048_576,
    ) == 0


def test_kv_overflow_lowers_estimate_confidence() -> None:
    estimate = ResourceEstimator().estimate(
        model_size_bytes=1 * GiB,
        config={
            "hidden_size": 1_048_576,
            "num_hidden_layers": 4096,
            "num_attention_heads": 4096,
            "num_key_value_heads": 4096,
        },
        context_length=1_048_576,
        max_concurrency=1_048_576,
        system_memory={"total_bytes": 128 * GiB},
    )
    assert estimate.kv_cache_bytes == MAX_KV_CACHE_BYTES
    assert estimate.confidence == "low"
    assert estimate.decision == "blocked"
    assert any("overflow" in reason.lower() for reason in estimate.reasons)
    assert any("physical" in reason.lower() for reason in estimate.reasons)


def test_reserve_bytes_uses_floor_and_minimum() -> None:
    assert reserve_bytes(128 * GiB, 0.10, 8 * GiB) == 13_743_895_347
    assert reserve_bytes(1, 0.10, 8 * GiB) == 8 * GiB


def test_resource_estimate_uses_host_memory_once() -> None:
    estimator = ResourceEstimator(reserve_fraction=0.10, reserve_min_bytes=8 * GiB)
    estimate = estimator.estimate(
        model_size_bytes=20 * GiB,
        draft_size_bytes=2 * GiB,
        config={
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
        context_length=8192,
        max_concurrency=2,
        system_memory={"total_bytes": 128 * GiB, "available_bytes": 80 * GiB},
    )
    assert estimate.total_bytes == 128 * GiB
    assert estimate.reserved_bytes == 13_743_895_347
    assert estimate.required_bytes < estimate.total_bytes
    assert estimate.decision == "ok"


def test_resource_estimate_blocks_physical_overcommit() -> None:
    estimator = ResourceEstimator(reserve_fraction=0.10, reserve_min_bytes=8 * GiB)
    estimate = estimator.estimate(
        model_size_bytes=125 * GiB,
        draft_size_bytes=0,
        config={},
        context_length=32768,
        max_concurrency=8,
        system_memory={"total_bytes": 128 * GiB, "available_bytes": 120 * GiB},
    )
    assert estimate.decision == "blocked"
    assert "physical" in estimate.reasons[0].lower()


def test_resource_estimate_warns_when_only_current_available_memory_is_short() -> None:
    estimator = ResourceEstimator(reserve_fraction=0.10, reserve_min_bytes=8 * GiB)
    estimate = estimator.estimate(
        model_size_bytes=40 * GiB,
        draft_size_bytes=0,
        config={},
        context_length=4096,
        max_concurrency=1,
        system_memory={"total_bytes": 128 * GiB, "available_bytes": 30 * GiB},
    )
    assert estimate.decision == "warning"


def test_missing_architecture_lowers_confidence_with_reason() -> None:
    estimate = ResourceEstimator().estimate(
        model_size_bytes=1 * GiB,
        config={},
        context_length=4096,
        max_concurrency=1,
        system_memory={"total_bytes": 128 * GiB},
    )
    assert estimate.confidence == "low"
    assert any("missing" in reason.lower() for reason in estimate.reasons)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_size_bytes": -1}, "model"),
        ({"context_length": 0}, "context"),
        ({"max_concurrency": True}, "concurrency"),
        ({"system_memory": {}}, "total"),
        ({"system_memory": {"total_bytes": 10, "available_bytes": -1}}, "available"),
    ],
)
def test_estimate_rejects_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    base = {
        "model_size_bytes": 1 * GiB,
        "config": {},
        "context_length": 4096,
        "max_concurrency": 1,
        "system_memory": {"total_bytes": 128 * GiB},
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ResourceEstimator().estimate(**base)


def test_available_memory_is_clamped_to_total() -> None:
    estimate = ResourceEstimator().estimate(
        model_size_bytes=1 * GiB,
        config={},
        context_length=4096,
        max_concurrency=1,
        system_memory={"total_bytes": 128 * GiB, "available_bytes": 256 * GiB},
    )
    assert estimate.available_bytes == estimate.total_bytes
    assert any("clamp" in reason.lower() for reason in estimate.reasons)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reserve_fraction": -0.1},
        {"reserve_fraction": True},
        {"reserve_fraction": math.inf},
        {"reserve_min_bytes": -1},
        {"reserve_min_bytes": True},
        {"reserve_min_bytes": 1.5},
    ],
)
def test_estimator_options_are_strictly_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ResourceEstimator(**kwargs)


def test_reserve_minimum_has_a_consistent_one_pib_upper_bound() -> None:
    assert MAX_RESERVE_MIN_BYTES == 1 << 50
    assert ResourceEstimator(reserve_min_bytes=1 << 50).reserve_min_bytes == 1 << 50
    with pytest.raises(ValueError, match="1 PiB"):
        ResourceEstimator(reserve_min_bytes=(1 << 50) + 1)


@pytest.mark.parametrize(("field", "value"), [("total_bytes", "1"), ("available_bytes", True)])
def test_resource_estimate_rejects_coerced_field_types(field: str, value: object) -> None:
    payload = {
        "total_bytes": 1,
        "available_bytes": 1,
        "reserved_bytes": 0,
        "weight_bytes": 0,
        "draft_weight_bytes": 0,
        "kv_cache_bytes": 0,
        "runtime_overhead_bytes": 0,
        "required_bytes": 0,
        "decision": "ok",
        "confidence": "high",
        "reasons": [],
    }
    payload[field] = value
    with pytest.raises(ValueError):
        ResourceEstimate.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("original_context_length", "1024"), ("final_context_length", True)],
)
def test_context_clamp_result_rejects_coerced_field_types(field: str, value: object) -> None:
    payload = {
        "original_context_length": 1024,
        "final_context_length": 1024,
        "final_decision": "ok",
        "fits": True,
        "explanation": "fits",
    }
    payload[field] = value
    with pytest.raises(ValueError):
        ContextClampResult.model_validate(payload)


def test_context_clamp_retries_until_not_blocked() -> None:
    calls: list[int] = []

    def factory(context: int) -> ResourceEstimate:
        calls.append(context)
        if context > 4096:
            return ResourceEstimate.blocked("physical memory budget exceeded")
        return ResourceEstimate(
            total_bytes=1,
            available_bytes=1,
            reserved_bytes=0,
            weight_bytes=0,
            draft_weight_bytes=0,
            kv_cache_bytes=0,
            runtime_overhead_bytes=0,
            required_bytes=0,
            decision="ok",
            confidence="high",
            reasons=[],
        )

    result = clamp_context_length(20000, 16384, factory)
    assert result.original_context_length == 20000
    assert result.final_context_length == 4096
    assert result.final_decision == "ok"
    assert result.fits is True
    assert "blocked estimates" in result.explanation
    assert calls == [16384, 8192, 4096]


def test_context_clamp_reports_hard_limit_without_claiming_blocked_estimates() -> None:
    calls: list[int] = []

    def factory(context: int) -> ResourceEstimate:
        calls.append(context)
        return ResourceEstimate(
            total_bytes=1,
            available_bytes=1,
            reserved_bytes=0,
            weight_bytes=0,
            draft_weight_bytes=0,
            kv_cache_bytes=0,
            runtime_overhead_bytes=0,
            required_bytes=0,
            decision="ok",
            confidence="high",
            reasons=[],
        )

    result = clamp_context_length(32768, 8192, factory)
    assert result.original_context_length == 32768
    assert result.final_context_length == 8192
    assert result.explanation == "context length capped at hard limit 8192"
    assert "blocked" not in result.explanation
    assert calls == [8192]


def test_context_clamp_explains_non_aligned_hard_limit() -> None:
    calls: list[int] = []

    def factory(context: int) -> ResourceEstimate:
        calls.append(context)
        return ResourceEstimate(
            total_bytes=1,
            available_bytes=1,
            reserved_bytes=0,
            weight_bytes=0,
            draft_weight_bytes=0,
            kv_cache_bytes=0,
            runtime_overhead_bytes=0,
            required_bytes=0,
            decision="ok",
            confidence="high",
            reasons=[],
        )

    result = clamp_context_length(8000, 5000, factory)
    assert result.original_context_length == 8000
    assert result.final_context_length == 4096
    assert "hard limit 5000" in result.explanation
    assert "aligned" in result.explanation
    assert "4096" in result.explanation
    assert calls == [4096]


def test_context_clamp_respects_lower_bound() -> None:
    calls: list[int] = []

    def factory(context: int) -> ResourceEstimate:
        calls.append(context)
        return ResourceEstimate.blocked("always blocked")

    result = clamp_context_length(2048, 2048, factory)
    assert result.final_context_length == 1024
    assert result.final_decision == "blocked"
    assert result.fits is False
    assert "minimum" in result.explanation
    assert "still blocked" in result.explanation
    assert calls == [2048, 1024]


def test_context_clamp_reports_requested_minimum_still_blocked() -> None:
    result = clamp_context_length(
        1024, 4096, lambda _: ResourceEstimate.blocked("physical memory exceeded")
    )
    assert result.final_context_length == 1024
    assert result.final_decision == "blocked"
    assert result.fits is False
    assert "minimum" in result.explanation
    assert "still blocked" in result.explanation
    assert "fits" not in result.explanation


def test_context_clamp_validates_requested_and_hard_limit() -> None:
    with pytest.raises(ValueError):
        clamp_context_length(0, 4096, lambda _: ResourceEstimate.blocked("x"))
    with pytest.raises(ValueError):
        clamp_context_length(4096, 512, lambda _: ResourceEstimate.blocked("x"))
