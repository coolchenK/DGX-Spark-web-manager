from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GiB = 1024**3
RUNTIME_WORKSPACE_BYTES = 2 * GiB
MAX_CONTEXT_LENGTH = 1_048_576
MAX_CONCURRENCY = 1_048_576
MAX_SIZE_BYTES = 1 << 60
MAX_RESERVE_MIN_BYTES = 1 << 60
MAX_HIDDEN_SIZE = 1_048_576
MAX_NUM_LAYERS = 4096
MAX_NUM_HEADS = 4096
MAX_KV_CACHE_BYTES = (1 << 63) - 1


def _positive_int(value: Any, *, maximum: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    if maximum is not None and value > maximum:
        return None
    return value


def _checked_product(values: tuple[int, ...], *, maximum: int) -> int | None:
    result = 1
    for value in values:
        if value > maximum // result:
            return None
        result *= value
    return result


def _bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _size_bytes(value: Any, *, name: str, allow_zero: bool = True) -> int:
    minimum = 0 if allow_zero else 1
    return _bounded_int(value, name=name, minimum=minimum, maximum=MAX_SIZE_BYTES)


def kv_cache_bytes(
    config: Mapping[str, Any] | None,
    context_length: int,
    max_concurrency: int,
) -> int:
    """Estimate fp16 KV cache bytes for one request batch.

    Missing or malformed architecture metadata deliberately returns zero. The
    caller can then lower confidence and surface that the estimate is partial.
    """
    if not isinstance(config, Mapping):
        return 0
    hidden = _positive_int(config.get("hidden_size"), maximum=MAX_HIDDEN_SIZE)
    layers = _positive_int(config.get("num_hidden_layers"), maximum=MAX_NUM_LAYERS)
    attention_heads = _positive_int(
        config.get("num_attention_heads"), maximum=MAX_NUM_HEADS
    )
    raw_kv_heads = config.get("num_key_value_heads")
    kv_heads = _positive_int(raw_kv_heads, maximum=MAX_NUM_HEADS)
    context = _positive_int(context_length, maximum=MAX_CONTEXT_LENGTH)
    concurrency = _positive_int(max_concurrency, maximum=MAX_CONCURRENCY)
    if (
        not hidden
        or not layers
        or not attention_heads
        or not context
        or not concurrency
    ):
        return 0
    if _positive_int(raw_kv_heads) and kv_heads is None:
        return 0
    kv_heads = kv_heads or attention_heads
    if hidden % attention_heads != 0:
        return 0
    head_dim = hidden // attention_heads
    if head_dim <= 0:
        return 0
    result = _checked_product(
        (2, layers, kv_heads, head_dim, context, concurrency, 2),
        maximum=MAX_KV_CACHE_BYTES,
    )
    return result or 0


def reserve_bytes(total_bytes: int, fraction: float, minimum: int) -> int:
    """Reserve the larger of a fraction (floored) and a fixed minimum."""
    return max(minimum, int(total_bytes * fraction))


class ResourceEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    reserved_bytes: int = Field(ge=0)
    weight_bytes: int = Field(ge=0)
    draft_weight_bytes: int = Field(ge=0)
    kv_cache_bytes: int = Field(ge=0)
    runtime_overhead_bytes: int = Field(ge=0)
    required_bytes: int = Field(ge=0)
    decision: Literal["ok", "warning", "blocked"]
    confidence: Literal["high", "low"]
    reasons: list[str] = Field(default_factory=list)

    @classmethod
    def blocked(cls, reason: str) -> ResourceEstimate:
        return cls(
            total_bytes=0,
            available_bytes=0,
            reserved_bytes=0,
            weight_bytes=0,
            draft_weight_bytes=0,
            kv_cache_bytes=0,
            runtime_overhead_bytes=0,
            required_bytes=0,
            decision="blocked",
            confidence="low",
            reasons=[reason],
        )


class ContextClampResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_context_length: int = Field(ge=1024)
    final_context_length: int = Field(ge=1024)
    explanation: str


class ResourceEstimator:
    def __init__(
        self,
        *,
        reserve_fraction: float = 0.10,
        reserve_min_bytes: int = 8 * GiB,
    ) -> None:
        if (
            isinstance(reserve_fraction, bool)
            or not isinstance(reserve_fraction, (int, float))
            or not math.isfinite(float(reserve_fraction))
            or not 0.05 <= float(reserve_fraction) <= 0.30
        ):
            raise ValueError("reserve_fraction must be finite and between 0.05 and 0.30")
        if (
            isinstance(reserve_min_bytes, bool)
            or not isinstance(reserve_min_bytes, int)
            or not 1 * GiB <= reserve_min_bytes <= MAX_RESERVE_MIN_BYTES
        ):
            raise ValueError("reserve_min_bytes must be an integer between 1 GiB and 1 PiB")
        self.reserve_fraction = float(reserve_fraction)
        self.reserve_min_bytes = reserve_min_bytes

    def estimate(
        self,
        *,
        model_size_bytes: int,
        config: Mapping[str, Any] | None,
        context_length: int,
        max_concurrency: int,
        system_memory: Mapping[str, Any],
        draft_size_bytes: int = 0,
    ) -> ResourceEstimate:
        model_size = _size_bytes(model_size_bytes, name="model_size_bytes")
        draft_size = _size_bytes(draft_size_bytes, name="draft_size_bytes")
        context = _bounded_int(
            context_length,
            name="context_length",
            minimum=1,
            maximum=MAX_CONTEXT_LENGTH,
        )
        concurrency = _bounded_int(
            max_concurrency,
            name="max_concurrency",
            minimum=1,
            maximum=MAX_CONCURRENCY,
        )
        if not isinstance(system_memory, Mapping):
            raise ValueError("system_memory must be a mapping")
        total = _size_bytes(system_memory.get("total_bytes"), name="total_bytes", allow_zero=False)
        available_value = system_memory.get("available_bytes", total)
        available = _size_bytes(available_value, name="available_bytes")
        reasons: list[str] = []
        if available > total:
            available = total
            reasons.append("available memory exceeded total memory and was clamped")

        kv_cache = kv_cache_bytes(config, context, concurrency)
        confidence: Literal["high", "low"] = "high"
        if kv_cache == 0:
            confidence = "low"
            reasons.append("architecture fields were unavailable or invalid; KV cache is omitted")

        # 1.15 weight multiplier is rounded up using integer arithmetic so the
        # result is deterministic and never underestimates model storage.
        weight_bytes = (model_size * 115 + 99) // 100
        draft_weight_bytes = (draft_size * 115 + 99) // 100
        required = weight_bytes + draft_weight_bytes + kv_cache + RUNTIME_WORKSPACE_BYTES
        reserved = reserve_bytes(total, self.reserve_fraction, self.reserve_min_bytes)
        if required > total:
            decision: Literal["ok", "warning", "blocked"] = "blocked"
            reasons.insert(0, "physical memory requirement exceeds total memory")
        elif required > max(0, available - reserved):
            decision = "warning"
            reasons.insert(0, "current available memory is insufficient after reserved memory")
        else:
            decision = "ok"
        return ResourceEstimate(
            total_bytes=total,
            available_bytes=available,
            reserved_bytes=reserved,
            weight_bytes=weight_bytes,
            draft_weight_bytes=draft_weight_bytes,
            kv_cache_bytes=kv_cache,
            runtime_overhead_bytes=RUNTIME_WORKSPACE_BYTES,
            required_bytes=required,
            decision=decision,
            confidence=confidence,
            reasons=reasons,
        )


def clamp_context_length(
    requested: int,
    hard_limit: int,
    estimate_factory: Callable[[int], ResourceEstimate],
) -> ContextClampResult:
    """Reduce context in 1024-token steps until a physical estimate recovers."""
    original = _bounded_int(
        requested, name="requested context_length", minimum=1024, maximum=MAX_CONTEXT_LENGTH
    )
    limit = _bounded_int(
        hard_limit, name="hard_limit", minimum=1024, maximum=MAX_CONTEXT_LENGTH
    )
    capped_by_hard_limit = original > limit
    current = min(original, limit)
    aligned_context = max(1024, (current // 1024) * 1024)
    aligned_down = aligned_context < current
    current = aligned_context
    attempts: list[int] = []
    blocked_reductions = 0
    while True:
        attempts.append(current)
        estimate = estimate_factory(current)
        if estimate.decision != "blocked" or current == 1024:
            break
        blocked_reductions += 1
        current = max(1024, ((current // 2) // 1024) * 1024)
    if blocked_reductions:
        explanation = (
            f"context length reduced from {original} to {current} after blocked estimates "
            f"({', '.join(map(str, attempts))})"
        )
    elif capped_by_hard_limit:
        explanation = f"context length capped at hard limit {limit}"
    elif aligned_down:
        explanation = f"context length aligned down from {original} to {current}"
    else:
        explanation = "requested context length fits the hard limit and resource estimate"
    return ContextClampResult(
        original_context_length=original,
        final_context_length=current,
        explanation=explanation,
    )
