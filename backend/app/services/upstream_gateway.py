"""Upstream gateway configuration, resolution, and model discovery.

The manager forwards requests for models it does not host to an
OpenAI-compatible upstream. Operators paste the same style of base URL the
README documents for this manager (`http://<host>:3000/v1`), so the configured
value may or may not carry the trailing `/v1`. Both forms must resolve to
exactly one `/v1` prefix on the wire.

Configuration is stored encrypted in `SecretSetting` so the panel can change it
without rebuilding the container. The `DGX_FALLBACK_*` environment variables
remain supported as the default seed for existing installations.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import SecretSetting, utc_now
from app.security import SecretBox

BASE_URL_KEY = "upstream_base_url"
API_KEY_KEY = "upstream_api_key"
EXPOSURE_KEY = "upstream_model_exposure"
MAX_BASE_URL_LENGTH = 500
MAX_SELECTED_MODELS = 500
MAX_MODEL_ID_LENGTH = 255
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
MODELS_TIMEOUT_SECONDS = 10.0
MAX_PROBE_DETAIL_CHARS = 200
V1_SUFFIX = "/v1"


@dataclass(frozen=True)
class UpstreamExposure:
    """Which upstream models this gateway is allowed to advertise and serve."""

    expose_all: bool = True
    selected_models: frozenset[str] = field(default_factory=frozenset)

    def allows(self, model_id: str) -> bool:
        return self.expose_all or model_id in self.selected_models


@dataclass(frozen=True)
class UpstreamGatewayConfig:
    base_url: str
    api_key: str | None
    source: Literal["database", "environment"]
    exposure: UpstreamExposure = field(default_factory=UpstreamExposure)

    @property
    def cache_key(self) -> str:
        """Separate cache entries per upstream without storing the key itself."""
        return f"{self.base_url}|{len(self.api_key or '')}"

    def allows_model(self, model_id: str) -> bool:
        return self.exposure.allows(model_id)


def upstream_api_root(base_url: str) -> str:
    """Return the base URL without a trailing `/v1`.

    Endpoints supplied by the gateway already start with `/v1`, so the root is
    what they are concatenated onto.
    """
    root = (base_url or "").strip().rstrip("/")
    if root.endswith(V1_SUFFIX):
        root = root[: -len(V1_SUFFIX)].rstrip("/")
    return root


def upstream_request_url(base_url: str, endpoint: str) -> str:
    """Compose an upstream URL with exactly one `/v1` prefix."""
    root = upstream_api_root(base_url)
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{root}{endpoint}"


def validate_upstream_base_url(value: str) -> str:
    """Validate and normalise an operator-supplied upstream base URL."""
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        raise ValueError("上游网关地址不能为空")
    if len(candidate) > MAX_BASE_URL_LENGTH:
        raise ValueError(f"上游网关地址不能超过 {MAX_BASE_URL_LENGTH} 个字符")
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"}:
        raise ValueError("上游网关地址必须使用 http 或 https")
    if not parts.hostname:
        raise ValueError("上游网关地址必须包含主机名")
    if parts.username or parts.password:
        raise ValueError("上游网关地址不能内嵌用户名或密码")
    if parts.fragment:
        raise ValueError("上游网关地址不能包含片段标识")
    return candidate


def validate_selected_models(values: Any) -> tuple[str, ...]:
    """Validate the explicitly selected upstream model ids."""
    if values is None:
        return ()
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError("selected_models 必须是模型名列表")
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError("selected_models 只能包含字符串")
        name = item.strip()
        if not name or len(name) > MAX_MODEL_ID_LENGTH or not MODEL_ID_PATTERN.fullmatch(name):
            raise ValueError(f"非法的模型名：{item!r}")
        if name not in normalized:
            normalized.append(name)
    if len(normalized) > MAX_SELECTED_MODELS:
        raise ValueError(f"最多只能选择 {MAX_SELECTED_MODELS} 个上游模型")
    return tuple(sorted(normalized))


def read_upstream_exposure(db: Session, secret_box: SecretBox) -> UpstreamExposure:
    """Read the stored exposure selection, defaulting to offering every model."""
    stored = db.get(SecretSetting, EXPOSURE_KEY)
    if stored is None:
        return UpstreamExposure()
    try:
        payload = json.loads(secret_box.decrypt(stored.encrypted_value))
    except (ValueError, json.JSONDecodeError):
        return UpstreamExposure()
    if not isinstance(payload, dict):
        return UpstreamExposure()
    expose_all = payload.get("expose_all")
    try:
        selected = validate_selected_models(payload.get("selected_models"))
    except ValueError:
        selected = ()
    return UpstreamExposure(
        expose_all=bool(expose_all) if isinstance(expose_all, bool) else True,
        selected_models=frozenset(selected),
    )


def write_upstream_exposure(
    db: Session,
    secret_box: SecretBox,
    *,
    expose_all: bool,
    selected_models: Any,
) -> UpstreamExposure:
    """Persist the exposure selection and return the normalized result."""
    exposure = UpstreamExposure(
        expose_all=bool(expose_all),
        selected_models=frozenset(validate_selected_models(selected_models)),
    )
    payload = json.dumps(
        {
            "expose_all": exposure.expose_all,
            "selected_models": sorted(exposure.selected_models),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    _write_secret(db, secret_box, EXPOSURE_KEY, payload)
    db.flush()
    return exposure


def _read_secret(box: SecretBox, row: SecretSetting | None) -> str | None:
    if row is None:
        return None
    value = box.decrypt(row.encrypted_value)
    return value or None


def resolve_upstream_gateway(
    db: Session,
    secret_box: SecretBox,
    settings: Settings,
) -> UpstreamGatewayConfig | None:
    """Resolve the effective upstream configuration.

    A value stored from the panel wins; environment variables are the default
    seed so installations configured through `.env` keep working.
    """
    exposure = read_upstream_exposure(db, secret_box)
    stored_url = db.get(SecretSetting, BASE_URL_KEY)
    if stored_url is not None:
        return UpstreamGatewayConfig(
            base_url=secret_box.decrypt(stored_url.encrypted_value).rstrip("/"),
            api_key=_read_secret(secret_box, db.get(SecretSetting, API_KEY_KEY)),
            source="database",
            exposure=exposure,
        )
    env_url = (settings.fallback_base_url or "").strip()
    if not env_url:
        return None
    return UpstreamGatewayConfig(
        base_url=env_url.rstrip("/"),
        api_key=(settings.fallback_api_key or "").strip() or None,
        source="environment",
        exposure=exposure,
    )


def _write_secret(db: Session, secret_box: SecretBox, key: str, value: str) -> None:
    stored = db.get(SecretSetting, key)
    encrypted = secret_box.encrypt(value)
    if stored is None:
        db.add(SecretSetting(key=key, encrypted_value=encrypted))
    else:
        stored.encrypted_value = encrypted
        stored.updated_at = utc_now()


def _delete_secret(db: Session, key: str) -> None:
    stored = db.get(SecretSetting, key)
    if stored is not None:
        db.delete(stored)


def set_upstream_base_url(
    db: Session,
    secret_box: SecretBox,
    base_url: str,
    *,
    api_key: str | None = None,
) -> None:
    """Store the upstream base URL.

    `api_key=None` leaves any stored key untouched; a non-empty value replaces
    it. Use `clear_upstream_api_key` to remove a key.
    """
    _write_secret(db, secret_box, BASE_URL_KEY, validate_upstream_base_url(base_url))
    if api_key is not None:
        _write_secret(db, secret_box, API_KEY_KEY, api_key)
    db.flush()


def clear_upstream_gateway(db: Session) -> None:
    """Remove the stored upstream configuration, reverting to the environment."""
    _delete_secret(db, BASE_URL_KEY)
    _delete_secret(db, API_KEY_KEY)
    db.flush()


def set_upstream_api_key(db: Session, secret_box: SecretBox, api_key: str) -> None:
    _write_secret(db, secret_box, API_KEY_KEY, api_key)
    db.flush()


def clear_upstream_api_key(db: Session) -> None:
    _delete_secret(db, API_KEY_KEY)
    db.flush()


def fetch_upstream_models(config: UpstreamGatewayConfig) -> list[dict[str, Any]]:
    """Fetch the upstream model list, raising httpx or ValueError on failure."""
    headers = {"authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    with httpx.Client(timeout=MODELS_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.get(upstream_request_url(config.base_url, "/v1/models"), headers=headers)
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("上游 /v1/models 响应缺少 data 列表")
    return [
        item
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    ]


def bounded_probe_detail(exc: Exception) -> str:
    """Return a bounded, credential-free description of a probe failure."""
    name = type(exc).__name__
    message = " ".join(str(exc).split())
    if len(message) > MAX_PROBE_DETAIL_CHARS:
        message = message[: MAX_PROBE_DETAIL_CHARS - 3] + "..."
    return f"{name}: {message}" if message else name


class UpstreamModelCache:
    """Bounded TTL cache holding only the most recent upstream model list."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._lock = Lock()

    def get(self, key: str) -> list[dict[str, Any]] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, models = entry
            if time.monotonic() - stored_at > self._ttl_seconds:
                self._entries.pop(key, None)
                return None
            return models

    def set(self, key: str, models: list[dict[str, Any]]) -> None:
        with self._lock:
            self._entries = {key: (time.monotonic(), models)}

    def invalidate(self) -> None:
        with self._lock:
            self._entries.clear()


def upstream_model_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Map an upstream model entry onto the gateway contract.

    Only the model id is actually known, so every capability and limit field is
    reported as unknown instead of being guessed.
    """
    created = raw.get("created")
    return {
        "id": raw["id"],
        "object": "model",
        "created": created if isinstance(created, int) and not isinstance(created, bool) else 0,
        "owned_by": "upstream",
        "dgx_source": "upstream",
        "capabilities": [],
        "input_modalities": [],
        "context_window": None,
        "configured_context_length": None,
        "runtime_token_capacity": None,
        "max_model_len": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "max_concurrency": None,
        "runtime": None,
        "performance": {"status": "unknown", "tokens_per_second": None},
    }
