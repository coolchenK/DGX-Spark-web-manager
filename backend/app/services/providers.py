from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from sqlalchemy.orm import Session

from app.models import Provider
from app.security import SecretBox, mask_secret

Resolver = Callable[..., list[tuple]]
HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
FORBIDDEN_CUSTOM_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "proxy-authorization",
    "transfer-encoding",
}


def _is_forbidden_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    )


def validate_provider_url(value: str, resolver: Resolver = socket.getaddrinfo) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Provider URL must use HTTP or HTTPS")
    if parsed.username or parsed.password:
        raise ValueError("Provider URL cannot include credentials")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Provider URL cannot target localhost")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            addresses = {item[4][0] for item in resolver(hostname, parsed.port or 443)}
        except socket.gaierror as exc:
            raise ValueError("Provider hostname could not be resolved") from exc
        if not addresses or any(_is_forbidden_ip(address) for address in addresses):
            raise ValueError("Provider hostname resolves to a restricted network") from None
    else:
        if _is_forbidden_ip(str(literal)):
            raise ValueError("Provider URL targets a restricted network")
    return value


def normalize_openai_base_url(value: str) -> str:
    parsed = urlparse(value.strip())
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def validate_custom_headers(headers: dict[str, str]) -> dict[str, str]:
    if len(headers) > 50:
        raise ValueError("At most 50 custom headers are allowed")
    for name, value in headers.items():
        if not HEADER_NAME.fullmatch(name) or name.lower() in FORBIDDEN_CUSTOM_HEADERS:
            raise ValueError(f"Unsafe custom header name: {name}")
        if len(value) > 8192 or any(character in value for character in "\r\n\0"):
            raise ValueError(f"Unsafe custom header value for: {name}")
    return headers


class ProviderService:
    def __init__(self, secret_box: SecretBox):
        self.secret_box = secret_box

    def create(
        self,
        db: Session,
        *,
        name: str,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout_seconds: int,
        headers: dict[str, str],
        enabled: bool,
    ) -> Provider:
        validate_provider_url(base_url)
        validate_custom_headers(headers)
        provider = Provider(
            name=name.strip(),
            base_url=normalize_openai_base_url(base_url),
            encrypted_api_key=self.secret_box.encrypt(api_key),
            default_model=default_model.strip(),
            timeout_seconds=timeout_seconds,
            headers=headers,
            enabled=enabled,
        )
        db.add(provider)
        db.commit()
        db.refresh(provider)
        return provider

    def update_secret(self, provider: Provider, api_key: str) -> None:
        provider.encrypted_api_key = self.secret_box.encrypt(api_key)

    def serialize(self, provider: Provider) -> dict[str, Any]:
        return {
            "id": provider.id,
            "name": provider.name,
            "base_url": provider.base_url,
            "default_model": provider.default_model,
            "api_key_masked": mask_secret(self.secret_box.decrypt(provider.encrypted_api_key)),
            "timeout_seconds": provider.timeout_seconds,
            "headers": provider.headers,
            "enabled": provider.enabled,
            "last_test_status": provider.last_test_status,
            "last_tested_at": provider.last_tested_at,
            "created_at": provider.created_at,
            "updated_at": provider.updated_at,
        }

    def authorization_headers(self, provider: Provider) -> dict[str, str]:
        return {
            **provider.headers,
            "Authorization": f"Bearer {self.secret_box.decrypt(provider.encrypted_api_key)}",
        }

    def test(self, db: Session, provider: Provider) -> dict[str, Any]:
        started = datetime.now(UTC)
        try:
            response = httpx.get(
                f"{provider.base_url}/models",
                headers=self.authorization_headers(provider),
                timeout=min(provider.timeout_seconds, 30),
                follow_redirects=False,
                trust_env=False,
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data", []) if isinstance(payload, dict) else []
            provider.last_test_status = "healthy"
            result = {
                "status": "healthy",
                "latency_ms": (datetime.now(UTC) - started).total_seconds() * 1000,
                "models": [item.get("id") for item in models[:20] if isinstance(item, dict)],
            }
        except (httpx.HTTPError, ValueError) as exc:
            provider.last_test_status = "failed"
            result = {"status": "failed", "error": str(exc)[:1000]}
        provider.last_tested_at = datetime.now(UTC)
        db.commit()
        return result
