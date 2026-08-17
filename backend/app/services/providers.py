from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse, urlunparse

from sqlalchemy.orm import Session

from app.models import Provider
from app.security import SecretBox, mask_secret

if TYPE_CHECKING:
    from app.services.ops_provider import AssistantTurn

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
MAX_SERIALIZED_TEST_STRING_CHARS = 500
MAX_SERIALIZED_TEST_COLLECTION_ITEMS = 50
MAX_SERIALIZED_TEST_DEPTH = 6


class OpsProviderProbe(Protocol):
    def list_models(self, provider: Provider) -> list[str]: ...

    def complete(
        self, provider: Provider, messages: list[dict[str, Any]]
    ) -> AssistantTurn: ...


def _sanitize_test_result_string(value: str, known_secret: str) -> str:
    redacted = value.replace(known_secret, "[REDACTED]") if known_secret else value
    return " ".join(redacted.replace("\x00", " ").split())[
        :MAX_SERIALIZED_TEST_STRING_CHARS
    ]


def _sanitize_test_result(value: Any, known_secret: str, *, depth: int = 0) -> Any:
    if depth >= MAX_SERIALIZED_TEST_DEPTH:
        return "[TRUNCATED]"
    if isinstance(value, str):
        return _sanitize_test_result_string(value, known_secret)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= MAX_SERIALIZED_TEST_COLLECTION_ITEMS:
                break
            key = _sanitize_test_result_string(str(raw_key), known_secret)
            if key in result:
                continue
            result[key] = _sanitize_test_result(
                raw_value, known_secret, depth=depth + 1
            )
        return result
    if isinstance(value, list | tuple):
        return [
            _sanitize_test_result(item, known_secret, depth=depth + 1)
            for item in value[:MAX_SERIALIZED_TEST_COLLECTION_ITEMS]
        ]
    return "[UNSUPPORTED]"


@dataclass(frozen=True, slots=True)
class PinnedProviderEndpoint:
    url: str
    host_header: str
    sni_hostname: str | None


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


def resolve_provider_endpoint(
    value: str, resolver: Resolver = socket.getaddrinfo
) -> PinnedProviderEndpoint:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Provider URL must use HTTP or HTTPS")
    if parsed.username or parsed.password:
        raise ValueError("Provider URL cannot include credentials")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Provider URL targets a restricted network")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    literal: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            records = resolver(hostname, port, 0, socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError("Provider hostname could not be resolved") from exc
        addresses = {str(record[4][0]) for record in records if len(record) > 4 and record[4]}
        if not addresses:
            raise ValueError("Provider hostname could not be resolved") from None
        parsed_addresses = [ipaddress.ip_address(address) for address in addresses]
    else:
        parsed_addresses = [literal]

    if any(_is_forbidden_ip(str(address)) or not address.is_global for address in parsed_addresses):
        raise ValueError("Provider hostname resolves to a restricted network")
    selected = sorted(parsed_addresses, key=lambda address: (address.version, int(address)))[0]
    pinned_host = f"[{selected}]" if selected.version == 6 else str(selected)
    pinned_netloc = f"{pinned_host}:{parsed.port}" if parsed.port is not None else pinned_host
    original_host = f"[{hostname}]" if literal is not None and literal.version == 6 else hostname
    host_header = f"{original_host}:{parsed.port}" if parsed.port is not None else original_host
    return PinnedProviderEndpoint(
        url=urlunparse(
            (
                parsed.scheme,
                pinned_netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                "",
            )
        ),
        host_header=host_header,
        sni_hostname=hostname if parsed.scheme == "https" else None,
    )


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
    def __init__(self, secret_box: SecretBox, ops_provider_client: OpsProviderProbe):
        self.secret_box = secret_box
        self.ops_provider_client = ops_provider_client

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
        encrypted_api_key = self.secret_box.encrypt(api_key)
        provider.encrypted_api_key = encrypted_api_key
        provider.last_test_result = {}
        provider.last_test_status = None
        provider.last_tested_at = None

    def serialize(self, provider: Provider) -> dict[str, Any]:
        api_key = self.secret_box.decrypt(provider.encrypted_api_key)
        return {
            "id": provider.id,
            "name": provider.name,
            "base_url": provider.base_url,
            "default_model": provider.default_model,
            "api_key_masked": mask_secret(api_key),
            "timeout_seconds": provider.timeout_seconds,
            "headers": provider.headers,
            "enabled": provider.enabled,
            "last_test_status": provider.last_test_status,
            "last_test_result": _sanitize_test_result(provider.last_test_result, api_key),
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
        try:
            models = self.ops_provider_client.list_models(provider)
        except Exception as exc:
            result = {
                "status": "failed",
                "connection": {
                    "status": "failed",
                    "error": self._probe_error(provider, exc),
                },
                "default_model": {
                    "status": "not_tested",
                    "model": provider.default_model,
                },
            }
        else:
            connection = {"status": "healthy", "models_seen": len(models)}
            try:
                self.ops_provider_client.complete(
                    provider,
                    [
                        {
                            "role": "system",
                            "content": (
                                "This is a connectivity probe. Return exactly one JSON object "
                                'with action "answer" and a short non-empty summary.'
                            ),
                        },
                        {"role": "user", "content": "Confirm this model can respond."},
                    ],
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "connection": connection,
                    "default_model": {
                        "status": "failed",
                        "model": provider.default_model,
                        "error": self._probe_error(provider, exc),
                    },
                }
            else:
                result = {
                    "status": "healthy",
                    "connection": connection,
                    "default_model": {
                        "status": "healthy",
                        "model": provider.default_model,
                    },
                }
        provider.last_test_status = result["status"]
        provider.last_test_result = result
        provider.last_tested_at = datetime.now(UTC)
        db.commit()
        return result

    def _probe_error(self, provider: Provider, exc: Exception) -> str:
        detail = getattr(exc, "detail", None)
        if not isinstance(detail, str):
            return "Provider probe failed"
        try:
            known_secret = self.secret_box.decrypt(provider.encrypted_api_key)
        except ValueError:
            known_secret = None
        if known_secret:
            detail = detail.replace(known_secret, "[REDACTED]")
        return " ".join(detail.replace("\x00", " ").split())[:500] or "Provider probe failed"
