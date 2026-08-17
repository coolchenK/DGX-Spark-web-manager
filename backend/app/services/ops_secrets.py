from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Provider, SecretSetting
from app.security import SecretBox

MIN_EMBEDDED_SECRET_CHARS = 3
MAX_STRUCTURE_DEPTH = 8
MAX_COLLECTION_ITEMS = 100
MAX_STRING_CHARS = 12_000
REDACTED = "[REDACTED]"
TRUNCATED = "[truncated]"
_SENSITIVE_KEY_NAMES = frozenset(
    {
        "auth",
        "authorization",
        "api_key",
        "apikey",
        "encrypted_api_key",
        "encrypted_value",
        "token",
        "access_token",
        "api_token",
        "auth_token",
        "bearer_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "secret_key",
    }
)
_SENSITIVE_KEY_SUFFIXES = (
    "_auth",
    "_authorization",
    "_api_key",
    "_apikey",
    "_token",
    "_password",
    "_passwd",
    "_secret",
    "_secret_key",
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<quote>[\"']?)"
    r"(?P<label>[A-Za-z0-9][A-Za-z0-9_-]{0,127})(?P=quote)\s*[:=]\s*"
    r"(?:(?:bearer|basic)\s+)?"
    r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|(?:\[REDACTED\]|[^\s,;&}\[\]]+)+)""",
    re.IGNORECASE,
)
_ACRONYM_BOUNDARY_PATTERN = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY_PATTERN = re.compile(r"([a-z0-9])([A-Z])")

SessionFactory = Callable[[], Session]


class SecretLoadError(RuntimeError):
    pass


def is_sensitive_key(value: str) -> bool:
    normalized = re.sub(r"[-\s]+", "_", value).strip("_")
    normalized = _ACRONYM_BOUNDARY_PATTERN.sub(r"\1_\2", normalized)
    normalized = _CAMEL_BOUNDARY_PATTERN.sub(r"\1_\2", normalized).casefold()
    return normalized in _SENSITIVE_KEY_NAMES or normalized.endswith(
        _SENSITIVE_KEY_SUFFIXES
    )


def _redact_credential_assignment(match: re.Match[str]) -> str:
    if is_sensitive_key(match.group("label")):
        return REDACTED
    return match.group(0)


@lru_cache(maxsize=128)
def _known_secret_pattern(secrets: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted({secret for secret in secrets if secret}, key=lambda item: (-len(item), item))
    marker_secrets = [
        re.escape(secret)
        for secret in ordered
        if len(secret) >= 8 and REDACTED in secret
    ]
    embedded = [
        re.escape(secret)
        for secret in ordered
        if len(secret) >= 8 and REDACTED not in secret
    ]
    bounded = [
        re.escape(secret)
        for secret in ordered
        if MIN_EMBEDDED_SECRET_CHARS <= len(secret) < 8
    ]
    alternatives: list[str] = []
    if marker_secrets:
        alternatives.append(f"(?P<marker_secret>{'|'.join(marker_secrets)})")
    alternatives.append(f"(?P<existing_marker>{re.escape(REDACTED)})")
    if embedded:
        alternatives.append(f"(?P<embedded_secret>{'|'.join(embedded)})")
    if bounded:
        alternatives.append(
            rf"(?P<bounded_secret>(?<![A-Za-z0-9_])(?:{'|'.join(bounded)})(?![A-Za-z0-9_]))"
        )
    return re.compile("|".join(alternatives))


def _replace_known_secret_match(match: re.Match[str]) -> str:
    return match.group(0) if match.lastgroup == "existing_marker" else REDACTED


def replace_known_secrets(value: str, secrets: tuple[str, ...]) -> str:
    if value in secrets:
        return REDACTED
    return _known_secret_pattern(secrets).sub(_replace_known_secret_match, value)


def sanitize_string(value: str, secrets: tuple[str, ...], limit: int) -> str:
    sanitized = redact_string(value, secrets)
    if len(sanitized) <= limit:
        return sanitized
    marker = f"\n{TRUNCATED}\n"
    available = max(0, limit - len(marker))
    head = (available + 1) // 2
    tail = available - head
    return f"{sanitized[:head]}{marker}{sanitized[-tail:] if tail else ''}"


def redact_string(value: str, secrets: tuple[str, ...]) -> str:
    sanitized = replace_known_secrets(value, secrets)
    return _CREDENTIAL_ASSIGNMENT_PATTERN.sub(_redact_credential_assignment, sanitized)


def sanitize_value(
    value: Any,
    secrets: tuple[str, ...],
    trusted_keys: frozenset[str],
    depth: int = 0,
) -> Any:
    if depth >= MAX_STRUCTURE_DEPTH:
        return TRUNCATED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return sanitize_string(value, secrets, MAX_STRING_CHARS)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_COLLECTION_ITEMS:
                result["truncated"] = True
                break
            raw_key = str(key)
            if raw_key in trusted_keys:
                key_text = raw_key
            elif is_sensitive_key(raw_key):
                result[f"redacted_field_{index}"] = REDACTED
                continue
            else:
                key_text = sanitize_string(raw_key, secrets, 256)
                if key_text != raw_key:
                    key_text = f"redacted_key_{index}"
            result[key_text] = sanitize_value(item, secrets, trusted_keys, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        result = [
            sanitize_value(item, secrets, trusted_keys, depth + 1)
            for item in value[:MAX_COLLECTION_ITEMS]
        ]
        if len(value) > MAX_COLLECTION_ITEMS:
            result.append(TRUNCATED)
        return result
    return f"[unsupported:{type(value).__name__}]"


class KnownSecrets:
    def __init__(self, values: tuple[str, ...]) -> None:
        self.values = tuple(
            sorted({value for value in values if value}, key=lambda item: (-len(item), item))
        )

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return redact_string(value, self.values)
        return sanitize_value(value, self.values, frozenset())

    def contains(self, value: Any) -> bool:
        return self.redact(value) != value


def load_known_secrets(
    session_factory: SessionFactory,
    secret_box: SecretBox | None,
) -> KnownSecrets:
    values: set[str] = set()

    def add_plaintext(value: str) -> None:
        if not value:
            raise SecretLoadError("configured secrets could not be loaded safely")
        if len(value) < MIN_EMBEDDED_SECRET_CHARS:
            raise SecretLoadError("configured secrets are too short to sanitize safely")
        values.add(value)

    def add_encrypted(value: Any) -> None:
        if not isinstance(value, str) or not value or secret_box is None:
            raise SecretLoadError("configured secrets could not be loaded safely")
        values.add(value)
        try:
            plaintext = secret_box.decrypt(value)
        except (TypeError, ValueError):
            raise SecretLoadError("configured secrets could not be loaded safely") from None
        add_plaintext(plaintext)

    try:
        with session_factory() as db:
            providers = db.execute(
                select(Provider.encrypted_api_key, Provider.headers)
            ).all()
            settings = db.scalars(select(SecretSetting.encrypted_value)).all()
            for encrypted_api_key, headers in providers:
                add_encrypted(encrypted_api_key)
                if isinstance(headers, Mapping):
                    for header_name, header_value in headers.items():
                        if not (
                            isinstance(header_name, str)
                            and isinstance(header_value, str)
                            and header_value
                        ):
                            continue
                        scheme, separator, credential = header_value.partition(" ")
                        has_auth_scheme = bool(
                            separator
                            and scheme.casefold() in {"bearer", "basic"}
                            and credential
                        )
                        if not is_sensitive_key(header_name) and not has_auth_scheme:
                            continue
                        add_plaintext(header_value)
                        if has_auth_scheme:
                            add_plaintext(credential)
            for encrypted_value in settings:
                add_encrypted(encrypted_value)
    except SecretLoadError:
        raise
    except Exception:
        raise SecretLoadError("configured secrets could not be loaded safely") from None
    return KnownSecrets(tuple(values))


__all__ = [
    "KnownSecrets",
    "MIN_EMBEDDED_SECRET_CHARS",
    "REDACTED",
    "SecretLoadError",
    "TRUNCATED",
    "is_sensitive_key",
    "load_known_secrets",
    "replace_known_secrets",
    "redact_string",
    "sanitize_string",
    "sanitize_value",
]
