from __future__ import annotations

import re

MAX_PROVIDER_ERROR_CHARS = 500

_SECRET_PATTERNS = (
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[^\s,;]+"),
    re.compile(r"(?i)(?:api[_-]?key|token)\s*[=:]\s*[^\s,;]+"),
    re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


def sanitize_provider_error(value: object, *, known_secret: str | None = None) -> str:
    text = str(value)
    if known_secret:
        text = text.replace(known_secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = " ".join(text.replace("\x00", " ").split())
    return text[:MAX_PROVIDER_ERROR_CHARS] or "Provider request failed"


class OpsProviderError(RuntimeError):
    def __init__(self, detail: str):
        self.detail = sanitize_provider_error(detail)
        super().__init__(self.detail)


class ProviderConfigurationChanged(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Provider configuration changed during test")
