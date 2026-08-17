from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.models import Provider
from app.security import SecretBox
from app.services.providers import PinnedProviderEndpoint, resolve_provider_endpoint

MAX_PROVIDER_RESPONSE_BYTES = 256 * 1024
MAX_PROVIDER_ERROR_CHARS = 500

ReadOnlyToolName = Literal[
    "host.memory",
    "host.disk",
    "host.gpu",
    "host.ports",
    "host.processes",
    "docker.list",
    "docker.inspect",
    "docker.logs",
    "docker.stats",
    "systemd.status",
    "systemd.journal",
    "manager.summary",
    "manager.tasks",
    "manager.gateway",
]
ChangeOperation = Literal[
    "start_deployment",
    "stop_deployment",
    "restart_deployment",
    "rescan_inventory",
    "shell",
    "explain_only",
]


class ReadOnlyToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: ReadOnlyToolName
    arguments: dict[str, Any] = Field(default_factory=dict, max_length=20)


class ChangeStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: ChangeOperation
    deployment_id: str | None = Field(default=None, min_length=1, max_length=255)
    command: str | None = Field(default=None, min_length=1, max_length=8000)
    cwd: str | None = Field(default=None, min_length=1, max_length=1000)
    timeout: int = Field(default=60, ge=1, le=600)
    reason: str = Field(min_length=1, max_length=2000)
    impact: str = Field(min_length=1, max_length=2000)
    rollback: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_target(self) -> ChangeStep:
        if self.operation == "shell" and self.command is None:
            raise ValueError("shell steps require a command")
        if self.operation != "shell" and self.command is not None:
            raise ValueError("only shell steps may include a command")
        if self.operation in {
            "start_deployment",
            "stop_deployment",
            "restart_deployment",
        } and self.deployment_id is None:
            raise ValueError("deployment steps require a deployment_id")
        return self


class AssistantTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["tool", "plan", "answer", "question"]
    summary: str = Field(min_length=1, max_length=4000)
    tool: ReadOnlyToolRequest | None = None
    steps: list[ChangeStep] = Field(default_factory=list, max_length=20)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must contain text")
        return value

    @model_validator(mode="after")
    def validate_action_payload(self) -> AssistantTurn:
        if self.action == "tool":
            if self.tool is None or self.steps:
                raise ValueError("tool actions require exactly one tool request")
        elif self.action == "plan":
            if self.tool is not None or not self.steps:
                raise ValueError("plan actions require one or more change steps")
        elif self.tool is not None or self.steps:
            raise ValueError("answer and question actions cannot include tools or steps")
        return self


class OpsProviderError(RuntimeError):
    def __init__(self, detail: str):
        self.detail = _sanitize_error(detail)
        super().__init__(self.detail)


class _IncompleteProviderResponse(ValueError):
    pass


class HttpClientFactory(Protocol):
    def __call__(self, **kwargs: Any) -> httpx.Client: ...


EndpointResolver = Callable[[str], PinnedProviderEndpoint]

_SECRET_PATTERNS = (
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[^\s,;]+"),
    re.compile(r"(?i)(?:api[_-]?key|token)\s*[=:]\s*[^\s,;]+"),
    re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


def _sanitize_error(value: object, *, known_secret: str | None = None) -> str:
    text = str(value)
    if known_secret:
        text = text.replace(known_secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = " ".join(text.replace("\x00", " ").split())
    return text[:MAX_PROVIDER_ERROR_CHARS] or "Provider request failed"


class OpsProviderClient:
    def __init__(
        self,
        secret_box: SecretBox,
        *,
        endpoint_resolver: EndpointResolver | None = None,
        http_client_factory: HttpClientFactory | None = None,
    ) -> None:
        self.secret_box = secret_box
        self.endpoint_resolver = endpoint_resolver or resolve_provider_endpoint
        self.http_client_factory = http_client_factory or httpx.Client

    def _headers(self, provider: Provider, endpoint: PinnedProviderEndpoint) -> dict[str, str]:
        forbidden = {
            "accept-encoding",
            "authorization",
            "connection",
            "content-length",
            "host",
            "proxy-authorization",
            "transfer-encoding",
        }
        headers = {
            name: value
            for name, value in provider.headers.items()
            if name.casefold() not in forbidden
        }
        headers.update(
            {
                "Accept-Encoding": "identity",
                "Authorization": f"Bearer {self.secret_box.decrypt(provider.encrypted_api_key)}",
                "Host": endpoint.host_header,
            }
        )
        return headers

    @staticmethod
    def _read_response(response: httpx.Response) -> bytes:
        content_encoding = (
            response.headers.get("Content-Encoding", "").strip().casefold()
        )
        if content_encoding not in {"", "identity"}:
            raise OpsProviderError("Compressed Provider responses are not accepted")
        body = bytearray()
        for chunk in response.iter_bytes():
            if len(body) + len(chunk) > MAX_PROVIDER_RESPONSE_BYTES:
                raise OpsProviderError("Provider response is too large")
            body.extend(chunk)
        return bytes(body)

    def _request_json(
        self,
        provider: Provider,
        method: Literal["GET", "POST"],
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        allow_response_format_fallback: bool = False,
    ) -> Any:
        endpoint = self.endpoint_resolver(f"{provider.base_url.rstrip('/')}{path}")
        headers = self._headers(provider, endpoint)
        extensions = (
            {"sni_hostname": endpoint.sni_hostname}
            if endpoint.sni_hostname is not None
            else {}
        )
        request_payload = payload
        response_format_fallback_used = False
        while True:
            try:
                with self.http_client_factory(
                    timeout=provider.timeout_seconds,
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    request_kwargs: dict[str, Any] = {
                        "headers": headers,
                        "extensions": extensions,
                    }
                    if request_payload is not None:
                        request_kwargs["json"] = request_payload
                    with client.stream(method, endpoint.url, **request_kwargs) as response:
                        body = self._read_response(response)
                        status_code = response.status_code
            except OpsProviderError:
                raise
            except (httpx.HTTPError, OSError, ValueError) as exc:
                raise OpsProviderError(
                    f"Provider request failed: {_sanitize_error(exc)}"
                ) from None

            if 200 <= status_code < 300:
                try:
                    return json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise OpsProviderError("Provider returned invalid JSON") from None

            detail = _sanitize_error(
                body.decode("utf-8", errors="replace"),
                known_secret=self.secret_box.decrypt(provider.encrypted_api_key),
            )
            can_fallback = (
                allow_response_format_fallback
                and not response_format_fallback_used
                and 400 <= status_code < 500
                and "response_format" in detail.casefold()
                and isinstance(request_payload, dict)
                and "response_format" in request_payload
            )
            if can_fallback:
                request_payload = {
                    key: value
                    for key, value in request_payload.items()
                    if key != "response_format"
                }
                response_format_fallback_used = True
                continue
            raise OpsProviderError(f"Provider returned HTTP {status_code}: {detail}")

    @staticmethod
    def _chat_payload(
        provider: Provider,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> dict[str, Any]:
        return {
            "model": provider.default_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

    @staticmethod
    def _parse_turn(payload: Any) -> AssistantTurn:
        if not isinstance(payload, Mapping):
            raise OpsProviderError("Provider response shape is invalid")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise OpsProviderError("Provider response shape is invalid")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise OpsProviderError("Provider response shape is invalid")
        if choice.get("finish_reason") == "length":
            raise _IncompleteProviderResponse("output limit reached")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise OpsProviderError("Provider response shape is invalid")
        content = message.get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            raise _IncompleteProviderResponse("missing final content")
        if not isinstance(content, str):
            raise OpsProviderError("Provider response shape is invalid")
        try:
            decoded = json.loads(content)
            return AssistantTurn.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise _IncompleteProviderResponse("invalid structured response") from exc

    def complete(
        self, provider: Provider, messages: list[dict[str, Any]]
    ) -> AssistantTurn:
        initial_payload = self._chat_payload(provider, messages, max_tokens=2048)
        response = self._request_json(
            provider,
            "POST",
            "/chat/completions",
            payload=initial_payload,
            allow_response_format_fallback=True,
        )
        try:
            return self._parse_turn(response)
        except _IncompleteProviderResponse as initial_error:
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Return one complete JSON object matching the requested assistant schema. "
                        "Do not include reasoning, markdown, or commentary."
                    ),
                },
                *messages,
                {
                    "role": "user",
                    "content": f"Repair the prior response: {initial_error}",
                },
            ]
            repair_payload = self._chat_payload(
                provider, repair_messages, max_tokens=4096
            )
            repaired = self._request_json(
                provider,
                "POST",
                "/chat/completions",
                payload=repair_payload,
                allow_response_format_fallback=True,
            )
            try:
                return self._parse_turn(repaired)
            except _IncompleteProviderResponse as exc:
                raise OpsProviderError(
                    f"Provider returned an invalid structured response after repair: {exc}"
                ) from None

    def list_models(self, provider: Provider) -> list[str]:
        payload = self._request_json(provider, "GET", "/models")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            raise OpsProviderError("Provider returned an invalid models response")
        return [
            item["id"]
            for item in payload["data"][:10_000]
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        ]
