from __future__ import annotations

import json
import re
import time
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
from app.services.provider_errors import (
    OpsProviderError,
)
from app.services.provider_errors import (
    sanitize_provider_error as _sanitize_error,
)
from app.services.providers import PinnedProviderEndpoint, resolve_provider_endpoint

MAX_PROVIDER_RESPONSE_BYTES = 256 * 1024
MAX_REPAIR_MESSAGES = 8
MAX_REPAIR_MESSAGE_CHARS = 2400
MAX_REPAIR_TOTAL_CHARS = 16_000

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

_CONTAINER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SERVICE_PATTERN = re.compile(r"[A-Za-z0-9_.@-]+\Z")


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmptyToolArguments(_ToolArguments):
    pass


class ContainerToolArguments(_ToolArguments):
    container: str = Field(min_length=1, max_length=128)

    @field_validator("container")
    @classmethod
    def validate_container(cls, value: str) -> str:
        if _CONTAINER_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid container")
        return value


class ContainerTailToolArguments(ContainerToolArguments):
    tail: int = Field(ge=1, le=5000, strict=True)


class ServiceToolArguments(_ToolArguments):
    service: str = Field(min_length=1, max_length=256)

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        if value.startswith("-") or value.isdecimal() or _SERVICE_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid service")
        return value


class ServiceTailToolArguments(ServiceToolArguments):
    tail: int = Field(ge=1, le=5000, strict=True)


class ManagerTasksToolArguments(_ToolArguments):
    limit: int = Field(default=20, ge=1, le=50, strict=True)


class ManagerGatewayToolArguments(_ToolArguments):
    minutes: int = Field(default=60, ge=1, le=1440, strict=True)
    limit: int = Field(default=20, ge=1, le=50, strict=True)


ReadOnlyToolArguments = (
    EmptyToolArguments
    | ContainerToolArguments
    | ContainerTailToolArguments
    | ServiceToolArguments
    | ServiceTailToolArguments
    | ManagerTasksToolArguments
    | ManagerGatewayToolArguments
)

_TOOL_ARGUMENT_MODELS: dict[str, type[_ToolArguments]] = {
    "host.memory": EmptyToolArguments,
    "host.disk": EmptyToolArguments,
    "host.gpu": EmptyToolArguments,
    "host.ports": EmptyToolArguments,
    "host.processes": EmptyToolArguments,
    "docker.list": EmptyToolArguments,
    "docker.inspect": ContainerToolArguments,
    "docker.logs": ContainerTailToolArguments,
    "docker.stats": ContainerToolArguments,
    "systemd.status": ServiceToolArguments,
    "systemd.journal": ServiceTailToolArguments,
    "manager.summary": EmptyToolArguments,
    "manager.tasks": ManagerTasksToolArguments,
    "manager.gateway": ManagerGatewayToolArguments,
}

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
    arguments: ReadOnlyToolArguments

    @model_validator(mode="before")
    @classmethod
    def validate_arguments_for_name(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        name = value.get("name")
        arguments_model = _TOOL_ARGUMENT_MODELS.get(name)
        if arguments_model is None:
            return value
        validated = arguments_model.model_validate(value.get("arguments", {}))
        return {**value, "arguments": validated}

    @model_validator(mode="after")
    def arguments_match_name(self) -> ReadOnlyToolRequest:
        expected = _TOOL_ARGUMENT_MODELS[self.name]
        if type(self.arguments) is not expected:
            raise ValueError("arguments do not match tool name")
        return self

    def argument_dict(self) -> dict[str, Any]:
        return self.arguments.model_dump(mode="python")


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


class _IncompleteProviderResponse(ValueError):
    pass


class HttpClientFactory(Protocol):
    def __call__(self, **kwargs: Any) -> httpx.Client: ...


EndpointResolver = Callable[[str], PinnedProviderEndpoint]

def _bounded_repair_content(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n...[truncated]...\n"
    available = max(0, limit - len(marker))
    head = (available + 1) // 2
    tail = available - head
    return f"{value[:head]}{marker}{value[-tail:] if tail else ''}"


def _compact_repair_messages(
    messages: list[dict[str, Any]], reason: str
) -> list[dict[str, str]]:
    usable = [
        (index, message["role"], message["content"])
        for index, message in enumerate(messages)
        if message.get("role") in {"system", "user", "assistant", "tool"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    selected_indexes: set[int] = set()
    first_system = next((item for item in usable if item[1] == "system"), None)
    if first_system is not None:
        selected_indexes.add(first_system[0])
    latest_user = next((item for item in reversed(usable) if item[1] == "user"), None)
    if latest_user is not None:
        selected_indexes.add(latest_user[0])
    recent_support = [
        item for item in reversed(usable) if item[1] in {"assistant", "tool"}
    ][:3]
    selected_indexes.update(item[0] for item in recent_support)

    repair_system = {
        "role": "system",
        "content": (
            "Return one complete JSON object matching the requested assistant schema. "
            "Do not include reasoning, markdown, or commentary."
        ),
    }
    retry_instruction = {
        "role": "user",
        "content": _bounded_repair_content(
            f"Repair the prior response: {reason}", MAX_REPAIR_MESSAGE_CHARS
        ),
    }
    result = [repair_system]
    remaining_chars = MAX_REPAIR_TOTAL_CHARS - sum(
        len(item["content"]) for item in (repair_system, retry_instruction)
    )
    context_slots = MAX_REPAIR_MESSAGES - 2
    for index, role, content in usable:
        if index not in selected_indexes or context_slots <= 0 or remaining_chars <= 0:
            continue
        bounded = _bounded_repair_content(
            content, min(MAX_REPAIR_MESSAGE_CHARS, remaining_chars)
        )
        if not bounded:
            continue
        result.append({"role": role, "content": bounded})
        remaining_chars -= len(bounded)
        context_slots -= 1
    result.append(retry_instruction)
    return result


class OpsProviderClient:
    def __init__(
        self,
        secret_box: SecretBox,
        *,
        endpoint_resolver: EndpointResolver | None = None,
        http_client_factory: HttpClientFactory | None = None,
        _monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.secret_box = secret_box
        self.endpoint_resolver = endpoint_resolver or resolve_provider_endpoint
        self.http_client_factory = http_client_factory or httpx.Client
        self._monotonic = _monotonic

    @staticmethod
    def _headers(
        provider: Provider, endpoint: PinnedProviderEndpoint, api_key: str
    ) -> dict[str, str]:
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
                "Authorization": f"Bearer {api_key}",
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
        deadline: float | None = None,
    ) -> Any:
        try:
            api_key = self.secret_box.decrypt(provider.encrypted_api_key)
        except ValueError:
            raise OpsProviderError("Provider credentials are unavailable") from None
        try:
            endpoint = self.endpoint_resolver(f"{provider.base_url.rstrip('/')}{path}")
            headers = self._headers(provider, endpoint, api_key)
            extensions = (
                {"sni_hostname": endpoint.sni_hostname}
                if endpoint.sni_hostname is not None
                else {}
            )
            request_payload = payload
            response_format_fallback_used = False
            while True:
                timeout = provider.timeout_seconds
                if deadline is not None:
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        raise OpsProviderError("Provider request deadline exceeded")
                    timeout = min(float(timeout), remaining)
                with self.http_client_factory(
                    timeout=timeout,
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

                if 200 <= status_code < 300:
                    try:
                        return json.loads(body)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        raise OpsProviderError("Provider returned invalid JSON") from None

                detail = _sanitize_error(
                    body.decode("utf-8", errors="replace"), known_secret=api_key
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
        except OpsProviderError as exc:
            raise OpsProviderError(
                _sanitize_error(exc.detail, known_secret=api_key)
            ) from None
        except (httpx.HTTPError, OSError, ValueError) as exc:
            detail = _sanitize_error(exc, known_secret=api_key)
            raise OpsProviderError(f"Provider request failed: {detail}") from None

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
        self,
        provider: Provider,
        messages: list[dict[str, Any]],
        *,
        timeout_seconds: float | None = None,
    ) -> AssistantTurn:
        deadline = (
            self._monotonic() + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        initial_payload = self._chat_payload(provider, messages, max_tokens=2048)
        response = self._request_json(
            provider,
            "POST",
            "/chat/completions",
            payload=initial_payload,
            allow_response_format_fallback=True,
            deadline=deadline,
        )
        try:
            return self._parse_turn(response)
        except _IncompleteProviderResponse as initial_error:
            repair_messages = _compact_repair_messages(messages, str(initial_error))
            repair_payload = self._chat_payload(
                provider, repair_messages, max_tokens=4096
            )
            repaired = self._request_json(
                provider,
                "POST",
                "/chat/completions",
                payload=repair_payload,
                allow_response_format_fallback=True,
                deadline=deadline,
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
