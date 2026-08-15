from __future__ import annotations

import json
import re
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Deployment, Provider
from app.services.providers import ProviderService

ALLOWED_OPERATIONS = {
    "start_deployment",
    "stop_deployment",
    "restart_deployment",
    "rescan_inventory",
}


def build_diagnostic_request(
    *, model: str, prompt: str, actual_context: dict[str, Any]
) -> dict[str, Any]:
    bounded_context = {
        **actual_context,
        "deployments": [
            {**deployment, "logs": str(deployment.get("logs") or "")[-2500:]}
            for deployment in actual_context.get("deployments", [])
        ],
    }
    return {
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 512,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You diagnose a DGX Spark. Return concise JSON with summary, diagnosis, risk "
                    "and steps. Each step may only use start_deployment, stop_deployment, "
                    "restart_deployment, rescan_inventory, or explain_only. "
                    "Never emit shell. Keep the entire response under 600 words."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"request": prompt, "observed_context": bounded_context},
                    ensure_ascii=True,
                ),
            },
        ],
    }


def parse_diagnostic_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Diagnostic response must be a JSON object")
    return value


def sanitize_steps(
    steps: list[dict[str, Any]], *, known_deployment_ids: set[str]
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for step in steps[:20]:
        operation = str(step.get("operation") or "explain_only")
        deployment_id = step.get("deployment_id")
        executable = operation in ALLOWED_OPERATIONS
        if operation != "rescan_inventory":
            executable = executable and deployment_id in known_deployment_ids
        sanitized.append(
            {
                "operation": operation,
                "deployment_id": deployment_id if deployment_id in known_deployment_ids else None,
                "reason": str(step.get("reason") or "No reason supplied")[:2000],
                "impact": str(step.get("impact") or "")[:2000],
                "rollback": str(step.get("rollback") or "")[:2000],
                "executable": bool(executable),
            }
        )
    return sanitized


def redact_log(value: str) -> str:
    patterns = [
        (re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s]+"), r"\1[REDACTED]"),
        (re.compile(r"(?i)(api[_-]?key[=:]\s*)[^\s,]+"), r"\1[REDACTED]"),
        (re.compile(r"(?i)(hf_[a-z0-9]{10,})"), "[REDACTED_HF_TOKEN]"),
    ]
    result = value
    for pattern, replacement in patterns:
        result = pattern.sub(replacement, result)
    return result[-30_000:]


class DiagnosticService:
    def __init__(self, provider_service: ProviderService, system_service, deployment_service):
        self.provider_service = provider_service
        self.system_service = system_service
        self.deployment_service = deployment_service

    def context(self, db: Session, deployment_id: str | None = None) -> dict[str, Any]:
        deployments = list(db.scalars(select(Deployment).order_by(Deployment.name)))
        deployment_rows: list[dict[str, Any]] = []
        for deployment in deployments:
            if deployment_id and deployment.id != deployment_id:
                continue
            logs = ""
            if deployment.container_id:
                try:
                    logs = redact_log(self.deployment_service.logs(deployment, tail=200))[-8000:]
                except Exception as exc:
                    logs = f"Logs unavailable: {exc}"
            deployment_rows.append(
                {
                    "id": deployment.id,
                    "name": deployment.name,
                    "runtime": deployment.runtime,
                    "model": deployment.api_model_name,
                    "status": deployment.status,
                    "health": deployment.health,
                    "managed": deployment.managed,
                    "logs": logs,
                }
            )
        return {"system": self.system_service.snapshot(), "deployments": deployment_rows}

    def diagnose(
        self,
        db: Session,
        *,
        provider: Provider,
        prompt: str,
        deployment_id: str | None,
    ) -> dict[str, Any]:
        actual_context = self.context(db, deployment_id)
        response = httpx.post(
            f"{provider.base_url}/chat/completions",
            headers=self.provider_service.authorization_headers(provider),
            json=build_diagnostic_request(
                model=provider.default_model,
                prompt=prompt,
                actual_context=actual_context,
            ),
            timeout=provider.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = parse_diagnostic_content(content)
        known_ids = {item["id"] for item in actual_context["deployments"]}
        return {
            "summary": str(parsed.get("summary") or "Diagnostic result")[:4000],
            "diagnosis": str(parsed.get("diagnosis") or "")[:20_000],
            "risk": str(parsed.get("risk") or "medium")[:32],
            "steps": sanitize_steps(parsed.get("steps") or [], known_deployment_ids=known_ids),
            "context": actual_context,
        }
