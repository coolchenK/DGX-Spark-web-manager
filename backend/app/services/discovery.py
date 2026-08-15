from __future__ import annotations

from pathlib import Path
from typing import Any

import docker
import httpx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Deployment, ModelAsset, utc_now


def parse_hf_cache_repository(directory_name: str) -> str | None:
    if not directory_name.startswith("models--"):
        return None
    parts = directory_name.removeprefix("models--").split("--", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        return None
    return "/".join(parts)


def infer_runtime(image: str, command: list[str]) -> str | None:
    haystack = " ".join([image, *command]).lower()
    if "sglang" in haystack:
        return "sglang"
    if "vllm" in haystack:
        return "vllm"
    if "llama-server" in haystack or "llama.cpp" in haystack:
        return "llama.cpp"
    if "tritonserver" in haystack:
        return "triton"
    return None


def _argument(command: list[str], *names: str) -> str | None:
    for index, value in enumerate(command):
        if value in names and index + 1 < len(command):
            return command[index + 1]
        for name in names:
            prefix = f"{name}="
            if value.startswith(prefix):
                return value.removeprefix(prefix)
    return None


def container_candidate(attrs: dict[str, Any]) -> dict[str, Any] | None:
    config = attrs.get("Config") or {}
    command = [str(item) for item in (config.get("Cmd") or [])]
    image = str(config.get("Image") or "")
    runtime = infer_runtime(image, command)
    if runtime is None:
        return None

    internal_port = _argument(command, "--port") or "8000"
    ports = (attrs.get("NetworkSettings") or {}).get("Ports") or {}
    bindings = ports.get(f"{internal_port}/tcp") or []
    host_port = bindings[0].get("HostPort") if bindings else internal_port
    if not host_port or not str(host_port).isdigit():
        return None

    model_name = _argument(command, "--served-model-name")
    model_path = _argument(command, "--model", "--model-path")
    model_name = model_name or model_path or attrs.get("Name", "").lstrip("/")
    state = attrs.get("State") or {}
    health = (state.get("Health") or {}).get("Status")
    labels = config.get("Labels") or {}
    return {
        "container_id": attrs.get("Id"),
        "container_name": str(attrs.get("Name") or "").lstrip("/"),
        "name": str(attrs.get("Name") or "").lstrip("/"),
        "runtime": runtime,
        "image": image,
        "endpoint_url": f"http://127.0.0.1:{host_port}",
        "port": int(host_port),
        "api_model_name": str(model_name),
        "status": state.get("Status", "unknown"),
        "health": health or ("healthy" if state.get("Status") == "running" else "unknown"),
        "managed": labels.get("com.dgx-spark-manager.managed") == "true",
        "config": {"command": command, "model_path": model_path},
    }


def directory_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


class DiscoveryService:
    def __init__(self, model_roots: tuple[Path, ...]):
        self.model_roots = model_roots

    @staticmethod
    def _docker_client():
        return docker.from_env()

    @staticmethod
    def _probe(candidate: dict[str, Any]) -> dict[str, Any]:
        try:
            response = httpx.get(f"{candidate['endpoint_url']}/v1/models", timeout=3)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data") if isinstance(payload, dict) else []
            first = models[0] if isinstance(models, list) and models else {}
            return {
                "health": "healthy",
                "api_model_name": first.get("id") or candidate["api_model_name"],
                "max_model_len": first.get("max_model_len"),
            }
        except (httpx.HTTPError, ValueError):
            return {"health": "unhealthy"}

    def scan_containers(self, db: Session) -> list[Deployment]:
        client = self._docker_client()
        discovered: list[Deployment] = []
        for container in client.containers.list(all=True):
            container.reload()
            candidate = container_candidate(container.attrs)
            if not candidate:
                continue
            probe = self._probe(candidate)
            candidate.update(probe)
            existing = db.scalar(
                select(Deployment).where(
                    or_(
                        Deployment.container_id == candidate["container_id"],
                        Deployment.container_name == candidate["container_name"],
                    )
                )
            )
            if existing is None:
                existing = Deployment(
                    name=candidate["name"],
                    runtime=candidate["runtime"],
                    endpoint_url=candidate["endpoint_url"],
                    api_model_name=candidate["api_model_name"],
                )
                db.add(existing)
            for key in (
                "container_id",
                "container_name",
                "runtime",
                "image",
                "endpoint_url",
                "port",
                "api_model_name",
                "status",
                "health",
                "managed",
                "config",
            ):
                setattr(existing, key, candidate[key])
            existing.capabilities = ["chat", "completion", "tools"]
            existing.last_checked_at = utc_now()
            discovered.append(existing)
        db.commit()
        return discovered

    def scan_models(self, db: Session) -> list[ModelAsset]:
        discovered: list[ModelAsset] = []
        for root in self.model_roots:
            if not root.exists() or not root.is_dir():
                continue
            for child in root.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                repository_id = parse_hf_cache_repository(child.name)
                source = "huggingface" if repository_id else "local"
                existing = db.scalar(select(ModelAsset).where(ModelAsset.local_path == str(child)))
                if existing is None:
                    existing = ModelAsset(
                        name=repository_id or child.name,
                        source=source,
                        repository_id=repository_id,
                        local_path=str(child),
                        capabilities=["chat", "completion"],
                    )
                    db.add(existing)
                existing.size_bytes = directory_size(child)
                existing.status = "available"
                discovered.append(existing)
        db.commit()
        return discovered

    def scan_all(self, db: Session) -> dict[str, int | str | None]:
        models = self.scan_models(db)
        try:
            deployments = self.scan_containers(db)
            error = None
        except docker.errors.DockerException as exc:
            deployments = []
            error = str(exc)
        return {"models": len(models), "deployments": len(deployments), "docker_error": error}

