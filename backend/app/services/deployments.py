from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import docker
import httpx
from docker.types import DeviceRequest, LogConfig
from sqlalchemy.orm import Session, sessionmaker

from app.models import Deployment
from app.runtime.base import DeploymentSpec, RuntimeAdapter, deterministic_container_name
from app.services.diagnostics import redact_log
from app.tasks.engine import TaskContext


def resolve_host_model_mount(
    model_path: Path,
    model_roots: tuple[Path, ...],
    host_model_roots: tuple[Path, ...],
) -> Path:
    if len(model_roots) != len(host_model_roots):
        raise ValueError("Model roots and host model roots must have the same length")
    resolved_model = model_path.resolve()
    for model_root, host_root in zip(model_roots, host_model_roots, strict=True):
        resolved_root = model_root.resolve()
        if resolved_model == resolved_root or resolved_model.is_relative_to(resolved_root):
            return host_root
    raise ValueError("Model path is outside configured model roots")


class DeploymentService:
    def __init__(
        self,
        *,
        adapters: dict[str, RuntimeAdapter],
        session_factory: sessionmaker[Session],
        model_roots: tuple[Path, ...],
        host_model_roots: tuple[Path, ...] | None = None,
        startup_timeout_seconds: int = 300,
    ):
        self.adapters = adapters
        self.session_factory = session_factory
        self.model_roots = model_roots
        self.host_model_roots = host_model_roots or model_roots
        if len(self.model_roots) != len(self.host_model_roots):
            raise ValueError("Model roots and host model roots must have the same length")
        self.startup_timeout_seconds = startup_timeout_seconds

    @staticmethod
    def docker_client():
        return docker.from_env()

    def adapter(self, runtime: str) -> RuntimeAdapter:
        try:
            return self.adapters[runtime]
        except KeyError as exc:
            raise ValueError(f"Unsupported runtime: {runtime}") from exc

    def preview(self, spec: DeploymentSpec) -> dict[str, Any]:
        return self.adapter(spec.runtime).preview(spec)

    def wait_for_health(
        self,
        context: TaskContext,
        endpoint: str,
        *,
        progress_start: float = 25,
        progress_end: float = 90,
    ) -> bool:
        attempts = max(1, math.ceil(self.startup_timeout_seconds / 2))
        for attempt in range(attempts):
            context.check_control()
            try:
                response = httpx.get(f"{endpoint}/v1/models", timeout=3)
                if response.is_success:
                    return True
            except httpx.HTTPError:
                pass
            fraction = (attempt + 1) / attempts
            progress = progress_start + fraction * (progress_end - progress_start)
            context.update(progress=min(progress, progress_end))
            time.sleep(2)
        return False

    def create_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        spec = DeploymentSpec.model_validate(payload)
        adapter = self.adapter(spec.runtime)
        preview = adapter.preview(spec)
        client = self.docker_client()
        name = deterministic_container_name(spec.name)
        created_container = False
        try:
            container = client.containers.get(name)
            labels = (container.attrs.get("Config") or {}).get("Labels") or {}
            if labels.get("com.dgx-spark-manager.managed") != "true":
                raise RuntimeError(f"Container name {name} is already used by an unmanaged service")
            if container.status != "running":
                container.start()
        except docker.errors.NotFound:
            model_path = Path(spec.model_path).resolve()
            host_mount_root = resolve_host_model_mount(
                model_path,
                self.model_roots,
                self.host_model_roots,
            )
            container = client.containers.run(
                spec.image,
                command=adapter.command(spec),
                name=name,
                detach=True,
                ports={"8000/tcp": spec.port},
                volumes={str(host_mount_root): {"bind": "/models", "mode": "ro"}},
                labels={
                    "com.dgx-spark-manager.managed": "true",
                    "com.dgx-spark-manager.model": spec.api_model_name,
                    "com.dgx-spark-manager.route": spec.route_alias or spec.api_model_name,
                    "com.dgx-spark-manager.runtime": spec.runtime,
                },
                restart_policy={"Name": "unless-stopped"},
                log_config=LogConfig(
                    type=LogConfig.types.JSON,
                    config={"max-size": "10m", "max-file": "5"},
                ),
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                environment={"HF_HUB_OFFLINE": "1"},
            )
            created_container = True
        endpoint = f"http://127.0.0.1:{spec.port}"
        context.update(progress=25, message=f"Container {name} started; waiting for health")
        healthy = self.wait_for_health(context, endpoint)
        if not healthy:
            try:
                logs = redact_log(
                    container.logs(tail=200, timestamps=True).decode(
                        "utf-8", errors="replace"
                    )
                )[-4000:]
            except Exception as exc:
                logs = f"Container logs unavailable: {exc}"
            context.update(message=f"Startup failed. Last container logs:\n{logs}")
            if created_container:
                container.stop(timeout=15)
                container.remove()
            detail = f" Last container logs:\n{logs}" if logs else ""
            raise RuntimeError(
                f"Deployment did not become healthy within "
                f"{self.startup_timeout_seconds} seconds.{detail}"
            )
        container.reload()
        with self.session_factory() as db:
            deployment = Deployment(
                name=spec.name,
                model_id=spec.model_id,
                runtime=spec.runtime,
                container_id=container.id,
                container_name=name,
                endpoint_url=endpoint,
                api_model_name=spec.api_model_name,
                status="running",
                health="healthy",
                managed=True,
                image=spec.image,
                port=spec.port,
                config=preview,
                capabilities=["chat", "completion"],
            )
            db.add(deployment)
            db.commit()
            db.refresh(deployment)
            deployment_id = deployment.id
        return {"deployment_id": deployment_id, "container_name": name, "endpoint_url": endpoint}

    def action_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        deployment_id = str(payload["deployment_id"])
        action = str(payload["action"])
        if action not in {"start", "stop", "restart", "delete"}:
            raise ValueError("Unsupported deployment action")
        with self.session_factory() as db:
            deployment = db.get(Deployment, deployment_id)
            if not deployment or not deployment.container_id:
                raise ValueError("Deployment or container was not found")
            if action == "delete" and not deployment.managed:
                raise ValueError("Discovered containers cannot be deleted by the manager")
            container_id = deployment.container_id
            endpoint_url = deployment.endpoint_url
            if action in {"start", "restart"}:
                deployment.status = "starting"
                deployment.health = "unknown"
                db.commit()
        container = self.docker_client().containers.get(container_id)
        context.update(progress=20, message=f"Executing {action} on {container.name}")
        if action == "start":
            container.start()
            new_status = "running"
        elif action == "stop":
            container.stop(timeout=30)
            new_status = "exited"
        elif action == "restart":
            container.restart(timeout=30)
            new_status = "running"
        else:
            container.stop(timeout=30)
            container.remove()
            new_status = "deleted"
        health = "unknown"
        if action in {"start", "restart"}:
            context.update(progress=30, message="Waiting for runtime health")
            if not self.wait_for_health(context, endpoint_url, progress_start=30, progress_end=95):
                try:
                    logs = redact_log(
                        container.logs(tail=200, timestamps=True).decode(
                            "utf-8", errors="replace"
                        )
                    )[-4000:]
                except Exception as exc:
                    logs = f"Container logs unavailable: {exc}"
                with self.session_factory() as db:
                    deployment = db.get(Deployment, deployment_id)
                    if deployment:
                        deployment.status = "running"
                        deployment.health = "unhealthy"
                        db.commit()
                context.update(message=f"Runtime health check failed:\n{logs}")
                raise RuntimeError(
                    f"Runtime did not become healthy within "
                    f"{self.startup_timeout_seconds} seconds. Last logs:\n{logs}"
                )
            health = "healthy"
        with self.session_factory() as db:
            deployment = db.get(Deployment, deployment_id)
            if deployment:
                if action == "delete":
                    db.delete(deployment)
                else:
                    deployment.status = new_status
                    deployment.health = health
                db.commit()
        return {
            "deployment_id": deployment_id,
            "action": action,
            "status": new_status,
            "health": health,
        }

    def logs(self, deployment: Deployment, tail: int = 500) -> str:
        if not deployment.container_id:
            raise ValueError("Deployment has no container")
        container = self.docker_client().containers.get(deployment.container_id)
        value = container.logs(tail=min(max(tail, 1), 5000), timestamps=True)
        return value.decode("utf-8", errors="replace")[-500_000:]

