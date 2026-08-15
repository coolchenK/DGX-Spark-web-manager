from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import docker
import httpx
from docker.types import DeviceRequest, LogConfig
from sqlalchemy import or_, select
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
        adapter: RuntimeAdapter | None = None,
        progress_start: float = 25,
        progress_end: float = 90,
    ) -> bool:
        attempts = max(1, math.ceil(self.startup_timeout_seconds / 2))
        for attempt in range(attempts):
            context.check_control()
            if adapter is not None:
                if adapter.health_check(endpoint, timeout=3):
                    return True
            else:
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

    def _run_container(
        self,
        client: Any,
        spec: DeploymentSpec,
        adapter: RuntimeAdapter,
        name: str,
    ) -> Any:
        model_path = Path(spec.model_path).resolve()
        host_mount_root = resolve_host_model_mount(
            model_path,
            self.model_roots,
            self.host_model_roots,
        )
        return client.containers.run(
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

    @staticmethod
    def _startup_logs(adapter: RuntimeAdapter, container: Any) -> str:
        try:
            return redact_log(adapter.logs(container, tail=200))[-4000:]
        except Exception as exc:
            return f"Container logs unavailable: {exc}"

    def create_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        spec = DeploymentSpec.model_validate(payload)
        adapter = self.adapter(spec.runtime)
        preview = adapter.preview(spec)
        client = self.docker_client()
        name = deterministic_container_name(spec.name)
        created_container = False
        with self.session_factory() as db:
            existing = db.scalar(
                select(Deployment).where(
                    or_(
                        Deployment.name == spec.name,
                        Deployment.api_model_name == spec.api_model_name,
                    )
                )
            )
            if existing:
                return {
                    "deployment_id": existing.id,
                    "container_name": existing.container_name,
                    "endpoint_url": existing.endpoint_url,
                    "idempotent": True,
                }
        try:
            container = client.containers.get(name)
            labels = (container.attrs.get("Config") or {}).get("Labels") or {}
            if labels.get("com.dgx-spark-manager.managed") != "true":
                raise RuntimeError(f"Container name {name} is already used by an unmanaged service")
            if container.status != "running":
                adapter.start(container)
        except docker.errors.NotFound:
            container = self._run_container(client, spec, adapter, name)
            created_container = True
        endpoint = f"http://127.0.0.1:{spec.port}"
        context.update(progress=25, message=f"Container {name} started; waiting for health")
        healthy = self.wait_for_health(context, endpoint, adapter=adapter)
        if not healthy:
            logs = self._startup_logs(adapter, container)
            context.update(message=f"Startup failed. Last container logs:\n{logs}")
            if created_container:
                adapter.stop(container, timeout=15)
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

    def update_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        deployment_id = str(payload["deployment_id"])
        spec = DeploymentSpec.model_validate(payload["spec"])
        adapter = self.adapter(spec.runtime)
        preview = adapter.preview(spec)
        with self.session_factory() as db:
            deployment = db.get(Deployment, deployment_id)
            if not deployment or not deployment.container_id:
                raise ValueError("Deployment or container was not found")
            if not deployment.managed:
                raise ValueError("Discovered containers cannot be edited")
            conflict = db.scalar(
                select(Deployment).where(
                    Deployment.id != deployment_id,
                    or_(
                        Deployment.name == spec.name,
                        Deployment.api_model_name == spec.api_model_name,
                    ),
                )
            )
            if conflict:
                raise ValueError("Another deployment uses this name or API model name")
            old_container_id = deployment.container_id
            old_model_id = deployment.model_id

        client = self.docker_client()
        old_container = client.containers.get(old_container_id)
        old_container.reload()
        old_name = old_container.name
        was_running = old_container.status == "running"
        backup_name = f"{old_name[:43]}-backup-{deployment_id[:8]}"
        new_name = deterministic_container_name(spec.name)
        new_container = None
        record_updated = False
        try:
            if was_running:
                adapter.stop(old_container, timeout=30)
            old_container.rename(backup_name)
            context.update(progress=20, message=f"Creating replacement container {new_name}")
            new_container = self._run_container(client, spec, adapter, new_name)
            endpoint = f"http://127.0.0.1:{spec.port}"
            if not self.wait_for_health(
                context,
                endpoint,
                adapter=adapter,
                progress_start=30,
                progress_end=92,
            ):
                logs = self._startup_logs(adapter, new_container)
                raise RuntimeError(
                    f"Updated deployment did not become healthy within "
                    f"{self.startup_timeout_seconds} seconds. Last logs:\n{logs}"
                )
            new_container.reload()
            with self.session_factory() as db:
                deployment = db.get(Deployment, deployment_id)
                if not deployment:
                    raise ValueError("Deployment was removed while update was running")
                deployment.name = spec.name
                deployment.model_id = spec.model_id or old_model_id
                deployment.runtime = spec.runtime
                deployment.container_id = new_container.id
                deployment.container_name = new_name
                deployment.endpoint_url = endpoint
                deployment.api_model_name = spec.api_model_name
                deployment.status = "running"
                deployment.health = "healthy"
                deployment.image = spec.image
                deployment.port = spec.port
                deployment.config = preview
                deployment.capabilities = adapter.openai_capabilities()
                db.commit()
            record_updated = True
            try:
                old_container.remove()
            except Exception as exc:
                context.update(message=f"Old stopped container cleanup deferred: {exc}")
            return {
                "deployment_id": deployment_id,
                "container_name": new_name,
                "endpoint_url": endpoint,
            }
        except Exception:
            if not record_updated:
                if new_container is not None:
                    try:
                        new_container.remove(force=True)
                    except Exception:
                        pass
                try:
                    old_container.rename(old_name)
                    if was_running:
                        adapter.start(old_container)
                except Exception:
                    pass
            raise

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
            adapter = self.adapter(deployment.runtime)
            if action in {"start", "restart"}:
                deployment.status = "starting"
                deployment.health = "unknown"
                db.commit()
        container = self.docker_client().containers.get(container_id)
        context.update(progress=20, message=f"Executing {action} on {container.name}")
        if action == "start":
            adapter.start(container)
            new_status = "running"
        elif action == "stop":
            adapter.stop(container, timeout=30)
            new_status = "exited"
        elif action == "restart":
            adapter.restart(container, timeout=30)
            new_status = "running"
        else:
            adapter.uninstall(container)
            new_status = "deleted"
        health = "unknown"
        if action in {"start", "restart"}:
            context.update(progress=30, message="Waiting for runtime health")
            if not self.wait_for_health(
                context,
                endpoint_url,
                adapter=adapter,
                progress_start=30,
                progress_end=95,
            ):
                logs = self._startup_logs(adapter, container)
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
        return redact_log(self.adapter(deployment.runtime).logs(container, tail=tail))

