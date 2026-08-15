from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import docker
import httpx
from docker.types import DeviceRequest
from sqlalchemy.orm import Session, sessionmaker

from app.models import Deployment
from app.runtime.base import DeploymentSpec, RuntimeAdapter, deterministic_container_name
from app.tasks.engine import TaskContext


class DeploymentService:
    def __init__(
        self,
        *,
        adapters: dict[str, RuntimeAdapter],
        session_factory: sessionmaker[Session],
        model_roots: tuple[Path, ...],
    ):
        self.adapters = adapters
        self.session_factory = session_factory
        self.model_roots = model_roots

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

    def create_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        spec = DeploymentSpec.model_validate(payload)
        adapter = self.adapter(spec.runtime)
        preview = adapter.preview(spec)
        client = self.docker_client()
        name = deterministic_container_name(spec.name)
        try:
            container = client.containers.get(name)
            labels = (container.attrs.get("Config") or {}).get("Labels") or {}
            if labels.get("com.dgx-spark-manager.managed") != "true":
                raise RuntimeError(f"Container name {name} is already used by an unmanaged service")
            if container.status != "running":
                container.start()
        except docker.errors.NotFound:
            model_path = Path(spec.model_path).resolve()
            mount_root = next(
                root.resolve()
                for root in self.model_roots
                if model_path == root.resolve() or model_path.is_relative_to(root.resolve())
            )
            container = client.containers.run(
                spec.image,
                command=adapter.command(spec),
                name=name,
                detach=True,
                ports={"8000/tcp": spec.port},
                volumes={str(mount_root): {"bind": "/models", "mode": "ro"}},
                labels={
                    "com.dgx-spark-manager.managed": "true",
                    "com.dgx-spark-manager.model": spec.api_model_name,
                    "com.dgx-spark-manager.runtime": spec.runtime,
                },
                restart_policy={"Name": "unless-stopped"},
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                environment={"HF_HUB_OFFLINE": "1"},
            )
        endpoint = f"http://127.0.0.1:{spec.port}"
        context.update(progress=25, message=f"Container {name} started; waiting for health")
        healthy = False
        for attempt in range(60):
            context.check_control()
            try:
                response = httpx.get(f"{endpoint}/v1/models", timeout=3)
                if response.is_success:
                    healthy = True
                    break
            except httpx.HTTPError:
                pass
            context.update(progress=min(25 + attempt, 90))
            time.sleep(2)
        if not healthy:
            container.stop(timeout=15)
            container.remove()
            raise RuntimeError("Deployment did not become healthy within 120 seconds")
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
                capabilities=["chat", "completion", "tools"],
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
        with self.session_factory() as db:
            deployment = db.get(Deployment, deployment_id)
            if deployment:
                if action == "delete":
                    db.delete(deployment)
                else:
                    deployment.status = new_status
                    deployment.health = "unknown" if action == "stop" else "healthy"
                db.commit()
        return {"deployment_id": deployment_id, "action": action, "status": new_status}

    def logs(self, deployment: Deployment, tail: int = 500) -> str:
        if not deployment.container_id:
            raise ValueError("Deployment has no container")
        container = self.docker_client().containers.get(deployment.container_id)
        value = container.logs(tail=min(max(tail, 1), 5000), timestamps=True)
        return value.decode("utf-8", errors="replace")[-500_000:]

