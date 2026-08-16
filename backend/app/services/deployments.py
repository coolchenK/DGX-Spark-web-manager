from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import docker
import httpx
from docker.types import DeviceRequest, LogConfig
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Deployment, ModelAsset
from app.runtime.base import (
    DeploymentSpec,
    GenerationDefaults,
    ResolvedDeploymentSpec,
    RuntimeAdapter,
    deterministic_container_name,
    validate_model_path,
)
from app.services.diagnostics import redact_log
from app.services.draft_models import DraftCompatibilityService
from app.services.model_evidence import ModelEvidenceLoader
from app.services.resource_estimator import ResourceEstimator
from app.services.runtime_capabilities import RuntimeCapabilityService
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
        runtime_capability_service: RuntimeCapabilityService | None = None,
        evidence_loader: ModelEvidenceLoader | None = None,
        draft_service: DraftCompatibilityService | None = None,
        resource_estimator: ResourceEstimator | None = None,
        system_snapshot: Callable[[], Mapping[str, Any] | Any] | None = None,
        docker_client: Any | None = None,
    ):
        self.adapters = adapters
        self.session_factory = session_factory
        self.model_roots = model_roots
        self.host_model_roots = host_model_roots or model_roots
        if len(self.model_roots) != len(self.host_model_roots):
            raise ValueError("Model roots and host model roots must have the same length")
        self.startup_timeout_seconds = startup_timeout_seconds
        self.runtime_capability_service = runtime_capability_service
        self.evidence_loader = evidence_loader or ModelEvidenceLoader()
        self.resource_estimator = resource_estimator or ResourceEstimator()
        self.draft_service = draft_service or DraftCompatibilityService(
            evidence_loader=self.evidence_loader,
            resource_estimator=self.resource_estimator,
        )
        self.system_snapshot = system_snapshot
        self._docker_client = docker_client

    def docker_client(self):
        return self._docker_client or docker.from_env()

    def adapter(self, runtime: str) -> RuntimeAdapter:
        try:
            return self.adapters[runtime]
        except KeyError as exc:
            raise ValueError(f"Unsupported runtime: {runtime}") from exc

    def _root_details(self, model_path: Path) -> tuple[Path, Path, Path]:
        resolved = validate_model_path(model_path, self.model_roots)
        for index, root in enumerate(self.model_roots):
            resolved_root = root.resolve()
            if resolved == resolved_root or resolved.is_relative_to(resolved_root):
                return resolved_root, self.host_model_roots[index], resolved.relative_to(
                    resolved_root
                )
        raise ValueError("Model path is outside configured model roots")

    @staticmethod
    def _container_path(bind: str, relative: Path) -> str:
        suffix = "/".join(relative.parts)
        return f"{bind}/{suffix}" if suffix else bind

    def resolve_spec(self, db: Session, spec: DeploymentSpec) -> ResolvedDeploymentSpec:
        target = db.get(ModelAsset, spec.model_id) if spec.model_id else None
        if target is None or target.status != "available" or not target.local_path:
            raise ValueError("Base model is missing or unavailable")
        try:
            base_path = validate_model_path(Path(target.local_path), self.model_roots)
            base_root, base_host_root, base_relative = self._root_details(base_path)
        except ValueError as exc:
            raise ValueError("Base model path is unavailable") from exc

        if self.runtime_capability_service is None:
            raise ValueError("Runtime capabilities could not be verified")
        try:
            capabilities = self.runtime_capability_service.get(spec.runtime, spec.image)
        except Exception as exc:
            raise ValueError("Runtime capabilities could not be verified") from exc

        internal: dict[str, Any] = {
            "model_path": str(base_path),
            "base_model_root": str(base_root),
            "base_host_model_root": str(base_host_root),
            "base_container_model_path": self._container_path("/models", base_relative),
        }
        if spec.speculative is not None:
            draft = db.get(ModelAsset, spec.speculative.draft_model_id)
            if draft is None or draft.status != "available" or not draft.local_path:
                raise ValueError("Draft Model is missing or unavailable")
            if draft.id == target.id:
                raise ValueError("Base model and Draft Model must be different")
            try:
                draft_path = validate_model_path(Path(draft.local_path), self.model_roots)
                draft_root, draft_host_root, draft_relative = self._root_details(draft_path)
            except ValueError as exc:
                raise ValueError("Draft Model path is unavailable") from exc
            method = spec.speculative.method
            if method not in capabilities.speculative_methods:
                raise ValueError("Speculative method is unsupported by the runtime")
            runtime_method = capabilities.method_mapping.get(method)
            if not runtime_method:
                raise ValueError("Speculative method mapping is unavailable")
            draft_bind = "/models" if draft_root == base_root else "/draft-models"
            internal.update(
                {
                    "resolved_draft_model_path": str(draft_path),
                    "draft_model_root": str(draft_root),
                    "draft_host_model_root": str(draft_host_root),
                    "draft_container_model_path": self._container_path(
                        draft_bind, draft_relative
                    ),
                    "speculative_runtime_method": runtime_method,
                }
            )
        return ResolvedDeploymentSpec.model_validate(
            {**spec.model_dump(mode="json"), **internal}
        )

    def preview(
        self,
        db: Session,
        spec: DeploymentSpec,
        *,
        exclude_deployment_id: str | None = None,
    ) -> dict[str, Any]:
        _, preview = self._preflight(
            db,
            spec,
            exclude_deployment_id=exclude_deployment_id,
        )
        return preview

    @staticmethod
    def _normalized_generation_defaults(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        try:
            return GenerationDefaults.model_validate(value).model_dump(
                mode="json", exclude_none=True
            )
        except (ValueError, TypeError):
            return {}

    def _validate_route_defaults(
        self,
        db: Session,
        spec: DeploymentSpec,
        *,
        exclude_deployment_id: str | None,
    ) -> None:
        if spec.route_alias is None:
            return
        current = spec.generation_defaults.model_dump(mode="json", exclude_none=True)
        for deployment in db.scalars(select(Deployment)).all():
            if deployment.id == exclude_deployment_id:
                continue
            config = deployment.config if isinstance(deployment.config, Mapping) else {}
            stored_spec = config.get("spec")
            if not isinstance(stored_spec, Mapping):
                continue
            if stored_spec.get("route_alias") != spec.route_alias:
                continue
            stored = self._normalized_generation_defaults(
                stored_spec.get("generation_defaults")
            )
            if stored != current:
                raise ValueError("Shared route generation defaults must match")

    @staticmethod
    def _system_memory(snapshot: Mapping[str, Any] | Any) -> Mapping[str, Any]:
        if hasattr(snapshot, "model_dump"):
            snapshot = snapshot.model_dump(mode="json")
        if not isinstance(snapshot, Mapping):
            raise ValueError("System resource snapshot is unavailable")
        memory = snapshot.get("memory", snapshot)
        if not isinstance(memory, Mapping):
            raise ValueError("System resource snapshot is unavailable")
        return memory

    @staticmethod
    def _mount_preview(resolved: ResolvedDeploymentSpec) -> dict[str, Any]:
        mounts: dict[str, Any] = {
            "base": {
                "host_root": resolved.base_host_model_root,
                "container_bind": "/models",
                "model_path": resolved.model_path,
                "container_model_path": resolved.base_container_model_path,
                "mode": "ro",
            }
        }
        if resolved.speculative is not None:
            mounts["draft"] = {
                "host_root": resolved.draft_host_model_root,
                "container_bind": (
                    "/models"
                    if resolved.draft_model_root == resolved.base_model_root
                    else "/draft-models"
                ),
                "model_path": resolved.resolved_draft_model_path,
                "container_model_path": resolved.draft_container_model_path,
                "mode": "ro",
                "reuses_base_mount": resolved.draft_model_root
                == resolved.base_model_root,
            }
        return mounts

    def _preflight(
        self,
        db: Session,
        spec: DeploymentSpec,
        *,
        exclude_deployment_id: str | None = None,
    ) -> tuple[ResolvedDeploymentSpec, dict[str, Any]]:
        resolved = self.resolve_spec(db, spec)
        adapter = self.adapter(resolved.runtime)
        model_path = adapter.validate(resolved)
        compatibility = adapter.check_model_compatibility(model_path)
        if not compatibility.get("compatible"):
            raise ValueError("Base model is incompatible")
        try:
            capabilities = self.runtime_capability_service.get(
                resolved.runtime, resolved.image
            )
        except Exception as exc:
            raise ValueError("Runtime capabilities could not be verified") from exc

        self._validate_route_defaults(
            db,
            resolved,
            exclude_deployment_id=exclude_deployment_id,
        )
        if self.system_snapshot is None:
            raise ValueError("System resource snapshot is unavailable")
        try:
            snapshot = self.system_snapshot()
            memory = self._system_memory(snapshot)
        except Exception as exc:
            raise ValueError("System resource snapshot is unavailable") from exc

        target = db.get(ModelAsset, resolved.model_id)
        if target is None:
            raise ValueError("Base model is missing or unavailable")
        draft = None
        selected_candidate = None
        if resolved.speculative is not None:
            draft = db.get(ModelAsset, resolved.speculative.draft_model_id)
            candidates = self.draft_service.list_candidates(
                db, target, capabilities, snapshot
            )
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.model_id == resolved.speculative.draft_model_id
                ),
                None,
            )
            if selected_candidate is None:
                raise ValueError("Draft Model compatibility could not be verified")
            if (
                selected_candidate.status == "incompatible"
                or selected_candidate.method != resolved.speculative.method
            ):
                raise ValueError("Draft Model is incompatible")
            if (
                selected_candidate.status == "review"
                and not resolved.speculative.manual_review_acknowledged
            ):
                raise ValueError("Draft Model review acknowledgement is required")

        try:
            evidence = self.evidence_loader.load(resolved.model_path)
            estimate = self.resource_estimator.estimate(
                model_size_bytes=target.size_bytes,
                draft_size_bytes=draft.size_bytes if draft is not None else 0,
                config=evidence.config,
                context_length=resolved.context_length,
                max_concurrency=resolved.max_concurrency,
                system_memory=memory,
            )
        except Exception as exc:
            raise ValueError("Deployment resource requirements could not be verified") from exc
        reason = "; ".join(estimate.reasons[:4]) or "resource limit exceeded"
        if estimate.decision == "blocked":
            raise ValueError(f"Deployment is blocked by current resources: {reason}")
        if estimate.decision == "warning" and not resolved.resource_warning_acknowledged:
            raise ValueError("Resource warning acknowledgement is required")

        preview = adapter.preview(resolved)
        warnings = list(capabilities.warnings)
        if estimate.decision == "warning":
            warnings.append("Current available unified memory requires deployment review")
        preview.update(
            {
                "spec": resolved.public_dump(),
                "runtime_capabilities": capabilities.model_dump(mode="json"),
                "resource_estimate": estimate.model_dump(mode="json"),
                "draft_candidate": (
                    selected_candidate.model_dump(mode="json")
                    if selected_candidate is not None
                    else None
                ),
                "mounts": self._mount_preview(resolved),
                "generation_defaults": resolved.generation_defaults.model_dump(
                    mode="json", exclude_none=True
                ),
                "speculative": (
                    resolved.speculative.model_dump(mode="json")
                    if resolved.speculative is not None
                    else None
                ),
                "recommendation": (
                    resolved.recommendation.model_dump(mode="json")
                    if resolved.recommendation is not None
                    else None
                ),
                "warnings": warnings,
            }
        )
        return resolved, preview

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
        spec: ResolvedDeploymentSpec,
        adapter: RuntimeAdapter,
        name: str,
    ) -> Any:
        if not isinstance(spec, ResolvedDeploymentSpec):
            raise ValueError("Resolved deployment spec is required")
        if not spec.base_host_model_root or not spec.base_container_model_path:
            raise ValueError("Resolved base model mount is required")
        volumes = {
            spec.base_host_model_root: {"bind": "/models", "mode": "ro"}
        }
        if (
            spec.speculative is not None
            and spec.draft_model_root != spec.base_model_root
        ):
            if not spec.draft_host_model_root:
                raise ValueError("Resolved Draft Model mount is required")
            volumes[spec.draft_host_model_root] = {
                "bind": "/draft-models",
                "mode": "ro",
            }
        return client.containers.run(
            spec.image,
            command=adapter.command(spec),
            name=name,
            detach=True,
            ports={"8000/tcp": spec.port},
            volumes=volumes,
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
            resolved, preview = self._preflight(db, spec)
        adapter = self.adapter(resolved.runtime)
        client = self.docker_client()
        name = deterministic_container_name(resolved.name)
        created_container = False
        try:
            container = client.containers.get(name)
            labels = (container.attrs.get("Config") or {}).get("Labels") or {}
            if labels.get("com.dgx-spark-manager.managed") != "true":
                raise RuntimeError(f"Container name {name} is already used by an unmanaged service")
            if container.status != "running":
                adapter.start(container)
        except docker.errors.NotFound:
            container = self._run_container(client, resolved, adapter, name)
            created_container = True
        endpoint = f"http://127.0.0.1:{resolved.port}"
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
                name=resolved.name,
                model_id=resolved.model_id,
                runtime=resolved.runtime,
                container_id=container.id,
                container_name=name,
                endpoint_url=endpoint,
                api_model_name=resolved.api_model_name,
                status="running",
                health="healthy",
                managed=True,
                image=resolved.image,
                port=resolved.port,
                config=preview,
                capabilities=adapter.openai_capabilities(),
            )
            db.add(deployment)
            db.commit()
            db.refresh(deployment)
            deployment_id = deployment.id
        return {"deployment_id": deployment_id, "container_name": name, "endpoint_url": endpoint}

    def update_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        deployment_id = str(payload["deployment_id"])
        spec = DeploymentSpec.model_validate(payload["spec"])
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
            resolved, preview = self._preflight(
                db,
                spec,
                exclude_deployment_id=deployment_id,
            )
            old_container_id = deployment.container_id

        adapter = self.adapter(resolved.runtime)
        client = self.docker_client()
        old_container = client.containers.get(old_container_id)
        old_container.reload()
        old_name = old_container.name
        was_running = old_container.status == "running"
        backup_name = f"{old_name[:43]}-backup-{deployment_id[:8]}"
        new_name = deterministic_container_name(resolved.name)
        new_container = None
        record_updated = False
        try:
            if was_running:
                adapter.stop(old_container, timeout=30)
            old_container.rename(backup_name)
            context.update(progress=20, message=f"Creating replacement container {new_name}")
            new_container = self._run_container(client, resolved, adapter, new_name)
            endpoint = f"http://127.0.0.1:{resolved.port}"
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
                deployment.name = resolved.name
                deployment.model_id = resolved.model_id
                deployment.runtime = resolved.runtime
                deployment.container_id = new_container.id
                deployment.container_name = new_name
                deployment.endpoint_url = endpoint
                deployment.api_model_name = resolved.api_model_name
                deployment.status = "running"
                deployment.health = "healthy"
                deployment.image = resolved.image
                deployment.port = resolved.port
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

