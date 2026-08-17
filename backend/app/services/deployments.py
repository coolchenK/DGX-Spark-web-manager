from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import docker
import httpx
from docker.errors import NotFound
from docker.types import DeviceRequest, LogConfig
from sqlalchemy import delete, or_, select, update
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
from app.services.runtime_capabilities import RuntimeCapabilities, RuntimeCapabilityService
from app.tasks.engine import TaskContext

LABEL_PREFIX = "com.dgx-spark-manager."
PROTECTED_CONTAINER_NAMES = {
    "dgx-spark-web-manager",
    "dgx-spark-ops-agent",
}
CONCURRENT_DEPLOYMENT_CHANGE_ERROR = "Deployment container identity changed concurrently"
MISSING_DEPLOYMENT_SNAPSHOT_ERROR = (
    "Deployment action is missing its container snapshot; retry the action"
)


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


def _deployment_spec_fingerprint(
    spec: DeploymentSpec, *, include_model_path: bool
) -> str:
    public = spec.public_dump() if isinstance(spec, ResolvedDeploymentSpec) else spec.model_dump(
        mode="json"
    )
    if not include_model_path:
        public.pop("model_path", None)
    canonical = json.dumps(
        public,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def deployment_spec_fingerprint(spec: DeploymentSpec) -> str:
    return _deployment_spec_fingerprint(spec, include_model_path=False)


class DeploymentService:
    def __init__(
        self,
        *,
        adapters: dict[str, RuntimeAdapter],
        session_factory: sessionmaker[Session],
        model_roots: tuple[Path, ...],
        host_model_roots: tuple[Path, ...] | None = None,
        startup_timeout_seconds: int = 1200,
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

    def _resolve_spec_with_capabilities(
        self, db: Session, spec: DeploymentSpec
    ) -> tuple[ResolvedDeploymentSpec, RuntimeCapabilities]:
        adapter = self.adapter(spec.runtime)
        if spec.runtime != adapter.runtime:
            raise ValueError(
                f"Adapter {adapter.runtime} cannot deploy runtime {spec.runtime}"
            )
        if spec.image not in adapter.allowed_images:
            raise ValueError(f"Image is not allowed for {spec.runtime}")
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
        resolved = ResolvedDeploymentSpec.model_validate(
            {**spec.model_dump(mode="json"), **internal}
        )
        return resolved, capabilities

    def resolve_spec(self, db: Session, spec: DeploymentSpec) -> ResolvedDeploymentSpec:
        resolved, _ = self._resolve_spec_with_capabilities(db, spec)
        return resolved

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
        effective_route = spec.route_alias or spec.api_model_name
        current = spec.generation_defaults.model_dump(mode="json", exclude_none=True)
        for deployment in db.scalars(select(Deployment)).all():
            if deployment.id == exclude_deployment_id:
                continue
            config = deployment.config if isinstance(deployment.config, Mapping) else {}
            stored_spec_value = config.get("spec")
            stored_spec = (
                stored_spec_value if isinstance(stored_spec_value, Mapping) else {}
            )
            stored_route = (
                config.get("route_alias")
                or stored_spec.get("route_alias")
                or deployment.api_model_name
            )
            if stored_route != effective_route:
                continue
            stored_generation = stored_spec.get("generation_defaults")
            if not isinstance(stored_generation, Mapping):
                stored_generation = config.get("generation_defaults")
            stored = self._normalized_generation_defaults(stored_generation)
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

    @staticmethod
    def _expected_container_labels(
        spec: DeploymentSpec,
        *,
        task_id: str,
        spec_fingerprint: str,
        deployment_id: str | None = None,
        replaces_container_id: str | None = None,
    ) -> dict[str, str]:
        labels = {
            f"{LABEL_PREFIX}managed": "true",
            f"{LABEL_PREFIX}task-id": task_id,
            f"{LABEL_PREFIX}spec-fingerprint": spec_fingerprint,
            f"{LABEL_PREFIX}runtime": spec.runtime,
            f"{LABEL_PREFIX}model-id": spec.model_id or "",
            f"{LABEL_PREFIX}route": spec.route_alias or spec.api_model_name,
            f"{LABEL_PREFIX}image": spec.image,
            f"{LABEL_PREFIX}port": str(spec.port),
        }
        if deployment_id is not None:
            labels[f"{LABEL_PREFIX}deployment-id"] = deployment_id
        if replaces_container_id is not None:
            labels[f"{LABEL_PREFIX}replaces-container-id"] = replaces_container_id
        return labels

    @staticmethod
    def _validate_container_labels(
        container: Any,
        expected: Mapping[str, str],
        *,
        require_task: bool = True,
        require_deployment: bool = True,
        accepted_fingerprints: set[str] | None = None,
    ) -> None:
        labels = (container.attrs.get("Config") or {}).get("Labels") or {}
        for key, value in expected.items():
            if not require_task and key == f"{LABEL_PREFIX}task-id":
                continue
            if (
                not require_deployment
                and key == f"{LABEL_PREFIX}deployment-id"
                and key not in labels
            ):
                continue
            if (
                accepted_fingerprints is not None
                and key == f"{LABEL_PREFIX}spec-fingerprint"
            ):
                if labels.get(key) not in accepted_fingerprints:
                    raise ValueError(
                        "Existing container labels do not match this task and spec"
                    )
                continue
            if labels.get(key) != value:
                raise ValueError("Existing container labels do not match this task and spec")

    @staticmethod
    def _get_container_optional(client: Any, identifier: str) -> Any | None:
        try:
            return client.containers.get(identifier)
        except docker.errors.NotFound:
            return None

    @staticmethod
    def _backup_container_name(deployment_id: str) -> str:
        return f"dgx-backup-{deployment_id}"[:63]

    @staticmethod
    def _stored_spec_fingerprint(deployment: Deployment) -> str | None:
        config = deployment.config if isinstance(deployment.config, Mapping) else {}
        stored_spec = config.get("spec")
        if not isinstance(stored_spec, Mapping):
            return None
        try:
            return deployment_spec_fingerprint(DeploymentSpec.model_validate(stored_spec))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _stored_container_fingerprints(deployment: Deployment) -> set[str]:
        config = deployment.config if isinstance(deployment.config, Mapping) else {}
        stored_spec = config.get("spec")
        if not isinstance(stored_spec, Mapping):
            return set()
        try:
            spec = DeploymentSpec.model_validate(stored_spec)
        except (TypeError, ValueError):
            return set()
        return {
            deployment_spec_fingerprint(spec),
            _deployment_spec_fingerprint(spec, include_model_path=True),
        }

    def _adapter_for_spec(self, spec: DeploymentSpec) -> RuntimeAdapter:
        adapter = self.adapter(spec.runtime)
        if spec.runtime != adapter.runtime:
            raise ValueError(
                f"Adapter {adapter.runtime} cannot deploy runtime {spec.runtime}"
            )
        if spec.image not in adapter.allowed_images:
            raise ValueError(f"Image is not allowed for {spec.runtime}")
        return adapter

    def _deployment_matches_spec(
        self, deployment: Deployment, spec: DeploymentSpec
    ) -> bool:
        return (
            self._stored_spec_fingerprint(deployment)
            == deployment_spec_fingerprint(spec)
            and deployment.name == spec.name
            and deployment.model_id == spec.model_id
            and deployment.runtime == spec.runtime
            and deployment.api_model_name == spec.api_model_name
            and deployment.image == spec.image
            and deployment.port == spec.port
        )

    def _cleanup_committed_backup(
        self,
        context: TaskContext,
        client: Any,
        target: Any,
        deployment_id: str,
    ) -> None:
        backup = self._get_container_optional(
            client, self._backup_container_name(deployment_id)
        )
        if backup is None or backup.id == target.id:
            return
        target_labels = (target.attrs.get("Config") or {}).get("Labels") or {}
        backup_labels = (backup.attrs.get("Config") or {}).get("Labels") or {}
        replaced_id = target_labels.get(f"{LABEL_PREFIX}replaces-container-id")
        if (
            backup_labels.get(f"{LABEL_PREFIX}managed") != "true"
            or not replaced_id
            or backup.id != replaced_id
        ):
            context.update(
                message="backup cleanup conflict: ownership could not be verified"
            )
            return
        self._remove_owned_container(backup)

    def _recover_committed_deployment(
        self,
        context: TaskContext,
        deployment: Deployment,
        spec: DeploymentSpec,
        *,
        cleanup_backup: bool,
    ) -> dict[str, Any]:
        deployment_id = deployment.id
        container_id = deployment.container_id
        original = {
            "updated_at": deployment.updated_at,
            "status": deployment.status,
            "health": deployment.health,
            "container_name": deployment.container_name,
            "endpoint_url": deployment.endpoint_url,
            "name": deployment.name,
            "model_id": deployment.model_id,
            "runtime": deployment.runtime,
            "api_model_name": deployment.api_model_name,
            "image": deployment.image,
            "port": deployment.port,
        }
        adapter = self._adapter_for_spec(spec)
        client = self.docker_client()
        target = self._get_container_optional(client, container_id)
        if target is None:
            raise ValueError("Persisted replacement container was not found")
        expected_labels = self._expected_container_labels(
            spec,
            task_id=str(getattr(context, "task_id", "manual")),
            spec_fingerprint=deployment_spec_fingerprint(spec),
            deployment_id=deployment.id,
        )
        self._validate_container_labels(
            target,
            expected_labels,
            require_task=False,
            require_deployment=cleanup_backup,
            accepted_fingerprints=self._stored_container_fingerprints(deployment),
        )
        endpoint = f"http://127.0.0.1:{spec.port}"
        started_here = target.status != "running"
        try:
            if started_here:
                adapter.start(target)
            if not self.wait_for_health(context, endpoint, adapter=adapter):
                raise RuntimeError("Persisted replacement container is unhealthy")
            self._sync_recovered_deployment(
                deployment_id=deployment_id,
                container_id=container_id,
                original=original,
                spec=spec,
                container_name=target.name,
                endpoint=endpoint,
            )
        except BaseException:
            if started_here:
                self._coordinate_failed_recovery_start(
                    adapter, target, deployment_id, container_id
                )
            raise
        if cleanup_backup:
            self._cleanup_committed_backup(context, client, target, deployment_id)
        return {
            "deployment_id": deployment_id,
            "container_name": target.name,
            "endpoint_url": endpoint,
            "idempotent": True,
        }

    def _sync_recovered_deployment(
        self,
        *,
        deployment_id: str,
        container_id: str,
        original: Mapping[str, Any],
        spec: DeploymentSpec,
        container_name: str,
        endpoint: str,
    ) -> None:
        with self.session_factory() as db:
            current = db.get(Deployment, deployment_id)
            if (
                current is None
                or current.container_id != container_id
                or not current.managed
                or current.updated_at != original["updated_at"]
                or current.status != original["status"]
                or current.health != original["health"]
                or current.container_name != original["container_name"]
                or current.endpoint_url != original["endpoint_url"]
                or not self._deployment_matches_spec(current, spec)
            ):
                raise ValueError("Deployment changed while recovery was running")

        conditions = (
            Deployment.id == deployment_id,
            Deployment.managed.is_(True),
            Deployment.container_id == container_id,
            Deployment.updated_at == original["updated_at"],
            Deployment.status == original["status"],
            Deployment.health == original["health"],
            Deployment.container_name == original["container_name"],
            Deployment.endpoint_url == original["endpoint_url"],
            Deployment.name == original["name"],
            Deployment.model_id == original["model_id"],
            Deployment.runtime == original["runtime"],
            Deployment.api_model_name == original["api_model_name"],
            Deployment.image == original["image"],
            Deployment.port == original["port"],
        )
        with self.session_factory() as db:
            result = db.execute(
                update(Deployment)
                .where(*conditions)
                .values(
                    status="running",
                    health="healthy",
                    container_name=container_name,
                    endpoint_url=endpoint,
                )
            )
            if result.rowcount != 1:
                db.rollback()
                db.expire_all()
                db.get(Deployment, deployment_id)
                raise ValueError("Deployment changed while recovery was running")
            db.expire_all()
            updated = db.get(Deployment, deployment_id)
            if (
                updated is None
                or updated.container_id != container_id
                or not updated.managed
                or not self._deployment_matches_spec(updated, spec)
            ):
                db.rollback()
                raise ValueError("Deployment changed while recovery was running")
            db.commit()

    def _coordinate_failed_recovery_start(
        self,
        adapter: RuntimeAdapter,
        target: Any,
        deployment_id: str,
        container_id: str,
    ) -> None:
        try:
            with self.session_factory() as db:
                reference = db.scalar(
                    select(Deployment).where(
                        Deployment.container_id == container_id
                    )
                )
                should_stop = reference is None or (
                    reference.id == deployment_id
                    and (
                        reference.status in {"deleted", "dead", "exited", "stopped"}
                        or reference.health == "unhealthy"
                    )
                )
            if should_stop:
                adapter.stop(target, timeout=30)
        except BaseException:
            pass

    @staticmethod
    def _remove_owned_container(container: Any) -> None:
        try:
            container.remove(force=True)
        except BaseException:
            pass

    def _preflight(
        self,
        db: Session,
        spec: DeploymentSpec,
        *,
        exclude_deployment_id: str | None = None,
    ) -> tuple[ResolvedDeploymentSpec, dict[str, Any]]:
        resolved, capabilities = self._resolve_spec_with_capabilities(db, spec)
        adapter = self.adapter(resolved.runtime)
        model_path = adapter.validate(resolved)
        compatibility = adapter.check_model_compatibility(model_path)
        if not compatibility.get("compatible"):
            raise ValueError("Base model is incompatible")
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
                "spec_fingerprint": deployment_spec_fingerprint(resolved),
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
        *,
        task_id: str = "manual",
        spec_fingerprint: str | None = None,
        deployment_id: str | None = None,
        replaces_container_id: str | None = None,
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
        for source, mount in adapter.extra_volumes(spec).items():
            if source in volumes or any(
                existing["bind"] == mount["bind"] for existing in volumes.values()
            ):
                raise ValueError("Runtime mount conflicts with a model mount")
            volumes[source] = mount
        fingerprint = spec_fingerprint or deployment_spec_fingerprint(spec)
        return client.containers.run(
            spec.image,
            command=adapter.command(spec),
            name=name,
            detach=True,
            ports={"8000/tcp": spec.port},
            volumes=volumes,
            labels=self._expected_container_labels(
                spec,
                task_id=task_id,
                spec_fingerprint=fingerprint,
                deployment_id=deployment_id,
                replaces_container_id=replaces_container_id,
            ),
            restart_policy={"Name": "unless-stopped"},
            log_config=LogConfig(
                type=LogConfig.types.JSON,
                config={"max-size": "10m", "max-file": "5"},
            ),
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            environment={"HF_HUB_OFFLINE": "1", **adapter.environment(spec)},
        )

    @staticmethod
    def _startup_logs(adapter: RuntimeAdapter, container: Any) -> str:
        try:
            return redact_log(adapter.logs(container, tail=200))[-4000:]
        except Exception as exc:
            return f"Container logs unavailable: {exc}"

    def create_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        spec = DeploymentSpec.model_validate(payload)
        committed = None
        with self.session_factory() as db:
            matches = list(
                db.scalars(
                select(Deployment).where(
                    or_(
                        Deployment.name == spec.name,
                        Deployment.api_model_name == spec.api_model_name,
                    )
                )
                ).all()
            )
            if matches:
                if len(matches) != 1:
                    raise ValueError("Deployment identity conflicts with existing deployments")
                existing = matches[0]
                if (
                    existing.name != spec.name
                    or existing.api_model_name != spec.api_model_name
                ):
                    raise ValueError("Deployment identity conflicts with an existing deployment")
                if not self._deployment_matches_spec(existing, spec):
                    raise ValueError("Existing deployment uses a different deployment spec")
                committed = existing
            else:
                resolved, preview = self._preflight(db, spec)
                fingerprint = preview["spec_fingerprint"]
        if committed is not None:
            return self._recover_committed_deployment(
                context, committed, spec, cleanup_backup=False
            )
        adapter = self.adapter(resolved.runtime)
        client = self.docker_client()
        name = deterministic_container_name(resolved.name)
        task_id = str(getattr(context, "task_id", "manual"))
        expected_labels = self._expected_container_labels(
            resolved,
            task_id=task_id,
            spec_fingerprint=fingerprint,
        )
        created_container = False
        persisted = False
        container = None
        try:
            try:
                container = client.containers.get(name)
                self._validate_container_labels(container, expected_labels)
                if container.status != "running":
                    adapter.start(container)
            except docker.errors.NotFound:
                container = self._run_container(
                    client,
                    resolved,
                    adapter,
                    name,
                    task_id=task_id,
                    spec_fingerprint=fingerprint,
                )
                created_container = True
            endpoint = f"http://127.0.0.1:{resolved.port}"
            context.update(
                progress=25,
                message=f"Container {name} started; waiting for health",
            )
            healthy = self.wait_for_health(context, endpoint, adapter=adapter)
            if not healthy:
                logs = self._startup_logs(adapter, container)
                context.update(message=f"Startup failed. Last container logs:\n{logs}")
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
                persisted = True
                db.refresh(deployment)
                deployment_id = deployment.id
            return {
                "deployment_id": deployment_id,
                "container_name": name,
                "endpoint_url": endpoint,
            }
        except BaseException:
            if created_container and not persisted and container is not None:
                self._remove_owned_container(container)
            raise

    def update_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        deployment_id = str(payload["deployment_id"])
        spec = DeploymentSpec.model_validate(payload["spec"])
        committed = None
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
            if self._deployment_matches_spec(deployment, spec):
                committed = deployment
            else:
                resolved, preview = self._preflight(
                    db,
                    spec,
                    exclude_deployment_id=deployment_id,
                )
                current_container_id = deployment.container_id
                current_container_name = (
                    deployment.container_name
                    or deterministic_container_name(deployment.name)
                )
                current_status = deployment.status

        if committed is not None:
            return self._recover_committed_deployment(
                context, committed, spec, cleanup_backup=True
            )

        adapter = self.adapter(resolved.runtime)
        client = self.docker_client()
        task_id = str(getattr(context, "task_id", "manual"))
        fingerprint = preview["spec_fingerprint"]
        new_name = deterministic_container_name(resolved.name)
        backup_name = self._backup_container_name(deployment_id)
        endpoint = f"http://127.0.0.1:{resolved.port}"
        expected_labels = self._expected_container_labels(
            resolved,
            task_id=task_id,
            spec_fingerprint=fingerprint,
            deployment_id=deployment_id,
            replaces_container_id=current_container_id,
        )

        old_container = client.containers.get(current_container_id)
        old_container.reload()
        restore_running = current_status == "running"
        replacement = self._get_container_optional(client, new_name)
        if replacement is not None and replacement.id == old_container.id:
            replacement = None
        new_container = None
        record_updated = False
        try:
            if replacement is not None:
                self._validate_container_labels(replacement, expected_labels)
            if old_container.name == current_container_name:
                if old_container.status == "running":
                    adapter.stop(old_container, timeout=30)
                old_container.rename(backup_name)
            elif old_container.name != backup_name:
                raise ValueError("Existing backup container state is invalid")

            if replacement is None:
                context.update(
                    progress=20,
                    message=f"Creating replacement container {new_name}",
                )
                new_container = self._run_container(
                    client,
                    resolved,
                    adapter,
                    new_name,
                    task_id=task_id,
                    spec_fingerprint=fingerprint,
                    deployment_id=deployment_id,
                    replaces_container_id=current_container_id,
                )
            else:
                new_container = replacement
                if new_container.status != "running":
                    adapter.start(new_container)
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
                old_container.remove(force=True)
            except Exception as exc:
                context.update(message=f"Old stopped container cleanup deferred: {exc}")
            return {
                "deployment_id": deployment_id,
                "container_name": new_name,
                "endpoint_url": endpoint,
            }
        except BaseException:
            if not record_updated:
                if new_container is not None:
                    self._remove_owned_container(new_container)
                try:
                    if old_container.name == backup_name:
                        old_container.rename(current_container_name)
                    if restore_running and old_container.status != "running":
                        adapter.start(old_container)
                except BaseException:
                    pass
            raise

    @staticmethod
    def _normalized_container_name(value: Any) -> str:
        return str(value or "").lstrip("/")

    @staticmethod
    def _container_ids_equivalent(recorded_id: Any, actual_id: Any) -> bool:
        recorded = str(recorded_id or "").strip().lower()
        actual = str(actual_id or "").strip().lower()
        if not recorded or not actual:
            return False
        if recorded == actual:
            return True
        shorter, longer = sorted((recorded, actual), key=len)
        return (
            len(shorter) >= 12
            and all(character in "0123456789abcdef" for character in shorter)
            and all(character in "0123456789abcdef" for character in longer)
            and longer.startswith(shorter)
        )

    def _validate_action_container(
        self,
        container: Any,
        *,
        recorded_id: str,
        recorded_name: str | None,
    ) -> None:
        container.reload()
        actual_id = getattr(container, "id", None)
        actual_name = self._normalized_container_name(getattr(container, "name", None))
        expected_name = str(recorded_name or "")
        if not self._container_ids_equivalent(recorded_id, actual_id) or (
            actual_name != expected_name
        ):
            raise ValueError("Container identity does not match deployment record")

        hostname = os.environ.get("HOSTNAME", "").strip()
        protected_name = actual_name in PROTECTED_CONTAINER_NAMES or (
            self._normalized_container_name(recorded_name) in PROTECTED_CONTAINER_NAMES
        )
        current_manager = self._container_ids_equivalent(hostname, actual_id)
        if protected_name or current_manager:
            raise ValueError("Refusing to operate on a protected manager container")

    @staticmethod
    def _stable_container_status(status: Any, *, fallback: str) -> str:
        normalized = str(status or "").lower()
        if normalized == "running":
            return "running"
        if normalized in {"dead", "exited", "stopped"}:
            return "exited"
        if normalized:
            return "unknown"
        if fallback in {"running", "exited"}:
            return fallback
        return "unknown"

    @staticmethod
    def _deployment_identity_filters(
        deployment_id: str,
        *,
        expected_container_id: str | None,
        expected_container_name: str | None,
    ) -> tuple[Any, ...]:
        container_id_filter = (
            Deployment.container_id.is_(None)
            if expected_container_id is None
            else Deployment.container_id == expected_container_id
        )
        container_name_filter = (
            Deployment.container_name.is_(None)
            if expected_container_name is None
            else Deployment.container_name == expected_container_name
        )
        return (
            Deployment.id == deployment_id,
            container_id_filter,
            container_name_filter,
        )

    def _recover_action_failure(
        self,
        deployment_id: str,
        *,
        container: Any | None,
        original_status: str,
        expected_container_id: str | None,
        expected_container_name: str | None,
        transient_status: str,
    ) -> bool:
        actual_status = None
        if container is not None:
            try:
                container.reload()
                actual_status = getattr(container, "status", None)
            except Exception:
                pass
        recovered_status = self._stable_container_status(
            actual_status,
            fallback=original_status,
        )
        with self.session_factory() as db:
            result = db.execute(
                update(Deployment)
                .where(
                    *self._deployment_identity_filters(
                        deployment_id,
                        expected_container_id=expected_container_id,
                        expected_container_name=expected_container_name,
                    ),
                    Deployment.status == transient_status,
                )
                .values(status=recovered_status, health="unhealthy")
            )
            if result.rowcount != 1:
                db.rollback()
                return False
            db.commit()
            return True

    def _delete_stale_deployment(
        self,
        deployment_id: str,
        *,
        expected_container_id: str | None,
        expected_container_name: str | None,
        transient_status: str,
    ) -> None:
        with self.session_factory() as db:
            result = db.execute(
                delete(Deployment).where(
                    *self._deployment_identity_filters(
                        deployment_id,
                        expected_container_id=expected_container_id,
                        expected_container_name=expected_container_name,
                    ),
                    Deployment.status == transient_status,
                )
            )
            if result.rowcount != 1:
                db.rollback()
                raise ValueError(CONCURRENT_DEPLOYMENT_CHANGE_ERROR)
            db.commit()

    def _best_effort_recover_action_failure(
        self,
        deployment_id: str,
        *,
        container: Any | None,
        original_status: str,
        expected_container_id: str | None,
        expected_container_name: str | None,
        transient_status: str,
    ) -> bool | None:
        try:
            return self._recover_action_failure(
                deployment_id,
                container=container,
                original_status=original_status,
                expected_container_id=expected_container_id,
                expected_container_name=expected_container_name,
                transient_status=transient_status,
            )
        except Exception:
            return None

    def _finalize_action(
        self,
        deployment_id: str,
        *,
        action: str,
        expected_container_id: str | None,
        expected_container_name: str | None,
        transient_status: str,
        status: str,
        health: str,
    ) -> None:
        filters = (
            *self._deployment_identity_filters(
                deployment_id,
                expected_container_id=expected_container_id,
                expected_container_name=expected_container_name,
            ),
            Deployment.status == transient_status,
        )
        statement = (
            delete(Deployment).where(*filters)
            if action == "delete"
            else update(Deployment).where(*filters).values(status=status, health=health)
        )
        with self.session_factory() as db:
            result = db.execute(statement)
            if result.rowcount != 1:
                db.rollback()
                raise ValueError(CONCURRENT_DEPLOYMENT_CHANGE_ERROR)
            db.commit()

    @staticmethod
    def _action_result(
        deployment_id: str,
        action: str,
        status: str,
        health: str,
        *,
        container_missing: bool,
    ) -> dict[str, Any]:
        return {
            "deployment_id": deployment_id,
            "action": action,
            "status": status,
            "health": health,
            "container_missing": container_missing,
        }

    def action_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        deployment_id = str(payload["deployment_id"])
        action = str(payload["action"])
        if action not in {"start", "stop", "restart", "delete"}:
            raise ValueError("Unsupported deployment action")
        has_expected_id = "expected_container_id" in payload
        has_expected_name = "expected_container_name" in payload
        if not has_expected_id or not has_expected_name:
            raise ValueError(MISSING_DEPLOYMENT_SNAPSHOT_ERROR)
        with self.session_factory() as db:
            deployment = db.get(Deployment, deployment_id)
            if not deployment:
                raise ValueError("Deployment was not found")
            expected_container_id = payload["expected_container_id"]
            expected_container_name = payload["expected_container_name"]
            if (
                deployment.container_id != expected_container_id
                or deployment.container_name != expected_container_name
            ):
                raise ValueError(CONCURRENT_DEPLOYMENT_CHANGE_ERROR)
            container_id = expected_container_id
            container_name = expected_container_name
            endpoint_url = deployment.endpoint_url
            runtime = deployment.runtime
            original_status = deployment.status
            if action != "delete" and not container_id:
                raise ValueError("Deployment or container was not found")
            transient_status = {
                "start": "starting",
                "restart": "starting",
                "stop": "stopping",
                "delete": "deleting",
            }[action]
            result = db.execute(
                update(Deployment)
                .where(
                    *self._deployment_identity_filters(
                        deployment_id,
                        expected_container_id=expected_container_id,
                        expected_container_name=expected_container_name,
                    ),
                    Deployment.status == original_status,
                )
                .values(status=transient_status, health="unknown")
            )
            if result.rowcount != 1:
                db.rollback()
                raise ValueError(CONCURRENT_DEPLOYMENT_CHANGE_ERROR)
            db.commit()

        if action == "delete" and not container_id:
            self._delete_stale_deployment(
                deployment_id,
                expected_container_id=expected_container_id,
                expected_container_name=expected_container_name,
                transient_status=transient_status,
            )
            return self._action_result(
                deployment_id,
                action,
                "deleted",
                "unknown",
                container_missing=True,
            )

        container = None
        try:
            container = self.docker_client().containers.get(container_id)
            self._validate_action_container(
                container,
                recorded_id=container_id,
                recorded_name=container_name,
            )
            adapter = self.adapter(runtime)
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
        except NotFound:
            if action == "delete":
                self._delete_stale_deployment(
                    deployment_id,
                    expected_container_id=expected_container_id,
                    expected_container_name=expected_container_name,
                    transient_status=transient_status,
                )
                return self._action_result(
                    deployment_id,
                    action,
                    "deleted",
                    "unknown",
                    container_missing=True,
                )
            recovered = self._best_effort_recover_action_failure(
                deployment_id,
                container=container,
                original_status=original_status,
                expected_container_id=expected_container_id,
                expected_container_name=expected_container_name,
                transient_status=transient_status,
            )
            if recovered is False:
                try:
                    context.update(message=CONCURRENT_DEPLOYMENT_CHANGE_ERROR)
                except Exception:
                    pass
            raise
        except Exception:
            recovered = self._best_effort_recover_action_failure(
                deployment_id,
                container=container,
                original_status=original_status,
                expected_container_id=expected_container_id,
                expected_container_name=expected_container_name,
                transient_status=transient_status,
            )
            if recovered is False:
                try:
                    context.update(message=CONCURRENT_DEPLOYMENT_CHANGE_ERROR)
                except Exception:
                    pass
            raise
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
                recovered = self._recover_action_failure(
                    deployment_id,
                    container=container,
                    original_status=original_status,
                    expected_container_id=expected_container_id,
                    expected_container_name=expected_container_name,
                    transient_status=transient_status,
                )
                if not recovered:
                    raise ValueError(CONCURRENT_DEPLOYMENT_CHANGE_ERROR)
                context.update(message=f"Runtime health check failed:\n{logs}")
                raise RuntimeError(
                    f"Runtime did not become healthy within "
                    f"{self.startup_timeout_seconds} seconds. Last logs:\n{logs}"
                )
            health = "healthy"
        self._finalize_action(
            deployment_id,
            action=action,
            expected_container_id=expected_container_id,
            expected_container_name=expected_container_name,
            transient_status=transient_status,
            status=new_status,
            health=health,
        )
        return self._action_result(
            deployment_id,
            action,
            new_status,
            health,
            container_missing=False,
        )

    def logs(self, deployment: Deployment, tail: int = 500) -> str:
        if not deployment.container_id:
            raise ValueError("Deployment has no container")
        container = self.docker_client().containers.get(deployment.container_id)
        return redact_log(self.adapter(deployment.runtime).logs(container, tail=tail))

