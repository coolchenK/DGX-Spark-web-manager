from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
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
TPS_BENCHMARK_PROMPT = (
    "不要解释，不要换行，不要输出思考过程。严格输出从 1 到 200 的阿拉伯数字，"
    "使用英文逗号分隔。"
)
TPS_BENCHMARK_MAX_TOKENS = 256


def run_deployment_tps_benchmark(
    endpoint: str,
    api_model_name: str,
    runtime: str,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": api_model_name,
        "messages": [{"role": "user", "content": TPS_BENCHMARK_PROMPT}],
        "temperature": 0,
        "max_tokens": TPS_BENCHMARK_MAX_TOKENS,
        "stream": False,
    }
    if runtime == "vllm":
        request["chat_template_kwargs"] = {"enable_thinking": False}
    warmup = {
        **request,
        "messages": [{"role": "user", "content": "只回答 OK"}],
        "max_tokens": 16,
    }
    timeout = httpx.Timeout(connect=5, read=360, write=30, pool=5)
    with httpx.Client(base_url=endpoint, timeout=timeout, trust_env=False) as client:
        warmup_response = client.post("/v1/chat/completions", json=warmup)
        warmup_response.raise_for_status()
        started_at = time.perf_counter()
        response = client.post("/v1/chat/completions", json=request)
        duration_seconds = time.perf_counter() - started_at
        response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") if isinstance(payload, dict) else None
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        raise RuntimeError("TPS benchmark response did not include completion_tokens")
    if duration_seconds <= 0:
        raise RuntimeError("TPS benchmark duration is invalid")
    choice = (payload.get("choices") or [{}])[0]
    return {
        "status": "succeeded",
        "tps": round(completion_tokens / duration_seconds, 3),
        "completion_tokens": completion_tokens,
        "duration_seconds": round(duration_seconds, 3),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "max_tokens": TPS_BENCHMARK_MAX_TOKENS,
    }


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


# Spec fields introduced after containers started carrying a spec fingerprint
# label. The label is baked into the container at creation and can never be
# rewritten, so hashing a newly added key would change the fingerprint of every
# spec that predates it and make each existing container fail the label check
# on the next in-place update. Omitting these while they are unset keeps an
# additive schema change invisible to deployments that do not use it; setting
# one changes the fingerprint, which is correct because the launch command
# changes too. Add to this tuple whenever DeploymentSpec gains an optional
# field.
FINGERPRINT_FIELDS_OMITTED_WHEN_UNSET = ("chat_template_kwargs",)


def _deployment_spec_fingerprint(
    spec: DeploymentSpec,
    *,
    include_model_path: bool,
    omit_unset_optional: bool = True,
) -> str:
    public = (
        spec.public_dump()
        if isinstance(spec, ResolvedDeploymentSpec)
        else spec.model_dump(mode="json")
    )
    if not include_model_path:
        public.pop("model_path", None)
    if omit_unset_optional:
        for field in FINGERPRINT_FIELDS_OMITTED_WHEN_UNSET:
            if public.get(field) is None:
                public.pop(field, None)
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
        benchmark_runner: Callable[[str, str, str], dict[str, Any]] | None = None,
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
        self.benchmark_runner = benchmark_runner

    _port_allocation_lock = threading.Lock()

    def docker_client(self):
        return self._docker_client or docker.from_env()

    def _docker_reserved_ports(self, *, excluded_container_ids: set[str] | None = None) -> set[int]:
        reserved: set[int] = set()
        excluded = excluded_container_ids or set()
        try:
            containers = self.docker_client().containers.list(all=True)
        except Exception:
            containers = []
        for container in containers:
            try:
                if str(getattr(container, "id", "")) in excluded:
                    continue
                bindings = container.attrs.get("HostConfig", {}).get("PortBindings") or {}
                for values in bindings.values():
                    for value in values or []:
                        host_port = value.get("HostPort")
                        if host_port:
                            reserved.add(int(host_port))
            except (TypeError, ValueError, AttributeError):
                continue
        return reserved

    def _db_reserved_ports(
        self, db: Session, *, exclude_deployment_id: str | None = None
    ) -> set[int]:
        conditions = [Deployment.port.is_not(None)]
        if exclude_deployment_id:
            conditions.append(Deployment.id != exclude_deployment_id)
        reserved = {
            int(port)
            for (port,) in db.execute(select(Deployment.port).where(*conditions)).all()
            if port is not None
        }
        return reserved

    def _deployment_reserved_ports(
        self, db: Session, *, exclude_deployment_id: str | None = None
    ) -> set[int]:
        reserved = self._db_reserved_ports(db, exclude_deployment_id=exclude_deployment_id)
        excluded_container_ids: set[str] = set()
        if exclude_deployment_id:
            deployment = db.get(Deployment, exclude_deployment_id)
            if deployment and deployment.container_id:
                excluded_container_ids.add(deployment.container_id)
        reserved.update(self._docker_reserved_ports(excluded_container_ids=excluded_container_ids))
        return reserved

    def _allocate_deployment_port(
        self, db: Session, *, exclude_deployment_id: str | None = None
    ) -> int:
        """Return the lowest free host port, starting at 8000.

        Deployment rows reserve ports even while stopped. Deleting a deployment
        releases its port, so the next creation naturally reuses the lowest gap.
        Live Docker bindings are also checked to protect against DB drift.
        """
        reserved = self._deployment_reserved_ports(db, exclude_deployment_id=exclude_deployment_id)
        port = 8000
        while port in reserved:
            port += 1
        return port

    def _prepare_spec_port(
        self,
        db: Session,
        spec: DeploymentSpec,
        *,
        exclude_deployment_id: str | None = None,
        include_docker: bool = True,
    ) -> DeploymentSpec:
        if spec.port is None:
            reserved = (
                self._deployment_reserved_ports(db, exclude_deployment_id=exclude_deployment_id)
                if include_docker
                else self._db_reserved_ports(db, exclude_deployment_id=exclude_deployment_id)
            )
            port = 8000
            while port in reserved:
                port += 1
        else:
            # The full Docker check runs after resource preflight. This DB-only check
            # catches duplicate reservations without probing the daemon first.
            reserved = self._db_reserved_ports(db, exclude_deployment_id=exclude_deployment_id)
            if spec.port in reserved:
                raise ValueError(f"Host port {spec.port} is already in use")
            port = spec.port
        return spec.model_copy(update={"port": port})

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
                return (
                    resolved_root,
                    self.host_model_roots[index],
                    resolved.relative_to(resolved_root),
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
            raise ValueError(f"Adapter {adapter.runtime} cannot deploy runtime {spec.runtime}")
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
            method = spec.speculative.method
            if method not in capabilities.speculative_methods:
                raise ValueError("Speculative method is unsupported by the runtime")
            runtime_method = capabilities.method_mapping.get(method)
            if not runtime_method:
                raise ValueError("Speculative method mapping is unavailable")
            if method == "mtp" and spec.speculative.draft_model_id is None:
                if spec.runtime != "vllm":
                    raise ValueError("Embedded MTP is only supported by vLLM")
                try:
                    evidence = self.evidence_loader.load(base_path)
                except Exception as exc:
                    raise ValueError("Embedded MTP evidence could not be verified") from exc
                if not evidence.embedded_mtp_available:
                    raise ValueError("Base model does not contain an embedded MTP head")
                internal["speculative_runtime_method"] = runtime_method
            else:
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
        auto_port = spec.port is None
        effective_spec = self._prepare_spec_port(
            db,
            spec,
            exclude_deployment_id=exclude_deployment_id,
            include_docker=not auto_port,
        )
        _, preview = self._preflight(
            db,
            effective_spec,
            exclude_deployment_id=exclude_deployment_id,
            auto_port=auto_port,
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
            stored_spec = stored_spec_value if isinstance(stored_spec_value, Mapping) else {}
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
        if (
            resolved.speculative is not None
            and resolved.speculative.draft_model_id is not None
        ):
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
                "reuses_base_mount": resolved.draft_model_root == resolved.base_model_root,
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
            if accepted_fingerprints is not None and key == f"{LABEL_PREFIX}spec-fingerprint":
                if labels.get(key) not in accepted_fingerprints:
                    raise ValueError("Existing container labels do not match this task and spec")
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
        # A container's label is written once at creation and is immutable,
        # so it holds whichever form was canonical back then. Accept the whole
        # matrix -- with and without model_path, with and without the fields
        # added later -- otherwise containers created on either side of a
        # schema change fail the label check on their next in-place update.
        return {
            _deployment_spec_fingerprint(
                spec, include_model_path=include_model_path, omit_unset_optional=omit
            )
            for include_model_path in (False, True)
            for omit in (False, True)
        }

    def _adapter_for_spec(self, spec: DeploymentSpec) -> RuntimeAdapter:
        adapter = self.adapter(spec.runtime)
        if spec.runtime != adapter.runtime:
            raise ValueError(f"Adapter {adapter.runtime} cannot deploy runtime {spec.runtime}")
        if spec.image not in adapter.allowed_images:
            raise ValueError(f"Image is not allowed for {spec.runtime}")
        return adapter

    def _deployment_matches_spec(self, deployment: Deployment, spec: DeploymentSpec) -> bool:
        return (
            self._stored_spec_fingerprint(deployment) == deployment_spec_fingerprint(spec)
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
        backup = self._get_container_optional(client, self._backup_container_name(deployment_id))
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
            context.update(message="backup cleanup conflict: ownership could not be verified")
            return
        self._remove_owned_container(backup)

    def _recover_committed_deployment(
        self,
        context: TaskContext,
        deployment: Deployment,
        spec: DeploymentSpec,
        *,
        cleanup_backup: bool,
        benchmark_if_missing: bool = False,
        force_benchmark: bool = False,
    ) -> dict[str, Any]:
        deployment_id = deployment.id
        container_id = deployment.container_id
        needs_benchmark = force_benchmark or (
            benchmark_if_missing and deployment.benchmark_status != "succeeded"
        )
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
                self._coordinate_failed_recovery_start(adapter, target, deployment_id, container_id)
            raise
        if cleanup_backup:
            self._cleanup_committed_backup(context, client, target, deployment_id)
        result = {
            "deployment_id": deployment_id,
            "container_name": target.name,
            "endpoint_url": endpoint,
            "idempotent": True,
        }
        if needs_benchmark:
            benchmark = self._benchmark_deployment(
                context,
                deployment_id=deployment_id,
                endpoint=endpoint,
                api_model_name=spec.api_model_name,
                runtime=spec.runtime,
            )
            if benchmark is not None:
                result["benchmark"] = benchmark
        return result

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
                    select(Deployment).where(Deployment.container_id == container_id)
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
        auto_port: bool = False,
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
        if (
            resolved.speculative is not None
            and resolved.speculative.draft_model_id is not None
        ):
            draft = db.get(ModelAsset, resolved.speculative.draft_model_id)
            candidates = self.draft_service.list_candidates(db, target, capabilities, snapshot)
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

        if resolved.port in self._deployment_reserved_ports(
            db, exclude_deployment_id=exclude_deployment_id
        ):
            if not auto_port:
                raise ValueError(f"Host port {resolved.port} is already in use")
            resolved = resolved.model_copy(
                update={
                    "port": self._allocate_deployment_port(
                        db, exclude_deployment_id=exclude_deployment_id
                    )
                }
            )

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
        volumes = {spec.base_host_model_root: {"bind": "/models", "mode": "ro"}}
        if (
            spec.speculative is not None
            and spec.draft_model_root is not None
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

    def _benchmark_deployment(
        self,
        context: TaskContext,
        *,
        deployment_id: str,
        endpoint: str,
        api_model_name: str,
        runtime: str,
    ) -> dict[str, Any] | None:
        if self.benchmark_runner is None:
            return None
        with self.session_factory() as db:
            deployment = db.get(Deployment, deployment_id)
            if deployment is None:
                return None
            deployment.benchmark_status = "running"
            deployment.benchmark_tps = None
            deployment.benchmark_completion_tokens = None
            deployment.benchmark_duration_seconds = None
            deployment.benchmark_tested_at = None
            deployment.benchmark_error = None
            db.commit()
        context.update(progress=94, message="Deployment healthy; running TPS benchmark")
        tested_at = datetime.now(UTC)
        try:
            raw = self.benchmark_runner(endpoint, api_model_name, runtime)
            tps = float(raw["tps"])
            completion_tokens = int(raw["completion_tokens"])
            duration_seconds = float(raw["duration_seconds"])
            if tps <= 0 or completion_tokens <= 0 or duration_seconds <= 0:
                raise ValueError("TPS benchmark returned invalid measurements")
            benchmark = {
                **raw,
                "status": "succeeded",
                "tps": round(tps, 3),
                "completion_tokens": completion_tokens,
                "duration_seconds": round(duration_seconds, 3),
                "tested_at": tested_at.isoformat(),
            }
            with self.session_factory() as db:
                deployment = db.get(Deployment, deployment_id)
                if deployment is not None:
                    deployment.benchmark_status = "succeeded"
                    deployment.benchmark_tps = benchmark["tps"]
                    deployment.benchmark_completion_tokens = completion_tokens
                    deployment.benchmark_duration_seconds = benchmark["duration_seconds"]
                    deployment.benchmark_tested_at = tested_at
                    deployment.benchmark_error = None
                    if deployment.model_id is not None:
                        model = db.get(ModelAsset, deployment.model_id)
                        if model is not None:
                            model.benchmark_tps = benchmark["tps"]
                            model.benchmark_tested_at = tested_at
                    db.commit()
            context.update(
                progress=99,
                message=f"TPS benchmark completed at {benchmark['tps']:.3f} tok/s",
            )
            return benchmark
        except Exception as exc:
            error = redact_log(str(exc)).strip()[:2000] or "TPS benchmark failed"
            with self.session_factory() as db:
                deployment = db.get(Deployment, deployment_id)
                if deployment is not None:
                    deployment.benchmark_status = "failed"
                    deployment.benchmark_tps = None
                    deployment.benchmark_completion_tokens = None
                    deployment.benchmark_duration_seconds = None
                    deployment.benchmark_tested_at = tested_at
                    deployment.benchmark_error = error
                    db.commit()
            context.update(progress=99, message=f"TPS benchmark failed: {error}")
            return {
                "status": "failed",
                "error": error,
                "tested_at": tested_at.isoformat(),
            }

    def create_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        with self._port_allocation_lock:
            return self._create_handler_locked(context, payload)

    def _create_handler_locked(
        self, context: TaskContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        committed = None
        with self.session_factory() as db:
            spec = DeploymentSpec.model_validate(payload)
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
                if existing.name != spec.name or existing.api_model_name != spec.api_model_name:
                    raise ValueError("Deployment identity conflicts with an existing deployment")
                if spec.port is None and existing.port is not None:
                    spec = spec.model_copy(update={"port": existing.port})
                if not self._deployment_matches_spec(existing, spec):
                    raise ValueError("Existing deployment uses a different deployment spec")
                committed = existing
            else:
                auto_port = spec.port is None
                spec = self._prepare_spec_port(db, spec, include_docker=not auto_port)
                resolved, preview = self._preflight(db, spec, auto_port=auto_port)
                fingerprint = preview["spec_fingerprint"]
        if committed is not None:
            return self._recover_committed_deployment(
                context,
                committed,
                spec,
                cleanup_backup=False,
                benchmark_if_missing=True,
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
                    benchmark_status="pending" if self.benchmark_runner is not None else None,
                    config=preview,
                    capabilities=adapter.openai_capabilities(),
                )
                db.add(deployment)
                db.commit()
                persisted = True
                db.refresh(deployment)
                deployment_id = deployment.id
            result = {
                "deployment_id": deployment_id,
                "container_name": name,
                "endpoint_url": endpoint,
            }
            benchmark = self._benchmark_deployment(
                context,
                deployment_id=deployment_id,
                endpoint=endpoint,
                api_model_name=resolved.api_model_name,
                runtime=resolved.runtime,
            )
            if benchmark is not None:
                result["benchmark"] = benchmark
            return result
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
            auto_port = spec.port is None
            spec = self._prepare_spec_port(
                db,
                spec,
                exclude_deployment_id=deployment_id,
                include_docker=not auto_port,
            )
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
                    auto_port=auto_port,
                )
                current_container_id = deployment.container_id
                current_container_name = deployment.container_name or deterministic_container_name(
                    deployment.name
                )
                current_status = deployment.status

        if committed is not None:
            return self._recover_committed_deployment(
                context,
                committed,
                spec,
                cleanup_backup=True,
                force_benchmark=True,
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
            result = {
                "deployment_id": deployment_id,
                "container_name": new_name,
                "endpoint_url": endpoint,
            }
            benchmark = self._benchmark_deployment(
                context,
                deployment_id=deployment_id,
                endpoint=endpoint,
                api_model_name=resolved.api_model_name,
                runtime=resolved.runtime,
            )
            if benchmark is not None:
                result["benchmark"] = benchmark
            return result
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

    @staticmethod
    def _launch_contract_snapshot(contract: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "image",
            "entrypoint",
            "command",
            "environment",
            "mounts",
            "network_mode",
            "ipc_mode",
            "restart_policy",
            "device_requests",
            "port",
        }
        missing = sorted(required - set(contract))
        if missing:
            raise ValueError(f"launch_contract is missing fields: {', '.join(missing)}")
        port = contract["port"]
        if not isinstance(port, Mapping) or not {
            "container_port",
            "host_port",
            "protocol",
        }.issubset(port):
            raise ValueError("launch_contract.port is invalid")
        try:
            snapshot = json.loads(json.dumps(contract))
        except (TypeError, ValueError) as exc:
            raise ValueError("launch_contract must contain JSON-compatible values") from exc
        if not isinstance(snapshot, dict):
            raise ValueError("launch_contract must be an object")
        return snapshot

    @staticmethod
    def _container_matches_launch_contract(container: Any, contract: Mapping[str, Any]) -> bool:
        attrs = getattr(container, "attrs", {}) or {}
        config = attrs.get("Config", {}) or {}
        host = attrs.get("HostConfig", {}) or {}
        actual_env: dict[str, str] = {}
        for item in config.get("Env") or []:
            key, separator, value = str(item).partition("=")
            actual_env[key] = value if separator else ""
        actual_mounts = sorted(
            (
                str(item.get("Type") or ""),
                str(item.get("Source") or ""),
                str(item.get("Destination") or ""),
                not bool(item.get("RW", False)),
            )
            for item in attrs.get("Mounts") or []
        )
        expected_mounts = sorted(
            (
                str(item.get("type") or "bind"),
                str(item.get("source") or ""),
                str(item.get("target") or ""),
                bool(item.get("read_only", False)),
            )
            for item in contract["mounts"]
        )
        port = contract["port"]
        binding_key = f"{int(port['container_port'])}/{port['protocol']}"
        bindings = (host.get("PortBindings") or {}).get(binding_key) or []
        actual_binding = sorted(
            (str(item.get("HostIp") or ""), str(item.get("HostPort") or "")) for item in bindings
        )
        expected_binding = [(str(port.get("host_ip") or ""), str(int(port["host_port"])))]
        expected_env = {str(key): str(value) for key, value in contract["environment"].items()}
        expected_labels = {
            str(key): str(value) for key, value in contract.get("labels", {}).items()
        }
        actual_labels = {
            str(key): str(value) for key, value in (config.get("Labels") or {}).items()
        }

        def normalize_device_requests(items: Any) -> list[dict[str, Any]]:
            return [
                {
                    "Driver": item.get("Driver") or "",
                    "Count": int(item.get("Count", 0)),
                    "DeviceIDs": item.get("DeviceIDs") or [],
                    "Capabilities": item.get("Capabilities") or [],
                    "Options": item.get("Options") or {},
                }
                for item in items or []
            ]

        return all(
            (
                config.get("Image") == contract["image"],
                (config.get("Entrypoint") or []) == (contract["entrypoint"] or []),
                (config.get("Cmd") or []) == (contract["command"] or []),
                all(actual_env.get(key) == value for key, value in expected_env.items()),
                all(actual_labels.get(key) == value for key, value in expected_labels.items()),
                actual_mounts == expected_mounts,
                host.get("NetworkMode") == contract["network_mode"],
                host.get("IpcMode") == contract["ipc_mode"],
                (host.get("RestartPolicy") or {}) == contract["restart_policy"],
                normalize_device_requests(host.get("DeviceRequests"))
                == normalize_device_requests(contract["device_requests"]),
                actual_binding == expected_binding,
            )
        )

    @staticmethod
    def _run_launch_contract_container(client: Any, contract: Mapping[str, Any], name: str) -> Any:
        volumes = {
            item["source"]: {
                "bind": item["target"],
                "mode": "ro" if item.get("read_only", False) else "rw",
            }
            for item in contract["mounts"]
        }
        port = contract["port"]
        device_requests = [
            DeviceRequest(
                driver=item.get("Driver", ""),
                count=item.get("Count", 0),
                device_ids=item.get("DeviceIDs") or [],
                capabilities=item.get("Capabilities") or [],
                options=item.get("Options") or {},
            )
            for item in contract["device_requests"]
        ]
        return client.containers.run(
            contract["image"],
            entrypoint=contract["entrypoint"],
            command=contract["command"],
            environment=contract["environment"],
            volumes=volumes,
            network_mode=contract["network_mode"],
            ipc_mode=contract["ipc_mode"],
            restart_policy=contract["restart_policy"],
            device_requests=device_requests,
            ports={
                f"{int(port['container_port'])}/{port['protocol']}": (
                    str(port.get("host_ip") or ""),
                    int(port["host_port"]),
                )
            },
            labels={str(key): str(value) for key, value in contract.get("labels", {}).items()},
            name=name,
            detach=True,
        )

    def _rebuild_action_container_from_contract(
        self,
        context: TaskContext,
        *,
        deployment_id: str,
        action: str,
        adapter: RuntimeAdapter,
        contract: Mapping[str, Any],
        old_container: Any | None,
        expected_container_id: str,
        expected_container_name: str,
        endpoint_url: str,
        original_status: str,
        transient_status: str,
    ) -> dict[str, Any]:
        client = self.docker_client()
        backup_name = self._backup_container_name(deployment_id)
        replacement = None
        old_was_running = old_container is not None and old_container.status == "running"
        container_missing = old_container is None
        try:
            if old_container is not None:
                if old_was_running:
                    adapter.stop(old_container, timeout=30)
                old_container.rename(backup_name)
            context.update(
                progress=20,
                message=f"Rebuilding drifted container {expected_container_name}",
            )
            replacement = self._run_launch_contract_container(
                client, contract, expected_container_name
            )
            if not self.wait_for_health(
                context,
                endpoint_url,
                adapter=adapter,
                progress_start=30,
                progress_end=95,
            ):
                logs = self._startup_logs(adapter, replacement)
                raise RuntimeError(
                    f"Runtime did not become healthy within {self.startup_timeout_seconds} "
                    f"seconds. Last logs:\n{logs}"
                )
            replacement.reload()
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
                    .values(
                        container_id=replacement.id,
                        container_name=expected_container_name,
                        status="running",
                        health="healthy",
                        managed=True,
                    )
                )
                if result.rowcount != 1:
                    db.rollback()
                    raise ValueError(CONCURRENT_DEPLOYMENT_CHANGE_ERROR)
                db.commit()
            if old_container is not None:
                self._remove_owned_container(old_container)
            return self._action_result(
                deployment_id,
                action,
                "running",
                "healthy",
                container_missing=container_missing,
            )
        except BaseException:
            if replacement is not None:
                self._remove_owned_container(replacement)
            if old_container is not None:
                try:
                    if old_container.name == backup_name:
                        old_container.rename(expected_container_name)
                    if old_was_running and old_container.status != "running":
                        adapter.start(old_container)
                except BaseException:
                    pass
            self._best_effort_recover_action_failure(
                deployment_id,
                container=old_container,
                original_status=original_status,
                expected_container_id=expected_container_id,
                expected_container_name=expected_container_name,
                transient_status=transient_status,
            )
            raise

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
            launch_contract = None
            if action in {"start", "restart"} and isinstance(deployment.config, Mapping):
                raw_contract = deployment.config.get("launch_contract")
                if raw_contract is not None:
                    if not isinstance(raw_contract, Mapping):
                        raise ValueError("launch_contract must be an object")
                    launch_contract = self._launch_contract_snapshot(raw_contract)
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
        adapter = None
        try:
            if action in {"start", "restart"}:
                adapter = self.adapter(runtime)
            container = self.docker_client().containers.get(container_id)
            self._validate_action_container(
                container,
                recorded_id=container_id,
                recorded_name=container_name,
            )
            if launch_contract is not None and not self._container_matches_launch_contract(
                container, launch_contract
            ):
                return self._rebuild_action_container_from_contract(
                    context,
                    deployment_id=deployment_id,
                    action=action,
                    adapter=adapter,
                    contract=launch_contract,
                    old_container=container,
                    expected_container_id=expected_container_id,
                    expected_container_name=expected_container_name,
                    endpoint_url=endpoint_url,
                    original_status=original_status,
                    transient_status=transient_status,
                )
            adapter = adapter or self.adapter(runtime)
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
            if launch_contract is not None and action in {"start", "restart"}:
                return self._rebuild_action_container_from_contract(
                    context,
                    deployment_id=deployment_id,
                    action=action,
                    adapter=adapter,
                    contract=launch_contract,
                    old_container=None,
                    expected_container_id=expected_container_id,
                    expected_container_name=expected_container_name,
                    endpoint_url=endpoint_url,
                    original_status=original_status,
                    transient_status=transient_status,
                )
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
