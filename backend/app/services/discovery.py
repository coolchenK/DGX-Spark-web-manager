from __future__ import annotations

import json
import os
import re
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


def resolve_hf_snapshot(repository_path: Path) -> Path | None:
    main_ref = repository_path / "refs" / "main"
    try:
        commit_hash = main_ref.read_text(encoding="utf-8").strip()
    except OSError:
        commit_hash = ""
    referenced = repository_path / "snapshots" / commit_hash
    if commit_hash and referenced.is_dir():
        return referenced

    snapshots = repository_path / "snapshots"
    if snapshots.is_dir():
        candidates: list[tuple[int, Path]] = []
        try:
            entries = snapshots.iterdir()
            for path in entries:
                try:
                    if path.is_dir():
                        candidates.append((path.stat().st_mtime_ns, path))
                except OSError:
                    continue
        except OSError:
            return None
        if candidates:
            return max(candidates, key=lambda candidate: candidate[0])[1]
    return None


def hf_snapshot_is_complete(repository_path: Path, snapshot_path: Path) -> bool:
    """Require every snapshot file link to resolve to this repository's blob store."""
    try:
        repository = Path(os.path.abspath(repository_path))
        snapshot = Path(os.path.abspath(snapshot_path))
        if repository.is_symlink() or not snapshot.is_dir() or snapshot.is_symlink():
            return False
        lexical_blob_root = repository / "blobs"
        blob_root: Path | None = None

        saw_file = False
        saw_nonempty_file = False
        for path in snapshot.rglob("*"):
            if path.is_symlink():
                if lexical_blob_root.is_symlink():
                    return False
                if blob_root is None:
                    blob_root = lexical_blob_root.resolve(strict=True)
                    if not blob_root.is_dir():
                        return False
                target = path.resolve(strict=True)
                if (
                    target.parent != blob_root
                    or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", target.name) is None
                    or not target.is_file()
                ):
                    return False
                size = target.stat().st_size
            elif path.is_dir():
                continue
            elif path.is_file():
                size = path.stat().st_size
            else:
                return False
            saw_file = True
            saw_nonempty_file = saw_nonempty_file or size > 0
        return saw_file and saw_nonempty_file
    except (OSError, RuntimeError, ValueError):
        return False


def infer_runtime(image: str, command: list[str]) -> str | None:
    haystack = " ".join([image, *command]).lower()
    if "sglang" in haystack:
        return "sglang"
    if "vllm" in haystack:
        return "vllm"
    if "llama-server" in haystack or "llama.cpp" in haystack:
        return "llama_cpp"
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
    if not bindings:
        port_bindings = (attrs.get("HostConfig") or {}).get("PortBindings") or {}
        bindings = port_bindings.get(f"{internal_port}/tcp") or []
    host_port = bindings[0].get("HostPort") if bindings else internal_port
    if not host_port or not str(host_port).isdigit():
        return None

    model_name = _argument(command, "--served-model-name", "--alias")
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
        "config": {
            "command": command,
            "model_path": model_path,
            "route_alias": labels.get("com.dgx-spark-manager.route"),
        },
    }


def directory_size(path: Path) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    stat = entry.stat()
                    identity = (stat.st_dev, stat.st_ino)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    total += stat.st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def infer_model_metadata(model_path: Path, name: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    try:
        value = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        if isinstance(value, dict):
            config = value
    except (OSError, json.JSONDecodeError):
        pass

    suffixes: set[str] = set()
    try:
        for path in model_path.rglob("*"):
            if path.is_file():
                suffixes.add(path.suffix.lower())
    except OSError:
        pass
    if ".safetensors" in suffixes:
        model_format = "safetensors"
    elif ".gguf" in suffixes:
        model_format = "gguf"
    elif ".bin" in suffixes:
        model_format = "pytorch-bin"
    else:
        model_format = None

    quantization = None
    quantization_config = config.get("quantization_config")
    if isinstance(quantization_config, dict):
        method = quantization_config.get("quant_method") or quantization_config.get("quantization")
        if method:
            quantization = str(method).lower()
    if not quantization:
        lowered_name = name.lower()
        quantization = next(
            (
                method
                for method in ("nvfp4", "fp8", "int8", "int4", "awq", "gptq")
                if method in lowered_name
            ),
            None,
        )

    parameter_match = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:$|[-_/])", name, re.IGNORECASE)
    parameter_count = f"{parameter_match.group(1)}B" if parameter_match else None
    architectures = [
        str(value).lower()
        for value in config.get("architectures", [])
        if isinstance(value, str)
    ]
    if any("causallm" in architecture for architecture in architectures):
        capabilities = ["chat", "completion"]
    elif any(
        marker in architecture
        for architecture in architectures
        for marker in ("embedding", "sequenceclassification")
    ):
        capabilities = ["embedding"]
    else:
        capabilities = []

    commit_hash = model_path.name if model_path.parent.name == "snapshots" else None
    return {
        "commit_hash": commit_hash,
        "format": model_format,
        "quantization": quantization,
        "parameter_count": parameter_count,
        "capabilities": capabilities,
    }


def merge_deployment_config(
    saved: dict[str, Any], observed: dict[str, Any], *, managed: bool
) -> dict[str, Any]:
    if not managed:
        return observed
    return {**saved, **{key: value for key, value in observed.items() if value is not None}}


def _mark_model_unavailable(
    asset: ModelAsset,
    *,
    local_path: Path | None = None,
    size_bytes: int = 0,
) -> None:
    if local_path is not None:
        asset.local_path = str(local_path)
    asset.size_bytes = size_bytes
    asset.status = "unavailable"
    asset.commit_hash = None
    asset.revision = None
    asset.format = None
    asset.quantization = None
    asset.parameter_count = None
    asset.capabilities = []
    asset.metadata_json = {}


def _lexically_within_root(local_path: str, root: Path) -> bool:
    try:
        candidate = Path(os.path.abspath(local_path))
        boundary = Path(os.path.abspath(root))
    except (OSError, TypeError, ValueError):
        return False
    return candidate.is_relative_to(boundary)


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
            # A stopped host-network container retains its old endpoint. Probing that
            # address can hit a different live server now using the same port.
            probe = (
                self._probe(candidate)
                if candidate["status"] == "running"
                else {}
            )
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
            candidate["config"] = merge_deployment_config(
                existing.config or {}, candidate["config"], managed=candidate["managed"]
            )
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
            existing.capabilities = ["chat", "completion"]
            existing.last_checked_at = utc_now()
            discovered.append(existing)
        db.commit()
        return discovered

    def scan_models(self, db: Session) -> list[ModelAsset]:
        discovered: list[ModelAsset] = []
        successful_roots: list[Path] = []
        for root in self.model_roots:
            try:
                if not root.exists() or not root.is_dir():
                    continue
                children = list(root.iterdir())
            except OSError:
                continue
            successful_roots.append(Path(os.path.abspath(root)))
            for child in children:
                try:
                    is_model_directory = child.is_dir() and not child.name.startswith(".")
                except OSError:
                    continue
                if not is_model_directory:
                    continue
                repository_id = parse_hf_cache_repository(child.name)
                source = "huggingface" if repository_id else "local"
                snapshot = resolve_hf_snapshot(child) if repository_id else None
                local_path = snapshot or child
                if repository_id:
                    existing = db.scalar(
                        select(ModelAsset).where(
                            ModelAsset.source == "huggingface",
                            ModelAsset.repository_id == repository_id,
                        )
                    )
                else:
                    existing = db.scalar(
                        select(ModelAsset).where(ModelAsset.local_path == str(local_path))
                    )
                if existing is None:
                    existing = ModelAsset(
                        name=repository_id or child.name,
                        source=source,
                        repository_id=repository_id,
                        local_path=str(local_path),
                        capabilities=[],
                    )
                    db.add(existing)
                lifecycle_status = (
                    existing.status
                    if existing.status in {"deleting", "delete_failed"}
                    else None
                )
                lifecycle_metadata = (
                    dict(existing.metadata_json or {})
                    if lifecycle_status is not None
                    else None
                )
                if repository_id and (
                    snapshot is None or not hf_snapshot_is_complete(child, snapshot)
                ):
                    _mark_model_unavailable(
                        existing,
                        local_path=child,
                        size_bytes=directory_size(child),
                    )
                    if lifecycle_status is not None:
                        existing.status = lifecycle_status
                        existing.metadata_json = lifecycle_metadata
                    discovered.append(existing)
                    continue
                metadata = infer_model_metadata(local_path, repository_id or child.name)
                existing.local_path = str(local_path)
                existing.size_bytes = directory_size(child)
                if lifecycle_status is None:
                    existing.status = "available"
                existing.commit_hash = metadata["commit_hash"]
                revision = None
                if repository_id and metadata["commit_hash"]:
                    try:
                        main_commit = (child / "refs" / "main").read_text(encoding="utf-8").strip()
                    except OSError:
                        main_commit = ""
                    revision = (
                        "main"
                        if main_commit == metadata["commit_hash"]
                        else metadata["commit_hash"]
                    )
                existing.revision = revision
                existing.format = metadata["format"]
                existing.quantization = metadata["quantization"]
                existing.parameter_count = metadata["parameter_count"]
                existing.capabilities = metadata["capabilities"]
                if lifecycle_status is not None:
                    existing.status = lifecycle_status
                    existing.metadata_json = lifecycle_metadata
                discovered.append(existing)
        db.flush()
        discovered_ids = {asset.id for asset in discovered}
        if successful_roots:
            available_assets = db.scalars(
                select(ModelAsset).where(ModelAsset.status == "available")
            ).all()
            for asset in available_assets:
                if asset.id in discovered_ids:
                    continue
                if any(
                    _lexically_within_root(asset.local_path, root)
                    for root in successful_roots
                ):
                    _mark_model_unavailable(asset)
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
