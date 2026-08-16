from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Deployment, ModelAsset
from app.services.discovery import directory_size
from app.tasks.huggingface import cache_repository_path, validate_repository_id

COMMAND_TIMEOUT_SECONDS = 120
PREVIEW_ID_KEYS = {"repository_id", "repo_id", "id", "target"}


@dataclass(frozen=True)
class ModelReference:
    deployment_id: str
    deployment_name: str
    usage: Literal["base", "draft", "legacy_path"]


class ModelInUseError(ValueError):
    def __init__(self, references: list[ModelReference]):
        super().__init__("Model is in use")
        self.references = list(references)


def _resolved(value: str | None) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return Path(value).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


class ModelLifecycleService:
    def __init__(
        self,
        model_roots: tuple[Path, ...],
        hf_cache_dir: Path,
        session_factory: Callable[[], Session] | None = None,
        discovery_service: Any | None = None,
        command_runner: Callable[[list[str]], Any] | None = None,
    ):
        self.model_roots = model_roots
        self.hf_cache_dir = hf_cache_dir
        self.session_factory = session_factory
        self.discovery_service = discovery_service
        self.command_runner = command_runner or self._default_command_runner

    @staticmethod
    def _default_command_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )

    def validate_local_target(self, target: Path) -> Path:
        try:
            if target.is_symlink():
                raise ValueError("Local model target cannot be a symbolic link")
            if not target.exists() or not target.is_dir():
                raise ValueError("Local model target must be an existing directory")
            resolved_target = target.resolve(strict=True)
            resolved_roots = [root.resolve(strict=False) for root in self.model_roots]
        except ValueError:
            raise
        except (OSError, RuntimeError) as exc:
            raise ValueError("Unable to resolve the local model target") from exc

        if not any(
            resolved_target != root and resolved_target.is_relative_to(root)
            for root in resolved_roots
        ):
            raise ValueError("Local model target must be inside a configured model root")
        return resolved_target

    def references(self, db: Session, model_id: str) -> list[ModelReference]:
        model = db.get(ModelAsset, model_id)
        if model is None:
            raise LookupError("Model was not found")

        target_path = _resolved(model.local_path)
        references: dict[str, ModelReference] = {}
        deployments = db.scalars(select(Deployment).order_by(Deployment.name)).all()

        for deployment in deployments:
            config = deployment.config if isinstance(deployment.config, dict) else {}
            spec = config.get("spec")
            spec = spec if isinstance(spec, dict) else {}
            mounts = config.get("mounts")
            mounts = mounts if isinstance(mounts, dict) else {}
            base_mount = mounts.get("base")
            base_mount = base_mount if isinstance(base_mount, dict) else {}
            draft_mount = mounts.get("draft")
            draft_mount = draft_mount if isinstance(draft_mount, dict) else {}
            speculative = config.get("speculative")
            speculative = speculative if isinstance(speculative, dict) else {}

            usage: Literal["base", "draft", "legacy_path"] | None = None
            if deployment.model_id == model_id:
                usage = "base"
            elif speculative.get("draft_model_id") == model_id:
                usage = "draft"
            elif target_path is not None and any(
                _resolved(path) == target_path
                for path in (
                    config.get("model_path"),
                    spec.get("model_path"),
                    base_mount.get("model_path"),
                    draft_mount.get("model_path"),
                    speculative.get("draft_model_path"),
                )
            ):
                usage = "legacy_path"

            if usage is not None:
                references[deployment.id] = ModelReference(
                    deployment_id=deployment.id,
                    deployment_name=deployment.name,
                    usage=usage,
                )

        return list(references.values())

    @staticmethod
    def _preview_identifies_repository(value: Any, repository_id: str) -> bool:
        expected = {repository_id, f"model/{repository_id}"}
        if isinstance(value, dict):
            for key, item in value.items():
                if key in PREVIEW_ID_KEYS and isinstance(item, str) and item in expected:
                    return True
                if ModelLifecycleService._preview_identifies_repository(
                    item, repository_id
                ):
                    return True
        elif isinstance(value, list):
            return any(
                ModelLifecycleService._preview_identifies_repository(
                    item, repository_id
                )
                for item in value
            )
        return False

    def _run_hf_command(self, argv: list[str], repository_id: str) -> None:
        try:
            completed = self.command_runner(argv)
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ) as exc:
            raise RuntimeError("Hugging Face cache command failed") from exc

        returncode = getattr(completed, "returncode", 0)
        if not isinstance(returncode, int) or returncode != 0:
            raise RuntimeError("Hugging Face cache command failed")
        stdout = getattr(completed, "stdout", None)
        if not isinstance(stdout, str):
            raise RuntimeError("Hugging Face cache command returned invalid JSON")
        try:
            preview = json.loads(stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("Hugging Face cache command returned invalid JSON") from exc
        if not self._preview_identifies_repository(preview, repository_id):
            raise RuntimeError("Hugging Face cache preview did not identify the target")

    @staticmethod
    def _path_is_absent(path: Path) -> bool:
        try:
            return not path.exists() and not path.is_symlink()
        except OSError:
            return False

    def _mark_delete_failed(self, model_id: str) -> None:
        if self.session_factory is None:
            return
        with self.session_factory() as db:
            model = db.get(ModelAsset, model_id)
            if model is not None:
                model.status = "delete_failed"
                db.commit()

    def _remove_database_record(self, model_id: str) -> None:
        if self.session_factory is None:
            raise RuntimeError("session_factory is required for model deletion")
        with self.session_factory() as db:
            model = db.get(ModelAsset, model_id)
            if model is not None:
                db.delete(model)
                db.commit()

    def _scan_inventory(self) -> int:
        if self.discovery_service is None:
            return 0
        if self.session_factory is None:
            raise RuntimeError("session_factory is required for model deletion")
        with self.session_factory() as db:
            return len(self.discovery_service.scan_models(db))

    def delete_handler(self, context: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if self.session_factory is None:
            raise RuntimeError("session_factory is required for model deletion")

        model_id = str(payload.get("model_id") or "")
        with self.session_factory() as db:
            model = db.get(ModelAsset, model_id)
            if model is None:
                raise ValueError("Model was not found")
            if model.status == "deleting":
                raise ValueError("Model is already being deleted")
            references = self.references(db, model_id)
            if references:
                raise ModelInUseError(references)

            source = model.source
            local_path = model.local_path
            repository_id = model.repository_id
            unavailable = model.status == "unavailable"
            model.status = "deleting"
            db.commit()

        estimated_bytes = 0
        released_bytes = 0
        try:
            if source == "huggingface":
                repository_id = validate_repository_id(repository_id or "")
                target = cache_repository_path(self.hf_cache_dir, repository_id)
                if unavailable and self._path_is_absent(target):
                    context.check_control()
                else:
                    estimated_bytes = directory_size(target)
                    usage_path = self.hf_cache_dir
                    usage_path.mkdir(parents=True, exist_ok=True)
                    free_before = shutil.disk_usage(usage_path).free
                    base_argv = [
                        "hf",
                        "cache",
                        "rm",
                        f"model/{repository_id}",
                        "--cache-dir",
                        str(self.hf_cache_dir),
                    ]
                    self._run_hf_command(
                        [*base_argv, "--dry-run", "--json"], repository_id
                    )
                    context.check_control()
                    self._run_hf_command([*base_argv, "--yes", "--json"], repository_id)
                    free_after = shutil.disk_usage(usage_path).free
                    released_bytes = max(0, free_after - free_before)
            else:
                target = Path(local_path)
                if unavailable and self._path_is_absent(target):
                    context.check_control()
                else:
                    resolved_target = self.validate_local_target(target)
                    estimated_bytes = directory_size(resolved_target)
                    free_before = shutil.disk_usage(resolved_target.parent).free
                    context.check_control()
                    try:
                        shutil.rmtree(target)
                    except OSError as exc:
                        raise RuntimeError("Local model deletion failed") from exc
                    free_after = shutil.disk_usage(resolved_target.parent).free
                    released_bytes = max(0, free_after - free_before)

            inventory_models = self._scan_inventory()
            self._remove_database_record(model_id)
        except Exception:
            self._mark_delete_failed(model_id)
            raise

        return {
            "model_id": model_id,
            "source": source,
            "released_bytes": released_bytes,
            "estimated_bytes": estimated_bytes,
            "inventory_models": inventory_models,
        }
