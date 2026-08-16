from __future__ import annotations

import json
import os
import shutil
import stat
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
HF_TARGET_ID_KEYS = {"repository_id", "repo_id", "id", "target"}
HF_TARGET_LIST_KEYS = {
    "targets",
    "repositories",
    "repository_ids",
    "repo_ids",
    "ids",
}


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

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(reparse_flag and attributes & reparse_flag)

    def validate_local_target(self, target: Path) -> Path:
        try:
            lexical_target = Path(os.path.abspath(target))
            if not lexical_target.exists() or not lexical_target.is_dir():
                raise ValueError("Local model target must be an existing directory")

            for configured_root in self.model_roots:
                lexical_root = Path(os.path.abspath(configured_root))
                if lexical_target == lexical_root or not lexical_target.is_relative_to(
                    lexical_root
                ):
                    continue

                relative_target = lexical_target.relative_to(lexical_root)
                components = [lexical_root]
                current = lexical_root
                for part in relative_target.parts:
                    current /= part
                    components.append(current)
                if any(self._is_link_or_reparse(component) for component in components):
                    raise ValueError(
                        "Local model path cannot contain a symbolic link or reparse point"
                    )

                resolved_root = lexical_root.resolve(strict=True)
                resolved_target = lexical_target.resolve(strict=True)
                if (
                    resolved_target != resolved_root
                    and resolved_target.is_relative_to(resolved_root)
                ):
                    return resolved_target
        except ValueError:
            raise
        except (OSError, RuntimeError) as exc:
            raise ValueError("Unable to resolve the local model target") from exc

        raise ValueError("Local model target must be inside a configured model root")

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

    def _run_hf_json(self, argv: list[str]) -> dict[str, Any]:
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
            payload = json.loads(stdout)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise RuntimeError("Hugging Face cache command returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Hugging Face cache command returned invalid JSON")
        return payload

    @staticmethod
    def _is_non_negative_int(value: Any) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        )

    @staticmethod
    def _normalize_hf_target(value: Any) -> str:
        if not isinstance(value, str):
            raise RuntimeError("Hugging Face cache preview target was invalid")
        candidate = value.strip()
        repository_id = (
            candidate.removeprefix("model/")
            if candidate.startswith("model/")
            else candidate
        )
        try:
            validated = validate_repository_id(repository_id)
        except ValueError as exc:
            raise RuntimeError("Hugging Face cache preview target was invalid") from exc
        return f"model/{validated}"

    @classmethod
    def _collect_hf_targets(cls, value: Any) -> tuple[bool, set[str]]:
        present = False
        targets: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key in HF_TARGET_ID_KEYS:
                    present = True
                    targets.add(cls._normalize_hf_target(item))
                elif key in HF_TARGET_LIST_KEYS:
                    present = True
                    if not isinstance(item, list):
                        raise RuntimeError("Hugging Face cache preview target was invalid")
                    for target in item:
                        if isinstance(target, dict):
                            nested_present, nested_targets = cls._collect_hf_targets(
                                target
                            )
                            if not nested_present:
                                raise RuntimeError(
                                    "Hugging Face cache preview target was invalid"
                                )
                            targets.update(nested_targets)
                        else:
                            targets.add(cls._normalize_hf_target(target))
                else:
                    nested_present, nested_targets = cls._collect_hf_targets(item)
                    present = present or nested_present
                    targets.update(nested_targets)
        elif isinstance(value, list):
            for item in value:
                nested_present, nested_targets = cls._collect_hf_targets(item)
                present = present or nested_present
                targets.update(nested_targets)
        return present, targets

    @classmethod
    def _validate_hf_preview(
        cls, payload: dict[str, Any], expected_target: str
    ) -> None:
        repos = payload.get("repos")
        if (
            payload.get("dry_run") is not True
            or not isinstance(repos, int)
            or isinstance(repos, bool)
            or repos != 1
            or (
                "revisions" in payload
                and not cls._is_non_negative_int(payload["revisions"])
            )
        ):
            raise RuntimeError("Hugging Face cache preview schema was invalid")

        normalized_expected = cls._normalize_hf_target(expected_target)
        targets_present, targets = cls._collect_hf_targets(payload)
        if targets_present and targets != {normalized_expected}:
            raise RuntimeError(
                "Hugging Face cache preview did not identify only the target"
            )

    @classmethod
    def _validate_hf_result(cls, payload: dict[str, Any]) -> None:
        repos_deleted = payload.get("repos_deleted")
        if (
            not isinstance(repos_deleted, int)
            or isinstance(repos_deleted, bool)
            or repos_deleted != 1
            or (
                "revisions_deleted" in payload
                and not cls._is_non_negative_int(payload["revisions_deleted"])
            )
        ):
            raise RuntimeError("Hugging Face cache deletion result was invalid")

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
                    preview = self._run_hf_json(
                        [*base_argv, "--dry-run", "--json"]
                    )
                    self._validate_hf_preview(preview, f"model/{repository_id}")
                    context.check_control()
                    result = self._run_hf_json([*base_argv, "--yes", "--json"])
                    self._validate_hf_result(result)
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
                    self.validate_local_target(target)
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
