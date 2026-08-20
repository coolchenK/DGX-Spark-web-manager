from __future__ import annotations

import errno
import json
import os
import re
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
from app.tasks.engine import TaskCancelled, TaskPaused
from app.tasks.huggingface import cache_repository_path, validate_repository_id

COMMAND_TIMEOUT_SECONDS = 120
OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
HF_TARGET_ID_KEYS = {"repository_id", "repo_id", "id", "target"}
HF_TARGET_LIST_KEYS = {
    "targets",
    "repositories",
    "repository_ids",
    "repo_ids",
    "ids",
}
HF_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


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
        local_remover: Callable[[Path, Path], None] | None = None,
    ):
        self.model_roots = model_roots
        self.hf_cache_dir = hf_cache_dir
        self.session_factory = session_factory
        self.discovery_service = discovery_service
        self.command_runner = command_runner or self._default_command_runner
        self.local_remover = local_remover or self._secure_rmtree

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

    @staticmethod
    def _secure_rmtree(root_lexical: Path, target_lexical: Path) -> None:
        if (
            os.name != "posix"
            or not getattr(shutil.rmtree, "avoids_symlink_attacks", False)
            or not hasattr(os, "O_DIRECTORY")
            or not hasattr(os, "O_NOFOLLOW")
            or not OPEN_SUPPORTS_DIR_FD
        ):
            raise RuntimeError("Secure local deletion is unsupported on this platform")

        relative = target_lexical.relative_to(root_lexical)
        if not relative.parts:
            raise ValueError("Local model target cannot equal a configured model root")

        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        opened: list[int] = []
        try:
            opened.append(os.open(root_lexical, flags))
            for component in relative.parts[:-1]:
                opened.append(os.open(component, flags, dir_fd=opened[-1]))
            shutil.rmtree(relative.parts[-1], dir_fd=opened[-1])
        except OSError as exc:
            # The bare message here used to swallow the cause, which made an
            # unwritable model root look like a bug in the delete itself. The
            # two that actually happen are a read-only bind mount and a root
            # owned directory the manager user cannot write, so name them.
            if exc.errno == errno.EROFS:
                hint = (
                    f"{root_lexical} is mounted read-only; remount it writable "
                    "or remove the files on the host"
                )
            elif exc.errno in (errno.EACCES, errno.EPERM):
                hint = (
                    f"the manager has no write permission under {root_lexical} "
                    f"(running as uid {os.geteuid()}); fix the ownership of the "
                    "model root on the host"
                )
            else:
                hint = exc.strerror or str(exc)
            raise RuntimeError(
                f"Secure local model deletion failed: {hint}"
            ) from exc
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    def _validated_local_paths(self, target: Path) -> tuple[Path, Path, Path]:
        try:
            lexical_target = Path(os.path.abspath(target))
            lexical_roots = [
                Path(os.path.abspath(configured_root))
                for configured_root in self.model_roots
            ]
            if any(
                lexical_root == lexical_target
                or lexical_root.is_relative_to(lexical_target)
                for lexical_root in lexical_roots
            ):
                raise ValueError(
                    "Local model target must be inside a configured model root"
                )
            if not lexical_target.exists() or not lexical_target.is_dir():
                raise ValueError("Local model target must be an existing directory")

            resolved_target = lexical_target.resolve(strict=True)
            normalized_roots: list[tuple[Path, Path]] = []
            for lexical_root in lexical_roots:
                target_within_root = lexical_target.is_relative_to(lexical_root)
                inspection_error: Exception | None = None
                try:
                    invalid_root = self._is_link_or_reparse(lexical_root)
                except (OSError, RuntimeError, ValueError) as exc:
                    invalid_root = True
                    inspection_error = exc
                try:
                    resolved_root = lexical_root.resolve(strict=True)
                except (OSError, RuntimeError, ValueError) as exc:
                    if target_within_root:
                        raise ValueError(
                            "Unable to resolve the configured model root"
                        ) from exc
                    continue

                if resolved_root == resolved_target or resolved_root.is_relative_to(
                    resolved_target
                ):
                    raise ValueError(
                        "Local model target must be inside a configured model root"
                    )
                resolved_target_within_root = resolved_target.is_relative_to(
                    resolved_root
                )
                if invalid_root:
                    if not target_within_root and not resolved_target_within_root:
                        continue
                    if inspection_error is not None:
                        raise ValueError(
                            "Unable to inspect the configured model root"
                        ) from inspection_error
                    raise ValueError(
                        "Local model path cannot contain a symbolic link or reparse point"
                    )
                normalized_roots.append((lexical_root, resolved_root))

            if any(
                resolved_target == resolved_root
                for _lexical_root, resolved_root in normalized_roots
            ):
                raise ValueError(
                    "Local model target must be inside a configured model root"
                )

            candidates = [
                (lexical_root, resolved_root)
                for lexical_root, resolved_root in normalized_roots
                if lexical_target.is_relative_to(lexical_root)
                and resolved_target.is_relative_to(resolved_root)
            ]
            candidates.sort(key=lambda item: len(item[0].parts), reverse=True)
            for lexical_root, _resolved_root in candidates:

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
                return lexical_root, lexical_target, resolved_target
        except ValueError:
            raise
        except (OSError, RuntimeError) as exc:
            raise ValueError("Unable to resolve the local model target") from exc

        raise ValueError("Local model target must be inside a configured model root")

    def validate_local_target(self, target: Path) -> Path:
        return self._validated_local_paths(target)[2]

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

    @classmethod
    def _path_is_absent(cls, path: Path) -> bool:
        try:
            if path.exists() or path.is_symlink():
                return False
            is_junction = getattr(path, "is_junction", None)
            if callable(is_junction) and is_junction():
                return False
            try:
                attributes = getattr(path.lstat(), "st_file_attributes", 0)
            except FileNotFoundError:
                return True
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            return not bool(reparse_flag and attributes & reparse_flag)
        except (OSError, RuntimeError):
            return False

    @classmethod
    def _is_orphaned_hf_cache(cls, target: Path) -> bool:
        """Return true only when a cache repository contains no model data."""
        try:
            if cls._is_link_or_reparse(target) or not target.is_dir():
                return False
            entries = list(target.iterdir())
            if not entries:
                return True
            allowed_roots = {".no_exist", "refs", "snapshots"}
            if any(entry.name not in allowed_roots for entry in entries):
                return False

            blob_root = Path(os.path.abspath(target / "blobs"))

            def inspect_tree(root: Path, kind: str) -> bool:
                if cls._is_link_or_reparse(root) or not root.is_dir():
                    return False
                for path in root.iterdir():
                    if path.is_symlink():
                        if kind != "snapshots":
                            return False
                        raw_target = Path(os.readlink(path))
                        if raw_target.is_absolute():
                            return False
                        linked_blob = Path(os.path.abspath(path.parent / raw_target))
                        if (
                            linked_blob.parent != blob_root
                            or HF_OBJECT_ID_PATTERN.fullmatch(linked_blob.name) is None
                            or not cls._path_is_absent(linked_blob)
                        ):
                            return False
                        continue
                    if cls._is_link_or_reparse(path):
                        return False
                    if path.is_dir():
                        if not inspect_tree(path, kind):
                            return False
                        continue
                    if not path.is_file():
                        return False
                    if kind == ".no_exist":
                        if path.stat().st_size != 0:
                            return False
                    elif kind == "refs":
                        if (
                            path.stat().st_size > 128
                            or HF_OBJECT_ID_PATTERN.fullmatch(
                                path.read_text(encoding="ascii").strip()
                            )
                            is None
                        ):
                            return False
                    else:
                        return False
                return True

            for entry in entries:
                if not inspect_tree(entry, entry.name):
                    return False
            return True
        except (OSError, RuntimeError, UnicodeError, ValueError):
            return False

    def _mark_delete_failed(self, model_id: str) -> None:
        if self.session_factory is None:
            return
        with self.session_factory() as db:
            model = db.get(ModelAsset, model_id)
            if model is not None:
                model.status = "delete_failed"
                db.commit()

    def _restore_delete_state(self, model_id: str, task_id: str) -> None:
        if self.session_factory is None:
            return
        with self.session_factory() as db:
            model = db.get(ModelAsset, model_id)
            if model is None:
                return
            metadata = dict(model.metadata_json or {})
            if metadata.get("_delete_task_id") != task_id:
                return
            original_status = metadata.pop("_delete_original_status", "available")
            if original_status not in {"available", "unavailable"}:
                original_status = "available"
            metadata.pop("_delete_task_id", None)
            model.metadata_json = metadata
            model.status = original_status
            db.commit()

    def _remove_database_record(self, model_id: str) -> None:
        if self.session_factory is None:
            raise RuntimeError("session_factory is required for model deletion")
        with self.session_factory() as db:
            model = db.get(ModelAsset, model_id)
            if model is not None:
                db.delete(model)
                db.commit()

    def _scan_inventory(self) -> tuple[int | None, str | None]:
        if self.discovery_service is None:
            return None, None
        if self.session_factory is None:
            raise RuntimeError("session_factory is required for model deletion")
        try:
            with self.session_factory() as db:
                return len(self.discovery_service.scan_models(db)), None
        except Exception:
            return None, "Inventory refresh failed"

    @staticmethod
    def _disk_free(path: Path) -> int | None:
        try:
            return shutil.disk_usage(path).free
        except (OSError, RuntimeError):
            return None

    def _deletion_result(
        self,
        *,
        model_id: str,
        source: str,
        released_bytes: int,
        estimated_bytes: int,
        warnings: list[str],
    ) -> dict[str, Any]:
        inventory_models, inventory_warning = self._scan_inventory()
        return {
            "model_id": model_id,
            "source": source,
            "released_bytes": released_bytes,
            "estimated_bytes": estimated_bytes,
            "inventory_models": inventory_models,
            "inventory_warning": inventory_warning,
            "warnings": warnings,
        }

    def delete_handler(self, context: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if self.session_factory is None:
            raise RuntimeError("session_factory is required for model deletion")

        model_id = str(payload.get("model_id") or "")
        task_id = getattr(context, "task_id", None)
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("Task context must provide a task_id")

        with self.session_factory() as db:
            model = db.get(ModelAsset, model_id)
            if model is None:
                raise ValueError("Model was not found")
            metadata = dict(model.metadata_json or {})
            original_status = model.status
            reentry = model.status == "deleting"
            failed_retry = model.status == "delete_failed"
            if model.status == "deleting":
                if metadata.get("_delete_task_id") != task_id:
                    raise ValueError("Model deletion is owned by another task")
                original_status = metadata.get(
                    "_delete_original_status", "available"
                )
            references = self.references(db, model_id)
            if references:
                if reentry:
                    restored_status = original_status
                    if restored_status not in {"available", "unavailable"}:
                        restored_status = "available"
                    metadata.pop("_delete_task_id", None)
                    metadata.pop("_delete_original_status", None)
                    model.metadata_json = metadata
                    model.status = restored_status
                    db.commit()
                raise ModelInUseError(references)

            source = model.source
            local_path = model.local_path
            repository_id = model.repository_id
            if not reentry:
                if model.status == "delete_failed":
                    original_status = metadata.get(
                        "_delete_original_status", "available"
                    )
                if original_status not in {"available", "unavailable"}:
                    original_status = "available"
                metadata["_delete_task_id"] = task_id
                metadata["_delete_original_status"] = original_status
                model.metadata_json = metadata
            unavailable = original_status == "unavailable"
            model.status = "deleting"
            db.commit()

        estimated_bytes = 0
        released_bytes = 0
        warnings: list[str] = []
        destructive_started = False
        trusted_target: Path | None = None
        usage_path: Path | None = None
        free_before: int | None = None

        def measure_released() -> None:
            nonlocal released_bytes
            if not destructive_started or usage_path is None:
                return
            free_after = self._disk_free(usage_path)
            if free_before is None or free_after is None:
                warning = "Disk usage could not be measured"
                if warning not in warnings:
                    warnings.append(warning)
                released_bytes = 0
            else:
                released_bytes = max(0, free_after - free_before)

        def complete_deletion() -> dict[str, Any]:
            self._remove_database_record(model_id)
            measure_released()
            return self._deletion_result(
                model_id=model_id,
                source=source,
                released_bytes=released_bytes,
                estimated_bytes=estimated_bytes,
                warnings=warnings,
            )

        try:
            if source == "huggingface":
                repository_id = validate_repository_id(repository_id or "")
                target = cache_repository_path(self.hf_cache_dir, repository_id)
                trusted_target = target
                missing_can_reconcile = unavailable or reentry or failed_retry
                if missing_can_reconcile and self._path_is_absent(target):
                    if not reentry:
                        context.check_control()
                    return complete_deletion()
                elif missing_can_reconcile and self._is_orphaned_hf_cache(target):
                    estimated_bytes = directory_size(target)
                    usage_path = self.hf_cache_dir
                    usage_path.mkdir(parents=True, exist_ok=True)
                    free_before = self._disk_free(usage_path)
                    context.check_control()
                    if not self._is_orphaned_hf_cache(target):
                        raise RuntimeError(
                            "Hugging Face orphaned cache changed before deletion"
                        )
                    destructive_started = True
                    self.local_remover(self.hf_cache_dir, target)
                    if not self._path_is_absent(target):
                        raise RuntimeError(
                            "Hugging Face orphaned cache deletion did not remove the target"
                        )
                    return complete_deletion()
                else:
                    estimated_bytes = directory_size(target)
                    usage_path = self.hf_cache_dir
                    usage_path.mkdir(parents=True, exist_ok=True)
                    free_before = self._disk_free(usage_path)
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
                    destructive_started = True
                    result = self._run_hf_json([*base_argv, "--yes", "--json"])
                    self._validate_hf_result(result)
                    if not self._path_is_absent(target):
                        raise RuntimeError("Hugging Face cache deletion did not remove the target")
            else:
                target = Path(local_path)
                trusted_target = target
                missing_can_reconcile = unavailable or reentry or failed_retry
                if missing_can_reconcile and self._path_is_absent(target):
                    if not reentry:
                        context.check_control()
                    return complete_deletion()
                else:
                    root_lexical, target_lexical, resolved_target = (
                        self._validated_local_paths(target)
                    )
                    estimated_bytes = directory_size(resolved_target)
                    usage_path = resolved_target.parent
                    free_before = self._disk_free(usage_path)
                    context.check_control()
                    root_lexical, target_lexical, _resolved_target = (
                        self._validated_local_paths(target)
                    )
                    destructive_started = True
                    try:
                        self.local_remover(root_lexical, target_lexical)
                    except OSError as exc:
                        raise RuntimeError("Local model deletion failed") from exc
                    if not self._path_is_absent(target_lexical):
                        raise RuntimeError("Local model deletion did not remove the target")

            self._remove_database_record(model_id)
            measure_released()
            return self._deletion_result(
                model_id=model_id,
                source=source,
                released_bytes=released_bytes,
                estimated_bytes=estimated_bytes,
                warnings=warnings,
            )
        except (TaskCancelled, TaskPaused):
            if not destructive_started:
                self._restore_delete_state(model_id, task_id)
                raise
            if trusted_target is not None and self._path_is_absent(trusted_target):
                warnings.append("Model files were removed before completion")
                return complete_deletion()
            self._mark_delete_failed(model_id)
            raise
        except Exception:
            if (
                destructive_started
                and trusted_target is not None
                and self._path_is_absent(trusted_target)
            ):
                warnings.append("Model files were removed before completion")
                return complete_deletion()
            self._mark_delete_failed(model_id)
            raise
