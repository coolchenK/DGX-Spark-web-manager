from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from app.services.discovery import directory_size, resolve_hf_snapshot
from app.tasks.engine import TaskCancelled, TaskContext, TaskPaused

REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_repository_id(value: str) -> str:
    if not REPOSITORY_PATTERN.fullmatch(value):
        raise ValueError("Repository ID must use the form owner/model")
    return value


def cache_repository_path(cache_dir: Path, repository_id: str) -> Path:
    safe_id = validate_repository_id(repository_id)
    target = cache_dir / f"models--{safe_id.replace('/', '--')}"
    resolved_cache = cache_dir.resolve()
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(resolved_cache):
        raise ValueError("Repository path escapes the configured cache")
    return target


def serialize_card_data(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        serialized = to_dict()
        if isinstance(serialized, dict):
            return serialized
    return {
        str(key): item
        for key, item in vars(value).items()
        if not str(key).startswith("_")
    }


def selected_file_names(
    files: list[dict[str, Any]],
    *,
    include: list[str],
    exclude: list[str],
) -> list[str]:
    selected: list[str] = []
    for item in files:
        name = str(item.get("name") or "")
        included = not include or any(fnmatch(name, pattern) for pattern in include)
        excluded = any(fnmatch(name, pattern) for pattern in exclude)
        if included and not excluded:
            selected.append(name)
    return selected


def selected_download_size(
    files: list[dict[str, Any]],
    *,
    include: list[str],
    exclude: list[str],
) -> int:
    selected = set(selected_file_names(files, include=include, exclude=exclude))
    return sum(int(item.get("size") or 0) for item in files if item.get("name") in selected)


def validate_disk_capacity(*, total_bytes: int, existing_bytes: int, free_bytes: int) -> int:
    required_bytes = max(0, total_bytes - existing_bytes)
    if required_bytes > free_bytes:
        raise RuntimeError(
            f"Insufficient free disk space: need {required_bytes} bytes, have {free_bytes} bytes"
        )
    return required_bytes


def verify_snapshot_files(
    snapshot: Path,
    files: list[dict[str, Any]],
    *,
    include: list[str],
    exclude: list[str],
) -> list[str]:
    expected = selected_file_names(files, include=include, exclude=exclude)
    resolved_snapshot = snapshot.resolve()
    missing: list[str] = []
    for name in expected:
        candidate = (snapshot / name).resolve()
        if not candidate.is_relative_to(resolved_snapshot) or not candidate.is_file():
            missing.append(name)
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Downloaded snapshot failed integrity check; missing: {preview}")
    return expected


class HuggingFaceService:
    def __init__(self, cache_dir: Path, token: str | None = None):
        self.cache_dir = cache_dir
        self.token = token
        self.api = HfApi(token=token)

    def set_token(self, token: str | None) -> None:
        self.token = token
        self.api = HfApi(token=token)

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        models = self.api.list_models(search=query, limit=min(max(limit, 1), 50), full=True)
        return [
            {
                "id": model.id,
                "downloads": model.downloads or 0,
                "likes": model.likes or 0,
                "pipeline_tag": model.pipeline_tag,
                "private": bool(model.private),
                "gated": bool(model.gated),
                "last_modified": model.last_modified,
                "tags": list(model.tags or [])[:20],
            }
            for model in models
        ]

    def info(self, repository_id: str, revision: str = "main") -> dict[str, Any]:
        repository_id = validate_repository_id(repository_id)
        model = self.api.model_info(repository_id, revision=revision, files_metadata=True)
        siblings = [
            {"name": item.rfilename, "size": getattr(item, "size", None)}
            for item in (model.siblings or [])
        ]
        total_size = sum(item["size"] or 0 for item in siblings)
        return {
            "id": model.id,
            "sha": model.sha,
            "pipeline_tag": model.pipeline_tag,
            "private": bool(model.private),
            "gated": bool(model.gated),
            "tags": list(model.tags or []),
            "siblings": siblings,
            "total_size": total_size,
            "card_data": serialize_card_data(model.card_data),
        }

    def download_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        repository_id = validate_repository_id(str(payload["repository_id"]))
        revision = str(payload.get("revision") or "main")
        info = self.info(repository_id, revision)
        include = [str(pattern) for pattern in (payload.get("include") or [])]
        exclude = [str(pattern) for pattern in (payload.get("exclude") or [])]
        total_bytes = selected_download_size(
            info["siblings"], include=include, exclude=exclude
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_repository_path(self.cache_dir, repository_id)
        existing_bytes = directory_size(target) if target.exists() else 0
        disk = shutil.disk_usage(self.cache_dir)
        validate_disk_capacity(
            total_bytes=total_bytes,
            existing_bytes=existing_bytes,
            free_bytes=disk.free,
        )
        executable = shutil.which("hf")
        if not executable:
            raise RuntimeError("Hugging Face CLI 'hf' is not installed")
        command = [
            executable,
            "download",
            repository_id,
            "--revision",
            revision,
            "--cache-dir",
            str(self.cache_dir),
        ]
        for pattern in include:
            command.extend(["--include", str(pattern)])
        for pattern in exclude:
            command.extend(["--exclude", str(pattern)])
        env = os.environ.copy()
        if self.token:
            env["HF_TOKEN"] = self.token
        context.update(total_bytes=total_bytes, message=f"Downloading {repository_id}@{revision}")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            while process.poll() is None:
                completed = directory_size(target) if target.exists() else 0
                progress = completed / total_bytes * 100 if total_bytes else 0
                context.update(
                    progress=min(progress, 99),
                    completed_bytes=completed,
                    total_bytes=total_bytes,
                )
                try:
                    context.check_control()
                except (TaskPaused, TaskCancelled):
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.terminate()
        if process.returncode != 0:
            raise RuntimeError(f"Hugging Face download exited with code {process.returncode}")
        completed = directory_size(target)
        snapshot = resolve_hf_snapshot(target)
        verified_files = verify_snapshot_files(
            snapshot,
            info["siblings"],
            include=include,
            exclude=exclude,
        )
        context.update(progress=100, completed_bytes=completed, total_bytes=total_bytes)
        return {
            "repository_id": repository_id,
            "revision": revision,
            "commit_hash": info["sha"],
            "local_path": str(snapshot),
            "size_bytes": completed,
            "verified_files": len(verified_files),
        }
