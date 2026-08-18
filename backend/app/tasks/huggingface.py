from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib.parse import quote

from huggingface_hub import HfApi, hf_hub_download

from app.services.discovery import directory_size, resolve_hf_snapshot
from app.tasks.engine import TaskContext

REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{10,}", re.IGNORECASE)
PARAMETER_SIZE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*([BT])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
MOE_PARAMETER_SIZE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*([BT])"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
QUANTIZATION_PATTERNS = {
    token: re.compile(
        rf"(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    for token in ("nvfp4", "fp8", "awq", "gptq")
}
NEGATED_QUANTIZATION_PREFIX = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:not|no|non)[^A-Za-z0-9]+$",
    re.IGNORECASE,
)
SPARK_LEVEL_RANK = {"review": 0, "compatible": 1, "recommended": 2}


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


def repository_file_url(repository_id: str, revision: str, filename: str) -> str:
    """Build a direct Hub URL that aria2 can resume with HTTP range requests."""
    return (
        f"https://huggingface.co/{quote(repository_id, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(filename, safe='/')}?download=true"
    )


def safe_relative_file_path(filename: str) -> Path:
    """Reject repository file names that could escape a download directory."""
    parts = filename.replace("\\", "/").split("/")
    if not filename or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe repository file name: {filename}")
    path = Path(*parts)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe repository file name: {filename}")
    return path


def stale_incomplete_files(repository: Path, *, max_age_seconds: int = 3600) -> int:
    """Remove old HF CLI temporary files without touching active downloads."""
    cutoff = time.time() - max_age_seconds
    removed = 0
    if not repository.is_dir():
        return removed
    for candidate in repository.rglob("*.incomplete"):
        try:
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _stop_aria2(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def verify_snapshot_files(
    snapshot: Path,
    files: list[dict[str, Any]],
    *,
    include: list[str],
    exclude: list[str],
) -> list[str]:
    expected = selected_file_names(files, include=include, exclude=exclude)
    missing: list[str] = []
    for name in expected:
        candidate = snapshot / name
        resolved_candidate = candidate.resolve()
        if (
            not is_safe_snapshot_file(snapshot, candidate, resolved_candidate)
            or not resolved_candidate.is_file()
        ):
            missing.append(name)
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Downloaded snapshot failed integrity check; missing: {preview}")
    return expected


def is_safe_snapshot_file(snapshot: Path, candidate: Path, resolved_target: Path) -> bool:
    lexical_snapshot = Path(os.path.abspath(snapshot))
    lexical_candidate = Path(os.path.abspath(candidate))
    repository_root = (
        snapshot.parent.parent.resolve()
        if snapshot.parent.name == "snapshots"
        else snapshot.resolve()
    )
    resolved_target = resolved_target.resolve()
    return lexical_candidate.is_relative_to(
        lexical_snapshot
    ) and resolved_target.is_relative_to(repository_root)


def huggingface_environment(cache_dir: Path, base_environment: dict[str, str]) -> dict[str, str]:
    cache_home = cache_dir.parent
    environment = dict(base_environment)
    environment.update(
        {
            "HOME": str(cache_home),
            "HF_HOME": str(cache_home),
            "HF_HUB_CACHE": str(cache_dir),
            "HF_XET_CACHE": str(cache_home / "xet"),
            "XDG_CACHE_HOME": str(cache_home / ".cache"),
        }
    )
    return environment


def sanitize_cli_output(value: str) -> str:
    sanitized = ANSI_ESCAPE.sub("", value)
    sanitized = HF_TOKEN_PATTERN.sub("[REDACTED_HF_TOKEN]", sanitized)
    sanitized = "".join(
        character
        for character in sanitized
        if character in "\n\t" or ord(character) >= 32
    )
    return sanitized.strip()[-4000:]


def has_quantization_token(repository_id: str, tags: list[str], token: str) -> bool:
    pattern = QUANTIZATION_PATTERNS[token]
    for value in (repository_id, *tags):
        for match in pattern.finditer(value):
            if not NEGATED_QUANTIZATION_PREFIX.search(value[: match.start()]):
                return True
    return False


def spark_compatibility(
    repository_id: str,
    tags: list[str],
    pipeline_tag: str | None,
) -> dict[str, Any]:
    normalized_tags = {tag.casefold() for tag in tags}
    has_nvfp4 = has_quantization_token(repository_id, tags, "nvfp4")
    has_compressed_tensors = "compressed-tensors" in normalized_tags
    has_safetensors = "safetensors" in normalized_tags
    has_runtime = bool({"vllm", "sglang"} & normalized_tags)
    has_fp8 = has_quantization_token(repository_id, tags, "fp8")
    has_low_bit = has_quantization_token(
        repository_id, tags, "awq"
    ) or has_quantization_token(repository_id, tags, "gptq")
    has_gguf = "gguf" in normalized_tags
    gguf_only = has_gguf and not has_compressed_tensors and not has_safetensors

    parameter_sizes = [
        float(value) * (1000 if unit.casefold() == "t" else 1)
        for value, unit in PARAMETER_SIZE_PATTERN.findall(repository_id)
    ]
    parameter_sizes.extend(
        float(experts) * float(value) * (1000 if unit.casefold() == "t" else 1)
        for experts, value, unit in MOE_PARAMETER_SIZE_PATTERN.findall(repository_id)
    )
    has_capacity_risk = bool(parameter_sizes and max(parameter_sizes) > 180)

    score = 0
    reasons: list[str] = []
    if has_nvfp4:
        score += 120
        reasons.append("NVFP4 量化")
    elif has_fp8:
        score += 35
        reasons.append("FP8 量化")
    elif has_low_bit:
        score += 30
        reasons.append("低比特量化")
    if has_capacity_risk:
        score -= 120
        reasons.append("模型规模需评估")
    if gguf_only:
        score -= 40
        reasons.append("需要额外运行时")
    if has_compressed_tensors:
        score += 30
        reasons.append("压缩权重格式")
    if has_safetensors:
        score += 20
        reasons.append("Safetensors 权重")
    if has_runtime:
        score += 20
        reasons.append("适配当前推理运行时")
    if (pipeline_tag or "").casefold() in {"text-generation", "image-text-to-text"}:
        score += 10
        reasons.append("生成任务")

    if has_capacity_risk or gguf_only:
        level = "review"
    elif (
        has_nvfp4
        and (has_compressed_tensors or has_runtime or has_safetensors)
    ):
        level = "recommended"
    elif score >= 20:
        level = "compatible"
    else:
        level = "review"
    return {"level": level, "score": score, "reasons": reasons[:3]}


class HuggingFaceService:
    def __init__(self, cache_dir: Path, token: str | None = None):
        self.cache_dir = cache_dir
        self.token = token
        self.api = HfApi(token=token)

    def set_token(self, token: str | None) -> None:
        self.token = token
        self.api = HfApi(token=token)

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = min(max(limit, 1), 50)
        models = self.api.list_models(search=query, limit=50, full=True)
        results = []
        for model in models:
            tags = list(model.tags or [])
            results.append(
                {
                    "id": model.id,
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "pipeline_tag": model.pipeline_tag,
                    "private": bool(model.private),
                    "gated": bool(model.gated),
                    "last_modified": model.last_modified,
                    "tags": tags[:20],
                    "spark_compatibility": spark_compatibility(
                        model.id,
                        tags,
                        model.pipeline_tag,
                    ),
                }
            )
        results.sort(
            key=lambda item: (
                SPARK_LEVEL_RANK[item["spark_compatibility"]["level"]],
                item["spark_compatibility"]["score"],
            ),
            reverse=True,
        )
        return results[:safe_limit]

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

    def model_card_text(
        self,
        repository_id: str,
        revision: str = "main",
        max_chars: int = 100_000,
    ) -> str:
        repository_id = validate_repository_id(repository_id)
        if (
            isinstance(max_chars, bool)
            or not isinstance(max_chars, int)
            or not 1 <= max_chars <= 500_000
        ):
            raise ValueError("max_chars must be between 1 and 500000")
        path = hf_hub_download(
            repo_id=repository_id,
            filename="README.md",
            revision=revision,
            cache_dir=self.cache_dir,
            token=self.token,
        )
        try:
            cache_root = self.cache_dir.resolve(strict=True)
            resolved_path = Path(path).resolve(strict=True)
            if (
                not resolved_path.is_relative_to(cache_root)
                or not resolved_path.is_file()
            ):
                raise ValueError("Downloaded model card is not a safe regular file")
            with resolved_path.open("rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Downloaded model card is not a safe regular file")
                payload = stream.read(max_chars * 4 + 1)
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError("Downloaded model card is not a safe regular file") from exc
        return payload.decode("utf-8", errors="replace")[:max_chars]

    def download_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        repository_id = validate_repository_id(str(payload["repository_id"]))
        revision = str(payload.get("revision") or "main")
        info = self.info(repository_id, revision)
        include = [str(pattern) for pattern in (payload.get("include") or [])]
        exclude = [str(pattern) for pattern in (payload.get("exclude") or [])]
        selected_names = selected_file_names(
            info["siblings"], include=include, exclude=exclude
        )
        sizes = {
            str(item.get("name")): max(0, int(item.get("size") or 0))
            for item in info["siblings"]
            if item.get("name") in selected_names
        }
        total_bytes = sum(sizes.values())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_repository_path(self.cache_dir, repository_id)
        removed = stale_incomplete_files(target)
        if removed:
            context.update(message=f"Removed {removed} stale incomplete cache files")
        snapshot = resolve_hf_snapshot(target)
        existing_sources: dict[str, Path] = {}
        for name in selected_names:
            relative = safe_relative_file_path(name)
            candidate = snapshot / relative if snapshot is not None else None
            if candidate is None:
                continue
            try:
                resolved = candidate.resolve(strict=True)
                if (
                    is_safe_snapshot_file(snapshot, candidate, resolved)
                    and resolved.is_file()
                    and (not sizes[name] or resolved.stat().st_size == sizes[name])
                ):
                    existing_sources[name] = resolved
            except (OSError, RuntimeError, ValueError):
                continue
        existing_bytes = sum(
            sizes[name] if sizes[name] else _safe_file_size(source)
            for name, source in existing_sources.items()
        )
        disk = shutil.disk_usage(self.cache_dir)
        validate_disk_capacity(
            total_bytes=total_bytes,
            existing_bytes=existing_bytes,
            free_bytes=disk.free,
        )
        executable = shutil.which("aria2c")
        if not executable:
            raise RuntimeError("aria2c is not installed")
        commit_hash = str(info.get("sha") or revision)
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", commit_hash):
            commit_hash = re.sub(r"[^A-Za-z0-9._-]+", "-", commit_hash)[:128]
        staging = (
            self.cache_dir
            / ".dgx-aria2"
            / f"models--{repository_id.replace('/', '--')}"
            / commit_hash
        )
        staging.mkdir(parents=True, exist_ok=True)
        cache_home = self.cache_dir.parent
        (cache_home / "xet").mkdir(parents=True, exist_ok=True)
        (cache_home / ".cache").mkdir(parents=True, exist_ok=True)
        env = huggingface_environment(self.cache_dir, dict(os.environ))
        completed_by_name = dict(existing_sources)
        context.update(
            total_bytes=total_bytes,
            completed_bytes=existing_bytes,
            progress=(existing_bytes / total_bytes * 100 if total_bytes else 0),
            message=f"Downloading {repository_id}@{revision} with aria2",
        )

        def completed_bytes() -> int:
            total = existing_bytes
            for name, path in completed_by_name.items():
                if name in existing_sources:
                    continue
                size = _safe_file_size(path)
                total += min(size, sizes[name]) if sizes[name] else size
            return total

        for name in selected_names:
            if name in existing_sources:
                continue
            relative = safe_relative_file_path(name)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected_size = sizes[name]
            current_size = _safe_file_size(destination)
            if expected_size and current_size == expected_size:
                completed_by_name[name] = destination
                continue
            if expected_size and current_size > expected_size:
                destination.unlink(missing_ok=True)
                destination.with_name(destination.name + ".aria2").unlink(missing_ok=True)
            url = repository_file_url(repository_id, revision, name)
            command = [
                executable,
                url,
                "--continue=true",
                "--always-resume=true",
                "--auto-file-renaming=false",
                "--allow-overwrite=false",
                "--file-allocation=none",
                "--max-tries=5",
                "--retry-wait=5",
                "--timeout=60",
                "--connect-timeout=30",
                "--summary-interval=1",
                "--console-log-level=warn",
                "--download-result=hide",
                f"--dir={destination.parent}",
                f"--out={destination.name}",
            ]
            if self.token:
                command.append(f"--header=Authorization: Bearer {self.token}")
            with tempfile.TemporaryFile() as cli_output:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=cli_output,
                    stderr=subprocess.STDOUT,
                )
                try:
                    while process.poll() is None:
                        context.check_control()
                        current = completed_bytes()
                        context.update(
                            progress=(current / total_bytes * 100 if total_bytes else 0),
                            completed_bytes=current,
                            total_bytes=total_bytes,
                        )
                        time.sleep(1)
                finally:
                    _stop_aria2(process)
                if process.returncode != 0:
                    cli_output.seek(0)
                    details = sanitize_cli_output(
                        cli_output.read().decode("utf-8", errors="replace")
                    )
                    suffix = f": {details}" if details else ""
                    raise RuntimeError(
                        f"aria2 download exited with code {process.returncode}{suffix}"
                    )
            if expected_size and _safe_file_size(destination) != expected_size:
                raise RuntimeError(
                    f"aria2 downloaded an unexpected size for {name}: "
                    f"{_safe_file_size(destination)} != {expected_size}"
                )
            completed_by_name[name] = destination
            current = completed_bytes()
            context.update(
                progress=(current / total_bytes * 100 if total_bytes else 100),
                completed_bytes=current,
                total_bytes=total_bytes,
                message=f"Downloaded {name}",
            )

        if not selected_names:
            raise RuntimeError("Downloaded repository has no valid snapshot")
        sources = {
            name: existing_sources.get(name) or completed_by_name[name]
            for name in selected_names
        }
        snapshots = target / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshots / commit_hash
        temporary_snapshot = snapshots / f".{commit_hash}.incomplete"
        if temporary_snapshot.exists():
            shutil.rmtree(temporary_snapshot)
        temporary_snapshot.mkdir(parents=True)
        try:
            for name, source in sources.items():
                relative = safe_relative_file_path(name)
                destination = temporary_snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            verify_snapshot_files(
                temporary_snapshot,
                info["siblings"],
                include=include,
                exclude=exclude,
            )
            if snapshot_path.exists():
                shutil.rmtree(snapshot_path)
            os.replace(temporary_snapshot, snapshot_path)
        except Exception:
            if temporary_snapshot.exists():
                shutil.rmtree(temporary_snapshot)
            raise
        refs = target / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        if revision == "main":
            (refs / "main").write_text(commit_hash, encoding="utf-8")
        elif re.fullmatch(r"[A-Za-z0-9._/-]+", revision):
            try:
                ref_path = refs / safe_relative_file_path(revision)
            except ValueError:
                ref_path = None
            if ref_path is not None:
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                ref_path.write_text(commit_hash, encoding="utf-8")
        shutil.rmtree(staging, ignore_errors=True)
        snapshot = resolve_hf_snapshot(target)
        if snapshot is None:
            raise RuntimeError("Downloaded repository has no valid snapshot")
        verified_files = verify_snapshot_files(
            snapshot,
            info["siblings"],
            include=include,
            exclude=exclude,
        )
        completed = sum(
            sizes[name] if sizes[name] else _safe_file_size(source)
            for name, source in sources.items()
        )
        context.update(progress=100, completed_bytes=completed, total_bytes=total_bytes)
        return {
            "repository_id": repository_id,
            "revision": revision,
            "commit_hash": info["sha"],
            "local_path": str(snapshot),
            "size_bytes": directory_size(snapshot),
            "verified_files": len(verified_files),
        }
