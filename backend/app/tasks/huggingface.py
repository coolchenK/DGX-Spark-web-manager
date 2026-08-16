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

from huggingface_hub import HfApi, hf_hub_download

from app.services.discovery import directory_size, resolve_hf_snapshot
from app.tasks.engine import TaskCancelled, TaskContext, TaskPaused

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
        cache_home = self.cache_dir.parent
        (cache_home / "xet").mkdir(parents=True, exist_ok=True)
        (cache_home / ".cache").mkdir(parents=True, exist_ok=True)
        env = huggingface_environment(self.cache_dir, dict(os.environ))
        if self.token:
            env["HF_TOKEN"] = self.token
        context.update(total_bytes=total_bytes, message=f"Downloading {repository_id}@{revision}")
        with tempfile.TemporaryFile() as cli_output:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=cli_output,
                stderr=subprocess.STDOUT,
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
                cli_output.seek(0)
                details = sanitize_cli_output(cli_output.read().decode("utf-8", errors="replace"))
                suffix = f": {details}" if details else ""
                raise RuntimeError(
                    f"Hugging Face download exited with code {process.returncode}{suffix}"
                )
        completed = directory_size(target)
        snapshot = resolve_hf_snapshot(target)
        if snapshot is None:
            raise RuntimeError("Downloaded repository has no valid snapshot")
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
