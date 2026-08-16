from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
from collections import namedtuple
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from huggingface_hub import ModelCard
from pydantic import BaseModel, ConfigDict, Field
from yaml.error import YAMLError

MAX_JSON_BYTES = 1024**2
MAX_CARD_CHARS = 500_000
MAX_TOKENIZER_FILE_BYTES = 128 * 1024**2

DEPLOYMENT_FLAGS = {
    "--max-model-len": "context_length",
    "--context-length": "context_length",
    "--gpu-memory-utilization": "memory_fraction",
    "--mem-fraction-static": "memory_fraction",
    "--max-num-seqs": "max_concurrency",
    "--max-running-requests": "max_concurrency",
    "--max-num-batched-tokens": "max_batched_tokens",
    "--quantization": "quantization",
}

GENERATION_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "stop",
}

TOKENIZER_FILES = {
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
TARGET_MODEL_KEYS = (
    "base_model",
    "base_models",
    "target_model",
    "target_models",
    "target_model_id",
    "target_model_ids",
)
SPECULATIVE_METHOD_KEYS = (
    "speculative_method",
    "speculative_decoding_method",
)
SPECULATIVE_METHODS = {"draft_model", "eagle", "eagle3", "mtp"}
SAFE_CARD_DATA_KEYS = (
    *TARGET_MODEL_KEYS,
    *SPECULATIVE_METHOD_KEYS,
    "datasets",
    "language",
    "library_name",
    "license",
    "model_name",
    "pipeline_tag",
    "tags",
)

OPEN_FENCE_LINE = re.compile(r"^[ \t]*```(?P<language>[A-Za-z0-9_-]*)[ \t]*(?:\r?\n)?$")
CLOSE_FENCE_LINE = re.compile(r"^[ \t]*```[ \t]*(?:\r?\n)?$")
QUANTIZATION_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
REPOSITORY_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


class ModelEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str
    config: dict[str, Any]
    generation_config: dict[str, Any]
    tokenizer_fingerprint: str | None
    card_text: str
    card_data: dict[str, Any]
    card_deployment_values: dict[str, int | float | bool | str]
    card_generation_values: dict[str, Any]
    local_generation_values: dict[str, Any]
    target_model_ids: list[str]
    speculative_method: str | None
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    warnings: list[str]


Fence = namedtuple("Fence", "language body start end closed")


def iter_fences(card_text: str) -> Iterator[Fence]:
    """Yield fenced blocks with one linear pass over the bounded card text."""
    lines = card_text.splitlines(keepends=True)
    offset = 0
    index = 0
    while index < len(lines):
        opening = OPEN_FENCE_LINE.fullmatch(lines[index])
        if opening is None:
            offset += len(lines[index])
            index += 1
            continue
        start = offset
        language = opening.group("language")
        offset += len(lines[index])
        index += 1
        body_parts: list[str] = []
        closed = False
        while index < len(lines):
            line = lines[index]
            if CLOSE_FENCE_LINE.fullmatch(line):
                offset += len(line)
                index += 1
                closed = True
                break
            body_parts.append(line)
            offset += len(line)
            index += 1
        yield Fence(language, "".join(body_parts), start, offset, closed)


@contextmanager
def _open_model_file(model_root: Path, filename: str) -> Iterator[Any]:
    if os.name == "posix":
        root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        root_fd = os.open(model_root, root_flags)
        file_fd: int | None = None
        try:
            file_fd = os.open(
                filename,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError("model evidence file is not regular")
            with os.fdopen(file_fd, "rb") as stream:
                file_fd = None
                yield stream
        finally:
            if file_fd is not None:
                os.close(file_fd)
            os.close(root_fd)
        return

    candidate = model_root / filename
    if candidate.is_symlink():
        raise ValueError("model evidence file is a symlink")
    with candidate.open("rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("model evidence file is not regular")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(model_root):
            raise ValueError("model evidence file escapes model directory")
        yield stream


def _read_model_file(
    model_root: Path, filename: str, max_bytes: int
) -> tuple[bytes | None, str]:
    try:
        with _open_model_file(model_root, filename) as stream:
            payload = stream.read(max_bytes + 1)
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError):
        return None, "unreadable"
    if len(payload) > max_bytes:
        return payload, "too_large"
    return payload, "ok"


def _read_json_dict(model_root: Path, filename: str, warnings: list[str]) -> dict[str, Any]:
    payload, status = _read_model_file(model_root, filename, MAX_JSON_BYTES)
    if status == "missing":
        return {}
    if status == "unreadable":
        warnings.append(f"{filename} is not a safe regular file")
        return {}
    if status == "too_large":
        warnings.append(f"{filename} exceeds the size limit")
        return {}
    assert payload is not None
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        warnings.append(f"{filename} is not valid JSON")
        return {}
    if not isinstance(value, dict):
        warnings.append(f"{filename} must contain a JSON object")
        return {}
    return value


def _read_card(model_root: Path, max_chars: int, warnings: list[str]) -> str:
    payload, status = _read_model_file(model_root, "README.md", max_chars * 4 + 1)
    if status == "missing":
        return ""
    if status == "unreadable" or payload is None:
        warnings.append("README.md is not a safe regular file")
        return ""
    text = payload.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > max_chars:
        warnings.append("README.md was truncated")
    return text[:max_chars]


def _tokenizer_fingerprint(model_root: Path) -> tuple[str | None, list[str]]:
    digest = hashlib.sha256()
    found = False
    warnings: list[str] = []
    for filename in sorted(TOKENIZER_FILES):
        content, status = _read_model_file(model_root, filename, MAX_TOKENIZER_FILE_BYTES)
        if status == "missing":
            continue
        if status == "too_large":
            warnings.append(f"{filename} is too large for tokenizer fingerprint")
            continue
        if status != "ok" or content is None:
            warnings.append(f"{filename} could not be read for tokenizer fingerprint")
            continue
        found = True
        encoded_name = filename.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return (digest.hexdigest() if found else None), warnings


def tokenizer_fingerprint(model_path: Path | str) -> str | None:
    supplied_path = Path(model_path)
    if not supplied_path.is_dir() or supplied_path.is_symlink():
        raise ValueError("model directory must be a regular directory")
    fingerprint, _warnings = _tokenizer_fingerprint(supplied_path.resolve(strict=True))
    return fingerprint


def _normalize_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized if normalized > 0 else None


def _normalize_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if normalized != normalized or normalized in {float("inf"), float("-inf")}:
        return None
    return normalized


def _normalize_generation_value(key: str, value: Any) -> Any | None:
    if key in {"top_k", "max_tokens"}:
        return _normalize_int(value)
    if key == "stop":
        if isinstance(value, str):
            return value[:256] if value else None
        if isinstance(value, list) and 0 < len(value) <= 16:
            normalized = [item[:256] for item in value if isinstance(item, str) and item]
            return normalized or None
        return None
    return _normalize_float(value)


def _extract_generation_values(value: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for key in GENERATION_KEYS:
        if key not in value:
            continue
        normalized = _normalize_generation_value(key, value[key])
        if normalized is not None:
            extracted[key] = normalized
    return extracted


def _normalize_deployment_value(key: str, value: str) -> int | float | str | None:
    if value.startswith("-"):
        return None
    if key in {"context_length", "max_concurrency", "max_batched_tokens"}:
        return _normalize_int(value)
    if key == "memory_fraction":
        return _normalize_float(value)
    if key == "quantization" and QUANTIZATION_VALUE.fullmatch(value):
        return value
    return None


def _extract_shell_values(body: str) -> dict[str, int | float | bool | str]:
    try:
        tokens = shlex.split(body, posix=True)
    except ValueError:
        raise
    extracted: dict[str, int | float | bool | str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag, separator, inline_value = token.partition("=")
        destination = DEPLOYMENT_FLAGS.get(flag)
        if destination is None:
            index += 1
            continue
        if separator:
            raw_value = inline_value
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            index += 1
            raw_value = tokens[index]
        else:
            index += 1
            continue
        normalized = _normalize_deployment_value(destination, raw_value)
        if normalized is not None:
            extracted[destination] = normalized
        index += 1
    return extracted


def _sanitize_shell_fences(card_text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for fence in iter_fences(card_text):
        if fence.language.casefold() not in {"bash", "sh", "shell"}:
            continue
        parts.append(card_text[cursor : fence.start])
        try:
            values = _extract_shell_values(fence.body)
        except ValueError:
            values = {}
        safe_lines = [f"{key}={json.dumps(value)}" for key, value in sorted(values.items())]
        body = "\n".join(safe_lines)
        parts.append(f"```{fence.language}\n{body}\n```")
        cursor = fence.end
    if not parts:
        return card_text
    parts.append(card_text[cursor:])
    return "".join(parts)


def _safe_card_data(value: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in SAFE_CARD_DATA_KEYS:
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            if item is not None:
                safe[key] = item
        elif isinstance(item, list):
            scalar_items = [
                entry for entry in item if isinstance(entry, (str, int, float, bool))
            ]
            if scalar_items:
                safe[key] = scalar_items[:100]
    return safe


def _extract_card_data(card_text: str, warnings: list[str]) -> dict[str, Any]:
    if not card_text.startswith("---"):
        return {}
    try:
        serialized = ModelCard(card_text).data.to_dict()
    except (TypeError, ValueError, YAMLError):
        warnings.append("README.md contains invalid model card metadata")
        return {}
    return _safe_card_data(serialized if isinstance(serialized, dict) else {})


def _target_model_ids(card_data: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for key in TARGET_MODEL_KEYS:
        value = card_data.get(key)
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            normalized = candidate.strip()
            if not REPOSITORY_ID.fullmatch(normalized):
                continue
            if normalized not in targets:
                targets.append(normalized)
    return targets


def _speculative_method(card_data: dict[str, Any]) -> str | None:
    for key in SPECULATIVE_METHOD_KEYS:
        value = card_data.get(key)
        if isinstance(value, str):
            normalized = value.strip().casefold().replace("-", "_")
            if normalized in SPECULATIVE_METHODS:
                return normalized
    return None


def _card_values(
    card_text: str, warnings: list[str]
) -> tuple[dict[str, int | float | bool | str], dict[str, Any]]:
    deployment: dict[str, int | float | bool | str] = {}
    generation: dict[str, Any] = {}
    for fence in iter_fences(card_text):
        language = fence.language.casefold()
        if not fence.closed and language in {"bash", "sh", "shell"}:
            warnings.append("README.md contains a malformed shell fence")
            continue
        body = fence.body
        if language in {"bash", "sh", "shell"}:
            try:
                deployment.update(_extract_shell_values(body))
            except ValueError:
                warnings.append("README.md contains a malformed shell fence")
        elif language == "json":
            try:
                value = json.loads(body, parse_constant=_reject_json_constant)
            except (ValueError, RecursionError):
                warnings.append("README.md contains a malformed JSON fence")
                continue
            if isinstance(value, dict):
                generation.update(_extract_generation_values(value))
    return deployment, generation


class ModelEvidenceLoader:
    def __init__(self, card_max_chars: int = 100_000):
        if (
            isinstance(card_max_chars, bool)
            or not isinstance(card_max_chars, int)
            or not 1 <= card_max_chars <= MAX_CARD_CHARS
        ):
            raise ValueError(f"card_max_chars must be between 1 and {MAX_CARD_CHARS}")
        self.card_max_chars = card_max_chars

    def load(self, model_path: Path | str) -> ModelEvidence:
        supplied_path = Path(model_path)
        if not supplied_path.exists() or not supplied_path.is_dir():
            raise ValueError("model directory must exist")
        if supplied_path.is_symlink():
            raise ValueError("model directory must not be a symlink")
        model_root = supplied_path.resolve(strict=True)
        warnings: list[str] = []
        config = _read_json_dict(model_root, "config.json", warnings)
        generation_config = _read_json_dict(model_root, "generation_config.json", warnings)
        raw_card_text = _read_card(model_root, self.card_max_chars, warnings)
        card_data = _extract_card_data(raw_card_text, warnings)
        card_deployment_values, card_generation_values = _card_values(raw_card_text, warnings)
        card_text = _sanitize_shell_fences(raw_card_text)
        local_generation_values = _extract_generation_values(generation_config)
        fingerprint, tokenizer_warnings = _tokenizer_fingerprint(model_root)
        warnings.extend(tokenizer_warnings)
        targets = _target_model_ids(card_data)
        method = _speculative_method(card_data)
        hash_payload = {
            "config": config,
            "generation_config": generation_config,
            "tokenizer_fingerprint": fingerprint,
            "card_text": card_text,
            "card_data": card_data,
            "card_deployment_values": card_deployment_values,
            "card_generation_values": card_generation_values,
            "local_generation_values": local_generation_values,
            "target_model_ids": targets,
            "speculative_method": method,
        }
        evidence_hash = hashlib.sha256(
            json.dumps(
                hash_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        return ModelEvidence(
            model_path=str(model_root),
            config=config,
            generation_config=generation_config,
            tokenizer_fingerprint=fingerprint,
            card_text=card_text,
            card_data=card_data,
            card_deployment_values=card_deployment_values,
            card_generation_values=card_generation_values,
            local_generation_values=local_generation_values,
            target_model_ids=targets,
            speculative_method=method,
            evidence_hash=evidence_hash,
            warnings=warnings,
        )
