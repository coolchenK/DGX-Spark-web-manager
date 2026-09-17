from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import ModelCard
from yaml import YAMLError

TEXT_GENERATION_PIPELINES = frozenset(
    {
        "conversational",
        "image-text-to-text",
        "text-generation",
        "text2text-generation",
        "video-text-to-text",
        "visual-question-answering",
    }
)
IMAGE_INPUT_PIPELINES = frozenset(
    {"image-text-to-text", "visual-question-answering", "video-text-to-text"}
)
VIDEO_INPUT_PIPELINES = frozenset({"video-text-to-text"})
IMAGE_INPUT_TAGS = frozenset(
    {"image-text-to-text", "multimodal", "vision", "vision-language", "vlm"}
)
VIDEO_INPUT_TAGS = frozenset({"video", "video-text-to-text", "video-understanding"})
MULTIMODAL_CAPABILITIES = ("image", "video")
DRAFT_ARCHITECTURE_MARKERS = ("draftmodel", "dsparkmodel", "dflashmodel")

RUNTIME_MULTIMODAL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "vllm": ("mm_processor_kwargs", "media_io_kwargs"),
    "sglang": ("images_config", "use_audio_in_video"),
    "llama_cpp": (),
}


def _model_card_data(model_path: Path) -> dict[str, Any]:
    card_path = model_path / "README.md"
    try:
        card_text = card_path.read_text(encoding="utf-8")
        serialized = ModelCard(card_text).data.to_dict()
    except (OSError, TypeError, ValueError, YAMLError):
        return {}
    return serialized if isinstance(serialized, dict) else {}


def _normalized_values(value: Any) -> set[str]:
    candidates = value if isinstance(value, list) else [value]
    return {
        str(candidate).strip().casefold().replace("_", "-")
        for candidate in candidates
        if isinstance(candidate, str) and candidate.strip()
    }


def _model_files(model_path: Path) -> set[str]:
    try:
        return {
            path.name.casefold()
            for path in model_path.iterdir()
            if path.is_file()
        }
    except OSError:
        return set()


def infer_model_capabilities(
    model_path: Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Infer OpenAI-facing capabilities from local config and structured card metadata."""
    if config is None:
        try:
            loaded = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        config = loaded if isinstance(loaded, dict) else {}

    architectures = _normalized_values(config.get("architectures"))
    card_data = _model_card_data(model_path)
    pipelines = _normalized_values(card_data.get("pipeline_tag"))
    tags = _normalized_values(card_data.get("tags"))
    files = _model_files(model_path)

    if any(
        marker in architecture
        for architecture in architectures
        for marker in DRAFT_ARCHITECTURE_MARKERS
    ):
        return []

    capabilities: list[str] = []
    if any("causallm" in architecture for architecture in architectures) or any(
        "conditionalgeneration" in architecture for architecture in architectures
    ) or bool(pipelines.intersection(TEXT_GENERATION_PIPELINES)):
        capabilities.extend(("chat", "completion"))
    elif any(
        marker in architecture
        for architecture in architectures
        for marker in ("embedding", "sequenceclassification")
    ):
        capabilities.append("embedding")

    image_input = (
        isinstance(config.get("vision_config"), Mapping)
        or config.get("image_token_id") is not None
        or config.get("image_token_index") is not None
        or bool(pipelines.intersection(IMAGE_INPUT_PIPELINES))
        or bool(tags.intersection(IMAGE_INPUT_TAGS))
        or any("mmproj" in filename for filename in files)
    )
    video_input = (
        isinstance(config.get("video_config"), Mapping)
        or config.get("video_token_id") is not None
        or config.get("video_token_index") is not None
        or "video_preprocessor_config.json" in files
        or bool(pipelines.intersection(VIDEO_INPUT_PIPELINES))
        or bool(tags.intersection(VIDEO_INPUT_TAGS))
    )
    if image_input:
        capabilities.append("image")
    if video_input:
        capabilities.append("video")
    return capabilities


def merge_runtime_model_capabilities(
    runtime_capabilities: Iterable[str],
    model_capabilities: Iterable[str],
) -> list[str]:
    runtime = list(dict.fromkeys(runtime_capabilities))
    model = set(model_capabilities)
    if "chat" not in runtime or "chat" not in model:
        return runtime
    return [*runtime, *(item for item in MULTIMODAL_CAPABILITIES if item in model)]


def input_modalities(capabilities: Iterable[str]) -> list[str]:
    names = set(capabilities)
    modalities = ["text"] if names.intersection({"chat", "completion"}) else []
    return [*modalities, *(item for item in MULTIMODAL_CAPABILITIES if item in names)]


def runtime_multimodal_parameters(runtime: str) -> list[str]:
    return list(RUNTIME_MULTIMODAL_PARAMETERS.get(runtime, ()))
