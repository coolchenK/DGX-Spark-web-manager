from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Deployment, ModelAsset


@dataclass(frozen=True)
class ModelReference:
    deployment_id: str
    deployment_name: str
    usage: Literal["base", "draft", "legacy_path"]


def _resolved(value: str | None) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return Path(value).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


class ModelLifecycleService:
    def __init__(self, model_roots: tuple[Path, ...], hf_cache_dir: Path):
        self.model_roots = model_roots
        self.hf_cache_dir = hf_cache_dir

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
