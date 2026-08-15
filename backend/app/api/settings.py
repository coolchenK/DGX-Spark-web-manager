from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import SecretSetting

router = APIRouter(prefix="/api/settings", tags=["settings"])


class HuggingFaceTokenUpdate(BaseModel):
    token: str | None = Field(default=None, max_length=4096)


@router.get("")
def get_settings(request: Request, db: DbSession, _: Admin) -> dict[str, Any]:
    stored = db.get(SecretSetting, "huggingface_token")
    settings = request.app.state.settings
    return {
        "huggingface": {
            "token_configured": bool(stored or request.app.state.huggingface_service.token),
            "cache_dir": str(settings.hf_cache_dir),
        },
        "models": {"roots": [str(path) for path in settings.model_root_paths]},
        "runtimes": {
            "vllm": sorted(settings.vllm_images),
            "sglang": sorted(settings.sglang_images),
        },
    }


@router.patch("/huggingface")
def update_huggingface_token(
    payload: HuggingFaceTokenUpdate,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    token = payload.token.strip() if payload.token else None
    stored = db.get(SecretSetting, "huggingface_token")
    if token:
        encrypted = request.app.state.secret_box.encrypt(token)
        if stored:
            stored.encrypted_value = encrypted
        else:
            db.add(SecretSetting(key="huggingface_token", encrypted_value=encrypted))
    elif stored:
        db.delete(stored)
    request.app.state.huggingface_service.set_token(token)
    record_audit(
        db,
        actor=str(admin["username"]),
        action="settings.huggingface_token.update",
        resource_type="settings",
        details={"configured": bool(token)},
    )
    db.commit()
    return {"token_configured": bool(token)}
