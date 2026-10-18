"""Keep operations-provider choices backed by Magpie without copying vendor secrets."""

import hashlib
import json
import logging
import re
from threading import Lock
from uuid import NAMESPACE_URL, uuid5

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Provider, SecretSetting
from app.security import SecretBox
from app.services.providers import (
    PinnedProviderEndpoint,
    ProviderService,
    resolve_provider_endpoint,
)

GATEWAY_BASE = "http://127.0.0.1:3425/v1"
MAPPING_KEY = "magpie_provider_mapping"
logger = logging.getLogger(__name__)


def resolve_magpie_provider_endpoint(value: str) -> PinnedProviderEndpoint:
    if value in {f"{GATEWAY_BASE}/models", f"{GATEWAY_BASE}/chat/completions"}:
        return PinnedProviderEndpoint(value, "127.0.0.1:3425", None)
    return resolve_provider_endpoint(value)


class MagpieProviderSync:
    def __init__(self, secret_box: SecretBox, web_key: str, gateway_key: str):
        self.secret_box = secret_box
        self.web_key = web_key
        self.gateway_key = gateway_key
        self._lock = Lock()

    def sync(self, db: Session) -> bool:
        with self._lock:
            try:
                providers = self._fetch()
            except (httpx.HTTPError, ValueError):
                logger.warning(
                    "Magpie provider sync unavailable; retaining previous provider snapshot"
                )
                return False
            self._apply(db, providers)
            return True

    def _fetch(self) -> list[dict]:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.get(
                "http://127.0.0.1:3430/api/providers",
                headers={"Cookie": f"magpie_web_3430={self.web_key}"},
            )
        response.raise_for_status()
        payload = response.json()
        providers = payload.get("providers") if isinstance(payload, dict) else None
        if not isinstance(providers, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", item["id"])
            or not isinstance(item.get("models"), list)
            for item in providers
        ):
            raise ValueError("Invalid Magpie provider snapshot")
        return providers

    def _apply(self, db: Session, snapshot: list[dict]) -> None:
        saved = db.get(SecretSetting, MAPPING_KEY)
        mapping = json.loads(self.secret_box.decrypt(saved.encrypted_value)) if saved else {}
        existing = list(db.scalars(select(Provider)))
        seen: set[str] = set()
        for source in snapshot:
            source_id = source["id"]
            if source_id == "dgx-local":
                continue
            models = [
                item["id"]
                for item in source["models"]
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"]
                and len(f"{source_id}/{item['id']}") <= 255
                and item.get("on") is True
                and not item.get("noAPI")
                and item.get("kind") not in {"image", "video"}
            ]
            entry = mapping.get(source_id, {})
            provider = db.get(Provider, entry["id"]) if entry else None
            name = str(source.get("name") or source_id).strip()[:220]
            if provider is None and not entry:
                provider = next(
                    (
                        item
                        for item in existing
                        if item.name == name
                        and item.default_model in models
                        and item.base_url != GATEWAY_BASE
                    ),
                    None,
                )
            if not models:
                continue
            if provider is None:
                provider = Provider(
                    id=str(uuid5(NAMESPACE_URL, f"magpie:{source_id}")),
                    name=name,
                    timeout_seconds=60,
                    config_revision=0,
                )
                db.add(provider)
                existing.append(provider)
            if any(item.id != provider.id and item.name == name for item in existing):
                name = f"{name} · {provider.id[:8]}"
            previous_model = (provider.default_model or "").removeprefix(f"{source_id}/")
            default_model = previous_model if previous_model in models else models[0]
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        key: source.get(key)
                        for key in (
                            "chat",
                            "responses",
                            "anthropic",
                            "gemini",
                            "headers",
                            "extraBody",
                            "baseAPI",
                            "key",
                            "off",
                            "ready",
                        )
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            values = {
                "name": name,
                "base_url": GATEWAY_BASE,
                "default_model": f"{source_id}/{default_model}",
                "headers": {},
                "enabled": not source.get("off") and source.get("ready") is True,
            }
            key_changed = not provider.encrypted_api_key or (
                self.secret_box.decrypt(provider.encrypted_api_key) != self.gateway_key
            )
            changed = (
                key_changed
                or entry.get("fingerprint") != fingerprint
                or any(
                    getattr(provider, key) != value
                    for key, value in values.items()
                    if key != "name"
                )
            )
            for key, value in values.items():
                setattr(provider, key, value)
            if key_changed:
                provider.encrypted_api_key = self.secret_box.encrypt(self.gateway_key)
            if changed:
                ProviderService._invalidate_probe(provider)
                provider.config_revision = (provider.config_revision or 0) + 1
            mapping[source_id] = {"id": provider.id, "fingerprint": fingerprint}
            seen.add(source_id)
        # Preserve IDs referenced by existing conversations when a provider is removed.
        for source_id, entry in mapping.items():
            if source_id not in seen and (provider := db.get(Provider, entry["id"])):
                if provider.enabled:
                    provider.enabled = False
                    ProviderService._invalidate_probe(provider)
                    provider.config_revision = (provider.config_revision or 0) + 1
        serialized = json.dumps(mapping, sort_keys=True)
        if saved is None:
            db.add(
                SecretSetting(key=MAPPING_KEY, encrypted_value=self.secret_box.encrypt(serialized))
            )
        elif self.secret_box.decrypt(saved.encrypted_value) != serialized:
            saved.encrypted_value = self.secret_box.encrypt(serialized)
        db.commit()
