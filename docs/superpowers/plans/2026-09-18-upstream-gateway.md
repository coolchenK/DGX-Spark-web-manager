# Upstream Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status:** implemented on 2026-09-18. Backend `1379 passed, 4 failed, 51 skipped`
> (baseline `1342 passed, 3 failed, 51 skipped`; the failures are the pre-existing Windows-only
> Host Agent socket and timing flakes, whose exact test names vary between runs). Frontend
> `127 passed` across 12 files (baseline 118). `ruff` and `oxlint` clean; production build succeeds.
>
> Two deviations from this plan were required and are recorded in the design document: the
> upstream URL is now normalised to a single `/v1` prefix (the shipped code doubled it), and
> `resolve_upstream_gateway` takes the `SecretBox` so stored keys can be decrypted.
**Goal:** Complete the upstream fallback added in `4435dee` so models this manager does not host are discoverable through `/v1/models`, configurable and testable from the panel without a container restart, and metered with real token usage.

**Architecture:** A dedicated `upstream_gateway` service owns config resolution (database `SecretSetting` over environment variables), URL validation, and encrypted storage. The gateway API merges upstream models into the existing OpenAI models response behind a short-lived bounded cache, with local routes always taking precedence. A shared usage extractor fixes the missing token metrics on the fallback path without altering the verbatim forwarding contract. The React gateway page gains an upstream section that never reveals the stored secret.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, Pydantic, HTTPX, respx, React 19, TypeScript 6, Ant Design 5, TanStack Query, Vitest, pytest.

**Local commands (Windows workspace):** Python is at
`C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe` and Node at
`C:\Program Files\nodejs`. Run backend tests as
`& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests -q`
with `DGX_SECRET_KEY` and `DGX_ADMIN_PASSWORD` set; run frontend commands as
`& 'C:\Program Files\nodejs\corepack.cmd' pnpm test` inside `frontend/`.
On the DGX Spark host use the repo-default `pytest` / `pnpm` from the README.

**Baseline before starting:** `1342 passed, 3 failed, 51 skipped` (backend; the 3 failures are Windows-only: two Host Agent socket tests and one 5 ms wall-clock assertion) and `118 passed` (frontend, 11 files). Treat only new failures as regressions.

---

## File Map

- Create `backend/app/services/upstream_gateway.py`: config resolution, validation, encrypted storage, model fetch, bounded cache.
- Create `backend/app/api/upstream.py`: `GET/PUT /api/gateway/upstream`, `POST /api/gateway/upstream/test`.
- Modify `backend/app/gateway/proxy.py`: shared JSON/SSE usage extractors.
- Modify `backend/app/api/gateway.py`: resolve upstream config from the service, add token usage to the three fallback metric calls, merge upstream models into `/v1/models` and `/v1/models/{model}`.
- Modify `backend/app/config.py`: upstream model cache TTL setting.
- Modify `backend/app/main.py`: register the upstream router and construct the upstream cache on `app.state`.
- Create `backend/tests/test_upstream_gateway.py`: config precedence, validation, secret handling, cache, model merge, usage metering.
- Modify `backend/tests/test_gateway.py`: `/v1/models` upstream merge and fallback usage coverage.
- Modify `frontend/src/api/types.ts`: upstream config and test-result contracts.
- Modify `frontend/src/pages/GatewayPage.tsx`: upstream section, configure dialog, test action.
- Create `frontend/src/pages/GatewayPage.test.tsx`: unset/configured/success/failure rendering and validation.
- Modify `.env.example`: document that the env pair now seeds the default and is no longer the only source.
- Modify `README.md`: config table entry and OpenAI API table note for upstream discovery.
- Modify `docs/API.md`: new management endpoints and the upstream fields in `/v1/models`.

---

### Task 1: Shared usage extraction and fallback token metrics

The fallback path records status and latency but never token counts, so `/api/gateway/stats` undercounts upstream traffic. Fix this first; it is independent of the rest.

**Files:**
- Modify: `backend/app/gateway/proxy.py`
- Test: `backend/tests/test_gateway.py`

- [ ] **Step 1: Write the failing extractor tests**

Add to `backend/tests/test_gateway.py`:

```python
from app.gateway.proxy import extract_usage_from_json, UsageScanner


def test_extract_usage_from_json_returns_usage_mapping():
    content = b'{"usage": {"prompt_tokens": 11, "completion_tokens": 7}}'
    assert extract_usage_from_json(content) == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
    }


def test_extract_usage_from_json_tolerates_non_json_and_missing_usage():
    assert extract_usage_from_json(b"not json") is None
    assert extract_usage_from_json(b'{"choices": []}') is None


def test_usage_scanner_reads_usage_from_split_sse_frames():
    scanner = UsageScanner()
    scanner.feed(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
    assert scanner.usage is None
    scanner.feed(b'data: {"choices":[],"usage":{"prompt_tokens":3,')
    scanner.feed(b'"completion_tokens":9}}\n\n')
    scanner.feed(b"data: [DONE]\n\n")
    assert scanner.usage == {"prompt_tokens": 3, "completion_tokens": 9}


def test_usage_scanner_keeps_buffer_bounded():
    scanner = UsageScanner()
    scanner.feed(b"x" * (UsageScanner.MAX_BUFFER * 2))
    assert len(scanner._buffer) <= UsageScanner.MAX_BUFFER
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_gateway.py -k usage -q`
Expected: collection error — `ImportError: cannot import name 'extract_usage_from_json'`.

- [ ] **Step 3: Add the extractors and refactor the local proxy to use them**

In `backend/app/gateway/proxy.py`, add above `record_request_metric`:

```python
def extract_usage_from_json(content: bytes) -> dict[str, Any] | None:
    """Return the OpenAI usage mapping from a JSON response body, if present."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict):
        return parsed["usage"]
    return None


class UsageScanner:
    """Collect the last usage mapping from an SSE byte stream without altering it.

    The buffer only ever holds the unterminated tail of the stream, so memory
    stays bounded even for very long completions.
    """

    MAX_BUFFER = 64 * 1024

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while True:
            positions = [(self._buffer.find(b"\n\n"), 2), (self._buffer.find(b"\r\n\r\n"), 4)]
            positions = [(index, length) for index, length in positions if index >= 0]
            if not positions:
                break
            index, length = min(positions)
            frame = self._buffer[:index]
            self._buffer = self._buffer[index + length:]
            for line in frame.replace(b"\r\n", b"\n").split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                    self.usage = event["usage"]
        if len(self._buffer) > self.MAX_BUFFER:
            self._buffer = self._buffer[-self.MAX_BUFFER:]
```

Replace the inline non-streaming parse in `proxy_openai_request` (currently lines 241-248) with:

```python
    usage = extract_usage_from_json(content)
```

- [ ] **Step 4: Run the extractor tests and verify GREEN**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_gateway.py -q`
Expected: all gateway tests pass, including the four new ones.

- [ ] **Step 5: Add a failing test that fallback metrics carry token usage**

Add to `backend/tests/test_upstream_gateway.py` (create the file):

```python
import respx
from httpx import Response


@respx.mock
def test_fallback_records_prompt_and_completion_tokens(authenticated_client, settings):
    settings.fallback_base_url = "https://upstream.test/v1"
    settings.fallback_api_key = "upstream-secret"
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 21, "completion_tokens": 13},
            },
        )
    )
    key = authenticated_client.post("/api/keys", json={"name": "test"}).json()["key"]

    response = authenticated_client.post(
        "/v1/chat/completions",
        json={"model": "remote-only-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert response.status_code == 200
    metrics = authenticated_client.get("/api/gateway/stats").json()
    assert metrics["prompt_tokens"] == 21
    assert metrics["completion_tokens"] == 13
```

Note: settings mutation alone does not hot-apply; this test asserts the environment source, which Task 2 preserves.

- [ ] **Step 6: Run it and verify RED**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_upstream_gateway.py -q`
Expected: FAIL — `prompt_tokens` is 0.

- [ ] **Step 7: Pass usage on every fallback metric call**

In `backend/app/api/gateway.py` `_proxy_fallback`:

Non-streaming branch — parse before closing the response:

```python
    if not forward_body.get("stream"):
        content = await upstream.aread()
        status_code = upstream.status_code
        content_type = upstream.headers.get("content-type", "application/json")
        await upstream.aclose()
        await client.aclose()
        record_request_metric(
            request.app.state.database.session_factory,
            model=str(body.get("model")),
            endpoint=endpoint,
            status_code=status_code,
            started_at=started_at,
            usage=extract_usage_from_json(content),
        )
        return Response(content=content, status_code=status_code, media_type=content_type)
```

Streaming branch — scan without altering the relayed bytes:

```python
    scanner = UsageScanner()

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                scanner.feed(chunk)
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            record_request_metric(
                request.app.state.database.session_factory,
                model=str(body.get("model")),
                endpoint=endpoint,
                status_code=upstream.status_code,
                started_at=started_at,
                usage=scanner.usage,
            )
```

Add `extract_usage_from_json` and `UsageScanner` to the existing `from app.gateway.proxy import (` block.

- [ ] **Step 8: Run the fallback tests and verify GREEN**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_upstream_gateway.py backend/tests/test_gateway.py -q`
Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add backend/app/gateway/proxy.py backend/app/api/gateway.py backend/tests/test_gateway.py backend/tests/test_upstream_gateway.py
git commit -m "fix: record token usage for upstream gateway traffic"
```

---

### Task 2: Upstream configuration service

**Files:**
- Create: `backend/app/services/upstream_gateway.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_upstream_gateway.py`

- [ ] **Step 1: Add the cache TTL setting**

In `backend/app/config.py`, next to the fallback pair:

```python
    upstream_models_cache_seconds: int = Field(default=30, ge=5, le=600)
```

- [ ] **Step 2: Write failing config tests**

Append to `backend/tests/test_upstream_gateway.py`:

```python
import pytest

from app.services import upstream_gateway


def test_database_config_wins_over_environment(settings):
    settings.fallback_base_url = "https://env.test/v1"
    settings.fallback_api_key = "env-key"
    session = {"db": None}  # replaced by the client fixture session below


@pytest.mark.parametrize(
    "value",
    ["ftp://x/v1", "https://", "https://user:pw@host/v1", "https://host/v1#frag", "  ", "https://" + "a" * 600],
)
def test_invalid_base_urls_are_rejected(value):
    with pytest.raises(ValueError):
        upstream_gateway.validate_upstream_base_url(value)


def test_valid_base_url_is_normalised():
    assert (
        upstream_gateway.validate_upstream_base_url("  https://upstream.test/v1/  ")
        == "https://upstream.test/v1"
    )
```

Replace the placeholder database test in Step 6 once the API exists; keep the parametrised validation tests here.

- [ ] **Step 3: Run and verify RED**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_upstream_gateway.py -q`
Expected: `ModuleNotFoundError: No module named 'app.services.upstream_gateway'`.

- [ ] **Step 4: Implement the service**

Create `backend/app/services/upstream_gateway.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import SecretSetting
from app.security import SecretBox

BASE_URL_KEY = "upstream_base_url"
API_KEY_KEY = "upstream_api_key"
MAX_BASE_URL_LENGTH = 500
MODELS_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class UpstreamGatewayConfig:
    base_url: str
    api_key: str | None
    source: Literal["database", "environment"]

    @property
    def cache_key(self) -> str:
        # The key never leaves the process; it only separates cache entries.
        return f"{self.base_url}|{len(self.api_key or '')}"


def validate_upstream_base_url(value: str) -> str:
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        raise ValueError("Upstream base URL is required")
    if len(candidate) > MAX_BASE_URL_LENGTH:
        raise ValueError(f"Upstream base URL must be at most {MAX_BASE_URL_LENGTH} characters")
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"}:
        raise ValueError("Upstream base URL must use http or https")
    if not parts.hostname:
        raise ValueError("Upstream base URL must include a host")
    if parts.username or parts.password:
        raise ValueError("Upstream base URL must not embed credentials")
    if parts.fragment:
        raise ValueError("Upstream base URL must not contain a fragment")
    return candidate


def resolve_upstream_gateway(
    db: Session, settings: Settings
) -> UpstreamGatewayConfig | None:
    """Database configuration wins; environment values act as the default seed."""
    stored_url = db.get(SecretSetting, BASE_URL_KEY)
    if stored_url is not None:
        stored_key = db.get(SecretSetting, API_KEY_KEY)
        return UpstreamGatewayConfig(
            base_url=urlsafely_decrypt_url(stored_url),
            api_key=urlsafely_decrypt_optional(stored_key),
            source="database",
        )
    env_url = (settings.fallback_base_url or "").strip()
    if not env_url:
        return None
    env_key = (settings.fallback_api_key or "").strip() or None
    return UpstreamGatewayConfig(
        base_url=env_url.rstrip("/"),
        api_key=env_key,
        source="environment",
    )
```

Implement the two decrypt helpers against the module-level `secret_box` passed in by the caller
(FastAPI holds it on `app.state.secret_box`), so keep them as small functions taking the box:

```python
def _read_secret(box: SecretBox, row: SecretSetting | None) -> str | None:
    if row is None:
        return None
    value = box.decrypt(row.encrypted_value)
    return value or None
```

Add the storage entry point used by the API:

```python
def save_upstream_gateway(
    db: Session,
    box: SecretBox,
    *,
    base_url: str | None,
    api_key: str | None,
) -> UpstreamGatewayConfig | None:
    """Persist or clear the upstream configuration.

    `base_url=None` clears everything. `api_key=None` keeps the stored key;
    an empty string clears only the key.
    """
    stored_url = db.get(SecretSetting, BASE_URL_KEY)
    stored_key = db.get(SecretSetting, API_KEY_KEY)

    if base_url is None:
        if stored_url is not None:
            db.delete(stored_url)
        if stored_key is not None:
            db.delete(stored_key)
        return None

    normalised = validate_upstream_base_url(base_url)
    encrypted_url = box.encrypt(normalised)
    if stored_url is None:
        db.add(SecretSetting(key=BASE_URL_KEY, encrypted_value=encrypted_url))
    else:
        stored_url.encrypted_value = encrypted_url
        stored_url.updated_at = utc_now()

    if api_key is not None:
        if api_key == "":
            if stored_key is not None:
                db.delete(stored_key)
        else:
            encrypted_key = box.encrypt(api_key)
            if stored_key is None:
                db.add(SecretSetting(key=API_KEY_KEY, encrypted_value=encrypted_key))
            else:
                stored_key.encrypted_value = encrypted_key
                stored_key.updated_at = utc_now()
    return UpstreamGatewayConfig(base_url=normalised, api_key=None, source="database")
```

Follow `app/api/settings.py` for the `utc_now` import (`from app.models import utc_now`) and the
`updated_at` handling used for `SecretSetting`.

- [ ] **Step 5: Run and verify GREEN**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_upstream_gateway.py -q`
Expected: validation and normalisation tests pass.

- [ ] **Step 6: Add the database-precedence and secret tests**

Replace the placeholder test from Step 2 with real coverage using the `authenticated_client`
fixture and the `/api/gateway/upstream` endpoints from Task 3 (write both tasks, then run):

```python
def test_clearing_database_config_falls_back_to_environment(authenticated_client, settings):
    settings.fallback_base_url = "https://env.test/v1"
    authenticated_client.put(
        "/api/gateway/upstream",
        json={"base_url": "https://db.test/v1", "api_key": "db-key"},
    )
    assert authenticated_client.get("/api/gateway/upstream").json() == {
        "base_url": "https://db.test/v1",
        "api_key_configured": True,
        "source": "database",
        "enabled": True,
    }

    authenticated_client.put("/api/gateway/upstream", json={"base_url": None})

    assert authenticated_client.get("/api/gateway/upstream").json() == {
        "base_url": "https://env.test/v1",
        "api_key_configured": True,
        "source": "environment",
        "enabled": True,
    }


def test_upstream_never_returns_the_stored_key(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream",
        json={"base_url": "https://db.test/v1", "api_key": "super-secret-value"},
    )
    body = authenticated_client.get("/api/gateway/upstream").text
    assert "super-secret-value" not in body
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/upstream_gateway.py backend/app/config.py backend/tests/test_upstream_gateway.py
git commit -m "feat: add upstream gateway configuration service"
```

---

### Task 3: Upstream management API

**Files:**
- Create: `backend/app/api/upstream.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_upstream_gateway.py`

- [ ] **Step 1: Write the failing endpoint tests**

Append to `backend/tests/test_upstream_gateway.py`:

```python
import respx
from httpx import Response


def test_upstream_endpoints_require_admin(client):
    assert client.get("/api/gateway/upstream").status_code == 401
    assert client.put("/api/gateway/upstream", json={"base_url": None}).status_code == 401
    assert client.post("/api/gateway/upstream/test").status_code == 401


def test_put_rejects_an_invalid_base_url(authenticated_client):
    response = authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "ftp://bad/v1"}
    )
    assert response.status_code == 422


def test_put_requires_csrf(client, authenticated_client):
    del authenticated_client.headers["X-CSRF-Token"]
    response = client.put("/api/gateway/upstream", json={"base_url": "https://x.test/v1"})
    assert response.status_code == 403


@respx.mock
def test_test_endpoint_reports_model_count(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": "a"}, {"id": "b"}]})
    )
    body = authenticated_client.post("/api/gateway/upstream/test").json()
    assert body["status"] == "ok"
    assert body["model_count"] == 2
    assert isinstance(body["latency_ms"], int)


@respx.mock
def test_test_endpoint_reports_unavailable_without_leaking_details(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "secret-key"}
    )
    respx.get("https://up.test/v1/models").mock(return_value=Response(503, text="boom"))
    body = authenticated_client.post("/api/gateway/upstream/test").json()
    assert body["status"] == "unavailable"
    assert "secret-key" not in str(body)


def test_test_endpoint_reports_unset_when_not_configured(authenticated_client):
    assert authenticated_client.post("/api/gateway/upstream/test").json()["status"] == "unset"


def test_config_changes_are_audited(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://audit.test/v1", "api_key": "abc"}
    )
    events = authenticated_client.get("/api/audit?limit=20").json()
    actions = [event["action"] for event in events]
    assert "gateway.upstream.update" in actions
```

- [ ] **Step 2: Run and verify RED**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_upstream_gateway.py -k "admin or invalid or csrf or test_endpoint or audited" -q`
Expected: 404 for the unregistered routes.

- [ ] **Step 3: Implement the router**

Create `backend/app/api/upstream.py`:

```python
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.services.upstream_gateway import (
    MODELS_TIMEOUT_SECONDS,
    resolve_upstream_gateway,
    save_upstream_gateway,
    validate_upstream_base_url,
)

router = APIRouter(prefix="/api/gateway/upstream", tags=["upstream-gateway"])


class UpstreamUpdate(BaseModel):
    base_url: str | None = Field(default=None, max_length=600)
    api_key: str | None = Field(default=None, max_length=4096)


@router.get("")
def get_upstream(request: Request, db: DbSession, _: Admin) -> dict[str, Any]:
    config = resolve_upstream_gateway(db, request.app.state.settings)
    if config is None:
        return {
            "base_url": None,
            "api_key_configured": False,
            "source": "unset",
            "enabled": False,
        }
    return {
        "base_url": config.base_url,
        "api_key_configured": bool(config.api_key),
        "source": config.source,
        "enabled": True,
    }
```

`PUT` validates, saves, invalidates the model cache, audits, and commits:

```python
@router.put("")
def put_upstream(
    payload: UpstreamUpdate,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    try:
        if payload.base_url is not None and payload.base_url.strip():
            validate_upstream_base_url(payload.base_url)
        save_upstream_gateway(
            db,
            request.app.state.secret_box,
            base_url=payload.base_url,
            api_key=payload.api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    request.app.state.upstream_model_cache.invalidate()
    config = resolve_upstream_gateway(db, request.app.state.settings)
    record_audit(
        db,
        actor=str(admin["username"]),
        action="gateway.upstream.update",
        resource_type="gateway",
        details={
            "base_url": config.base_url if config else None,
            "api_key_configured": bool(config and config.api_key),
            "source": config.source if config else "unset",
        },
    )
    db.commit()
    return get_upstream(request, db, admin)
```

`POST /test` uses `fetch_upstream_models` from the service, maps failures to bounded detail, and audits:

```python
@router.post("/test")
def test_upstream(request: Request, db: DbSession, admin: CsrfAdmin) -> dict[str, Any]:
    started = time.perf_counter()
    config = resolve_upstream_gateway(db, request.app.state.settings)
    if config is None:
        return {"status": "unset", "latency_ms": 0, "model_count": 0, "detail": None}
    try:
        models = fetch_upstream_models(config)
    except httpx.HTTPError as exc:
        result: dict[str, Any] = {
            "status": "unavailable",
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "model_count": 0,
            "detail": bounded_probe_detail(exc),
        }
    else:
        result = {
            "status": "ok",
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "model_count": len(models),
            "detail": None,
        }
    record_audit(
        db,
        actor=str(admin["username"]),
        action="gateway.upstream.test",
        resource_type="gateway",
        outcome="success" if result["status"] == "ok" else "failed",
        details={"status": result["status"], "model_count": result["model_count"]},
    )
    db.commit()
    return result
```

`bounded_probe_detail` returns the exception class name plus at most 200 characters, and never the
URL query string or the API key.

- [ ] **Step 4: Register the router and the cache in main.py**

```python
from app.api.upstream import router as upstream_router
from app.services.upstream_gateway import UpstreamModelCache
...
app.include_router(upstream_router)
```

Construct the cache next to the other `app.state` singletons:

```python
app.state.upstream_model_cache = UpstreamModelCache(
    ttl_seconds=settings.upstream_models_cache_seconds
)
```

- [ ] **Step 5: Run and verify GREEN**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_upstream_gateway.py -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/upstream.py backend/app/main.py backend/tests/test_upstream_gateway.py
git commit -m "feat: expose upstream gateway management API"
```

---

### Task 4: Discover upstream models through /v1/models

**Files:**
- Modify: `backend/app/services/upstream_gateway.py` (`fetch_upstream_models`, `UpstreamModelCache`)
- Modify: `backend/app/api/gateway.py`
- Test: `backend/tests/test_upstream_gateway.py`

- [ ] **Step 1: Write the failing merge tests**

```python
@respx.mock
def test_models_merge_upstream_entries(authenticated_client, gateway_key, healthy_deployment):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": "remote-a"}, {"id": "remote-b"}]})
    )
    data = authenticated_client.get(
        "/v1/models", headers={"Authorization": f"Bearer {gateway_key}"}
    ).json()
    ids = [item["id"] for item in data["data"]]
    upstream = [item for item in data["data"] if item.get("owned_by") == "upstream"]
    assert {"remote-a", "remote-b"} <= set(ids)
    assert all(item["dgx_source"] == "upstream" for item in upstream)
    assert all(item["capabilities"] == [] for item in upstream)
    assert all(item["context_window"] is None for item in upstream)


@respx.mock
def test_local_route_wins_over_same_named_upstream_model(authenticated_client, gateway_key, healthy_deployment):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": healthy_deployment.api_model_name}]})
    )
    data = authenticated_client.get(
        "/v1/models", headers={"Authorization": f"Bearer {gateway_key}"}
    ).json()
    matches = [item for item in data["data"] if item["id"] == healthy_deployment.api_model_name]
    assert len(matches) == 1
    assert matches[0]["dgx_source"] == "local"


@respx.mock
def test_upstream_failure_keeps_local_models(authenticated_client, gateway_key, healthy_deployment):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(return_value=Response(503))
    data = authenticated_client.get(
        "/v1/models", headers={"Authorization": f"Bearer {gateway_key}"}
    ).json()
    assert data["data"]
    assert data["upstream"]["status"] == "unavailable"


def test_models_report_unset_upstream(authenticated_client, gateway_key, healthy_deployment):
    data = authenticated_client.get(
        "/v1/models", headers={"Authorization": f"Bearer {gateway_key}"}
    ).json()
    assert data["upstream"] == {"status": "unset", "detail": None}


@respx.mock
def test_upstream_models_are_cached(authenticated_client, gateway_key, healthy_deployment):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    route = respx.get("https://up.test/v1/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": "remote-a"}]})
    )
    headers = {"Authorization": f"Bearer {gateway_key}"}
    authenticated_client.get("/v1/models", headers=headers)
    authenticated_client.get("/v1/models", headers=headers)
    assert route.call_count == 1
```

Add a `gateway_key` fixture and a `healthy_deployment` helper in `conftest.py`-style inside this
file (create the key through `/api/keys` and insert a healthy `Deployment` row directly through the
database session, mirroring `backend/tests/test_gateway.py`).

- [ ] **Step 2: Run and verify RED**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_upstream_gateway.py -k "merge or wins or keeps_local or unset_upstream or cached" -q`
Expected: FAIL — `KeyError: 'upstream'`.

- [ ] **Step 3: Implement fetch and cache in the service**

Append to `backend/app/services/upstream_gateway.py`:

```python
def fetch_upstream_models(config: UpstreamGatewayConfig) -> list[dict[str, Any]]:
    headers = {"authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    with httpx.Client(timeout=MODELS_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.get(f"{config.base_url}/models", headers=headers)
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("Upstream /models response did not contain a data list")
    return [item for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]


class UpstreamModelCache:
    """Bounded TTL cache for the upstream model list."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._lock = Lock()

    def get(self, key: str) -> list[dict[str, Any]] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, models = entry
            if time.monotonic() - stored_at > self._ttl:
                self._entries.pop(key, None)
                return None
            return models

    def set(self, key: str, models: list[dict[str, Any]]) -> None:
        with self._lock:
            self._entries = {key: (time.monotonic(), models)}

    def invalidate(self) -> None:
        with self._lock:
            self._entries.clear()


def upstream_model_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Map an upstream /models entry to the local contract without inventing values."""
    return {
        "id": raw["id"],
        "object": "model",
        "created": raw.get("created") if isinstance(raw.get("created"), int) else 0,
        "owned_by": "upstream",
        "dgx_source": "upstream",
        "capabilities": [],
        "input_modalities": [],
        "context_window": None,
        "configured_context_length": None,
        "runtime_token_capacity": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "max_concurrency": None,
        "runtime": None,
        "performance": {"status": "unknown", "tokens_per_second": None},
    }
```

- [ ] **Step 4: Merge into the models endpoint**

In `backend/app/api/gateway.py`, make both endpoints async and append upstream entries after the
local loop in `openai_models`:

```python
    upstream_status: dict[str, Any] = {"status": "unset", "detail": None}
    config = resolve_upstream_gateway(db, request.app.state.settings)
    if config is not None:
        cache: UpstreamModelCache = request.app.state.upstream_model_cache
        cached = cache.get(config.cache_key)
        if cached is None:
            try:
                cached = fetch_upstream_models(config)
            except (httpx.HTTPError, ValueError):
                cached = []
                upstream_status = {"status": "unavailable", "detail": None}
            else:
                cache.set(config.cache_key, cached)
        if upstream_status["status"] != "unavailable":
            upstream_status = {"status": "ok", "detail": None}
        for raw in cached:
            route_name = raw["id"]
            if route_name in routes:
                continue  # local routes always win
            routes[route_name] = upstream_model_entry(raw)
    return {"object": "list", "data": list(routes.values()), "upstream": upstream_status}
```

Change the signatures to accept `request: Request` and drop the unused `_`/`key` parameter name
collisions carefully: keep `GatewayKey` for authentication, and pass `request` for `app.state`.
Update `openai_model` to `await openai_models(...)`.

- [ ] **Step 5: Run and verify GREEN**

Run: `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests/test_upstream_gateway.py backend/tests/test_gateway.py -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/upstream_gateway.py backend/app/api/gateway.py backend/tests/test_upstream_gateway.py
git commit -m "feat: expose upstream models through the OpenAI models endpoint"
```

---

### Task 5: Frontend upstream section

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/pages/GatewayPage.tsx`
- Create: `frontend/src/pages/GatewayPage.test.tsx`

- [ ] **Step 1: Add the contracts**

```typescript
export interface UpstreamGateway {
  base_url: string | null
  api_key_configured: boolean
  source: 'database' | 'environment' | 'unset'
  enabled: boolean
}

export interface UpstreamTestResult {
  status: 'ok' | 'unavailable' | 'unset' | 'error'
  latency_ms: number
  model_count: number
  detail: string | null
}
```

- [ ] **Step 2: Write the failing page tests**

Create `frontend/src/pages/GatewayPage.test.tsx` following the existing page-test pattern
(`QueryClientProvider` + `MemoryRouter` + mocked `fetch`): assert that an unset upstream renders a
配置 entry point, a configured upstream renders the base URL plus a 环境变量/数据库 source label, a
successful test renders the model count, a failed test renders the reason, and that saving an
invalid `base_url` blocks submission.

- [ ] **Step 3: Run and verify RED**

Run: `& 'C:\Program Files\nodejs\corepack.cmd' pnpm test -- GatewayPage`
Expected: FAIL — the upstream section is absent.

- [ ] **Step 4: Implement the section**

Add to `GatewayPage` below the summary block: a `useQuery` for `/api/gateway/upstream`, a
`useMutation` for the `PUT`, a `useMutation` for `POST /api/gateway/upstream/test`, a config
`Modal` with `base_url` (required, URL rule) and `api_key` (`Input.Password`, placeholder
"留空表示不修改"), a `Popconfirm`-guarded clear action, and a status line that renders the test
result. Reuse `settings-section`, `Descriptions`, `Tag` and the existing button styles; never render
the API key value.

- [ ] **Step 5: Run and verify GREEN**

Run: `& 'C:\Program Files\nodejs\corepack.cmd' pnpm test` then `pnpm lint` and `pnpm build`
Expected: all tests pass, no lint errors, production build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api/types.ts frontend/src/pages/GatewayPage.tsx frontend/src/pages/GatewayPage.test.tsx
git commit -m "feat: manage the upstream gateway from the panel"
```

---

### Task 6: Documentation

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/API.md`

- [ ] **Step 1: Update `.env.example`**

State that the pair now seeds the default and can be overridden from the panel without a restart.

- [ ] **Step 2: Update `README.md`**

Add the upstream behaviour to the "OpenAI 兼容 API" table (`/v1/models` includes upstream models
tagged `dgx_source: upstream` when configured) and a row in the configuration table explaining that
the upstream gateway is normally configured in the panel and the env pair is only the default.

- [ ] **Step 3: Update `docs/API.md`**

Document `GET/PUT /api/gateway/upstream`, `POST /api/gateway/upstream/test`, and the new
`upstream` object plus `dgx_source` field on `/v1/models`.

- [ ] **Step 4: Commit**

```bash
git add .env.example README.md docs/API.md
git commit -m "docs: document upstream gateway management and discovery"
```

---

## Final verification

- [ ] Run the full backend suite and compare against the baseline of `1342 passed, 3 failed, 51 skipped`:
  `& 'C:\Users\chenw\AppData\Local\Programs\Python\Python312\python.exe' -m pytest backend/tests -q`
- [ ] Run `ruff check backend/app backend/tests` (or the repo-pinned `ruff` in CI).
- [ ] Run the full frontend suite from `frontend/`: `pnpm test`, `pnpm lint`, `pnpm build`.
- [ ] Confirm no secret leaks: `GET /api/gateway/upstream` and `POST /api/gateway/upstream/test`
  must never contain the configured API key.
- [ ] On the DGX Spark host, deploy the manager and verify end to end: configure a real upstream,
  confirm `/v1/models` lists both local and upstream entries with correct `dgx_source`, stream a
  chat completion for an upstream model, and confirm `/api/gateway/stats` token totals increase.