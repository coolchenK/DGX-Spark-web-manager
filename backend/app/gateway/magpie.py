"""Magpie's public API, with a non-recursive bridge to managed local models."""

import hmac
import json

import httpx
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, StreamingResponse

from app.config import Settings

LOCAL_PREFIX = "/_magpie/local"
GATEWAY_URL = "http://127.0.0.1:3425"
FORWARD_HEADERS = {
    "authorization",
    "x-api-key",
    "x-goog-api-key",
    "anthropic-version",
    "anthropic-beta",
    "openai-beta",
    "content-type",
    "accept",
    "user-agent",
    # Magpie otherwise hashes the first user message, shared by some agent tasks.
    "x-magpie-session",
    "x-opencode-session",
    "x-session-affinity",
    "x-session-id",
    "session_id",
    "session-id",
    "x-claude-code-session-id",
    "x-conversation-id",
    "x-deepseek-harness-session-id",
}


def _live_local_model(request: Request, model: str) -> bool:
    from app.api.gateway import _healthy_gateway_deployments, deployment_route_name

    with request.app.state.database.session_factory() as db:
        return any(
            deployment_route_name(item) == model for item in _healthy_gateway_deployments(db)
        )


class MagpieGatewayMiddleware:
    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.settings.magpie_enabled:
            return await self.app(scope, receive, send)
        path = scope["path"]
        request = Request(scope, receive)
        authorization = request.headers.get("authorization", "")
        if path.startswith(LOCAL_PREFIX + "/"):
            # Only Magpie has this random credential. The marker cannot be set
            # by a client header, and prevents discovery/fallback recursion.
            expected = self.settings.magpie_bridge_key
            supplied = authorization.removeprefix("Bearer ")
            if not expected or not hmac.compare_digest(supplied, expected):
                return await JSONResponse({"error": "Invalid bridge key"}, 401)(
                    scope, receive, send
                )
            rewritten = path.removeprefix(LOCAL_PREFIX)
            if not rewritten.startswith("/v1/"):
                return await JSONResponse({"error": "Not found"}, 404)(scope, receive, send)
            scope = dict(scope, path=rewritten, raw_path=rewritten.encode())
            scope["state"] = {**scope.get("state", {}), "magpie_local_only": True}
            return await self.app(scope, receive, send)
        if not path.startswith(("/v1/", "/v1beta/")):
            return await self.app(scope, receive, send)
        legacy = authorization.startswith("Bearer dgx_")
        if legacy:
            if path not in {"/v1/chat/completions", "/v1/completions", "/v1/responses"}:
                return await self.app(scope, receive, send)
            from app.api.gateway import GatewayAuthError, require_gateway_key

            def check_legacy_key():
                with scope["app"].state.database.session_factory() as db:
                    require_gateway_key(request, db, authorization)

            try:
                await run_in_threadpool(check_legacy_key)
            except GatewayAuthError:
                return await JSONResponse(
                    {"error": {"message": "Invalid API key", "type": "authentication_error"}}, 401
                )(scope, receive, send)
            if not self.settings.magpie_gateway_key:
                return await JSONResponse(
                    {"error": {"message": "Magpie gateway is not configured"}}, 503
                )(scope, receive, send)

        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_HEADERS}
        if legacy:
            headers["authorization"] = "Bearer " + self.settings.magpie_gateway_key
            headers.pop("x-api-key", None)
        # Magpie otherwise trusts loopback requests, including invalid keys.
        # Always mark this proxy hop as remote and keep MAGPIE_TRUST_PROXY off.
        headers["x-forwarded-for"] = "192.0.2.1"
        url = httpx.URL(GATEWAY_URL + path, query=scope.get("query_string", b""))
        timeout = httpx.Timeout(connect=10, read=600, write=600, pool=10)
        started = False

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            content = request.stream()
            if request.method == "POST" and path in {
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/responses",
            }:
                raw_body = await request.body()
                content = raw_body
                try:
                    body = json.loads(raw_body)
                except ValueError:
                    body = None
                model = body.get("model") if isinstance(body, dict) else None
                if isinstance(model, str) and await run_in_threadpool(
                    _live_local_model, request, model
                ):
                    # Explicit provider IDs work before Magpie refreshes its catalog.
                    body["model"] = "dgx-local/" + model
                    content = json.dumps(body).encode()
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                async with client.stream(
                    request.method, url, headers=headers, content=content
                ) as upstream:
                    response_headers = {
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower() in {"content-type", "retry-after", "x-request-id"}
                    }
                    response_headers.update(
                        {"cache-control": "no-store", "x-accel-buffering": "no"}
                    )
                    response = StreamingResponse(
                        upstream.aiter_bytes(),
                        status_code=upstream.status_code,
                        headers=response_headers,
                    )
                    await response(scope, receive, tracked_send)
        except ClientDisconnect:
            return
        except httpx.HTTPError:
            if started:
                raise
            # Do not leak URLs, credentials, or provider response bodies.
            response = JSONResponse(
                {"error": {"message": "Magpie gateway is unavailable", "type": "api_error"}}, 502
            )
            await response(scope, receive, send)
