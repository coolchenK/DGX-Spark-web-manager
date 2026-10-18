import ipaddress
import re
from urllib.parse import quote, urlencode, urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from app.dependencies import Admin

router = APIRouter(tags=["magpie"])
WEB_URL = "http://127.0.0.1:3430"
WEB_PREFIX = "/magpie/"
_SCRIPT_URL = re.compile(r"""(\bfetch\(\s*|\.(?:src|href)\s*=\s*)(["'`])/api/""")


def _web_key(request: Request) -> str:
    settings = request.app.state.settings
    key = settings.magpie_web_key
    if not settings.magpie_enabled or not re.fullmatch(r"[A-Za-z0-9._~-]{16,}", key):
        raise HTTPException(503, "Magpie 服务尚未配置")
    return key


def _check_origin(request: Request) -> None:
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "Magpie 管理仅允许同源访问")
    if request.method in {"GET", "HEAD"}:
        return
    scheme, host = request.url.scheme, request.headers.get("host", "")
    try:
        local_proxy = bool(request.client and ipaddress.ip_address(request.client.host).is_loopback)
    except ValueError:
        local_proxy = False
    # Only the local reverse proxy may describe the browser's public origin.
    if local_proxy:
        scheme = request.headers.get("x-forwarded-proto", scheme)
        host = request.headers.get("x-forwarded-host", host)
    if scheme not in {"http", "https"} or request.headers.get("origin") != f"{scheme}://{host}":
        raise HTTPException(403, "Magpie 管理操作需要同源 Origin")


def _rewrite_script(content: bytes) -> bytes:
    # Rewrite request/DOM URLs, never provider API path comparisons or examples.
    return _SCRIPT_URL.sub(r"\1\2/magpie/api/", content.decode("utf-8")).encode("utf-8")


@router.get("/api/magpie/status")
async def magpie_status(request: Request, _: Admin) -> dict[str, str]:
    try:
        key = _web_key(request)
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            response = await client.get(
                f"{WEB_URL}/boot.js", headers={"Cookie": f"magpie_web_3430={key}"}
            )
        if response.status_code == 200 and "window.bootPrefs" in response.text:
            return {"status": "ok"}
    except (HTTPException, httpx.HTTPError):
        pass
    return {"status": "unavailable", "detail": "Magpie 管理服务暂不可用，请稍后重试"}


@router.get("/magpie", include_in_schema=False)
async def magpie_redirect(request: Request, _: Admin) -> RedirectResponse:
    _web_key(request)
    return RedirectResponse(WEB_PREFIX, status_code=307)


@router.api_route(
    "/magpie/{path:path}",
    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
async def magpie_web(path: str, request: Request, _: Admin) -> Response:
    key = _web_key(request)
    _check_origin(request)
    if "\\" in path or any(part in {".", ".."} for part in path.split("/")):
        raise HTTPException(400, "Invalid Magpie path")
    # Sandboxed third-party plugin pages have no admin cookie; do not weaken panel auth for them.
    if path.startswith("_magpie/"):
        raise HTTPException(404, "此入口不提供 Magpie 独立登录或第三方插件页面")
    query = urlencode([(k, v) for k, v in request.query_params.multi_items() if k != "k"])
    url = f"{WEB_URL}/{quote(path, safe='/')}" + (f"?{query}" if query else "")
    headers = {"Cookie": f"magpie_web_3430={key}", "Accept-Encoding": "identity"}
    for name in ("accept", "content-type", "range"):
        if value := request.headers.get(name):
            headers[name] = value
    client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0), trust_env=False)
    try:
        upstream = await client.send(
            client.build_request(request.method, url, headers=headers, content=request.stream()),
            stream=True,
        )
    except (httpx.HTTPError, ClientDisconnect) as exc:
        await client.aclose()
        if isinstance(exc, ClientDisconnect):
            return Response(status_code=499)
        raise HTTPException(502, "Magpie 管理服务暂不可用") from exc
    outgoing = {
        "Cache-Control": "no-store",
        "X-Frame-Options": "SAMEORIGIN",
        "Content-Security-Policy": "frame-ancestors 'self'",
        "X-Content-Type-Options": "nosniff",
        "X-Accel-Buffering": "no",
        "Referrer-Policy": "same-origin",
    }
    for name in ("content-type", "content-disposition", "content-range", "accept-ranges"):
        if value := upstream.headers.get(name):
            outgoing[name] = value
    if location := upstream.headers.get("location"):
        target = urlsplit(location)
        await upstream.aclose()
        await client.aclose()
        if target.netloc or target.scheme or not target.path.startswith("/"):
            raise HTTPException(502, "Magpie 返回了不支持的跳转")
        outgoing["Location"] = WEB_PREFIX + target.path.lstrip("/")
        if target.query:
            outgoing["Location"] += "?" + target.query
        return Response(status_code=upstream.status_code, headers=outgoing)
    if upstream.status_code in {401, 403}:
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(502, "Magpie 内部管理认证失败")
    if path.endswith(".js") and "javascript" in upstream.headers.get("content-type", ""):
        try:
            content = _rewrite_script(await upstream.aread())
        except (httpx.HTTPError, UnicodeError) as exc:
            raise HTTPException(502, "无法读取 Magpie 页面资源") from exc
        finally:
            await upstream.aclose()
            await client.aclose()
        return Response(content, status_code=upstream.status_code, headers=outgoing)

    async def relay():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=outgoing)
