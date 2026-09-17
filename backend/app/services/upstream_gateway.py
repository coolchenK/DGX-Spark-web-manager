"""Upstream gateway configuration helpers.

The manager forwards requests for models it does not host to an
OpenAI-compatible upstream. Operators paste the same style of base URL the
README documents for this manager (`http://<host>:3000/v1`), so the configured
value may or may not carry the trailing `/v1`. Both forms must resolve to
exactly one `/v1` prefix on the wire.
"""

from __future__ import annotations

V1_SUFFIX = "/v1"


def upstream_api_root(base_url: str) -> str:
    """Return the base URL without a trailing `/v1`.

    Endpoints supplied by the gateway already start with `/v1`, so the root is
    what they are concatenated onto.
    """
    root = (base_url or "").strip().rstrip("/")
    if root.endswith(V1_SUFFIX):
        root = root[: -len(V1_SUFFIX)].rstrip("/")
    return root


def upstream_request_url(base_url: str, endpoint: str) -> str:
    """Compose an upstream URL with exactly one `/v1` prefix."""
    root = upstream_api_root(base_url)
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{root}{endpoint}"
