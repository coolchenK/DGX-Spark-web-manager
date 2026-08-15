# API Reference

## Authentication

Management endpoints use the `dgx_session` HttpOnly cookie. Login returns a CSRF value; send it as `X-CSRF-Token` on POST, PATCH, and DELETE requests.

```http
POST /api/auth/login
Content-Type: application/json

{"username":"admin","password":"..."}
```

OpenAI endpoints use a separately generated gateway key:

```http
Authorization: Bearer dgx_...
```

## Management Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Service and database liveness |
| GET | `/api/system` | Current host and GPU snapshot |
| GET | `/api/models` | Registered model assets |
| GET | `/api/deployments` | Discovered and managed deployments |
| POST | `/api/discovery/scan` | Re-scan Docker and model roots |
| GET | `/api/huggingface/search?query=` | Search Hub models |
| GET | `/api/huggingface/models/{repo}` | Repository files and metadata |
| POST | `/api/huggingface/downloads` | Create persistent download task |
| POST | `/api/deployments/preview` | Validate and preview deployment |
| POST | `/api/deployments` | Create deployment task |
| POST | `/api/deployments/{id}/{start|stop|restart|delete}` | Lifecycle task |
| GET | `/api/deployments/{id}/logs` | Sanitized container log tail |
| GET | `/api/tasks` | Persistent task history |
| POST | `/api/tasks/{id}/{pause|resume}` | Task control |
| DELETE | `/api/tasks/{id}` | Cancel task |
| GET/POST/PATCH/DELETE | `/api/providers` | Encrypted online AI providers |
| POST | `/api/providers/{id}/test` | Provider connectivity test |
| GET/POST | `/api/diagnostics` | Plans and new diagnosis |
| POST | `/api/diagnostics/{id}/{approve|reject}` | Human decision |
| GET | `/api/keys` | Gateway key metadata |
| POST | `/api/keys` | Create and reveal a key once |
| DELETE | `/api/keys/{id}` | Revoke key |
| GET | `/api/audit` | Audit history |
| GET | `/api/settings` | Non-secret manager configuration |
| PATCH | `/api/settings/huggingface` | Set or clear encrypted HF token |

## OpenAI-Compatible Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings` for deployments advertising `embedding`

The gateway replaces the request model only with the selected deployment's upstream model name. SSE bytes are relayed without buffering. Upstream status/content types are preserved. Manager-generated failures use the standard OpenAI `error` object.

Set the optional deployment `route_alias` to the same value on multiple instances to expose one
gateway model name. Healthy instances are selected round-robin; `/v1/models` reports the instance
count and only capabilities shared by every instance.
