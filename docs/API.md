# API Reference

## Authentication and CSRF

Management endpoints use the `dgx_session` HttpOnly cookie. Login also returns a CSRF value and
sets the `dgx_csrf` cookie:

```http
POST /api/auth/login
Content-Type: application/json

{"username":"admin","password":"..."}
```

Send the returned value as `X-CSRF-Token` on state-changing management requests. The deployment
recommendation endpoint also requires it because a request can call a configured third-party AI
provider and records an audit event. `POST /api/deployments/preview` is read-only and requires the
administrator session but does not enforce the CSRF dependency. The web client sends the header
whenever a CSRF value is available.

OpenAI endpoints do not use the administrator session. They require a separately generated gateway
key:

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
| POST | `/api/deployments/recommendations` | Recommend settings for a model/runtime/image tuple |
| POST | `/api/deployments/preview` | Revalidate and preview a deployment spec |
| POST | `/api/deployments` | Create a deployment task |
| PATCH | `/api/deployments/{id}` | Replace a managed deployment from a validated spec |
| POST | `/api/deployments/{id}/{start|stop|restart|delete}` | Create a lifecycle task |
| GET | `/api/deployments/{id}/logs` | Sanitized container log tail |
| GET | `/api/tasks` | Persistent task history |
| POST | `/api/tasks/{id}/{pause|resume}` | Task control |
| DELETE | `/api/tasks/{id}` | Cancel task |
| GET/POST/PATCH/DELETE | `/api/providers` | Encrypted online AI provider configuration |
| POST | `/api/providers/{id}/test` | Provider connectivity test |
| GET/POST | `/api/diagnostics` | Plans and new diagnosis |
| POST | `/api/diagnostics/{id}/{approve|reject}` | Human decision |
| GET | `/api/keys` | Gateway key metadata |
| POST | `/api/keys` | Create and reveal a gateway key once |
| DELETE | `/api/keys/{id}` | Revoke a gateway key |
| GET | `/api/audit` | Audit history |
| GET | `/api/settings` | Non-secret manager configuration |
| PATCH | `/api/settings/huggingface` | Set or clear the encrypted HF token |

### Hugging Face Spark compatibility

`GET /api/huggingface/search?query=Qwen&limit=20` evaluates up to 50 Hub candidates for
DGX Spark suitability before applying the requested result limit. NVFP4 receives the strongest
positive signal. Results are ordered by compatibility level, then score; Hub relevance remains
the tie-breaker for candidates with the same level and score.

Each search result includes an additive compatibility object:

```json
{
  "id": "unsloth/Qwen3.8-27B-NVFP4",
  "spark_compatibility": {
    "level": "recommended",
    "score": 180,
    "reasons": ["NVFP4 quantization", "compressed weights", "Safetensors weights"]
  }
}
```

`level` is `recommended`, `compatible`, or `review`. This metadata controls search order and
presentation only; it does not bypass download, runtime, image, quantization, or resource checks.

## Deployment Recommendations

`POST /api/deployments/recommendations` accepts the stable selection tuple. Unknown fields are
rejected. `provider_id` is optional; when omitted, recommendations remain deterministic and no
online AI provider is called.

```json
{
  "model_id": "model-asset-id",
  "runtime": "vllm",
  "image": "vllm/vllm-openai:v0.27.1",
  "provider_id": "provider-id"
}
```

Use `?refresh_ai=true` to bypass a matching AI cache entry. Model evidence, runtime capability,
device memory, resource estimation, and Draft Model classification are still evaluated on every
request. A missing provider returns `404`; a disabled provider or one whose last test failed returns
`409`.

The response has this shape (values are illustrative):

```json
{
  "status": "complete",
  "generated_at": "2026-08-16T00:00:00Z",
  "model_id": "model-asset-id",
  "runtime": "vllm",
  "image_digest": "sha256:...",
  "evidence_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "fields": {
    "context_length": {
      "value": 32768,
      "source": "model_card",
      "confidence": "high",
      "reason": "Model card explicitly sets context_length",
      "warning": null
    },
    "quantization": {
      "value": "modelopt_fp4",
      "source": "local_config",
      "confidence": "high",
      "reason": "Local model asset metadata declares quantization",
      "warning": null
    }
  },
  "generation_defaults": {
    "temperature": {
      "value": 0.6,
      "source": "ai",
      "confidence": "medium",
      "reason": "AI filled a bounded generation recommendation",
      "warning": null
    }
  },
  "resource_snapshot": {
    "total_bytes": 130595860480,
    "available_bytes": 100000000000,
    "reserved_bytes": 13059586048
  },
  "resource_estimate": {
    "total_bytes": 130595860480,
    "available_bytes": 100000000000,
    "reserved_bytes": 13059586048,
    "weight_bytes": 45000000000,
    "draft_weight_bytes": 0,
    "kv_cache_bytes": 8000000000,
    "runtime_overhead_bytes": 2147483648,
    "required_bytes": 55147483648,
    "decision": "ok",
    "confidence": "high",
    "reasons": []
  },
  "runtime_capabilities": {
    "runtime": "vllm",
    "image": "vllm/vllm-openai:v0.27.1",
    "image_digest": "sha256:...",
    "source": "probe",
    "generation_defaults": [
      "temperature",
      "top_p",
      "top_k",
      "min_p",
      "repetition_penalty",
      "presence_penalty",
      "frequency_penalty",
      "max_tokens",
      "stop"
    ],
    "quantization_methods": [
      "auto",
      "awq",
      "gptq",
      "fp8",
      "bitsandbytes",
      "marlin",
      "gguf",
      "modelopt",
      "modelopt_fp4",
      "nvfp4_online",
      "compressed-tensors"
    ],
    "quantization_mapping": {"nvfp4": "modelopt_fp4"},
    "speculative_methods": ["draft_model", "eagle3", "mtp"],
    "method_mapping": {"draft_model": "draft_model", "eagle3": "eagle3", "mtp": "mtp"},
    "speculative_transport": "json",
    "warnings": []
  },
  "draft_candidates": [],
  "warnings": []
}
```

`status` is:

- `complete`: every critical deployment field is resolved and no AI fallback failed.
- `partial`: safe values are returned, but unresolved fields or an unavailable/invalid AI result
  still require administrator review.
- `unavailable`: a required model, capability, evidence, resource snapshot, or physical resource
  check could not be verified. The response can retain partial fields and warnings for inspection.

Every recommended value has a `source`: `model_card`, `local_config`, `runtime_default`,
`device_rule`, or `ai`. `draft_candidates` use `compatible`, `review`, or `incompatible` status.
A `review` candidate requires `speculative.manual_review_acknowledged=true`; an `incompatible`
candidate cannot pass preview. Resource `warning` requires `resource_warning_acknowledged=true`,
while `blocked` cannot be acknowledged through.

A Draft candidate contains the local asset identity, detected method, classification reasons, raw
size, and the combined estimate when it could be calculated:

```json
{
  "model_id": "draft-model-asset-id",
  "name": "qwen-eagle3-draft",
  "repository_id": "org/qwen-eagle3-draft",
  "method": "eagle3",
  "status": "review",
  "reasons": ["Auxiliary Draft Model target pairing evidence is missing"],
  "size_bytes": 2147483648,
  "estimated_total_bytes": 57365372928
}
```

## Deployment Spec, Preview, Create, Edit, and Clone

Preview, create, and edit accept the same strict `DeploymentSpec` JSON body:

```json
{
  "name": "qwen-nvfp4",
  "model_id": "model-asset-id",
  "model_path": "/models/qwen-nvfp4",
  "api_model_name": "qwen-nvfp4",
  "route_alias": "qwen-production",
  "runtime": "vllm",
  "image": "vllm/vllm-openai:v0.27.1",
  "port": 8010,
  "context_length": 32768,
  "memory_fraction": 0.8,
  "max_concurrency": 4,
  "max_batched_tokens": 8192,
  "quantization": "modelopt_fp4",
  "trust_remote_code": false,
  "generation_defaults": {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 2048,
    "stop": ["<|end|>"]
  },
  "speculative": {
    "draft_model_id": "draft-model-asset-id",
    "method": "eagle3",
    "num_speculative_tokens": 5,
    "manual_review_acknowledged": false
  },
  "recommendation": {
    "generated_at": "2026-08-16T00:00:00Z",
    "evidence_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "provider_id": "provider-id",
    "resource_snapshot": {
      "total_bytes": 130595860480,
      "available_bytes": 100000000000,
      "reserved_bytes": 13059586048
    },
    "modified_fields": ["generation_defaults.temperature"],
    "sources": {
      "context_length": "model_card",
      "quantization": "local_config",
      "generation_defaults.temperature": "ai"
    }
  },
  "resource_warning_acknowledged": false
}
```

`recommendation` is provenance, not an instruction to call AI during deployment. `sources` records
the origin of retained recommended fields. `modified_fields` records recommendation-managed fields
the administrator changed after application; both use dotted generation paths such as
`generation_defaults.temperature`. The backend persists JSON-safe provenance with the task/spec.

For SGLang, replace `num_speculative_tokens` with all three grouped values `num_steps`,
`eagle_top_k`, and `num_draft_tokens`, or omit all three. These tuning groups are mutually exclusive
by runtime.

After login, these commands exercise recommendation, preview, and create using JSON files containing
the request bodies shown in this reference:

```bash
BASE_URL=http://dgx-spark.local:3000

curl -sS -c cookies.txt \
  -H 'Content-Type: application/json' \
  --data-binary @login.json \
  "$BASE_URL/api/auth/login" > login-response.json
CSRF=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["csrf_token"])' \
  < login-response.json)

curl -sS -b cookies.txt \
  -H "X-CSRF-Token: $CSRF" \
  -H 'Content-Type: application/json' \
  --data-binary @recommendation-request.json \
  "$BASE_URL/api/deployments/recommendations?refresh_ai=true"

curl -sS -b cookies.txt \
  -H 'Content-Type: application/json' \
  --data-binary @deployment.json \
  "$BASE_URL/api/deployments/preview"

curl -sS -b cookies.txt \
  -H "X-CSRF-Token: $CSRF" \
  -H 'Content-Type: application/json' \
  --data-binary @deployment.json \
  "$BASE_URL/api/deployments"
```

Preview returns the normalized `spec`, command, mounts, `runtime_capabilities`, recomputed
`resource_estimate`, selected `draft_candidate`, `generation_defaults`, `speculative`,
`recommendation`, warnings, and `spec_fingerprint`. The panel submits the exact payload snapshot
that produced the displayed preview. Any subsequent form or tuple change invalidates that preview.

For an edit, pass the record ID as a preview query parameter so route/name checks exclude that
record, then use the same ID in the PATCH path:

```text
POST  /api/deployments/preview?deployment_id=<deployment-id>
PATCH /api/deployments/<deployment-id>
```

`deployment_id` is not a field in `DeploymentSpec`. Create uses `POST /api/deployments`. Clone is a
panel workflow that starts from an existing saved spec, requires a new name/API model name/port,
previews it, then uses the create endpoint. Create and edit return a persistent task with HTTP 202.

Preview and submit both repeat allowlist, path, image, capability, Draft compatibility, shared-route,
and resource checks. An edit uses a health-gated container replacement and rollback path.

## OpenAI-Compatible Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings` for deployments advertising `embedding`

`GET /v1/models` lists only running, healthy deployments. The gateway replaces the requested route
name with the selected deployment's upstream `api_model_name`. SSE bytes are relayed without
buffering, upstream status/content types are preserved, and manager-generated failures use the
standard OpenAI `error` object.

Set the optional `route_alias` to the same value on multiple instances to expose one gateway model
name. Healthy instances are selected round-robin. `/v1/models` reports `instances` and only the
capabilities shared by every instance. Preview rejects members of one effective route when their
normalized `generation_defaults` differ.

For `/v1/chat/completions` and `/v1/completions`, the gateway fills only missing saved defaults that
also appear in the deployment's saved runtime-capability snapshot. Explicit request values,
including `0`, `false`, and an empty `stop`, win. `max_completion_tokens` also prevents the saved
`max_tokens` default from being inserted. Invalid, unknown, or unsupported saved values are ignored;
embeddings are unchanged. Applied field names, not their values or prompts, are recorded under the
`gateway.defaults.apply` audit action.
