# Hugging Face DGX Spark Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prioritize NVFP4 and other DGX Spark-friendly Hugging Face models while keeping all search results visible and explaining each compatibility decision in the Ant Design UI.

**Architecture:** The Hugging Face service enriches a 50-result candidate pool using deterministic repository metadata, then stably sorts by compatibility level, score, and Hub order. The frontend consumes the new compatibility object, maps levels to consistent Ant Design tags, and renders short reasons without duplicating ranking logic.

**Tech Stack:** Python 3.12, FastAPI, huggingface_hub, pytest, React 19, TypeScript 6, Ant Design 5, Vitest, Docker Compose

---

## File Structure

- Modify `backend/app/tasks/huggingface.py`: compatibility scoring, model-size risk parsing, result enrichment, and stable sorting.
- Modify `backend/tests/test_huggingface.py`: scoring and search-order regression tests.
- Modify `frontend/src/api/types.ts`: typed compatibility response contract.
- Modify `frontend/src/utils/huggingface.ts`: compatibility level presentation mapping.
- Modify `frontend/src/utils/huggingface.test.ts`: presentation mapping tests.
- Modify `frontend/src/pages/HuggingFacePage.tsx`: compatibility tags and reasons.
- Create `frontend/src/pages/HuggingFacePage.test.tsx`: search-result rendering and stable class regression test.
- Modify `frontend/src/styles.css`: stable responsive layout and theme-safe accessible contrast.
- Modify `docs/API.md`: document the additive search response fields.

### Task 1: Backend compatibility ranking

**Files:**
- Modify: `backend/tests/test_huggingface.py`
- Modify: `backend/app/tasks/huggingface.py`

- [ ] **Step 1: Write the failing search-order test**

Add a small `SimpleNamespace` model factory and a test that returns a popular base model before an NVFP4 model from the mocked Hub API:

```python
from types import SimpleNamespace


def hub_model(model_id: str, *, tags: list[str], downloads: int = 0):
    return SimpleNamespace(
        id=model_id,
        tags=tags,
        downloads=downloads,
        likes=0,
        pipeline_tag="text-generation",
        private=False,
        gated=False,
        last_modified=None,
    )


def test_search_prioritizes_nvfp4_for_dgx_spark(tmp_path, monkeypatch):
    service = huggingface.HuggingFaceService(tmp_path / "hub")
    candidates = [
        hub_model("Qwen/Qwen3.8-27B", tags=["transformers", "safetensors"], downloads=1_000_000),
        hub_model("unsloth/Qwen3.8-27B-NVFP4", tags=["safetensors", "compressed-tensors"], downloads=100),
    ]
    monkeypatch.setattr(service.api, "list_models", lambda **_kwargs: candidates)

    results = service.search("Qwen", limit=2)

    assert [item["id"] for item in results] == [
        "unsloth/Qwen3.8-27B-NVFP4",
        "Qwen/Qwen3.8-27B",
    ]
    assert results[0]["spark_compatibility"] == {
        "level": "recommended",
        "score": 180,
        "reasons": ["NVFP4 量化", "压缩权重格式", "Safetensors 权重"],
    }
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest backend/tests/test_huggingface.py::test_search_prioritizes_nvfp4_for_dgx_spark -q`

Expected: FAIL because the current service preserves Hub order and does not return `spark_compatibility`.

- [ ] **Step 3: Add edge-case tests**

Add tests that assert:

```python
def test_search_keeps_hub_order_when_spark_scores_match(tmp_path, monkeypatch):
    service = huggingface.HuggingFaceService(tmp_path / "hub")
    candidates = [
        hub_model("org/first", tags=["safetensors"]),
        hub_model("org/second", tags=["safetensors"]),
    ]
    monkeypatch.setattr(service.api, "list_models", lambda **_kwargs: candidates)
    assert [item["id"] for item in service.search("model", 2)] == ["org/first", "org/second"]


def test_search_marks_gguf_only_model_for_review(tmp_path, monkeypatch):
    service = huggingface.HuggingFaceService(tmp_path / "hub")
    monkeypatch.setattr(
        service.api,
        "list_models",
        lambda **_kwargs: [hub_model("org/model-GGUF", tags=["gguf", "llama.cpp"])],
    )
    result = service.search("model", 1)[0]["spark_compatibility"]
    assert result["level"] == "review"
    assert "需要额外运行时" in result["reasons"]


def test_search_scores_a_wider_pool_before_applying_limit(tmp_path, monkeypatch):
    service = huggingface.HuggingFaceService(tmp_path / "hub")
    calls = []
    candidates = [
        hub_model("org/base", tags=["safetensors"]),
        hub_model("org/model-NVFP4", tags=["compressed-tensors"]),
    ]
    monkeypatch.setattr(service.api, "list_models", lambda **kwargs: calls.append(kwargs) or candidates)
    result = service.search("model", 1)
    assert calls[0]["limit"] == 50
    assert result[0]["id"] == "org/model-NVFP4"
```

- [ ] **Step 4: Implement deterministic scoring and sorting**

In `backend/app/tasks/huggingface.py`, add helpers that normalize metadata, detect bounded and non-negated quantization tokens, and detect explicit `B`/`T` plus MoE `NxMB/T` scale tokens. Capacity risk and GGUF-only results are always `review`; risk reasons are kept in the three visible reasons.

```python
MODEL_SCALE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*([BT])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
MOE_MODEL_SCALE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*([BT])"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
QUANTIZATION_PATTERNS = {
    token: re.compile(rf"(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])", re.IGNORECASE)
    for token in ("nvfp4", "fp8", "awq", "gptq")
}
NEGATED_QUANTIZATION_PREFIX = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:not|no|non)[^A-Za-z0-9]+$",
    re.IGNORECASE,
)


def repository_parameter_billions(repository_id: str) -> float | None:
    values = [
        float(value) * (1_000 if unit.upper() == "T" else 1)
        for value, unit in MODEL_SCALE_PATTERN.findall(repository_id)
    ]
    values.extend(
        float(experts) * float(value) * (1_000 if unit.upper() == "T" else 1)
        for experts, value, unit in MOE_MODEL_SCALE_PATTERN.findall(repository_id)
    )
    return max(values, default=None)


def has_quantization_token(repository_id: str, tags: list[str], token: str) -> bool:
    pattern = QUANTIZATION_PATTERNS[token]
    for value in (repository_id, *tags):
        for match in pattern.finditer(value):
            if not NEGATED_QUANTIZATION_PREFIX.search(value[:match.start()]):
                return True
    return False


def spark_compatibility(model: Any) -> dict[str, Any]:
    repository_id = str(model.id)
    model_tags = [str(tag) for tag in (model.tags or [])]
    tags = {tag.casefold() for tag in model_tags}
    has_nvfp4 = has_quantization_token(repository_id, model_tags, "nvfp4")
    has_fp8 = has_quantization_token(repository_id, model_tags, "fp8")
    has_awq_or_gptq = has_quantization_token(
        repository_id, model_tags, "awq"
    ) or has_quantization_token(repository_id, model_tags, "gptq")
    has_compressed = "compressed-tensors" in tags
    has_safetensors = "safetensors" in tags
    has_runtime = bool({"vllm", "sglang"} & tags)
    gguf_only = "gguf" in tags and not (has_compressed or has_safetensors)

    score = 0
    reasons: list[str] = []
    if has_nvfp4:
        score += 120
        reasons.append("NVFP4 量化")
    elif has_fp8:
        score += 35
        reasons.append("FP8 量化")
    elif has_awq_or_gptq:
        score += 30
        reasons.append("低比特量化")
    parameter_billions = repository_parameter_billions(repository_id)
    capacity_risk = parameter_billions is not None and parameter_billions > 180
    if capacity_risk:
        score -= 120
        reasons.append("模型规模需评估")
    if has_compressed:
        score += 30
        reasons.append("压缩权重格式")
    if has_safetensors:
        score += 20
        reasons.append("Safetensors 权重")
    if has_runtime:
        score += 20
        reasons.append("适配当前推理运行时")
    if str(model.pipeline_tag or "").casefold() in {"text-generation", "image-text-to-text"}:
        score += 10
        reasons.append("生成任务")
    if gguf_only:
        score -= 40
        reasons.append("需要额外运行时")

    if capacity_risk or gguf_only:
        level = "review"
    elif has_nvfp4 and (has_compressed or has_runtime or has_safetensors):
        level = "recommended"
    elif score >= 20:
        level = "compatible"
    else:
        level = "review"
    return {"level": level, "score": score, "reasons": reasons[:3]}
```

Update `HuggingFaceService.search()` to request `limit=50`, enrich each serialized result, perform a stable descending level-and-score sort, and return only `safe_limit` entries:

```python
def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
    safe_limit = min(max(limit, 1), 50)
    models = self.api.list_models(search=query, limit=50, full=True)
    results = []
    for model in models:
        results.append(
            {
                "id": model.id,
                "downloads": model.downloads or 0,
                "likes": model.likes or 0,
                "pipeline_tag": model.pipeline_tag,
                "private": bool(model.private),
                "gated": bool(model.gated),
                "last_modified": model.last_modified,
                "tags": list(model.tags or [])[:20],
                "spark_compatibility": spark_compatibility(model),
            }
        )
    level_rank = {"review": 0, "compatible": 1, "recommended": 2}
    return sorted(
        results,
        key=lambda item: (
            level_rank[item["spark_compatibility"]["level"]],
            item["spark_compatibility"]["score"],
        ),
        reverse=True,
    )[:safe_limit]
```

- [ ] **Step 5: Run backend tests and lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_huggingface.py -q
.\.venv\Scripts\ruff.exe check backend/app/tasks/huggingface.py backend/tests/test_huggingface.py
```

Expected: all Hugging Face tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit backend behavior**

```powershell
git add backend/app/tasks/huggingface.py backend/tests/test_huggingface.py
git commit -m "feat: prioritize Spark-compatible Hugging Face models"
```

### Task 2: Frontend compatibility presentation

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/utils/huggingface.ts`
- Modify: `frontend/src/utils/huggingface.test.ts`
- Modify: `frontend/src/pages/HuggingFacePage.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Write the failing presentation test**

Add this import and test to `frontend/src/utils/huggingface.test.ts`:

```typescript
import { getSparkCompatibilityPresentation } from './huggingface'

test('maps Spark compatibility levels to concise labels', () => {
  expect(getSparkCompatibilityPresentation('recommended')).toEqual({ label: 'DGX Spark 推荐', color: 'success' })
  expect(getSparkCompatibilityPresentation('compatible')).toEqual({ label: '可部署', color: 'processing' })
  expect(getSparkCompatibilityPresentation('review')).toEqual({ label: '需评估', color: 'default' })
})
```

- [ ] **Step 2: Run the focused frontend test and verify RED**

Run: `corepack pnpm --dir frontend test -- src/utils/huggingface.test.ts`

Expected: FAIL because `getSparkCompatibilityPresentation` is not exported.

- [ ] **Step 3: Add the response type and presentation helper**

Add to `HuggingFaceModel` in `frontend/src/api/types.ts`:

```typescript
spark_compatibility: {
  level: 'recommended' | 'compatible' | 'review'
  score: number
  reasons: string[]
}
```

Add to `frontend/src/utils/huggingface.ts`:

```typescript
const presentations = {
  recommended: { label: 'DGX Spark 推荐', color: 'success' },
  compatible: { label: '可部署', color: 'processing' },
  review: { label: '需评估', color: 'default' },
} as const

export function getSparkCompatibilityPresentation(level: keyof typeof presentations) {
  return presentations[level]
}
```

- [ ] **Step 4: Render tags and reasons in the search list**

In `HuggingFacePage.tsx`, import `ThunderboltOutlined` and the presentation helper. Replace the inline result renderer with this block:

```tsx
renderItem={(model) => {
  const presentation = getSparkCompatibilityPresentation(model.spark_compatibility.level)
  return (
    <List.Item actions={[<Button key="download" icon={<CloudDownloadOutlined />} onClick={() => setSelected(model)}>下载</Button>]}>
      <List.Item.Meta
        title={(
          <div className="hf-result-title">
            <strong>{model.id}</strong>
            <Tag
              className={`hf-spark-tag hf-spark-tag-${model.spark_compatibility.level}`}
              icon={<ThunderboltOutlined />}
              color={presentation.color}
            >
              {presentation.label}
            </Tag>
            {model.gated && <Tag icon={<LockOutlined />} color="warning">需授权</Tag>}
          </div>
        )}
        description={(
          <div className="hf-result-meta">
            <Tag>{model.pipeline_tag ?? '未分类'}</Tag>
            <Typography.Text type="secondary">{model.downloads.toLocaleString()} 次下载</Typography.Text>
            <Typography.Text type="secondary">{model.likes.toLocaleString()} 赞</Typography.Text>
            <Typography.Text className="hf-compatibility-reason" type="secondary">
              {model.spark_compatibility.reasons.slice(0, 2).join(' · ') || '模型元数据不足'}
            </Typography.Text>
          </div>
        )}
      />
    </List.Item>
  )
}}
```

Use these stable classes in `frontend/src/styles.css`:

```css
.hf-result-title { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; min-width: 0; }
.hf-result-title strong { overflow-wrap: anywhere; }
.hf-result-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; margin-top: 4px; }
.hf-spark-tag { color: var(--color-ink) !important; }
.hf-spark-tag-recommended { background: var(--color-primary-soft) !important; border-color: var(--color-primary) !important; }
.hf-spark-tag-compatible { background: color-mix(in oklch, var(--color-info) 14%, var(--color-bg)) !important; border-color: var(--color-info) !important; }
.hf-spark-tag-review { background: var(--color-surface) !important; border-color: var(--color-border) !important; }
.hf-compatibility-reason { flex-basis: 100%; color: var(--color-muted) !important; overflow-wrap: anywhere; }
```

Add a Testing Library page test that performs a search against a mocked result, verifies the
`hf-spark-tag-recommended` class and compatibility reason text, and restores global mocks in
`afterEach`.

- [ ] **Step 5: Run frontend tests, lint, and build**

Run:

```powershell
corepack pnpm --dir frontend test
corepack pnpm --dir frontend lint
corepack pnpm --dir frontend build
```

Expected: Vitest passes, Oxlint reports zero errors, and Vite produces `frontend/dist`.

- [ ] **Step 6: Commit frontend behavior**

```powershell
git add frontend/src/api/types.ts frontend/src/utils/huggingface.ts frontend/src/utils/huggingface.test.ts frontend/src/pages/HuggingFacePage.tsx frontend/src/pages/HuggingFacePage.test.tsx frontend/src/styles.css
git commit -m "feat: show DGX Spark model compatibility"
```

### Task 3: API documentation and full verification

**Files:**
- Modify: `docs/API.md`

- [ ] **Step 1: Document the additive search contract**

Add an example under the Hugging Face endpoint table showing `spark_compatibility.level`, `score`, and `reasons`, and state that the service scores up to 50 Hub candidates before applying the requested response limit.

- [ ] **Step 2: Run complete local verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\ruff.exe check backend/app backend/tests
corepack pnpm --dir frontend test
corepack pnpm --dir frontend lint
corepack pnpm --dir frontend build
git diff --check
```

Expected: all commands exit 0 with no test failures, lint findings, TypeScript errors, or whitespace errors.

- [ ] **Step 3: Commit documentation**

```powershell
git add docs/API.md
git commit -m "docs: describe Spark compatibility ranking"
```

### Task 4: Deploy and verify on DGX Spark

**Files:**
- No source files changed.

- [ ] **Step 1: Push the verified commits**

Run: `git push origin main`

Expected: the remote `main` branch advances without rejection.

- [ ] **Step 2: Update the manager on `192.168.6.202`**

Using the configured SSH key and pinned known-hosts file, run `git pull --ff-only` in the existing manager checkout, then `docker compose build` and `docker compose up -d`.

Expected: `docker compose ps` reports `dgx-spark-web-manager` as healthy.

- [ ] **Step 3: Verify the live API ranking**

Authenticate with the existing administrator session and call:

```text
GET http://192.168.6.202:3000/api/huggingface/search?query=Qwen&limit=10
```

Expected: `unsloth/Qwen3.8-27B-NVFP4` appears before the non-NVFP4 Qwen result and includes `spark_compatibility.level = recommended` with `NVFP4 量化` in `reasons`.

- [ ] **Step 4: Verify desktop, mobile, and dark mode**

Open `http://192.168.6.202:3000/huggingface`, search for `Qwen`, and capture desktop and mobile screenshots. Confirm the repository name, compatibility tag, authorization tag, action button, and reasons do not overlap at either viewport, and confirm tag contrast in dark mode.

- [ ] **Step 5: Confirm repository and service state**

Run `git status --short`, compare local and remote HEAD, and re-run `/api/health`.

Expected: clean worktree, matching commits, and a healthy API response.
