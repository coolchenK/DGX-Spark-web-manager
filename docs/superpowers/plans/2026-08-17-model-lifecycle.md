# Model Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe permanent model deletion and make deployment stop/uninstall semantics complete for managed and discovered inference containers.

**Architecture:** A focused `ModelLifecycleService` performs dependency checks and source-specific deletion inside the existing task engine. Deployment removal keeps the current runtime adapter boundary but adds container identity validation and explicit confirmation for discovered containers. React pages expose the lifecycle actions through Ant Design confirmation flows and task feedback.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, Hugging Face CLI, Docker SDK, pytest, React 19, TypeScript, Ant Design, TanStack Query, Vitest.

---

## File Map

- Create `backend/app/services/model_lifecycle.py`: dependency discovery, safe path validation, HF/local deletion handlers.
- Modify `backend/app/api/inventory.py`: model delete request and task creation endpoint.
- Modify `backend/app/services/deployments.py`: discovered-container uninstall and identity checks.
- Modify `backend/app/api/deployments.py`: confirmation payload for discovered-container removal.
- Modify `backend/app/main.py`: construct lifecycle service and register `model.delete` task.
- Create `backend/tests/test_model_lifecycle.py`: unit and API coverage for deletion and reference protection.
- Modify `backend/tests/test_deployments.py`: managed/discovered uninstall behavior and stale-container behavior.
- Modify `frontend/src/api/client.ts`: allow an optional JSON body on DELETE requests.
- Modify `frontend/src/api/types.ts`: deletion conflict and task result types.
- Modify `frontend/src/pages/ModelsPage.tsx`: permanent-delete action and confirmation dialog.
- Create `frontend/src/pages/ModelsPage.test.tsx`: desktop/mobile deletion behavior.
- Modify `frontend/src/pages/DeploymentsPage.tsx`: stop/uninstall language and discovered-container confirmation.
- Modify `frontend/src/pages/DeploymentsPage.test.tsx`: managed and discovered uninstall flows.
- Modify `docs/API.md`: lifecycle API contracts.

### Task 1: Model Dependency Detection

**Files:**
- Create: `backend/app/services/model_lifecycle.py`
- Test: `backend/tests/test_model_lifecycle.py`

- [ ] **Step 1: Write failing dependency tests**

```python
def test_references_include_base_and_draft_deployments(database, model_assets):
    service = ModelLifecycleService(model_roots=(model_assets.root,), hf_cache_dir=model_assets.hf)
    with database.session_factory() as db:
        references = service.references(db, model_assets.target.id)
    assert [(item.deployment_name, item.usage) for item in references] == [
        ("base-service", "base"),
        ("draft-service", "draft"),
    ]


def test_path_reference_supports_legacy_deployment(database, model_assets):
    with database.session_factory() as db:
        db.add(Deployment(
            name="legacy", runtime="vllm", endpoint_url="http://127.0.0.1:8100",
            api_model_name="legacy", config={"model_path": str(model_assets.target_path)},
        ))
        db.commit()
        references = ModelLifecycleService(
            model_roots=(model_assets.root,), hf_cache_dir=model_assets.hf,
        ).references(db, model_assets.target.id)
    assert references[0].usage == "legacy_path"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest backend/tests/test_model_lifecycle.py -k references -v`

Expected: FAIL because `app.services.model_lifecycle` does not exist.

- [ ] **Step 3: Implement normalized dependency detection**

```python
@dataclass(frozen=True)
class ModelReference:
    deployment_id: str
    deployment_name: str
    usage: Literal["base", "draft", "legacy_path"]


def _resolved(value: str | None) -> Path | None:
    if not value:
        return None
    try:
        return Path(value).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def references(self, db: Session, model_id: str) -> list[ModelReference]:
    asset = db.get(ModelAsset, model_id)
    if asset is None:
        raise LookupError("Model was not found")
    target = _resolved(asset.local_path)
    found: dict[str, ModelReference] = {}
    for deployment in db.scalars(select(Deployment).order_by(Deployment.name)):
        config = deployment.config or {}
        speculative = config.get("speculative") if isinstance(config, dict) else None
        draft_id = speculative.get("draft_model_id") if isinstance(speculative, dict) else None
        usage = None
        if deployment.model_id == model_id:
            usage = "base"
        elif draft_id == model_id:
            usage = "draft"
        elif target is not None and target in {
            _resolved(config.get("model_path")),
            _resolved(speculative.get("draft_model_path") if isinstance(speculative, dict) else None),
        }:
            usage = "legacy_path"
        if usage:
            found[deployment.id] = ModelReference(deployment.id, deployment.name, usage)
    return list(found.values())
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest backend/tests/test_model_lifecycle.py -k references -v`

Expected: PASS.

- [ ] **Step 5: Commit dependency protection**

```bash
git add backend/app/services/model_lifecycle.py backend/tests/test_model_lifecycle.py
git commit -m "feat: detect model deployment references"
```

### Task 2: Safe Source-Specific Model Deletion

**Files:**
- Modify: `backend/app/services/model_lifecycle.py`
- Modify: `backend/tests/test_model_lifecycle.py`

- [ ] **Step 1: Write failing path and HF deletion tests**

```python
def test_local_delete_rejects_root_and_symlink_escape(tmp_path):
    root = tmp_path / "models"
    outside = tmp_path / "outside"
    root.mkdir(); outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    service = ModelLifecycleService(model_roots=(root,), hf_cache_dir=tmp_path / "hf")
    with pytest.raises(ValueError, match="model root"):
        service.validate_local_target(root)
    with pytest.raises(ValueError, match="outside configured model roots"):
        service.validate_local_target(root / "escape")


def test_hf_delete_uses_dry_run_then_confirmed_rm(lifecycle_fixture):
    result = lifecycle_fixture.service.delete_handler(
        lifecycle_fixture.context, {"model_id": lifecycle_fixture.hf_asset.id}
    )
    assert lifecycle_fixture.runner.calls == [
        ["hf", "cache", "rm", "model/org/model", "--cache-dir", str(lifecycle_fixture.hf), "--dry-run", "--json"],
        ["hf", "cache", "rm", "model/org/model", "--cache-dir", str(lifecycle_fixture.hf), "--yes", "--json"],
    ]
    assert result["source"] == "huggingface"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest backend/tests/test_model_lifecycle.py -k "local_delete or hf_delete" -v`

Expected: FAIL because safe target validation and deletion handlers are missing.

- [ ] **Step 3: Implement deletion with a testable command runner**

```python
def validate_local_target(self, value: Path) -> Path:
    target = value.resolve(strict=True)
    roots = [root.resolve(strict=True) for root in self.model_roots]
    if target in roots:
        raise ValueError("Refusing to delete a model root")
    if not any(target.is_relative_to(root) for root in roots):
        raise ValueError("Model path is outside configured model roots")
    lexical = Path(os.path.abspath(value))
    if lexical != target and value.is_symlink():
        raise ValueError("Model symlink resolves outside configured model roots")
    return target


def _hf_commands(self, repository_id: str) -> tuple[list[str], list[str]]:
    target = f"model/{validate_repository_id(repository_id)}"
    common = ["hf", "cache", "rm", target, "--cache-dir", str(self.hf_cache_dir)]
    return [*common, "--dry-run", "--json"], [*common, "--yes", "--json"]


def delete_handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
    model_id = str(payload["model_id"])
    with self.session_factory() as db:
        asset = db.get(ModelAsset, model_id)
        if asset is None:
            raise ValueError("Model was not found")
        blocking = self.references(db, model_id)
        if blocking:
            raise ModelInUseError(blocking)
        asset.status = "deleting"
        source, repository_id, local_path = asset.source, asset.repository_id, asset.local_path
        before = directory_size(self._repository_root(asset)) if Path(local_path).exists() else 0
        db.commit()
    try:
        if source == "huggingface" and repository_id:
            dry_run, remove = self._hf_commands(repository_id)
            preview = json.loads(self.command_runner(dry_run).stdout)
            validate_hf_preview(preview, repository_id)
            context.check_control()
            removed = json.loads(self.command_runner(remove).stdout)
            released_bytes = parse_released_bytes(removed, fallback=before)
        elif Path(local_path).exists():
            shutil.rmtree(self.validate_local_target(Path(local_path)))
            released_bytes = before
        else:
            released_bytes = 0
        with self.session_factory() as db:
            asset = db.get(ModelAsset, model_id)
            if asset:
                db.delete(asset)
                db.commit()
    except Exception:
        with self.session_factory() as db:
            asset = db.get(ModelAsset, model_id)
            if asset:
                asset.status = "delete_failed"
                db.commit()
        raise
    return {"model_id": model_id, "source": source, "released_bytes": released_bytes}
```

`parse_released_bytes` accepts only the documented numeric JSON size fields and falls back to the pre-delete unique directory size when the CLI version omits them. Tests cover malformed CLI JSON without exposing raw command output.

- [ ] **Step 4: Verify service tests pass**

Run: `python -m pytest backend/tests/test_model_lifecycle.py -v`

Expected: PASS, including an execution-time reference test that leaves the model untouched.

- [ ] **Step 5: Commit deletion service**

```bash
git add backend/app/services/model_lifecycle.py backend/tests/test_model_lifecycle.py
git commit -m "feat: safely delete local and Hugging Face models"
```

### Task 3: Model Delete API and Task Registration

**Files:**
- Modify: `backend/app/api/inventory.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_model_lifecycle.py`
- Modify: `docs/API.md`

- [ ] **Step 1: Write failing API tests**

```python
def test_delete_model_requires_exact_name(authenticated_client, available_model):
    response = authenticated_client.request(
        "DELETE", f"/api/models/{available_model.id}", json={"confirmation": "wrong"}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Confirmation does not match model name"


def test_delete_model_returns_reference_conflict(authenticated_client, referenced_model):
    response = authenticated_client.request(
        "DELETE", f"/api/models/{referenced_model.id}",
        json={"confirmation": referenced_model.name},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "model_in_use"
    assert response.json()["detail"]["references"][0]["deployment_name"] == "service"
```

- [ ] **Step 2: Run API tests and verify RED**

Run: `python -m pytest backend/tests/test_model_lifecycle.py -k "delete_model_requires or reference_conflict" -v`

Expected: FAIL with 405 Method Not Allowed.

- [ ] **Step 3: Add endpoint, audit, service wiring, and task handler**

```python
class ModelDeleteRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=255)


@router.delete("/models/{model_id}", status_code=status.HTTP_202_ACCEPTED)
def delete_model(model_id: str, payload: ModelDeleteRequest, request: Request,
                 db: DbSession, admin: CsrfAdmin) -> dict[str, Any]:
    asset = db.get(ModelAsset, model_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if payload.confirmation != asset.name:
        raise HTTPException(status_code=422, detail="Confirmation does not match model name")
    references = request.app.state.model_lifecycle_service.references(db, model_id)
    if references:
        raise HTTPException(status_code=409, detail={
            "code": "model_in_use",
            "references": [asdict(item) for item in references],
        })
    task = request.app.state.task_engine.create_task(
        db, task_type="model.delete", title=f"删除模型 {asset.name}",
        input_json={"model_id": asset.id}, idempotency_key=f"model:{asset.id}:delete",
    )
    record_audit(db, actor=str(admin["username"]), action="model.delete.create",
                 resource_type="model", resource_id=asset.id, details={"task_id": task.id})
    db.commit()
    return serialize_task(task)
```

In `create_app`, construct the service with `app_settings.model_root_paths`, `hf_cache_dir`, `database.session_factory`, and `discovery_service`; store it on `app.state` and register `model.delete`.

Deployment preview, create, and task handlers must load `ModelAsset` by `model_id` and reject states `deleting`, `delete_failed`, or a missing row. This second handler check prevents an already queued deployment from using a model deleted by an earlier task.

- [ ] **Step 4: Run API and task tests**

Run: `python -m pytest backend/tests/test_model_lifecycle.py -v`

Expected: PASS; audit details contain no local credentials or tokens.

- [ ] **Step 5: Document and commit the API**

```bash
git add backend/app/api/inventory.py backend/app/main.py backend/tests/test_model_lifecycle.py docs/API.md
git commit -m "feat: expose audited model deletion tasks"
```

### Task 4: Uninstall Managed and Discovered Services

**Files:**
- Modify: `backend/app/api/deployments.py`
- Modify: `backend/app/services/deployments.py`
- Modify: `backend/tests/test_deployments.py`

- [ ] **Step 1: Write failing discovered-container tests**

```python
def test_discovered_delete_requires_container_name(authenticated_client, discovered):
    response = authenticated_client.post(f"/api/deployments/{discovered.id}/delete")
    assert response.status_code == 422


def test_discovered_delete_accepts_matching_container_name(authenticated_client, discovered):
    response = authenticated_client.post(
        f"/api/deployments/{discovered.id}/delete",
        json={"confirm_container_name": discovered.container_name},
    )
    assert response.status_code == 202


def test_delete_missing_container_cleans_stale_record(deployment_service, missing_container):
    result = deployment_service.action_handler(
        FakeContext(), {"deployment_id": missing_container.id, "action": "delete"}
    )
    assert result["status"] == "deleted"
    assert result["container_missing"] is True


def test_stop_marks_deployment_stopping_before_docker_call(deployment_service, running):
    running.adapter.on_stop = lambda: assert_database_status(running.id, "stopping")
    deployment_service.action_handler(
        FakeContext(), {"deployment_id": running.id, "action": "stop"}
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest backend/tests/test_deployments.py -k "discovered_delete or missing_container" -v`

Expected: FAIL because discovered containers return 409 and missing containers raise.

- [ ] **Step 3: Add explicit confirmation and identity-safe removal**

```python
class DeploymentActionRequest(BaseModel):
    confirm_container_name: str | None = Field(default=None, max_length=255)


if action == "delete" and not deployment.managed:
    if payload is None or payload.confirm_container_name != deployment.container_name:
        raise HTTPException(status_code=422, detail="Container name confirmation is required")
```

Before stopping, set `status="stopping"` and `health="unknown"` in a committed transaction so the gateway immediately stops routing new requests. Before removing a discovered container, reload it and verify both `container.id == container_id` and `container.name == container_name`. Mark the row `deleting` before Docker calls. Catch `docker.errors.NotFound` only for delete, then remove the stale database row and return `container_missing=True`. On any other error retain the row and store `health="unhealthy"`.

- [ ] **Step 4: Run deployment tests**

Run: `python -m pytest backend/tests/test_deployments.py -k "action or delete or uninstall" -v`

Expected: PASS; managed deletion remains backward compatible.

- [ ] **Step 5: Commit service uninstall support**

```bash
git add backend/app/api/deployments.py backend/app/services/deployments.py backend/tests/test_deployments.py
git commit -m "feat: uninstall discovered inference services"
```

### Task 5: Model Library Delete UI

**Files:**
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/pages/ModelsPage.tsx`
- Create: `frontend/src/pages/ModelsPage.test.tsx`

- [ ] **Step 1: Write failing component tests**

```tsx
it('requires the exact model name before permanent deletion', async () => {
  renderModelsPage([modelFixture])
  await user.click(screen.getByRole('button', { name: '删除模型' }))
  expect(screen.getByRole('button', { name: '永久删除' })).toBeDisabled()
  await user.type(screen.getByLabelText('输入模型名称确认'), modelFixture.name)
  await user.click(screen.getByRole('button', { name: '永久删除' }))
  expect(api.delete).toHaveBeenCalledWith(`/api/models/${modelFixture.id}`, {
    confirmation: modelFixture.name,
  })
})

it('renders deployment references returned by a conflict', async () => {
  mockDeleteConflict([{ deployment_id: 'dep-1', deployment_name: 'chat', usage: 'base' }])
  renderModelsPage([modelFixture])
  await confirmDelete(modelFixture.name)
  expect(await screen.findByText('chat')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run component tests and verify RED**

Run: `pnpm --dir frontend test -- ModelsPage.test.tsx`

Expected: FAIL because the delete button and request body are absent.

- [ ] **Step 3: Implement the Ant Design deletion dialog**

Extend `api.delete` to accept `body?: unknown`. Add a danger icon button to desktop and mobile renderers. Use `Modal`, an exact-name `Input`, model source/path/size summary, and a disabled primary danger button until names match. On 409, retain the modal and render `ApiError.detail.references`; on 202, close it, show the created task message, and invalidate `models` and `tasks`.

```tsx
const remove = useMutation({
  mutationFn: (model: ModelAsset) => api.delete<TaskRecord>(`/api/models/${model.id}`, {
    confirmation,
  }),
  onSuccess: () => {
    message.success('模型删除任务已创建')
    setDeleting(null)
    queryClient.invalidateQueries({ queryKey: ['models'] })
    queryClient.invalidateQueries({ queryKey: ['tasks'] })
  },
})
```

- [ ] **Step 4: Run frontend tests, lint, and type build**

Run: `pnpm --dir frontend test -- ModelsPage.test.tsx && pnpm --dir frontend lint && pnpm --dir frontend build`

Expected: all commands succeed.

- [ ] **Step 5: Commit model deletion UI**

```bash
git add frontend/src/api/client.ts frontend/src/api/types.ts frontend/src/pages/ModelsPage.tsx frontend/src/pages/ModelsPage.test.tsx
git commit -m "feat: delete unused models from the library"
```

### Task 6: Deployment Stop and Uninstall UI

**Files:**
- Modify: `frontend/src/pages/DeploymentsPage.tsx`
- Modify: `frontend/src/pages/DeploymentsPage.test.tsx`

- [ ] **Step 1: Write failing wording and confirmation tests**

```tsx
it('describes stop as releasing runtime resources', async () => {
  renderDeploymentsPage([runningManaged])
  expect(screen.getByRole('button', { name: '停止实例' })).toBeInTheDocument()
})

it('requires a discovered container name before uninstall', async () => {
  renderDeploymentsPage([runningDiscovered])
  await user.click(screen.getByRole('button', { name: '卸载服务' }))
  await user.type(screen.getByLabelText('输入容器名称确认'), runningDiscovered.container_name!)
  await user.click(screen.getByRole('button', { name: '确认卸载' }))
  expect(api.post).toHaveBeenCalledWith(
    `/api/deployments/${runningDiscovered.id}/delete`,
    { confirm_container_name: runningDiscovered.container_name },
  )
})
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pnpm --dir frontend test -- DeploymentsPage.test.tsx`

Expected: FAIL because discovered containers have no uninstall action and labels differ.

- [ ] **Step 3: Implement stop/uninstall controls**

Rename visible stop text to `停止实例` and the delete tooltip/label to `卸载服务`. Managed containers keep a standard `Popconfirm` stating that model files remain. Discovered containers use a `Modal` with an exact container-name input and warning that saved deployment parameters cannot recreate the service. Both success paths invalidate deployments, gateway models, and tasks.

- [ ] **Step 4: Run all frontend checks**

Run: `pnpm --dir frontend test && pnpm --dir frontend lint && pnpm --dir frontend build`

Expected: all tests, lint, and build pass.

- [ ] **Step 5: Commit lifecycle UI**

```bash
git add frontend/src/pages/DeploymentsPage.tsx frontend/src/pages/DeploymentsPage.test.tsx
git commit -m "feat: clarify stop and service uninstall actions"
```

### Task 7: Lifecycle Regression and DGX Smoke Test

**Files:**
- Modify: `docs/TROUBLESHOOTING.md`

- [ ] **Step 1: Run backend quality gates**

Run: `python -m ruff check backend && python -m pytest backend/tests -q`

Expected: lint succeeds and the full backend suite passes.

- [ ] **Step 2: Run frontend quality gates**

Run: `pnpm --dir frontend test && pnpm --dir frontend lint && pnpm --dir frontend build`

Expected: all frontend checks pass.

- [ ] **Step 3: Build the ARM64 manager image on DGX Spark**

Run on DGX: `cd ~/dgx-spark-web-manager && docker compose build manager`

Expected: build succeeds and `docker image inspect dgx-spark-web-manager:local --format '{{.Architecture}}'` prints `arm64`.

- [ ] **Step 4: Exercise deletion with a disposable model and container**

Create a small disposable local model directory under the configured model root, scan it, deploy only a disposable test container, then verify in order: reference conflict, stop, uninstall, model delete, model directory absence, and audit entries. Do not use an existing production model for this test.

- [ ] **Step 5: Document recovery and commit**

Document that `delete_failed` preserves the record, a missing container can be cleaned by uninstall, and model deletion never removes container images.

```bash
git add docs/TROUBLESHOOTING.md
git commit -m "docs: add model lifecycle recovery guidance"
```
