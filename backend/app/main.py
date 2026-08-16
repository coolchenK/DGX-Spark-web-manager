from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app import models  # noqa: F401
from app.api.deployments import router as deployments_router
from app.api.diagnostics import router as diagnostics_router
from app.api.gateway import GatewayActivity, GatewayAuthError
from app.api.gateway import router as gateway_router
from app.api.huggingface import router as huggingface_router
from app.api.inventory import router as inventory_router
from app.api.providers import router as providers_router
from app.api.settings import router as settings_router
from app.api.system import router as system_router
from app.api.tasks import router as tasks_router
from app.audit import router as audit_router
from app.auth import router as auth_router
from app.config import Settings
from app.db import Database
from app.models import SecretSetting
from app.operations.executor import OperationExecutor
from app.runtime.sglang import SGLangAdapter
from app.runtime.vllm import VllmAdapter
from app.security import PasswordManager, SecretBox, SessionManager
from app.services.deployment_recommendations import DeploymentRecommendationService
from app.services.deployments import DeploymentService
from app.services.diagnostics import DiagnosticService
from app.services.discovery import DiscoveryService
from app.services.draft_models import DraftCompatibilityService
from app.services.model_evidence import ModelEvidenceLoader
from app.services.providers import ProviderService
from app.services.resource_estimator import ResourceEstimator
from app.services.runtime_capabilities import RuntimeCapabilityService
from app.services.system import SystemService
from app.tasks.engine import TaskEngine
from app.tasks.huggingface import HuggingFaceService


class _LazyDockerClient:
    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._client: Any | None = None
        self._factory = factory or self._default_factory
        self._lock = Lock()
        self._closed = False

    @staticmethod
    def _default_factory() -> Any:
        import docker

        return docker.from_env()

    def _get_client(self) -> Any:
        client = self._client
        if client is None:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Docker client is closed")
                if self._client is None:
                    self._client = self._factory()
                client = self._client
        return client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_client(), name)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()
    database = Database(app_settings.database_url)
    task_engine = TaskEngine(database.session_factory)
    huggingface_service = HuggingFaceService(app_settings.hf_cache_dir, app_settings.hf_token)
    discovery_service = DiscoveryService(app_settings.model_root_paths)
    system_service = SystemService(
        app_settings.data_dir, os_release_path=app_settings.host_os_release
    )
    provider_service = ProviderService(SecretBox(app_settings.secret_key))
    lazy_docker_client = _LazyDockerClient()
    evidence_loader = ModelEvidenceLoader(card_max_chars=app_settings.recommendation_card_max_chars)
    resource_estimator = ResourceEstimator(
        reserve_fraction=app_settings.memory_reserve_fraction,
        reserve_min_bytes=app_settings.memory_reserve_min_bytes,
    )
    runtime_capability_service = RuntimeCapabilityService(
        settings=app_settings, docker_client=lazy_docker_client
    )
    draft_service = DraftCompatibilityService(
        evidence_loader=evidence_loader,
        resource_estimator=resource_estimator,
    )
    deployment_service = DeploymentService(
        adapters={
            "vllm": VllmAdapter(
                allowed_images=app_settings.vllm_images,
                model_roots=app_settings.model_root_paths,
            ),
            "sglang": SGLangAdapter(
                allowed_images=app_settings.sglang_images,
                model_roots=app_settings.model_root_paths,
            ),
        },
        session_factory=database.session_factory,
        model_roots=app_settings.model_root_paths,
        host_model_roots=app_settings.host_model_root_paths,
        startup_timeout_seconds=app_settings.deployment_startup_timeout_seconds,
        runtime_capability_service=runtime_capability_service,
        evidence_loader=evidence_loader,
        draft_service=draft_service,
        resource_estimator=resource_estimator,
        system_snapshot=system_service.snapshot,
        docker_client=lazy_docker_client,
    )
    deployment_recommendation_service = DeploymentRecommendationService(
        evidence_loader=evidence_loader,
        runtime_capability_service=runtime_capability_service,
        resource_estimator=resource_estimator,
        draft_service=draft_service,
        system_snapshot=system_service.snapshot,
        huggingface_service=huggingface_service,
        provider_service=provider_service,
        cache_ttl_seconds=app_settings.recommendation_cache_ttl_seconds,
        card_max_chars=app_settings.recommendation_card_max_chars,
    )
    diagnostic_service = DiagnosticService(
        provider_service,
        system_service,
        deployment_service,
    )
    operation_executor = OperationExecutor(
        session_factory=database.session_factory,
        deployment_service=deployment_service,
        discovery_service=discovery_service,
    )

    def download_and_discover(context, payload):
        result = huggingface_service.download_handler(context, payload)
        with database.session_factory() as db:
            discovery_service.scan_models(db)
        return result

    task_engine.register("model.download", download_and_discover)
    task_engine.register("deployment.create", deployment_service.create_handler)
    task_engine.register("deployment.update", deployment_service.update_handler)
    task_engine.register("deployment.action", deployment_service.action_handler)
    task_engine.register("operation.execute", operation_executor.handler)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task_engine_started = False
        try:
            database.create_schema()
            app.state.settings = app_settings
            app.state.database = database
            with database.session_factory() as db:
                stored_token = db.get(SecretSetting, "huggingface_token")
                if stored_token:
                    huggingface_service.set_token(
                        app.state.secret_box.decrypt(stored_token.encrypted_value)
                    )
                if app_settings.auto_discovery:
                    app.state.discovery_service.scan_all(db)
            task_engine.start()
            task_engine_started = True
            yield
        finally:
            try:
                if task_engine_started:
                    task_engine.stop()
            finally:
                try:
                    lazy_docker_client.close()
                finally:
                    database.dispose()

    app = FastAPI(title="DGX Spark Web Manager", version="0.1.0", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.database = database
    app.state.password_manager = PasswordManager(app_settings.admin_password)
    app.state.session_manager = SessionManager(
        app_settings.secret_key, app_settings.session_ttl_seconds
    )
    app.state.secret_box = SecretBox(app_settings.secret_key)
    app.state.system_service = system_service
    app.state.discovery_service = discovery_service
    app.state.huggingface_service = huggingface_service
    app.state.task_engine = task_engine
    app.state.deployment_service = deployment_service
    app.state.provider_service = provider_service
    app.state.deployment_recommendation_service = deployment_recommendation_service
    app.state.diagnostic_service = diagnostic_service
    app.state.operation_executor = operation_executor
    app.state.gateway_activity = GatewayActivity()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(GatewayAuthError)
    async def gateway_auth_error_handler(request: Request, exc: GatewayAuthError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Invalid or missing API key",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    app.include_router(auth_router)
    app.include_router(audit_router)
    app.include_router(system_router)
    app.include_router(inventory_router)
    app.include_router(tasks_router)
    app.include_router(huggingface_router)
    app.include_router(deployments_router)
    app.include_router(gateway_router)
    app.include_router(providers_router)
    app.include_router(diagnostics_router)
    app.include_router(settings_router)

    @app.get("/api/health")
    def health(request: Request) -> dict[str, str]:
        with request.app.state.database.session_factory() as session:
            session.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "service": app_settings.app_name,
            "database": "ok",
        }

    static_dir = app_settings.static_dir.resolve()
    if static_dir.is_dir() and (static_dir / "index.html").is_file():
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            is_api_path = path in {"api", "v1"} or path.startswith(("api/", "v1/"))
            if is_api_path:
                raise HTTPException(status_code=404, detail="Not found")
            candidate = (static_dir / path).resolve()
            if candidate.is_relative_to(static_dir) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()
