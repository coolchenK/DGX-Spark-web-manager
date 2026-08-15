from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app import models  # noqa: F401
from app.api.deployments import router as deployments_router
from app.api.diagnostics import router as diagnostics_router
from app.api.gateway import GatewayAuthError
from app.api.gateway import router as gateway_router
from app.api.huggingface import router as huggingface_router
from app.api.inventory import router as inventory_router
from app.api.providers import router as providers_router
from app.api.system import router as system_router
from app.api.tasks import router as tasks_router
from app.audit import router as audit_router
from app.auth import router as auth_router
from app.config import Settings
from app.db import Database
from app.operations.executor import OperationExecutor
from app.runtime.sglang import SGLangAdapter
from app.runtime.vllm import VllmAdapter
from app.security import PasswordManager, SecretBox, SessionManager
from app.services.deployments import DeploymentService
from app.services.diagnostics import DiagnosticService
from app.services.discovery import DiscoveryService
from app.services.providers import ProviderService
from app.services.system import SystemService
from app.tasks.engine import TaskEngine
from app.tasks.huggingface import HuggingFaceService


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()
    database = Database(app_settings.database_url)
    task_engine = TaskEngine(database.session_factory)
    huggingface_service = HuggingFaceService(app_settings.hf_cache_dir, app_settings.hf_token)
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
    )
    provider_service = ProviderService(SecretBox(app_settings.secret_key))
    diagnostic_service = DiagnosticService(
        provider_service,
        SystemService(app_settings.data_dir),
        deployment_service,
    )
    operation_executor = OperationExecutor(
        session_factory=database.session_factory,
        deployment_service=deployment_service,
        discovery_service=DiscoveryService(app_settings.model_root_paths),
    )
    task_engine.register("model.download", huggingface_service.download_handler)
    task_engine.register("deployment.create", deployment_service.create_handler)
    task_engine.register("deployment.action", deployment_service.action_handler)
    task_engine.register("operation.execute", operation_executor.handler)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database.create_schema()
        app.state.settings = app_settings
        app.state.database = database
        task_engine.start()
        yield
        task_engine.stop()
        database.dispose()

    app = FastAPI(title="DGX Spark Web Manager", version="0.1.0", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.database = database
    app.state.password_manager = PasswordManager(app_settings.admin_password)
    app.state.session_manager = SessionManager(
        app_settings.secret_key, app_settings.session_ttl_seconds
    )
    app.state.secret_box = SecretBox(app_settings.secret_key)
    app.state.system_service = SystemService(app_settings.data_dir)
    app.state.discovery_service = DiscoveryService(app_settings.model_root_paths)
    app.state.huggingface_service = huggingface_service
    app.state.task_engine = task_engine
    app.state.deployment_service = deployment_service
    app.state.provider_service = provider_service
    app.state.diagnostic_service = diagnostic_service
    app.state.operation_executor = operation_executor
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

    @app.get("/api/health")
    def health(request: Request) -> dict[str, str]:
        with request.app.state.database.session_factory() as session:
            session.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "service": app_settings.app_name,
            "database": "ok",
        }

    return app


app = create_app()
