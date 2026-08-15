from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app import models  # noqa: F401
from app.api.huggingface import router as huggingface_router
from app.api.inventory import router as inventory_router
from app.api.system import router as system_router
from app.api.tasks import router as tasks_router
from app.audit import router as audit_router
from app.auth import router as auth_router
from app.config import Settings
from app.db import Database
from app.security import PasswordManager, SecretBox, SessionManager
from app.services.discovery import DiscoveryService
from app.services.system import SystemService
from app.tasks.engine import TaskEngine
from app.tasks.huggingface import HuggingFaceService


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()
    database = Database(app_settings.database_url)
    task_engine = TaskEngine(database.session_factory)
    huggingface_service = HuggingFaceService(app_settings.hf_cache_dir, app_settings.hf_token)
    task_engine.register("model.download", huggingface_service.download_handler)

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router)
    app.include_router(audit_router)
    app.include_router(system_router)
    app.include_router(inventory_router)
    app.include_router(tasks_router)
    app.include_router(huggingface_router)

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
