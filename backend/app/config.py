from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DGX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "dgx-spark-web-manager"
    database_url: str = "sqlite:///./data/manager.db"
    secret_key: str = Field(min_length=32)
    admin_username: str = "admin"
    admin_password: str = Field(min_length=12)
    data_dir: Path = Path("./data")
    model_roots: str = "/models,/root/.cache/huggingface/hub"
    hf_cache_dir: Path = Path("/root/.cache/huggingface/hub")
    hf_token: str | None = None
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    session_ttl_seconds: int = 43_200
    cookie_secure: bool = False

    @field_validator("database_url")
    @classmethod
    def ensure_sqlite_parent(cls, value: str) -> str:
        if value.startswith("sqlite:///") and value != "sqlite:///:memory:":
            path = Path(value.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
        return value

    @property
    def model_root_paths(self) -> tuple[Path, ...]:
        return tuple(Path(item.strip()) for item in self.model_roots.split(",") if item.strip())

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]
