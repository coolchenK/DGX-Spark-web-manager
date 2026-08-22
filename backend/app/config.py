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
    model_root_mappings: str = ""
    hf_cache_dir: Path = Path("/root/.cache/huggingface/hub")
    hf_token: str | None = None
    static_dir: Path = Path("frontend/dist")
    host_os_release: Path = Path("/host/etc/os-release")
    auto_discovery: bool = True
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    session_ttl_seconds: int = 43_200
    cookie_secure: bool = False
    deployment_startup_timeout_seconds: int = Field(default=1200, ge=30, le=1800)
    runtime_probe_timeout_seconds: int = Field(default=45, ge=5, le=180)
    ops_agent_socket: Path = Path("/run/dgx-spark-manager/ops-agent.sock")
    ops_agent_key_file: Path = Path("/run/secrets/ops-agent.key")
    ops_agent_connect_timeout_seconds: float = Field(default=3, ge=0.5, le=30)
    ops_agent_read_timeout_seconds: float = Field(default=30, ge=22, le=120)
    ops_agent_output_limit_bytes: int = Field(default=1_000_000, ge=10_000, le=10_000_000)
    recommendation_cache_ttl_seconds: int = Field(default=900, ge=60, le=86_400)
    recommendation_card_max_chars: int = Field(default=100_000, ge=10_000, le=500_000)
    memory_reserve_fraction: float = Field(default=0.10, ge=0.05, le=0.30)
    memory_reserve_min_bytes: int = Field(default=8 * 1024**3, ge=1024**3)
    allowed_vllm_images: str = (
        "vllm/vllm-openai:v0.27.1,vllm/vllm-openai:muse-glimmer"
    )
    allowed_sglang_images: str = (
        "dgx-local/sglang-qwen38-dflash2:61fa64a,lmsysorg/sglang:qwen38-27b,"
        "sglang-inkling:specforge,"
        "lmsysorg/sglang:dev-cu13-inkling-dspark"
    )
    allowed_llama_cpp_images: str = "nvidia/cuda:12.9.0-devel-ubuntu24.04"
    llama_cpp_host_dir: Path = Path("/opt/llamacpp")
    llama_cpp_manager_dir: Path = Path("/llamacpp-runtime")

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
    def host_model_root_paths(self) -> tuple[Path, ...]:
        mappings: dict[str, Path] = {}
        for item in self.model_root_mappings.split(";"):
            if not item.strip():
                continue
            container_path, separator, host_path = item.partition("=")
            if not separator or not container_path.strip() or not host_path.strip():
                raise ValueError("Model root mappings must use container_path=host_path")
            mappings[str(Path(container_path.strip()))] = Path(host_path.strip())
        return tuple(mappings.get(str(root), root) for root in self.model_root_paths)

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    @staticmethod
    def _csv_set(value: str) -> set[str]:
        return {item.strip() for item in value.split(",") if item.strip()}

    @property
    def vllm_images(self) -> set[str]:
        return self._csv_set(self.allowed_vllm_images)

    @property
    def sglang_images(self) -> set[str]:
        return self._csv_set(self.allowed_sglang_images)

    @property
    def llama_cpp_images(self) -> set[str]:
        return self._csv_set(self.allowed_llama_cpp_images)
