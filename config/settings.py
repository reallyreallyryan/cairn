from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:32b"
    ollama_model_light: str = "qwen3:8b"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_provider: Literal["ollama", "openai"] = "openai"

    # API Keys (optional)
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Agent
    agent_model: Literal["local", "cloud"] = "local"
    max_iterations: int = 10
    max_tool_calls_per_step: int = 5

    # File access (expanduser applied at runtime)
    allowed_directories: list[str] = ["~/Documents", "~/Projects"]
    notes_directory: str = "~/Documents/cairn/notes"

    # Code execution
    code_execution_timeout: int = 30
    allow_subprocess: bool = False

    # Docker / Sandbox
    docker_host: str = ""  # Auto-detected if empty (Colima, default socket)
    sandbox_image: str = "cairn-sandbox"
    sandbox_policy_path: str = "config/sandbox_policy.yaml"

    # GitHub (optional — for github_search tool)
    github_token: str = ""

    # Daemon
    daemon_poll_interval: int = 30
    model_routing_config: str = "config/model_routing.yaml"
    daily_budget_usd: float = 5.00
    budget_warn_threshold: float = 0.80

    # Notifications
    notification_method: str = "macos"  # "macos" or "log_only"
    notification_log_path: str = "~/Documents/cairn/notifications.log"

    # Digest
    digest_config_path: str = "config/digest_sources.yaml"

    # Logging
    log_level: str = "INFO"


settings = Settings()
