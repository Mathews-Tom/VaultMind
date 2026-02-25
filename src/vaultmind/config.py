"""Configuration management for VaultMind.

Loads from environment variables, .env files, and config/default.toml.
All secrets come from env vars; structural config from TOML.

Default base directory: ~/.vaultmind/
  vault/              — Obsidian vault (or symlink to existing)
  data/chromadb/      — Vector store
  data/knowledge_graph.json — Graph persistence
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VAULTMIND_HOME = Path.home() / ".vaultmind"


class VaultConfig(BaseSettings):
    """Obsidian vault configuration."""

    path: Path = Field(description="Absolute path to the Obsidian vault root")
    inbox_folder: str = "00-inbox"
    daily_folder: str = "01-daily"
    templates_folder: str = "06-templates"
    meta_folder: str = "_meta"
    sync_interval_seconds: int = 300
    excluded_folders: list[str] = Field(default_factory=lambda: [".obsidian", ".git", ".trash"])

    @field_validator("path")
    @classmethod
    def validate_vault_path(cls, v: Path) -> Path:
        v = v.expanduser()
        if not v.exists():
            raise ValueError(f"Vault path does not exist: {v}")
        return v.resolve()


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    provider: Literal["openai", "voyage"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64
    cache_enabled: bool = True


class ChromaConfig(BaseSettings):
    """ChromaDB configuration."""

    persist_dir: Path = Field(default_factory=lambda: VAULTMIND_HOME / "data" / "chromadb")
    collection_name: str = "vault_chunks"
    distance_fn: Literal["cosine", "l2", "ip"] = "cosine"

    @field_validator("persist_dir")
    @classmethod
    def expand_persist_dir(cls, v: Path) -> Path:
        return v.expanduser()


class GraphConfig(BaseSettings):
    """Knowledge graph configuration."""

    persist_path: Path = Field(
        default_factory=lambda: VAULTMIND_HOME / "data" / "knowledge_graph.json"
    )
    extraction_model: str = ""  # Empty = use llm.thinking_model
    min_confidence: float = 0.7
    enrichment_schedule: str = "0 3 * * *"  # 3 AM daily

    @field_validator("persist_path")
    @classmethod
    def expand_persist_path(cls, v: Path) -> Path:
        return v.expanduser()


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    provider: Literal["anthropic", "openai", "gemini", "ollama"] = "anthropic"
    thinking_model: str = "claude-sonnet-4-20250514"
    fast_model: str = "claude-sonnet-4-20250514"
    max_context_notes: int = 8
    max_tokens: int = 4096
    ollama_base_url: str = "http://localhost:11434"


class TelegramConfig(BaseSettings):
    """Telegram bot configuration."""

    bot_token: str = Field(default="", description="Telegram bot token from @BotFather")
    allowed_user_ids: list[int] = Field(
        default_factory=list,
        description="Telegram user IDs allowed to use the bot. Empty = allow all.",
    )
    thinking_session_ttl: int = 3600  # seconds


class RoutingConfig(BaseSettings):
    """Message routing configuration for the Telegram bot."""

    chat_model: str = ""  # Empty = use llm.fast_model
    chat_max_tokens: int = 1024
    vault_context_enabled: bool = True
    max_context_chunks: int = 4
    capture_all: bool = False  # Escape hatch: route all text to capture


class WatchConfig(BaseSettings):
    """Incremental watch mode configuration."""

    debounce_ms: int = 500
    hash_stability_check: bool = True
    reextract_graph: bool = False
    batch_graph_interval_seconds: int = 300


class DuplicateDetectionConfig(BaseSettings):
    """Semantic duplicate detection configuration."""

    enabled: bool = True
    min_content_length: int = 100


class NoteSuggestionsConfig(BaseSettings):
    """Context-aware note suggestions configuration."""

    enabled: bool = True
    min_content_length: int = 100
    entity_weight: float = 0.1
    graph_weight: float = 0.05


class MCPConfig(BaseSettings):
    """MCP server configuration."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765


class Settings(BaseSettings):
    """Root configuration — aggregates all sub-configs."""

    model_config = SettingsConfigDict(
        env_prefix="VAULTMIND_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vault: VaultConfig = Field(default_factory=lambda: VaultConfig(path=VAULTMIND_HOME / "vault"))
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    duplicate_detection: DuplicateDetectionConfig = Field(
        default_factory=DuplicateDetectionConfig
    )
    note_suggestions: NoteSuggestionsConfig = Field(
        default_factory=NoteSuggestionsConfig
    )
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    # API keys — always from env vars
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    voyage_api_key: str = ""

    @property
    def llm_api_key(self) -> str:
        """Resolve the API key for the configured LLM provider."""
        keys = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "ollama": "",
        }
        return keys.get(self.llm.provider, "")

    @classmethod
    def from_toml(cls, path: Path | None = None) -> Settings:
        """Load settings from TOML file, with env var overrides."""
        config_path = path or Path("config/default.toml")
        if config_path.exists():
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            return cls(**data)
        return cls()


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings. Entry point for all config access."""
    return Settings.from_toml(config_path)
