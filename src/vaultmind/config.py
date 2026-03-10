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
from typing import Any, Literal

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
    graph_context_enabled: bool = True
    graph_hop_depth: int = 2
    graph_min_confidence: float = 0.6
    graph_max_relationships: int = 20


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


class SearchConfig(BaseSettings):
    """Paginated search configuration for the Telegram bot."""

    page_size: int = 5
    max_results: int = 25
    session_ttl: int = 300
    hybrid_enabled: bool = True
    bm25_db_path: str = ""  # Empty = default ~/.vaultmind/data/bm25.db


class RankingConfig(BaseSettings):
    """Search result ranking configuration."""

    enabled: bool = True


class EvolutionConfig(BaseSettings):
    """Belief evolution tracking configuration."""

    enabled: bool = True
    confidence_drift_threshold: float = 0.3
    stale_days: int = 180
    min_confidence_for_stale: float = 0.8
    max_results: int = 10
    include_in_digest: bool = True


class MaturationConfig(BaseSettings):
    """Zettelkasten maturation pipeline configuration."""

    enabled: bool = True
    schedule_day: str = "sunday"
    schedule_hour: int = 9
    timezone: str = "UTC"
    min_cluster_size: int = 3
    max_clusters_per_digest: int = 3
    cluster_eps: float = 0.25
    synthesis_max_tokens: int = 1500
    synthesis_model: str = ""  # Empty = use llm.thinking_model
    target_note_types: list[str] = Field(default_factory=lambda: ["fleeting", "literature"])
    dismissed_cluster_expiry_days: int = 90
    inbox_folder: str = "00-inbox"


class DigestConfig(BaseSettings):
    """Smart Daily Digest configuration."""

    enabled: bool = True
    period_days: int = 7
    schedule_hour: int = 8
    timezone: str = "UTC"
    save_to_vault: bool = True
    send_telegram: bool = True
    max_trending: int = 10
    max_suggestions: int = 5
    connection_threshold_low: float = 0.70
    connection_threshold_high: float = 0.85
    inbox_folder: str = "00-inbox"
    inbox_age_warning_days: int = 7
    max_inbox_shown: int = 10


class AutoTagConfig(BaseSettings):
    """Auto-tagging configuration."""

    enabled: bool = True
    max_tags_per_note: int = 2
    min_content_length: int = 100
    tagging_model: str = ""  # Empty = use llm.fast_model


class VoiceConfig(BaseSettings):
    """Voice note capture configuration."""

    enabled: bool = True
    whisper_model: str = "whisper-1"
    language: str = ""  # Empty = auto-detect


class ImageConfig(BaseSettings):
    """Image/photo capture configuration."""

    enabled: bool = True
    vision_model: str = ""  # Empty = use llm.fast_model
    max_image_size_bytes: int = 10_000_000
    save_originals: bool = True


class IngestConfig(BaseSettings):
    """URL ingestion configuration."""

    enabled: bool = True
    youtube_language: str = "en"
    max_content_length: int = 100_000
    inbox_folder: str = "00-inbox"


class ResearchConfig(BaseSettings):
    """Research pipeline configuration."""

    max_results: int = 5
    output_folder: str = "research"
    youtube_language: str = "en"


class TrackingConfig(BaseSettings):
    """User preference tracking configuration."""

    enabled: bool = True
    db_path: str = ""  # Empty = default (~/.vaultmind/data/preferences.db)


class ActivationConfig(BaseSettings):
    """Activation-based note decay configuration."""

    enabled: bool = True
    half_life_days: float = 14.0
    db_path: str = ""  # Empty = default ~/.vaultmind/data/activations.db


class EpisodicConfig(BaseSettings):
    """Episodic memory configuration."""

    enabled: bool = True
    db_path: str = ""  # default: ~/.vaultmind/data/episodes.db
    auto_extract: bool = False  # LLM extraction from notes (opt-in)
    extraction_model: str = ""  # Empty = use llm.fast_model


class ProceduralConfig(BaseSettings):
    """Procedural memory configuration."""

    enabled: bool = False  # Off by default — experimental
    db_path: str = ""  # Empty = default ~/.vaultmind/data/procedural.db
    min_episodes_for_pattern: int = 3
    synthesis_model: str = ""  # Empty = use llm.fast_model


class MCPConfig(BaseSettings):
    """MCP server configuration."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)


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
    duplicate_detection: DuplicateDetectionConfig = Field(default_factory=DuplicateDetectionConfig)
    note_suggestions: NoteSuggestionsConfig = Field(default_factory=NoteSuggestionsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    maturation: MaturationConfig = Field(default_factory=MaturationConfig)
    digest: DigestConfig = Field(default_factory=DigestConfig)
    auto_tag: AutoTagConfig = Field(default_factory=AutoTagConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    episodic: EpisodicConfig = Field(default_factory=EpisodicConfig)
    procedural: ProceduralConfig = Field(default_factory=ProceduralConfig)
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
