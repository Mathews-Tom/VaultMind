"""Configuration management for VaultMind.

Loads defaults, TOML, .env, environment, and deliberate constructor overrides.
Higher-priority sources override individual nested keys, not whole sections.

Default base directory: ~/.vaultmind/
  vault/              — Obsidian vault (or symlink to existing)
  data/chromadb/      — Vector store
  data/knowledge_graph.json — Graph persistence
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal, cast, get_args, get_origin

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from vaultmind.vault.security import (
    validate_ordinary_output_path,
    validate_ordinary_write_path,
)

VAULTMIND_HOME = Path.home() / ".vaultmind"


class VaultConfig(BaseSettings):
    """Obsidian vault configuration."""

    path: Path = Field(description="Absolute path to the Obsidian vault root")
    inbox_folder: str = "00-inbox"
    daily_folder: str = "01-daily"
    templates_folder: str = "06-templates"
    meta_folder: str = "_meta"
    sync_interval_seconds: int = 300
    excluded_folders: list[str] = Field(
        default_factory=lambda: [".obsidian", ".git", ".trash", "wiki", "_wiki"]
    )

    @field_validator("excluded_folders")
    @classmethod
    def require_wiki_exclusions(cls, folders: list[str]) -> list[str]:
        missing = [name for name in ("wiki", "_wiki") if name not in folders]
        if missing:
            raise ValueError(
                "vault.excluded_folders is missing "
                + ", ".join(missing)
                + "; add these exact root folder names to the selected configuration"
            )
        return folders

    @field_validator("path")
    @classmethod
    def validate_vault_path(cls, v: Path) -> Path:
        v = v.expanduser()
        if not v.exists():
            raise ValueError(f"Vault path does not exist: {v}")
        return v.resolve()

    @model_validator(mode="after")
    def validate_output_folders(self) -> VaultConfig:
        for field in ("inbox_folder", "daily_folder", "templates_folder", "meta_folder"):
            validate_ordinary_write_path(str(getattr(self, field)), self.path)
        return self


class MemoryProfileConfig(BaseSettings):
    """Curated identity-memory layout and promotion guardrails."""

    enabled: bool = True
    identity_folder: str = "_meta/identity"
    user_note_name: str = "user.md"
    system_note_name: str = "system.md"
    promoted_memory_note_name: str = "memory.md"
    priorities_note_name: str = "priorities.md"
    daily_log_folder: str = "_meta/daily-logs"
    recent_daily_logs: int = 3
    promoted_memory_max_chars: int = 12_000


class ProactivityConfig(BaseSettings):
    """Global autonomy and confirmation boundaries."""

    mode: Literal["observer", "advisor", "assistant", "partner"] = "observer"
    require_confirmation_for_external_send: bool = True
    require_confirmation_for_destructive_actions: bool = True
    allow_local_automation: bool = False
    allow_external_side_effects: bool = False


class HeartbeatConfig(BaseSettings):
    """Deterministic proactive-state collection boundaries."""

    enabled: bool = False
    schedule: str = "*/30 * * * *"
    timezone: str = "UTC"
    active_hours_start: int = 8
    active_hours_end: int = 18
    snapshot_state_path: str = ""  # Empty = default ~/.vaultmind/data/heartbeat_state.json
    max_snapshot_items: int = 50
    max_actions_per_run: int = 5


class DraftsConfig(BaseSettings):
    """Draft lifecycle storage and approval defaults."""

    enabled: bool = False
    active_folder: str = "_meta/drafts/active"
    sent_folder: str = "_meta/drafts/sent"
    expired_folder: str = "_meta/drafts/expired"
    expiry_hours: int = 24
    require_human_approval: bool = True
    voice_match_limit: int = 5


class IntegrationsConfig(BaseSettings):
    """External system policy defaults and manifest location."""

    enabled: bool = False
    manifest_path: str = ""  # Empty = default ~/.vaultmind/data/integrations.json
    default_capability: Literal["read", "draft", "write"] = "read"
    sanitize_external_text: bool = True
    audit_retention_days: int = 90


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
    single_pass_extraction_enabled: bool = True
    extraction_confidence_threshold: float = 0.7


class TelegramConfig(BaseSettings):
    """Telegram bot configuration."""

    bot_token: str = Field(default="", description="Telegram bot token from @BotFather")
    allowed_user_ids: list[int] = Field(
        default_factory=list,
        description="Telegram user IDs allowed to use the bot. Empty = allow all.",
    )
    thinking_session_ttl: int = 3600  # seconds
    notification_chat_id: int = 0  # chat ID for proactive notifications; 0 = disabled
    thinking_summarization_enabled: bool = True
    thinking_message_count_threshold: int = 20
    thinking_recent_turns_to_keep: int = 6
    thinking_batch_size: int = 4
    thinking_summary_max_tokens: int = 400


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
    cache_enabled: bool = True
    cache_max_entries: int = 50
    cache_similarity_threshold: float = 0.85


class RankingConfig(BaseSettings):
    """Search result ranking configuration."""

    enabled: bool = True
    semantic_weight: float = 0.40
    recency_weight: float = 0.20
    connection_density_weight: float = 0.25
    activation_weight: float = 0.05
    note_type_weight: float = 0.10
    connection_max_hops: int = 2
    entity_confidence_threshold: float = 0.6
    recency_half_life_days: float = 30.0
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 20
    importance_scoring_enabled: bool = True
    authority_default: int = 3  # Neutral authority level applied when a note has none stamped
    authority_weight: dict[int, float] = Field(
        default_factory=lambda: {1: 0.85, 2: 0.92, 3: 1.0, 4: 1.08, 5: 1.15}
    )

    @model_validator(mode="after")
    def _check_weights_sum(self) -> RankingConfig:
        total = (
            self.semantic_weight
            + self.recency_weight
            + self.connection_density_weight
            + self.activation_weight
            + self.note_type_weight
        )
        if abs(total - 1.0) > 0.05:
            msg = f"Ranking weights must sum to ~1.0 (±0.05), got {total:.4f}"
            raise ValueError(msg)
        return self


class BenchConfig(BaseSettings):
    """Retrieval self-benchmark configuration (`vaultmind bench`)."""

    enabled: bool = True
    golden_path: str = "benchmarks/golden.yaml"
    k: int = 5
    score_floor: float = 0.5
    recall_at_k_threshold: float = 0.8
    mrr_threshold: float = 0.6
    retrieval_decline_threshold: float = 0.8
    llm_decline_threshold: float = 0.8
    llm_model: str = ""  # Empty = use llm.fast_model
    trend_path: str = ""  # Empty = default ~/.vaultmind/data/bench/trend.jsonl


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


class DistillConfig(BaseSettings):
    """Conversation distillation configuration."""

    enabled: bool = False
    min_turns: int = 3
    model: str = ""  # Empty = use llm.thinking_model
    output_folder: str = "qa-artifacts"
    max_tokens: int = 800


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
    short_term_days: int = 30  # Episodes older than this auto-promote to long_term


class GapConfig(BaseSettings):
    """Knowledge gap ledger configuration."""

    enabled: bool = True
    db_path: str = ""  # Empty = default ~/.vaultmind/data/gaps.db
    stale_after_days: int = 30
    max_shown: int = 10


class ContradictionConfig(BaseSettings):
    """Contradiction detection + deterministic resolution configuration."""

    enabled: bool = True
    # Escalate-only by default: Kosha's own real-model Gate-0 run recorded
    # contradiction routing accuracy 0.17 vs a 1.00 prompt-only baseline
    # (docs/gate0-status.md, M13 Gate-0). Flip only after `vaultmind eval
    # contradict` shows this detector beating the trivial always-escalate
    # baseline on your own vault.
    auto_resolve: bool = False
    detection_model: str = ""  # Empty = use llm.fast_model
    max_tokens: int = 300
    eval_path: str = "benchmarks/contradict_eval.yaml"


class AutonomyConfig(BaseSettings):
    """Graduated autonomy review-queue configuration (M7).

    Confidence cutoffs mirror Kosha's own `AutonomyThresholds` defaults
    (`kosha/approve/autonomy.py`) — a proposal's confidence must clear
    `skim_below` (and its impact must be `low`) to auto-apply.
    """

    enabled: bool = True
    db_path: str = ""  # Empty = default ~/.vaultmind/data/review_queue.db
    block_below: float = 0.4
    skim_below: float = 0.9
    force_block: bool = False


class SourcesConfig(BaseSettings):
    """Source connector framework configuration (M8).

    Per-instance connector config (`SourceInstance`: name, kind, target,
    schedule) lives in `config/sources.toml`, not here — this is only the
    global toggle + cursor-store path, matching `db_path`-only settings
    like `GapConfig`/`AutonomyConfig` (no matching `config/default.toml`
    section, per M5's own established precedent for these opt-in stores).
    """

    enabled: bool = True
    db_path: str = ""  # Empty = default ~/.vaultmind/data/sources.db
    config_path: str = ""  # Empty = default config/sources.toml


class ConsolidationConfig(BaseSettings):
    """Memory consolidation pipeline configuration."""

    enabled: bool = False  # Opt-in — runs monthly by default
    retention_days: int = 365
    min_references_for_promotion: int = 3
    schedule: str = "0 0 1 * *"  # Monthly on 1st


class ProceduralConfig(BaseSettings):
    """Procedural memory configuration."""

    enabled: bool = False  # Off by default — experimental
    db_path: str = ""  # Empty = default ~/.vaultmind/data/procedural.db
    min_episodes_for_pattern: int = 3
    synthesis_model: str = ""  # Empty = use llm.fast_model


class MCPRetryConfig(BaseSettings):
    """MCP tool execution retry configuration."""

    enabled: bool = True
    max_retries: int = 1
    use_llm_correction: bool = True
    correction_model: str = ""  # Empty = use llm.fast_model
    timeout_seconds: int = 30
    retryable_errors: list[str] = Field(
        default_factory=lambda: ["ValueError", "LLMError", "TimeoutError", "ConnectionError"]
    )


class MCPAuditConfig(BaseSettings):
    """MCP audit logging configuration."""

    enabled: bool = True
    level: str = "standard"  # minimal / standard / verbose
    log_search_queries: bool = False
    retention_days: int = 90


class MCPProfileConfig(BaseModel):
    """Validated profile values, independent of the optional MCP server package."""

    description: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    folder_scope: list[str] = Field(default_factory=lambda: ["*"])
    write_enabled: bool = False
    max_note_size_bytes: int | None = Field(default=None, ge=0)
    requires_confirmation: bool = False

    @field_validator("max_note_size_bytes", mode="before")
    @classmethod
    def reject_boolean_size(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("max_note_size_bytes must be an integer or null")
        return value


class MCPConfig(BaseSettings):
    """MCP server configuration."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    profiles: dict[str, MCPProfileConfig] = Field(default_factory=dict)
    audit: MCPAuditConfig = Field(default_factory=MCPAuditConfig)


class LoopsConfig(BaseSettings):
    """Compound loop job configuration."""

    insight_enabled: bool = True
    insight_interval_days: int = 7
    insight_schedule: str = ""  # Cron expression; empty = derive from interval_days
    evolution_enabled: bool = True
    evolution_interval_days: int = 7
    evolution_schedule: str = ""
    procedural_enabled: bool = False  # requires [procedural].enabled = true
    procedural_interval_days: int = 7
    procedural_schedule: str = ""


class HealthConfig(BaseSettings):
    """System health monitoring configuration."""

    enabled: bool = True
    check_interval_seconds: int = 60
    check_chromadb: bool = True
    check_sqlite: bool = True
    check_graph_file: bool = True
    check_llm: bool = True
    check_bot: bool = True
    check_vault_access: bool = True
    chromadb_latency_warn_ms: int = 1000
    retention_days: int = 30


class SchedulerConfig(BaseSettings):
    """Background scheduler configuration."""

    state_path: str = ""  # Empty = default ~/.vaultmind/data/scheduler_state.json


class _DictionaryEnvSource(EnvSettingsSource):
    """Consume mapping keys before inspecting fields on their value model."""

    def next_field(
        self, field: FieldInfo | Any | None, key: str, case_sensitive: bool | None = None
    ) -> FieldInfo | None:
        annotation = field.annotation if isinstance(field, FieldInfo) else field
        if get_origin(annotation) is dict:
            # pydantic-settings 2.13 searches value-model fields first, so a
            # profile named "allowed_tools" otherwise skips the mapping level.
            return FieldInfo.from_annotation(get_args(annotation)[1])
        return super().next_field(field, key, case_sensitive)


class _DictionaryDotEnvSource(DotEnvSettingsSource, _DictionaryEnvSource):
    """Use the same dictionary traversal for dotenv values."""


def _dictionary_source(source: EnvSettingsSource) -> EnvSettingsSource:
    # Reuse resolved BaseSettings controls, including per-load overrides, rather
    # than rebuilding either source from model_config defaults.
    options: dict[str, Any] = {
        "case_sensitive": source.case_sensitive,
        "env_prefix": source.env_prefix,
        "env_prefix_target": source.env_prefix_target,
        "env_nested_delimiter": source.env_nested_delimiter,
        "env_nested_max_split": source.env_nested_max_split,
        "env_ignore_empty": source.env_ignore_empty,
        "env_parse_none_str": source.env_parse_none_str,
        "env_parse_enums": source.env_parse_enums,
    }
    if isinstance(source, DotEnvSettingsSource):
        return _DictionaryDotEnvSource(
            source.settings_cls,
            env_file=source.env_file,
            env_file_encoding=source.env_file_encoding,
            **options,
        )
    return _DictionaryEnvSource(source.settings_cls, **options)


class Settings(BaseSettings):
    """Root configuration — aggregates all sub-configs."""

    model_config = SettingsConfigDict(
        env_prefix="VAULTMIND_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    vault: VaultConfig = Field(default_factory=lambda: VaultConfig(path=VAULTMIND_HOME / "vault"))
    memory_profile: MemoryProfileConfig = Field(default_factory=MemoryProfileConfig)
    proactivity: ProactivityConfig = Field(default_factory=ProactivityConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    drafts: DraftsConfig = Field(default_factory=DraftsConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
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
    bench: BenchConfig = Field(default_factory=BenchConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    maturation: MaturationConfig = Field(default_factory=MaturationConfig)
    digest: DigestConfig = Field(default_factory=DigestConfig)
    auto_tag: AutoTagConfig = Field(default_factory=AutoTagConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    distill: DistillConfig = Field(default_factory=DistillConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    episodic: EpisodicConfig = Field(default_factory=EpisodicConfig)
    gaps: GapConfig = Field(default_factory=GapConfig)
    contradiction: ContradictionConfig = Field(default_factory=ContradictionConfig)
    autonomy: AutonomyConfig = Field(default_factory=AutonomyConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    procedural: ProceduralConfig = Field(default_factory=ProceduralConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    mcp_retry: MCPRetryConfig = Field(default_factory=MCPRetryConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loops: LoopsConfig = Field(default_factory=LoopsConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)

    # API keys — always from env vars
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    voyage_api_key: str = ""

    def __init__(self, *, _toml_file: Path | None = None, **values: Any) -> None:
        """Select TOML per load while forwarding all BaseSettings controls."""
        super().__init__(_toml_file=_toml_file, **values)

    @model_validator(mode="after")
    def validate_output_destinations(self) -> Settings:
        """Preflight selected outputs without creating directories or state.

        This prevents configured aliases from entering wiki-owned storage before
        assemblies construct providers/stores. Writers still check at use time;
        configuration validation is not a filesystem or concurrency sandbox.
        """
        root = self.vault.path
        folders: tuple[tuple[BaseModel, tuple[str, ...]], ...] = (
            (self.memory_profile, ("identity_folder", "daily_log_folder")),
            (self.drafts, ("active_folder", "sent_folder", "expired_folder")),
            (self.maturation, ("inbox_folder",)),
            (self.digest, ("inbox_folder",)),
            (self.ingest, ("inbox_folder",)),
            (self.research, ("output_folder",)),
            (self.distill, ("output_folder",)),
        )
        for model, fields in folders:
            for field in fields:
                validate_ordinary_write_path(str(getattr(model, field)), root)
        for field in (
            "user_note_name",
            "system_note_name",
            "promoted_memory_note_name",
            "priorities_note_name",
        ):
            validate_ordinary_write_path(
                str(
                    Path(self.memory_profile.identity_folder) / getattr(self.memory_profile, field)
                ),
                root,
            )

        validate_ordinary_output_path(self.chroma.persist_dir, root)
        validate_ordinary_output_path(self.graph.persist_path, root)
        data = VAULTMIND_HOME / "data"
        outputs: tuple[tuple[BaseModel, str, Path | None], ...] = (
            (self.search, "bm25_db_path", data / "bm25.db"),
            (self.bench, "trend_path", data / "bench" / "trend.jsonl"),
            (self.tracking, "db_path", data / "preferences.db"),
            (self.activation, "db_path", None),
            (self.episodic, "db_path", data / "episodes.db"),
            (self.gaps, "db_path", data / "gaps.db"),
            (self.autonomy, "db_path", data / "review_queue.db"),
            (self.sources, "db_path", data / "sources.db"),
            (self.procedural, "db_path", data / "procedural.db"),
            (
                self.scheduler,
                "state_path",
                Path.home() / ".vaultmind" / "data" / "scheduler_state.json",
            ),
            (self.heartbeat, "snapshot_state_path", None),
            (self.integrations, "manifest_path", None),
        )
        for model, field, default in outputs:
            configured = str(getattr(model, field))
            destination = Path(configured) if configured else default
            if destination is not None:
                validate_ordinary_output_path(destination, root)
        # These outputs have no configurable path field. Check their actual
        # defaults too: a data-directory symlink must not bypass the boundary.
        for name in (
            "embedding_cache.db",
            "sessions.db",
            "lineage.json",
            "tag_vocabulary.json",
            "mcp_audit.jsonl",
        ):
            validate_ordinary_output_path(data / name, root)
        home_data = Path.home() / ".vaultmind" / "data"
        for name in ("maturation_state.json", "evolution_dismissed.json"):
            validate_ordinary_output_path(home_data / name, root)
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Keep the file selection local to this load. TOML values must not become
        # constructor overrides, which would outrank environment values.
        init_values = init_settings().copy()
        config_path: Path | None = init_values.pop("_toml_file", None)
        toml_values: dict[str, Any] = {}
        if config_path is not None and config_path.exists():
            with config_path.open("rb") as stream:
                toml_values = tomllib.load(stream)

        return (
            InitSettingsSource(settings_cls, init_values),
            _dictionary_source(cast("EnvSettingsSource", env_settings)),
            _dictionary_source(cast("DotEnvSettingsSource", dotenv_settings)),
            InitSettingsSource(settings_cls, toml_values),
            file_secret_settings,
        )

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
    def from_toml(cls, path: Path | None = None, **overrides: Any) -> Settings:
        """Load TOML below environment/.env and deliberate call-site overrides."""
        return cls(_toml_file=path or Path("config/default.toml"), **overrides)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings. Entry point for all config access."""
    return Settings.from_toml(config_path)
