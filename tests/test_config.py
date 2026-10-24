"""Behavioral coverage for isolated, layered settings loads."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from vaultmind.config import Settings, load_settings

SettingsLoader = Callable[[Path], Settings]


@pytest.fixture(autouse=True)
def isolated_config_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)


@pytest.fixture(params=[Settings.from_toml, load_settings], ids=["from_toml", "load_settings"])
def loader(request: pytest.FixtureRequest) -> SettingsLoader:
    return request.param


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "settings.toml"
    path.write_text(
        f"""
[vault]
path = {json.dumps(str(vault))}
inbox_folder = "toml-inbox"
sync_interval_seconds = 731

[embedding]
provider = "voyage"
model = "toml-embedding-model"
dimensions = 768
batch_size = 41

[llm]
provider = "anthropic"
fast_model = "toml-fast-model"
""",
        encoding="utf-8",
    )
    return path


def test_nested_environment_overrides_only_conflicting_toml_values(
    loader: SettingsLoader, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAULTMIND_EMBEDDING__BATCH_SIZE", "23")
    monkeypatch.setenv("VAULTMIND_VAULT__SYNC_INTERVAL_SECONDS", "83")

    settings = loader(config_path)

    assert settings.embedding.batch_size == 23
    assert settings.embedding.dimensions == 768
    assert settings.embedding.model == "toml-embedding-model"
    assert settings.vault.sync_interval_seconds == 83
    assert settings.vault.inbox_folder == "toml-inbox"


def test_absent_environment_preserves_explicit_toml_values(
    loader: SettingsLoader, config_path: Path
) -> None:
    settings = loader(config_path)

    assert settings.vault.path == (config_path.parent.parent / "vault").resolve()
    assert settings.vault.inbox_folder == "toml-inbox"
    assert settings.vault.sync_interval_seconds == 731
    assert settings.embedding.provider == "voyage"
    assert settings.embedding.dimensions == 768
    assert settings.embedding.batch_size == 41
    assert settings.llm.fast_model == "toml-fast-model"


def test_explicit_constructor_kwargs_beat_environment(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAULTMIND_EMBEDDING__BATCH_SIZE", "23")
    monkeypatch.setenv("VAULTMIND_EMBEDDING__DIMENSIONS", "1024")

    settings = Settings(
        vault={"path": config_path.parent.parent / "vault"},
        embedding={"batch_size": 17},
    )

    assert settings.embedding.batch_size == 17
    assert settings.embedding.dimensions == 1024


def test_deliberate_from_toml_kwargs_beat_environment_and_preserve_siblings(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAULTMIND_EMBEDDING__BATCH_SIZE", "23")
    monkeypatch.setenv("VAULTMIND_EMBEDDING__DIMENSIONS", "1024")

    settings = Settings.from_toml(config_path, embedding={"batch_size": 17})

    assert settings.embedding.batch_size == 17
    assert settings.embedding.dimensions == 1024
    assert settings.embedding.model == "toml-embedding-model"


def test_cwd_dotenv_is_below_environment_but_above_toml(
    loader: SettingsLoader, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_path = Path.cwd() / ".env"
    dotenv_path.write_text(
        "VAULTMIND_EMBEDDING__BATCH_SIZE=29\nVAULTMIND_EMBEDDING__DIMENSIONS=1024\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VAULTMIND_EMBEDDING__BATCH_SIZE", "23")

    settings = loader(config_path)

    assert settings.embedding.batch_size == 23
    assert settings.embedding.dimensions == 1024
    assert settings.embedding.model == "toml-embedding-model"

    monkeypatch.delenv("VAULTMIND_EMBEDDING__BATCH_SIZE")
    dotenv_only = loader(config_path)

    assert dotenv_only.embedding.batch_size == 29
    assert dotenv_only.embedding.dimensions == 1024

    dotenv_path.write_text("", encoding="utf-8")
    toml_only = loader(config_path)

    assert toml_only.embedding.batch_size == 41
    assert toml_only.embedding.dimensions == 768


@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize(
    ("name", "value", "location", "error_type"),
    [
        (
            "VAULTMIND_EMBEDDING__BATCH_SIZE",
            "not-an-integer",
            ("embedding", "batch_size"),
            "int_parsing",
        ),
        (
            "VAULTMIND_LLM__PROVIDER",
            "unsupported-provider",
            ("llm", "provider"),
            "literal_error",
        ),
    ],
    ids=["nested-integer", "nested-enum"],
)
def test_invalid_overrides_fail_instead_of_falling_back_to_valid_toml(
    loader: SettingsLoader,
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    name: str,
    value: str,
    location: tuple[str, str],
    error_type: str,
) -> None:
    if source == "environment":
        monkeypatch.setenv(name, value)
    else:
        (Path.cwd() / ".env").write_text(f"{name}={value}\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        loader(config_path)

    assert any(
        error["loc"] == location and error["type"] == error_type
        for error in exc_info.value.errors()
    )


def test_repeated_loads_do_not_leak_toml_paths_or_environment(
    loader: SettingsLoader, config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_vault = tmp_path / "other-vault"
    other_vault.mkdir()
    other_path = tmp_path / "other.toml"
    other_path.write_text(
        f"""
[vault]
path = {json.dumps(str(other_vault))}
inbox_folder = "other-inbox"

[embedding]
dimensions = 2048
batch_size = 59
model = "other-embedding-model"
""",
        encoding="utf-8",
    )

    first = loader(config_path)
    monkeypatch.setenv("VAULTMIND_EMBEDDING__BATCH_SIZE", "13")
    second = loader(other_path)
    monkeypatch.delenv("VAULTMIND_EMBEDDING__BATCH_SIZE")
    third = loader(config_path)
    fourth = loader(other_path)

    assert first.vault.inbox_folder == third.vault.inbox_folder == "toml-inbox"
    assert first.embedding.dimensions == third.embedding.dimensions == 768
    assert first.embedding.batch_size == third.embedding.batch_size == 41
    assert second.vault.path == fourth.vault.path == other_vault.resolve()
    assert second.vault.inbox_folder == fourth.vault.inbox_folder == "other-inbox"
    assert second.embedding.dimensions == fourth.embedding.dimensions == 2048
    assert second.embedding.model == fourth.embedding.model == "other-embedding-model"
    assert second.embedding.batch_size == 13
    assert fourth.embedding.batch_size == 59
    assert first.embedding.model == third.embedding.model == "toml-embedding-model"


def test_overridden_provider_drives_api_key_selection(
    loader: SettingsLoader, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAULTMIND_LLM__PROVIDER", "gemini")
    monkeypatch.setenv("VAULTMIND_GEMINI_API_KEY", "test-only-gemini-key")
    monkeypatch.setenv("VAULTMIND_ANTHROPIC_API_KEY", "test-only-anthropic-key")

    settings = loader(config_path)

    assert settings.llm_api_key == "test-only-gemini-key"
    assert settings.llm.fast_model == "toml-fast-model"


@pytest.fixture
def profile_config_path(config_path: Path) -> Path:
    with config_path.open("a", encoding="utf-8") as config:
        config.write(
            """
[mcp.profiles.editor]
description = "TOML editor"
allowed_tools = ["vault_read", "vault_write"]
folder_scope = ["02-projects"]
write_enabled = true
requires_confirmation = true
max_note_size_bytes = 512

[mcp.profiles.reader]
description = "Unaffected reader"
allowed_tools = ["vault_read"]
folder_scope = ["04-resources"]
write_enabled = false
"""
        )
    return config_path


@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize("enabled", [False, True], ids=["false", "true"])
def test_profile_boolean_overrides_parse_values_and_preserve_siblings(
    loader: SettingsLoader,
    profile_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    enabled: bool,
) -> None:
    # Both directions must override a conflicting TOML boolean, not use string truthiness.
    toml = profile_config_path.read_text(encoding="utf-8")
    profile_config_path.write_text(
        toml.replace("= true", f"= {str(not enabled).lower()}"), encoding="utf-8"
    )
    overrides = {
        f"VAULTMIND_MCP__PROFILES__EDITOR__{field}": str(enabled).lower()
        for field in ("WRITE_ENABLED", "REQUIRES_CONFIRMATION")
    }
    if source == "environment":
        for name, value in overrides.items():
            monkeypatch.setenv(name, value)
    else:
        (Path.cwd() / ".env").write_text(
            "".join(f"{name}={value}\n" for name, value in overrides.items()), encoding="utf-8"
        )

    settings = loader(profile_config_path)

    editor = settings.mcp.profiles["editor"]
    assert editor.write_enabled is enabled
    assert editor.requires_confirmation is enabled
    assert editor.description == "TOML editor"
    assert editor.allowed_tools == ["vault_read", "vault_write"]
    assert editor.folder_scope == ["02-projects"]
    assert editor.max_note_size_bytes == 512
    reader = settings.mcp.profiles["reader"]
    assert reader.description == "Unaffected reader"
    assert reader.allowed_tools == ["vault_read"]
    assert reader.folder_scope == ["04-resources"]
    assert reader.write_enabled is False


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_profile_json_list_leaves_override_toml(
    loader: SettingsLoader,
    profile_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    overrides = {
        "VAULTMIND_MCP__PROFILES__EDITOR__ALLOWED_TOOLS": '["capture_note"]',
        "VAULTMIND_MCP__PROFILES__EDITOR__FOLDER_SCOPE": '["00-inbox", "04-resources"]',
    }
    if source == "environment":
        for name, value in overrides.items():
            monkeypatch.setenv(name, value)
    else:
        (Path.cwd() / ".env").write_text(
            "".join(f"{name}='{value}'\n" for name, value in overrides.items()), encoding="utf-8"
        )

    settings = loader(profile_config_path)

    assert settings.mcp.profiles["editor"].allowed_tools == ["capture_note"]
    assert settings.mcp.profiles["editor"].folder_scope == ["00-inbox", "04-resources"]
    assert settings.mcp.profiles["editor"].write_enabled is True
    assert settings.mcp.profiles["reader"].allowed_tools == ["vault_read"]


@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize("profile_name", ["allowed_tools", "description", "write_enabled"])
def test_profile_names_matching_fields_preserve_decoding_and_enforcement(
    loader: SettingsLoader,
    profile_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    profile_name: str,
) -> None:
    with profile_config_path.open("a", encoding="utf-8") as config:
        config.write(
            f"""
[mcp.profiles.{profile_name}]
description = "Collision-name reader"
allowed_tools = ["vault_read"]
folder_scope = ["04-resources"]
write_enabled = false
requires_confirmation = true
max_note_size_bytes = 4
"""
        )
    name = f"VAULTMIND_MCP__PROFILES__{profile_name.upper()}__FOLDER_SCOPE"
    value = '["00-inbox"]'
    if source == "environment":
        monkeypatch.setenv(name, value)
    else:
        (Path.cwd() / ".env").write_text(f"{name}='{value}'\n", encoding="utf-8")

    settings = loader(profile_config_path)
    profile = settings.mcp.profiles[profile_name]
    assert profile.folder_scope == ["00-inbox"]
    assert profile.allowed_tools == ["vault_read"]
    assert profile.write_enabled is False
    assert profile.description == "Collision-name reader"
    assert profile.requires_confirmation is True
    assert profile.max_note_size_bytes == 4
    assert settings.mcp.profiles["reader"].folder_scope == ["04-resources"]

    from vaultmind.mcp.auth import ProfileEnforcer, ProfileError
    from vaultmind.mcp.profiles import load_profile

    policy = load_profile(profile_name, settings.mcp.profiles)
    assert policy.folder_scope == ("00-inbox",)
    assert policy.allowed_tools == frozenset({"vault_read"})
    assert policy.write_enabled is False
    assert policy.requires_confirmation is True
    enforcer = ProfileEnforcer(policy, settings.vault.path)
    enforcer.check_path(Path("00-inbox/note.md"))
    with pytest.raises(ProfileError, match="outside folder scope"):
        enforcer.check_path(Path("04-resources/note.md"))
    enforcer.check_tool("vault_read")
    with pytest.raises(ProfileError, match="not allowed"):
        enforcer.check_tool("vault_write")
    with pytest.raises(ProfileError, match="Write operations disabled"):
        enforcer.check_write()
    enforcer.check_size("1234")
    with pytest.raises(ProfileError, match="exceeds limit"):
        enforcer.check_size("12345")


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_invalid_profile_json_for_collision_name_fails_at_source(
    loader: SettingsLoader,
    profile_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    name = "VAULTMIND_MCP__PROFILES__ALLOWED_TOOLS__FOLDER_SCOPE"
    if source == "environment":
        monkeypatch.setenv(name, "[invalid-json")
    else:
        (Path.cwd() / ".env").write_text(f"{name}='[invalid-json'\n", encoding="utf-8")

    with pytest.raises(SettingsError, match='field "mcp"'):
        loader(profile_config_path)


@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize("profile_name", ["editor", "allowed_tools"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("write_enabled", "not-a-boolean"),
        ("requires_confirmation", "not-a-boolean"),
        ("allowed_tools", '"vault_read"'),
        ("allowed_tools", '["vault_read", 7]'),
        ("folder_scope", '{"folder": "00-inbox"}'),
        ("folder_scope", '["00-inbox", false]'),
        ("max_note_size_bytes", "-1"),
        ("max_note_size_bytes", "not-an-integer"),
        ("max_note_size_bytes", "1.5"),
    ],
)
def test_invalid_profile_leaves_do_not_fall_back_to_valid_toml(
    loader: SettingsLoader,
    profile_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    profile_name: str,
    field: str,
    value: str,
) -> None:
    toml = profile_config_path.read_text(encoding="utf-8")
    profile_config_path.write_text(
        toml.replace("[mcp.profiles.editor]", f"[mcp.profiles.{profile_name}]"), encoding="utf-8"
    )
    name = f"VAULTMIND_MCP__PROFILES__{profile_name.upper()}__{field.upper()}"
    if source == "environment":
        monkeypatch.setenv(name, value)
    else:
        (Path.cwd() / ".env").write_text(f"{name}='{value}'\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        loader(profile_config_path)

    assert any(
        error["loc"][:4] == ("mcp", "profiles", profile_name, field)
        for error in exc_info.value.errors()
    )


def test_explicit_null_size_override_removes_toml_limit_without_losing_profile_siblings(
    profile_config_path: Path,
) -> None:
    settings = Settings.from_toml(
        profile_config_path, mcp={"profiles": {"editor": {"max_note_size_bytes": None}}}
    )

    editor = settings.mcp.profiles["editor"]
    assert editor.max_note_size_bytes is None
    assert editor.write_enabled is True
    assert editor.folder_scope == ["02-projects"]
    assert editor.allowed_tools == ["vault_read", "vault_write"]
    assert settings.mcp.profiles["reader"].folder_scope == ["04-resources"]


@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize("size", [0, 1, 4096])
def test_profile_size_override_preserves_zero_and_positive_limits(
    loader: SettingsLoader,
    profile_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    size: int,
) -> None:
    name = "VAULTMIND_MCP__PROFILES__EDITOR__MAX_NOTE_SIZE_BYTES"
    if source == "environment":
        monkeypatch.setenv(name, str(size))
    else:
        (Path.cwd() / ".env").write_text(f"{name}={size}\n", encoding="utf-8")

    settings = loader(profile_config_path)

    assert settings.mcp.profiles["editor"].max_note_size_bytes == size
    assert settings.mcp.profiles["reader"].max_note_size_bytes is None


def test_explicit_toml_constructor_preserves_custom_dotenv_controls(
    profile_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (Path.cwd() / ".env").write_text(
        "VAULTMIND_MCP__PROFILES__EDITOR__WRITE_ENABLED=true\n", encoding="utf-8"
    )
    custom_dotenv = tmp_path / "custom.env"
    custom_dotenv.write_text(
        "VAULTMIND_MCP__PROFILES__EDITOR__WRITE_ENABLED=false\n"
        "VAULTMIND_MCP__PROFILES__EDITOR__FOLDER_SCOPE='[\"00-inbox\"]'\n",
        encoding="utf-8",
    )

    settings = Settings(
        _toml_file=profile_config_path, _env_file=custom_dotenv, _env_file_encoding="utf-8"
    )

    assert settings.mcp.profiles["editor"].write_enabled is False
    assert settings.mcp.profiles["editor"].folder_scope == ["00-inbox"]
    assert settings.mcp.profiles["editor"].description == "TOML editor"
    assert settings.mcp.profiles["reader"].folder_scope == ["04-resources"]

    monkeypatch.setenv("VAULTMIND_MCP__PROFILES__EDITOR__WRITE_ENABLED", "true")
    environment_wins = Settings(_toml_file=profile_config_path, _env_file=custom_dotenv)
    assert environment_wins.mcp.profiles["editor"].write_enabled is True
    assert environment_wins.mcp.profiles["editor"].folder_scope == ["00-inbox"]

    monkeypatch.delenv("VAULTMIND_MCP__PROFILES__EDITOR__WRITE_ENABLED")
    dotenv_disabled = Settings(_toml_file=profile_config_path, _env_file=None)
    assert dotenv_disabled.mcp.profiles["editor"].write_enabled is True
    assert dotenv_disabled.mcp.profiles["editor"].folder_scope == ["02-projects"]


@pytest.mark.parametrize(
    ("excluded", "missing"),
    [
        ([".git"], ("wiki", "_wiki")),
        ([".git", "wiki"], ("_wiki",)),
        ([".git", "_wiki"], ("wiki",)),
        (["Wiki", "_Wiki"], ("wiki", "_wiki")),
    ],
)
def test_explicit_missing_exclusions_fail_without_rewriting_setup(
    loader: SettingsLoader, config_path: Path, excluded: list[str], missing: tuple[str, ...]
) -> None:
    config_path.write_text(
        config_path.read_text().replace(
            'inbox_folder = "toml-inbox"',
            f'inbox_folder = "toml-inbox"\nexcluded_folders = {json.dumps(excluded)}',
        ),
        encoding="utf-8",
    )
    before = config_path.read_bytes()
    vault = config_path.parent.parent / "vault"
    with pytest.raises(ValidationError) as error:
        loader(config_path)
    for name in missing:
        assert name in str(error.value)
    assert "add these exact root folder names" in str(error.value)
    assert config_path.read_bytes() == before
    assert list(vault.iterdir()) == []


def test_new_config_is_read_only_and_excludes_both_owned_trees(
    loader: SettingsLoader, config_path: Path
) -> None:
    before = config_path.read_bytes()
    settings = loader(config_path)
    assert {"wiki", "_wiki"}.issubset(settings.vault.excluded_folders)
    assert list(settings.vault.path.iterdir()) == []
    assert config_path.read_bytes() == before


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("vault", "inbox_folder"),
        ("vault", "daily_folder"),
        ("vault", "templates_folder"),
        ("vault", "meta_folder"),
        ("memory_profile", "identity_folder"),
        ("memory_profile", "daily_log_folder"),
        ("drafts", "active_folder"),
        ("drafts", "sent_folder"),
        ("drafts", "expired_folder"),
        ("maturation", "inbox_folder"),
        ("digest", "inbox_folder"),
        ("ingest", "inbox_folder"),
        ("research", "output_folder"),
        ("distill", "output_folder"),
    ],
)
@pytest.mark.parametrize("destination", ["wiki/new", "_WiKi/new", "alias/new"])
def test_configured_ordinary_folders_fail_before_any_setup_writes(
    config_path: Path, section: str, field: str, destination: str
) -> None:
    vault = config_path.parent.parent / "vault"
    reserved = vault / "_wiki"
    reserved.mkdir()
    source = reserved / "source.json"
    source.write_bytes(b"preserved source")
    (vault / "alias").symlink_to(reserved, target_is_directory=True)
    with pytest.raises(ValidationError, match="Ordinary writes cannot target"):
        Settings.from_toml(config_path, **{section: {field: destination}})
    assert source.read_bytes() == b"preserved source"
    assert sorted(path.name for path in reserved.iterdir()) == ["source.json"]


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("chroma", "persist_dir"),
        ("graph", "persist_path"),
        ("search", "bm25_db_path"),
        ("bench", "trend_path"),
        ("tracking", "db_path"),
        ("activation", "db_path"),
        ("episodic", "db_path"),
        ("gaps", "db_path"),
        ("autonomy", "db_path"),
        ("sources", "db_path"),
        ("procedural", "db_path"),
        ("scheduler", "state_path"),
        ("heartbeat", "snapshot_state_path"),
        ("integrations", "manifest_path"),
    ],
)
def test_configured_durable_outputs_deny_aliases_but_allow_external_locations(
    config_path: Path, tmp_path: Path, section: str, field: str
) -> None:
    vault = config_path.parent.parent / "vault"
    reserved = vault / "_wiki"
    reserved.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = outside / "alias"
    alias.symlink_to(reserved, target_is_directory=True)
    with pytest.raises(ValidationError, match="Ordinary writes cannot target"):
        Settings.from_toml(config_path, **{section: {field: str(alias / "state.json")}})
    destination = outside / "ordinary.json"
    settings = Settings.from_toml(config_path, **{section: {field: str(destination)}})
    assert Path(getattr(getattr(settings, section), field)) == destination
    assert not destination.exists()
    assert list(reserved.iterdir()) == []


def test_folder_preflight_preserves_safe_selected_alias_spelling(config_path: Path) -> None:
    vault = config_path.parent.parent / "vault"
    target = vault / "ordinary"
    target.mkdir()
    (vault / "alias").symlink_to(target, target_is_directory=True)
    settings = Settings.from_toml(config_path, vault={"inbox_folder": "alias"})
    assert settings.vault.inbox_folder == "alias"
    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "filename",
    ["embedding_cache.db", "sessions.db", "lineage.json", "tag_vocabulary.json", "mcp_audit.jsonl"],
)
def test_fixed_state_defaults_cannot_alias_wiki_storage(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    import vaultmind.config as config_module

    home = tmp_path / "runtime"
    data = home / "data"
    data.mkdir(parents=True)
    monkeypatch.setattr(config_module, "VAULTMIND_HOME", home)
    vault = config_path.parent.parent / "vault"
    reserved = vault / "_wiki"
    reserved.mkdir()
    retained = reserved / "retained"
    retained.write_bytes(b"unaltered")
    (data / filename).symlink_to(retained)
    with pytest.raises(ValidationError, match="Ordinary writes cannot target"):
        load_settings(config_path)
    assert retained.read_bytes() == b"unaltered"


def test_configured_state_loop_fails_without_creating_outputs(config_path: Path) -> None:
    vault = config_path.parent.parent / "vault"
    cycle = vault / "cycle"
    cycle.symlink_to(cycle)
    with pytest.raises(ValidationError, match="Path cannot be resolved"):
        Settings.from_toml(config_path, autonomy={"db_path": str(cycle / "queue.db")})
    assert cycle.is_symlink()
    assert not (vault / "_wiki").exists()


def test_output_preflight_preserves_explicit_empty_sentinels_and_relative_paths(
    config_path: Path,
) -> None:
    settings = Settings.from_toml(
        config_path,
        tracking={"db_path": ""},
        bench={"trend_path": ""},
        scheduler={"state_path": ""},
        autonomy={"db_path": "relative/queue.db"},
    )
    assert settings.tracking.db_path == ""
    assert settings.bench.trend_path == ""
    assert settings.scheduler.state_path == ""
    assert settings.autonomy.db_path == "relative/queue.db"
    assert not (Path.cwd() / "relative").exists()


@pytest.mark.parametrize("filename", ["maturation_state.json", "evolution_dismissed.json"])
def test_home_relative_state_defaults_cannot_alias_wiki_storage(
    config_path: Path, filename: str
) -> None:
    data = Path.home() / ".vaultmind" / "data"
    data.mkdir(parents=True)
    vault = config_path.parent.parent / "vault"
    reserved = vault / "_wiki"
    reserved.mkdir()
    retained = reserved / "retained.json"
    retained.write_bytes(b'{"preserved":true}')
    (data / filename).symlink_to(retained)
    with pytest.raises(ValidationError, match="Ordinary writes cannot target"):
        load_settings(config_path)
    assert retained.read_bytes() == b'{"preserved":true}'
