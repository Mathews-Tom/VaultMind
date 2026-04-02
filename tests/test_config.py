"""Tests for root configuration loading and architectural control-plane defaults."""

from __future__ import annotations

from pathlib import Path

from vaultmind.config import Settings, VaultConfig


class TestSettingsArchitecturalSections:
    def test_new_sections_exist_in_root_settings(self) -> None:
        assert "memory_profile" in Settings.model_fields
        assert "proactivity" in Settings.model_fields
        assert "heartbeat" in Settings.model_fields
        assert "drafts" in Settings.model_fields
        assert "integrations" in Settings.model_fields

    def test_defaults_are_conservative(self) -> None:
        settings = Settings(vault=VaultConfig(path=Path.cwd()))

        assert settings.memory_profile.identity_folder == "_meta/identity"
        assert settings.proactivity.mode == "observer"
        assert settings.proactivity.allow_local_automation is False
        assert settings.heartbeat.enabled is False
        assert settings.drafts.enabled is False
        assert settings.integrations.enabled is False
        assert settings.integrations.default_capability == "read"

    def test_from_toml_loads_new_sections(self, tmp_path: Path) -> None:
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        config_path = tmp_path / "settings.toml"
        config_path.write_text(
            """
[vault]
path = "__VAULT_PATH__"

[memory_profile]
promoted_memory_max_chars = 9000

[proactivity]
mode = "advisor"
allow_local_automation = true

[heartbeat]
enabled = true
max_actions_per_run = 7

[drafts]
enabled = true
expiry_hours = 12

[integrations]
enabled = true
default_capability = "draft"
""".replace("__VAULT_PATH__", str(vault_root)),
            encoding="utf-8",
        )

        settings = Settings.from_toml(config_path)

        assert settings.memory_profile.promoted_memory_max_chars == 9000
        assert settings.proactivity.mode == "advisor"
        assert settings.proactivity.allow_local_automation is True
        assert settings.heartbeat.enabled is True
        assert settings.heartbeat.max_actions_per_run == 7
        assert settings.drafts.enabled is True
        assert settings.drafts.expiry_hours == 12
        assert settings.integrations.enabled is True
        assert settings.integrations.default_capability == "draft"
