"""Tests for sources/registry.py and sources/store.py — closed-set enforcement,
config loading, and durable cursor storage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from vaultmind.sources.models import (
    ConnectorDefinition,
    ConnectorState,
    FetchResult,
    RunSummary,
    SourceInstance,
)
from vaultmind.sources.registry import (
    ALLOWED_KINDS,
    REGISTRY,
    get_connector,
    load_source_instances,
    register_connector,
)
from vaultmind.sources.store import SourceStore, advance_cursor

if TYPE_CHECKING:
    from pathlib import Path


async def _noop_fetch(  # pragma: no cover - stub
    instance: SourceInstance, state: ConnectorState
) -> FetchResult:
    raise NotImplementedError


class TestAllowedKinds:
    def test_closed_set_has_exactly_three_kinds(self) -> None:
        assert frozenset({"rss", "youtube-channel", "github-activity"}) == ALLOWED_KINDS

    def test_allowed_kinds_is_frozen(self) -> None:
        assert isinstance(ALLOWED_KINDS, frozenset)


class TestRegisterConnector:
    def test_registers_known_kind(self) -> None:
        REGISTRY.pop("rss", None)
        register_connector(ConnectorDefinition(kind="rss", fetch=_noop_fetch, description="test"))
        assert "rss" in REGISTRY
        assert REGISTRY["rss"].description == "test"

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="Unknown connector kind"):
            register_connector(
                ConnectorDefinition(kind="email", fetch=_noop_fetch, description="not allowed")
            )
        assert "email" not in REGISTRY


class TestGetConnector:
    def test_raises_for_unregistered_kind(self) -> None:
        REGISTRY.pop("github-activity", None)
        with pytest.raises(KeyError, match="No connector implementation registered"):
            get_connector("github-activity")

    def test_returns_registered_definition(self) -> None:
        REGISTRY.pop("youtube-channel", None)
        definition = ConnectorDefinition(
            kind="youtube-channel", fetch=_noop_fetch, description="yt"
        )
        register_connector(definition)
        assert get_connector("youtube-channel") is definition


class TestLoadSourceInstances:
    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        assert load_source_instances(tmp_path / "nonexistent.toml") == []

    def test_loads_valid_instance(self, tmp_path: Path) -> None:
        config = tmp_path / "sources.toml"
        config.write_text(
            """
            [[instance]]
            name = "rss-fixture"
            kind = "rss"
            target = "tests/fixtures/sources/rss_feed.xml"
            enabled = true
            interval_hours = 12
            output_folder = "00-inbox/sources"
            """
        )
        instances = load_source_instances(config)
        assert len(instances) == 1
        inst = instances[0]
        assert inst.name == "rss-fixture"
        assert inst.kind == "rss"
        assert inst.target == "tests/fixtures/sources/rss_feed.xml"
        assert inst.enabled is True
        assert inst.interval_hours == 12

    def test_defaults_when_optional_fields_omitted(self, tmp_path: Path) -> None:
        config = tmp_path / "sources.toml"
        config.write_text(
            """
            [[instance]]
            name = "minimal"
            kind = "rss"
            target = "https://example.com/feed.xml"
            """
        )
        inst = load_source_instances(config)[0]
        assert inst.enabled is False
        assert inst.interval_hours == 24
        assert inst.output_folder == "00-inbox/sources"

    def test_rejects_unknown_kind(self, tmp_path: Path) -> None:
        config = tmp_path / "sources.toml"
        config.write_text(
            """
            [[instance]]
            name = "bad"
            kind = "email"
            target = "https://example.com"
            """
        )
        with pytest.raises(ValueError, match="unknown kind"):
            load_source_instances(config)

    def test_rejects_duplicate_name(self, tmp_path: Path) -> None:
        config = tmp_path / "sources.toml"
        config.write_text(
            """
            [[instance]]
            name = "dup"
            kind = "rss"
            target = "https://example.com/a.xml"

            [[instance]]
            name = "dup"
            kind = "rss"
            target = "https://example.com/b.xml"
            """
        )
        with pytest.raises(ValueError, match="duplicate instance name"):
            load_source_instances(config)

    def test_rejects_missing_name(self, tmp_path: Path) -> None:
        config = tmp_path / "sources.toml"
        config.write_text(
            """
            [[instance]]
            kind = "rss"
            target = "https://example.com/a.xml"
            """
        )
        with pytest.raises(ValueError, match="requires a non-empty 'name'"):
            load_source_instances(config)

    def test_rejects_missing_target(self, tmp_path: Path) -> None:
        config = tmp_path / "sources.toml"
        config.write_text(
            """
            [[instance]]
            name = "no-target"
            kind = "rss"
            """
        )
        with pytest.raises(ValueError, match="requires a non-empty 'target'"):
            load_source_instances(config)

    def test_no_instance_tables_returns_empty_list(self, tmp_path: Path) -> None:
        config = tmp_path / "sources.toml"
        config.write_text("# no instances configured\n")
        assert load_source_instances(config) == []


class TestSourceStore:
    def test_unknown_instance_returns_fresh_empty_cursor(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "sources.db")
        state = store.get_state("never-run")
        assert state.instance_name == "never-run"
        assert state.last_seen_id == ""
        assert state.etag == ""
        assert state.last_run is None
        assert state.run_count == 0
        store.close()

    def test_save_and_get_state_roundtrip(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "sources.db")
        now = datetime.now(UTC)
        store.save_state(
            ConnectorState(
                instance_name="rss-fixture",
                last_seen_id="fixture-post-3",
                etag="etag-123",
                last_run=now,
                run_count=1,
            )
        )
        state = store.get_state("rss-fixture")
        assert state.last_seen_id == "fixture-post-3"
        assert state.etag == "etag-123"
        assert state.run_count == 1
        assert state.last_run is not None
        store.close()

    def test_save_state_upserts(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "sources.db")
        store.save_state(ConnectorState(instance_name="x", last_seen_id="a", run_count=1))
        store.save_state(ConnectorState(instance_name="x", last_seen_id="b", run_count=2))
        state = store.get_state("x")
        assert state.last_seen_id == "b"
        assert state.run_count == 2
        store.close()

    def test_record_and_list_runs_most_recent_first(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "sources.db")
        now = datetime.now(UTC)
        for i in range(3):
            store.record_run(
                RunSummary(
                    instance_name="rss-fixture",
                    started=now,
                    finished=now,
                    items_fetched=i,
                    items_ingested=i,
                )
            )
        runs = store.list_runs("rss-fixture", limit=10)
        assert len(runs) == 3
        assert [r.items_fetched for r in runs] == [2, 1, 0]
        store.close()

    def test_run_history_bounded_per_instance(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "sources.db")
        now = datetime.now(UTC)
        for i in range(25):
            store.record_run(
                RunSummary(
                    instance_name="rss-fixture",
                    started=now,
                    finished=now,
                    items_fetched=i,
                    items_ingested=i,
                )
            )
        runs = store.list_runs("rss-fixture", limit=100)
        assert len(runs) == 20
        assert runs[0].items_fetched == 24
        store.close()

    def test_advance_cursor_updates_last_seen_id_and_etag(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "sources.db")
        advance_cursor(store, "rss-fixture", next_cursor_id="fixture-post-3", next_etag="e1")
        state = store.get_state("rss-fixture")
        assert state.last_seen_id == "fixture-post-3"
        assert state.etag == "e1"
        assert state.run_count == 1
        assert state.last_run is not None
        store.close()

    def test_advance_cursor_none_leaves_prior_cursor_unchanged(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "sources.db")
        advance_cursor(store, "rss-fixture", next_cursor_id="fixture-post-3", next_etag="e1")
        advance_cursor(store, "rss-fixture", next_cursor_id=None, next_etag=None)
        state = store.get_state("rss-fixture")
        assert state.last_seen_id == "fixture-post-3"
        assert state.etag == "e1"
        assert state.run_count == 2
        store.close()
