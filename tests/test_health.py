"""Tests for vaultmind.services.health — health monitoring subsystem."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vaultmind.config import HealthConfig
from vaultmind.services.health import (
    HealthAlert,
    HealthMonitor,
    HealthStatus,
)


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    d = tmp_path / "vault"
    d.mkdir()
    return d


@pytest.fixture
def graph_file(tmp_path: Path) -> Path:
    f = tmp_path / "graph.json"
    f.write_text('{"nodes": [], "edges": []}')
    return f


@pytest.fixture
def monitor(vault_dir: Path, graph_file: Path) -> HealthMonitor:
    return HealthMonitor(
        vault_root=vault_dir,
        graph_persist_path=graph_file,
        check_chromadb=False,
        check_sqlite=False,
        check_llm=False,
        check_bot=False,
    )


# ---------------------------------------------------------------------------
# A. Individual checks
# ---------------------------------------------------------------------------


class TestIndividualChecks:
    def test_check_vault_access_existing_dir_passes(
        self, vault_dir: Path, graph_file: Path
    ) -> None:
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_graph_file=False,
        )
        report = m.run_check()
        vault_signal = [s for s in report.signals if s.check_type.value == "vault_access"]
        assert len(vault_signal) == 1
        assert vault_signal[0].status == "pass"

    def test_check_vault_access_missing_dir_fails(self, tmp_path: Path, graph_file: Path) -> None:
        missing = tmp_path / "nonexistent"
        m = HealthMonitor(
            vault_root=missing,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_graph_file=False,
        )
        report = m.run_check()
        vault_signal = [s for s in report.signals if s.check_type.value == "vault_access"]
        assert len(vault_signal) == 1
        assert vault_signal[0].status == "fail"
        assert vault_signal[0].severity == 0.9

    def test_check_graph_file_existing_file_passes(self, vault_dir: Path, graph_file: Path) -> None:
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
        )
        report = m.run_check()
        gf_signal = [s for s in report.signals if s.check_type.value == "graph_file"]
        assert len(gf_signal) == 1
        assert gf_signal[0].status == "pass"

    def test_check_graph_file_missing_file_fails(self, vault_dir: Path, tmp_path: Path) -> None:
        missing = tmp_path / "no_graph.json"
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=missing,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
        )
        report = m.run_check()
        gf_signal = [s for s in report.signals if s.check_type.value == "graph_file"]
        assert len(gf_signal) == 1
        assert gf_signal[0].status == "fail"

    def test_check_graph_file_empty_file_fails(self, vault_dir: Path, tmp_path: Path) -> None:
        empty = tmp_path / "empty_graph.json"
        empty.write_text("")
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=empty,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
        )
        report = m.run_check()
        gf_signal = [s for s in report.signals if s.check_type.value == "graph_file"]
        assert len(gf_signal) == 1
        assert gf_signal[0].status == "fail"
        assert gf_signal[0].severity == 0.5

    def test_check_chromadb_skipped_when_no_store(self, vault_dir: Path, graph_file: Path) -> None:
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=True,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
            check_graph_file=False,
            store=None,
        )
        report = m.run_check()
        chroma_signal = [s for s in report.signals if s.check_type.value == "chromadb"]
        assert len(chroma_signal) == 1
        assert chroma_signal[0].status == "pass"

    def test_check_chromadb_success_returns_pass(self, vault_dir: Path, graph_file: Path) -> None:
        mock_store = MagicMock()
        mock_store.search.return_value = []
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=True,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
            check_graph_file=False,
            store=mock_store,
        )
        report = m.run_check()
        chroma_signal = [s for s in report.signals if s.check_type.value == "chromadb"]
        assert len(chroma_signal) == 1
        assert chroma_signal[0].status == "pass"
        assert chroma_signal[0].latency_ms is not None
        assert chroma_signal[0].latency_ms >= 0

    def test_check_chromadb_exception_returns_fail(self, vault_dir: Path, graph_file: Path) -> None:
        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("connection refused")
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=True,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
            check_graph_file=False,
            store=mock_store,
        )
        report = m.run_check()
        chroma_signal = [s for s in report.signals if s.check_type.value == "chromadb"]
        assert len(chroma_signal) == 1
        assert chroma_signal[0].status == "fail"
        assert chroma_signal[0].severity == 0.9

    def test_check_bot_token_set_passes(self, vault_dir: Path, graph_file: Path) -> None:
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=True,
            check_vault_access=False,
            check_graph_file=False,
            bot_token="123:ABC",
        )
        report = m.run_check()
        bot_signal = [s for s in report.signals if s.check_type.value == "bot"]
        assert len(bot_signal) == 1
        assert bot_signal[0].status == "pass"

    def test_check_bot_token_empty_fails(self, vault_dir: Path, graph_file: Path) -> None:
        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=True,
            check_vault_access=False,
            check_graph_file=False,
            bot_token="",
        )
        report = m.run_check()
        bot_signal = [s for s in report.signals if s.check_type.value == "bot"]
        assert len(bot_signal) == 1
        assert bot_signal[0].status == "fail"


# ---------------------------------------------------------------------------
# B. Overall status
# ---------------------------------------------------------------------------


class TestOverallStatus:
    def test_all_passing_returns_healthy(self, monitor: HealthMonitor) -> None:
        report = monitor.run_check()
        assert report.overall_status == HealthStatus.HEALTHY

    def test_one_failure_returns_healthy(self, tmp_path: Path, graph_file: Path) -> None:
        """One low-severity failure keeps overall HEALTHY."""
        missing_vault = tmp_path / "gone"
        m = HealthMonitor(
            vault_root=missing_vault,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
        )
        report = m.run_check()
        fail_count = sum(1 for s in report.signals if s.status == "fail")
        assert fail_count >= 1
        assert report.overall_status == HealthStatus.HEALTHY

    def test_three_failures_returns_degraded(self, tmp_path: Path) -> None:
        """Three or more failures trigger DEGRADED."""
        missing_vault = tmp_path / "gone"
        missing_graph = tmp_path / "no_graph.json"
        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("boom")
        m = HealthMonitor(
            vault_root=missing_vault,
            graph_persist_path=missing_graph,
            check_chromadb=True,
            check_sqlite=False,
            check_llm=False,
            check_bot=True,
            store=mock_store,
            bot_token="",
        )
        report = m.run_check()
        fail_count = sum(1 for s in report.signals if s.status == "fail")
        assert fail_count >= 3
        assert report.overall_status in (
            HealthStatus.DEGRADED,
            HealthStatus.CRITICAL,
        )

    def test_two_critical_failures_returns_critical(self, tmp_path: Path) -> None:
        """Two failures with severity >= 0.7 trigger CRITICAL."""
        missing_vault = tmp_path / "gone"
        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("boom")
        m = HealthMonitor(
            vault_root=missing_vault,
            graph_persist_path=tmp_path / "graph.json",
            check_chromadb=True,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_graph_file=False,
            store=mock_store,
        )
        report = m.run_check()
        critical_count = sum(1 for s in report.signals if s.status == "fail" and s.severity >= 0.7)
        assert critical_count >= 2
        assert report.overall_status == HealthStatus.CRITICAL


# ---------------------------------------------------------------------------
# C. Alert transitions
# ---------------------------------------------------------------------------


class TestAlertTransitions:
    @pytest.mark.asyncio
    async def test_pass_to_fail_emits_alert(self, vault_dir: Path, tmp_path: Path) -> None:
        graph_file = tmp_path / "graph.json"
        graph_file.write_text('{"nodes": []}')
        alerts: list[HealthAlert] = []

        async def collector(alert: HealthAlert) -> None:
            alerts.append(alert)

        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
        )
        m.on_alert(collector)

        # First run: pass
        m.run_check()
        await m.dispatch_alerts()
        assert len(alerts) == 0

        # Remove graph file to cause failure
        graph_file.unlink()
        m.run_check()
        await m.dispatch_alerts()
        assert len(alerts) == 1
        assert alerts[0].check_type.value == "graph_file"

    @pytest.mark.asyncio
    async def test_fail_to_pass_emits_recovery_alert(self, vault_dir: Path, tmp_path: Path) -> None:
        graph_file = tmp_path / "graph.json"
        alerts: list[HealthAlert] = []

        async def collector(alert: HealthAlert) -> None:
            alerts.append(alert)

        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
        )
        m.on_alert(collector)

        # First run: fail (file missing)
        m.run_check()
        await m.dispatch_alerts()
        assert len(alerts) == 0  # no baseline yet

        # Restore graph file
        graph_file.write_text('{"nodes": []}')
        m.run_check()
        await m.dispatch_alerts()
        assert len(alerts) == 1
        assert "recovered" in alerts[0].message

    @pytest.mark.asyncio
    async def test_no_transition_no_alert(self, vault_dir: Path, graph_file: Path) -> None:
        alerts: list[HealthAlert] = []

        async def collector(alert: HealthAlert) -> None:
            alerts.append(alert)

        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=graph_file,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
        )
        m.on_alert(collector)

        m.run_check()
        await m.dispatch_alerts()
        m.run_check()
        await m.dispatch_alerts()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_first_report_no_alerts(self, vault_dir: Path, tmp_path: Path) -> None:
        missing_graph = tmp_path / "no_graph.json"
        alerts: list[HealthAlert] = []

        async def collector(alert: HealthAlert) -> None:
            alerts.append(alert)

        m = HealthMonitor(
            vault_root=vault_dir,
            graph_persist_path=missing_graph,
            check_chromadb=False,
            check_sqlite=False,
            check_llm=False,
            check_bot=False,
            check_vault_access=False,
        )
        m.on_alert(collector)

        m.run_check()
        await m.dispatch_alerts()
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# D. HealthConfig
# ---------------------------------------------------------------------------


class TestHealthConfig:
    def test_default_config_valid(self) -> None:
        cfg = HealthConfig()
        assert cfg.enabled is True
        assert cfg.check_interval_seconds == 60

    def test_config_fields_present(self) -> None:
        expected_fields = {
            "enabled",
            "check_interval_seconds",
            "check_chromadb",
            "check_sqlite",
            "check_graph_file",
            "check_llm",
            "check_bot",
            "check_vault_access",
            "chromadb_latency_warn_ms",
            "retention_days",
        }
        actual_fields = set(HealthConfig.model_fields.keys())
        assert expected_fields.issubset(actual_fields)
