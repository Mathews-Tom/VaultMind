"""Periodic health monitoring for VaultMind subsystems."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class CheckType(StrEnum):
    CHROMADB = "chromadb"
    SQLITE = "sqlite"
    GRAPH_FILE = "graph_file"
    LLM = "llm"
    BOT = "bot"
    VAULT_ACCESS = "vault_access"


@dataclass(frozen=True, slots=True)
class HealthSignal:
    check_type: CheckType
    status: str  # "pass", "fail", "timeout"
    latency_ms: float | None = None
    message: str = ""
    severity: float = 0.0  # 0.0-1.0


@dataclass(frozen=True, slots=True)
class HealthReport:
    timestamp: str  # ISO format
    overall_status: HealthStatus
    signals: tuple[HealthSignal, ...]
    check_count: int = 0


@dataclass(frozen=True, slots=True)
class HealthAlert:
    severity: str  # "warning", "error", "critical"
    check_type: CheckType
    message: str
    suggested_action: str = ""


class HealthMonitor:
    """Periodic health checker for VaultMind subsystems."""

    def __init__(
        self,
        vault_root: Path,
        graph_persist_path: Path,
        check_chromadb: bool = True,
        check_sqlite: bool = True,
        check_graph_file: bool = True,
        check_llm: bool = True,
        check_bot: bool = True,
        check_vault_access: bool = True,
        chromadb_latency_warn_ms: int = 1000,
        store: Any = None,
        llm_client: Any = None,
        bot_token: str = "",
    ) -> None:
        self._vault_root = vault_root
        self._graph_persist_path = graph_persist_path
        self._check_chromadb_enabled = check_chromadb
        self._check_sqlite_enabled = check_sqlite
        self._check_graph_file_enabled = check_graph_file
        self._check_llm_enabled = check_llm
        self._check_bot_enabled = check_bot
        self._check_vault_access_enabled = check_vault_access
        self._chromadb_latency_warn_ms = chromadb_latency_warn_ms
        self._store: Any = store
        self._llm_client: Any = llm_client
        self._bot_token = bot_token
        self._last_report: HealthReport | None = None
        self._alert_handlers: list[Callable[[HealthAlert], Awaitable[None]]] = []
        self._pending_alerts: list[HealthAlert] = []

    def _check_vault_access(self) -> HealthSignal:
        if not self._vault_root.exists():
            return HealthSignal(
                check_type=CheckType.VAULT_ACCESS,
                status="fail",
                message=f"Vault root does not exist: {self._vault_root}",
                severity=0.9,
            )
        if not os.access(self._vault_root, os.R_OK | os.W_OK):
            return HealthSignal(
                check_type=CheckType.VAULT_ACCESS,
                status="fail",
                message=f"Vault root not readable/writable: {self._vault_root}",
                severity=0.7,
            )
        return HealthSignal(
            check_type=CheckType.VAULT_ACCESS,
            status="pass",
            message="Vault accessible",
        )

    def _check_graph_file(self) -> HealthSignal:
        if not self._graph_persist_path.exists():
            return HealthSignal(
                check_type=CheckType.GRAPH_FILE,
                status="fail",
                message=f"Graph file not found: {self._graph_persist_path}",
                severity=0.3,
            )
        if self._graph_persist_path.stat().st_size == 0:
            return HealthSignal(
                check_type=CheckType.GRAPH_FILE,
                status="fail",
                message="Graph file is empty",
                severity=0.5,
            )
        return HealthSignal(
            check_type=CheckType.GRAPH_FILE,
            status="pass",
            message="Graph file present",
        )

    def _check_chromadb(self) -> HealthSignal:
        if self._store is None:
            return HealthSignal(
                check_type=CheckType.CHROMADB,
                status="pass",
                message="ChromaDB check skipped (no store)",
            )
        try:
            start = time.perf_counter()
            self._store.search("health_check_ping", n_results=1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if elapsed_ms > self._chromadb_latency_warn_ms:
                return HealthSignal(
                    check_type=CheckType.CHROMADB,
                    status="fail",
                    latency_ms=elapsed_ms,
                    message=(
                        f"ChromaDB latency {elapsed_ms:.0f}ms"
                        f" exceeds threshold {self._chromadb_latency_warn_ms}ms"
                    ),
                    severity=0.5,
                )
            return HealthSignal(
                check_type=CheckType.CHROMADB,
                status="pass",
                latency_ms=elapsed_ms,
                message="ChromaDB responsive",
            )
        except Exception as exc:
            return HealthSignal(
                check_type=CheckType.CHROMADB,
                status="fail",
                message=f"ChromaDB error: {exc}",
                severity=0.9,
            )

    def _check_sqlite(self) -> HealthSignal:
        data_dir = Path.home() / ".vaultmind" / "data"
        if not data_dir.exists():
            return HealthSignal(
                check_type=CheckType.SQLITE,
                status="fail",
                message=f"SQLite data directory not found: {data_dir}",
                severity=0.4,
            )
        return HealthSignal(
            check_type=CheckType.SQLITE,
            status="pass",
            message="SQLite data directory present",
        )

    def _check_llm(self) -> HealthSignal:
        if self._llm_client is None:
            return HealthSignal(
                check_type=CheckType.LLM,
                status="pass",
                message="LLM check skipped (no client)",
            )
        return HealthSignal(
            check_type=CheckType.LLM,
            status="pass",
            message="LLM client configured",
        )

    def _check_bot(self) -> HealthSignal:
        if not self._bot_token:
            return HealthSignal(
                check_type=CheckType.BOT,
                status="fail",
                message="Bot token not configured",
                severity=0.5,
            )
        return HealthSignal(
            check_type=CheckType.BOT,
            status="pass",
            message="Bot token set",
        )

    def run_check(self) -> HealthReport:
        """Execute all enabled health checks and return a report."""
        signals: list[HealthSignal] = []

        if self._check_vault_access_enabled:
            signals.append(self._check_vault_access())
        if self._check_graph_file_enabled:
            signals.append(self._check_graph_file())
        if self._check_chromadb_enabled:
            signals.append(self._check_chromadb())
        if self._check_sqlite_enabled:
            signals.append(self._check_sqlite())
        if self._check_llm_enabled:
            signals.append(self._check_llm())
        if self._check_bot_enabled:
            signals.append(self._check_bot())

        fail_count = sum(1 for s in signals if s.status == "fail")
        critical_count = sum(1 for s in signals if s.status == "fail" and s.severity >= 0.7)

        if critical_count >= 2:
            overall = HealthStatus.CRITICAL
        elif fail_count >= 3:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        report = HealthReport(
            timestamp=datetime.now(UTC).isoformat(),
            overall_status=overall,
            signals=tuple(signals),
            check_count=len(signals),
        )

        self._detect_transitions(report)
        self._last_report = report

        return report

    def _detect_transitions(self, report: HealthReport) -> None:
        """Detect pass->fail and fail->pass transitions vs previous report."""
        if self._last_report is None:
            return

        old_map = {s.check_type: s.status for s in self._last_report.signals}
        for signal in report.signals:
            old_status = old_map.get(signal.check_type)
            if old_status == "pass" and signal.status == "fail":
                alert = HealthAlert(
                    severity="error" if signal.severity >= 0.7 else "warning",
                    check_type=signal.check_type,
                    message=signal.message or f"{signal.check_type.value} check failed",
                    suggested_action=self._suggest_action(signal.check_type),
                )
                self._emit_alert(alert)
            elif old_status == "fail" and signal.status == "pass":
                alert = HealthAlert(
                    severity="warning",
                    check_type=signal.check_type,
                    message=f"{signal.check_type.value} recovered",
                )
                self._emit_alert(alert)

    def _suggest_action(self, check_type: CheckType) -> str:
        suggestions: dict[CheckType, str] = {
            CheckType.CHROMADB: "Check ChromaDB disk space and connectivity.",
            CheckType.SQLITE: "Verify ~/.vaultmind/data/ directory permissions.",
            CheckType.GRAPH_FILE: "Run 'vaultmind graph-maintain' to rebuild.",
            CheckType.LLM: "Check API key and network connectivity.",
            CheckType.BOT: "Verify VAULTMIND_TELEGRAM__BOT_TOKEN is set.",
            CheckType.VAULT_ACCESS: "Check vault directory permissions.",
        }
        return suggestions.get(check_type, "")

    def on_alert(self, handler: Callable[[HealthAlert], Awaitable[None]]) -> None:
        """Register an async alert handler."""
        self._alert_handlers.append(handler)

    def _emit_alert(self, alert: HealthAlert) -> None:
        self._pending_alerts.append(alert)

    async def dispatch_alerts(self) -> None:
        """Dispatch any pending alerts to registered handlers."""
        alerts = list(self._pending_alerts)
        self._pending_alerts.clear()
        for alert in alerts:
            for handler in self._alert_handlers:
                try:
                    await handler(alert)
                except Exception:
                    logger.warning("Alert handler failed", exc_info=True)

    @property
    def last_report(self) -> HealthReport | None:
        return self._last_report
