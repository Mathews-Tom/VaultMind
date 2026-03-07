"""Zettelkasten maturation pipeline orchestrator.

Coordinates cluster discovery, digest delivery, and synthesis.
Maintains persistent state for dismissed/synthesized clusters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vaultmind.pipeline.clustering import NoteCluster, discover_clusters

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

    from vaultmind.config import MaturationConfig
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class MaturationState:
    """Persistent state for the maturation pipeline."""

    last_run: str = ""
    dismissed_clusters: dict[str, str] = field(default_factory=dict)  # fingerprint -> iso date
    synthesized: list[str] = field(default_factory=list)  # fingerprints


def load_state(state_path: Path) -> MaturationState:
    """Load maturation state from JSON file."""
    if state_path.exists():
        data = json.loads(state_path.read_text())
        return MaturationState(
            last_run=data.get("last_run", ""),
            dismissed_clusters=data.get("dismissed_clusters", {}),
            synthesized=data.get("synthesized", []),
        )
    return MaturationState()


def save_state(state: MaturationState, state_path: Path) -> None:
    """Persist maturation state to JSON."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "last_run": state.last_run,
        "dismissed_clusters": state.dismissed_clusters,
        "synthesized": state.synthesized,
    }
    state_path.write_text(json.dumps(data, indent=2))


class MaturationPipeline:
    """Orchestrates the Zettelkasten maturation pipeline."""

    def __init__(
        self,
        config: MaturationConfig,
        collection: Collection,
        knowledge_graph: KnowledgeGraph,
        llm: LLMClient,
        vault_root: Path,
        state_path: Path | None = None,
    ) -> None:
        self._config = config
        self._collection = collection
        self._graph = knowledge_graph
        self._llm = llm
        self._vault_root = vault_root
        default_state = Path.home() / ".vaultmind" / "data" / "maturation_state.json"
        self._state_path = state_path or default_state
        self._state = load_state(self._state_path)

    def discover(self) -> list[NoteCluster]:
        """Run cluster discovery, filtering dismissed and synthesized clusters.

        Returns:
            Ranked list of actionable clusters.
        """
        # Expire old dismissed clusters
        self._expire_dismissed()

        all_clusters = discover_clusters(
            collection=self._collection,
            knowledge_graph=self._graph,
            target_types=self._config.target_note_types,
            eps=self._config.cluster_eps,
            min_samples=self._config.min_cluster_size,
        )

        # Filter out dismissed and already-synthesized
        excluded = set(self._state.dismissed_clusters.keys()) | set(self._state.synthesized)
        clusters = [c for c in all_clusters if c.fingerprint not in excluded]

        return clusters[: self._config.max_clusters_per_digest]

    def dismiss(self, fingerprint: str) -> None:
        """Dismiss a cluster so it won't be shown again (until expiry)."""
        self._state.dismissed_clusters[fingerprint] = datetime.now(UTC).isoformat()
        save_state(self._state, self._state_path)

    def synthesize(self, cluster: NoteCluster) -> str:
        """Synthesize a permanent note from a cluster.

        Returns:
            Path to the synthesized note or error message.
        """
        from vaultmind.pipeline.synthesis import synthesize_cluster

        result = synthesize_cluster(
            cluster=cluster,
            vault_root=self._vault_root,
            inbox_folder=self._config.inbox_folder,
            llm=self._llm,
            model=self._config.synthesis_model,
            max_tokens=self._config.synthesis_max_tokens,
        )

        if result.success:
            self._state.synthesized.append(cluster.fingerprint)
            self._state.last_run = datetime.now(UTC).isoformat()
            save_state(self._state, self._state_path)
            return result.output_path

        return f"Synthesis failed: {result.error}"

    def mark_run(self) -> None:
        """Record that a digest was delivered."""
        self._state.last_run = datetime.now(UTC).isoformat()
        save_state(self._state, self._state_path)

    def _expire_dismissed(self) -> None:
        """Remove dismissed clusters older than expiry threshold."""
        expiry_days = self._config.dismissed_cluster_expiry_days
        cutoff = datetime.now(UTC) - timedelta(days=expiry_days)
        expired = [
            fp
            for fp, ts in self._state.dismissed_clusters.items()
            if datetime.fromisoformat(ts) < cutoff
        ]
        for fp in expired:
            del self._state.dismissed_clusters[fp]
        if expired:
            save_state(self._state, self._state_path)
            logger.info("Expired %d dismissed clusters", len(expired))
