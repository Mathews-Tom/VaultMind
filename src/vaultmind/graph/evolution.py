"""Belief evolution detection.

Identifies confidence drift, relationship shifts, and stale claims.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vaultmind.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BeliefEvolution:
    """A detected belief evolution signal."""

    entity_a: str
    entity_b: str
    signal_type: Literal["confidence_drift", "relationship_shift", "stale_claim"]
    detail: str
    source_notes: list[str] = field(hash=False)
    severity: float  # 0.0-1.0
    detected_at: datetime
    evolution_id: str


class EvolutionDetector:
    """Detects belief evolution signals from knowledge graph edge data.

    Three signals:
    1. Confidence drift: same entity pair, same relation,
       different confidence from different sources
    2. Relationship shift: same entity pair, different relation types from different sources
    3. Stale high-confidence: high-confidence claim whose source notes are old
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        confidence_drift_threshold: float = 0.3,
        stale_days: int = 180,
        min_confidence_for_stale: float = 0.8,
        store_path: Path | None = None,
    ) -> None:
        self._graph = knowledge_graph
        self._drift_threshold = confidence_drift_threshold
        self._stale_days = stale_days
        self._min_stale_confidence = min_confidence_for_stale
        default_path = Path.home() / ".vaultmind" / "data" / "evolution_dismissed.json"
        self._store_path = store_path or default_path
        self._dismissed: set[str] = self._load_dismissed()

    def _load_dismissed(self) -> set[str]:
        if self._store_path.exists():
            data = json.loads(self._store_path.read_text())
            return set(data.get("dismissed", []))
        return set()

    def _save_dismissed(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(json.dumps({"dismissed": sorted(self._dismissed)}, indent=2))

    def scan(self) -> list[BeliefEvolution]:
        """Single-pass O(E) scan over all graph edges for evolution signals.

        Detects confidence drift and relationship shifts by examining edges
        where multiple source notes contributed (indicating the relationship
        was observed in different contexts). Since DiGraph merges edges,
        drift is detected when an edge has multiple sources and high confidence
        variance would have been flattened by the max() merge strategy.

        For multi-edge scenarios (when edges are added directly to the graph
        bypassing the merge logic), groups edges by (src, tgt) pair.
        """
        g = self._graph._graph
        now = datetime.now(UTC)
        signals: list[BeliefEvolution] = []

        # Group edges by (source, target) pair
        edge_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for src, tgt, data in g.edges(data=True):
            edge_groups[(src, tgt)].append(dict(data))

        for (src, tgt), edges in edge_groups.items():
            src_label = str(g.nodes[src].get("label", src))
            tgt_label = str(g.nodes[tgt].get("label", tgt))

            if len(edges) >= 2:
                # Check confidence drift across multiple edges
                confidences = [float(e.get("confidence", 0.5)) for e in edges]
                drift = max(confidences) - min(confidences)
                if drift >= self._drift_threshold:
                    all_sources: list[str] = []
                    for e in edges:
                        all_sources.extend(str(s) for s in (e.get("source_notes") or []))
                    evo_id = self._make_id(src_label, tgt_label, "confidence_drift", all_sources)
                    if not self._is_dismissed(evo_id):
                        signals.append(
                            BeliefEvolution(
                                entity_a=src_label,
                                entity_b=tgt_label,
                                signal_type="confidence_drift",
                                detail=(
                                    f"Confidence ranges from {min(confidences):.1f} to "
                                    f"{max(confidences):.1f} across {len(all_sources)} source(s)"
                                ),
                                source_notes=all_sources,
                                severity=min(drift / 0.5, 1.0),
                                detected_at=now,
                                evolution_id=evo_id,
                            )
                        )

                # Check relationship shift across multiple edges
                relations = {str(e.get("relation", "related_to")) for e in edges}
                if len(relations) >= 2:
                    all_sources = []
                    for e in edges:
                        all_sources.extend(str(s) for s in (e.get("source_notes") or []))
                    evo_id = self._make_id(src_label, tgt_label, "relationship_shift", all_sources)
                    if not self._is_dismissed(evo_id):
                        signals.append(
                            BeliefEvolution(
                                entity_a=src_label,
                                entity_b=tgt_label,
                                signal_type="relationship_shift",
                                detail=f"Relationship changed: {' vs '.join(sorted(relations))}",
                                source_notes=all_sources,
                                severity=0.8,
                                detected_at=now,
                                evolution_id=evo_id,
                            )
                        )

        # Sort by severity descending
        signals.sort(key=lambda s: s.severity, reverse=True)
        return signals

    def scan_with_file_ages(
        self,
        file_modified: dict[str, datetime],
    ) -> list[BeliefEvolution]:
        """Scan including stale claim detection using file modification times.

        Args:
            file_modified: Mapping of note path -> last modified datetime.
        """
        g = self._graph._graph
        now = datetime.now(UTC)
        signals = self.scan()  # Get drift and shift signals

        # Add stale claim signals
        for src, tgt, data in g.edges(data=True):
            confidence = float(data.get("confidence", 0.5))
            if confidence < self._min_stale_confidence:
                continue

            source_notes: list[str] = [str(s) for s in (data.get("source_notes") or [])]
            if not source_notes:
                continue

            src_label = str(g.nodes[src].get("label", src))
            tgt_label = str(g.nodes[tgt].get("label", tgt))

            # Check if ALL source notes are old
            all_stale = True
            oldest_days = 0
            for note_path in source_notes:
                mod_time = file_modified.get(note_path)
                if mod_time is None:
                    all_stale = False
                    break
                if mod_time.tzinfo is None:
                    mod_time = mod_time.replace(tzinfo=UTC)
                age_days = (now - mod_time).days
                oldest_days = max(oldest_days, age_days)
                if age_days < self._stale_days:
                    all_stale = False
                    break

            if all_stale and oldest_days >= self._stale_days:
                evo_id = self._make_id(src_label, tgt_label, "stale_claim", source_notes)
                if not self._is_dismissed(evo_id):
                    signals.append(
                        BeliefEvolution(
                            entity_a=src_label,
                            entity_b=tgt_label,
                            signal_type="stale_claim",
                            detail=(
                                f"{data.get('relation', 'related_to')} ({confidence:.2f}) "
                                f"-- last reinforced {oldest_days} days ago"
                            ),
                            source_notes=source_notes,
                            severity=min(oldest_days / 365, 1.0),
                            detected_at=now,
                            evolution_id=evo_id,
                        )
                    )

        signals.sort(key=lambda s: s.severity, reverse=True)
        return signals

    def dismiss(self, evolution_id_prefix: str) -> bool:
        """Dismiss a signal by ID prefix (first 8+ chars).

        Returns True if a matching ID was found and dismissed.
        """
        for signal in self.scan():
            if signal.evolution_id.startswith(evolution_id_prefix):
                self._dismissed.add(signal.evolution_id)
                self._save_dismissed()
                return True
        return False

    def dismiss_by_id(self, evolution_id: str) -> None:
        """Dismiss by full evolution ID."""
        self._dismissed.add(evolution_id)
        self._save_dismissed()

    def _is_dismissed(self, evolution_id: str) -> bool:
        return evolution_id in self._dismissed

    @staticmethod
    def _make_id(
        entity_a: str,
        entity_b: str,
        signal_type: str,
        source_notes: list[str],
    ) -> str:
        raw = f"{entity_a}:{entity_b}:{signal_type}:{':'.join(sorted(source_notes))}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
