"""Cluster discovery for Zettelkasten maturation.

Extracts embeddings from ChromaDB for unprocessed notes, runs DBSCAN
clustering, and optionally merges clusters that share knowledge graph entities.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.cluster import DBSCAN

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

    from vaultmind.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NoteCluster:
    """A cluster of semantically related notes."""

    note_paths: list[str] = field(hash=False)
    note_titles: list[str] = field(hash=False)
    top_entity: str
    score: float
    fingerprint: str


def discover_clusters(
    collection: Collection,
    knowledge_graph: KnowledgeGraph | None = None,
    target_types: list[str] | None = None,
    eps: float = 0.25,
    min_samples: int = 3,
) -> list[NoteCluster]:
    """Discover clusters of unprocessed notes using DBSCAN on ChromaDB embeddings.

    Args:
        collection: ChromaDB collection with indexed chunks.
        knowledge_graph: Optional graph for entity-based cluster merging.
        target_types: Note types to cluster (default: fleeting, literature).
        eps: DBSCAN neighborhood radius (cosine distance).
        min_samples: Minimum notes per cluster.

    Returns:
        List of NoteCluster sorted by score descending.
    """
    types = target_types or ["fleeting", "literature"]

    # Extract embeddings for target note types
    results = collection.get(
        where={"note_type": {"$in": types}},  # type: ignore[dict-item]
        include=["embeddings", "metadatas"],  # type: ignore[list-item]
    )

    if not results["ids"] or not results["embeddings"]:
        logger.info("No embeddings found for types %s", types)
        return []

    embeddings = np.array(results["embeddings"])
    metadatas: list[dict[str, Any]] = [dict(m) for m in (results["metadatas"] or [])]

    if len(embeddings) < min_samples:
        logger.info("Only %d notes found, need at least %d", len(embeddings), min_samples)
        return []

    # Deduplicate to one embedding per note (use first chunk per note)
    note_map: dict[str, int] = {}  # note_path -> index in deduplicated arrays
    dedup_indices: list[int] = []
    for i, meta in enumerate(metadatas):
        path = str(meta.get("note_path", ""))
        if path and path not in note_map:
            note_map[path] = len(dedup_indices)
            dedup_indices.append(i)

    if len(dedup_indices) < min_samples:
        logger.info("Only %d unique notes, need at least %d", len(dedup_indices), min_samples)
        return []

    dedup_embeddings = embeddings[dedup_indices]
    dedup_metas = [metadatas[i] for i in dedup_indices]
    note_paths = [str(m.get("note_path", "")) for m in dedup_metas]

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(dedup_embeddings)

    # Group notes by cluster label (ignore noise label -1)
    cluster_groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label >= 0:
            cluster_groups.setdefault(int(label), []).append(idx)

    if not cluster_groups:
        logger.info("DBSCAN found no clusters (all noise)")
        return []

    # Optionally merge clusters that share graph entities
    if knowledge_graph is not None:
        cluster_groups = _merge_by_entities(cluster_groups, note_paths, knowledge_graph)

    # Build NoteCluster objects
    clusters: list[NoteCluster] = []
    for members in cluster_groups.values():
        paths = [note_paths[i] for i in members]
        titles = [str(dedup_metas[i].get("note_title", note_paths[i])) for i in members]
        top_entity = _find_top_entity(paths, knowledge_graph)
        score = _score_cluster(members, dedup_metas)
        fingerprint = _cluster_fingerprint(paths)
        clusters.append(
            NoteCluster(
                note_paths=paths,
                note_titles=titles,
                top_entity=top_entity,
                score=score,
                fingerprint=fingerprint,
            )
        )

    clusters.sort(key=lambda c: c.score, reverse=True)
    return clusters


def _merge_by_entities(
    cluster_groups: dict[int, list[int]],
    note_paths: list[str],
    graph: KnowledgeGraph,
) -> dict[int, list[int]]:
    """Merge clusters that share at least one entity in the knowledge graph."""
    # Build entity sets per cluster
    cluster_entities: dict[int, set[str]] = {}
    for label, members in cluster_groups.items():
        entities: set[str] = set()
        for idx in members:
            path = note_paths[idx]
            # Check graph nodes that reference this note
            for node_id in graph._graph.nodes:
                node_data = graph._graph.nodes[node_id]
                sources = node_data.get("source_notes", [])
                if path in sources:
                    entities.add(str(node_id))
        cluster_entities[label] = entities

    # Union-find merge
    parent: dict[int, int] = {label: label for label in cluster_groups}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    labels = list(cluster_groups.keys())
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            if cluster_entities[la] & cluster_entities[lb]:
                ra, rb = find(la), find(lb)
                if ra != rb:
                    parent[rb] = ra

    # Rebuild merged groups
    merged: dict[int, list[int]] = {}
    for label, members in cluster_groups.items():
        root = find(label)
        merged.setdefault(root, []).extend(members)

    return merged


def _find_top_entity(
    paths: list[str],
    graph: KnowledgeGraph | None,
) -> str:
    """Find the most-referenced entity across cluster notes."""
    if graph is None:
        return paths[0].rsplit("/", maxsplit=1)[-1].replace(".md", "")

    entity_counts: dict[str, int] = {}
    for node_id in graph._graph.nodes:
        node_data = graph._graph.nodes[node_id]
        sources = node_data.get("source_notes", [])
        overlap = sum(1 for p in paths if p in sources)
        if overlap > 0:
            entity_counts[str(node_data.get("label", node_id))] = overlap

    if entity_counts:
        return max(entity_counts, key=lambda k: entity_counts[k])
    return paths[0].rsplit("/", maxsplit=1)[-1].replace(".md", "")


def _score_cluster(members: list[int], metas: list[dict[str, Any]]) -> float:
    """Score a cluster by note count and recency."""
    count_score = len(members) / 10.0  # normalize
    return min(count_score, 1.0)


def _cluster_fingerprint(paths: list[str]) -> str:
    """Deterministic fingerprint for a cluster (hash of sorted paths)."""
    raw = ":".join(sorted(paths))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
