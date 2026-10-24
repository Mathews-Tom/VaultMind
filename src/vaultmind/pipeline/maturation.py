"""Zettelkasten maturation pipeline orchestrator.

Coordinates cluster discovery, digest delivery, and synthesis.
Maintains persistent state for dismissed/synthesized clusters.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import frontmatter

from vaultmind.errors import VaultMindError
from vaultmind.pipeline.clustering import NoteCluster, discover_clusters
from vaultmind.pipeline.distill import validate_derivative_parent
from vaultmind.pipeline.synthesis import (
    WIKILINK_PATTERN,
    SynthesisGeneration,
    generate_synthesis,
    render_synthesis_prompt,
)
from vaultmind.vault.security import validate_ordinary_output_path, validate_ordinary_write_path
from vaultmind.wiki.workspace import (
    MAX_INPUT_BYTES,
    WikiError,
    _canonical,
    _metadata,
    _regular_file,
)

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

    from vaultmind.config import MaturationConfig
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.knowledge.intake import SourceIntake
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
        intake: SourceIntake,
        state_path: Path | None = None,
    ) -> None:
        self._config = config
        self._collection = collection
        self._graph = knowledge_graph
        self._llm = llm
        self._intake = intake
        self._vault_root = intake.workspace.root
        default_state = Path.home() / ".vaultmind" / "data" / "maturation_state.json"
        self._state_path = validate_ordinary_output_path(
            state_path or default_state, self._vault_root
        )
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

    def synthesize(self, cluster: NoteCluster, *, invocation_id: str | None = None) -> str:
        """Preserve a selected cluster and return only its real unapproved projection."""
        output_dir = validate_ordinary_write_path(self._config.inbox_folder, self._vault_root)
        validate_ordinary_write_path(str(output_dir / "synthesis.md"), self._vault_root)
        try:
            relative = self._synthesize_retained(cluster, invocation_id)
        except (ValueError, OSError) as exc:
            raise VaultMindError(f"Synthesis failed: {exc}") from exc
        self._state.synthesized.append(cluster.fingerprint)
        self._state.last_run = datetime.now(UTC).isoformat()
        save_state(self._state, self._state_path)
        return relative

    def _synthesize_retained(self, cluster: NoteCluster, invocation_id: str | None) -> str:
        intake = self._intake
        root = self._vault_root.resolve()
        if not cluster.note_paths or len(set(cluster.note_paths)) != len(cluster.note_paths):
            raise WikiError("Select a nonempty cluster without duplicate source paths")
        selected: list[tuple[str, bytes, list[str]]] = []
        for relative in cluster.note_paths:
            checked = validate_ordinary_write_path(relative, root)
            _regular_file(root / relative, "Selected source note")
            fd = os.open(checked, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise WikiError("Selected source note must be a regular file")
                raw = stream.read(MAX_INPUT_BYTES + 1)
            if len(raw) > MAX_INPUT_BYTES:
                raise WikiError("Selected source note exceeds its input limit")
            text = raw.decode("utf-8")
            metadata, _ = _metadata(text)
            parents = metadata.get("parent_source_ids", [])
            if not isinstance(parents, list) or any(not isinstance(item, str) for item in parents):
                raise WikiError("Selected source note has invalid retained parent identities")
            parents = list(parents)
            source_id = metadata.get("source_id")
            if source_id is not None:
                if not isinstance(source_id, str):
                    raise WikiError("Selected source note has invalid retained source identity")
                parents.append(source_id)
            selected.append((relative, raw, sorted(set(parents))))
        event = invocation_id or f"cluster:{cluster.fingerprint}"
        originals = [
            intake.preserve_text(
                raw,
                original_name=Path(relative).name,
                origin=f"vault:{relative}",
                adapter="maturation",
                role="source-note",
                event_id=f"{event}:{relative}",
                parents=parents,
            )
            for relative, raw, parents in selected
        ]
        source_texts: list[tuple[str, str]] = []
        for (relative, _, _), original in zip(selected, originals, strict=True):
            retained = intake.read(original.source_id)
            assert retained.text is not None
            source_texts.append((relative, retained.text))
        prompt = render_synthesis_prompt(
            source_texts, cluster.top_entity, datetime.now(UTC).strftime("%Y-%m-%d")
        )
        request = intake.preserve_text(
            _canonical(
                dict(
                    messages=[dict(role="user", content=prompt)],
                    model=self._config.synthesis_model,
                    max_tokens=self._config.synthesis_max_tokens,
                    cluster=dict(
                        fingerprint=cluster.fingerprint,
                        note_paths=cluster.note_paths,
                        note_titles=cluster.note_titles,
                        top_entity=cluster.top_entity,
                    ),
                )
            ),
            original_name="synthesis-request.json",
            origin=f"maturation:{cluster.fingerprint}",
            adapter="maturation",
            role="model-request",
            event_id=event,
            parents=[original.source_id for original in originals],
            original_retained=False,
        )
        retained_request = intake.read(request.source_id)
        validate_derivative_parent(retained_request)
        values = json.loads(retained_request.text or "")
        generated: SynthesisGeneration | None = None

        def generate() -> None:
            nonlocal generated
            generated = generate_synthesis(
                values["messages"][0]["content"],
                self._llm,
                values["model"],
                values["max_tokens"],
            )
            if not generated.success:
                raise WikiError(generated.error)
            self._validate_links(generated.content)

        disposition = request.run_stage("provider", generate)
        if disposition.status != "succeeded" or generated is None:
            raise WikiError(
                generated.error
                if generated is not None and not generated.success
                else disposition.error or "Generation was not completed; originals preserved"
            )
        post = frontmatter.loads(generated.content)
        post.metadata["parent_source_ids"] = [request.source_id]
        content = str(frontmatter.dumps(post))
        derivative = request.run_stage(
            "derivative",
            lambda: intake.preserve_text(
                content,
                original_name="synthesis.md",
                origin=f"maturation:{cluster.fingerprint}",
                adapter="maturation",
                role="synthesis",
                event_id=event,
                revision=request.source_id,
                parents=[request.source_id],
                original_retained=False,
            ),
        )
        if derivative.status != "succeeded":
            raise WikiError(derivative.error or "Derivative was not retained")
        output_dir = validate_ordinary_write_path(self._config.inbox_folder, root)
        folder = output_dir.relative_to(root).as_posix()
        projection = intake.project_text(derivative.value, folder)
        if projection.status != "succeeded" or not isinstance(projection.value, str):
            raise WikiError(projection.error or "Projection was not completed; sources preserved")
        return projection.value

    def _validate_links(self, content: str) -> None:
        links = set(WIKILINK_PATTERN.findall(content))
        if not links:
            return
        available: set[str] = set()
        for path in self._vault_root.rglob("*.md"):
            try:
                checked = validate_ordinary_write_path(str(path), self._vault_root)
            except ValueError:
                continue
            if checked.is_file():
                available.add(path.stem)
                available.add(str(path.relative_to(self._vault_root).with_suffix("")))
        unresolved = links - available
        if unresolved:
            raise WikiError(f"Unresolved wikilinks: {', '.join(sorted(unresolved))}")

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
