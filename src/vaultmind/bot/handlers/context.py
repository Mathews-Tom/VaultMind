"""Shared context dataclass for all handler modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vaultmind.bot.thinking import ThinkingPartner
    from vaultmind.config import Settings
    from vaultmind.graph.knowledge_graph import KnowledgeGraph
    from vaultmind.indexer.store import VaultStore
    from vaultmind.llm.client import LLMClient
    from vaultmind.vault.parser import VaultParser


@dataclass
class HandlerContext:
    settings: Settings
    store: VaultStore
    graph: KnowledgeGraph
    parser: VaultParser
    thinking: ThinkingPartner
    llm_client: LLMClient
    vault_root: Path
