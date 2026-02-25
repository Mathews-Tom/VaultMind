"""Indexer — embedding pipeline, vector store, duplicate detection, and note suggestions."""

from vaultmind.indexer.duplicate_detector import DuplicateDetector, DuplicateMatch, MatchType
from vaultmind.indexer.embedder import Embedder
from vaultmind.indexer.embedding_cache import EmbeddingCache
from vaultmind.indexer.note_suggester import NoteSuggester, NoteSuggestion
from vaultmind.indexer.store import VaultStore

__all__ = [
    "DuplicateDetector",
    "DuplicateMatch",
    "Embedder",
    "EmbeddingCache",
    "MatchType",
    "NoteSuggester",
    "NoteSuggestion",
    "VaultStore",
]
