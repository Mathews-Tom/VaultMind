"""Approved-current discovery and shared answer policy on isolated real workspaces."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from vaultmind.config import Settings, load_settings
from vaultmind.knowledge.query import query_knowledge
from vaultmind.llm.client import LLMResponse
from vaultmind.wiki.workspace import WikiError, WikiWorkspace, proposal_digest

if TYPE_CHECKING:
    from pathlib import Path

CACHE = "wiki/cache.md"
LATE = "The cache retention policy is 30 days."


@dataclass
class Corpus:
    settings: Settings
    vault: Path

    def source(self, name: str, text: str) -> None:
        original = self.vault.parent / f"{name}-input.md"
        original.write_text(text)
        with WikiWorkspace(self.settings) as workspace:
            workspace.add_source(original, name, uri=f"urn:fixture:{name}")

    def publish(self, page: str, body: str, source: str = "cache-source") -> None:
        draft = self.vault.parent / "draft.md"
        draft.write_text(body)
        with WikiWorkspace(self.settings) as workspace:
            proposal = workspace.propose(page, [source], draft_file=draft)
            workspace.review(proposal.proposal_id)
            workspace.apply(proposal.proposal_id, proposal_digest(proposal.payload))


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def forbidden_provider() -> Any:
    raise AssertionError("A provider must not be constructed for this query")


class ScriptedProvider:
    provider_name = "scripted-approved-query"

    def __init__(self, response: dict[str, Any], mutate: Any = None) -> None:
        self.response = response
        self.mutate = mutate
        self.calls: list[Any] = []

    def complete(
        self, messages: Any, model: str, max_tokens: int = 4096, system: str | None = None
    ) -> LLMResponse:
        self.calls.append(messages)
        if self.mutate is not None:
            self.mutate()
        return LLMResponse(text=json.dumps(self.response), model=model, usage={})

    def complete_multimodal(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise AssertionError("Query must not use multimodal generation")


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Corpus:
    for key in tuple(os.environ):
        if key.upper().startswith("VAULTMIND_"):
            monkeypatch.delenv(key)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.chdir(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    config = tmp_path / "query.toml"
    config.write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        'excluded_folders = ["wiki", "_wiki"]\n'
        '[llm]\nprovider = "ollama"\nthinking_model = "scripted-query"\n'
    )
    result = Corpus(load_settings(config), vault)
    result.source("cache-source", "Original cache retention policy: 30 days or 7 days.\n")
    return result


def cache_body() -> str:
    return (
        "# Cache\n\n## Current understanding\n\n"
        + "Background material. " * 50
        + LATE
        + " [[_wiki/sources/cache-source]]\n"
    )


def test_automatic_late_evidence_matches_explicit_snapshot(corpus: Corpus) -> None:
    corpus.publish(CACHE, cache_body())
    corpus.source("garden-source", "Water seedlings twice a week.\n")
    corpus.publish(
        "wiki/gardening.md",
        "# Gardening\n\n## Current understanding\n\nWater seedlings. "
        "[[_wiki/sources/garden-source]]\n",
        "garden-source",
    )
    before = snapshot(corpus.vault)
    question = "What retention policy is agreed?"
    automatic = query_knowledge(
        corpus.settings, question, evidence_only=True, llm_factory=forbidden_provider
    )
    explicit = query_knowledge(
        corpus.settings, question, [CACHE], evidence_only=True, llm_factory=forbidden_provider
    )
    for key in (
        "status",
        "answer",
        "citations",
        "evidence",
        "unresolved_conflicts",
        "selected_pages",
        "sources",
    ):
        assert automatic[key] == explicit[key]
    assert automatic["selection"]["mode"] == "automatic"
    assert explicit["selection"]["mode"] == "explicit"
    assert automatic["selection"]["selected_pages"] == [CACHE]
    assert automatic["selection"]["limited"] is False
    late = next(row for row in automatic["evidence"] if LATE in row["text"])
    assert late["text"].index(LATE) > 500
    assert [row["id"] for row in automatic["sources"]] == ["cache-source"]
    assert snapshot(corpus.vault) == before


def test_lexical_title_preference_ties_and_eight_page_limit(corpus: Corpus) -> None:
    # A lexically earlier body-only hit must lose to heading/title coverage.
    corpus.publish(
        "wiki/a-body.md",
        "# Archive\n\n## Current understanding\n\nRetention policy. "
        "[[_wiki/sources/cache-source]]\n",
    )
    for index in reversed(range(9)):
        corpus.publish(
            f"wiki/title-{index:02d}.md",
            "# Retention policy\n\n## Current understanding\n\nAgreed. "
            "[[_wiki/sources/cache-source]]\n",
        )
    result = query_knowledge(
        corpus.settings, "retention policy", evidence_only=True, llm_factory=forbidden_provider
    )
    expected = [f"wiki/title-{index:02d}.md" for index in range(8)]
    assert result["selection"]["selected_pages"] == expected
    assert result["selection"]["limited"] is True
    assert [row["path"] for row in result["selected_pages"]] == expected
    assert {row["page"] for row in result["evidence"]} == set(expected)
    assert result["selection"]["limitation"]


@pytest.mark.parametrize("has_page", [False, True], ids=["empty", "no-lexical-support"])
@pytest.mark.parametrize("evidence_only", [False, True])
def test_missing_coverage_never_constructs_provider_or_reads_raw(
    corpus: Corpus, has_page: bool, evidence_only: bool
) -> None:
    if has_page:
        corpus.publish(CACHE, cache_body())
    # Even an exact raw-note/source match does not establish approved coverage.
    (corpus.vault / "raw.md").write_text("Horticultural lexicon: secret raw answer.\n")
    corpus.source("unreviewed", "Horticultural lexicon: secret original answer.\n")
    before = snapshot(corpus.vault)
    result = query_knowledge(
        corpus.settings,
        "horticultural lexicon",
        evidence_only=evidence_only,
        llm_factory=forbidden_provider,
    )
    assert result["status"] == "unsupported"
    assert result["reason"] == ("no_lexical_support" if has_page else "no_eligible_pages")
    assert (
        result["evidence"]
        == result["citations"]
        == result["selected_pages"]
        == result["sources"]
        == []
    )
    assert result["selection"]["selected_pages"] == []
    assert result["selection"]["limited"] is False
    assert "synonym" in result["selection"]["limitation"].lower()
    assert "secret" not in json.dumps(result)
    assert snapshot(corpus.vault) == before


def test_invalid_irrelevant_permitted_page_is_not_silently_dropped(corpus: Corpus) -> None:
    corpus.publish(CACHE, cache_body())
    hidden = corpus.vault / "wiki/gardening.md"
    hidden.write_text("---\napproved: true\nauthority: 5\n---\n# Gardening\nFORGED_BODY\n")
    before = snapshot(corpus.vault)
    with pytest.raises(WikiError):
        query_knowledge(corpus.settings, "cache retention", llm_factory=forbidden_provider)
    assert snapshot(corpus.vault) == before


def test_scope_filters_invalid_pages_before_validation_or_disclosure(corpus: Corpus) -> None:
    corpus.publish(CACHE, cache_body())
    denied = corpus.vault / "wiki/private-secret.md"
    denied.write_text("# Hidden retention\nPRIVATE_BODY_PRIVATE_URI\n")
    before = snapshot(corpus.vault)

    def allowed(path: str) -> bool:
        return path == "_wiki" or path.startswith("_wiki/") or path == CACHE

    result = query_knowledge(
        corpus.settings,
        "retention",
        evidence_only=True,
        llm_factory=forbidden_provider,
        allows_path=allowed,
    )
    assert result["selection"]["selected_pages"] == [CACHE]
    assert "private-secret" not in json.dumps(result)
    assert "PRIVATE_BODY_PRIVATE_URI" not in json.dumps(result)
    with pytest.raises(WikiError) as error:
        query_knowledge(
            corpus.settings,
            "retention",
            [CACHE, "wiki/private-secret.md"],
            llm_factory=forbidden_provider,
            allows_path=allowed,
        )
    assert "PRIVATE_BODY_PRIVATE_URI" not in str(error.value)
    assert snapshot(corpus.vault) == before


def test_denied_provenance_is_invisible_even_for_invalid_current_page(corpus: Corpus) -> None:
    corpus.publish(CACHE, cache_body())
    (corpus.vault / CACHE).write_text("PRIVATE_CURRENT_BODY\n")
    before = snapshot(corpus.vault)

    def allowed(path: str) -> bool:
        return path.startswith("wiki/")

    result = query_knowledge(
        corpus.settings, "cache", llm_factory=forbidden_provider, allows_path=allowed
    )
    assert result["reason"] == "no_eligible_pages"
    assert result["selected_pages"] == result["sources"] == result["evidence"] == []
    assert "PRIVATE_CURRENT_BODY" not in json.dumps(result)
    with pytest.raises(WikiError) as error:
        query_knowledge(
            corpus.settings,
            "cache",
            [CACHE],
            llm_factory=forbidden_provider,
            allows_path=allowed,
        )
    assert "cache-source" not in str(error.value)
    assert "PRIVATE_CURRENT_BODY" not in str(error.value)
    assert snapshot(corpus.vault) == before


@pytest.mark.parametrize("pages", [None, [CACHE]])
@pytest.mark.parametrize("invalid_question", ["", " ", "x" * 4097])
def test_question_bounds_precede_provider(
    corpus: Corpus, pages: list[str] | None, invalid_question: str
) -> None:
    corpus.publish(CACHE, cache_body())
    with pytest.raises(WikiError):
        query_knowledge(corpus.settings, invalid_question, pages, llm_factory=forbidden_provider)


@pytest.mark.parametrize("pages", [[], [CACHE, CACHE], [f"wiki/page-{i}.md" for i in range(9)]])
def test_explicit_selection_errors_do_not_construct_provider(
    corpus: Corpus, pages: list[str]
) -> None:
    corpus.publish(CACHE, cache_body())
    with pytest.raises(WikiError):
        query_knowledge(corpus.settings, "cache", pages, llm_factory=forbidden_provider)


@pytest.mark.parametrize("pages", [None, [CACHE]])
def test_whole_selected_evidence_overflow_refuses_instead_of_truncating(
    corpus: Corpus, pages: list[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vaultmind.knowledge.query as module

    corpus.publish(CACHE, cache_body())
    monkeypatch.setattr(module, "MAX_QUERY_CONTEXT_BYTES", 512)
    before = snapshot(corpus.vault)
    with pytest.raises(WikiError):
        query_knowledge(corpus.settings, "cache retention", pages, llm_factory=forbidden_provider)
    assert snapshot(corpus.vault) == before


@pytest.mark.parametrize(
    "fence",
    ["```markdown\n## Resolution\nExample only.\n```", "~~~md\n## Resolution\nExample only.\n~~~"],
)
def test_all_conflict_blocks_and_fenced_resolution_have_identical_policy(
    corpus: Corpus, fence: str
) -> None:
    body = (
        "# Cache\n\n## Current understanding\n\nCache retention remains disputed.\n\n"
        "## Conflicting information\n\n30 days from policy. [[_wiki/sources/cache-source]]\n\n"
        "## Conflicting information\n\n7 days from operations. [[_wiki/sources/cache-source]]\n\n"
        + fence
        + "\n"
    )
    corpus.publish(CACHE, body)
    automatic = query_knowledge(corpus.settings, "cache retention", evidence_only=True)
    explicit = query_knowledge(corpus.settings, "cache retention", [CACHE], evidence_only=True)
    assert automatic["evidence"] == explicit["evidence"]
    assert automatic["unresolved_conflicts"] == explicit["unresolved_conflicts"] == [CACHE]
    conflicts = [
        row["id"] for row in automatic["evidence"] if row["heading"] == "Conflicting information"
    ]
    assert len(conflicts) == 2
    for pages in (None, [CACHE]):
        for status, citations in (("supported", conflicts), ("conflicting", conflicts[:1])):
            client = ScriptedProvider(
                dict(status=status, answer="Unresolved.", citations=citations)
            )
            with pytest.raises(WikiError):
                query_knowledge(
                    corpus.settings,
                    "cache retention",
                    pages,
                    llm_factory=lambda client=client: client,
                )
            assert len(client.calls) == 1
        client = ScriptedProvider(
            dict(status="conflicting", answer="30 and 7 days conflict.", citations=conflicts)
        )
        result = query_knowledge(
            corpus.settings, "cache retention", pages, llm_factory=lambda client=client: client
        )
        assert result["status"] == "conflicting"
        assert result["citations"] == conflicts
        assert len(client.calls) == 1


@pytest.mark.parametrize("kind", ["current", "source", "metadata", "operation", "history"])
@pytest.mark.parametrize("pages", [None, [CACHE]])
def test_changed_selected_identity_discards_generated_answer(
    corpus: Corpus, kind: str, pages: list[str] | None
) -> None:
    corpus.publish(CACHE, cache_body())
    corpus.publish(CACHE, cache_body().replace("30 days", "31 days"))
    if kind == "current":
        path = corpus.vault / CACHE
    elif kind == "source":
        path = corpus.vault / "_wiki/sources/cache-source.md"
    elif kind == "metadata":
        path = corpus.vault / "_wiki/sources/cache-source.json"
    elif kind == "operation":
        path = next((corpus.vault / "_wiki/operations").glob("*.json"))
    else:
        path = next((corpus.vault / "_wiki/history/cache").glob("*.md"))
    original = path.read_bytes()
    before = snapshot(corpus.vault)
    client = ScriptedProvider(
        dict(status="unsupported", answer="No decision.", citations=[]),
        mutate=lambda: path.write_bytes(original + b"\nMUTATION_DURING_COMPLETION"),
    )
    with pytest.raises(WikiError):
        query_knowledge(corpus.settings, "cache retention", pages, llm_factory=lambda: client)
    assert len(client.calls) == 1
    # Restore only the deliberate synthetic mutation; querying made no other writes.
    path.write_bytes(original)
    assert snapshot(corpus.vault) == before
    restored = query_knowledge(corpus.settings, "cache retention", pages, evidence_only=True)
    assert restored["evidence"]


def test_historical_match_is_not_automatic_current_evidence(corpus: Corpus) -> None:
    corpus.publish(CACHE, cache_body())
    corpus.publish(
        CACHE,
        "# Archive\n\n## Current understanding\n\nThe project is paused. "
        "[[_wiki/sources/cache-source]]\n",
    )
    before = snapshot(corpus.vault)
    result = query_knowledge(corpus.settings, "retention", llm_factory=forbidden_provider)
    assert result["reason"] == "no_lexical_support"
    assert result["evidence"] == []
    assert "30 days" not in json.dumps(result)
    assert snapshot(corpus.vault) == before


def test_generated_automatic_and_explicit_answers_share_citation_identity(corpus: Corpus) -> None:
    corpus.publish(CACHE, cache_body())
    evidence = query_knowledge(corpus.settings, "cache retention", evidence_only=True)
    citation = next(row["id"] for row in evidence["evidence"] if LATE in row["text"])
    results = []
    for pages in (None, [CACHE]):
        client = ScriptedProvider(
            dict(status="supported", answer="The policy is 30 days.", citations=[citation])
        )
        result = query_knowledge(
            corpus.settings, "cache retention", pages, llm_factory=lambda client=client: client
        )
        assert len(client.calls) == 1
        results.append(result)
    for key in (
        "status",
        "answer",
        "citations",
        "evidence",
        "unresolved_conflicts",
        "selected_pages",
        "sources",
    ):
        assert results[0][key] == results[1][key]
    assert results[0]["status"] == "supported"
    assert results[0]["citations"] == [citation]


@pytest.mark.parametrize(
    "kind",
    ["current", "source", "metadata", "operation", "history", "stale", "missing-operation"],
)
def test_invalid_automatic_authority_refuses_before_provider_and_restores_exactly(
    corpus: Corpus, kind: str
) -> None:
    corpus.publish(CACHE, cache_body())
    old = (corpus.vault / CACHE).read_bytes()
    earlier_operations = set((corpus.vault / "_wiki/operations").glob("*.json"))
    corpus.publish(CACHE, cache_body().replace("30 days", "31 days"))
    latest_operation = next(
        path
        for path in (corpus.vault / "_wiki/operations").glob("*.json")
        if path not in earlier_operations
    )
    if kind in {"current", "stale"}:
        path = corpus.vault / CACHE
    elif kind == "source":
        path = corpus.vault / "_wiki/sources/cache-source.md"
    elif kind == "metadata":
        path = corpus.vault / "_wiki/sources/cache-source.json"
    elif kind in {"operation", "missing-operation"}:
        path = latest_operation
    else:
        path = next((corpus.vault / "_wiki/history/cache").glob("*.md"))
    original = path.read_bytes()
    valid = snapshot(corpus.vault)
    if kind == "missing-operation":
        path.unlink()
    elif kind == "stale":
        path.write_bytes(old)
    else:
        path.write_bytes(original + b"\nUNAPPROVED_MUTATION")
    invalid = snapshot(corpus.vault)
    with pytest.raises(WikiError):
        query_knowledge(corpus.settings, "cache retention", llm_factory=forbidden_provider)
    assert snapshot(corpus.vault) == invalid
    path.write_bytes(original)
    assert snapshot(corpus.vault) == valid
    result = query_knowledge(
        corpus.settings, "cache retention", evidence_only=True, llm_factory=forbidden_provider
    )
    assert result["selection"]["selected_pages"] == [CACHE]
    assert any("31 days" in row["text"] for row in result["evidence"])


@pytest.mark.parametrize("term", ["current", "understanding"])
def test_structural_heading_alone_cannot_select_unrelated_evidence(
    corpus: Corpus, term: str
) -> None:
    corpus.publish(CACHE, cache_body())
    before = snapshot(corpus.vault)
    result = query_knowledge(corpus.settings, f"{term} xylophone", llm_factory=forbidden_provider)
    assert result["reason"] == "no_lexical_support"
    assert result["selected_pages"] == []
    assert snapshot(corpus.vault) == before


@pytest.mark.parametrize("position", ["title", "body"])
def test_structural_words_in_user_content_remain_discoverable(
    corpus: Corpus, position: str
) -> None:
    title = "Current account" if position == "title" else "Account"
    body = "The current account is active." if position == "body" else "The account is active."
    corpus.publish(
        CACHE,
        f"# {title}\n\n## Current understanding\n\n{body} [[_wiki/sources/cache-source]]\n",
    )
    result = query_knowledge(corpus.settings, "current", evidence_only=True)
    assert result["selection"]["selected_pages"] == [CACHE]
    assert any(body in row["text"] for row in result["evidence"])
