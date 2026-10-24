"""Tests for SessionStore — SQLite-backed thinking session persistence."""

from __future__ import annotations

import time

import pytest

from vaultmind.bot.session_store import SessionStore


@pytest.fixture
def store(tmp_path: object) -> SessionStore:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    return SessionStore(tmp_path / "sessions.db")


class TestSaveAndLoad:
    def test_roundtrip(self, store: SessionStore) -> None:
        history = [{"user": "hello", "assistant": "hi there"}]
        store.save(1, history)
        loaded = store.load(1)
        assert loaded == history

    def test_load_nonexistent_returns_none(self, store: SessionStore) -> None:
        assert store.load(999) is None

    def test_upsert_updates_not_duplicates(self, store: SessionStore) -> None:
        store.save(1, [{"user": "a", "assistant": "b"}])
        store.save(1, [{"user": "a", "assistant": "b"}, {"user": "c", "assistant": "d"}])
        loaded = store.load(1)
        assert loaded is not None
        assert len(loaded) == 2

    def test_multi_turn_history(self, store: SessionStore) -> None:
        history = [
            {"user": "first", "assistant": "reply1"},
            {"user": "second", "assistant": "reply2"},
            {"user": "third", "assistant": "reply3"},
        ]
        store.save(42, history)
        assert store.load(42) == history


class TestDelete:
    def test_delete_existing(self, store: SessionStore) -> None:
        store.save(1, [{"user": "x", "assistant": "y"}])
        store.delete(1)
        assert store.load(1) is None

    def test_delete_nonexistent_no_error(self, store: SessionStore) -> None:
        store.delete(999)  # should not raise


class TestArchiveExpired:
    def test_preserves_expired_sessions(self, store: SessionStore) -> None:
        store.save(1, [{"user": "a", "assistant": "b"}])
        # Manually backdate the last_active
        store._conn.execute(
            "UPDATE thinking_sessions SET last_active = ? WHERE user_id = 1",
            (time.time() - 7200,),
        )
        store._conn.commit()
        claimed = store.claim_expired(3600)
        assert len(claimed) == 1
        assert store.load(1) is None
        record = claimed[0]
        assert record["history"] == [{"user": "a", "assistant": "b"}]
        assert store.claim_expired(3600) == []

    def test_keeps_fresh_sessions(self, store: SessionStore) -> None:
        store.save(1, [{"user": "a", "assistant": "b"}])
        assert store.claim_expired(3600) == []
        assert store.load(1) is not None


class TestHasSession:
    def test_true_when_exists(self, store: SessionStore) -> None:
        store.save(1, [{"user": "a", "assistant": "b"}])
        assert store.has_session(1) is True

    def test_false_when_missing(self, store: SessionStore) -> None:
        assert store.has_session(999) is False


class TestCompositeKey:
    def test_different_session_names_isolated(self, store: SessionStore) -> None:
        store.save(1, [{"user": "a", "assistant": "b"}], session_name="work")
        store.save(1, [{"user": "c", "assistant": "d"}], session_name="personal")
        assert store.load(1, session_name="work") == [{"user": "a", "assistant": "b"}]
        assert store.load(1, session_name="personal") == [{"user": "c", "assistant": "d"}]
        assert store.load(1, session_name="default") is None

    def test_delete_one_session_name_keeps_other(self, store: SessionStore) -> None:
        store.save(1, [{"user": "a", "assistant": "b"}], session_name="alpha")
        store.save(1, [{"user": "c", "assistant": "d"}], session_name="beta")
        store.delete(1, session_name="alpha")
        assert store.load(1, session_name="alpha") is None
        assert store.load(1, session_name="beta") is not None

    def test_has_session_respects_name(self, store: SessionStore) -> None:
        store.save(1, [{"user": "a", "assistant": "b"}], session_name="named")
        assert store.has_session(1, session_name="named") is True
        assert store.has_session(1, session_name="default") is False


def test_interrupted_claim_survives_restart_without_retry(store: SessionStore) -> None:
    import json
    from pathlib import Path

    history = [{"user": "cache retention", "assistant": "Original assistant answer"}]
    store.save(1, history)
    store.save_summary(1, "default", 0, 0, 0, "Original summary", ["cache"], ["why?"])
    original_summary = store._conn.execute("SELECT * FROM thinking_session_summaries").fetchone()
    store._conn.execute("UPDATE thinking_sessions SET last_active = ?", (time.time() - 7200,))
    store._conn.commit()
    claimed = store.claim_expired(3600)
    assert len(claimed) == 1
    path = Path(store._conn.execute("PRAGMA database_list").fetchone()[2])
    store.close()
    restarted = SessionStore(path)
    assert restarted.claim_expired(3600) == []
    row = restarted._conn.execute(
        "SELECT history, summaries, state FROM thinking_session_archives"
    ).fetchone()
    assert json.loads(row[0]) == history
    assert row[2] == "claimed"
    archived = json.loads(row[1])[0]
    assert tuple(archived.values()) == original_summary


def test_explicit_clear_removes_conversation_generations_not_other_sessions(
    store: SessionStore,
) -> None:
    for uid, name in [(1, "default"), (1, "thinking:old"), (1, "work"), (2, "default")]:
        store.save(uid, [{"user": "cache", "assistant": "answer"}], name)
    store._conn.execute("UPDATE thinking_sessions SET last_active = ?", (time.time() - 7200,))
    store._conn.commit()
    store.claim_expired(3600)
    store.save(1, [{"user": "new cache", "assistant": "new answer"}], "thinking:new")
    store.delete_conversations(1)
    assert store.load(1, "thinking:new") is None
    assert store._conn.execute(
        "SELECT user_id, session_name FROM thinking_session_archives ORDER BY user_id"
    ).fetchall() == [(1, "work"), (2, "default")]
