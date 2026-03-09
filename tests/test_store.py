"""Tests for the engram memory store."""

import os
import tempfile
import pytest
from engram.store import MemoryStore


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = MemoryStore(path)
    yield s
    s.close()
    os.unlink(path)


def test_remember_and_recall(store):
    m = store.remember("TrueNAS is at 192.168.0.192", category="network", key="truenas_ip")
    assert m.id
    assert m.content == "TrueNAS is at 192.168.0.192"
    assert m.category == "network"

    results = store.recall("TrueNAS IP address")
    assert len(results) >= 1
    assert any("192.168.0.192" in r.content for r in results)


def test_fact_deduplication(store):
    store.remember("NAS is at .192", category="network", key="nas_ip")
    store.remember("NAS is at .193", category="network", key="nas_ip")

    results = store.recall("NAS", category="network")
    assert len(results) == 1
    assert ".193" in results[0].content  # updated value


def test_episode_no_dedup(store):
    store.remember("Debugged timing chain issue", memory_type="episode", category="debug")
    store.remember("Fixed NAS mount timeout", memory_type="episode", category="debug")

    results = store.recall("debug", category="debug")
    assert len(results) == 2


def test_forget_by_id(store):
    m = store.remember("temporary note", category="temp", key="note1")
    assert store.forget(m.id)
    results = store.recall("temporary note")
    assert len(results) == 0


def test_forget_by_key(store):
    store.remember("old fact", category="test", key="deleteme")
    assert store.forget_by_key("test", "deleteme")
    results = store.recall("old fact")
    assert len(results) == 0


def test_update(store):
    m = store.remember("initial value", category="test", key="updatable")
    updated = store.update(m.id, content="new value", tags=["updated"])
    assert updated.content == "new value"
    assert "updated" in updated.tags


def test_relationships(store):
    r = store.relate("Immich", "TrueNAS", "runs_on", metadata={"port": 30041})
    assert r.entity_from == "Immich"
    assert r.relation_type == "runs_on"

    entity = store.get_entity("Immich")
    assert len(entity["relationships"]) >= 1
    assert entity["relationships"][0]["relation_type"] == "runs_on"


def test_relationship_upsert(store):
    store.relate("A", "B", "connects_to", metadata={"port": 80})
    store.relate("A", "B", "connects_to", metadata={"port": 443})

    entity = store.get_entity("A")
    rels = [r for r in entity["relationships"] if r["relation_type"] == "connects_to"]
    assert len(rels) == 1
    assert rels[0]["metadata"]["port"] == 443  # updated


def test_categories(store):
    store.remember("fact 1", category="network", key="k1")
    store.remember("fact 2", category="network", key="k2")
    store.remember("pref 1", memory_type="preference", category="user")

    cats = store.list_categories()
    assert len(cats) >= 2


def test_stats(store):
    store.remember("test", category="test", key="t1")
    s = store.stats()
    assert s["total_memories"] == 1
    assert s["by_type"]["fact"] == 1


def test_stale_detection(store):
    m = store.remember("old memory", category="test", key="old")
    # Manually age it
    store._conn.execute(
        "UPDATE memories SET accessed_at = datetime('now', '-60 days') WHERE id = ?",
        (m.id,),
    )
    store._conn.commit()

    stale = store.get_stale(days=30)
    assert len(stale) == 1
    assert stale[0].id == m.id


def test_tags_filter(store):
    store.remember("tagged memory", category="test", key="tagged", tags=["important", "network"])
    store.remember("untagged memory", category="test", key="untagged")

    results = store.recall("memory", tags=["important"])
    assert len(results) == 1
    assert "tagged" in results[0].content


def test_confidence_filter(store):
    store.remember("sure thing", category="test", key="sure", confidence=1.0)
    store.remember("maybe", category="test", key="maybe", confidence=0.3)

    results = store.recall("thing maybe", min_confidence=0.5)
    assert all(r.confidence >= 0.5 for r in results)


def test_get_context(store):
    store.remember("user prefers dark mode", memory_type="preference", category="user")
    store.remember("NAS IP is .192", category="network", key="nas_ip")

    ctx = store.get_context()
    assert len(ctx["preferences"]) >= 1
    assert len(ctx["recent"]) >= 1


def test_access_count_increments(store):
    store.remember("accessed memory", category="test", key="accessed")
    store.recall("accessed memory")
    store.recall("accessed memory")

    m = store.recall("accessed memory")[0]
    assert m.access_count >= 2


# ── Provenance / History Tests ──────────────────────────────────


def test_remember_records_model_and_context(store):
    m = store.remember(
        "NAS IP is 192.168.0.192",
        category="network",
        key="nas_ip",
        model="claude-opus-4-6",
        context="User stated this directly",
    )
    assert m.model == "claude-opus-4-6"
    assert m.context == "User stated this directly"


def test_creation_recorded_in_history(store):
    m = store.remember(
        "test fact",
        category="test",
        key="hist1",
        model="claude-opus-4-6",
        context="Testing history",
    )
    history = store.get_history(m.id)
    assert len(history) == 1
    assert history[0].action == "created"
    assert history[0].model == "claude-opus-4-6"
    assert history[0].new_content == "test fact"
    assert history[0].context == "Testing history"


def test_upsert_records_old_and_new_in_history(store):
    """When two models write the same fact, history shows both perspectives."""
    store.remember(
        "The best approach is X",
        category="decisions",
        key="approach",
        model="claude-opus-4-6",
        context="Based on analyzing logs, X handles the edge case better",
    )
    store.remember(
        "The best approach is Y",
        category="decisions",
        key="approach",
        model="qwen3:32b",
        context="X has a performance bottleneck, Y avoids it entirely",
    )

    # Memory should have the latest value
    results = store.recall("approach", category="decisions")
    assert len(results) == 1
    assert "Y" in results[0].content
    assert results[0].model == "qwen3:32b"

    # History should show both perspectives
    history = store.get_history(results[0].id)
    assert len(history) == 2

    # First entry: creation by Opus
    assert history[0].action == "created"
    assert history[0].model == "claude-opus-4-6"
    assert "X" in history[0].new_content

    # Second entry: update by Qwen, with old value preserved
    assert history[1].action == "updated"
    assert history[1].model == "qwen3:32b"
    assert "X" in history[1].old_content
    assert "Y" in history[1].new_content
    assert "performance bottleneck" in history[1].context


def test_forget_records_history(store):
    m = store.remember("obsolete fact", category="test", key="obsolete")
    store.forget(m.id, model="claude-opus-4-6", context="User confirmed this is no longer true")

    history = store.get_history(m.id)
    forgotten = [h for h in history if h.action == "forgotten"]
    assert len(forgotten) == 1
    assert forgotten[0].model == "claude-opus-4-6"
    assert "no longer true" in forgotten[0].context
    assert forgotten[0].old_content == "obsolete fact"


def test_update_records_history(store):
    m = store.remember("initial", category="test", key="upd")
    store.update(
        m.id,
        content="revised",
        model="qwen3:32b",
        context="Found a more accurate value",
    )

    history = store.get_history(m.id)
    updates = [h for h in history if h.action == "updated"]
    assert len(updates) == 1
    assert updates[0].old_content == "initial"
    assert updates[0].new_content == "revised"
    assert updates[0].model == "qwen3:32b"


def test_get_history_by_model(store):
    store.remember("fact A", category="test", key="a", model="claude-opus-4-6")
    store.remember("fact B", category="test", key="b", model="qwen3:32b")
    store.remember("fact C", category="test", key="c", model="claude-opus-4-6")

    opus_history = store.get_history_by_model("claude-opus-4-6")
    assert len(opus_history) == 2
    assert all(h.model == "claude-opus-4-6" for h in opus_history)

    qwen_history = store.get_history_by_model("qwen3:32b")
    assert len(qwen_history) == 1
