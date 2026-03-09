"""SQLite + FTS5 memory store with relevance-ranked recall."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(content: str, salt: str = "") -> str:
    return hashlib.sha256(f"{content}{salt}{time.time_ns()}".encode()).hexdigest()[:12]


@dataclass
class Memory:
    id: str
    content: str
    memory_type: str  # "fact", "episode", "preference"
    category: str
    key: Optional[str]
    tags: list[str]
    confidence: float
    source: Optional[str]
    created_at: str
    updated_at: str
    accessed_at: str
    access_count: int
    model: Optional[str] = None  # which model created/last updated this
    context: Optional[str] = None  # reasoning/justification for the memory
    score: float = 0.0  # populated during recall

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HistoryEntry:
    id: str
    memory_id: str
    action: str  # "created", "updated", "forgotten"
    model: Optional[str]
    old_content: Optional[str]
    new_content: Optional[str]
    old_confidence: Optional[float]
    new_confidence: Optional[float]
    context: Optional[str]  # reasoning for the change
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Relationship:
    id: str
    entity_from: str
    entity_to: str
    relation_type: str
    metadata: Optional[dict]
    confidence: float
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL CHECK(memory_type IN ('fact', 'episode', 'preference')),
    category TEXT NOT NULL DEFAULT 'general',
    key TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    model TEXT,
    context TEXT,
    UNIQUE(category, key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, category, key, tags,
    content=memories, content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, category, key, tags)
    VALUES (new.rowid, new.content, new.category, new.key, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, category, key, tags)
    VALUES ('delete', old.rowid, old.content, old.category, old.key, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, category, key, tags)
    VALUES ('delete', old.rowid, old.content, old.category, old.key, old.tags);
    INSERT INTO memories_fts(rowid, content, category, key, tags)
    VALUES (new.rowid, new.content, new.category, new.key, new.tags);
END;

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    entity_from TEXT NOT NULL,
    entity_to TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    metadata TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(entity_from, entity_to, relation_type)
);

CREATE TABLE IF NOT EXISTS memory_history (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('created', 'updated', 'forgotten')),
    model TEXT,
    old_content TEXT,
    new_content TEXT,
    old_confidence REAL,
    new_confidence REAL,
    context TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_memory ON memory_history(memory_id);
CREATE INDEX IF NOT EXISTS idx_history_model ON memory_history(model);
CREATE INDEX IF NOT EXISTS idx_history_created ON memory_history(created_at);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(category, key);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at);
CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(entity_from);
CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(entity_to);
"""


class MemoryStore:
    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".engram" / "memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self):
        """Add columns that may be missing from older schema versions."""
        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        migrations = [
            ("model", "TEXT"),
            ("context", "TEXT"),
        ]
        for col, col_type in migrations:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {col_type}")

    def close(self):
        self._conn.close()

    # ── Remember ────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        category: str = "general",
        key: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        model: str | None = None,
        context: str | None = None,
    ) -> Memory:
        """Store a new memory. If a fact with the same category+key exists, update it."""
        now = _now()
        tags = tags or []
        mid = _short_id(content)

        if key and memory_type == "fact":
            # Upsert: update existing fact with same category+key
            existing = self._conn.execute(
                "SELECT * FROM memories WHERE category = ? AND key = ?",
                (category, key),
            ).fetchone()
            if existing:
                # Record history before overwriting
                self._record_history(
                    memory_id=existing["id"],
                    action="updated",
                    model=model,
                    old_content=existing["content"],
                    new_content=content,
                    old_confidence=existing["confidence"],
                    new_confidence=confidence,
                    context=context,
                )
                self._conn.execute(
                    """UPDATE memories SET content = ?, tags = ?, confidence = ?,
                       source = ?, updated_at = ?, accessed_at = ?,
                       model = ?, context = ?
                       WHERE id = ?""",
                    (content, json.dumps(tags), confidence, source, now, now,
                     model, context, existing["id"]),
                )
                self._conn.commit()
                return self._get_memory(existing["id"])

        try:
            self._conn.execute(
                """INSERT INTO memories (id, content, memory_type, category, key, tags,
                   confidence, source, created_at, updated_at, accessed_at, access_count,
                   model, context)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (mid, content, memory_type, category, key, json.dumps(tags),
                 confidence, source, now, now, now, model, context),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # ID collision (extremely unlikely) — retry with different salt
            mid = _short_id(content, salt="retry")
            self._conn.execute(
                """INSERT INTO memories (id, content, memory_type, category, key, tags,
                   confidence, source, created_at, updated_at, accessed_at, access_count,
                   model, context)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (mid, content, memory_type, category, key, json.dumps(tags),
                 confidence, source, now, now, now, model, context),
            )
            self._conn.commit()

        # Record creation in history
        self._record_history(
            memory_id=mid,
            action="created",
            model=model,
            old_content=None,
            new_content=content,
            old_confidence=None,
            new_confidence=confidence,
            context=context,
        )

        return self._get_memory(mid)

    # ── Recall ──────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        category: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[Memory]:
        """Search memories using FTS5 with relevance + recency + frequency scoring."""
        # FTS5 search
        fts_query = self._build_fts_query(query)

        sql = """
            SELECT m.*, fts.rank AS fts_rank
            FROM memories_fts fts
            JOIN memories m ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ?
        """
        params: list = [fts_query]

        if category:
            sql += " AND m.category = ?"
            params.append(category)
        if memory_type:
            sql += " AND m.memory_type = ?"
            params.append(memory_type)
        if min_confidence > 0:
            sql += " AND m.confidence >= ?"
            params.append(min_confidence)

        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(limit * 3)  # overfetch for re-ranking

        rows = self._conn.execute(sql, params).fetchall()
        memories = [self._row_to_memory(r) for r in rows]

        # Tag filter (post-query since tags are JSON)
        if tags:
            tag_set = set(tags)
            memories = [m for m in memories if tag_set & set(m.tags)]

        # Re-rank with composite score
        for m in memories:
            m.score = self._compute_score(m, rows)

        memories.sort(key=lambda m: m.score, reverse=True)
        memories = memories[:limit]

        # Update access timestamps
        now = _now()
        for m in memories:
            self._conn.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (now, m.id),
            )
        self._conn.commit()

        return memories

    def recall_by_id(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID."""
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        now = _now()
        self._conn.execute(
            "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            (now, memory_id),
        )
        self._conn.commit()
        return self._row_to_memory(row)

    # ── Forget ──────────────────────────────────────────────────────

    def forget(self, memory_id: str, model: str | None = None, context: str | None = None) -> bool:
        """Delete a memory by ID."""
        existing = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not existing:
            return False
        self._record_history(
            memory_id=memory_id,
            action="forgotten",
            model=model,
            old_content=existing["content"],
            new_content=None,
            old_confidence=existing["confidence"],
            new_confidence=None,
            context=context,
        )
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return True

    def forget_by_key(self, category: str, key: str, model: str | None = None, context: str | None = None) -> bool:
        """Delete a fact by category + key."""
        existing = self._conn.execute(
            "SELECT * FROM memories WHERE category = ? AND key = ?", (category, key)
        ).fetchone()
        if not existing:
            return False
        self._record_history(
            memory_id=existing["id"],
            action="forgotten",
            model=model,
            old_content=existing["content"],
            new_content=None,
            old_confidence=existing["confidence"],
            new_confidence=None,
            context=context,
        )
        self._conn.execute("DELETE FROM memories WHERE category = ? AND key = ?", (category, key))
        self._conn.commit()
        return True

    # ── Update ──────────────────────────────────────────────────────

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
        key: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        model: str | None = None,
        context: str | None = None,
    ) -> Memory | None:
        """Update fields of an existing memory."""
        existing = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not existing:
            return None

        # Record history
        self._record_history(
            memory_id=memory_id,
            action="updated",
            model=model,
            old_content=existing["content"],
            new_content=content or existing["content"],
            old_confidence=existing["confidence"],
            new_confidence=confidence if confidence is not None else existing["confidence"],
            context=context,
        )

        now = _now()
        updates = {"updated_at": now}
        if content is not None:
            updates["content"] = content
        if category is not None:
            updates["category"] = category
        if key is not None:
            updates["key"] = key
        if tags is not None:
            updates["tags"] = json.dumps(tags)
        if confidence is not None:
            updates["confidence"] = confidence
        if model is not None:
            updates["model"] = model
        if context is not None:
            updates["context"] = context

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [memory_id]
        self._conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        self._conn.commit()
        return self._get_memory(memory_id)

    # ── Relationships ───────────────────────────────────────────────

    def relate(
        self,
        entity_from: str,
        entity_to: str,
        relation_type: str,
        metadata: dict | None = None,
        confidence: float = 1.0,
    ) -> Relationship:
        """Create or update a relationship between two entities."""
        now = _now()
        rid = _short_id(f"{entity_from}:{entity_to}:{relation_type}")

        existing = self._conn.execute(
            "SELECT id FROM relationships WHERE entity_from = ? AND entity_to = ? AND relation_type = ?",
            (entity_from, entity_to, relation_type),
        ).fetchone()

        if existing:
            self._conn.execute(
                "UPDATE relationships SET metadata = ?, confidence = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata) if metadata else None, confidence, now, existing["id"]),
            )
            self._conn.commit()
            rid = existing["id"]
        else:
            self._conn.execute(
                """INSERT INTO relationships (id, entity_from, entity_to, relation_type,
                   metadata, confidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (rid, entity_from, entity_to, relation_type,
                 json.dumps(metadata) if metadata else None, confidence, now, now),
            )
            self._conn.commit()

        return self._get_relationship(rid)

    def get_entity(self, entity: str) -> dict:
        """Get all memories and relationships for an entity."""
        # Search memories mentioning this entity
        memories = self.recall(entity, limit=20)

        # Get relationships
        rows = self._conn.execute(
            """SELECT * FROM relationships
               WHERE entity_from = ? OR entity_to = ?
               ORDER BY confidence DESC""",
            (entity, entity),
        ).fetchall()
        relationships = [self._row_to_relationship(r) for r in rows]

        return {
            "entity": entity,
            "memories": [m.to_dict() for m in memories],
            "relationships": [r.to_dict() for r in relationships],
        }

    def unrelate(self, entity_from: str, entity_to: str, relation_type: str) -> bool:
        """Remove a relationship."""
        cur = self._conn.execute(
            "DELETE FROM relationships WHERE entity_from = ? AND entity_to = ? AND relation_type = ?",
            (entity_from, entity_to, relation_type),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Introspection ───────────────────────────────────────────────

    def list_categories(self) -> list[dict]:
        """List all categories with memory counts."""
        rows = self._conn.execute(
            "SELECT category, memory_type, COUNT(*) as count FROM memories GROUP BY category, memory_type ORDER BY count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Get overall memory statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = self._conn.execute(
            "SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type"
        ).fetchall()
        relationships = self._conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        oldest = self._conn.execute("SELECT MIN(created_at) FROM memories").fetchone()[0]
        newest = self._conn.execute("SELECT MAX(updated_at) FROM memories").fetchone()[0]
        stale = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE accessed_at < datetime('now', '-30 days')"
        ).fetchone()[0]

        return {
            "total_memories": total,
            "by_type": {r["memory_type"]: r["count"] for r in by_type},
            "total_relationships": relationships,
            "oldest_memory": oldest,
            "newest_memory": newest,
            "stale_memories_30d": stale,
            "db_path": str(self.db_path),
        }

    def get_stale(self, days: int = 30, limit: int = 20) -> list[Memory]:
        """Get memories not accessed in N days."""
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE accessed_at < datetime('now', ? || ' days')
               ORDER BY accessed_at ASC LIMIT ?""",
            (f"-{days}", limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # ── Context (priming) ───────────────────────────────────────────

    def get_context(self, topic: str | None = None, limit: int = 20) -> dict:
        """Get a priming context for conversation start."""
        result = {"preferences": [], "recent": [], "topic_memories": []}

        # Always include preferences
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE memory_type = 'preference' ORDER BY confidence DESC LIMIT 10"
        ).fetchall()
        result["preferences"] = [self._row_to_memory(r).to_dict() for r in rows]

        # Recently updated facts
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE memory_type = 'fact' ORDER BY updated_at DESC LIMIT 10"
        ).fetchall()
        result["recent"] = [self._row_to_memory(r).to_dict() for r in rows]

        # Topic-specific if provided
        if topic:
            result["topic_memories"] = [m.to_dict() for m in self.recall(topic, limit=limit)]

        return result

    # ── History / Provenance ────────────────────────────────────────

    def get_history(self, memory_id: str) -> list[HistoryEntry]:
        """Get the full audit trail for a memory."""
        rows = self._conn.execute(
            "SELECT * FROM memory_history WHERE memory_id = ? ORDER BY created_at ASC",
            (memory_id,),
        ).fetchall()
        return [self._row_to_history(r) for r in rows]

    def get_history_by_model(self, model: str, limit: int = 20) -> list[HistoryEntry]:
        """Get all history entries from a specific model."""
        rows = self._conn.execute(
            "SELECT * FROM memory_history WHERE model = ? ORDER BY created_at DESC LIMIT ?",
            (model, limit),
        ).fetchall()
        return [self._row_to_history(r) for r in rows]

    def _record_history(
        self,
        memory_id: str,
        action: str,
        model: str | None,
        old_content: str | None,
        new_content: str | None,
        old_confidence: float | None,
        new_confidence: float | None,
        context: str | None,
    ):
        """Record a history entry for a memory mutation."""
        hid = _short_id(f"{memory_id}:{action}")
        now = _now()
        self._conn.execute(
            """INSERT INTO memory_history (id, memory_id, action, model,
               old_content, new_content, old_confidence, new_confidence,
               context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (hid, memory_id, action, model, old_content, new_content,
             old_confidence, new_confidence, context, now),
        )
        # Don't commit here — caller manages the transaction

    def _row_to_history(self, row: sqlite3.Row) -> HistoryEntry:
        return HistoryEntry(
            id=row["id"],
            memory_id=row["memory_id"],
            action=row["action"],
            model=row["model"],
            old_content=row["old_content"],
            new_content=row["new_content"],
            old_confidence=row["old_confidence"],
            new_confidence=row["new_confidence"],
            context=row["context"],
            created_at=row["created_at"],
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _get_memory(self, mid: str) -> Memory:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        return self._row_to_memory(row)

    def _get_relationship(self, rid: str) -> Relationship:
        row = self._conn.execute("SELECT * FROM relationships WHERE id = ?", (rid,)).fetchone()
        return self._row_to_relationship(row)

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            category=row["category"],
            key=row["key"],
            tags=json.loads(row["tags"]),
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            accessed_at=row["accessed_at"],
            access_count=row["access_count"],
            model=row["model"],
            context=row["context"],
        )

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        return Relationship(
            id=row["id"],
            entity_from=row["entity_from"],
            entity_to=row["entity_to"],
            relation_type=row["relation_type"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _build_fts_query(self, query: str) -> str:
        """Build an FTS5 query from natural language input.

        Tokenizes, strips FTS operators, and joins with OR for broad matching.
        """
        # Split into words, strip punctuation, remove FTS5 operators
        fts_operators = {"AND", "OR", "NOT", "NEAR"}
        words = []
        for word in query.split():
            cleaned = word.strip('".,:;!?()[]{}')
            if cleaned and cleaned.upper() not in fts_operators and len(cleaned) > 1:
                words.append(f'"{cleaned}"')
        if not words:
            return f'"{query}"'
        return " OR ".join(words)

    def _compute_score(self, memory: Memory, raw_rows: list[sqlite3.Row]) -> float:
        """Composite relevance score: FTS rank + recency + frequency + confidence."""
        # Find the raw FTS rank for this memory
        fts_rank = 0.0
        for row in raw_rows:
            if row["id"] == memory.id:
                fts_rank = abs(row["fts_rank"])  # FTS5 rank is negative
                break

        # Normalize FTS rank (higher = better)
        fts_score = fts_rank

        # Recency: exponential decay from last update
        try:
            updated = datetime.fromisoformat(memory.updated_at)
            age_days = (datetime.now(timezone.utc) - updated).total_seconds() / 86400
            recency_score = math.exp(-0.02 * age_days)  # half-life ~35 days
        except (ValueError, TypeError):
            recency_score = 0.5

        # Frequency: log scale of access count
        freq_score = math.log1p(memory.access_count) / 10.0

        # Composite: weighted sum
        return (fts_score * 0.5) + (recency_score * 0.3) + (freq_score * 0.1) + (memory.confidence * 0.1)
