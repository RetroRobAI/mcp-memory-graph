import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import sqlite_vec

from config import DB_PATH, EMBEDDING_DIM


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = _connect()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id           TEXT PRIMARY KEY,
                content      TEXT NOT NULL,
                memory_type  TEXT NOT NULL DEFAULT 'general',
                priority     TEXT NOT NULL DEFAULT 'medium',
                authority_score REAL NOT NULL DEFAULT 0.6,
                status       TEXT NOT NULL DEFAULT 'active',
                tags         TEXT NOT NULL DEFAULT '[]',
                metadata     TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vecs USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_edges (
                id         TEXT PRIMARY KEY,
                from_id    TEXT NOT NULL REFERENCES memories(id),
                to_id      TEXT NOT NULL REFERENCES memories(id),
                edge_type  TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_from ON memory_edges(from_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_to ON memory_edges(to_id)")
    conn.close()


def store_memory(
    content: str,
    embedding: list[float],
    memory_type: str = "general",
    priority: str = "medium",
    authority_score: float = 0.6,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    memory_id = str(uuid.uuid4())
    now = _now()
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT INTO memories
               (id, content, memory_type, priority, authority_score, status, tags, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)""",
            (
                memory_id, content, memory_type, priority, authority_score,
                json.dumps(tags or []), json.dumps(metadata or {}), now, now,
            ),
        )
        conn.execute(
            "INSERT INTO memory_vecs(id, embedding) VALUES (?, ?)",
            (memory_id, sqlite_vec.serialize_float32(embedding)),
        )
    conn.close()
    return memory_id


def retrieve_memories(
    query_embedding: list[float],
    limit: int = 10,
    min_authority: float = 0.0,
    status: str = "active",
) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT m.id, m.content, m.memory_type, m.priority, m.authority_score,
               m.status, m.tags, m.metadata, m.created_at, m.updated_at,
               v.distance
        FROM memory_vecs v
        JOIN memories m ON m.id = v.id
        WHERE v.embedding MATCH ?
          AND v.k = ?
          AND m.status = ?
          AND m.authority_score >= ?
        ORDER BY (v.distance / (m.authority_score + 0.001))
        """,
        (sqlite_vec.serialize_float32(query_embedding), limit * 3, status, min_authority),
    ).fetchall()
    conn.close()
    results = [dict(r) for r in rows]
    for r in results:
        r["tags"] = json.loads(r["tags"])
        r["metadata"] = json.loads(r["metadata"])
        # Weighted score: lower is better (distance), higher authority = better rank
        r["weighted_score"] = 1.0 - (r["distance"] / (r["authority_score"] + 0.001) / 10)
    results.sort(key=lambda x: x["weighted_score"], reverse=True)
    return results[:limit]


def get_memory_by_id(memory_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    conn.close()
    if not row:
        return None
    r = dict(row)
    r["tags"] = json.loads(r["tags"])
    r["metadata"] = json.loads(r["metadata"])
    return r


def update_memory(
    memory_id: str,
    content: str | None = None,
    priority: str | None = None,
    authority_score: float | None = None,
    status: str | None = None,
    embedding: list[float] | None = None,
) -> bool:
    conn = _connect()
    existing = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not existing:
        conn.close()
        return False
    updates = {"updated_at": _now()}
    if content is not None:
        updates["content"] = content
    if priority is not None:
        updates["priority"] = priority
    if authority_score is not None:
        updates["authority_score"] = authority_score
    if status is not None:
        updates["status"] = status
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with conn:
        conn.execute(
            f"UPDATE memories SET {set_clause} WHERE id = ?",
            (*updates.values(), memory_id),
        )
        if embedding is not None:
            conn.execute("UPDATE memory_vecs SET embedding = ? WHERE id = ?",
                         (sqlite_vec.serialize_float32(embedding), memory_id))
    conn.close()
    return True


def add_edge(from_id: str, to_id: str, edge_type: str) -> str:
    edge_id = str(uuid.uuid4())
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO memory_edges (id, from_id, to_id, edge_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (edge_id, from_id, to_id, edge_type, _now()),
        )
    conn.close()
    return edge_id


def get_edges(memory_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT e.*,
                  mf.content as from_content, mf.priority as from_priority,
                  mt.content as to_content, mt.priority as to_priority
           FROM memory_edges e
           JOIN memories mf ON mf.id = e.from_id
           JOIN memories mt ON mt.id = e.to_id
           WHERE e.from_id = ? OR e.to_id = ?""",
        (memory_id, memory_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_memories(
    status: str = "active",
    memory_type: str | None = None,
    priority: str | None = None,
    limit: int = 50,
) -> list[dict]:
    conn = _connect()
    query = "SELECT * FROM memories WHERE status = ?"
    params: list[Any] = [status]
    if memory_type:
        query += " AND memory_type = ?"
        params.append(memory_type)
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    query += " ORDER BY authority_score DESC, updated_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    results = [dict(r) for r in rows]
    for r in results:
        r["tags"] = json.loads(r["tags"])
        r["metadata"] = json.loads(r["metadata"])
    return results


def find_similar(embedding: list[float], threshold: float, limit: int = 5) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT m.id, m.content, m.priority, m.authority_score, m.status, v.distance
        FROM memory_vecs v
        JOIN memories m ON m.id = v.id
        WHERE v.embedding MATCH ? AND v.k = ? AND m.status = 'active'
        ORDER BY v.distance
        """,
        (sqlite_vec.serialize_float32(embedding), limit),
    ).fetchall()
    conn.close()
    # Convert distance to similarity: 1 - (distance/2) for cosine distance in [0,2]
    results = []
    for r in rows:
        d = dict(r)
        d["similarity"] = max(0.0, 1.0 - d["distance"] / 2.0)
        if d["similarity"] >= threshold:
            results.append(d)
    return results
