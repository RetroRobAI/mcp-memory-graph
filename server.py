import sys
import os

# Allow imports from this directory
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

import db
import resolver
from config import (
    AUTHORITY_SCORES, CONFLICT_THRESHOLD, DEFAULT_RESULTS, EMBEDDING_MODEL,
    VALID_EDGE_TYPES, VALID_MEMORY_TYPES, VALID_PRIORITIES, VALID_STATUSES,
)

mcp = FastMCP("mcp-memory-graph")
db.init_db()

_model = None

def _embed(text: str) -> list[float]:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model.encode(text, normalize_embeddings=True).tolist()


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def store_memory(
    content: str,
    priority: str = "medium",
    memory_type: str = "general",
    tags: list[str] | None = None,
    auto_resolve_conflicts: bool = True,
) -> dict:
    """
    Store a new memory. Embeds the content, checks for conflicts, and
    optionally auto-resolves them by authority score.

    priority: high | medium | low
    memory_type: user | feedback | project | reference | general
    auto_resolve_conflicts: if True, supersedes lower-authority conflicting memories automatically
    """
    if priority not in VALID_PRIORITIES:
        return {"error": f"priority must be one of {VALID_PRIORITIES}"}
    if memory_type not in VALID_MEMORY_TYPES:
        return {"error": f"memory_type must be one of {VALID_MEMORY_TYPES}"}

    embedding = _embed(content)
    authority = AUTHORITY_SCORES[priority]

    # Conflict detection
    conflicts = resolver.check_conflicts(embedding)
    conflict_summary = [
        {"id": c["id"], "content": c["content"][:100], "similarity": round(c["similarity"], 3),
         "priority": c["priority"]}
        for c in conflicts
    ]

    memory_id = db.store_memory(
        content=content,
        embedding=embedding,
        memory_type=memory_type,
        priority=priority,
        authority_score=authority,
        tags=tags,
    )

    superseded = []
    if auto_resolve_conflicts and conflicts:
        superseded = resolver.auto_resolve(new_id=memory_id, conflicts=conflicts)

    return {
        "id": memory_id,
        "priority": priority,
        "authority_score": authority,
        "conflicts_found": len(conflicts),
        "conflicts": conflict_summary,
        "superseded": superseded,
        "message": f"Stored. {len(superseded)} conflicting memories superseded.",
    }


@mcp.tool()
def retrieve_memories(
    query: str,
    limit: int = DEFAULT_RESULTS,
    min_authority: float = 0.0,
) -> list[dict]:
    """
    Retrieve memories ranked by weighted score (semantic similarity × authority).
    High-priority explicit instructions rank above lower-priority inferred memories
    even when the latter are semantically closer.

    min_authority: filter out memories below this authority score (0.0 = no filter)
    """
    embedding = _embed(query)
    results = db.retrieve_memories(
        query_embedding=embedding,
        limit=limit,
        min_authority=min_authority,
    )
    return [
        {
            "id": r["id"],
            "content": r["content"],
            "memory_type": r["memory_type"],
            "priority": r["priority"],
            "authority_score": r["authority_score"],
            "weighted_score": round(r["weighted_score"], 4),
            "distance": round(r["distance"], 4),
            "tags": r["tags"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in results
    ]


@mcp.tool()
def check_conflicts(content: str) -> dict:
    """
    Check if a piece of content conflicts with any existing active memories
    before storing it. Returns potential conflicts above the similarity threshold.
    """
    embedding = _embed(content)
    conflicts = resolver.check_conflicts(embedding)
    return {
        "threshold": CONFLICT_THRESHOLD,
        "conflicts_found": len(conflicts),
        "conflicts": [
            {
                "id": c["id"],
                "content": c["content"],
                "priority": c["priority"],
                "authority_score": c["authority_score"],
                "similarity": round(c["similarity"], 3),
            }
            for c in conflicts
        ],
    }


@mcp.tool()
def update_memory(
    memory_id: str,
    content: str | None = None,
    priority: str | None = None,
    supersedes_id: str | None = None,
) -> dict:
    """
    Update an existing memory. Optionally mark another memory as superseded.
    If supersedes_id is provided, that memory is marked status=superseded and
    a supersedes edge is created.
    """
    if priority and priority not in VALID_PRIORITIES:
        return {"error": f"priority must be one of {VALID_PRIORITIES}"}

    embedding = _embed(content) if content else None
    authority = AUTHORITY_SCORES[priority] if priority else None

    ok = db.update_memory(
        memory_id=memory_id,
        content=content,
        priority=priority,
        authority_score=authority,
        embedding=embedding,
    )
    if not ok:
        return {"error": f"Memory {memory_id} not found"}

    result: dict = {"id": memory_id, "updated": True}

    if supersedes_id:
        resolver.supersede(new_id=memory_id, old_id=supersedes_id)
        result["superseded"] = supersedes_id

    return result


@mcp.tool()
def delete_memory(memory_id: str) -> dict:
    """Soft-delete a memory (sets status=deleted, preserves history)."""
    ok = db.update_memory(memory_id=memory_id, status="deleted")
    if not ok:
        return {"error": f"Memory {memory_id} not found"}
    return {"id": memory_id, "status": "deleted"}


@mcp.tool()
def add_memory_edge(from_id: str, to_id: str, edge_type: str) -> dict:
    """
    Add a typed relationship between two memories.
    edge_type: supersedes | relates_to | contradicts | referenced_by
    """
    if edge_type not in VALID_EDGE_TYPES:
        return {"error": f"edge_type must be one of {VALID_EDGE_TYPES}"}
    edge_id = db.add_edge(from_id=from_id, to_id=to_id, edge_type=edge_type)
    return {"edge_id": edge_id, "from_id": from_id, "to_id": to_id, "edge_type": edge_type}


@mcp.tool()
def get_related_memories(memory_id: str) -> dict:
    """
    Get all memories connected to this one via edges.
    Returns the full relationship graph for a memory.
    """
    mem = db.get_memory_by_id(memory_id)
    if not mem:
        return {"error": f"Memory {memory_id} not found"}
    edges = db.get_edges(memory_id)
    return {
        "memory_id": memory_id,
        "content": mem["content"],
        "priority": mem["priority"],
        "status": mem["status"],
        "edges": edges,
    }


@mcp.tool()
def list_memories(
    status: str = "active",
    memory_type: str | None = None,
    priority: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    List memories with optional filters.
    status: active | superseded | deleted
    priority: high | medium | low
    memory_type: user | feedback | project | reference | general
    """
    if status not in VALID_STATUSES:
        return [{"error": f"status must be one of {VALID_STATUSES}"}]
    results = db.list_memories(
        status=status,
        memory_type=memory_type,
        priority=priority,
        limit=limit,
    )
    return [
        {
            "id": r["id"],
            "content": r["content"][:200],
            "memory_type": r["memory_type"],
            "priority": r["priority"],
            "authority_score": r["authority_score"],
            "tags": r["tags"],
            "updated_at": r["updated_at"],
        }
        for r in results
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")