from config import AUTHORITY_SCORES, CONFLICT_THRESHOLD
import db


def authority_score_for(priority: str) -> float:
    return AUTHORITY_SCORES.get(priority, 0.6)


def check_conflicts(embedding: list[float]) -> list[dict]:
    """Return active memories that are semantically similar enough to conflict."""
    return db.find_similar(embedding, threshold=CONFLICT_THRESHOLD, limit=5)


def supersede(new_id: str, old_id: str) -> None:
    """Mark old_id as superseded and create a supersedes edge from new_id."""
    db.update_memory(old_id, status="superseded")
    db.add_edge(from_id=new_id, to_id=old_id, edge_type="supersedes")


def auto_resolve(new_id: str, conflicts: list[dict]) -> list[str]:
    """
    Auto-resolve conflicts by authority + recency.
    Returns list of memory IDs that were superseded.
    """
    new_mem = db.get_memory_by_id(new_id)
    if not new_mem:
        return []

    new_score = new_mem["authority_score"]
    superseded = []

    for conflict in conflicts:
        if conflict["id"] == new_id:
            continue
        existing_score = conflict["authority_score"]
        # New memory wins if it has equal or higher authority
        if new_score >= existing_score:
            supersede(new_id=new_id, old_id=conflict["id"])
            superseded.append(conflict["id"])
        else:
            # Existing memory outranks new one — add a relates_to edge, don't supersede
            db.add_edge(from_id=new_id, to_id=conflict["id"], edge_type="relates_to")

    return superseded
