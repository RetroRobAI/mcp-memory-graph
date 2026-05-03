import os

# Storage
DB_PATH = os.environ.get(
    "MEMORY_GRAPH_DB_PATH",
    os.path.join(os.path.expanduser("~"), ".mcp-memory-graph", "memories.db")
)

# Embedding model — any sentence-transformers model works; 384-dim default
EMBEDDING_MODEL = os.environ.get("MEMORY_GRAPH_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.environ.get("MEMORY_GRAPH_DIM", "384"))

# Authority scores per priority level (0.0 – 1.0)
AUTHORITY_SCORES: dict[str, float] = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
}

# Cosine similarity threshold above which two memories are flagged as conflicting
CONFLICT_THRESHOLD = float(os.environ.get("MEMORY_GRAPH_CONFLICT_THRESHOLD", "0.85"))

# How many results to return by default
DEFAULT_RESULTS = int(os.environ.get("MEMORY_GRAPH_DEFAULT_RESULTS", "10"))

# Valid values
VALID_PRIORITIES = {"high", "medium", "low"}
VALID_MEMORY_TYPES = {"user", "feedback", "project", "reference", "general"}
VALID_EDGE_TYPES = {"supersedes", "relates_to", "contradicts", "referenced_by"}
VALID_STATUSES = {"active", "superseded", "deleted"}
