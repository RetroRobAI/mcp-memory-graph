# mcp-memory-graph

A context-aware memory MCP server for Claude Code and any MCP-compatible AI agent.

Goes beyond basic vector search by adding **authority weighting**, **conflict detection**, and **typed relationship edges** between memories — so your agent always retrieves the right answer when sources disagree.

Inspired by the context engine architecture described in [Unblocked's "How a Context Engine Actually Works"](https://getunblocked.com/blog/how-a-context-engine-actually-works-and-why-you-need-to-care-now/).

---

## Why this exists

Standard memory MCP servers store and retrieve memories by semantic similarity. That works until you have conflicting memories — an old instruction saying one thing and a new one saying another. Without authority weighting, the agent retrieves whichever is semantically closer to the query, not whichever is more trustworthy.

**mcp-memory-graph** solves this with three mechanisms:

| Problem | Solution |
|---|---|
| All memories treated equally | Priority tiers: `high` / `medium` / `low` → authority scores 1.0 / 0.6 / 0.3 |
| Stale memories persist silently | Supersession tracking: old memories marked `status=superseded` with typed edges |
| Duplicates accumulate over time | Conflict detection before every store; auto-resolve by authority |

---

## Installation

```bash
pip install mcp-memory-graph
```

Or run directly:
```bash
git clone <repo>
cd mcp-memory-graph
pip install -r requirements.txt
python server.py
```

---

## Claude Code setup

Add to `~/.claude.json` under `mcpServers`:

```json
"mcp-memory-graph": {
  "type": "stdio",
  "command": "python",
  "args": ["/path/to/mcp-memory-graph/server.py"],
  "env": {
    "MEMORY_GRAPH_DB_PATH": "/path/to/memories.db"
  }
}
```

---

## Migrating from an existing memory service

If you have an existing memory service (mcp-memory-service, Mem0, or a markdown-based memory system), you can import your memories into mcp-memory-graph using the included migration script.

**Migration is manual and opt-in** — it never runs automatically. Nothing is written until you explicitly confirm.

### Run the migration script

```bash
python migrate.py
```

The script will:
1. Auto-detect any existing `mcp-memory-service` SQLite database
2. Ask if you have a markdown memory directory to import
3. Show you how many memories it found
4. Present three choices:
   - **[1] Migrate** — import everything into mcp-memory-graph
   - **[2] Run in parallel** — start mcp-memory-graph fresh, keep your old service running
   - **[3] Skip** — do nothing
5. Ask for a final confirmation before writing anything

Your existing memory service is never modified — the script only reads from it.

### When to run it

Run `migrate.py` once, after installation, before your first Claude session with mcp-memory-graph. If you choose to run in parallel first and decide to migrate later, simply run the script again.

---

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_GRAPH_DB_PATH` | `~/.mcp-memory-graph/memories.db` | SQLite database path |
| `MEMORY_GRAPH_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `MEMORY_GRAPH_DIM` | `384` | Embedding dimensions |
| `MEMORY_GRAPH_CONFLICT_THRESHOLD` | `0.85` | Cosine similarity above which memories are flagged as conflicting |
| `MEMORY_GRAPH_DEFAULT_RESULTS` | `10` | Default retrieval limit |

---

## Tools

| Tool | Description |
|---|---|
| `store_memory` | Store with conflict detection and optional auto-resolve |
| `retrieve_memories` | Semantic search ranked by similarity × authority |
| `check_conflicts` | Preview conflicts before storing |
| `update_memory` | Update content/priority with supersession tracking |
| `delete_memory` | Soft delete (preserves history) |
| `add_memory_edge` | Manually add typed relationship |
| `get_related_memories` | Traverse relationship graph for a memory |
| `list_memories` | List with filters (status, type, priority) |

---

## Priority system

```python
priority="high"    # authority_score=1.0  — explicit instructions, confirmed preferences
priority="medium"  # authority_score=0.6  — inferred preferences, reference data
priority="low"     # authority_score=0.3  — session summaries, historical context
```

Retrieval ranking: `weighted_score = 1 - (distance / (authority_score + 0.001) / 10)`

A high-authority memory will rank above a semantically closer low-authority one when their similarity scores are within ~3x of each other.

---

## Edge types

- `supersedes` — this memory replaces another
- `relates_to` — connected but not conflicting
- `contradicts` — explicitly conflicting, unresolved
- `referenced_by` — another memory cites this one

---

## Stack

- [sqlite-vec](https://github.com/asg017/sqlite-vec) — vector similarity search
- [sentence-transformers](https://www.sbert.net/) — local embeddings, no API key needed
- [FastMCP](https://github.com/jlowin/fastmcp) — MCP server framework

---

## License

MIT
