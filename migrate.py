#!/usr/bin/env python3
"""
migrate.py — Import existing memories into mcp-memory-graph.

Supports migrating from:
  - mcp-memory-service  (SQLite database)
  - Flat markdown files (e.g. Agent Neo file-based memory system)

The user is always asked before any data is written.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import sqlite3

from sentence_transformers import SentenceTransformer

import db as graph_db
from config import AUTHORITY_SCORES, DB_PATH, EMBEDDING_MODEL


# ── Source detection ───────────────────────────────────────────────────────────

def _detect_mcp_memory_service() -> str | None:
    """Return path to mcp-memory-service SQLite DB if it exists and has data."""
    candidates = [
        os.path.join(os.path.expanduser("~"), ".mcp-memory", "memories.db"),
        os.path.join(os.path.expanduser("~"), ".mcp-memory", "memory.db"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                conn = sqlite3.connect(path)
                count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                conn.close()
                if count > 0:
                    return path
            except Exception:
                pass
    return None


def _detect_markdown_memories(directory: str | None = None) -> list[str]:
    """Return markdown memory files found in a directory."""
    if not directory:
        return []
    files = []
    for f in os.listdir(directory):
        if f.endswith(".md") and f != "MEMORY.md" and not f.startswith("session_"):
            fp = os.path.join(directory, f)
            if os.path.isfile(fp) and os.path.getsize(fp) > 50:
                files.append(fp)
    return files


# ── Source readers ─────────────────────────────────────────────────────────────

def _read_mcp_memory_service(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # mcp-memory-service schema varies by version; try common column names
    try:
        rows = conn.execute("SELECT * FROM memories WHERE status != 'deleted'").fetchall()
    except Exception:
        rows = conn.execute("SELECT * FROM memories").fetchall()
    conn.close()
    records = []
    for row in rows:
        r = dict(row)
        records.append({
            "content": r.get("content", r.get("text", "")),
            "memory_type": r.get("type", "general"),
            "priority": r.get("priority", "medium"),
            "tags": json.loads(r["tags"]) if isinstance(r.get("tags"), str) else [],
            "source": "mcp-memory-service",
        })
    return [r for r in records if r["content"]]


def _read_markdown_file(filepath: str) -> dict | None:
    """Parse a memory markdown file with YAML-style frontmatter."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract frontmatter
    fm_match = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
    if not fm_match:
        return None

    fm_raw, body = fm_match.group(1), fm_match.group(2).strip()
    frontmatter: dict = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            frontmatter[key.strip()] = val.strip()

    memory_type = frontmatter.get("type", "general")
    priority = frontmatter.get("priority", "medium")
    status = frontmatter.get("status", "active")

    if status == "superseded":
        return None  # Skip already-superseded memories

    content = body if body else frontmatter.get("name", "")
    if not content:
        return None

    return {
        "content": content,
        "memory_type": memory_type,
        "priority": priority,
        "tags": [],
        "source": os.path.basename(filepath),
    }


# ── Migration ─────────────────────────────────────────────────────────────────

def _confirm(prompt: str) -> bool:
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in ("y", "yes")


def _migrate_records(records: list[dict], model: SentenceTransformer) -> int:
    imported = 0
    for i, record in enumerate(records, 1):
        print(f"  [{i}/{len(records)}] {record['content'][:70]}...")
        embedding = model.encode(record["content"], normalize_embeddings=True).tolist()
        authority = AUTHORITY_SCORES.get(record["priority"], 0.6)
        graph_db.store_memory(
            content=record["content"],
            embedding=embedding,
            memory_type=record.get("memory_type", "general"),
            priority=record.get("priority", "medium"),
            authority_score=authority,
            tags=record.get("tags", []),
            metadata={"migrated_from": record.get("source", "unknown")},
        )
        imported += 1
    return imported


def main() -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║        mcp-memory-graph Migration Tool           ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # ── Detect sources ─────────────────────────────────────────────────────────
    sources_found: list[tuple[str, str | list]] = []

    mcp_db = _detect_mcp_memory_service()
    if mcp_db:
        records = _read_mcp_memory_service(mcp_db)
        sources_found.append(("mcp-memory-service", records))
        print(f"  Found mcp-memory-service: {len(records)} memories at {mcp_db}")

    md_dir = input("\nEnter path to a markdown memory directory to import (or press Enter to skip): ").strip()
    if md_dir and os.path.isdir(md_dir):
        md_files = _detect_markdown_memories(md_dir)
        md_records = []
        for fp in md_files:
            r = _read_markdown_file(fp)
            if r:
                md_records.append(r)
        if md_records:
            sources_found.append(("markdown-files", md_records))
            print(f"  Found markdown memory files: {len(md_records)} importable memories")
        else:
            print("  No importable markdown memories found in that directory.")
    elif md_dir:
        print(f"  Directory not found: {md_dir}")

    if not sources_found:
        print("\nNo existing memory sources detected. Nothing to migrate.")
        print("You can start fresh — mcp-memory-graph will build its DB as you use it.\n")
        return

    # ── Present options ────────────────────────────────────────────────────────
    total = sum(len(r) if isinstance(r, list) else 0 for _, r in sources_found)
    print(f"\n{total} total memories found across {len(sources_found)} source(s).")
    print("\nOptions:")
    print("  [1] Migrate all memories into mcp-memory-graph (recommended)")
    print("  [2] Run in parallel — keep existing service, start mcp-memory-graph fresh")
    print("  [3] Skip — do nothing")

    choice = input("\nYour choice [1/2/3]: ").strip()

    if choice == "3" or choice == "":
        print("\nSkipped. No changes made.")
        return

    if choice == "2":
        print("\nRunning in parallel. Your existing memory service is untouched.")
        print("mcp-memory-graph will start building its own DB as you use it.")
        print("You can re-run this script at any time to migrate later.\n")
        return

    if choice != "1":
        print("\nInvalid choice. No changes made.")
        return

    # ── Confirm migration ─────────────────────────────────────────────────────
    print(f"\nThis will import {total} memories into: {DB_PATH}")
    print("Your existing memory service will NOT be modified.")
    if not _confirm("Proceed with migration?"):
        print("Migration cancelled. No changes made.")
        return

    # ── Run migration ─────────────────────────────────────────────────────────
    print(f"\nLoading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    graph_db.init_db()

    total_imported = 0
    for source_name, records in sources_found:
        print(f"\nImporting from {source_name} ({len(records)} memories)...")
        n = _migrate_records(records, model)
        total_imported += n
        print(f"  ✓ {n} memories imported from {source_name}")

    print(f"\n✓ Migration complete. {total_imported} memories imported into mcp-memory-graph.")
    print("  Restart Claude Code to activate the new memory server.\n")


if __name__ == "__main__":
    main()
