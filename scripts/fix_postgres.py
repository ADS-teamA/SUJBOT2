#!/usr/bin/env python3
"""Fix PostgreSQL data quality issues.

This script addresses the following problems:
1. Empty metadata.documents table - populate from vector layers
2. Empty vector_store_stats table - populate with current counts
3. NULL content_tsv fields - generate for full-text search
4. Short chunks (<10 chars) - identify and optionally remove

Usage:
    uv run python scripts/fix_postgres.py [--dry-run] [--cleanup-short]
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import asyncpg

load_dotenv()


async def get_connection():
    """Create database connection."""
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL environment variable required")
    return await asyncpg.connect(connection_string)


async def check_current_state(conn) -> dict:
    """Check current database state."""
    state = {}

    # Count vectors in each layer
    for layer in [1, 2, 3]:
        count = await conn.fetchval(f"SELECT COUNT(*) FROM vectors.layer{layer}")
        state[f"layer{layer}_count"] = count

    # Count documents
    state["documents_count"] = await conn.fetchval(
        "SELECT COUNT(*) FROM metadata.documents"
    )

    # Count stats entries
    state["stats_count"] = await conn.fetchval(
        "SELECT COUNT(*) FROM metadata.vector_store_stats"
    )

    # Count NULL content_tsv
    for layer in [1, 2, 3]:
        null_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM vectors.layer{layer} WHERE content_tsv IS NULL"
        )
        state[f"layer{layer}_null_tsv"] = null_count

    # Count short chunks
    state["short_chunks"] = await conn.fetchval(
        "SELECT COUNT(*) FROM vectors.layer3 WHERE LENGTH(TRIM(content)) < 10"
    )

    # Get unique document IDs
    docs = await conn.fetch(
        "SELECT DISTINCT document_id FROM vectors.layer2 ORDER BY document_id"
    )
    state["unique_docs"] = [r["document_id"] for r in docs]

    return state


async def populate_metadata_documents(conn, dry_run: bool = False):
    """Populate metadata.documents from vector layers."""
    print("\n--- Populating metadata.documents ---")

    # Get documents from vectors
    docs = await conn.fetch("""
        SELECT DISTINCT
            l2.document_id,
            l1.content as summary,
            COUNT(DISTINCT l2.chunk_id) as section_count,
            COUNT(DISTINCT l3.chunk_id) as chunk_count
        FROM vectors.layer2 l2
        LEFT JOIN vectors.layer1 l1 ON l1.document_id = l2.document_id
        LEFT JOIN vectors.layer3 l3 ON l3.document_id = l2.document_id
        GROUP BY l2.document_id, l1.content
    """)

    print(f"  Found {len(docs)} documents to register")

    if dry_run:
        for doc in docs:
            print(f"    Would insert: {doc['document_id']} "
                  f"({doc['section_count']} sections, {doc['chunk_count']} chunks)")
        return

    inserted = 0
    for doc in docs:
        try:
            await conn.execute("""
                INSERT INTO metadata.documents
                    (document_id, title, hierarchy_depth, total_chunks, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (document_id) DO UPDATE
                SET total_chunks = $4, metadata = $5, indexed_at = NOW()
            """,
                doc["document_id"],
                doc["document_id"],  # title = document_id
                5,  # hierarchy_depth (estimated)
                doc["chunk_count"],
                f'{{"sections": {doc["section_count"]}, "source": "fix_postgres.py"}}'
            )
            inserted += 1
        except Exception as e:
            print(f"    Error inserting {doc['document_id']}: {e}")

    print(f"  Inserted/updated {inserted} documents")


async def populate_vector_store_stats(conn, state: dict, dry_run: bool = False):
    """Populate vector_store_stats table."""
    print("\n--- Populating vector_store_stats ---")

    total_vectors = (
        state["layer1_count"] +
        state["layer2_count"] +
        state["layer3_count"]
    )

    print(f"  Layer 1: {state['layer1_count']} vectors")
    print(f"  Layer 2: {state['layer2_count']} vectors")
    print(f"  Layer 3: {state['layer3_count']} vectors")
    print(f"  Total: {total_vectors} vectors")
    print(f"  Documents: {len(state['unique_docs'])}")

    if dry_run:
        print("  Would insert stats row")
        return

    # Clear existing stats
    await conn.execute("DELETE FROM metadata.vector_store_stats")

    # Insert new stats
    await conn.execute("""
        INSERT INTO metadata.vector_store_stats (
            dimensions, layer1_count, layer2_count, layer3_count,
            total_vectors, document_count
        ) VALUES ($1, $2, $3, $4, $5, $6)
    """,
        4096,  # dimensions
        state["layer1_count"],
        state["layer2_count"],
        state["layer3_count"],
        total_vectors,
        len(state["unique_docs"])
    )

    print("  Stats row inserted")


async def generate_content_tsv(conn, dry_run: bool = False):
    """Generate content_tsv for full-text search."""
    print("\n--- Generating content_tsv (full-text search) ---")

    for layer in [1, 2, 3]:
        null_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM vectors.layer{layer} WHERE content_tsv IS NULL"
        )
        print(f"  Layer {layer}: {null_count} rows need content_tsv")

        if null_count == 0:
            continue

        if dry_run:
            print(f"    Would update {null_count} rows")
            continue

        # Generate tsvector - use 'simple' config for Czech text
        # (Czech config may not be installed)
        result = await conn.execute(f"""
            UPDATE vectors.layer{layer}
            SET content_tsv = to_tsvector('simple', COALESCE(content, ''))
            WHERE content_tsv IS NULL
        """)

        print(f"    Updated rows in layer{layer}")


async def identify_short_chunks(conn, cleanup: bool = False, dry_run: bool = False):
    """Identify and optionally remove short chunks."""
    print("\n--- Short chunks (<10 chars) ---")

    short_chunks = await conn.fetch("""
        SELECT chunk_id, document_id, LENGTH(content) as len,
               LEFT(content, 50) as preview
        FROM vectors.layer3
        WHERE LENGTH(TRIM(content)) < 10
        ORDER BY len, document_id
        LIMIT 20
    """)

    total_short = await conn.fetchval(
        "SELECT COUNT(*) FROM vectors.layer3 WHERE LENGTH(TRIM(content)) < 10"
    )

    print(f"  Found {total_short} short chunks")

    if short_chunks:
        print("  Examples:")
        for chunk in short_chunks[:10]:
            print(f"    {chunk['chunk_id']}: '{chunk['preview']}' ({chunk['len']} chars)")

    if cleanup:
        if dry_run:
            print(f"  Would delete {total_short} short chunks")
        else:
            result = await conn.execute("""
                DELETE FROM vectors.layer3
                WHERE LENGTH(TRIM(content)) < 10
            """)
            print(f"  Deleted {total_short} short chunks")


async def run_fixes(dry_run: bool = False, cleanup_short: bool = False):
    """Run all PostgreSQL fixes."""
    print("=" * 60)
    print("SUJBOT2 PostgreSQL Database Fixes")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (applying changes)'}")

    conn = await get_connection()

    try:
        # Check current state
        print("\n--- Current State ---")
        state = await check_current_state(conn)

        print(f"  Layer 1: {state['layer1_count']} vectors "
              f"({state['layer1_null_tsv']} NULL tsv)")
        print(f"  Layer 2: {state['layer2_count']} vectors "
              f"({state['layer2_null_tsv']} NULL tsv)")
        print(f"  Layer 3: {state['layer3_count']} vectors "
              f"({state['layer3_null_tsv']} NULL tsv)")
        print(f"  Documents registered: {state['documents_count']}")
        print(f"  Stats entries: {state['stats_count']}")
        print(f"  Short chunks (<10 chars): {state['short_chunks']}")
        print(f"  Unique documents: {state['unique_docs']}")

        # Run fixes
        await populate_metadata_documents(conn, dry_run)
        await populate_vector_store_stats(conn, state, dry_run)
        await generate_content_tsv(conn, dry_run)
        await identify_short_chunks(conn, cleanup_short, dry_run)

        print("\n" + "=" * 60)
        print("Fixes complete!")
        if dry_run:
            print("Run without --dry-run to apply changes.")
        print("=" * 60)

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fix PostgreSQL data quality issues"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--cleanup-short",
        action="store_true",
        help="Delete short chunks (<10 chars)"
    )
    args = parser.parse_args()

    asyncio.run(run_fixes(dry_run=args.dry_run, cleanup_short=args.cleanup_short))


if __name__ == "__main__":
    main()
