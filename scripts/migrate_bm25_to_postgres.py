#!/usr/bin/env python3
"""
Migrate BM25 data from FAISS vector_db to PostgreSQL database.

This script:
1. Loads BM25 indexes from FAISS (vector_db/bm25_layer*.pkl)
2. Connects to PostgreSQL database
3. Inserts BM25 data into PostgreSQL tables
4. Verifies data integrity
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_search_multilang import MultiLangBM25Store


async def migrate_bm25_to_postgres():
    """Main migration function."""
    print("="*80)
    print("BM25 MIGRATION: FAISS → PostgreSQL")
    print("="*80)

    # Load environment variables
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set in .env file!")
        return False

    # Step 1: Load BM25 from FAISS
    print("\n[1/4] Loading BM25 data from FAISS...")
    vector_db_path = Path("vector_db")
    if not vector_db_path.exists():
        print(f"ERROR: {vector_db_path} not found!")
        return False

    try:
        bm25_store = MultiLangBM25Store.load(vector_db_path)
        print(f"✓ Loaded BM25 store")
        print(f"  - Languages: {bm25_store.languages}")
        print(f"  - Layer 1: {len(bm25_store.index_layer1.corpus)} documents")
        print(f"  - Layer 2: {len(bm25_store.index_layer2.corpus)} sections")
        print(f"  - Layer 3: {len(bm25_store.index_layer3.corpus)} chunks")
    except Exception as e:
        print(f"ERROR loading BM25: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Connect to PostgreSQL
    print("\n[2/4] Connecting to PostgreSQL...")
    try:
        conn = await asyncpg.connect(database_url)
        print("✓ Connected to PostgreSQL")
    except Exception as e:
        print(f"ERROR connecting to PostgreSQL: {e}")
        return False

    try:
        # Step 3: Clear existing BM25 data
        print("\n[3/4] Clearing existing BM25 data...")
        await conn.execute("TRUNCATE bm25_layer1, bm25_layer2, bm25_layer3, bm25_config RESTART IDENTITY")
        print("✓ Cleared old data")

        # Step 4: Insert BM25 configuration
        print("\n[4/4] Migrating BM25 data...")
        await conn.execute("""
            INSERT INTO bm25_config (languages, primary_language, format_version)
            VALUES ($1, $2, $3)
        """, bm25_store.languages, bm25_store.lang, "3.0")
        print("✓ Inserted config")

        # Insert Layer 1
        print("  Migrating Layer 1...")
        for corpus_text, chunk_id, metadata in zip(
            bm25_store.index_layer1.corpus,
            bm25_store.index_layer1.chunk_ids,
            bm25_store.index_layer1.metadata
        ):
            await conn.execute("""
                INSERT INTO bm25_layer1 (chunk_id, document_id, corpus, metadata)
                VALUES ($1, $2, $3, $4)
            """, chunk_id, metadata.get("document_id", ""), corpus_text, json.dumps(metadata))

        print(f"  ✓ Migrated {len(bm25_store.index_layer1.corpus)} Layer 1 documents")

        # Insert Layer 2
        print("  Migrating Layer 2...")
        for corpus_text, chunk_id, metadata in zip(
            bm25_store.index_layer2.corpus,
            bm25_store.index_layer2.chunk_ids,
            bm25_store.index_layer2.metadata
        ):
            await conn.execute("""
                INSERT INTO bm25_layer2 (chunk_id, document_id, section_id, corpus, metadata)
                VALUES ($1, $2, $3, $4, $5)
            """, chunk_id, metadata.get("document_id", ""), metadata.get("section_id"), corpus_text, json.dumps(metadata))

        print(f"  ✓ Migrated {len(bm25_store.index_layer2.corpus)} Layer 2 sections")

        # Insert Layer 3 (with progress)
        print("  Migrating Layer 3...")
        total = len(bm25_store.index_layer3.corpus)
        for i, (corpus_text, chunk_id, metadata) in enumerate(zip(
            bm25_store.index_layer3.corpus,
            bm25_store.index_layer3.chunk_ids,
            bm25_store.index_layer3.metadata
        )):
            await conn.execute("""
                INSERT INTO bm25_layer3 (chunk_id, document_id, section_id, corpus, metadata)
                VALUES ($1, $2, $3, $4, $5)
            """, chunk_id, metadata.get("document_id", ""), metadata.get("section_id"), corpus_text, json.dumps(metadata))

            if (i + 1) % 1000 == 0:
                print(f"    Progress: {i+1}/{total} chunks...")

        print(f"  ✓ Migrated {total} Layer 3 chunks")

        # Verify counts
        print("\n[VERIFICATION] Checking data integrity...")
        count_l1 = await conn.fetchval("SELECT COUNT(*) FROM bm25_layer1")
        count_l2 = await conn.fetchval("SELECT COUNT(*) FROM bm25_layer2")
        count_l3 = await conn.fetchval("SELECT COUNT(*) FROM bm25_layer3")

        print(f"  Layer 1: {count_l1} documents (expected: {len(bm25_store.index_layer1.corpus)})")
        print(f"  Layer 2: {count_l2} sections (expected: {len(bm25_store.index_layer2.corpus)})")
        print(f"  Layer 3: {count_l3} chunks (expected: {len(bm25_store.index_layer3.corpus)})")

        if (count_l1 == len(bm25_store.index_layer1.corpus) and
            count_l2 == len(bm25_store.index_layer2.corpus) and
            count_l3 == len(bm25_store.index_layer3.corpus)):
            print("\n" + "="*80)
            print("✓ MIGRATION SUCCESSFUL! All data verified.")
            print("="*80)
            return True
        else:
            print("\n✗ Migration count mismatch!")
            return False

    finally:
        await conn.close()


if __name__ == "__main__":
    success = asyncio.run(migrate_bm25_to_postgres())
    sys.exit(0 if success else 1)
