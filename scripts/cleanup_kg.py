#!/usr/bin/env python3
"""One-time cleanup script for Knowledge Graph issues.

This script fixes the following issues in Neo4j:
1. Orphaned entities with group_id = "Neznm" (assign legacy_ prefix)
2. Self-loop relationships (delete them)
3. Missing entity types (infer from name patterns)
4. Missing RELATES_TO confidence scores (set placeholder 0.7)
5. Orphaned Episodic nodes without MENTIONS relationships (identify)

Usage:
    uv run python scripts/cleanup_kg.py [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def get_neo4j_driver():
    """Create Neo4j driver from environment variables."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "graphiti123")
    return GraphDatabase.driver(uri, auth=(user, password))


def run_cleanup(dry_run: bool = False):
    """Run all cleanup queries."""
    driver = get_neo4j_driver()

    cleanup_queries = [
        {
            "name": "Fix orphaned group_ids (Neznm)",
            "check": """
                MATCH (e)
                WHERE e.group_id = "Neznm" OR e.group_id IS NULL OR e.group_id = ""
                RETURN count(e) as count
            """,
            "fix": """
                MATCH (e)
                WHERE e.group_id = "Neznm" OR e.group_id IS NULL OR e.group_id = ""
                SET e.group_id = "legacy_" + substring(toString(id(e)), 0, 8)
                RETURN count(e) as fixed_count
            """,
        },
        {
            "name": "Remove self-loop relationships",
            "check": """
                MATCH (e)-[r]->(e)
                RETURN count(r) as count
            """,
            "fix": """
                MATCH (e)-[r]->(e)
                DELETE r
                RETURN count(r) as deleted_count
            """,
        },
        {
            "name": "Infer entity types from name patterns",
            "check": """
                MATCH (e)
                WHERE e.entity_type IS NULL OR e.entity_type = "Entity" OR e.entity_type = ""
                RETURN count(e) as count
            """,
            "fix": """
                MATCH (e)
                WHERE e.entity_type IS NULL OR e.entity_type = "Entity" OR e.entity_type = ""
                WITH e,
                     CASE
                       WHEN toLower(e.name) =~ '.*zákon.*|.*vyhláška.*|.*nařízení.*|.*směrnice.*' THEN 'Dokument'
                       WHEN e.name =~ '.*[0-9]{1,2}\\.[0-9]{1,2}\\.[0-9]{4}.*' THEN 'Datum'
                       WHEN toLower(e.name) =~ '.*kč.*|.*eur.*|.*czk.*' THEN 'Částka'
                       WHEN toLower(e.name) =~ '.*sújb.*|.*čez.*|.*súrao.*|.*ministerstvo.*|.*úřad.*' THEN 'Organizace'
                       WHEN e.name =~ '^[A-ZŽŠČŘĎŤŇÁÉÍÓÚŮÝĚ][a-zžščřďťňáéíóúůýě]+ [A-ZŽŠČŘĎŤŇÁÉÍÓÚŮÝĚ][a-zžščřďťňáéíóúůýě]+$' THEN 'Osoba'
                       ELSE 'Entity'
                     END as inferred_type
                SET e.entity_type = inferred_type
                RETURN count(e) as updated_count
            """,
        },
        {
            "name": "Add placeholder confidence to RELATES_TO",
            "check": """
                MATCH ()-[r:RELATES_TO]-()
                WHERE r.confidence IS NULL
                RETURN count(r) as count
            """,
            "fix": """
                MATCH ()-[r:RELATES_TO]-()
                WHERE r.confidence IS NULL
                SET r.confidence = 0.7
                RETURN count(r) as updated_count
            """,
        },
        {
            "name": "Identify orphaned Episodic nodes (no MENTIONS)",
            "check": """
                MATCH (e:Episodic)
                WHERE NOT (e)<-[:MENTIONS]-()
                RETURN count(e) as count
            """,
            "fix": """
                MATCH (e:Episodic)
                WHERE NOT (e)<-[:MENTIONS]-()
                SET e.orphaned = true
                RETURN count(e) as marked_count
            """,
        },
    ]

    print("=" * 60)
    print("SUJBOT2 Knowledge Graph Cleanup")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (applying changes)'}")
    print()

    with driver.session() as session:
        for query in cleanup_queries:
            print(f"\n--- {query['name']} ---")

            # Check current state
            result = session.run(query["check"])
            record = result.single()
            count = record["count"] if record else 0
            print(f"  Found: {count} items to fix")

            if count == 0:
                print("  Status: Nothing to do")
                continue

            if dry_run:
                print("  Status: Would fix (dry run)")
            else:
                # Apply fix
                result = session.run(query["fix"])
                record = result.single()
                fixed = list(record.values())[0] if record else 0
                print(f"  Status: Fixed {fixed} items")

    driver.close()

    print("\n" + "=" * 60)
    print("Cleanup complete!")
    if dry_run:
        print("Run without --dry-run to apply changes.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup Knowledge Graph issues in Neo4j"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    args = parser.parse_args()

    run_cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
