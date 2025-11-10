#!/usr/bin/env python
"""
Migrate Knowledge Graph from JSON to Neo4j Aura.

This script loads a KnowledgeGraph from JSON file and uploads it to Neo4j
database with batch processing and progress tracking.

Usage:
    # Dry-run (analyze without uploading)
    uv run python scripts/migrate_kg_to_neo4j.py --kg-file vector_db/unified_kg.json --dry-run

    # Actual migration
    uv run python scripts/migrate_kg_to_neo4j.py --kg-file vector_db/unified_kg.json

    # Custom batch size
    uv run python scripts/migrate_kg_to_neo4j.py --kg-file vector_db/unified_kg.json --batch-size 500
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.models import KnowledgeGraph
from src.graph.config import Neo4jConfig, GraphStorageConfig, GraphBackend
from src.graph.graph_builder import create_graph_builder
from src.graph.health_check import check_neo4j_health

logger = logging.getLogger(__name__)


def analyze_kg(kg: KnowledgeGraph) -> None:
    """Print analysis of knowledge graph."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH ANALYSIS")
    print("=" * 60)

    print(f"\nEntities: {len(kg.entities)}")
    print(f"Relationships: {len(kg.relationships)}")

    # Entity type breakdown
    from collections import Counter
    entity_types = Counter(e.type.value for e in kg.entities)
    print("\nEntity Types:")
    for entity_type, count in entity_types.most_common():
        print(f"  {entity_type:20s}: {count:5d}")

    # Relationship type breakdown
    rel_types = Counter(r.type.value for r in kg.relationships)
    print("\nRelationship Types:")
    for rel_type, count in rel_types.most_common():
        print(f"  {rel_type:20s}: {count:5d}")

    print("\n" + "=" * 60)


def migrate_to_neo4j(
    kg_file: str,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> None:
    """
    Migrate knowledge graph from JSON to Neo4j.

    Args:
        kg_file: Path to KnowledgeGraph JSON file
        batch_size: Entities per batch (default: 1000)
        dry_run: If True, only analyze without uploading
    """
    # Load KnowledgeGraph from JSON
    logger.info(f"Loading knowledge graph from {kg_file}...")

    if not Path(kg_file).exists():
        logger.error(f"File not found: {kg_file}")
        sys.exit(1)

    try:
        kg = KnowledgeGraph.load_json(kg_file)
    except Exception as e:
        logger.error(f"Failed to load knowledge graph: {e}")
        sys.exit(1)

    logger.info(
        f"Loaded {len(kg.entities)} entities and {len(kg.relationships)} relationships"
    )

    # Analyze
    analyze_kg(kg)

    if dry_run:
        print("\n✓ DRY-RUN MODE: No changes will be made to Neo4j")
        print(f"\nWould migrate:")
        print(f"  - {len(kg.entities)} entities ({len(kg.entities)//batch_size + 1} batches)")
        print(f"  - {len(kg.relationships)} relationships")
        return

    # Load Neo4j config from environment
    logger.info("Loading Neo4j configuration from environment...")
    neo4j_config = Neo4jConfig.from_env()

    # Health check
    print("\nChecking Neo4j health...")
    health = check_neo4j_health(neo4j_config)

    if not health.get("connected", False):
        logger.error(f"Neo4j health check failed: {health.get('error')}")
        logger.error("Please check your NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in .env")
        sys.exit(1)

    print(f"✓ Neo4j connected ({health['response_time_ms']:.0f}ms)")

    # Create graph builder
    storage_config = GraphStorageConfig(
        backend=GraphBackend.NEO4J,
        neo4j_config=neo4j_config,
    )

    builder = create_graph_builder(storage_config)

    try:
        # Migrate entities
        print(f"\nMigrating {len(kg.entities)} entities...")
        start_time = time.time()

        # Use tqdm for progress if available, fallback to simple logging
        try:
            from tqdm import tqdm

            # Process in batches with progress bar
            batch_count = len(kg.entities) // batch_size + 1
            with tqdm(total=len(kg.entities), desc="Entities", unit="entity") as pbar:
                for i in range(0, len(kg.entities), batch_size):
                    batch = kg.entities[i : i + batch_size]
                    builder.add_entities(batch)
                    pbar.update(len(batch))

        except ImportError:
            # Fallback without progress bar
            for i in range(0, len(kg.entities), batch_size):
                batch = kg.entities[i : i + batch_size]
                builder.add_entities(batch)
                print(f"  Progress: {min(i + batch_size, len(kg.entities))}/{len(kg.entities)} entities")

        entity_duration = time.time() - start_time
        print(f"✓ Entities migrated in {entity_duration:.1f}s")

        # Migrate relationships
        print(f"\nMigrating {len(kg.relationships)} relationships...")
        start_time = time.time()

        try:
            from tqdm import tqdm

            rel_batch_size = 500  # Smaller batch for relationships
            with tqdm(total=len(kg.relationships), desc="Relationships", unit="rel") as pbar:
                for i in range(0, len(kg.relationships), rel_batch_size):
                    batch = kg.relationships[i : i + rel_batch_size]
                    builder.add_relationships(batch)
                    pbar.update(len(batch))

        except ImportError:
            # Fallback without progress bar
            rel_batch_size = 500
            for i in range(0, len(kg.relationships), rel_batch_size):
                batch = kg.relationships[i : i + rel_batch_size]
                builder.add_relationships(batch)
                print(
                    f"  Progress: {min(i + rel_batch_size, len(kg.relationships))}/{len(kg.relationships)} relationships"
                )

        rel_duration = time.time() - start_time
        print(f"✓ Relationships migrated in {rel_duration:.1f}s")

        # Verify migration
        print("\nVerifying migration...")
        exported = builder.export_to_knowledge_graph()

        entities_match = len(exported.entities) == len(kg.entities)
        rels_match = len(exported.relationships) == len(kg.relationships)

        print(f"\nVerification:")
        print(f"  Source entities:      {len(kg.entities)}")
        print(f"  Neo4j entities:       {len(exported.entities)} {'✓' if entities_match else '✗'}")
        print(f"  Source relationships: {len(kg.relationships)}")
        print(f"  Neo4j relationships:  {len(exported.relationships)} {'✓' if rels_match else '✗'}")

        if entities_match and rels_match:
            total_duration = entity_duration + rel_duration
            print(f"\n✓ Migration completed successfully in {total_duration:.1f}s")
            print(f"  Throughput: {len(kg.entities)/entity_duration:.0f} entities/sec")
        else:
            logger.warning("Count mismatch detected - some data may not have migrated")

    finally:
        builder.close()


def main():
    """Parse arguments and run migration."""
    parser = argparse.ArgumentParser(
        description="Migrate Knowledge Graph from JSON to Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run to preview migration
  uv run python scripts/migrate_kg_to_neo4j.py --kg-file vector_db/unified_kg.json --dry-run

  # Actual migration
  uv run python scripts/migrate_kg_to_neo4j.py --kg-file vector_db/unified_kg.json

  # Custom batch size (smaller if getting timeouts)
  uv run python scripts/migrate_kg_to_neo4j.py --kg-file vector_db/unified_kg.json --batch-size 500

Environment variables required:
  NEO4J_URI         - Neo4j connection URI (from .env)
  NEO4J_USERNAME    - Neo4j username (from .env)
  NEO4J_PASSWORD    - Neo4j password (from .env)
  NEO4J_DATABASE    - Neo4j database name (from .env)
        """,
    )

    parser.add_argument(
        "--kg-file",
        required=True,
        help="Path to KnowledgeGraph JSON file (e.g., vector_db/unified_kg.json)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of entities per batch (default: 1000)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze knowledge graph without uploading to Neo4j",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        migrate_to_neo4j(
            kg_file=args.kg_file,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
