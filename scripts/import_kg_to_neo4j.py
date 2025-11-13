#!/usr/bin/env python3
"""
Import unified_kg.json into Neo4j database.

Usage:
    python scripts/import_kg_to_neo4j.py --kg-file vector_db/unified_kg.json

Or from Docker:
    docker exec sujbot_backend python scripts/import_kg_to_neo4j.py --kg-file /import/unified_kg.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

try:
    from neo4j import GraphDatabase
    import neo4j.exceptions
except ImportError:
    print("ERROR: neo4j Python driver not installed")
    print("Install with: pip install neo4j")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Neo4jKGImporter:
    """Import knowledge graph from JSON to Neo4j."""

    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships (WARNING: destructive!)."""
        with self.driver.session() as session:
            logger.warning("Clearing all existing nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")

    def create_indexes(self):
        """Create indexes for faster queries."""
        with self.driver.session() as session:
            # Index on entity ID
            session.run("CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            # Index on entity type
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            # Index on entity name
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            logger.info("Indexes created")

    def import_entities(self, entities: List[Dict[str, Any]], batch_size: int = 500):
        """Import entities in batches."""
        total = len(entities)
        logger.info(f"Importing {total} entities...")

        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = entities[i:i+batch_size]

                # Create entities (all with Entity label, type stored as property)
                # Handle both old format (name, definition, examples) and new format (value, normalized_value)
                # Note: metadata must be serialized to JSON string (Neo4j doesn't support nested maps)
                import json as json_module
                for entity in batch:
                    if "metadata" in entity and entity["metadata"]:
                        entity["metadata_json"] = json_module.dumps(entity["metadata"])
                        del entity["metadata"]

                query = """
                UNWIND $entities AS entity
                CREATE (n:Entity)
                SET n.id = entity.id,
                    n.type = entity.type,
                    n.value = COALESCE(entity.value, entity.name),
                    n.normalized_value = COALESCE(entity.normalized_value, entity.name),
                    n.confidence = entity.confidence,
                    n.source_chunk_ids = COALESCE(entity.source_chunk_ids, entity.chunk_ids, []),
                    n.document_id = entity.document_id,
                    n.first_mention_chunk_id = entity.first_mention_chunk_id,
                    n.extraction_method = entity.extraction_method,
                    n.metadata_json = entity.metadata_json
                RETURN count(n) AS created
                """

                result = session.run(query, entities=batch)
                count = result.single()["created"]
                logger.info(f"Imported {i + count}/{total} entities")

        logger.info(f"✓ Imported all {total} entities")

    def import_relationships(self, relationships: List[Dict[str, Any]], batch_size: int = 500):
        """Import relationships in batches (using APOC for dynamic relationship types)."""
        total = len(relationships)
        logger.info(f"Importing {total} relationships...")

        # Group relationships by type for efficient batch processing
        from collections import defaultdict
        rels_by_type = defaultdict(list)
        for rel in relationships:
            # Handle both formats: 'type' or 'relationship_type'
            rel_type = rel.get("type") or rel.get("relationship_type", "RELATED_TO")
            rels_by_type[rel_type].append(rel)

        with self.driver.session() as session:
            total_imported = 0
            for rel_type, rels in rels_by_type.items():
                # Sanitize relationship type (Neo4j relationship types can't have spaces or special chars)
                safe_rel_type = rel_type.upper().replace(" ", "_").replace("-", "_")

                for i in range(0, len(rels), batch_size):
                    batch = rels[i:i+batch_size]

                    # Create relationships with static type per batch
                    # Handle both formats: source_entity_id/target_entity_id or source_id/target_id
                    query = f"""
                    UNWIND $relationships AS rel
                    MATCH (source:Entity {{id: COALESCE(rel.source_entity_id, rel.source_id)}})
                    MATCH (target:Entity {{id: COALESCE(rel.target_entity_id, rel.target_id)}})
                    CREATE (source)-[r:{safe_rel_type}]->(target)
                    SET r.confidence = rel.confidence,
                        r.evidence = COALESCE(rel.evidence_text, rel.evidence),
                        r.chunk_id = COALESCE(rel.source_chunk_id, rel.chunk_id),
                        r.original_type = COALESCE(rel.type, rel.relationship_type),
                        r.extraction_method = rel.extraction_method
                    RETURN count(r) AS created
                    """

                    result = session.run(query, relationships=batch)
                    count = result.single()["created"]
                    total_imported += count
                    logger.info(f"Imported {total_imported}/{total} relationships ({safe_rel_type})")

        logger.info(f"✓ Imported all {total} relationships")

    def verify_import(self) -> Dict[str, int]:
        """Verify import by counting nodes and relationships."""
        with self.driver.session() as session:
            # Count entities
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]

            # Count relationships
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

            # Get entity type distribution
            type_dist = session.run("""
                MATCH (e:Entity)
                RETURN e.type AS type, count(*) AS count
                ORDER BY count DESC
            """).data()

            logger.info(f"Verification results:")
            logger.info(f"  Entities: {entity_count}")
            logger.info(f"  Relationships: {rel_count}")
            logger.info(f"  Top entity types:")
            for row in type_dist[:10]:
                logger.info(f"    - {row['type']}: {row['count']}")

            return {
                "entity_count": entity_count,
                "relationship_count": rel_count,
                "type_distribution": type_dist
            }


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Import knowledge graph to Neo4j")
    parser.add_argument(
        "--kg-file",
        type=str,
        default="vector_db/unified_kg.json",
        help="Path to unified_kg.json file"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear database before import (WARNING: destructive!)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for import"
    )

    args = parser.parse_args()

    # Validate inputs
    kg_file = Path(args.kg_file)
    if not kg_file.exists():
        logger.error(f"Knowledge graph file not found: {kg_file}")
        sys.exit(1)

    if not args.password:
        logger.error("Neo4j password required (set NEO4J_PASSWORD env var or use --password)")
        sys.exit(1)

    # Load knowledge graph
    logger.info(f"Loading knowledge graph from {kg_file}...")
    with open(kg_file, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)

    entities = kg_data.get("entities", [])
    relationships = kg_data.get("relationships", [])
    stats = kg_data.get("stats", {})

    logger.info(f"Loaded knowledge graph:")
    logger.info(f"  Entities: {len(entities)}")
    logger.info(f"  Relationships: {len(relationships)}")
    logger.info(f"  Created: {kg_data.get('created_at', 'unknown')}")

    # Import to Neo4j
    importer = Neo4jKGImporter(args.uri, args.username, args.password)

    try:
        if args.clear:
            importer.clear_database()

        logger.info("Creating indexes...")
        importer.create_indexes()

        logger.info("Importing entities...")
        importer.import_entities(entities, batch_size=args.batch_size)

        logger.info("Importing relationships...")
        importer.import_relationships(relationships, batch_size=args.batch_size)

        # Verify
        logger.info("\n" + "="*60)
        logger.info("Verifying import...")
        logger.info("="*60)
        verification = importer.verify_import()

        # Check for data loss
        expected_entities = len(entities)
        actual_entities = verification["entity_count"]
        expected_rels = len(relationships)
        actual_rels = verification["relationship_count"]

        data_loss_detected = False

        if actual_entities != expected_entities:
            entity_loss = expected_entities - actual_entities
            logger.error(f"CRITICAL: Entity count mismatch! Expected {expected_entities}, got {actual_entities}")
            logger.error(f"Data loss: {entity_loss} entities not imported")
            data_loss_detected = True

        if actual_rels != expected_rels:
            rel_loss = expected_rels - actual_rels
            logger.error(f"CRITICAL: Relationship count mismatch! Expected {expected_rels}, got {actual_rels}")
            logger.error(f"Data loss: {rel_loss} relationships not imported")
            data_loss_detected = True

        if data_loss_detected:
            logger.error("\n✗ Import FAILED with data loss!")
            logger.error("Database is in INCONSISTENT state - manual investigation required")
            sys.exit(1)
        else:
            logger.info("\n✓ Import successful! All entities and relationships imported.")
            sys.exit(0)

    except neo4j.exceptions.ServiceUnavailable as e:
        logger.error(f"Neo4j database unavailable: {e}")
        logger.error("Check that Neo4j is running and credentials are correct")
        sys.exit(1)
    except neo4j.exceptions.AuthError as e:
        logger.error(f"Neo4j authentication failed: {e}")
        logger.error("Verify NEO4J_USERNAME and NEO4J_PASSWORD")
        sys.exit(1)
    except neo4j.exceptions.CypherSyntaxError as e:
        logger.error(f"Cypher query syntax error: {e}")
        logger.error("This is a bug in the import script - report to developers")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Import failed with unexpected error: {e}", exc_info=True)
        logger.error("Database may be in inconsistent state - review logs and consider rollback")
        sys.exit(1)
    finally:
        importer.close()


if __name__ == "__main__":
    main()
