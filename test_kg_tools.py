"""
Test KG tools with benchmark database.
"""

import sys
import logging
from pathlib import Path

# CRITICAL: Validate config.json before doing anything else
try:
    from src.config import get_config
    _config = get_config()
except (FileNotFoundError, ValueError) as e:
    print(f"\n❌ ERROR: Invalid or missing config.json!")
    print(f"\n{e}")
    print(f"\nPlease create config.json from config.json.example")
    sys.exit(1)

from src.agent.tools.tier2_advanced import GraphSearchTool, BrowseEntitiesTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test KG tools."""
    logger.info("=" * 80)
    logger.info("TESTING KG TOOLS WITH BENCHMARK DATABASE")
    logger.info("=" * 80)

    # Test 1: BrowseEntitiesTool
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: BrowseEntitiesTool")
    logger.info("=" * 60)

    browse_tool = BrowseEntitiesTool(vector_store="benchmark_db")

    logger.info("\nBrowsing ALL entities:")
    result = browse_tool.execute(limit=20)

    if result.success:
        entities = result.data.get("entities", [])
        logger.info(f"  ✓ Found {len(entities)} entities total")
        for i, entity in enumerate(entities, 1):
            logger.info(f"    {i}. {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
    else:
        logger.error(f"  ✗ Failed: {result.error}")

    # Test 2: GraphSearchTool (entity_mentions mode)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: GraphSearchTool (entity_mentions)")
    logger.info("=" * 60)

    graph_tool = GraphSearchTool(vector_store="benchmark_db")

    # Get first entity from browse results
    if entities:
        test_entity = entities[0].get('name', 'unknown')
        logger.info(f"\nSearching mentions for entity: '{test_entity}'")

        result = graph_tool.execute(
            mode="entity_mentions",
            entity_value=test_entity,
            limit=5
        )

        if result.success:
            chunks = result.data.get("chunks", [])
            logger.info(f"  ✓ Found {len(chunks)} chunk mentions")
            for i, chunk in enumerate(chunks[:3], 1):
                logger.info(f"    {i}. {chunk.get('id', 'N/A')}")
        else:
            logger.error(f"  ✗ Failed: {result.error}")

    # Test 3: GraphSearchTool (relationships mode)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: GraphSearchTool (relationships)")
    logger.info("=" * 60)

    if entities:
        test_entity = entities[0].get('name', 'unknown')
        logger.info(f"\nSearching relationships for entity: '{test_entity}'")

        result = graph_tool.execute(
            mode="relationships",
            entity_value=test_entity,
            direction="both",
            limit=10
        )

        if result.success:
            relationships = result.data.get("relationships", [])
            logger.info(f"  ✓ Found {len(relationships)} relationships")
            for i, rel in enumerate(relationships[:3], 1):
                source = rel.get('source', {}).get('name', 'N/A')
                target = rel.get('target', {}).get('name', 'N/A')
                rel_type = rel.get('type', 'N/A')
                logger.info(f"    {i}. {source} --[{rel_type}]--> {target}")
        else:
            logger.error(f"  ✗ Failed: {result.error}")

    logger.info("\n" + "=" * 80)
    logger.info("KG TOOLS TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
