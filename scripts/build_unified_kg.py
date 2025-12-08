#!/usr/bin/env python3
"""
Build Unified Knowledge Graph.

Utility script to create unified_kg.json from existing per-document KGs in vector_db/.

Features:
- Entity deduplication across documents
- Cross-document relationship detection
- Document tracking per entity

Usage:
    python build_unified_kg.py
    python build_unified_kg.py --storage-dir vector_db --enable-cross-doc
"""

import sys
import argparse
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

from src.graph import KnowledgeGraphPipeline, KnowledgeGraphConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build unified knowledge graph")
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="vector_db",
        help="Directory with per-document KG files (default: vector_db)",
    )
    parser.add_argument(
        "--enable-cross-doc",
        action="store_true",
        help="Enable cross-document relationship detection (default: True)",
    )
    parser.add_argument(
        "--disable-cross-doc",
        action="store_true",
        help="Disable cross-document relationship detection",
    )

    args = parser.parse_args()

    # Determine cross-doc setting
    enable_cross_doc = not args.disable_cross_doc
    if args.enable_cross_doc:
        enable_cross_doc = True

    # Load configuration
    kg_config = KnowledgeGraphConfig.from_env()

    # Build unified KG
    with KnowledgeGraphPipeline(kg_config) as pipeline:
        unified_kg = pipeline.build_unified_kg(
            storage_dir=args.storage_dir,
            enable_cross_document_relationships=enable_cross_doc,
        )

    print("\n" + "=" * 60)
    print("✅ UNIFIED KNOWLEDGE GRAPH BUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"Output: {Path(args.storage_dir) / 'unified_kg.json'}")
    print(f"Entities: {len(unified_kg.entities)}")
    print(f"Relationships: {len(unified_kg.relationships)}")
    print("\nYou can now run the agent CLI to use the unified graph:")
    print(f"  python -m src.agent.cli --vector-store {args.storage_dir}")


if __name__ == "__main__":
    main()
