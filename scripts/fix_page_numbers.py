#!/usr/bin/env python3
"""
Fix page_number in PostgreSQL from phase3_chunks.json

This script updates page_number values in PostgreSQL vectors.layer3
using the correct values from phase3_chunks.json without regenerating
embeddings (which would be expensive).

Usage:
    # Fix single document
    uv run python scripts/fix_page_numbers.py output/BZ_VR1/phase3_chunks.json

    # Fix all documents
    uv run python scripts/fix_page_numbers.py output/

    # Dry run (show what would be updated)
    uv run python scripts/fix_page_numbers.py output/BZ_VR1/phase3_chunks.json --dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fix_page_numbers_for_document(
    conn: asyncpg.Connection,
    phase3_path: Path,
    dry_run: bool = False
) -> dict:
    """
    Update page_number in PostgreSQL from phase3_chunks.json.

    Returns dict with statistics.
    """
    # Load phase3 chunks
    with open(phase3_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract chunk_id -> page_number mapping
    page_map = {}
    for layer_key in ["layer1", "layer2", "layer3"]:
        for chunk in data.get(layer_key, []):
            chunk_id = chunk.get("chunk_id")
            page_number = chunk.get("metadata", {}).get("page_number")
            if chunk_id and page_number is not None:
                page_map[chunk_id] = page_number

    if not page_map:
        logger.warning(f"No chunks with page_number found in {phase3_path}")
        return {"total": 0, "updated": 0, "skipped": 0, "errors": 0}

    logger.info(f"Loaded {len(page_map)} chunks from {phase3_path.name}")

    # Get document_id from first chunk
    first_chunk = data.get("layer3", [{}])[0]
    document_id = first_chunk.get("metadata", {}).get("document_id")

    if not document_id:
        # Fallback: extract from parent directory name
        document_id = phase3_path.parent.name

    logger.info(f"Document ID: {document_id}")

    # Check current state in PostgreSQL
    current_stats = await conn.fetch(
        """
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN page_number = 0 THEN 1 END) as zero_pages,
            COUNT(CASE WHEN page_number IS NULL THEN 1 END) as null_pages
        FROM vectors.layer3
        WHERE document_id = $1
        """,
        document_id
    )

    if current_stats:
        stats = current_stats[0]
        logger.info(
            f"Current PostgreSQL state: "
            f"total={stats['total']}, zero_pages={stats['zero_pages']}, null_pages={stats['null_pages']}"
        )

    # Get mismatched chunks
    mismatches = await conn.fetch(
        """
        SELECT chunk_id, page_number
        FROM vectors.layer3
        WHERE document_id = $1
        """,
        document_id
    )

    # Count and prepare updates
    updates_needed = []
    for row in mismatches:
        chunk_id = row["chunk_id"]
        db_page = row["page_number"]
        json_page = page_map.get(chunk_id)

        if json_page is not None and db_page != json_page:
            updates_needed.append({
                "chunk_id": chunk_id,
                "old_page": db_page,
                "new_page": json_page
            })

    logger.info(f"Found {len(updates_needed)} chunks needing page_number update")

    if dry_run:
        logger.info("[DRY RUN] Would update the following chunks:")
        for update in updates_needed[:10]:
            logger.info(f"  {update['chunk_id']}: {update['old_page']} -> {update['new_page']}")
        if len(updates_needed) > 10:
            logger.info(f"  ... and {len(updates_needed) - 10} more")
        return {
            "total": len(page_map),
            "updated": 0,
            "would_update": len(updates_needed),
            "skipped": len(page_map) - len(updates_needed),
            "errors": 0
        }

    # Perform updates
    updated = 0
    errors = 0

    # Also update layer1 and layer2
    for layer in [1, 2, 3]:
        layer_key = f"layer{layer}"
        layer_chunks = data.get(layer_key, [])

        for chunk in layer_chunks:
            chunk_id = chunk.get("chunk_id")
            page_number = chunk.get("metadata", {}).get("page_number")

            if chunk_id and page_number is not None:
                try:
                    result = await conn.execute(
                        f"""
                        UPDATE vectors.layer{layer}
                        SET page_number = $1
                        WHERE chunk_id = $2 AND (page_number != $1 OR page_number IS NULL)
                        """,
                        page_number,
                        chunk_id
                    )
                    # Parse "UPDATE N" to get count
                    count = int(result.split()[-1])
                    updated += count
                except Exception as e:
                    logger.error(f"Failed to update {chunk_id}: {e}")
                    errors += 1

    logger.info(f"Updated {updated} rows across all layers")

    # Verify fix
    verify_stats = await conn.fetchrow(
        """
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN page_number = 0 THEN 1 END) as zero_pages,
            COUNT(CASE WHEN page_number IS NULL THEN 1 END) as null_pages
        FROM vectors.layer3
        WHERE document_id = $1
        """,
        document_id
    )

    if verify_stats:
        logger.info(
            f"After fix: "
            f"total={verify_stats['total']}, zero_pages={verify_stats['zero_pages']}, null_pages={verify_stats['null_pages']}"
        )

        if verify_stats['zero_pages'] > 0:
            logger.warning(f"Still {verify_stats['zero_pages']} chunks with page_number=0!")

    return {
        "total": len(page_map),
        "updated": updated,
        "skipped": len(page_map) - updated,
        "errors": errors
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Fix page_number in PostgreSQL from phase3_chunks.json"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to phase3_chunks.json file or directory containing document outputs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without actually doing it"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    # Get connection string
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    # Find phase3_chunks.json files
    if path.is_file() and path.name == "phase3_chunks.json":
        phase3_files = [path]
    elif path.is_dir():
        phase3_files = list(path.glob("**/phase3_chunks.json"))
        if not phase3_files:
            logger.error(f"No phase3_chunks.json files found in {path}")
            sys.exit(1)
    else:
        logger.error(f"Expected phase3_chunks.json file or directory, got: {path}")
        sys.exit(1)

    logger.info(f"Found {len(phase3_files)} phase3_chunks.json files to process")

    # Connect to database
    conn = await asyncpg.connect(connection_string)

    try:
        total_stats = {
            "total": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0
        }

        for phase3_path in phase3_files:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"Processing: {phase3_path}")
            logger.info("=" * 60)

            stats = await fix_page_numbers_for_document(
                conn, phase3_path, dry_run=args.dry_run
            )

            for key in total_stats:
                total_stats[key] += stats.get(key, 0)

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total chunks processed: {total_stats['total']}")
        logger.info(f"Updated: {total_stats['updated']}")
        logger.info(f"Skipped (already correct): {total_stats['skipped']}")
        logger.info(f"Errors: {total_stats['errors']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
