#!/usr/bin/env python3
"""
Document Reindexing Script with Delete-Then-Insert Logic

This script:
1. Deletes existing document data from PostgreSQL (all 3 vector layers + metadata)
2. Runs the indexing pipeline for fresh extraction
3. Validates the results (page numbering, chunk counts, etc.)

Usage:
    # Reindex single document
    uv run python scripts/reindex_document.py data/Sb_2025_157_PZZ.pdf

    # Reindex all documents in directory
    uv run python scripts/reindex_document.py data/

    # Dry run (show what would be deleted, don't actually delete)
    uv run python scripts/reindex_document.py data/Sb_2025_157_PZZ.pdf --dry-run

    # Skip validation (faster, less verbose)
    uv run python scripts/reindex_document.py data/Sb_2025_157_PZZ.pdf --skip-validation
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def get_document_stats(conn: asyncpg.Connection, document_id: str) -> dict:
    """Get statistics for a document in PostgreSQL."""
    stats = {}

    for layer in [1, 2, 3]:
        count = await conn.fetchval(
            f"SELECT COUNT(*) FROM vectors.layer{layer} WHERE document_id = $1",
            document_id
        )
        stats[f"layer{layer}_count"] = count

    # Get page number range for layer3
    page_range = await conn.fetchrow(
        """
        SELECT MIN(page_number) as min_page, MAX(page_number) as max_page
        FROM vectors.layer3
        WHERE document_id = $1 AND page_number IS NOT NULL
        """,
        document_id
    )
    if page_range:
        stats["min_page"] = page_range["min_page"]
        stats["max_page"] = page_range["max_page"]

    return stats


async def delete_document(
    conn: asyncpg.Connection,
    document_id: str,
    dry_run: bool = False
) -> dict:
    """
    Delete all data for a document from PostgreSQL.

    Deletes from:
    - vectors.layer1, layer2, layer3
    - graphs.entities, graphs.relationships (if source_document_id matches)
    - metadata.documents

    Returns dict with deletion counts.
    """
    deleted = {}

    if dry_run:
        logger.info(f"[DRY RUN] Would delete document: {document_id}")

    # Delete from vector layers (order matters for foreign keys)
    for layer in [3, 2, 1]:
        if dry_run:
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM vectors.layer{layer} WHERE document_id = $1",
                document_id
            )
            deleted[f"layer{layer}"] = count
            logger.info(f"[DRY RUN] Would delete {count} rows from vectors.layer{layer}")
        else:
            result = await conn.execute(
                f"DELETE FROM vectors.layer{layer} WHERE document_id = $1",
                document_id
            )
            count = int(result.split()[-1])
            deleted[f"layer{layer}"] = count
            logger.info(f"Deleted {count} rows from vectors.layer{layer}")

    # Delete from graphs (entities and relationships)
    for table in ["relationships", "entities"]:
        if dry_run:
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM graphs.{table} WHERE source_document_id = $1",
                document_id
            )
            deleted[f"graphs_{table}"] = count
            if count > 0:
                logger.info(f"[DRY RUN] Would delete {count} rows from graphs.{table}")
        else:
            result = await conn.execute(
                f"DELETE FROM graphs.{table} WHERE source_document_id = $1",
                document_id
            )
            count = int(result.split()[-1])
            deleted[f"graphs_{table}"] = count
            if count > 0:
                logger.info(f"Deleted {count} rows from graphs.{table}")

    # Delete from metadata.documents
    if dry_run:
        exists = await conn.fetchval(
            "SELECT 1 FROM metadata.documents WHERE document_id = $1",
            document_id
        )
        deleted["metadata_documents"] = 1 if exists else 0
        if exists:
            logger.info(f"[DRY RUN] Would delete from metadata.documents")
    else:
        result = await conn.execute(
            "DELETE FROM metadata.documents WHERE document_id = $1",
            document_id
        )
        count = int(result.split()[-1])
        deleted["metadata_documents"] = count
        if count > 0:
            logger.info(f"Deleted from metadata.documents")

    # Update stats
    if not dry_run:
        await conn.execute("SELECT metadata.update_vector_store_stats()")
        logger.info("Updated vector store statistics")

    return deleted


async def find_document_id_for_file(conn: asyncpg.Connection, file_path: Path) -> str | None:
    """
    Try to find document_id in PostgreSQL that matches the file.

    Matching strategies:
    1. Exact match on file stem (e.g., "Sb_2025_157_PZZ" -> "157/2025 Sb.")
    2. Pattern matching on document_id
    """
    file_stem = file_path.stem

    # Get all document IDs
    rows = await conn.fetch("SELECT DISTINCT document_id FROM vectors.layer1")
    doc_ids = [row["document_id"] for row in rows]

    # Strategy 1: Direct patterns for Czech legal documents
    # Sb_2025_157_PZZ.pdf -> 157/2025 Sb.
    # Sb_1997_18_2017-01-01_IZ.pdf -> 18/1997 Sb.
    if file_stem.startswith("Sb_"):
        parts = file_stem.split("_")
        if len(parts) >= 3:
            year = parts[1]
            number = parts[2]
            expected_id = f"{number}/{year} Sb."
            if expected_id in doc_ids:
                return expected_id

    # Strategy 2: Direct match (e.g., BZ_VR1)
    if file_stem in doc_ids:
        return file_stem

    # Strategy 3: Check if file_stem is contained in any document_id
    for doc_id in doc_ids:
        if file_stem.lower() in doc_id.lower() or doc_id.lower() in file_stem.lower():
            return doc_id

    return None


async def validate_indexing(
    conn: asyncpg.Connection,
    document_id: str,
    output_dir: Path
) -> dict:
    """
    Validate indexed document:
    - Check chunk counts
    - Verify page numbering
    - Compare with phase files
    """
    validation = {"passed": True, "issues": []}

    # Get database stats
    db_stats = await get_document_stats(conn, document_id)

    # Load phase3 for comparison
    phase3_path = output_dir / "phase3_chunks.json"
    if phase3_path.exists():
        with open(phase3_path, "r", encoding="utf-8") as f:
            phase3 = json.load(f)

        expected_l3 = len(phase3.get("layer3", []))
        actual_l3 = db_stats.get("layer3_count", 0)

        if expected_l3 != actual_l3:
            validation["passed"] = False
            validation["issues"].append(
                f"Layer3 count mismatch: expected {expected_l3}, got {actual_l3}"
            )
        else:
            logger.info(f"Layer3 count: {actual_l3} chunks (matches phase3)")

        # Check page numbering
        phase3_pages = [
            c.get("metadata", {}).get("page_number")
            for c in phase3.get("layer3", [])
            if c.get("metadata", {}).get("page_number") is not None
        ]
        if phase3_pages:
            expected_max = max(phase3_pages)
            actual_max = db_stats.get("max_page", 0)

            if expected_max != actual_max:
                validation["issues"].append(
                    f"Max page mismatch: phase3={expected_max}, db={actual_max}"
                )
            else:
                logger.info(f"Page range: {db_stats.get('min_page', 0)}-{actual_max}")

    # Check for null page numbers
    null_pages = await conn.fetchval(
        """
        SELECT COUNT(*) FROM vectors.layer3
        WHERE document_id = $1 AND page_number IS NULL
        """,
        document_id
    )
    if null_pages > 0:
        validation["issues"].append(f"{null_pages} chunks have NULL page_number")

    validation["db_stats"] = db_stats

    if validation["issues"]:
        logger.warning(f"Validation issues found:")
        for issue in validation["issues"]:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Validation passed!")

    return validation


def run_indexing_pipeline(file_path: Path, output_dir: Path) -> bool:
    """Run the indexing pipeline for a document."""
    from src.indexing_pipeline import IndexingPipeline, IndexingConfig
    from src.cost_tracker import reset_global_tracker

    reset_global_tracker()

    config = IndexingConfig.from_env()
    pipeline = IndexingPipeline(config)

    try:
        result = pipeline.index_document(
            document_path=file_path,
            save_intermediate=True,
            output_dir=output_dir,
            resume=False  # Force fresh extraction
        )

        if result is None:
            logger.warning("Document marked as duplicate - was skipped")
            return False

        return True

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        return False


async def reindex_document(
    file_path: Path,
    dry_run: bool = False,
    skip_validation: bool = False
) -> bool:
    """
    Reindex a single document with delete-then-insert logic.

    Steps:
    1. Find existing document_id in database
    2. Delete all existing data for that document
    3. Run indexing pipeline
    4. Validate results
    """
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        logger.error("DATABASE_URL environment variable not set")
        return False

    logger.info("=" * 80)
    logger.info(f"REINDEXING: {file_path.name}")
    logger.info("=" * 80)

    # Connect to database
    conn = await asyncpg.connect(connection_string)

    try:
        # Step 1: Find existing document_id
        doc_id = await find_document_id_for_file(conn, file_path)

        if doc_id:
            logger.info(f"Found existing document: {doc_id}")

            # Get current stats
            stats = await get_document_stats(conn, doc_id)
            logger.info(
                f"Current data: L1={stats.get('layer1_count', 0)}, "
                f"L2={stats.get('layer2_count', 0)}, "
                f"L3={stats.get('layer3_count', 0)}, "
                f"pages={stats.get('min_page', '?')}-{stats.get('max_page', '?')}"
            )

            # Step 2: Delete existing data
            logger.info("")
            logger.info("Deleting existing data...")
            deleted = await delete_document(conn, doc_id, dry_run=dry_run)

            if dry_run:
                logger.info("[DRY RUN] Would delete and reindex. Stopping here.")
                return True
        else:
            logger.info("No existing document found in database - fresh indexing")
            if dry_run:
                logger.info("[DRY RUN] Nothing to delete. Stopping here.")
                return True

    finally:
        await conn.close()

    # Step 3: Run indexing pipeline
    logger.info("")
    logger.info("Starting indexing pipeline...")

    # Determine output directory
    doc_name = file_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    output_dir = Path("output") / doc_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing phase files to force fresh extraction
    for phase_file in output_dir.glob("phase*.json"):
        phase_file.unlink()
        logger.info(f"Removed cached: {phase_file.name}")

    success = run_indexing_pipeline(file_path, output_dir)

    if not success:
        logger.error("Indexing pipeline failed!")
        return False

    # Step 4: Validate
    if not skip_validation:
        logger.info("")
        logger.info("Validating indexed document...")

        conn = await asyncpg.connect(connection_string)
        try:
            # Re-find document_id (may have changed)
            new_doc_id = await find_document_id_for_file(conn, file_path)
            if new_doc_id:
                validation = await validate_indexing(conn, new_doc_id, output_dir)

                if not validation["passed"]:
                    logger.warning("Validation found issues - review above")
            else:
                logger.warning("Could not find document after indexing!")
        finally:
            await conn.close()

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"REINDEXING COMPLETE: {file_path.name}")
    logger.info("=" * 80)

    return True


async def main():
    parser = argparse.ArgumentParser(
        description="Reindex documents with delete-then-insert logic"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to document file or directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually doing it"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-indexing validation"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    # Get list of files to process
    if path.is_dir():
        supported = [".pdf", ".docx", ".pptx", ".xlsx", ".html"]
        files = [f for f in path.iterdir() if f.suffix.lower() in supported]
        files.sort()

        if not files:
            logger.error(f"No supported documents found in {path}")
            sys.exit(1)

        logger.info(f"Found {len(files)} documents to reindex:")
        for f in files:
            logger.info(f"  - {f.name}")
        logger.info("")
    else:
        files = [path]

    # Process each file
    success_count = 0
    for file_path in files:
        try:
            success = await reindex_document(
                file_path,
                dry_run=args.dry_run,
                skip_validation=args.skip_validation
            )
            if success:
                success_count += 1
        except Exception as e:
            logger.error(f"Failed to reindex {file_path.name}: {e}", exc_info=True)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"REINDEXING SUMMARY: {success_count}/{len(files)} documents successful")
    logger.info("=" * 80)

    if success_count < len(files):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
