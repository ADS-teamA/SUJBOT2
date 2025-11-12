"""
FAISS to PostgreSQL Migration Script

Migrates existing FAISS vector store to PostgreSQL with pgvector.
Preserves all 3 layers, metadata, and embeddings.

Usage:
    python scripts/migrate_faiss_to_postgres.py \
        --faiss-dir vector_db/ \
        --db-url postgresql://user:pass@localhost:5432/sujbot \
        --batch-size 500

Requirements:
    - FAISS vector store in vector_db/
    - PostgreSQL with pgvector extension running
    - Database schema initialized (01-init.sql)
"""

import asyncio
import asyncpg
import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.faiss_vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FAISSToPostgresMigrator:
    """Migrates FAISS vector store to PostgreSQL."""

    def __init__(self, db_url: str, batch_size: int = 500):
        """
        Initialize migrator.

        Args:
            db_url: PostgreSQL connection string
            batch_size: Batch insert size (default: 500)
        """
        self.db_url = db_url
        self.batch_size = batch_size
        self.pool: asyncpg.Pool = None

    async def initialize(self):
        """Create database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=600,  # 10 minutes for large batches
            )
            logger.info("PostgreSQL connection pool created")

            # Verify extensions
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT extname FROM pg_extension WHERE extname='vector'")
                if not result:
                    raise RuntimeError("pgvector extension not found! Run 01-init.sql first.")
            logger.info("pgvector extension verified")

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    async def migrate_layer(
        self, layer: int, index, metadata: list, doc_id_map: dict
    ):
        """
        Migrate single FAISS layer to PostgreSQL.

        Args:
            layer: Layer number (1, 2, or 3)
            index: FAISS index object
            metadata: List of metadata dicts
            doc_id_map: Document ID to indices mapping
        """
        logger.info(f"Starting Layer {layer} migration...")

        # Extract embeddings from FAISS
        n_total = index.ntotal
        if n_total == 0:
            logger.warning(f"Layer {layer} is empty, skipping")
            return

        logger.info(f"Layer {layer}: Extracting {n_total} embeddings from FAISS...")
        embeddings = np.zeros((n_total, index.d), dtype=np.float32)
        for i in tqdm(range(n_total), desc=f"Layer {layer} extraction"):
            embeddings[i] = index.reconstruct(i)

        # Prepare records for batch insert
        logger.info(f"Layer {layer}: Preparing {len(metadata)} records...")
        records = []
        for i, meta in enumerate(metadata):
            # Convert embedding to list for PostgreSQL
            embedding_list = embeddings[i].tolist()

            # Extract metadata fields
            chunk_id = meta["chunk_id"]
            document_id = meta["document_id"]
            content = meta.get("content", "")
            section_id = meta.get("section_id")
            section_title = meta.get("section_title")
            section_path = meta.get("section_path")
            hierarchical_path = meta.get("hierarchical_path")
            page_number = meta.get("page_number")

            # Remaining metadata as JSONB
            metadata_json = {
                k: v
                for k, v in meta.items()
                if k
                not in [
                    "chunk_id",
                    "document_id",
                    "content",
                    "section_id",
                    "section_title",
                    "section_path",
                    "hierarchical_path",
                    "page_number",
                ]
            }

            record = (
                chunk_id,
                document_id,
                embedding_list,
                content,
                section_id,
                section_title,
                section_path,
                hierarchical_path,
                page_number,
                metadata_json,
            )
            records.append(record)

        # Batch insert
        logger.info(f"Layer {layer}: Inserting {len(records)} records in batches of {self.batch_size}...")
        async with self.pool.acquire() as conn:
            # Disable indexes temporarily for faster inserts
            await conn.execute(f"SET maintenance_work_mem = '2GB';")

            insert_sql = f"""
                INSERT INTO vectors.layer{layer}
                (chunk_id, document_id, embedding, content, section_id, section_title,
                 section_path, hierarchical_path, page_number, metadata)
                VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9, $10::jsonb)
                ON CONFLICT (chunk_id) DO NOTHING
            """

            total_inserted = 0
            for i in tqdm(range(0, len(records), self.batch_size), desc=f"Layer {layer} insert"):
                batch = records[i : i + self.batch_size]

                # Use executemany for batch insert
                await conn.executemany(insert_sql, batch)
                total_inserted += len(batch)

            logger.info(f"Layer {layer}: Inserted {total_inserted} records")

        # Update statistics
        await self._update_stats()

        logger.info(f"✓ Layer {layer} migration complete!")

    async def migrate_faiss_store(self, faiss_dir: Path):
        """
        Migrate entire FAISS vector store.

        Args:
            faiss_dir: Path to FAISS vector store directory
        """
        logger.info(f"Loading FAISS store from {faiss_dir}...")

        # Load FAISS store
        try:
            store = FAISSVectorStore.load(faiss_dir)
            logger.info(f"FAISS store loaded: {store.get_stats()}")
        except Exception as e:
            logger.error(f"Failed to load FAISS store: {e}")
            raise

        # Migrate each layer
        try:
            await self.migrate_layer(
                layer=1,
                index=store.index_layer1,
                metadata=store.metadata_layer1,
                doc_id_map=store.doc_id_to_indices.get(1, {}),
            )

            await self.migrate_layer(
                layer=2,
                index=store.index_layer2,
                metadata=store.metadata_layer2,
                doc_id_map=store.doc_id_to_indices.get(2, {}),
            )

            await self.migrate_layer(
                layer=3,
                index=store.index_layer3,
                metadata=store.metadata_layer3,
                doc_id_map=store.doc_id_to_indices.get(3, {}),
            )

            logger.info("✓ ✓ ✓ All layers migrated successfully!")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

    async def _update_stats(self):
        """Update vector store statistics."""
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT metadata.update_vector_store_stats();")

    async def verify_migration(self):
        """Verify migration by comparing counts."""
        logger.info("Verifying migration...")

        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                "SELECT * FROM metadata.vector_store_stats WHERE id = 1"
            )

            logger.info(f"Migration verification:")
            logger.info(f"  - Layer 1: {stats['layer1_count']} vectors")
            logger.info(f"  - Layer 2: {stats['layer2_count']} vectors")
            logger.info(f"  - Layer 3: {stats['layer3_count']} vectors")
            logger.info(f"  - Total: {stats['total_vectors']} vectors")
            logger.info(f"  - Documents: {stats['document_count']} documents")

            # Test search query
            test_vec = np.random.rand(3072).tolist()
            result = await conn.fetchrow(
                """
                SELECT chunk_id, content, 1 - (embedding <=> $1::vector) AS score
                FROM vectors.layer3
                ORDER BY embedding <=> $1::vector
                LIMIT 1
                """,
                test_vec,
            )

            if result:
                logger.info(f"✓ Test search successful: {result['chunk_id']} (score={result['score']:.3f})")
            else:
                logger.warning("⚠ No results found in test search")

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Connection pool closed")


async def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(description="Migrate FAISS vector store to PostgreSQL")
    parser.add_argument(
        "--faiss-dir",
        type=str,
        default="vector_db/",
        help="Path to FAISS vector store directory (default: vector_db/)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        required=True,
        help="PostgreSQL connection string (postgresql://user:pass@host:port/db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch insert size (default: 500)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration after completion",
    )

    args = parser.parse_args()

    # Initialize migrator
    migrator = FAISSToPostgresMigrator(db_url=args.db_url, batch_size=args.batch_size)

    try:
        await migrator.initialize()

        # Run migration
        faiss_dir = Path(args.faiss_dir)
        if not faiss_dir.exists():
            logger.error(f"FAISS directory not found: {faiss_dir}")
            return 1

        await migrator.migrate_faiss_store(faiss_dir)

        # Verify if requested
        if args.verify:
            await migrator.verify_migration()

        logger.info("✓ ✓ ✓ Migration completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1

    finally:
        await migrator.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
