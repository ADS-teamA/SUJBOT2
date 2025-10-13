"""
PostgreSQL + pgvector Vector Store
Drop-in replacement for FAISS-based MultiDocumentVectorStore
Optimized for 100,000+ page legal document corpus
"""

import asyncio
import logging
import hashlib
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import asyncpg
except ImportError:
    raise ImportError(
        "asyncpg is required for PostgreSQL vector store. "
        "Install with: pip install asyncpg"
    )

import numpy as np

from .embeddings import LegalEmbedder
from .chunker import LegalChunk
from .indexing import SearchResult, ReferenceMap

logger = logging.getLogger(__name__)


@dataclass
class PostgreSQLConfig:
    """Configuration for PostgreSQL vector store"""

    # Connection
    host: str = "localhost"
    port: int = 5432
    database: str = "sujbot2"
    user: str = "sujbot_app"
    password: str = ""

    # Connection pool
    min_pool_size: int = 5
    max_pool_size: int = 20

    # Performance
    vector_search_probes: int = 10  # IVFFlat probes
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 3600


class PostgreSQLVectorStore:
    """
    PostgreSQL + pgvector implementation replacing FAISS MultiDocumentVectorStore

    Compatible API with the original MultiDocumentVectorStore:
    - add_document()
    - search()
    - search_by_reference()
    - get_document_chunks()
    - get_document_info()
    - get_document_count()
    - get_chunk_count()
    """

    def __init__(
        self,
        embedder: LegalEmbedder,
        config: Optional[PostgreSQLConfig] = None
    ):
        """
        Initialize PostgreSQL vector store

        Args:
            embedder: Legal embedder instance
            config: PostgreSQL configuration
        """
        self.embedder = embedder
        self.config = config or PostgreSQLConfig()

        # Connection pool (initialized lazily)
        self._pool: Optional[asyncpg.Pool] = None

        # Reference map (in-memory cache)
        self.reference_map = ReferenceMap()

        # Document metadata cache
        self.document_info: Dict[str, Dict[str, Any]] = {}

    async def _ensure_pool(self):
        """Initialize connection pool if not already done"""
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    command_timeout=60,
                    timeout=30  # Connection acquisition timeout
                )
                logger.info(
                    f"Created PostgreSQL connection pool: "
                    f"{self.config.host}:{self.config.port}/{self.config.database}"
                )

                # Set pgvector search parameters
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        f"SET ivfflat.probes = {self.config.vector_search_probes}"
                    )
                    logger.info(f"Set ivfflat.probes = {self.config.vector_search_probes}")

            except asyncpg.InvalidPasswordError as e:
                logger.error(
                    f"PostgreSQL authentication failed for user {self.config.user}: {e}. "
                    f"Check POSTGRES_PASSWORD in environment."
                )
                raise ConnectionError(
                    f"Database authentication failed for user '{self.config.user}'. "
                    f"Verify POSTGRES_PASSWORD is set correctly."
                ) from e
            except asyncpg.InvalidCatalogNameError as e:
                logger.error(
                    f"Database '{self.config.database}' does not exist: {e}. "
                    f"Run database init script first."
                )
                raise ConnectionError(
                    f"Database '{self.config.database}' not found on "
                    f"{self.config.host}:{self.config.port}. "
                    f"Create the database using: psql -c 'CREATE DATABASE {self.config.database}'"
                ) from e
            except asyncio.TimeoutError as e:
                logger.error(
                    f"Connection timeout to {self.config.host}:{self.config.port}: {e}. "
                    f"Check PostgreSQL is running and network is reachable."
                )
                raise ConnectionError(
                    f"Database connection timeout to {self.config.host}:{self.config.port}. "
                    f"Ensure PostgreSQL is running and accessible."
                ) from e
            except OSError as e:
                # Covers connection refused, network unreachable, etc.
                logger.error(
                    f"Network error connecting to PostgreSQL at {self.config.host}:{self.config.port}: {e}"
                )
                raise ConnectionError(
                    f"Cannot connect to PostgreSQL at {self.config.host}:{self.config.port}. "
                    f"Check that PostgreSQL is running and the host/port are correct."
                ) from e
            except Exception as e:
                logger.error(
                    f"Unexpected error creating connection pool: {type(e).__name__}: {e}",
                    exc_info=True
                )
                raise ConnectionError(
                    f"Database connection failed: {type(e).__name__}: {e}"
                ) from e

    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            logger.info("Closed PostgreSQL connection pool")

    async def add_document(
        self,
        chunks: List[LegalChunk],
        document_id: str,
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Add document to PostgreSQL store

        Args:
            chunks: List of legal chunks
            document_id: Unique document identifier
            document_type: Type of document (law_code, contract, regulation)
            metadata: Additional document metadata
            progress_callback: Optional callback(chunks_processed, total_chunks) for progress tracking
        """
        if not chunks:
            logger.warning(f"No chunks provided for document {document_id}")
            return

        logger.info(f"Adding document {document_id} with {len(chunks)} chunks to PostgreSQL")

        await self._ensure_pool()

        # 1. Generate embeddings with progress tracking
        total_chunks = len(chunks)
        embeddings = []

        # Process in batches to report progress
        batch_size = self.embedder.config.batch_size if hasattr(self.embedder, 'config') else 32

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = await self.embedder.embed_chunks(batch)
            embeddings.extend(batch_embeddings)

            # Report progress
            if progress_callback:
                chunks_processed = min(i + batch_size, total_chunks)
                progress_callback(chunks_processed, total_chunks)

        embeddings = np.array(embeddings)

        # 2. Insert into PostgreSQL with error handling
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Insert document record
                    await self._insert_document(
                        conn, document_id, document_type, chunks, metadata
                    )

                    # Batch insert chunks with embeddings
                    await self._insert_chunks_batch(
                        conn, chunks, embeddings, document_id, document_type
                    )

                    # Build references table
                    await self._insert_references(conn, chunks, document_id)

                    # Build knowledge graph edges (if applicable)
                    if document_type == 'law_code':
                        await self._build_knowledge_graph(conn, chunks, document_id)

            # 3. Update in-memory caches ONLY if transaction succeeded
            await self.reference_map.build(chunks)

            self.document_info[document_id] = {
                'document_type': document_type,
                'num_chunks': len(chunks),
                'metadata': metadata or {}
            }

            logger.info(f"Successfully added document {document_id} to PostgreSQL")

        except asyncpg.UniqueViolationError as e:
            logger.error(
                f"Document {document_id} already exists in database: {e}. "
                f"Delete the existing document first or use a different document_id."
            )
            raise ValueError(
                f"Document '{document_id}' already exists. "
                f"Use delete_document() first or choose a different ID."
            ) from e
        except asyncpg.CheckViolationError as e:
            logger.error(
                f"Data constraint violation for {document_id}: {e}. "
                f"Check embedding dimensions match schema (vector(768))."
            )
            raise ValueError(
                f"Data validation error for document '{document_id}': {e}. "
                f"Ensure embeddings have correct dimensions (768)."
            ) from e
        except asyncpg.DiskFullError as e:
            logger.error(f"Database disk full while indexing {document_id}: {e}")
            raise OSError(
                f"Database storage full. Free up space and retry indexing document '{document_id}'."
            ) from e
        except Exception as e:
            logger.error(
                f"Failed to add document {document_id} to PostgreSQL: {type(e).__name__}: {e}",
                exc_info=True
            )
            # Don't update caches if transaction failed
            raise RuntimeError(
                f"Database indexing failed for document '{document_id}': {type(e).__name__}: {e}"
            ) from e

    async def _insert_document(
        self,
        conn: asyncpg.Connection,
        document_id: str,
        document_type: str,
        chunks: List[LegalChunk],
        metadata: Optional[Dict[str, Any]]
    ):
        """Insert document metadata"""
        # Extract document-level metadata from chunks
        first_chunk = chunks[0] if chunks else None

        title = None
        document_number = None
        law_citation = None

        if first_chunk and first_chunk.metadata:
            title = first_chunk.metadata.get('document_title', '')
            document_number = first_chunk.metadata.get('document_number', '')
            law_citation = first_chunk.metadata.get('law_citation', '')

        # Calculate statistics
        total_tokens = sum(
            chunk.metadata.get('token_count', 0) for chunk in chunks
        )

        # For asyncpg execute(), also need JSON string with ::jsonb cast
        metadata_json = json.dumps(metadata or {})

        await conn.execute(
            """
            INSERT INTO documents (
                document_id, filename, document_type, title,
                document_number, law_citation, status, total_chunks,
                total_words, metadata, indexed_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, NOW())
            ON CONFLICT (document_id) DO UPDATE SET
                status = EXCLUDED.status,
                total_chunks = EXCLUDED.total_chunks,
                indexed_at = NOW(),
                updated_at = NOW()
            """,
            document_id,
            metadata.get('filename', document_id) if metadata else document_id,
            document_type,
            title,
            document_number,
            law_citation,
            'indexed',
            len(chunks),
            total_tokens,
            metadata_json,
        )

    async def _insert_chunks_batch(
        self,
        conn: asyncpg.Connection,
        chunks: List[LegalChunk],
        embeddings: np.ndarray,
        document_id: str,
        document_type: str
    ):
        """Batch insert chunks with embeddings"""

        # Prepare batch data
        records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Convert embedding to string representation for pgvector executemany()
            # pgvector expects: '[1.0, 2.0, 3.0]' format for batch inserts
            embedding_str = str(embedding.tolist())

            # Extract hierarchical components
            metadata = chunk.metadata
            part_num = metadata.get('part')
            chapter_num = metadata.get('chapter')
            paragraph_num = metadata.get('paragraph')
            article_num = metadata.get('article')
            subsection_num = metadata.get('subsection')

            # Extract content classification
            content_type = metadata.get('content_type', 'general')
            contains_obligation = content_type == 'obligation'
            contains_prohibition = content_type == 'prohibition'
            contains_definition = content_type == 'definition'

            # References
            references_to = metadata.get('references_to', [])

            # Page number
            page_num = metadata.get('page_number')

            # Token count
            token_count = metadata.get('token_count', 0)
            word_count = len(chunk.content.split())

            # For executemany(), pass metadata as JSON string (asyncpg requirement)
            metadata_json = json.dumps(metadata or {})

            record = (
                chunk.chunk_id,
                i,  # chunk_index
                document_id,
                chunk.content,
                embedding_str,  # String representation for pgvector
                chunk.hierarchy_path,
                chunk.legal_reference,
                chunk.structural_level,
                part_num,
                chapter_num,
                paragraph_num,
                article_num,
                subsection_num,
                page_num,
                content_type,
                contains_obligation,
                contains_prohibition,
                contains_definition,
                token_count,
                word_count,
                references_to,
                metadata_json  # JSON string for executemany()
            )
            records.append(record)

        # Batch insert
        await conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id, chunk_index, document_id, content,
                embedding, hierarchy_path, legal_reference, structural_level,
                part_number, chapter_number, paragraph_number,
                article_number, subsection_number, page_number,
                content_type, contains_obligation, contains_prohibition,
                contains_definition, token_count, word_count,
                references_to, metadata
            )
            VALUES (
                $1, $2, $3, $4, $5::vector, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22::jsonb
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding
            """,
            records
        )

        logger.info(f"Inserted {len(records)} chunks into PostgreSQL")

    async def _insert_references(
        self,
        conn: asyncpg.Connection,
        chunks: List[LegalChunk],
        document_id: str
    ):
        """Build references table from chunks"""
        records = []

        for chunk in chunks:
            if chunk.legal_reference:
                # Add reference mapping: reference → chunk
                ref_normalized = self._normalize_reference(chunk.legal_reference)
                ref_type = self._classify_reference_type(chunk.legal_reference)

                # Parse components
                para_num, article_num, subsection_num = self._parse_reference_components(
                    chunk.legal_reference
                )

                # For executemany(), pass metadata as JSON string
                ref_metadata_json = json.dumps({})

                record = (
                    chunk.legal_reference,
                    ref_normalized,
                    ref_type,
                    para_num,
                    article_num,
                    subsection_num,
                    chunk.chunk_id,
                    document_id,
                    1.0,  # confidence
                    True,  # is_explicit
                    ref_metadata_json  # metadata as JSON string
                )
                records.append(record)

        if records:
            await conn.executemany(
                """
                INSERT INTO "references" (
                    reference_text, reference_normalized, reference_type,
                    paragraph_num, article_num, subsection_num,
                    chunk_id, document_id, confidence, is_explicit, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
                ON CONFLICT DO NOTHING
                """,
                records
            )
            logger.info(f"Inserted {len(records)} references")

    async def _build_knowledge_graph(
        self,
        conn: asyncpg.Connection,
        chunks: List[LegalChunk],
        document_id: str
    ):
        """Build knowledge graph edges for law documents"""
        records = []

        # Build hierarchical edges (part_of relationships)
        chunk_id_map = {chunk.chunk_id: chunk for chunk in chunks}

        for chunk in chunks:
            # Get references_to from metadata
            references_to = chunk.metadata.get('references_to', [])

            for ref in references_to:
                # Find target chunk by reference
                target_chunks = [
                    c for c in chunks if c.legal_reference == ref
                ]

                if target_chunks:
                    target_chunk = target_chunks[0]

                    # For executemany(), pass metadata as JSON string
                    graph_metadata_json = json.dumps({})

                    record = (
                        chunk.chunk_id,
                        target_chunk.chunk_id,
                        'references',  # edge_type
                        1.0,  # weight
                        1.0,  # confidence
                        f"{chunk.legal_reference} references {target_chunk.legal_reference}",
                        'reference_extractor',  # detected_by
                        graph_metadata_json  # metadata as JSON string
                    )
                    records.append(record)

        if records:
            await conn.executemany(
                """
                INSERT INTO cross_references (
                    source_chunk_id, target_chunk_id, edge_type,
                    weight, confidence, explanation, detected_by, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                ON CONFLICT (source_chunk_id, target_chunk_id, edge_type) DO NOTHING
                """,
                records
            )
            logger.info(f"Inserted {len(records)} knowledge graph edges")

    def _normalize_reference(self, reference: str) -> str:
        """Normalize legal reference (e.g., '§89 odst. 2' → '§89.2')"""
        import re

        # Simple normalization - can be enhanced
        normalized = reference

        # §89 odst. 2 → §89.2
        normalized = re.sub(r'§(\d+)\s+odst\.\s*(\d+)', r'§\1.\2', normalized)

        # Článek 5 odst. 2 → Článek 5.2
        normalized = re.sub(r'([Čč]lánek)\s+(\d+)\s+odst\.\s*(\d+)', r'\1 \2.\3', normalized)

        return normalized

    def _classify_reference_type(self, reference: str) -> str:
        """Classify reference type"""
        if '§' in reference:
            if 'odst' in reference.lower():
                return 'subsection'
            return 'paragraph'
        elif 'článek' in reference.lower() or 'Článek' in reference:
            return 'article'
        return 'section'

    def _parse_reference_components(self, reference: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Parse reference into components (paragraph, article, subsection)"""
        import re

        para_num = None
        article_num = None
        subsection_num = None

        # Try to extract paragraph number
        para_match = re.search(r'§(\d+)', reference)
        if para_match:
            para_num = int(para_match.group(1))

            # Check for subsection
            subsec_match = re.search(r'odst\.\s*(\d+)', reference)
            if subsec_match:
                subsection_num = int(subsec_match.group(1))

        # Try to extract article number
        article_match = re.search(r'[Čč]lánek\s+(\d+)', reference)
        if article_match:
            article_num = int(article_match.group(1))

            # Check for subsection
            subsec_match = re.search(r'odst\.\s*(\d+)', reference)
            if subsec_match:
                subsection_num = int(subsec_match.group(1))

        return para_num, article_num, subsection_num

    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Semantic search using pgvector

        Args:
            query: Search query
            document_ids: Filter by document IDs (None = all)
            top_k: Number of results
            filter_metadata: Metadata filters

        Returns:
            List of search results with scores
        """
        await self._ensure_pool()

        # 1. Embed query
        query_embedding = await self.embedder.embed_query(query)
        # Convert to string representation for pgvector
        embedding_str = str(query_embedding.tolist())

        # 2. Build SQL query
        sql = """
        SELECT
            c.chunk_id,
            c.chunk_index,
            c.content,
            c.hierarchy_path,
            c.legal_reference,
            c.structural_level,
            c.document_id,
            c.metadata,
            1 - (c.embedding <=> $1::vector) AS similarity_score
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        """

        params = [embedding_str]
        param_idx = 2

        # Filter by document IDs
        if document_ids:
            sql += f" AND c.document_id = ANY(${param_idx})"
            params.append(document_ids)
            param_idx += 1

        # Metadata filters
        if filter_metadata:
            for key, value in filter_metadata.items():
                if key == 'content_type':
                    sql += f" AND c.content_type = ${param_idx}"
                    params.append(value)
                    param_idx += 1
                elif key == 'contains_obligation':
                    sql += f" AND c.contains_obligation = ${param_idx}"
                    params.append(value)
                    param_idx += 1

        # Order and limit
        sql += f"""
        ORDER BY c.embedding <=> $1::vector
        LIMIT ${param_idx}
        """
        params.append(top_k)

        # 3. Execute query
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        # 4. Convert to SearchResult objects
        results = []
        for rank, row in enumerate(rows, start=1):
            # Ensure metadata is a dict
            metadata = row['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            elif metadata is None:
                metadata = {}

            chunk = LegalChunk(
                chunk_id=row['chunk_id'],
                chunk_index=row['chunk_index'],
                content=row['content'],
                document_id=row['document_id'],
                document_type=self.document_info.get(row['document_id'], {}).get('document_type', ''),
                hierarchy_path=row['hierarchy_path'],
                legal_reference=row['legal_reference'],
                structural_level=row['structural_level'],
                metadata=metadata
            )

            results.append(SearchResult(
                chunk_id=row['chunk_id'],
                chunk=chunk,
                score=float(row['similarity_score']),
                document_id=row['document_id'],
                rank=rank
            ))

        return results

    async def search_by_reference(
        self,
        legal_ref: str,
        document_id: Optional[str] = None
    ) -> Optional[LegalChunk]:
        """
        Direct lookup by legal reference

        Args:
            legal_ref: Legal reference (e.g., "§89 odst. 2")
            document_id: Filter by document ID (optional)

        Returns:
            First matching chunk or None
        """
        await self._ensure_pool()

        sql = """
        SELECT
            c.chunk_id, c.chunk_index, c.content, c.document_id,
            c.document_type, c.hierarchy_path, c.legal_reference,
            c.structural_level, c.metadata
        FROM chunks c
        WHERE c.legal_reference = $1
        """

        params = [legal_ref]

        if document_id:
            sql += " AND c.document_id = $2"
            params.append(document_id)

        sql += " LIMIT 1"

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *params)

        if not row:
            return None

        # Ensure metadata is a dict
        metadata = row['metadata']
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}

        return LegalChunk(
            chunk_id=row['chunk_id'],
            chunk_index=row['chunk_index'],
            content=row['content'],
            document_id=row['document_id'],
            document_type=self.document_info.get(row['document_id'], {}).get('document_type', ''),
            hierarchy_path=row['hierarchy_path'],
            legal_reference=row['legal_reference'],
            structural_level=row['structural_level'],
            metadata=metadata
        )

    async def get_document_chunks(self, document_id: str) -> List[LegalChunk]:
        """
        Get all chunks for a document

        Args:
            document_id: Document identifier

        Returns:
            List of all chunks for the document
        """
        await self._ensure_pool()

        sql = """
        SELECT
            chunk_id, chunk_index, content, document_id,
            hierarchy_path, legal_reference, structural_level, metadata
        FROM chunks
        WHERE document_id = $1
        ORDER BY chunk_index
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, document_id)

        chunks = []
        for row in rows:
            # Ensure metadata is a dict (asyncpg should return JSONB as dict, but double-check)
            metadata = row['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            elif metadata is None:
                metadata = {}

            chunk = LegalChunk(
                chunk_id=row['chunk_id'],
                chunk_index=row['chunk_index'],
                content=row['content'],
                document_id=row['document_id'],
                document_type=self.document_info.get(document_id, {}).get('document_type', ''),
                hierarchy_path=row['hierarchy_path'],
                legal_reference=row['legal_reference'],
                structural_level=row['structural_level'],
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks

    def get_document_count(self) -> int:
        """Get total number of indexed documents"""
        return len(self.document_info)

    def get_chunk_count(self, document_id: Optional[str] = None) -> int:
        """Get total number of indexed chunks"""
        if document_id:
            return self.document_info.get(document_id, {}).get('num_chunks', 0)
        return sum(info.get('num_chunks', 0) for info in self.document_info.values())

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a document"""
        return self.document_info.get(document_id)

    async def load_document_info_from_db(self):
        """Load document info from database into memory cache"""
        await self._ensure_pool()

        sql = """
        SELECT document_id, document_type, total_chunks, metadata
        FROM documents
        WHERE status = 'indexed'
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)

        for row in rows:
            self.document_info[row['document_id']] = {
                'document_type': row['document_type'],
                'num_chunks': row['total_chunks'],
                'metadata': row['metadata']
            }

        logger.info(f"Loaded {len(rows)} documents from database")

    async def delete_document(self, document_id: str):
        """
        Delete document and all its chunks

        Args:
            document_id: Document identifier
        """
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM documents WHERE document_id = $1",
                document_id
            )

        # Remove from cache
        if document_id in self.document_info:
            del self.document_info[document_id]

        logger.info(f"Deleted document {document_id} from PostgreSQL")


class PostgreSQLIndexPersistence:
    """
    Compatibility layer for index persistence
    With PostgreSQL, persistence is automatic (data is always in DB)
    """

    def __init__(self, vector_store: PostgreSQLVectorStore):
        self.vector_store = vector_store

    async def save(
        self,
        document_id: str,
        index: Any,  # Ignored (data already in DB)
        metadata: Dict,
        chunks: List[LegalChunk],
        reference_map: Optional[ReferenceMap] = None
    ):
        """No-op: PostgreSQL persists automatically"""
        logger.debug(f"PostgreSQL auto-persists: {document_id}")

    async def load(
        self,
        document_id: str
    ) -> Tuple[None, Dict, List[LegalChunk], Optional[ReferenceMap]]:
        """Load from PostgreSQL"""
        # Load document info
        doc_info = self.vector_store.get_document_info(document_id)
        if not doc_info:
            raise ValueError(f"Document {document_id} not found")

        # Load chunks
        chunks = await self.vector_store.get_document_chunks(document_id)

        # Rebuild reference map
        reference_map = ReferenceMap()
        await reference_map.build(chunks)

        return None, doc_info['metadata'], chunks, reference_map

    def exists(self, document_id: str) -> bool:
        """Check if document exists"""
        return document_id in self.vector_store.document_info

    def list_documents(self) -> List[str]:
        """List all documents"""
        return list(self.vector_store.document_info.keys())

    async def delete(self, document_id: str):
        """Delete document"""
        await self.vector_store.delete_document(document_id)
