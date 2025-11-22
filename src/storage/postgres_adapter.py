"""
PostgreSQL Vector Store Adapter with pgvector

New implementation using PostgreSQL + pgvector for vector similarity search.
Replaces FAISS in-memory store with persistent database backend.
"""

import asyncio
import asyncpg
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import nest_asyncio

from .vector_store_adapter import VectorStoreAdapter

logger = logging.getLogger(__name__)

# Enable nested event loops
nest_asyncio.apply()


def _run_async_safe(coro):
    """
    Safely run async coroutine from sync context.

    Uses asyncio.run() with nest-asyncio to handle nested event loops.
    nest_asyncio patches asyncio.run() to work even when called from within
    an existing event loop (e.g., FastAPI's loop).
    """
    # With nest_asyncio.apply() at module level, asyncio.run() works
    # even when called from within FastAPI's async context
    return asyncio.run(coro)


class PostgresVectorStoreAdapter(VectorStoreAdapter):
    """
    PostgreSQL + pgvector implementation of vector store.

    Architecture:
    - 3 tables: vectors.layer1, vectors.layer2, vectors.layer3
    - HNSW indexes for fast approximate nearest neighbor search
    - Full-text search integrated (tsvector + GIN indexes)
    - Connection pooling via asyncpg

    Performance:
    - Search latency: 10-50ms (vs FAISS 5-10ms)
    - Supports millions of vectors
    - ACID transactions, persistent storage
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 20,
        dimensions: int = 3072,
    ):
        """
        Initialize PostgreSQL adapter.

        Args:
            connection_string: PostgreSQL DSN (postgresql://user:pass@host:port/db)
            pool_size: Connection pool size (default: 20)
            dimensions: Embedding dimensionality (default: 3072 for text-embedding-3-large)
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.dimensions = dimensions
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

        # Metadata cache (materialized on-demand)
        self._metadata_cache = {1: None, 2: None, 3: None}

    async def initialize(self):
        """Initialize connection pool. Must be called before using the adapter."""
        if not self._initialized:
            await self._initialize_pool()
            self._initialized = True

    async def _initialize_pool(self):
        """Create asyncpg connection pool with retry logic."""
        import time

        max_retries = 5
        retry_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempting to connect to PostgreSQL (attempt {attempt}/{max_retries})...")

                self.pool = await asyncpg.create_pool(
                    dsn=self.connection_string,
                    min_size=2,
                    max_size=self.pool_size,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300,
                    command_timeout=60,
                )
                logger.info(f"PostgreSQL connection pool created (size={self.pool_size})")

                # Verify extensions
                async with self.pool.acquire() as conn:
                    extensions = await conn.fetch(
                        "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'age')"
                    )
                    ext_names = {row["extname"] for row in extensions}
                    if "vector" not in ext_names:
                        raise RuntimeError("pgvector extension not installed")
                    logger.info("PostgreSQL extensions verified: vector")

                return  # Success!

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Connection attempt {attempt} failed: {e}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to initialize PostgreSQL pool after {max_retries} attempts: {e}")
                    raise ConnectionError(f"PostgreSQL connection failed after {max_retries} attempts: {e}") from e

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity (pgvector requires this)."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _vector_to_pgvector_string(self, vec: np.ndarray) -> str:
        """
        Convert numpy array to PostgreSQL vector string format.

        asyncpg doesn't automatically serialize Python lists to pgvector type,
        so we need to manually convert to string format: '[1.0,2.0,3.0]'

        Args:
            vec: Numpy array or list (can be nested)

        Returns:
            String representation compatible with pgvector (e.g., '[0.1,0.2,0.3]')
        """
        # Flatten array if needed (handles 2D arrays from some embedding models)
        if isinstance(vec, np.ndarray):
            vec = vec.flatten().tolist()
        elif isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
            # Handle nested lists (e.g., [[0.1, 0.2, ...]])
            vec = vec[0] if len(vec) == 1 else sum(vec, [])

        return '[' + ','.join(map(str, vec)) + ']'

    # ============================================================================
    # Core Search Methods
    # ============================================================================

    def hierarchical_search(
        self,
        query_embedding: np.ndarray,
        k_layer3: int = 6,
        use_doc_filtering: bool = True,
        similarity_threshold_offset: float = 0.25,
        query_text: Optional[str] = None,
        document_filter: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Hierarchical 3-layer search using PostgreSQL.

        Strategy:
        1. Search Layer 1 (document-level) → find top document (unless document_filter provided)
        2. Use document_id to filter Layer 3 search
        3. Retrieve top-k chunks from Layer 3
        4. Apply similarity threshold filtering
        5. Search Layer 2 (section-level) for context

        Args:
            query_embedding: Query embedding vector
            k_layer3: Number of layer3 chunks to retrieve
            use_doc_filtering: Enable document filtering (ignored if document_filter provided)
            similarity_threshold_offset: Threshold offset for filtering
            query_text: Original query text (for logging)
            document_filter: Explicit document ID to filter by (overrides use_doc_filtering)
        """
        return _run_async_safe(
            self._async_hierarchical_search(
                query_embedding,
                k_layer3,
                use_doc_filtering,
                similarity_threshold_offset,
                query_text,
                document_filter,
            )
        )

    async def _async_hierarchical_search(
        self,
        query_embedding: np.ndarray,
        k_layer3: int,
        use_doc_filtering: bool,
        similarity_threshold_offset: float,
        query_text: Optional[str],
        document_filter: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """Async implementation of hierarchical search."""
        query_vec = self._normalize_vector(query_embedding)

        async with self.pool.acquire() as conn:
            # Step 1: Search Layer 1 (find top document) - skip if document_filter provided
            if document_filter:
                # Explicit document filter provided - skip layer1 search
                layer1_results = []
                logger.debug(f"Using explicit document filter: {document_filter}")
            else:
                # Search Layer 1 to find top document
                layer1_results = await self._search_layer(
                    conn, layer=1, query_vec=query_vec, k=1
                )

                # Get document filter from layer1 results
                if use_doc_filtering and layer1_results:
                    document_filter = layer1_results[0]["document_id"]
                    logger.debug(f"Document filter from layer1: {document_filter}")

            # Step 2: Search Layer 3 (PRIMARY retrieval)
            layer3_results = await self._search_layer(
                conn,
                layer=3,
                query_vec=query_vec,
                k=k_layer3,
                document_filter=document_filter,
                query_text=query_text,
            )

            # Apply similarity threshold
            if layer3_results and similarity_threshold_offset > 0:
                top_score = layer3_results[0]["score"]
                threshold = top_score - similarity_threshold_offset
                layer3_results = [r for r in layer3_results if r["score"] >= threshold]
                logger.debug(
                    f"Threshold filtering: {len(layer3_results)} results kept (threshold={threshold:.3f})"
                )

            # Step 3: Search Layer 2 (section context)
            layer2_results = await self._search_layer(
                conn, layer=2, query_vec=query_vec, k=3, document_filter=document_filter
            )

            return {
                "layer1": layer1_results,
                "layer2": layer2_results,
                "layer3": layer3_results,
            }

    async def _search_layer(
        self,
        conn: asyncpg.Connection,
        layer: int,
        query_vec: np.ndarray,
        k: int,
        document_filter: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search specific layer using pgvector cosine similarity.

        Uses <=> operator (cosine distance) from pgvector.
        Score = 1 - distance (0=dissimilar, 1=identical)
        """
        # Convert query vector to PostgreSQL-compatible string format
        query_str = self._vector_to_pgvector_string(query_vec)

        # Build SQL query with layer-specific columns
        # Layer 1 (documents) does not have section_id, section_title
        # Layers 2-3 (sections/chunks) have these columns
        # All layers have hierarchical_path
        section_columns = (
            "section_id, section_title,"
            if layer > 1
            else ""
        )

        if query_text:
            # Hybrid search: Dense (pgvector) + Sparse (full-text)
            return await self._hybrid_search_layer(
                conn, layer, query_vec, query_text, k, document_filter
            )
        else:
            # Pure vector search
            if document_filter:
                sql = f"""
                    SELECT
                        chunk_id,
                        document_id,
                        metadata,
                        content,
                        {section_columns}
                        hierarchical_path,
                        1 - (embedding <=> $1::vector) AS score
                    FROM vectors.layer{layer}
                    WHERE document_id = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """
                rows = await conn.fetch(sql, query_str, document_filter, k)
            else:
                sql = f"""
                    SELECT
                        chunk_id,
                        document_id,
                        metadata,
                        content,
                        {section_columns}
                        hierarchical_path,
                        1 - (embedding <=> $1::vector) AS score
                    FROM vectors.layer{layer}
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """
                rows = await conn.fetch(sql, query_str, k)

        # Convert rows to dicts
        results = []
        for row in rows:
            result = {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "score": float(row["score"]),
                "section_id": row.get("section_id"),
                "section_title": row.get("section_title"),
                "hierarchical_path": row.get("hierarchical_path"),
            }
            # Merge JSONB metadata (PostgreSQL returns JSONB as dict)
            if row["metadata"] and isinstance(row["metadata"], dict):
                result.update(row["metadata"])
            results.append(result)

        return results

    async def _hybrid_search_layer(
        self,
        conn: asyncpg.Connection,
        layer: int,
        query_vec: np.ndarray,
        query_text: str,
        k: int,
        document_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Hybrid search: Dense (pgvector) + Sparse (full-text) with RRF fusion.

        RRF (Reciprocal Rank Fusion): score = 1/(k + rank)
        k=60 (standard parameter from research)
        """
        # Convert query vector to PostgreSQL-compatible string format
        query_str = self._vector_to_pgvector_string(query_vec)

        # Layer-specific columns (Layer 1 doesn't have section_id, section_title)
        # All layers have hierarchical_path
        section_columns = (
            "l.section_id, l.section_title,"
            if layer > 1
            else ""
        )

        # Build hybrid search with RRF fusion
        if document_filter:
            sql = f"""
                WITH dense AS (
                    SELECT chunk_id, 1 - (embedding <=> $1::vector) AS dense_score,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS dense_rank
                    FROM vectors.layer{layer}
                    WHERE document_id = $3
                    ORDER BY embedding <=> $1::vector
                    LIMIT 50
                ),
                sparse AS (
                    SELECT chunk_id, ts_rank(content_tsv, to_tsquery('english', $2)) AS sparse_score,
                           ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsv, to_tsquery('english', $2)) DESC) AS sparse_rank
                    FROM vectors.layer{layer}
                    WHERE content_tsv @@ to_tsquery('english', $2)
                      AND document_id = $3
                    ORDER BY sparse_score DESC
                    LIMIT 50
                ),
                fused AS (
                    SELECT
                        COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
                        COALESCE(1.0 / (60 + d.dense_rank), 0) + COALESCE(1.0 / (60 + s.sparse_rank), 0) AS rrf_score
                    FROM dense d
                    FULL OUTER JOIN sparse s USING (chunk_id)
                )
                SELECT
                    f.chunk_id,
                    l.document_id,
                    l.content,
                    l.metadata,
                    {section_columns}
                    l.hierarchical_path,
                    f.rrf_score AS score
                FROM fused f
                JOIN vectors.layer{layer} l ON f.chunk_id = l.chunk_id
                ORDER BY f.rrf_score DESC
                LIMIT $4
            """
            # Preprocess query text for tsquery (replace spaces with &)
            tsquery = query_text.replace(" ", " & ")
            rows = await conn.fetch(sql, query_str, tsquery, document_filter, k)
        else:
            # Same query without document filter
            sql = f"""
                WITH dense AS (
                    SELECT chunk_id, 1 - (embedding <=> $1::vector) AS dense_score,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS dense_rank
                    FROM vectors.layer{layer}
                    ORDER BY embedding <=> $1::vector
                    LIMIT 50
                ),
                sparse AS (
                    SELECT chunk_id, ts_rank(content_tsv, to_tsquery('english', $2)) AS sparse_score,
                           ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsv, to_tsquery('english', $2)) DESC) AS sparse_rank
                    FROM vectors.layer{layer}
                    WHERE content_tsv @@ to_tsquery('english', $2)
                    ORDER BY sparse_score DESC
                    LIMIT 50
                ),
                fused AS (
                    SELECT
                        COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
                        COALESCE(1.0 / (60 + d.dense_rank), 0) + COALESCE(1.0 / (60 + s.sparse_rank), 0) AS rrf_score
                    FROM dense d
                    FULL OUTER JOIN sparse s USING (chunk_id)
                )
                SELECT
                    f.chunk_id,
                    l.document_id,
                    l.content,
                    l.metadata,
                    {section_columns}
                    l.hierarchical_path,
                    f.rrf_score AS score
                FROM fused f
                JOIN vectors.layer{layer} l ON f.chunk_id = l.chunk_id
                ORDER BY f.rrf_score DESC
                LIMIT $3
            """
            tsquery = query_text.replace(" ", " & ")
            rows = await conn.fetch(sql, query_str, tsquery, k)

        # Convert to dicts
        results = []
        for row in rows:
            result = {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "score": float(row["score"]),
                "section_id": row.get("section_id"),
                "section_title": row.get("section_title"),
                "hierarchical_path": row.get("hierarchical_path"),
            }
            # Merge JSONB metadata (PostgreSQL returns JSONB as dict)
            if row["metadata"] and isinstance(row["metadata"], dict):
                result.update(row["metadata"])
            results.append(result)

        return results

    def search_layer1(
        self,
        query_embedding: np.ndarray,
        k: int = 1,
        **kwargs
    ) -> List[Dict]:
        """Direct Layer 1 search (document-level)."""
        return _run_async_safe(
            self._async_search_layer1(query_embedding, k)
        )

    async def _async_search_layer1(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[Dict]:
        """Async Layer 1 search."""
        query_vec = self._normalize_vector(query_embedding)

        async with self.pool.acquire() as conn:
            results = await self._search_layer(
                conn, layer=1, query_vec=query_vec, k=k
            )
            return results

    def search_layer2(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        document_filter: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """Direct Layer 2 search (section-level)."""
        return _run_async_safe(
            self._async_search_layer2(query_embedding, k, document_filter)
        )

    async def _async_search_layer2(
        self,
        query_embedding: np.ndarray,
        k: int,
        document_filter: Optional[str],
    ) -> List[Dict]:
        """Async Layer 2 search."""
        query_vec = self._normalize_vector(query_embedding)

        async with self.pool.acquire() as conn:
            results = await self._search_layer(
                conn, layer=2, query_vec=query_vec, k=k, document_filter=document_filter
            )
            return results

    def search_layer3(
        self,
        query_embedding: np.ndarray,
        k: int = 6,
        document_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Direct Layer 3 search."""
        return _run_async_safe(
            self._async_search_layer3(query_embedding, k, document_filter, similarity_threshold)
        )

    async def _async_search_layer3(
        self,
        query_embedding: np.ndarray,
        k: int,
        document_filter: Optional[str],
        similarity_threshold: Optional[float],
    ) -> List[Dict]:
        """Async Layer 3 search."""
        query_vec = self._normalize_vector(query_embedding)

        async with self.pool.acquire() as conn:
            results = await self._search_layer(
                conn, layer=3, query_vec=query_vec, k=k, document_filter=document_filter
            )

            # Apply similarity threshold
            if similarity_threshold:
                results = [r for r in results if r["score"] >= similarity_threshold]

            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return _run_async_safe(self._async_get_stats())

    async def _async_get_stats(self) -> Dict[str, Any]:
        """Async get stats."""
        async with self.pool.acquire() as conn:
            stats_row = await conn.fetchrow(
                "SELECT * FROM metadata.vector_store_stats WHERE id = 1"
            )

            if not stats_row:
                # Fallback: compute stats manually
                layer1_count = await conn.fetchval("SELECT COUNT(*) FROM vectors.layer1")
                layer2_count = await conn.fetchval("SELECT COUNT(*) FROM vectors.layer2")
                layer3_count = await conn.fetchval("SELECT COUNT(*) FROM vectors.layer3")
                doc_count = await conn.fetchval(
                    "SELECT COUNT(DISTINCT document_id) FROM vectors.layer1"
                )

                return {
                    "documents": doc_count,
                    "total_vectors": layer1_count + layer2_count + layer3_count,
                    "layer1_count": layer1_count,
                    "layer2_count": layer2_count,
                    "layer3_count": layer3_count,
                    "dimensions": self.dimensions,
                    "backend": "postgresql",
                }
            else:
                return {
                    "documents": stats_row["document_count"],
                    "total_vectors": stats_row["total_vectors"],
                    "layer1_count": stats_row["layer1_count"],
                    "layer2_count": stats_row["layer2_count"],
                    "layer3_count": stats_row["layer3_count"],
                    "dimensions": stats_row["dimensions"],
                    "backend": "postgresql",
                }

    # ============================================================================
    # Properties - Metadata Access
    # ============================================================================

    @property
    def metadata_layer1(self) -> List[Dict]:
        """Lazily load Layer 1 metadata."""
        if self._metadata_cache[1] is None:
            self._metadata_cache[1] = _run_async_safe(self._load_metadata(layer=1))
        return self._metadata_cache[1]

    @property
    def metadata_layer2(self) -> List[Dict]:
        """Lazily load Layer 2 metadata."""
        if self._metadata_cache[2] is None:
            self._metadata_cache[2] = _run_async_safe(self._load_metadata(layer=2))
        return self._metadata_cache[2]

    @property
    def metadata_layer3(self) -> List[Dict]:
        """Lazily load Layer 3 metadata."""
        if self._metadata_cache[3] is None:
            self._metadata_cache[3] = _run_async_safe(self._load_metadata(layer=3))
        return self._metadata_cache[3]

    async def _load_metadata(self, layer: int) -> List[Dict]:
        """Load all metadata for a layer (expensive, use sparingly)."""
        async with self.pool.acquire() as conn:
            # Layer 1 (documents) doesn't have section_id, section_title
            # Layers 2-3 (sections/chunks) have these columns
            # All layers have hierarchical_path
            if layer == 1:
                sql = f"""
                    SELECT chunk_id, document_id, metadata, content, hierarchical_path
                    FROM vectors.layer{layer}
                    ORDER BY id
                """
            else:
                sql = f"""
                    SELECT chunk_id, document_id, metadata, content,
                           section_id, section_title, hierarchical_path
                    FROM vectors.layer{layer}
                    ORDER BY id
                """

            rows = await conn.fetch(sql)

            results = []
            for row in rows:
                result = {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "hierarchical_path": row.get("hierarchical_path"),
                }
                # Only add section columns for layers 2-3
                if layer > 1:
                    result["section_id"] = row.get("section_id")
                    result["section_title"] = row.get("section_title")

                # Merge JSONB metadata (PostgreSQL returns JSONB as dict)
                if row["metadata"] and isinstance(row["metadata"], dict):
                    result.update(row["metadata"])
                results.append(result)

            logger.info(f"Loaded {len(results)} metadata entries from layer {layer}")
            return results

    # ============================================================================
    # Conversation Ownership Management
    # ============================================================================

    async def create_conversation(
        self,
        conversation_id: str,
        user_id: int,
        title: Optional[str] = None
    ) -> None:
        """
        Create new conversation owned by user.

        Args:
            conversation_id: UUID string for conversation
            user_id: Owner user ID
            title: Optional conversation title
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO auth.conversations (id, user_id, title, created_at, updated_at)
                VALUES ($1, $2, $3, NOW(), NOW())
                """,
                conversation_id, user_id, title
            )
            logger.debug(f"Created conversation {conversation_id} for user {user_id}")

    async def get_user_conversations(
        self,
        user_id: int,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get all conversations for user (ordered by most recent).

        Args:
            user_id: User ID
            limit: Maximum conversations to return

        Returns:
            List of conversation dicts with id, title, created_at, updated_at
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, created_at, updated_at
                FROM auth.conversations
                WHERE user_id = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                user_id, limit
            )
            conversations = [dict(row) for row in rows]
            logger.debug(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations

    async def verify_conversation_ownership(
        self,
        conversation_id: str,
        user_id: int
    ) -> bool:
        """
        Check if user owns conversation.

        Args:
            conversation_id: Conversation ID
            user_id: User ID to check

        Returns:
            True if user owns conversation, False otherwise
        """
        async with self.pool.acquire() as conn:
            owner_id = await conn.fetchval(
                "SELECT user_id FROM auth.conversations WHERE id = $1",
                conversation_id
            )
            is_owner = owner_id == user_id
            logger.debug(
                f"Ownership check: conversation {conversation_id}, "
                f"user {user_id}, owner {owner_id}, result {is_owner}"
            )
            return is_owner

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get message history for conversation.

        Args:
            conversation_id: Conversation ID
            limit: Maximum messages to return

        Returns:
            List of message dicts with id, role, content, metadata, created_at
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, metadata, created_at
                FROM auth.messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                conversation_id, limit
            )
            # Convert rows to dicts and parse metadata JSON string back to dict
            import json
            messages = []
            for row in rows:
                msg = dict(row)
                # Parse metadata from JSON string to dict (asyncpg returns JSONB as string)
                if msg.get('metadata') and isinstance(msg['metadata'], str):
                    msg['metadata'] = json.loads(msg['metadata'])
                messages.append(msg)

            logger.debug(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Append message to conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata (tool calls, costs, etc.)

        Returns:
            Message ID
        """
        import json

        async with self.pool.acquire() as conn:
            # Convert metadata dict to JSON string for JSONB column
            metadata_json = json.dumps(metadata) if metadata else None

            # Insert message
            message_id = await conn.fetchval(
                """
                INSERT INTO auth.messages (conversation_id, role, content, metadata, created_at)
                VALUES ($1, $2, $3, $4::jsonb, NOW())
                RETURNING id
                """,
                conversation_id, role, content, metadata_json
            )

            # Update conversation updated_at timestamp
            await conn.execute(
                "UPDATE auth.conversations SET updated_at = NOW() WHERE id = $1",
                conversation_id
            )

            logger.debug(
                f"Appended {role} message (id={message_id}) to conversation {conversation_id}"
            )
            return message_id

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: int
    ) -> bool:
        """
        Delete conversation (with ownership check).

        Args:
            conversation_id: Conversation ID
            user_id: User ID (must be owner)

        Returns:
            True if deleted, False if not found or not owned
        """
        async with self.pool.acquire() as conn:
            # Verify ownership and delete in single transaction
            result = await conn.execute(
                "DELETE FROM auth.conversations WHERE id = $1 AND user_id = $2",
                conversation_id, user_id
            )
            # result is like "DELETE 1" or "DELETE 0"
            deleted = result.split()[-1] == "1"
            logger.debug(f"Delete conversation {conversation_id} for user {user_id}: {deleted}")
            return deleted

    # ============================================================================
    # Indexing: Add Chunks (for pipeline)
    # ============================================================================

    def add_chunks(
        self,
        chunks_dict: Dict[str, List],
        embeddings_dict: Dict[str, np.ndarray]
    ) -> None:
        """
        Add chunks and embeddings to PostgreSQL (batch INSERT with deduplication).

        Sync wrapper for async _async_add_chunks (pipeline is synchronous).

        Args:
            chunks_dict: Dict with keys 'layer1', 'layer2', 'layer3' containing Chunk objects
            embeddings_dict: Dict with keys 'layer1', 'layer2', 'layer3' containing embedding arrays

        Example:
            >>> adapter = PostgresVectorStoreAdapter(connection_string="...")
            >>> await adapter.initialize()
            >>> adapter.add_chunks(chunks_dict, embeddings_dict)
        """
        return _run_async_safe(self._async_add_chunks(chunks_dict, embeddings_dict))

    async def _async_add_chunks(
        self,
        chunks_dict: Dict[str, List],
        embeddings_dict: Dict[str, np.ndarray]
    ) -> None:
        """
        Async implementation of add_chunks.

        Uses batch INSERT with ON CONFLICT DO NOTHING for incremental indexing.
        """
        logger.info("Adding chunks to PostgreSQL...")

        # Add each layer
        for layer in [1, 2, 3]:
            layer_key = f"layer{layer}"
            chunks = chunks_dict.get(layer_key, [])
            embeddings = embeddings_dict.get(layer_key)

            if chunks and embeddings is not None and embeddings.size > 0:
                await self._add_layer_batch(layer, chunks, embeddings)
            else:
                logger.info(f"Layer {layer}: Skipped (empty layer)")

        # Update stats
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT metadata.update_vector_store_stats()")

        # Clear metadata cache (invalidate after adding new chunks)
        self._metadata_cache = {1: None, 2: None, 3: None}

        logger.info("✓ Chunks added to PostgreSQL")

    async def _add_layer_batch(
        self,
        layer: int,
        chunks: List,
        embeddings: np.ndarray
    ) -> None:
        """
        Add chunks to a specific layer using batch INSERT.

        Args:
            layer: Layer number (1, 2, or 3)
            chunks: List of Chunk objects
            embeddings: Embedding matrix (N x dimensions)
        """
        import json

        # Ensure embeddings are float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize embeddings for cosine similarity (pgvector requires unit vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1)

        # Prepare batch records
        records = []
        for i, chunk in enumerate(chunks):
            # Build hierarchical path
            hierarchical_path = chunk.metadata.document_id
            if chunk.metadata.section_path:
                hierarchical_path = f"{chunk.metadata.document_id} > {chunk.metadata.section_path}"

            # Convert embedding to pgvector string format
            embedding_str = self._vector_to_pgvector_string(embeddings[i])

            # Build metadata JSONB (flexible storage)
            metadata_json = json.dumps({
                "layer": layer,
                "section_path": chunk.metadata.section_path,
                "section_level": getattr(chunk.metadata, 'section_level', None),
                "section_depth": getattr(chunk.metadata, 'section_depth', None),
                "char_start": getattr(chunk.metadata, 'char_start', None),
                "char_end": getattr(chunk.metadata, 'char_end', None),
            })

            # Layer-specific record structure
            if layer == 1:
                # Layer 1: Documents (no section columns)
                record = (
                    chunk.chunk_id,
                    chunk.metadata.document_id,
                    chunk.metadata.section_title or chunk.metadata.document_id,  # title
                    embedding_str,
                    chunk.raw_content,  # content without SAC
                    chunk.metadata.cluster_id,
                    chunk.metadata.cluster_label,
                    chunk.metadata.cluster_confidence,
                    hierarchical_path,
                    chunk.metadata.page_number,
                    metadata_json,
                )
            else:
                # Layer 2/3: Sections/Chunks (have section columns)
                record = (
                    chunk.chunk_id,
                    chunk.metadata.document_id,
                    chunk.metadata.section_id,
                    chunk.metadata.section_title,
                    chunk.metadata.section_path,
                    getattr(chunk.metadata, 'section_level', None),
                    getattr(chunk.metadata, 'section_depth', None),
                    hierarchical_path,
                    chunk.metadata.page_number,
                    getattr(chunk.metadata, 'char_start', None) if layer == 3 else None,
                    getattr(chunk.metadata, 'char_end', None) if layer == 3 else None,
                    embedding_str,
                    chunk.raw_content,
                    chunk.metadata.cluster_id,
                    chunk.metadata.cluster_label,
                    chunk.metadata.cluster_confidence,
                    metadata_json,
                )

            records.append(record)

        # Batch INSERT with ON CONFLICT DO NOTHING (deduplication)
        async with self.pool.acquire() as conn:
            if layer == 1:
                sql = """
                    INSERT INTO vectors.layer1
                    (chunk_id, document_id, title, embedding, content,
                     cluster_id, cluster_label, cluster_confidence,
                     hierarchical_path, page_number, metadata)
                    VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10, $11::jsonb)
                    ON CONFLICT (chunk_id) DO NOTHING
                """
            elif layer == 2:
                sql = """
                    INSERT INTO vectors.layer2
                    (chunk_id, document_id, section_id, section_title, section_path,
                     section_level, section_depth, hierarchical_path, page_number,
                     embedding, content, cluster_id, cluster_label, cluster_confidence, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11, $12, $13, $14, $15::jsonb)
                    ON CONFLICT (chunk_id) DO NOTHING
                """
            else:  # layer == 3
                sql = """
                    INSERT INTO vectors.layer3
                    (chunk_id, document_id, section_id, section_title, section_path,
                     section_level, section_depth, hierarchical_path, page_number,
                     char_start, char_end, embedding, content,
                     cluster_id, cluster_label, cluster_confidence, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::vector, $13, $14, $15, $16, $17::jsonb)
                    ON CONFLICT (chunk_id) DO NOTHING
                """

            # Execute batch insert
            await conn.executemany(sql, records)

        logger.info(f"Layer {layer}: Added {len(records)} chunks to PostgreSQL")

    # ============================================================================
    # Cleanup
    # ============================================================================

    def __del__(self):
        """Close connection pool on destruction."""
        if self.pool:
            try:
                # Try graceful close - may fail during Python shutdown
                _run_async_safe(self.pool.close())
            except Exception:
                # Ignore errors during destruction (Python may be shutting down)
                pass

# ====================================================================================
# PostgreSQL Storage Adapter for Authentication & User Data
# ====================================================================================

class PostgreSQLStorageAdapter:
    """
    PostgreSQL connection pool for auth/user data.
    
    Separate from PostgresVectorStoreAdapter (vector search).
    This adapter provides connection pooling for:
    - auth.users (user accounts)
    - auth.conversations (chat sessions)
    - auth.messages (message history)
    
    Used by AuthQueries for user management and conversation storage.
    """
    
    def __init__(self):
        """Initialize adapter (connection pool created in initialize())."""
        import os
        self.pool: Optional[asyncpg.Pool] = None
        self.db_url = os.getenv("DATABASE_URL")
        
        if not self.db_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Set in .env file (e.g., postgresql://postgres:password@postgres:5432/sujbot)"
            )
    
    async def initialize(self):
        """Create connection pool."""
        logger.info("Creating PostgreSQL connection pool for auth/user storage...")
        
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        
        # Test connection
        async with self.pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info(f"PostgreSQL connected: {version}")
        
        logger.info("✓ Connection pool created (2-10 connections)")
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    # ============================================================================
    # Conversation Management Methods
    # ============================================================================

    async def create_conversation(
        self,
        conversation_id: str,
        user_id: int,
        title: Optional[str] = None
    ) -> None:
        """
        Create new conversation owned by user.

        Args:
            conversation_id: UUID string for conversation
            user_id: Owner user ID
            title: Optional conversation title
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO auth.conversations (id, user_id, title, created_at, updated_at)
                VALUES ($1, $2, $3, NOW(), NOW())
                """,
                conversation_id, user_id, title
            )
            logger.debug(f"Created conversation {conversation_id} for user {user_id}")

    async def get_user_conversations(
        self,
        user_id: int,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get all conversations for user (ordered by most recent).

        Args:
            user_id: User ID
            limit: Maximum conversations to return

        Returns:
            List of conversation dicts with id, title, created_at, updated_at
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, created_at, updated_at
                FROM auth.conversations
                WHERE user_id = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                user_id, limit
            )
            conversations = [dict(row) for row in rows]
            logger.debug(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations

    async def verify_conversation_ownership(
        self,
        conversation_id: str,
        user_id: int
    ) -> bool:
        """
        Check if user owns conversation.

        Args:
            conversation_id: Conversation ID
            user_id: User ID to check

        Returns:
            True if user owns conversation, False otherwise
        """
        async with self.pool.acquire() as conn:
            owner_id = await conn.fetchval(
                "SELECT user_id FROM auth.conversations WHERE id = $1",
                conversation_id
            )
            is_owner = owner_id == user_id
            logger.debug(
                f"Ownership check: conversation {conversation_id}, "
                f"user {user_id}, owner {owner_id}, result {is_owner}"
            )
            return is_owner

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get message history for conversation.

        Args:
            conversation_id: Conversation ID
            limit: Maximum messages to return

        Returns:
            List of message dicts with id, role, content, metadata, created_at
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, metadata, created_at
                FROM auth.messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                conversation_id, limit
            )
            # Convert rows to dicts and parse metadata JSON string back to dict
            import json
            messages = []
            for row in rows:
                msg = dict(row)
                # Parse metadata from JSON string to dict (asyncpg returns JSONB as string)
                if msg.get('metadata') and isinstance(msg['metadata'], str):
                    msg['metadata'] = json.loads(msg['metadata'])
                messages.append(msg)

            logger.debug(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Append message to conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata (tool calls, costs, etc.)

        Returns:
            Message ID
        """
        import json

        async with self.pool.acquire() as conn:
            # Convert metadata dict to JSON string for JSONB column
            metadata_json = json.dumps(metadata) if metadata else None

            # Insert message
            message_id = await conn.fetchval(
                """
                INSERT INTO auth.messages (conversation_id, role, content, metadata, created_at)
                VALUES ($1, $2, $3, $4::jsonb, NOW())
                RETURNING id
                """,
                conversation_id, role, content, metadata_json
            )

            # Update conversation updated_at timestamp
            await conn.execute(
                "UPDATE auth.conversations SET updated_at = NOW() WHERE id = $1",
                conversation_id
            )

            logger.debug(
                f"Appended {role} message (id={message_id}) to conversation {conversation_id}"
            )
            return message_id

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: int
    ) -> bool:
        """
        Delete conversation (with ownership check).

        Args:
            conversation_id: Conversation ID
            user_id: User ID (must be owner)

        Returns:
            True if deleted, False if not found or not owned
        """
        async with self.pool.acquire() as conn:
            # Verify ownership and delete in single transaction
            result = await conn.execute(
                "DELETE FROM auth.conversations WHERE id = $1 AND user_id = $2",
                conversation_id, user_id
            )
            # result is like "DELETE 1" or "DELETE 0"
            deleted = result.split()[-1] == "1"
            logger.debug(f"Delete conversation {conversation_id} for user {user_id}: {deleted}")
            return deleted
