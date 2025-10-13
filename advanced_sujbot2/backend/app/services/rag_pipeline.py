"""
RAG Pipeline Service - Coordinates all RAG components for production use.

This service manages:
- Component initialization (lazy loading)
- Document indexing pipeline
- Query processing and retrieval
- Cross-document retrieval
- Knowledge graph operations
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from app.rag.config import Config
from app.rag.document_reader import LegalDocumentReader
from app.rag.chunker import LawCodeChunker
from app.rag.embeddings import LegalEmbedder
from app.rag.pg_vector_store import PostgreSQLVectorStore
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.cross_doc_retrieval import ComparativeRetriever
from app.rag.reranker import RerankingPipeline
from app.rag.knowledge_graph import LegalKnowledgeGraph
from app.rag.exceptions import (
    DocumentProcessingError,
    IndexingError,
    RetrievalError,
    ConfigurationError
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production RAG pipeline that orchestrates all components.

    Features:
    - Lazy initialization of heavy models (embeddings, reranker)
    - Progress tracking for long-running operations
    - Error handling and recovery
    - Component reuse across requests
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG pipeline.

        Args:
            config: Optional configuration dict (uses defaults if not provided)
        """
        # Load configuration
        if config:
            self.config = Config(config_dict=config)
        else:
            # Use default config from app/rag/config.py
            self.config = Config()

        # Override with backend settings
        self._apply_backend_settings()

        # Initialize lightweight components
        self.document_reader = LegalDocumentReader()

        # Create ChunkingConfig from main config
        from app.rag.chunker import ChunkingConfig
        chunking_config = ChunkingConfig(
            chunk_size=512,
            min_chunk_size=128,
            max_chunk_size=1024,
            strategy='hierarchical_legal',
            law_chunk_by='paragraph',
            law_include_context=True,
            law_aggregate_small=True,
            law_split_large=True
        )
        self.chunker = LawCodeChunker(chunking_config)

        # Heavy components - lazy load on first use
        self._embedding_service: Optional[LegalEmbedder] = None
        self._vector_store: Optional[PostgreSQLVectorStore] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._cross_doc_retriever: Optional[ComparativeRetriever] = None
        self._reranker: Optional[RerankingPipeline] = None
        self._knowledge_graph: Optional[LegalKnowledgeGraph] = None
        # Note: ComplianceAnalyzer removed - use AdvancedCompliancePipeline directly

        # Index directory
        self.index_dir = Path(settings.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        logger.info("RAG pipeline initialized")

    def _apply_backend_settings(self):
        """Apply backend settings to RAG config."""
        # Override Claude API settings
        if settings.CLAUDE_API_KEY:
            os.environ["CLAUDE_API_KEY"] = settings.CLAUDE_API_KEY

        # Override model settings
        self.config.set("claude.main_model", settings.MAIN_AGENT_MODEL)
        self.config.set("claude.subagent_model", settings.SUBAGENT_MODEL)

        # Override logging
        if settings.VERBOSE_LOGGING:
            self.config.set("api.logging.level", "DEBUG")

    @property
    def embedding_service(self) -> LegalEmbedder:
        """Lazy load embedding service."""
        if self._embedding_service is None:
            logger.info("Initializing embedding service...")
            from app.rag.embeddings import EmbeddingConfig
            embedding_config = EmbeddingConfig(
                model_name="joelniklaus/legal-xlm-roberta-base",  # 768-dim legal multilingual (trained on MultiLegalPile with Czech)
                device="auto",  # Auto-detect: CUDA > MPS > CPU
                batch_size=32,
                max_sequence_length=512,  # RoBERTa limit
                normalize=True,
                add_hierarchical_context=True
            )
            self._embedding_service = LegalEmbedder(embedding_config)
            logger.info("Embedding service ready")
        return self._embedding_service

    @property
    def vector_store(self) -> PostgreSQLVectorStore:
        """Lazy load PostgreSQL vector store."""
        if self._vector_store is None:
            logger.info("Initializing PostgreSQL vector store...")
            from app.rag.pg_vector_store import PostgreSQLConfig

            # Load PostgreSQL config from environment
            pg_config = PostgreSQLConfig(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "sujbot2"),
                user=os.getenv("POSTGRES_USER", "sujbot_app"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
                min_pool_size=5,
                max_pool_size=20,
                vector_search_probes=10,
                enable_query_cache=True
            )

            self._vector_store = PostgreSQLVectorStore(
                embedder=self.embedding_service,
                config=pg_config
            )

            # Note: Document info loading is deferred to first async operation
            # This avoids asyncio.run() from within an event loop
            logger.info("PostgreSQL vector store ready (document loading deferred)")
        return self._vector_store

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        """Lazy load hybrid retriever."""
        if self._hybrid_retriever is None:
            logger.info("Initializing hybrid retriever...")

            # Import necessary components
            from app.rag.hybrid_retriever import (
                SemanticSearcher,
                KeywordSearcher,
                StructuralSearcher,
                RetrievalConfig
            )

            # Create retrieval configuration
            retrieval_config = RetrievalConfig(
                semantic_weight=0.5,
                keyword_weight=0.3,
                structural_weight=0.2,
                top_k=20,
                bm25_k1=1.5,
                bm25_b=0.75,
                normalize_scores=True,
                enable_caching=True,
                parallel_retrieval=True
            )

            # Create component searchers
            semantic_searcher = SemanticSearcher(
                self.embedding_service,
                self.vector_store
            )
            keyword_searcher = KeywordSearcher(
                self.vector_store,
                k1=retrieval_config.bm25_k1,
                b=retrieval_config.bm25_b
            )
            structural_searcher = StructuralSearcher(self.vector_store)

            # Create hybrid retriever with all required components
            self._hybrid_retriever = HybridRetriever(
                semantic_searcher,
                keyword_searcher,
                structural_searcher,
                retrieval_config
            )
            logger.info("Hybrid retriever ready")
        return self._hybrid_retriever

    @property
    def cross_doc_retriever(self) -> ComparativeRetriever:
        """Lazy load cross-document retriever."""
        if self._cross_doc_retriever is None:
            logger.info("Initializing cross-document retriever...")

            # Create empty reference map (will be populated when documents are loaded)
            from app.rag.indexing import ReferenceMap
            reference_map = ReferenceMap()

            # Create cross-document retriever with correct arguments
            self._cross_doc_retriever = ComparativeRetriever(
                vector_store=self.vector_store,
                embedder=self.embedding_service,
                reference_map=reference_map,
                config={}
            )
            logger.info("Cross-document retriever ready")
        return self._cross_doc_retriever

    @property
    def reranker(self) -> RerankingPipeline:
        """Lazy load reranker."""
        if self._reranker is None:
            logger.info("Initializing reranker...")
            from app.rag.reranker import RerankingConfig

            # Create RerankingConfig from main config
            reranking_config = RerankingConfig(
                cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Smaller, faster model (already cached)
                cross_encoder_batch_size=16,
                cross_encoder_device="auto",  # Auto-detect: CUDA > MPS > CPU
                cross_encoder_max_length=512,
                enable_graph_reranking=False,  # Disable for now (no knowledge graph ready yet)
                enable_precedence_weighting=True,
                ensemble_method="weighted_average",
                final_top_k=5,
                min_confidence_threshold=0.1,
                explain_reranking=True
            )

            self._reranker = RerankingPipeline(reranking_config, knowledge_graph=None)
            logger.info("Reranker ready")
        return self._reranker

    @property
    def knowledge_graph(self) -> LegalKnowledgeGraph:
        """Lazy load knowledge graph."""
        if self._knowledge_graph is None:
            logger.info("Initializing knowledge graph...")
            self._knowledge_graph = LegalKnowledgeGraph(config=self.config)
            logger.info("Knowledge graph ready")
        return self._knowledge_graph

    # NOTE: Old ComplianceAnalyzer removed
    # Use AdvancedCompliancePipeline directly instead (see chat_service.py)

    async def index_document(
        self,
        document_path: str,
        document_id: str,
        document_type: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Index a document with full RAG pipeline.

        Args:
            document_path: Path to document file
            document_id: Unique document identifier
            document_type: 'contract' | 'law_code' | 'regulation'
            progress_callback: Optional callback(current, total, message)

        Returns:
            Metadata dict with indexing results

        Raises:
            DocumentProcessingError: If document cannot be read
            IndexingError: If indexing fails
        """
        try:
            doc_path = Path(document_path)
            if not doc_path.exists():
                raise DocumentProcessingError(f"Document not found: {document_path}")

            # Progress tracking helper
            def update_progress(current: int, total: int, message: str):
                if progress_callback:
                    progress_callback(current, total, message)
                logger.info(f"[{current}/{total}] {message}")

            update_progress(0, 100, "Starting indexing...")
            update_progress(5, 100, "Initializing document reader...")

            # Step 1: Read document
            update_progress(10, 100, "Reading document...")
            update_progress(15, 100, "Parsing document structure...")
            doc = await self.document_reader.read_legal_document(str(doc_path), document_type)
            update_progress(22, 100, "Extracting text content...")

            # Convert LegalDocument to metadata dict
            # Note: Keep structure and references as objects to avoid recursion issues
            doc_metadata = {
                "content": doc.cleaned_text,
                "raw_content": doc.raw_text,
                "document_type": doc.document_type,
                "structure": doc.structure,  # Keep as object
                "references": doc.references,  # Keep as objects
                "title": doc.metadata.title,
                "file_format": doc.metadata.file_format,
                "page_count": doc.metadata.total_pages,
                "word_count": doc.metadata.total_words,
                "section_count": doc.metadata.total_sections,
            }

            if not doc_metadata:
                raise DocumentProcessingError("Failed to extract document metadata")

            update_progress(28, 100, "Document read successfully")

            # Step 2: Chunk document
            update_progress(32, 100, "Initializing chunker...")
            update_progress(35, 100, "Chunking document...")
            chunks = await self.chunker.chunk(doc)
            update_progress(48, 100, "Analyzing chunk boundaries...")

            # Fallback: if no chunks created, use simple text-based chunking
            if len(chunks) == 0:
                logger.warning("No structured chunks created, using simple text chunking fallback")
                from app.rag.chunker import LegalChunk

                # Simple chunking: split by paragraphs
                text = doc.cleaned_text
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

                for i, para in enumerate(paragraphs):
                    if len(para) > 50:  # Skip very short paragraphs
                        chunks.append(LegalChunk(
                            chunk_id=f"{document_id}_chunk_{i}",
                            document_id=document_id,
                            document_type=document_type,
                            content=para,
                            chunk_index=i,
                            hierarchy_path="",
                            legal_reference=f"Paragraph {i+1}",
                            structural_level="paragraph",
                            metadata={"simple_chunking": True, "page_estimate": i // 5 + 1}
                        ))

            chunk_count = len(chunks)
            logger.info(f"Generated {chunk_count} chunks")
            update_progress(52, 100, f"Generated {chunk_count} chunks")

            # Step 3: Generate embeddings and build index
            update_progress(55, 100, "Initializing embedding model...")

            # Create simple progress callback for chunk batch processing
            # Map embedding progress from 55% to 85% (30% of total time)
            EMBEDDING_START = 55
            EMBEDDING_END = 85
            EMBEDDING_RANGE = EMBEDDING_END - EMBEDDING_START

            def embedding_batch_progress(chunks_processed, total_chunks):
                """
                Track progress through chunk batches with smooth linear progression.

                Args:
                    chunks_processed: Number of chunks processed so far
                    total_chunks: Total number of chunks to process
                """
                if total_chunks == 0:
                    update_progress(EMBEDDING_END, 100, "No chunks to process")
                    return

                # Calculate linear progress from 55% to 85%
                chunk_progress = chunks_processed / total_chunks
                overall_progress = EMBEDDING_START + int(chunk_progress * EMBEDDING_RANGE)
                overall_progress = min(overall_progress, EMBEDDING_END)  # Cap at 85%

                # Create descriptive message
                message = f"Generating embeddings: {chunks_processed}/{total_chunks} chunks"

                update_progress(overall_progress, 100, message)

            # Add chunks to vector store with simple progress tracking
            await self.vector_store.add_document(
                document_id=document_id,
                document_type=document_type,
                chunks=chunks,
                progress_callback=embedding_batch_progress
            )
            update_progress(85, 100, "Embeddings generated and indexed")

            # PostgreSQL persists automatically, no separate save step needed
            update_progress(88, 100, "Data persisted to PostgreSQL")
            logger.info(f"Document indexed in PostgreSQL: {document_id}")
            update_progress(93, 100, "Index built and saved successfully")
            update_progress(96, 100, "Finalizing...")

            index_metadata = {
                "vector_count": len(chunks),
                "storage_backend": "postgresql"
            }

            # Step 4: Build knowledge graph (if applicable)
            if document_type in ["law_code", "regulation"]:
                update_progress(97, 100, "Building knowledge graph...")
                try:
                    self.knowledge_graph.add_document(
                        document_id=document_id,
                        chunks=chunks,
                        document_type=document_type
                    )
                except Exception as e:
                    logger.warning(f"Knowledge graph build failed: {e}")

            update_progress(100, 100, "Indexing complete")

            # Return comprehensive metadata
            return {
                "document_id": document_id,
                "document_type": document_type,
                "page_count": doc_metadata.get("page_count", 0),
                "word_count": doc_metadata.get("word_count", 0),
                "char_count": doc_metadata.get("char_count", 0),
                "chunk_count": chunk_count,
                "indexed_at": datetime.now().isoformat(),
                **index_metadata
            }

        except Exception as e:
            logger.error(f"Indexing failed for {document_id}: {e}")
            raise IndexingError(f"Failed to index document: {e}") from e

    async def query(
        self,
        query: str,
        document_ids: List[str],
        top_k: Optional[int] = None,
        rerank: bool = True,
        language: str = "cs"
    ) -> Dict[str, Any]:
        """
        Query documents with hybrid retrieval.

        Args:
            query: User query
            document_ids: List of document IDs to search
            top_k: Number of results to return (uses config default if None)
            rerank: Whether to apply reranking
            language: Query language ('cs' or 'en')

        Returns:
            Query results with sources and metadata

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Load document info from PostgreSQL if not already loaded
            if not self.vector_store.document_info:
                logger.info("Loading document info from PostgreSQL...")
                await self.vector_store.load_document_info_from_db()

            # Verify that documents are indexed in PostgreSQL
            available_docs = []
            for doc_id in document_ids:
                if doc_id in self.vector_store.document_info:
                    available_docs.append(doc_id)
                else:
                    logger.warning(f"Document not found in database: {doc_id}")

            if not available_docs:
                raise RetrievalError("No indexed documents found")

            # Retrieve relevant chunks using HybridRetriever.search()
            logger.info(f"Retrieving for query: {query[:100]}...")
            results = await self.hybrid_retriever.search(
                query=query,
                document_ids=available_docs,
                top_k=top_k or self.config.get("retrieval.rerank_top_k", 5)
            )

            # Apply reranking if requested
            if rerank and self.config.get("retrieval.enable_reranking", True):
                logger.info("Applying reranking...")
                # Convert results to SearchResult objects for reranker
                from app.rag.reranker import SearchResult as RerankSearchResult

                search_results = []
                for i, result in enumerate(results):
                    # SearchResult is a dataclass with attributes, not a dict
                    search_result = RerankSearchResult(
                        chunk_id=result.chunk_id,
                        content=result.chunk.content,
                        legal_reference=result.chunk.legal_reference or "",
                        document_id=result.document_id,
                        document_type=result.chunk.document_type,
                        hierarchy_path=result.chunk.hierarchy_path or "",
                        rank=i+1,
                        hybrid_score=result.score,
                        metadata=result.chunk.metadata or {}
                    )
                    search_results.append(search_result)

                # Run reranking
                reranked_results = await self.reranker.rerank(
                    query=query,
                    initial_results=search_results,
                    query_context=None
                )

                # Convert RankedResult objects back to dict format
                results = []
                for ranked_result in reranked_results:
                    results.append({
                        "chunk_id": ranked_result.chunk_id,
                        "content": ranked_result.content,
                        "score": ranked_result.scores.ensemble_score,
                        "document_id": ranked_result.document_id,
                        "metadata": {
                            "section": ranked_result.legal_reference,
                            "document_type": ranked_result.document_type,
                            "hierarchy_path": "",
                            "rank": ranked_result.final_rank,
                            "confidence": ranked_result.confidence,
                            "reranking_explanation": ranked_result.reranking_explanation
                        }
                    })

            return {
                "query": query,
                "results": results,
                "document_count": len(document_ids),
                "total_results": len(results),
                "language": language,
                "reranked": rerank
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise RetrievalError(f"Failed to execute query: {e}") from e

    def cross_document_query(
        self,
        query: str,
        contract_ids: List[str],
        law_ids: List[str],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query across contracts and laws with cross-document linking.

        Args:
            query: User query
            contract_ids: List of contract document IDs
            law_ids: List of law/regulation document IDs
            top_k: Number of results per document type

        Returns:
            Cross-document query results with provision links
        """
        try:
            # Verify documents exist in database
            contract_indexes = [
                cid for cid in contract_ids
                if cid in self.vector_store.document_info
            ]
            law_indexes = [
                lid for lid in law_ids
                if lid in self.vector_store.document_info
            ]

            # Execute cross-document retrieval
            results = self.cross_doc_retriever.retrieve_cross_document(
                query=query,
                contract_indexes=contract_indexes,
                law_indexes=law_indexes,
                top_k=top_k or self.config.get("retrieval.rerank_top_k", 5)
            )

            return {
                "query": query,
                "results": results,
                "contract_count": len(contract_ids),
                "law_count": len(law_ids)
            }

        except Exception as e:
            logger.error(f"Cross-document query failed: {e}")
            raise RetrievalError(f"Failed to execute cross-document query: {e}") from e

    async def reload_document_index(self, document_id: str) -> bool:
        """
        Reload document metadata from PostgreSQL into cache.

        This is useful after a document has been indexed by a Celery worker,
        so the backend can immediately use the new document without restarting.

        Args:
            document_id: Document ID to reload

        Returns:
            True if successfully reloaded, False otherwise
        """
        try:
            logger.info(f"Reloading document info for {document_id} from PostgreSQL...")

            # Small delay to ensure connection pool is released from previous operations
            import asyncio
            await asyncio.sleep(0.5)

            # Reload document info from database
            await self.vector_store.load_document_info_from_db()

            if document_id in self.vector_store.document_info:
                doc_info = self.vector_store.document_info[document_id]
                logger.info(f"Successfully reloaded {document_id}: {doc_info.get('num_chunks', 0)} chunks")
                return True
            else:
                logger.warning(f"Document {document_id} not found in database")
                return False

        except Exception as e:
            logger.error(f"Failed to reload document info for {document_id}: {e}")
            return False

    # NOTE: Old analyze_compliance method removed
    # Use AdvancedCompliancePipeline directly instead (see chat_service.py or tasks/compliance.py)


# Global singleton instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


# Add missing import
import os
