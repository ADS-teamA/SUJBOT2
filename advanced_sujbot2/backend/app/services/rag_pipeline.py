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
from app.rag.indexing import MultiDocumentVectorStore
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.cross_doc_retrieval import ComparativeRetriever
from app.rag.reranker import RerankingPipeline
from app.rag.knowledge_graph import LegalKnowledgeGraph
from app.rag.compliance_analyzer import ComplianceAnalyzer
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
        self._vector_store: Optional[MultiDocumentVectorStore] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._cross_doc_retriever: Optional[ComparativeRetriever] = None
        self._reranker: Optional[RerankingPipeline] = None
        self._knowledge_graph: Optional[LegalKnowledgeGraph] = None
        self._compliance_analyzer: Optional[ComplianceAnalyzer] = None

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
                model_name="BAAI/bge-m3",
                device="cpu",
                batch_size=32,
                max_sequence_length=8192,
                normalize=True,
                add_hierarchical_context=True
            )
            self._embedding_service = LegalEmbedder(embedding_config)
            logger.info("Embedding service ready")
        return self._embedding_service

    @property
    def vector_store(self) -> MultiDocumentVectorStore:
        """Lazy load vector store."""
        if self._vector_store is None:
            logger.info("Initializing vector store...")
            from app.rag.indexing import VectorStoreConfig
            vector_config = VectorStoreConfig(
                index_type="flat",
                vector_size=1024,
                enable_gpu=False
            )
            self._vector_store = MultiDocumentVectorStore(
                embedder=self.embedding_service,
                config=vector_config
            )
            logger.info("Vector store ready")
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
            self._cross_doc_retriever = ComparativeRetriever(
                self.config,
                self.hybrid_retriever
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
                cross_encoder_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                cross_encoder_batch_size=16,
                cross_encoder_device="cpu",
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
            self._knowledge_graph = LegalKnowledgeGraph(self.config)
            logger.info("Knowledge graph ready")
        return self._knowledge_graph

    @property
    def compliance_analyzer(self) -> ComplianceAnalyzer:
        """Lazy load compliance analyzer."""
        if self._compliance_analyzer is None:
            logger.info("Initializing compliance analyzer...")
            self._compliance_analyzer = ComplianceAnalyzer(
                self.config,
                self.hybrid_retriever
            )
            logger.info("Compliance analyzer ready")
        return self._compliance_analyzer

    def index_document(
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

            # Step 1: Read document
            update_progress(10, 100, "Reading document...")
            import asyncio
            doc = asyncio.run(self.document_reader.read_legal_document(str(doc_path), document_type))

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

            update_progress(20, 100, "Document read successfully")

            # Step 2: Chunk document
            update_progress(30, 100, "Chunking document...")
            chunks = asyncio.run(self.chunker.chunk(doc))

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
            update_progress(50, 100, f"Generated {chunk_count} chunks")

            # Step 3: Generate embeddings and build index
            update_progress(60, 100, "Generating embeddings...")
            index_path = self.index_dir / document_id
            index_path.mkdir(parents=True, exist_ok=True)

            # Add chunks to vector store
            asyncio.run(self.vector_store.add_document(
                document_id=document_id,
                document_type=document_type,
                chunks=chunks
            ))
            update_progress(80, 100, "Embeddings generated")

            # Persist index (handled by add_document internally, skip for now)
            update_progress(95, 100, "Index built successfully")

            index_metadata = {
                "vector_count": len(chunks),
                "index_size": sum(f.stat().st_size for f in index_path.glob("*") if f.is_file())
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
                "index_path": str(index_path),
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
            # Verify that documents are indexed
            available_docs = []
            for doc_id in document_ids:
                index_path = self.index_dir / doc_id
                if index_path.exists():
                    available_docs.append(doc_id)
                else:
                    logger.warning(f"Index not found for document: {doc_id}")

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
                    search_result = RerankSearchResult(
                        chunk_id=result.get("chunk_id", f"chunk_{i}"),
                        content=result.get("content", ""),
                        legal_reference=result.get("metadata", {}).get("section", ""),
                        document_id=result.get("document_id", ""),
                        document_type=result.get("metadata", {}).get("document_type", "law_code"),
                        hierarchy_path=result.get("metadata", {}).get("hierarchy_path", ""),
                        rank=i+1,
                        hybrid_score=result.get("score", 0.0),
                        metadata=result.get("metadata", {})
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
            # Prepare index paths
            contract_indexes = [
                str(self.index_dir / cid) for cid in contract_ids
                if (self.index_dir / cid).exists()
            ]
            law_indexes = [
                str(self.index_dir / lid) for lid in law_ids
                if (self.index_dir / lid).exists()
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

    def analyze_compliance(
        self,
        contract_id: str,
        law_ids: List[str],
        mode: str = "exhaustive"
    ) -> Dict[str, Any]:
        """
        Analyze contract compliance against laws.

        Args:
            contract_id: Contract document ID
            law_ids: List of law/regulation document IDs
            mode: Analysis mode ('quick' or 'exhaustive')

        Returns:
            Compliance analysis results
        """
        try:
            contract_index = str(self.index_dir / contract_id)
            law_indexes = [
                str(self.index_dir / lid) for lid in law_ids
                if (self.index_dir / lid).exists()
            ]

            if not Path(contract_index).exists():
                raise RetrievalError(f"Contract index not found: {contract_id}")

            if not law_indexes:
                raise RetrievalError("No law indexes found")

            # Run compliance analysis
            results = self.compliance_analyzer.analyze(
                contract_index=contract_index,
                law_indexes=law_indexes,
                mode=mode
            )

            return results

        except Exception as e:
            logger.error(f"Compliance analysis failed: {e}")
            raise


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
