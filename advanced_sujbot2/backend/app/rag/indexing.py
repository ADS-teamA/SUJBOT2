"""
Multi-Document Vector Store - Legal Document Indexing System
Manages separate FAISS indices for each document with cross-document references
"""

import asyncio
import json
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is required for indexing. Install with: "
        "pip install faiss-cpu (or faiss-gpu for CUDA support)"
    )

from .embeddings import LegalEmbedder, LegalChunk, EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""

    # Index type
    index_type: str = "flat"  # flat | ivf | hnsw
    vector_size: int = 1024  # BGE-M3 dimension

    # IVF settings
    ivf_nlist: int = 100  # Number of clusters
    ivf_nprobe: int = 10  # Number of clusters to search

    # HNSW settings
    hnsw_m: int = 32  # Number of connections
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50

    # Performance
    enable_gpu: bool = False


@dataclass
class SearchResult:
    """Single search result"""

    chunk_id: str
    chunk: LegalChunk
    score: float
    document_id: str
    rank: int = 0


class ReferenceMap:
    """
    Maps legal references across documents
    Enables fast lookup: "§89" → list of chunks
    """

    def __init__(self):
        """Initialize empty reference maps"""
        self.ref_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self.chunk_to_refs: Dict[str, List[str]] = {}

    async def build(self, chunks: List[LegalChunk]):
        """
        Build reference map from chunks

        Args:
            chunks: List of legal chunks with references
        """
        for chunk in chunks:
            # Index by legal reference
            ref = chunk.legal_reference
            if ref:
                self.ref_to_chunks[ref].append(chunk.chunk_id)

            # Index outgoing references
            outgoing = chunk.metadata.get('references_to', [])
            if outgoing:
                self.chunk_to_refs[chunk.chunk_id] = outgoing

        logger.info(
            f"Built reference map: {len(self.ref_to_chunks)} references, "
            f"{len(self.chunk_to_refs)} chunks with outgoing refs"
        )

    def get_chunks_by_reference(self, legal_ref: str) -> List[str]:
        """
        Get all chunks matching a legal reference

        Args:
            legal_ref: Legal reference (e.g., "§89", "Článek 5")

        Returns:
            List of chunk IDs
        """
        return self.ref_to_chunks.get(legal_ref, [])

    def get_references_from_chunk(self, chunk_id: str) -> List[str]:
        """
        Get all references cited by a chunk

        Args:
            chunk_id: Chunk identifier

        Returns:
            List of referenced legal references
        """
        return self.chunk_to_refs.get(chunk_id, [])

    def serialize(self) -> Dict[str, Any]:
        """Serialize reference map for storage"""
        return {
            'ref_to_chunks': dict(self.ref_to_chunks),
            'chunk_to_refs': self.chunk_to_refs
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ReferenceMap':
        """Deserialize reference map from storage"""
        ref_map = cls()
        ref_map.ref_to_chunks = defaultdict(list, data.get('ref_to_chunks', {}))
        ref_map.chunk_to_refs = data.get('chunk_to_refs', {})
        return ref_map


class MultiDocumentVectorStore:
    """
    Separate FAISS indices for each document type
    Enables filtering and cross-document comparison
    """

    def __init__(
        self,
        embedder: LegalEmbedder,
        config: Optional[VectorStoreConfig] = None
    ):
        """
        Initialize multi-document vector store

        Args:
            embedder: Legal embedder instance
            config: Vector store configuration
        """
        self.embedder = embedder
        self.config = config or VectorStoreConfig()

        # Storage
        self.indices: Dict[str, faiss.Index] = {}
        self.metadata_stores: Dict[str, Dict[str, LegalChunk]] = {}
        self.reference_map = ReferenceMap()

        # Document metadata
        self.document_info: Dict[str, Dict[str, Any]] = {}

    async def add_document(
        self,
        chunks: List[LegalChunk],
        document_id: str,
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ):
        """
        Add document to store with batch processing for memory efficiency

        Args:
            chunks: List of legal chunks
            document_id: Unique document identifier
            document_type: Type of document (law_code, contract, etc.)
            metadata: Additional document metadata
            batch_size: Number of chunks to process at once (default: 100)
            progress_callback: Optional callback(chunks_processed, total_chunks)
        """
        if not chunks:
            logger.warning(f"No chunks provided for document {document_id}")
            return

        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        logger.info(f"Adding document {document_id} with {total_chunks} chunks in {total_batches} batches (batch_size={batch_size})")

        # 1. Create/get index for this document
        if document_id not in self.indices:
            self.indices[document_id] = self._create_faiss_index()

        # 2. Process chunks in batches to avoid OOM
        import numpy as np
        all_embeddings = []

        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1

            logger.info(f"Processing batch {batch_num}/{total_batches}: chunks {batch_start}-{batch_end-1}")

            # Generate embeddings for this batch (no sub-batch progress for smoother UI)
            batch_embeddings = await self.embedder.embed_chunks(
                batch_chunks,
                progress_callback=None  # Don't pass embedding-level progress to avoid flickering
            )

            # Add to index immediately
            self.indices[document_id].add(batch_embeddings)

            # Store for later use if needed
            all_embeddings.append(batch_embeddings)

            logger.info(f"Batch complete: added {len(batch_chunks)} chunks to index")

            # Report progress after batch completion for smooth progression
            if progress_callback:
                progress_callback(batch_end, total_chunks)

        # 3. Store metadata for all chunks
        self.metadata_stores[document_id] = {
            chunk.chunk_id: chunk for chunk in chunks
        }

        # 4. Update reference map
        await self.reference_map.build(chunks)

        # 5. Store document info
        self.document_info[document_id] = {
            'document_type': document_type,
            'num_chunks': total_chunks,
            'metadata': metadata or {}
        }

        logger.info(f"Successfully added document {document_id} with {total_chunks} chunks in {len(all_embeddings)} batches")

    def _create_faiss_index(
        self,
        training_vectors: Optional[np.ndarray] = None
    ) -> faiss.Index:
        """
        Create FAISS index optimized for legal retrieval

        Args:
            training_vectors: Vectors for training (required for IVF)

        Returns:
            Configured FAISS index
        """
        vector_size = self.config.vector_size

        if self.config.index_type == 'flat':
            # Exact search with inner product
            index = faiss.IndexFlatIP(vector_size)
            logger.info("Created FAISS IndexFlatIP (exact search)")

        elif self.config.index_type == 'ivf':
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatIP(vector_size)
            index = faiss.IndexIVFFlat(
                quantizer,
                vector_size,
                self.config.ivf_nlist
            )

            # Train index if vectors provided
            if training_vectors is not None and len(training_vectors) > 0:
                logger.info(f"Training IVF index with {len(training_vectors)} vectors")
                index.train(training_vectors)
            else:
                logger.warning("IVF index created but not trained")

            # Set search parameters
            index.nprobe = self.config.ivf_nprobe

            logger.info(f"Created FAISS IndexIVFFlat (nlist={self.config.ivf_nlist})")

        elif self.config.index_type == 'hnsw':
            # HNSW graph-based search
            index = faiss.IndexHNSWFlat(
                vector_size,
                self.config.hnsw_m
            )
            index.hnsw.efConstruction = self.config.hnsw_ef_construction
            index.hnsw.efSearch = self.config.hnsw_ef_search

            logger.info(
                f"Created FAISS IndexHNSWFlat "
                f"(M={self.config.hnsw_m}, ef={self.config.hnsw_ef_search})"
            )

        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        # Move to GPU if enabled and available
        if self.config.enable_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

        return index

    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Semantic search across one or more documents

        Args:
            query: Search query
            document_ids: Filter by document IDs (None = all)
            top_k: Number of results
            filter_metadata: Metadata filters (e.g., {'content_type': 'obligation'})

        Returns:
            List of search results with scores
        """
        # 1. Embed query
        query_embedding = await self.embedder.embed_query(query)
        query_vector = query_embedding.reshape(1, -1)

        # 2. Search in selected indices
        all_results = []

        indices_to_search = document_ids or list(self.indices.keys())

        for doc_id in indices_to_search:
            if doc_id not in self.indices:
                logger.warning(f"Document {doc_id} not found in index")
                continue

            index = self.indices[doc_id]

            # FAISS search
            scores, indices = index.search(query_vector, top_k)

            # Map to chunks
            metadata_store = self.metadata_stores[doc_id]
            chunk_ids = list(metadata_store.keys())

            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(chunk_ids):
                    chunk_id = chunk_ids[idx]
                    chunk = metadata_store[chunk_id]

                    # Apply metadata filters
                    if filter_metadata and not self._matches_filter(chunk, filter_metadata):
                        continue

                    all_results.append(SearchResult(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        score=float(score),
                        document_id=doc_id
                    ))

        # 3. Merge and sort
        all_results.sort(key=lambda x: x.score, reverse=True)

        # 4. Assign ranks
        for rank, result in enumerate(all_results[:top_k], start=1):
            result.rank = rank

        return all_results[:top_k]

    def _matches_filter(self, chunk: LegalChunk, filters: Dict) -> bool:
        """Check if chunk matches metadata filters"""
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

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
        chunk_ids = self.reference_map.get_chunks_by_reference(legal_ref)

        if not chunk_ids:
            return None

        # Filter by document if specified
        if document_id:
            filtered_ids = []
            for cid in chunk_ids:
                chunk = self._get_chunk(cid)
                if chunk and chunk.document_id == document_id:
                    filtered_ids.append(cid)
            chunk_ids = filtered_ids

        if not chunk_ids:
            return None

        # Return first match
        return self._get_chunk(chunk_ids[0])

    def _get_chunk(self, chunk_id: str) -> Optional[LegalChunk]:
        """Get chunk by ID from any document"""
        for metadata_store in self.metadata_stores.values():
            if chunk_id in metadata_store:
                return metadata_store[chunk_id]
        return None

    def get_document_count(self) -> int:
        """Get total number of indexed documents"""
        return len(self.indices)

    def get_chunk_count(self, document_id: Optional[str] = None) -> int:
        """Get total number of indexed chunks"""
        if document_id:
            return len(self.metadata_stores.get(document_id, {}))
        return sum(len(store) for store in self.metadata_stores.values())

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a document"""
        return self.document_info.get(document_id)

    def get_document_chunks(self, document_id: str) -> List[LegalChunk]:
        """
        Get all chunks for a document from metadata store.

        This method is used by structural matching in cross-document retrieval
        to iterate through target document chunks.

        Args:
            document_id: Document identifier

        Returns:
            List of all chunks for the document (empty if not found)
        """
        metadata_store = self.metadata_stores.get(document_id, {})
        return list(metadata_store.values())


class IndexPersistence:
    """Save and load indices from disk"""

    def __init__(self, index_dir: Path = Path("./indexes")):
        """
        Initialize index persistence

        Args:
            index_dir: Directory for storing indices
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        document_id: str,
        index: faiss.Index,
        metadata: Dict,
        chunks: List[LegalChunk],
        reference_map: Optional[ReferenceMap] = None
    ):
        """
        Save index and metadata to disk

        Args:
            document_id: Document identifier
            index: FAISS index
            metadata: Document metadata
            chunks: List of chunks
            reference_map: Optional reference map
        """
        doc_dir = self.index_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save FAISS index
            index_path = doc_dir / "faiss.index"
            faiss.write_index(index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")

            # Save metadata
            metadata_path = doc_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {metadata_path}")

            # Save chunks
            chunks_path = doc_dir / "chunks.pkl"
            with open(chunks_path, 'wb') as f:
                pickle.dump(chunks, f)
            logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")

            # Save reference map if provided
            if reference_map:
                refmap_path = doc_dir / "reference_map.json"
                with open(refmap_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        reference_map.serialize(),
                        f,
                        indent=2,
                        ensure_ascii=False
                    )
                logger.info(f"Saved reference map to {refmap_path}")

        except Exception as e:
            logger.error(f"Failed to save index for {document_id}: {e}")
            raise IndexPersistenceError(f"Save failed: {e}")

    async def load(
        self,
        document_id: str
    ) -> Tuple[faiss.Index, Dict, List[LegalChunk], Optional[ReferenceMap]]:
        """
        Load index and metadata from disk

        Args:
            document_id: Document identifier

        Returns:
            Tuple of (index, metadata, chunks, reference_map)
        """
        doc_dir = self.index_dir / document_id

        if not doc_dir.exists():
            raise IndexPersistenceError(f"Index directory not found: {doc_dir}")

        try:
            # Load FAISS index
            index_path = doc_dir / "faiss.index"
            index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index from {index_path}")

            # Load metadata
            metadata_path = doc_dir / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")

            # Load chunks
            chunks_path = doc_dir / "chunks.pkl"
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

            # Load reference map if exists
            reference_map = None
            refmap_path = doc_dir / "reference_map.json"
            if refmap_path.exists():
                with open(refmap_path, 'r', encoding='utf-8') as f:
                    refmap_data = json.load(f)
                reference_map = ReferenceMap.deserialize(refmap_data)
                logger.info(f"Loaded reference map from {refmap_path}")

            return index, metadata, chunks, reference_map

        except Exception as e:
            logger.error(f"Failed to load index for {document_id}: {e}")
            raise IndexPersistenceError(f"Load failed: {e}")

    def exists(self, document_id: str) -> bool:
        """Check if index exists for document"""
        doc_dir = self.index_dir / document_id
        return (doc_dir / "faiss.index").exists()

    def list_documents(self) -> List[str]:
        """List all indexed documents"""
        if not self.index_dir.exists():
            return []

        documents = []
        for doc_dir in self.index_dir.iterdir():
            if doc_dir.is_dir() and (doc_dir / "faiss.index").exists():
                documents.append(doc_dir.name)

        return documents

    async def delete(self, document_id: str):
        """
        Delete index for document

        Args:
            document_id: Document identifier
        """
        doc_dir = self.index_dir / document_id

        if doc_dir.exists():
            import shutil
            shutil.rmtree(doc_dir)
            logger.info(f"Deleted index for {document_id}")


# Custom Exceptions

class IndexError(Exception):
    """Base exception for indexing errors"""
    pass


class IndexCorruptedError(IndexError):
    """FAISS index is corrupted"""
    pass


class IndexPersistenceError(IndexError):
    """Error during index save/load"""
    pass


# Utility Functions

def create_training_vectors(
    chunks: List[LegalChunk],
    embedder: LegalEmbedder,
    sample_size: Optional[int] = None
) -> np.ndarray:
    """
    Create training vectors for IVF index

    Args:
        chunks: List of chunks
        embedder: Embedder instance
        sample_size: Number of samples (None = use all)

    Returns:
        Training vectors
    """
    import asyncio

    if sample_size and len(chunks) > sample_size:
        import random
        chunks = random.sample(chunks, sample_size)

    # Run synchronously in this context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    embeddings = loop.run_until_complete(embedder.embed_chunks(chunks))
    loop.close()

    return embeddings
