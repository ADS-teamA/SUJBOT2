"""
Vector Store Management for Document Analyzer
Handles embeddings storage and retrieval for millions of documents
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import hashlib
import json
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, Range, SearchRequest, NamedVector
)
import chromadb
from chromadb.config import Settings
import faiss
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    chunk_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.chunk_id:
            # Generate chunk ID from content hash
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class SearchResult:
    """Result from vector search"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """Abstract base class for vector stores"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Initialize the embedding model"""
        model_name = self.config.get('embeddings', {}).get('model', 'all-MiniLM-L6-v2')
        device = self.config.get('embeddings', {}).get('device', 'cpu')

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.embedding_model = SentenceTransformer(model_name, device=device)

        # Set normalization
        self.normalize_embeddings = self.config.get('embeddings', {}).get('normalize', True)

    async def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts"""
        embeddings = []

        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.to_thread(
                self.embedding_model.encode,
                batch,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings
            )
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    async def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the store"""
        raise NotImplementedError

    async def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[SearchResult]:
        """Search for similar documents"""
        raise NotImplementedError

    async def delete_collection(self, collection_name: str):
        """Delete a collection"""
        raise NotImplementedError

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.client = QdrantClient(
            host=config.get('vector_db', {}).get('host', 'localhost'),
            port=config.get('vector_db', {}).get('port', 6333)
        )
        self.collection_name = config.get('vector_db', {}).get('collection', 'documents')
        self.vector_size = config.get('vector_db', {}).get('vector_size', 384)

        self._init_collection()

    def _init_collection(self):
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except:
            logger.info(f"Creating collection '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    async def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to Qdrant"""
        if not chunks:
            return

        # Generate embeddings for chunks without them
        texts_to_embed = [c.content for c in chunks if c.embedding is None]
        if texts_to_embed:
            embeddings = await self.generate_embeddings(texts_to_embed)
            embed_idx = 0
            for chunk in chunks:
                if chunk.embedding is None:
                    chunk.embedding = embeddings[embed_idx]
                    embed_idx += 1

        # Prepare points for Qdrant
        points = []
        for chunk in chunks:
            point = PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding.tolist(),
                payload={
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "indexed_at": datetime.now().isoformat()
                }
            )
            points.append(point)

        # Batch upsert
        batch_size = self.config.get('vector_db', {}).get('batch_upsert_size', 100)
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

    async def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[SearchResult]:
        """Search Qdrant for similar documents"""
        # Generate query embedding
        query_embedding = await self.generate_embeddings([query])
        query_vector = query_embedding[0].tolist()

        # Build filter if provided
        qdrant_filter = None
        if filters:
            # Convert filters to Qdrant format
            # Example: {"document_id": "doc1", "page": {"gte": 10, "lte": 20}}
            conditions = []
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Range filter
                    conditions.append(FieldCondition(
                        key=f"metadata.{key}",
                        range=Range(**value)
                    ))
                else:
                    # Exact match
                    conditions.append(FieldCondition(
                        key=f"metadata.{key}",
                        match=value
                    ))

            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )

        # Convert to SearchResult
        results = []
        for hit in search_result:
            results.append(SearchResult(
                chunk_id=hit.id,
                content=hit.payload.get("content", ""),
                score=hit.score,
                metadata=hit.payload.get("metadata", {})
            ))

        return results

    async def delete_collection(self, collection_name: str = None):
        """Delete a Qdrant collection"""
        name = collection_name or self.collection_name
        self.client.delete_collection(name)
        logger.info(f"Deleted collection: {name}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "total_vectors": info.vectors_count,
            "indexed_vectors": info.indexed_vectors_count,
            "collection_name": self.collection_name,
            "vector_size": self.vector_size
        }


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=config.get('paths', {}).get('index_dir', './indexes')
        ))

        self.collection_name = config.get('vector_db', {}).get('collection', 'documents')
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    async def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to ChromaDB"""
        if not chunks:
            return

        # Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = await self.generate_embeddings(texts)

        # Add to collection
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[c.metadata for c in chunks]
        )

        logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    async def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[SearchResult]:
        """Search ChromaDB for similar documents"""
        # Generate query embedding
        query_embedding = await self.generate_embeddings([query])

        # Search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=filters  # ChromaDB accepts filters directly
        )

        # Convert to SearchResult
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i],
                score=1 - results['distances'][0][i],  # Convert distance to similarity
                metadata=results['metadatas'][0][i] if results['metadatas'] else {}
            ))

        return search_results

    async def delete_collection(self, collection_name: str = None):
        """Delete a ChromaDB collection"""
        name = collection_name or self.collection_name
        self.client.delete_collection(name)
        logger.info(f"Deleted collection: {name}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        count = self.collection.count()
        return {
            "total_vectors": count,
            "collection_name": self.collection_name
        }


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.vector_size = config.get('vector_db', {}).get('vector_size', 384)
        self.index_path = Path(config.get('paths', {}).get('index_dir', './indexes'))
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / "faiss.index"
        self.metadata_file = self.index_path / "metadata.json"

        self.index = None
        self.metadata = {}
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if self.index_file.exists():
            logger.info(f"Loading FAISS index from {self.index_file}")
            self.index = faiss.read_index(str(self.index_file))

            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
        else:
            logger.info("Creating new FAISS index")
            # Use simple flat index for now (works with any number of vectors)
            self.index = faiss.IndexFlatL2(self.vector_size)
            # IVF can be added later for larger datasets (requires min 100 vectors)

    async def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to FAISS"""
        if not chunks:
            return

        # Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = await self.generate_embeddings(texts)

        # Add vectors (IndexFlatL2 doesn't need training)
        start_idx = self.index.ntotal
        self.index.add(embeddings)

        # Store metadata
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.metadata[str(idx)] = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata
            }

        # Save index and metadata
        self._save_index()
        logger.info(f"Added {len(chunks)} chunks to FAISS")

    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    async def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[SearchResult]:
        """Search FAISS for similar documents"""
        # Generate query embedding
        query_embedding = await self.generate_embeddings([query])

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert to SearchResult with filtering
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            meta = self.metadata.get(str(idx), {})

            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if meta.get("metadata", {}).get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            results.append(SearchResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                content=meta.get("content", ""),
                score=1 / (1 + dist),  # Convert distance to similarity score
                metadata=meta.get("metadata", {})
            ))

        return results

    async def delete_collection(self, collection_name: str = None):
        """Delete FAISS index"""
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

        self.index = faiss.IndexFlatL2(self.vector_size)
        self.metadata = {}
        logger.info("Deleted FAISS index")

    async def get_stats(self) -> Dict[str, Any]:
        """Get FAISS statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "vector_size": self.vector_size,
            "index_type": type(self.index).__name__,
            "is_trained": True  # IndexFlatL2 is always "trained" (doesn't need training)
        }


def create_vector_store(config: Dict[str, Any]) -> VectorStore:
    """Factory function to create appropriate vector store"""

    # Load config from file if path provided
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    store_type = config.get('vector_db', {}).get('type', 'qdrant').lower()

    if store_type == 'qdrant':
        return QdrantVectorStore(config)
    elif store_type == 'chroma' or store_type == 'chromadb':
        return ChromaVectorStore(config)
    elif store_type == 'faiss':
        return FAISSVectorStore(config)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


# Example usage
async def main():
    """Example usage of vector store"""
    import yaml

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create vector store
    store = create_vector_store(config)

    # Create sample chunks
    chunks = [
        DocumentChunk(
            chunk_id="1",
            content="This is a test document about machine learning.",
            metadata={"document_id": "doc1", "page": 1}
        ),
        DocumentChunk(
            chunk_id="2",
            content="Neural networks are powerful tools for pattern recognition.",
            metadata={"document_id": "doc1", "page": 2}
        )
    ]

    # Add documents
    await store.add_documents(chunks)

    # Search
    results = await store.search("machine learning", top_k=5)
    for result in results:
        print(f"Score: {result.score:.4f} - {result.content[:50]}...")

    # Get stats
    stats = await store.get_stats()
    print(f"Vector store stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())