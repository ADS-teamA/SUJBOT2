"""
FAISS-only Vector Store for Document Analyzer
Simplified version using only FAISS for vector storage
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
import pickle

import yaml
from sentence_transformers import SentenceTransformer
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
    metadata: Dict[str, Any] = None


class FAISSVectorStore:
    """FAISS-based vector store"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)

        # Initialize embedding model
        model_name = self.config.get('embeddings', {}).get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        device = self.config.get('embeddings', {}).get('device', 'cpu')

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.faiss_index = None
        self.chunk_metadata = {}  # Store metadata separately
        self.chunk_ids = []       # Store chunk IDs in order

        # Index files
        self.index_dir = Path(self.config.get('paths', {}).get('index_dir', './indexes'))
        self.index_dir.mkdir(exist_ok=True)
        self.faiss_index_path = self.index_dir / 'faiss.index'
        self.metadata_path = self.index_dir / 'metadata.json'

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'embeddings': {
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'device': 'cpu',
                'batch_size': 32,
                'normalize': True
            },
            'paths': {
                'index_dir': './indexes'
            }
        }

    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if not texts:
            return np.array([])

        batch_size = self.config.get('embeddings', {}).get('batch_size', 32)
        normalize = self.config.get('embeddings', {}).get('normalize', True)

        # Generate embeddings in batches
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def _create_faiss_index(self) -> faiss.IndexFlatIP:
        """Create a new FAISS index"""
        logger.info("Creating new FAISS index")
        # Using Inner Product for cosine similarity (with normalized vectors)
        return faiss.IndexFlatIP(self.vector_size)

    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.faiss_index_path))

        # Save metadata
        metadata_to_save = {
            'chunk_metadata': self.chunk_metadata,
            'chunk_ids': self.chunk_ids,
            'vector_size': self.vector_size,
            'indexed_at': datetime.now().isoformat()
        }

        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

    def _load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            if self.faiss_index_path.exists() and self.metadata_path.exists():
                logger.info(f"Loading FAISS index from {self.faiss_index_path}")
                self.faiss_index = faiss.read_index(str(self.faiss_index_path))

                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                self.chunk_metadata = metadata.get('chunk_metadata', {})
                self.chunk_ids = metadata.get('chunk_ids', [])

                logger.info(f"Loaded {len(self.chunk_ids)} chunks from index")
                return True

        except Exception as e:
            logger.error(f"Error loading index: {e}")

        return False

    async def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to FAISS index"""
        if not chunks:
            return

        # Load existing index if available
        if self.faiss_index is None:
            if not self._load_index():
                self.faiss_index = self._create_faiss_index()

        # Generate embeddings for chunks without them
        texts_to_embed = [c.content for c in chunks if c.embedding is None]
        if texts_to_embed:
            embeddings = await self.generate_embeddings(texts_to_embed)
            embed_idx = 0
            for chunk in chunks:
                if chunk.embedding is None:
                    chunk.embedding = embeddings[embed_idx]
                    embed_idx += 1

        # Add to FAISS index
        embeddings_to_add = []
        for chunk in chunks:
            embeddings_to_add.append(chunk.embedding)
            self.chunk_ids.append(chunk.chunk_id)
            self.chunk_metadata[chunk.chunk_id] = {
                'content': chunk.content,
                'metadata': chunk.metadata,
                'indexed_at': datetime.now().isoformat()
            }

        if embeddings_to_add:
            embeddings_matrix = np.vstack(embeddings_to_add).astype('float32')
            self.faiss_index.add(embeddings_matrix)

            logger.info(f"Added {len(chunks)} chunks to FAISS")

            # Save to disk
            self._save_index()

    async def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[SearchResult]:
        """Search for similar documents"""
        if self.faiss_index is None:
            if not self._load_index():
                return []

        # Generate query embedding
        query_embedding = await self.generate_embeddings([query])
        if query_embedding.size == 0:
            return []

        # Search in FAISS
        query_vector = query_embedding[0].astype('float32').reshape(1, -1)
        scores, indices = self.faiss_index.search(query_vector, top_k)

        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_ids):  # Valid index
                chunk_id = self.chunk_ids[idx]
                if chunk_id in self.chunk_metadata:
                    metadata = self.chunk_metadata[chunk_id]

                    # Apply filters if provided
                    if filters:
                        skip = False
                        chunk_meta = metadata.get('metadata', {})
                        for key, value in filters.items():
                            if chunk_meta.get(key) != value:
                                skip = True
                                break
                        if skip:
                            continue

                    result = SearchResult(
                        chunk_id=chunk_id,
                        content=metadata['content'],
                        score=float(score),
                        metadata=metadata.get('metadata', {})
                    )
                    results.append(result)

        return results

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = {
            'total_chunks': len(self.chunk_ids),
            'vector_size': self.vector_size,
            'index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }
        return stats


def create_vector_store(config_path: str = "config.yaml") -> FAISSVectorStore:
    """Factory function to create vector store"""
    return FAISSVectorStore(config_path)