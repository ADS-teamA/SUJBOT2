"""
PHASE 4: FAISS Vector Store with Multi-Layer Indexing

Based on research:
- LegalBench-RAG: Dense-only retrieval (no BM25)
- Multi-Layer Embeddings (Lima, 2024): 3 separate indexes
- Retrieval: K=6 on Layer 3 (primary)
- DRM prevention: Document-level filtering

Implementation:
- 3 separate FAISS indexes (IndexFlatIP for cosine similarity)
- Metadata tracking for chunk IDs, document IDs, sections
- Save/load functionality
- Document-level filtering during retrieval
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS required for vector store. " "Install with: uv pip install faiss-cpu (or faiss-gpu)"
    )

# Import utilities
from src.utils.persistence import PersistenceManager, VectorStoreLoader

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Multi-layer FAISS vector store for RAG pipeline.

    Creates 3 separate FAISS indexes:
    - Layer 1: Document-level (1 vector per document)
    - Layer 2: Section-level (N vectors per document)
    - Layer 3: Chunk-level (M vectors per document) - PRIMARY

    Based on:
    - LegalBench-RAG: Dense-only retrieval
    - Multi-Layer Embeddings (Lima, 2024)
    """

    def __init__(self, dimensions: int):
        """
        Initialize multi-layer FAISS vector store.

        Args:
            dimensions: Embedding dimensionality (e.g., 3072 for text-embedding-3-large)
        """
        self.dimensions = dimensions

        # Create 3 separate FAISS indexes
        # Using IndexFlatIP (inner product) for cosine similarity with normalized vectors
        self.index_layer1 = faiss.IndexFlatIP(dimensions)
        self.index_layer2 = faiss.IndexFlatIP(dimensions)
        self.index_layer3 = faiss.IndexFlatIP(dimensions)

        # Metadata storage (chunk objects)
        self.metadata_layer1: List[Dict] = []
        self.metadata_layer2: List[Dict] = []
        self.metadata_layer3: List[Dict] = []

        # Document ID mapping for DRM prevention
        self.doc_id_to_indices = {
            1: {},  # Layer 1: doc_id -> [indices]
            2: {},  # Layer 2: doc_id -> [indices]
            3: {},  # Layer 3: doc_id -> [indices]
        }

        logger.info(f"FAISSVectorStore initialized: {dimensions}D, 3 layers")

    def add_chunks(self, chunks_dict: Dict[str, List], embeddings_dict: Dict[str, np.ndarray]):
        """
        Add chunks and embeddings to appropriate layers.

        Args:
            chunks_dict: Dict with keys 'layer1', 'layer2', 'layer3' containing Chunk objects
            embeddings_dict: Dict with keys 'layer1', 'layer2', 'layer3' containing embeddings
        """
        logger.info("Adding chunks to FAISS indexes...")

        # Add Layer 1 (Document level)
        self._add_layer(layer=1, chunks=chunks_dict["layer1"], embeddings=embeddings_dict["layer1"])

        # Add Layer 2 (Section level)
        self._add_layer(layer=2, chunks=chunks_dict["layer2"], embeddings=embeddings_dict["layer2"])

        # Add Layer 3 (Chunk level - PRIMARY)
        self._add_layer(layer=3, chunks=chunks_dict["layer3"], embeddings=embeddings_dict["layer3"])

        logger.info(
            f"Chunks added: "
            f"L1={self.index_layer1.ntotal}, "
            f"L2={self.index_layer2.ntotal}, "
            f"L3={self.index_layer3.ntotal}"
        )

    def _add_layer(self, layer: int, chunks: List, embeddings: np.ndarray):
        """Add chunks and embeddings to a specific layer."""
        # Skip empty layers (optimization for flat documents)
        if not chunks or embeddings is None or embeddings.size == 0:
            logger.info(f"Layer {layer}: Skipped (empty layer)")
            return

        # Select index and metadata
        if layer == 1:
            index = self.index_layer1
            metadata = self.metadata_layer1
        elif layer == 2:
            index = self.index_layer2
            metadata = self.metadata_layer2
        elif layer == 3:
            index = self.index_layer3
            metadata = self.metadata_layer3
        else:
            raise ValueError(f"Invalid layer: {layer}")

        # Add embeddings to FAISS index
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        index.add(embeddings)

        # Add metadata
        for chunk in chunks:
            # Build hierarchical path: document_id > section_path
            hierarchical_path = chunk.metadata.document_id
            if chunk.metadata.section_path:
                hierarchical_path = f"{chunk.metadata.document_id} > {chunk.metadata.section_path}"

            chunk_meta = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.metadata.document_id,
                "section_id": chunk.metadata.section_id,
                "section_title": chunk.metadata.section_title,
                "section_path": chunk.metadata.section_path,
                "hierarchical_path": hierarchical_path,  # NEW: Full path for breadcrumb navigation
                "page_number": chunk.metadata.page_number,
                "layer": layer,
                # Store raw_content for generation (without SAC)
                "content": chunk.raw_content,
                # Semantic clustering (PHASE 4.5)
                "cluster_id": chunk.metadata.cluster_id,
                "cluster_label": chunk.metadata.cluster_label,
                "cluster_confidence": chunk.metadata.cluster_confidence,
            }
            metadata.append(chunk_meta)

            # Update doc_id mapping
            doc_id = chunk.metadata.document_id
            if doc_id not in self.doc_id_to_indices[layer]:
                self.doc_id_to_indices[layer][doc_id] = []
            self.doc_id_to_indices[layer][doc_id].append(len(metadata) - 1)

        logger.info(f"Layer {layer}: Added {len(chunks)} chunks")

    def search_layer3(
        self,
        query_embedding: np.ndarray,
        k: int = 6,
        document_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Search Layer 3 (PRIMARY retrieval layer).

        Args:
            query_embedding: Query embedding vector (1D or 2D array)
            k: Number of results to retrieve (default: 6, per research)
            document_filter: Optional document_id to filter results (DRM prevention)
            similarity_threshold: Optional minimum similarity score

        Returns:
            List of chunk metadata dicts with 'score' field added
        """
        return self._search_layer(
            layer=3,
            query_embedding=query_embedding,
            k=k,
            document_filter=document_filter,
            similarity_threshold=similarity_threshold,
        )

    def search_layer2(
        self, query_embedding: np.ndarray, k: int = 3, document_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search Layer 2 (Section level)."""
        return self._search_layer(
            layer=2, query_embedding=query_embedding, k=k, document_filter=document_filter
        )

    def search_layer1(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        """Search Layer 1 (Document level)."""
        return self._search_layer(layer=1, query_embedding=query_embedding, k=k)

    def _search_layer(
        self,
        layer: int,
        query_embedding: np.ndarray,
        k: int,
        document_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Search a specific layer."""
        # Select index and metadata
        if layer == 1:
            index = self.index_layer1
            metadata = self.metadata_layer1
        elif layer == 2:
            index = self.index_layer2
            metadata = self.metadata_layer2
        elif layer == 3:
            index = self.index_layer3
            metadata = self.metadata_layer3
        else:
            raise ValueError(f"Invalid layer: {layer}")

        if index.ntotal == 0:
            logger.warning(f"Layer {layer}: Index is empty")
            return []

        # Prepare query embedding
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2:
            raise ValueError(
                f"Query embedding must be 1D or 2D array, got shape {query_embedding.shape}"
            )

        if query_embedding.shape[1] != self.dimensions:
            msg = (
                f"Query embedding dimension {query_embedding.shape[1]} does not match "
                f"FAISS index dimension {self.dimensions}. "
                "Rebuild the vector store with the current embedding model or switch "
                "the embedder back to the model used when this store was created."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Document filtering for DRM prevention
        if document_filter:
            # Get indices for this document
            doc_indices = self.doc_id_to_indices[layer].get(document_filter, [])
            if not doc_indices:
                logger.warning(f"Layer {layer}: No chunks for document {document_filter}")
                return []

            # Search only within document indices
            # Note: FAISS doesn't natively support filtering, so we search all
            # and filter results. For production, use IDSelectorArray.
            k_search = min(k * 10, index.ntotal)  # Search more, filter later
        else:
            k_search = min(k, index.ntotal)

        # Search
        scores, indices = index.search(query_embedding, k_search)

        # Flatten results
        scores = scores[0]
        indices = indices[0]

        # Build results
        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            # Apply document filter if specified
            if document_filter:
                if metadata[idx]["document_id"] != document_filter:
                    continue

            # Apply similarity threshold if specified
            if similarity_threshold and score < similarity_threshold:
                continue

            result = metadata[idx].copy()
            result["score"] = float(score)
            result["index"] = int(idx)
            results.append(result)

            if len(results) >= k:
                break

        logger.info(f"Layer {layer}: Retrieved {len(results)}/{k} chunks")
        return results

    def hierarchical_search(
        self,
        query_embedding: np.ndarray,
        k_layer3: int = 6,
        use_doc_filtering: bool = True,
        similarity_threshold_offset: float = 0.25,
    ) -> Dict[str, List[Dict]]:
        """
        Hierarchical search across all layers with DRM prevention.

        Strategy:
        1. Search Layer 1 (Document) to identify relevant document
        2. Use document_id to filter Layer 3 search (DRM prevention)
        3. Retrieve K=6 chunks from Layer 3 (PRIMARY)
        4. Optional: Retrieve Layer 2 sections for context

        Args:
            query_embedding: Query embedding vector
            k_layer3: Number of chunks to retrieve from Layer 3 (default: 6)
            use_doc_filtering: Enable document-level filtering (DRM prevention)
            similarity_threshold_offset: Threshold = top_score - offset (default: 0.25)

        Returns:
            Dict with keys 'layer1', 'layer2', 'layer3' containing results
        """
        logger.info("Performing hierarchical search...")

        results = {}

        # Step 1: Search Layer 1 (Document level)
        layer1_results = self.search_layer1(query_embedding, k=1)
        results["layer1"] = layer1_results

        # Get document filter for DRM prevention
        document_filter = None
        if use_doc_filtering and layer1_results:
            document_filter = layer1_results[0]["document_id"]
            logger.info(f"Document filter: {document_filter}")

        # Step 2: Search Layer 3 (PRIMARY - Chunk level)
        layer3_results = self.search_layer3(
            query_embedding=query_embedding, k=k_layer3, document_filter=document_filter
        )

        # Apply similarity threshold (top_score - 25%)
        if layer3_results and similarity_threshold_offset > 0:
            top_score = layer3_results[0]["score"]
            threshold = top_score - similarity_threshold_offset
            layer3_results = [r for r in layer3_results if r["score"] >= threshold]
            logger.info(
                f"Similarity threshold applied: {threshold:.4f} "
                f"({len(layer3_results)} chunks remaining)"
            )

        results["layer3"] = layer3_results

        # Step 3: Optional Layer 2 (Section context)
        layer2_results = self.search_layer2(
            query_embedding=query_embedding, k=3, document_filter=document_filter
        )
        results["layer2"] = layer2_results

        logger.info(
            f"Hierarchical search complete: "
            f"L1={len(layer1_results)}, "
            f"L2={len(layer2_results)}, "
            f"L3={len(layer3_results)}"
        )

        return results

    def merge(self, other: "FAISSVectorStore"):
        """
        Merge another vector store into this one with deduplication.

        This allows adding new documents to an existing vector store without
        rebuilding from scratch. Automatically detects and skips duplicate
        chunk_ids to prevent data duplication.

        Args:
            other: Another FAISSVectorStore to merge into this one

        Raises:
            ValueError: If dimensions don't match
        """
        if self.dimensions != other.dimensions:
            raise ValueError(
                f"Cannot merge stores with different dimensions: "
                f"{self.dimensions} vs {other.dimensions}"
            )

        logger.info(f"Merging vector store with {other.get_stats()['documents']} documents...")

        # Track deduplication statistics
        stats = {"added": 0, "skipped": 0, "layers": {}}

        # Merge Layer 1 with deduplication
        if other.index_layer1.ntotal > 0:
            layer_stats = self._merge_layer_with_deduplication(
                layer=1,
                other_index=other.index_layer1,
                other_metadata=other.metadata_layer1,
                other_doc_indices=other.doc_id_to_indices[1]
            )
            stats["layers"][1] = layer_stats
            stats["added"] += layer_stats["added"]
            stats["skipped"] += layer_stats["skipped"]

        # Merge Layer 2 with deduplication
        if other.index_layer2.ntotal > 0:
            layer_stats = self._merge_layer_with_deduplication(
                layer=2,
                other_index=other.index_layer2,
                other_metadata=other.metadata_layer2,
                other_doc_indices=other.doc_id_to_indices[2]
            )
            stats["layers"][2] = layer_stats
            stats["added"] += layer_stats["added"]
            stats["skipped"] += layer_stats["skipped"]

        # Merge Layer 3 with deduplication
        if other.index_layer3.ntotal > 0:
            layer_stats = self._merge_layer_with_deduplication(
                layer=3,
                other_index=other.index_layer3,
                other_metadata=other.metadata_layer3,
                other_doc_indices=other.doc_id_to_indices[3]
            )
            stats["layers"][3] = layer_stats
            stats["added"] += layer_stats["added"]
            stats["skipped"] += layer_stats["skipped"]

        final_stats = self.get_stats()
        logger.info(
            f"Merge complete: {stats['added']} vectors added, {stats['skipped']} duplicates skipped"
        )
        logger.info(
            f"Total: {final_stats['documents']} documents, {final_stats['total_vectors']} vectors"
        )

        return stats

    def _merge_layer_with_deduplication(
        self, layer: int, other_index, other_metadata: List[Dict], other_doc_indices: Dict
    ) -> Dict:
        """
        Merge a single layer with chunk_id deduplication.

        Args:
            layer: Layer number (1, 2, or 3)
            other_index: Other layer's FAISS index
            other_metadata: Other layer's metadata list
            other_doc_indices: Other layer's doc_id_to_indices mapping

        Returns:
            Dict with 'added' and 'skipped' counts
        """
        # Select current layer's structures
        if layer == 1:
            current_index = self.index_layer1
            current_metadata = self.metadata_layer1
            current_doc_indices = self.doc_id_to_indices[1]
        elif layer == 2:
            current_index = self.index_layer2
            current_metadata = self.metadata_layer2
            current_doc_indices = self.doc_id_to_indices[2]
        elif layer == 3:
            current_index = self.index_layer3
            current_metadata = self.metadata_layer3
            current_doc_indices = self.doc_id_to_indices[3]
        else:
            raise ValueError(f"Invalid layer: {layer}")

        # Build set of existing chunk_ids for O(1) lookup
        existing_chunk_ids = set()
        for meta in current_metadata:
            if "chunk_id" in meta:
                existing_chunk_ids.add(meta["chunk_id"])

        # Track which vectors to add
        vectors_to_add = []
        metadata_to_add = []
        new_doc_indices = {}

        # Iterate through other's vectors
        vectors = other_index.reconstruct_n(0, other_index.ntotal)
        vectors = vectors.reshape(other_index.ntotal, self.dimensions).astype(np.float32)

        added = 0
        skipped = 0

        for i, (vector, meta) in enumerate(zip(vectors, other_metadata)):
            chunk_id = meta.get("chunk_id")

            # Check for duplicate
            if chunk_id and chunk_id in existing_chunk_ids:
                skipped += 1
                logger.debug(f"Layer {layer}: Skipping duplicate chunk_id: {chunk_id}")
                continue

            # Not a duplicate - add it
            vectors_to_add.append(vector)
            metadata_to_add.append(meta)

            # Track for doc_id_to_indices update
            doc_id = meta.get("document_id")
            if doc_id:
                if doc_id not in new_doc_indices:
                    new_doc_indices[doc_id] = []
                # Index will be: current_index.ntotal + position in vectors_to_add
                new_idx = current_index.ntotal + len(vectors_to_add) - 1
                new_doc_indices[doc_id].append(new_idx)

            # Add to existing_chunk_ids to prevent duplicates within same merge
            if chunk_id:
                existing_chunk_ids.add(chunk_id)

            added += 1

        # Add vectors to FAISS index
        if vectors_to_add:
            vectors_array = np.array(vectors_to_add).astype(np.float32)
            current_index.add(vectors_array)
            current_metadata.extend(metadata_to_add)

            # Update doc_id_to_indices
            for doc_id, indices in new_doc_indices.items():
                if doc_id not in current_doc_indices:
                    current_doc_indices[doc_id] = []
                current_doc_indices[doc_id].extend(indices)

        logger.info(f"Layer {layer}: +{added} vectors, ~{skipped} duplicates skipped")

        return {"added": added, "skipped": skipped}

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "dimensions": self.dimensions,
            "layer1_count": self.index_layer1.ntotal,
            "layer2_count": self.index_layer2.ntotal,
            "layer3_count": self.index_layer3.ntotal,
            "total_vectors": (
                self.index_layer1.ntotal + self.index_layer2.ntotal + self.index_layer3.ntotal
            ),
            "documents": len(
                set(
                    m["document_id"]
                    for m in self.metadata_layer1 + self.metadata_layer2 + self.metadata_layer3
                )
            ),
        }

    def save(self, output_dir: Path):
        """
        Save FAISS indexes and metadata to disk.

        Uses hybrid serialization:
        - JSON for config (dimensions, human-readable)
        - Pickle for large arrays (metadata lists, doc_id_to_indices)

        Args:
            output_dir: Directory to save indexes
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving vector store to {output_dir}")

        # Save FAISS indexes
        faiss.write_index(self.index_layer1, str(output_dir / "faiss_layer1.index"))
        faiss.write_index(self.index_layer2, str(output_dir / "faiss_layer2.index"))
        faiss.write_index(self.index_layer3, str(output_dir / "faiss_layer3.index"))

        # Save config (JSON - human-readable)
        config = {
            "dimensions": self.dimensions,
            "layer1_count": self.index_layer1.ntotal,
            "layer2_count": self.index_layer2.ntotal,
            "layer3_count": self.index_layer3.ntotal,
            "format_version": "1.0",
        }
        PersistenceManager.save_json(output_dir / "faiss_metadata.json", config)

        # Save arrays (Pickle - performance)
        arrays = {
            "metadata_layer1": self.metadata_layer1,
            "metadata_layer2": self.metadata_layer2,
            "metadata_layer3": self.metadata_layer3,
            "doc_id_to_indices": self.doc_id_to_indices,
        }
        PersistenceManager.save_pickle(output_dir / "faiss_arrays.pkl", arrays)

        logger.info(f"Vector store saved: {self.get_stats()}")

    @classmethod
    def load(cls, input_dir: Path) -> "FAISSVectorStore":
        """
        Load FAISS indexes and metadata from disk.

        Supports backward compatibility:
        - Old format: layer1.index, metadata.pkl
        - New format: faiss_layer1.index, faiss_metadata.json, faiss_arrays.pkl

        Args:
            input_dir: Directory containing saved indexes

        Returns:
            FAISSVectorStore instance
        """
        input_dir = Path(input_dir)

        logger.info(f"Loading vector store from {input_dir}")

        # Detect format
        format_type = VectorStoreLoader.detect_format(input_dir)
        logger.info(f"Detected format: {format_type}")

        if format_type == "new":
            # Load config (JSON)
            config = PersistenceManager.load_json(input_dir / "faiss_metadata.json")
            dimensions = config["dimensions"]

            # Load arrays (Pickle)
            arrays = PersistenceManager.load_pickle(input_dir / "faiss_arrays.pkl")

            # Create instance
            store = cls(dimensions=dimensions)

            # Load FAISS indexes (new naming)
            store.index_layer1 = faiss.read_index(str(input_dir / "faiss_layer1.index"))
            store.index_layer2 = faiss.read_index(str(input_dir / "faiss_layer2.index"))
            store.index_layer3 = faiss.read_index(str(input_dir / "faiss_layer3.index"))

            # Load metadata arrays
            store.metadata_layer1 = arrays["metadata_layer1"]
            store.metadata_layer2 = arrays["metadata_layer2"]
            store.metadata_layer3 = arrays["metadata_layer3"]
            store.doc_id_to_indices = arrays["doc_id_to_indices"]

        else:  # old format
            logger.warning(
                "Loading old format vector store. "
                "Consider re-saving in new format for better performance."
            )

            # Load metadata (old pickle format)
            metadata = PersistenceManager.load_pickle(input_dir / "metadata.pkl")

            # Create instance
            store = cls(dimensions=metadata["dimensions"])

            # Load FAISS indexes (old naming)
            store.index_layer1 = faiss.read_index(str(input_dir / "layer1.index"))
            store.index_layer2 = faiss.read_index(str(input_dir / "layer2.index"))
            store.index_layer3 = faiss.read_index(str(input_dir / "layer3.index"))

            # Load metadata
            store.metadata_layer1 = metadata["metadata_layer1"]
            store.metadata_layer2 = metadata["metadata_layer2"]
            store.metadata_layer3 = metadata["metadata_layer3"]
            store.doc_id_to_indices = metadata["doc_id_to_indices"]

        logger.info(f"Vector store loaded: {store.get_stats()}")

        return store

    # ------------------------------------------------------------------
    # Utilities for analytics/clustering
    # ------------------------------------------------------------------
    def get_layer_embeddings_and_metadata(self, layer: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Reconstruct and return all embeddings and metadata for a given layer.

        Args:
            layer: 1 (document), 2 (section), or 3 (chunk)

        Returns:
            Tuple of (embeddings, metadata_list)
            - embeddings: np.ndarray of shape (N, D)
            - metadata_list: list of metadata dicts aligned with embeddings order

        Notes:
            - Embeddings are returned as float32 and re-normalized for cosine operations.
            - Order matches FAISS index order and corresponding metadata list.
        """
        if layer == 1:
            index = self.index_layer1
            metadata = self.metadata_layer1
        elif layer == 2:
            index = self.index_layer2
            metadata = self.metadata_layer2
        elif layer == 3:
            index = self.index_layer3
            metadata = self.metadata_layer3
        else:
            raise ValueError(f"Invalid layer: {layer}")

        n_total = index.ntotal
        if n_total == 0:
            return np.zeros((0, self.dimensions), dtype=np.float32), []

        # Reconstruct vectors from FAISS index
        vectors = np.zeros((n_total, self.dimensions), dtype=np.float32)
        for i in range(n_total):
            # IndexFlat supports reconstruct
            vec = index.reconstruct(i)
            # Faiss returns np array of float32
            vectors[i] = vec

        # Normalize for cosine operations (safety)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        return vectors, metadata


# Example usage
if __name__ == "__main__":
    from config import ExtractionConfig
    from multi_layer_chunker import MultiLayerChunker
    from docling_extractor_v2 import DoclingExtractorV2
    from embedding_generator import EmbeddingGenerator, EmbeddingConfig

    # Extract and chunk
    config = ExtractionConfig(enable_smart_hierarchy=True, generate_summaries=True)

    extractor = DoclingExtractorV2(config)
    result = extractor.extract("document.pdf")

    chunker = MultiLayerChunker(chunk_size=500, enable_sac=True)
    chunks = chunker.chunk_document(result)

    # Generate embeddings
    embedder = EmbeddingGenerator(EmbeddingConfig(model="text-embedding-3-large"))

    embeddings = {
        "layer1": embedder.embed_chunks(chunks["layer1"], layer=1),
        "layer2": embedder.embed_chunks(chunks["layer2"], layer=2),
        "layer3": embedder.embed_chunks(chunks["layer3"], layer=3),
    }

    # Create vector store
    vector_store = FAISSVectorStore(dimensions=embedder.dimensions)
    vector_store.add_chunks(chunks, embeddings)

    # Search
    query_embedding = embedder.embed_texts(["What is the safety specification?"])
    results = vector_store.hierarchical_search(query_embedding)

    print(f"Layer 3 results: {len(results['layer3'])}")
    for i, result in enumerate(results["layer3"][:3], 1):
        print(f"{i}. Score: {result['score']:.4f} - {result['section_title']}")

    # Save
    vector_store.save(Path("output/vector_store"))
