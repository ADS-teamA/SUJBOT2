"""
PHASE 5B: Hybrid Search (BM25 + Dense + RRF)

Implements hybrid retrieval combining:
- Dense retrieval (FAISS): Semantic similarity via embeddings
- Sparse retrieval (BM25): Keyword/exact match via term frequency
- Reciprocal Rank Fusion (RRF): Combines both rankings

Based on research:
- LegalBench-RAG: Hybrid search +23% precision over dense-only
- RRF formula: score = 1/(k + rank), k=60 (research-optimal)
- Contextual indexing: Index same text as FAISS (context + raw_content)

Architecture:
1. BM25Index: Single-layer BM25 index (L1, L2, or L3)
2. BM25Store: Multi-layer wrapper (3 BM25 indexes)
3. HybridVectorStore: Orchestrates FAISS + BM25 with RRF fusion

Usage:
    from hybrid_search import BM25Store, HybridVectorStore
    from faiss_vector_store import FAISSVectorStore

    # Build BM25 indexes
    bm25_store = BM25Store()
    bm25_store.build_from_chunks(chunks)

    # Create hybrid store
    hybrid_store = HybridVectorStore(
        faiss_store=vector_store,
        bm25_store=bm25_store,
        fusion_k=60
    )

    # Search with both text and embedding
    results = hybrid_store.hierarchical_search(
        query_text="safety requirements",
        query_embedding=embedding,
        k_layer3=6
    )
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank-bm25 required for hybrid search. " "Install with: pip install rank-bm25"
    )

# Import utilities
from src.utils.persistence import PersistenceManager

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 sparse index for a single layer (L1, L2, or L3).

    Indexes the same content as FAISS (context + raw_content) for consistency,
    but stores raw_content for retrieval (without context).

    Based on BM25Okapi algorithm with default parameters:
    - k1=1.5 (term frequency saturation)
    - b=0.75 (length normalization)
    """

    def __init__(self):
        """Initialize empty BM25 index."""
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_ids: List[str] = []
        self.metadata: List[Dict] = []
        self.doc_id_map: Dict[str, List[int]] = {}  # For document filtering

    def build_from_chunks(self, chunks: List) -> None:
        """
        Build BM25 index from chunks.

        Indexes: chunk.content (context + raw_content, same as FAISS)
        Stores: chunk.raw_content in metadata (for retrieval)

        Args:
            chunks: List of Chunk objects from MultiLayerChunker
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        for chunk in chunks:
            # Index with context (better retrieval)
            self.corpus.append(chunk.content)
            self.chunk_ids.append(chunk.chunk_id)

            # Tokenize for BM25
            tokens = self._tokenize(chunk.content)
            self.tokenized_corpus.append(tokens)

            # Build hierarchical path: document_id > section_path
            hierarchical_path = chunk.metadata.document_id
            if chunk.metadata.section_path:
                hierarchical_path = f"{chunk.metadata.document_id} > {chunk.metadata.section_path}"

            # Store metadata (same structure as FAISSVectorStore)
            meta = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.metadata.document_id,
                "section_id": chunk.metadata.section_id,
                "section_title": chunk.metadata.section_title,
                "section_path": chunk.metadata.section_path,
                "hierarchical_path": hierarchical_path,  # NEW: Full path for breadcrumb navigation
                "page_number": chunk.metadata.page_number,
                "layer": chunk.metadata.layer,
                "content": chunk.raw_content,  # Store without context for generation
            }
            self.metadata.append(meta)

            # Build doc_id mapping for filtering
            doc_id = chunk.metadata.document_id
            if doc_id not in self.doc_id_map:
                self.doc_id_map[doc_id] = []
            self.doc_id_map[doc_id].append(len(self.metadata) - 1)

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"BM25 index built: {len(self.corpus)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (whitespace splitting with lowercasing).

        Can be enhanced with:
        - Stemming (Porter stemmer)
        - Stopword removal
        - Legal-specific tokenization

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return text.lower().split()

    def search(self, query: str, k: int = 50, document_filter: Optional[str] = None) -> List[Dict]:
        """
        Search BM25 index.

        Args:
            query: Query text
            k: Number of results to retrieve
            document_filter: Optional document_id to filter results

        Returns:
            List of dicts with keys: chunk_id, content, score, index, ...
        """
        if self.bm25 is None or len(self.metadata) == 0:
            logger.warning("BM25 index is empty")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Apply document filter if specified
        if document_filter:
            indices = self.doc_id_map.get(document_filter, [])
            if not indices:
                logger.warning(f"No chunks for document {document_filter}")
                return []

            # Filter scores to only these indices
            filtered = [(i, scores[i]) for i in indices]
            top_k = sorted(filtered, key=lambda x: x[1], reverse=True)[:k]
        else:
            # Get top-k globally
            top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        # Build results
        results = []
        for idx, score in top_k:
            result = self.metadata[idx].copy()
            result["score"] = float(score)
            result["index"] = int(idx)
            results.append(result)

        return results

    def merge(self, other: "BM25Index") -> None:
        """
        Merge another BM25Index into this one (incremental indexing).

        Args:
            other: Another BM25Index to merge
        """
        if not other.corpus:
            logger.warning("Other BM25Index is empty, skipping merge")
            return

        logger.info(f"Merging BM25 index with {len(other.corpus)} documents...")

        # Track base index for doc_id mapping
        base_idx = len(self.metadata)

        # Extend corpus and metadata
        self.corpus.extend(other.corpus)
        self.chunk_ids.extend(other.chunk_ids)
        self.tokenized_corpus.extend(other.tokenized_corpus)
        self.metadata.extend(other.metadata)

        # Update doc_id_map with offset indices using centralized utility
        PersistenceManager.update_doc_id_indices(self.doc_id_map, other.doc_id_map, base_idx)

        # Rebuild BM25 index with merged corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"BM25 merge complete: {len(self.corpus)} total documents")

    def save(self, path: Path) -> None:
        """
        Save BM25 index to disk using hybrid serialization.

        Args:
            path: Base path (without extension) to save files
                  Creates: {path}_config.json and {path}_arrays.pkl
        """
        path = Path(path)

        # Config (JSON - human-readable)
        config = {"corpus_count": len(self.corpus), "format_version": "1.0"}
        config_path = path.parent / f"{path.stem}_config.json"
        PersistenceManager.save_json(config_path, config)

        # Arrays (Pickle - performance)
        arrays = {
            "corpus": self.corpus,
            "chunk_ids": self.chunk_ids,
            "metadata": self.metadata,
            "doc_id_map": self.doc_id_map,
        }
        arrays_path = path.parent / f"{path.stem}_arrays.pkl"
        PersistenceManager.save_pickle(arrays_path, arrays)

        logger.info(f"BM25 index saved: {config_path}, {arrays_path}")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """
        Load BM25 index from disk with backward compatibility.

        Supports:
        - Old format: {path}.pkl (single pickle file)
        - New format: {path}_config.json + {path}_arrays.pkl

        Args:
            path: Base path to load from

        Returns:
            BM25Index instance
        """
        path = Path(path)

        # Try new format first
        config_path = path.parent / f"{path.stem}_config.json"
        arrays_path = path.parent / f"{path.stem}_arrays.pkl"

        if config_path.exists() and arrays_path.exists():
            # New format - hybrid serialization
            logger.info(f"Loading BM25 index (new format): {path}")

            # Load arrays
            arrays = PersistenceManager.load_pickle(arrays_path)

            index = cls()
            index.corpus = arrays["corpus"]
            index.chunk_ids = arrays["chunk_ids"]
            index.metadata = arrays["metadata"]
            index.doc_id_map = arrays["doc_id_map"]

        else:
            # Old format - single pickle file
            logger.warning(
                f"Loading BM25 index (old format): {path}. " "Consider re-saving in new format."
            )

            data = PersistenceManager.load_pickle(path)

            index = cls()
            index.corpus = data["corpus"]
            index.chunk_ids = data["chunk_ids"]
            index.metadata = data["metadata"]
            index.doc_id_map = data["doc_id_map"]

        # Rebuild BM25 index and tokenized corpus
        index.tokenized_corpus = [index._tokenize(doc) for doc in index.corpus]

        # Handle empty corpus (skip BM25 initialization to avoid division by zero)
        if index.tokenized_corpus:
            index.bm25 = BM25Okapi(index.tokenized_corpus)
        else:
            index.bm25 = None
            logger.warning("Loaded empty BM25 index (0 documents) - BM25 model not initialized")

        logger.info(f"BM25 index loaded: {len(index.corpus)} documents")

        return index


class BM25Store:
    """
    Multi-layer BM25 store (mirrors FAISSVectorStore structure).

    Creates 3 separate BM25 indexes:
    - Layer 1: Document-level (1 per document)
    - Layer 2: Section-level (N per document)
    - Layer 3: Chunk-level (M per document) - PRIMARY
    """

    def __init__(self):
        """Initialize empty BM25 stores for 3 layers."""
        self.index_layer1 = BM25Index()
        self.index_layer2 = BM25Index()
        self.index_layer3 = BM25Index()

        logger.info("BM25Store initialized: 3 layers")

    def build_from_chunks(self, chunks_dict: Dict[str, List]) -> None:
        """
        Build all 3 BM25 indexes from chunks dict.

        Args:
            chunks_dict: Dict with keys 'layer1', 'layer2', 'layer3'
        """
        logger.info("Building BM25 indexes for all layers...")

        self.index_layer1.build_from_chunks(chunks_dict["layer1"])
        self.index_layer2.build_from_chunks(chunks_dict["layer2"])
        self.index_layer3.build_from_chunks(chunks_dict["layer3"])

        logger.info(
            f"BM25 indexes built: "
            f"L1={len(self.index_layer1.corpus)}, "
            f"L2={len(self.index_layer2.corpus)}, "
            f"L3={len(self.index_layer3.corpus)}"
        )

    def search_layer1(self, query: str, k: int = 1) -> List[Dict]:
        """Search Layer 1 (Document level)."""
        return self.index_layer1.search(query, k)

    def search_layer2(
        self, query: str, k: int = 3, document_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search Layer 2 (Section level)."""
        return self.index_layer2.search(query, k, document_filter)

    def search_layer3(
        self, query: str, k: int = 50, document_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search Layer 3 (Chunk level - PRIMARY)."""
        return self.index_layer3.search(query, k, document_filter)

    def merge(self, other: "BM25Store") -> None:
        """
        Merge another BM25Store into this one (incremental indexing).

        Args:
            other: Another BM25Store to merge
        """
        logger.info("Merging BM25 stores...")

        self.index_layer1.merge(other.index_layer1)
        self.index_layer2.merge(other.index_layer2)
        self.index_layer3.merge(other.index_layer3)

        logger.info("BM25 store merge complete")

    def save(self, output_dir: Path) -> None:
        """
        Save all 3 BM25 indexes.

        Args:
            output_dir: Directory to save indexes
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving BM25 store to {output_dir}")

        self.index_layer1.save(output_dir / "bm25_layer1.pkl")
        self.index_layer2.save(output_dir / "bm25_layer2.pkl")
        self.index_layer3.save(output_dir / "bm25_layer3.pkl")

        logger.info("BM25 store saved")

    @classmethod
    def load(cls, input_dir: Path) -> "BM25Store":
        """
        Load all 3 BM25 indexes.

        Args:
            input_dir: Directory containing indexes

        Returns:
            BM25Store instance
        """
        input_dir = Path(input_dir)
        logger.info(f"Loading BM25 store from {input_dir}")

        store = cls()
        store.index_layer1 = BM25Index.load(input_dir / "bm25_layer1.pkl")
        store.index_layer2 = BM25Index.load(input_dir / "bm25_layer2.pkl")
        store.index_layer3 = BM25Index.load(input_dir / "bm25_layer3.pkl")

        logger.info("BM25 store loaded")

        return store


class HybridVectorStore:
    """
    Hybrid vector store combining FAISS (dense) + BM25 (sparse) with RRF fusion.

    Architecture:
    - Wraps FAISSVectorStore for dense retrieval
    - Wraps BM25Store for sparse retrieval
    - Applies Reciprocal Rank Fusion (RRF) to combine rankings
    - Supports all 3 layers (document, section, chunk)

    RRF Formula:
        RRF_score(chunk) = 1/(k + rank_dense) + 1/(k + rank_sparse)
        where k=60 (research-validated constant)

    Based on research:
    - LegalBench-RAG: +23% precision improvement over dense-only
    - RRF k=60 optimal for legal documents
    - Retrieve k_dense=50, k_sparse=50, fuse to top_k
    """

    def __init__(self, faiss_store, bm25_store: BM25Store, fusion_k: int = 60):  # FAISSVectorStore
        """
        Initialize hybrid vector store.

        Args:
            faiss_store: FAISSVectorStore instance (existing dense retrieval)
            bm25_store: BM25Store instance (new sparse retrieval)
            fusion_k: RRF constant (default: 60, research-optimal)
        """
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.fusion_k = fusion_k

        logger.info(
            f"HybridVectorStore initialized: "
            f"fusion_k={fusion_k}, "
            f"dense_dims={faiss_store.dimensions}"
        )

    def hierarchical_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k_layer3: int = 6,
        use_doc_filtering: bool = True,
        similarity_threshold_offset: float = 0.25,
    ) -> Dict[str, List[Dict]]:
        """
        Hierarchical hybrid search across all layers with RRF fusion.

        Strategy:
        1. Search Layer 1 (Document) - hybrid fusion
        2. Use document_id to filter Layer 3 search (DRM prevention)
        3. Search Layer 3 (PRIMARY) - hybrid fusion with k=50 each
        4. Optional Layer 2 (Section context) - hybrid fusion

        Args:
            query_text: Query text for BM25 sparse retrieval
            query_embedding: Query embedding for FAISS dense retrieval
            k_layer3: Number of final chunks to return (default: 6)
            use_doc_filtering: Enable document-level filtering (DRM prevention)
            similarity_threshold_offset: Threshold = top_score - offset

        Returns:
            Dict with keys 'layer1', 'layer2', 'layer3' containing fused results
        """
        logger.info("Performing hierarchical hybrid search...")

        results = {}

        # Step 1: Layer 1 (Document identification) - Hybrid
        dense_l1 = self.faiss_store.search_layer1(query_embedding, k=3)
        sparse_l1 = self.bm25_store.search_layer1(query_text, k=3)
        fused_l1 = self._rrf_fusion(dense_l1, sparse_l1, k=1)
        results["layer1"] = fused_l1

        # Get document filter
        document_filter = None
        if use_doc_filtering and fused_l1:
            document_filter = fused_l1[0]["document_id"]
            logger.info(f"Document filter: {document_filter}")

        # Step 2: Layer 3 (PRIMARY - Chunk level) - Hybrid
        # Retrieve more candidates for RRF (k=50 each, fuse to top k_layer3)
        dense_l3 = self.faiss_store.search_layer3(
            query_embedding=query_embedding, k=50, document_filter=document_filter
        )
        sparse_l3 = self.bm25_store.search_layer3(
            query=query_text, k=50, document_filter=document_filter
        )
        fused_l3 = self._rrf_fusion(dense_l3, sparse_l3, k=k_layer3)

        # Apply similarity threshold (optional)
        if fused_l3 and similarity_threshold_offset > 0:
            top_score = fused_l3[0]["rrf_score"]
            threshold = top_score - similarity_threshold_offset
            fused_l3 = [r for r in fused_l3 if r["rrf_score"] >= threshold]
            logger.info(
                f"Similarity threshold applied: {threshold:.4f} "
                f"({len(fused_l3)} chunks remaining)"
            )

        # Section-aware deduplication to avoid redundant chunks
        fused_l3 = self._deduplicate_by_section(fused_l3, max_results=k_layer3)

        results["layer3"] = fused_l3

        # Step 3: Optional Layer 2 (Section context) - Hybrid
        dense_l2 = self.faiss_store.search_layer2(
            query_embedding=query_embedding, k=10, document_filter=document_filter
        )
        sparse_l2 = self.bm25_store.search_layer2(
            query=query_text, k=10, document_filter=document_filter
        )
        fused_l2 = self._rrf_fusion(dense_l2, sparse_l2, k=3)
        results["layer2"] = fused_l2

        logger.info(
            f"Hierarchical hybrid search complete: "
            f"L1={len(results['layer1'])}, "
            f"L2={len(results['layer2'])}, "
            f"L3={len(results['layer3'])}"
        )

        return results

    def merge(self, other: "HybridVectorStore") -> None:
        """
        Merge another HybridVectorStore into this one (incremental indexing).

        Args:
            other: Another HybridVectorStore to merge

        Raises:
            ValueError: If fusion_k parameters don't match
        """
        if self.fusion_k != other.fusion_k:
            logger.warning(
                f"Fusion k mismatch: {self.fusion_k} vs {other.fusion_k}. "
                f"Using this store's value: {self.fusion_k}"
            )

        logger.info("Merging hybrid vector stores...")

        # Merge FAISS stores
        self.faiss_store.merge(other.faiss_store)

        # Merge BM25 stores
        self.bm25_store.merge(other.bm25_store)

        logger.info("Hybrid vector store merge complete")

    def _deduplicate_by_section(
        self,
        chunks: List[Dict],
        max_results: int
    ) -> List[Dict]:
        """
        Deduplicate chunks so we don't return multiple results from the same section.

        Preference order:
        1. Keep the first occurrence (highest-ranked) for each (document_id, section_id)
        2. If section metadata missing, fall back to chunk_id
        3. Once unique sections are exhausted, refill with remaining chunks until max_results
        """
        if not chunks:
            return []

        seen_sections = set()
        unique_chunks = []
        duplicates = []

        for chunk in chunks:
            section_id = chunk.get("section_id") or chunk.get("section_title")
            doc_id = chunk.get("document_id", "")
            dedup_key = (doc_id, section_id) if section_id else (doc_id, chunk.get("chunk_id"))

            if dedup_key not in seen_sections:
                seen_sections.add(dedup_key)
                unique_chunks.append(chunk)
            else:
                duplicates.append(chunk)

        # Refill with duplicates if we need more results
        deduped = unique_chunks[:max_results]
        if len(deduped) < max_results:
            for chunk in duplicates:
                deduped.append(chunk)
                if len(deduped) >= max_results:
                    break

        if len(deduped) != len(chunks):
            logger.info(
                f"Section-aware deduplication removed {len(chunks) - len(deduped)} redundant chunks"
            )

        return deduped

    def _rrf_fusion(
        self, dense_results: List[Dict], sparse_results: List[Dict], k: int
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF).

        Combines dense and sparse rankings using RRF formula:
            RRF_score = 1/(fusion_k + rank_dense) + 1/(fusion_k + rank_sparse)

        Preserves intermediate scores (bm25_score, dense_score) for debugging
        and explainability via explain_search_results tool.

        Args:
            dense_results: Results from FAISS (with 'chunk_id', 'score')
            sparse_results: Results from BM25 (with 'chunk_id', 'score')
            k: Number of final results to return

        Returns:
            Sorted list of results with 'rrf_score', 'bm25_score', 'dense_score', 'fusion_method'
        """
        rrf_scores = defaultdict(float)
        all_chunks = {}
        dense_scores = {}  # Track original dense scores
        sparse_scores = {}  # Track original sparse (BM25) scores

        # Accumulate scores from dense retrieval (by rank)
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] += 1.0 / (self.fusion_k + rank)
            all_chunks[chunk_id] = result
            dense_scores[chunk_id] = result.get("score", 0.0)

        # Accumulate scores from sparse retrieval (by rank)
        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] += 1.0 / (self.fusion_k + rank)
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = result
            sparse_scores[chunk_id] = result.get("score", 0.0)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results (top k) with all scores preserved
        results = []
        for chunk_id, rrf_score in sorted_ids[:k]:
            result = all_chunks[chunk_id].copy()
            result["rrf_score"] = rrf_score
            result["bm25_score"] = sparse_scores.get(chunk_id, 0.0)
            result["dense_score"] = dense_scores.get(chunk_id, 0.0)
            result["fusion_method"] = "rrf"
            results.append(result)

        logger.info(
            f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(results)} fused (k={self.fusion_k})"
        )

        return results

    def save(self, output_dir: Path) -> None:
        """
        Save both FAISS and BM25 stores.

        Args:
            output_dir: Directory to save stores
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving hybrid store to {output_dir}")

        # Save FAISS store (uses existing save method)
        self.faiss_store.save(output_dir)

        # Save BM25 store
        self.bm25_store.save(output_dir)

        # Save config (JSON - human-readable)
        config = {"fusion_k": self.fusion_k, "format_version": "1.0"}
        PersistenceManager.save_json(output_dir / "hybrid_config.json", config)

        logger.info("Hybrid store saved")

    @classmethod
    def load(cls, input_dir: Path) -> "HybridVectorStore":
        """
        Load both FAISS and BM25 stores with backward compatibility.

        Supports:
        - Old format: hybrid_config.pkl
        - New format: hybrid_config.json

        Args:
            input_dir: Directory containing stores

        Returns:
            HybridVectorStore instance
        """
        try:
            from src.faiss_vector_store import FAISSVectorStore
        except ImportError:
            from .faiss_vector_store import FAISSVectorStore

        input_dir = Path(input_dir)
        logger.info(f"Loading hybrid store from {input_dir}")

        # Load FAISS store
        faiss_store = FAISSVectorStore.load(input_dir)

        # Load BM25 store
        bm25_store = BM25Store.load(input_dir)

        # Load config (try new format first, fallback to old)
        json_config_path = input_dir / "hybrid_config.json"
        pkl_config_path = input_dir / "hybrid_config.pkl"

        if json_config_path.exists():
            # New format (JSON)
            config = PersistenceManager.load_json(json_config_path)
        elif pkl_config_path.exists():
            # Old format (pickle)
            logger.warning(
                "Loading hybrid config (old format). " "Consider re-saving in new format."
            )
            config = PersistenceManager.load_pickle(pkl_config_path)
        else:
            raise FileNotFoundError(
                f"No hybrid config found in {input_dir}. "
                "Expected hybrid_config.json or hybrid_config.pkl"
            )

        logger.info("Hybrid store loaded")

        return cls(faiss_store=faiss_store, bm25_store=bm25_store, fusion_k=config["fusion_k"])

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a chunk by its ID from metadata_layer3.

        Used by graph_search tool for entity_mentions mode.

        Args:
            chunk_id: The chunk ID to retrieve

        Returns:
            Dict with chunk metadata and content, or None if not found
        """
        # Search in FAISS metadata_layer3
        for meta in self.faiss_store.metadata_layer3:
            if meta.get("chunk_id") == chunk_id:
                return meta

        # Not found
        return None

    def get_stats(self) -> Dict:
        """
        Get combined statistics from both stores.

        Returns:
            Dict with statistics from FAISS and BM25
        """
        faiss_stats = self.faiss_store.get_stats()

        return {
            **faiss_stats,
            "hybrid_enabled": True,
            "fusion_k": self.fusion_k,
            "bm25_layer1_count": len(self.bm25_store.index_layer1.corpus),
            "bm25_layer2_count": len(self.bm25_store.index_layer2.corpus),
            "bm25_layer3_count": len(self.bm25_store.index_layer3.corpus),
        }


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from src.faiss_vector_store import FAISSVectorStore
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

    print("=== PHASE 5B: Hybrid Search Example ===\n")

    # Load existing FAISS vector store
    print("1. Loading FAISS vector store...")
    vector_store = FAISSVectorStore.load(Path("output/vector_store"))
    print(f"   Loaded: {vector_store.get_stats()}\n")

    # Build BM25 indexes (assuming we have chunks dict from pipeline)
    # In practice, this is done during indexing by IndexingPipeline
    print("2. Building BM25 indexes...")
    print("   (Normally done by IndexingPipeline during indexing)\n")

    # Create hybrid store
    print("3. Creating hybrid store...")
    # hybrid_store = HybridVectorStore(
    #     faiss_store=vector_store,
    #     bm25_store=bm25_store,
    #     fusion_k=60
    # )
    # print(f"   Hybrid store ready: {hybrid_store.get_stats()}\n")

    # Search with hybrid retrieval
    print("4. Hybrid search example:")
    print("   query_text = 'safety specification requirements'")
    print("   query_embedding = embedder.embed_texts([query_text])")
    print("")
    print("   results = hybrid_store.hierarchical_search(")
    print("       query_text=query_text,")
    print("       query_embedding=query_embedding,")
    print("       k_layer3=6")
    print("   )")
    print("")
    print("   # Results contain RRF-fused chunks:")
    print("   for chunk in results['layer3']:")
    print("       print(f'RRF Score: {chunk[\"rrf_score\"]:.4f}')")
    print("       print(f'Content: {chunk[\"content\"][:100]}...')")
    print("")

    # Save
    print("5. Saving hybrid store:")
    print("   hybrid_store.save(Path('output/hybrid_store'))")
    print("")

    # Load
    print("6. Loading hybrid store:")
    print("   loaded = HybridVectorStore.load(Path('output/hybrid_store'))")
    print("")

    print("=== Implementation complete! ===")
