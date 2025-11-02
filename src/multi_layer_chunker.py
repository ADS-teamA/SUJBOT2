"""
PHASE 3: Multi-Layer Chunking with Contextual Retrieval

Based on research:
- LegalBench-RAG: RecursiveCharacterTextSplitter > Fixed-size (Prec@1: 6.41% vs 2.40%)
- Anthropic, 2024: Contextual Retrieval reduces retrieval failures by 67%
- Lima, 2024: Multi-layer embeddings improve essential chunks by 2.3x

Implementation:
- Layer 1: Document level (1 chunk per document)
- Layer 2: Section level (1 chunk per section)
- Layer 3: Chunk level (RCTS 500 chars + Contextual Retrieval)

Contextual Retrieval:
- Generates LLM-based context for each chunk
- 67% reduction in retrieval failures (Anthropic research)
- Fallback to basic chunking if context generation fails
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import contextual retrieval
try:
    from .contextual_retrieval import ContextualRetrieval
    from .config import ChunkingConfig, ContextGenerationConfig
except ImportError:
    from contextual_retrieval import ContextualRetrieval
    from config import ChunkingConfig, ContextGenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""

    chunk_id: str
    layer: int  # 1=document, 2=section, 3=chunk
    document_id: str
    section_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    # Position info
    page_number: int = 0
    char_start: int = 0
    char_end: int = 0

    # Document info (Layer 1)
    title: Optional[str] = None

    # Hierarchy context (Layer 2-3)
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    section_level: int = 0
    section_depth: int = 0

    # Semantic clustering (PHASE 4.5)
    cluster_id: Optional[int] = None  # Semantic cluster assignment
    cluster_label: Optional[str] = None  # Human-readable cluster topic
    cluster_confidence: Optional[float] = None  # Distance to cluster centroid (0-1, lower is better)


@dataclass
class Chunk:
    """
    A single chunk with content and metadata.

    For Layer 3 (chunk level), content includes SAC summary prepended.
    For embedding, use 'content'.
    For generation, use 'raw_content' (without SAC summary).
    """

    chunk_id: str
    content: str  # For embedding (with SAC if Layer 3)
    raw_content: str  # For generation (without SAC)
    metadata: ChunkMetadata

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "raw_content": self.raw_content,
            "metadata": {
                "chunk_id": self.metadata.chunk_id,
                "layer": self.metadata.layer,
                "document_id": self.metadata.document_id,
                "title": self.metadata.title,
                "section_id": self.metadata.section_id,
                "parent_chunk_id": self.metadata.parent_chunk_id,
                "page_number": self.metadata.page_number,
                "char_start": self.metadata.char_start,
                "char_end": self.metadata.char_end,
                "section_title": self.metadata.section_title,
                "section_path": self.metadata.section_path,
                "section_level": self.metadata.section_level,
                "section_depth": self.metadata.section_depth,
            },
        }


class MultiLayerChunker:
    """
    Multi-layer chunker with Contextual Retrieval.

    Creates 3 layers:
    - Layer 1: Document level (summary only)
    - Layer 2: Section level (section summaries)
    - Layer 3: Chunk level (RCTS 500 chars + Contextual Retrieval)

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024)
    - Contextual Retrieval (Anthropic, Sept 2024)
    - Multi-Layer Embeddings (Lima, 2024)
    """

    def __init__(self, config: Optional[ChunkingConfig] = None, api_key: Optional[str] = None):
        """
        Initialize multi-layer chunker.

        Args:
            config: ChunkingConfig instance (uses defaults if None)
            api_key: API key for LLM provider (for contextual retrieval)
        """
        self.config = config or ChunkingConfig()

        # Extract params for convenience
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.enable_contextual = self.config.enable_contextual

        # Initialize RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.config.separators,
        )

        # Initialize contextual retrieval if enabled
        self.context_generator = None
        if self.enable_contextual:
            try:
                context_config = self.config.context_config or ContextGenerationConfig()
                self.context_generator = ContextualRetrieval(config=context_config, api_key=api_key)
                logger.info("Contextual Retrieval initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Contextual Retrieval: {e}")
                if not self.config.context_config.fallback_to_basic:
                    raise
                logger.info("Falling back to basic chunking")
                self.enable_contextual = False

        chunking_mode = "Contextual" if self.enable_contextual else "Basic"
        logger.info(
            f"MultiLayerChunker initialized: "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"mode={chunking_mode}"
        )

    def chunk_document(self, extracted_doc) -> Dict[str, List[Chunk]]:
        """
        Create multi-layer chunks from ExtractedDocument.

        OPTIMIZATION: For documents without hierarchy (depth=1, flat structure),
        only Layer 3 (chunks) is created to save embeddings and storage.
        Documents with hierarchy (depth>1) use all 3 layers.

        Args:
            extracted_doc: ExtractedDocument from DoclingExtractorV2

        Returns:
            Dict with keys 'layer1', 'layer2', 'layer3' containing chunks
        """

        logger.info(f"Chunking document: {extracted_doc.document_id}")

        # Detect if document has hierarchy
        has_hierarchy = extracted_doc.hierarchy_depth > 1

        if has_hierarchy:
            # Multi-layer indexing: Document → Sections → Chunks
            logger.info("Document has hierarchy (depth > 1) - using 3-layer indexing")

            # Layer 1: Document level
            layer1_chunks = self._create_layer1_document(extracted_doc)

            # Layer 2: Section level
            layer2_chunks = self._create_layer2_sections(extracted_doc)

            # Layer 3: Chunk level with SAC
            layer3_chunks = self._create_layer3_chunks(extracted_doc)

            logger.info(
                f"Created {len(layer1_chunks)} L1, "
                f"{len(layer2_chunks)} L2, "
                f"{len(layer3_chunks)} L3 chunks"
            )
        else:
            # Single-layer indexing: Only chunks (no hierarchy)
            logger.info("Document is flat (depth = 1) - using single-layer indexing (Layer 3 only)")

            layer1_chunks = []
            layer2_chunks = []

            # Layer 3: Chunk level with SAC
            layer3_chunks = self._create_layer3_chunks(extracted_doc)

            logger.info(
                f"Created {len(layer3_chunks)} L3 chunks "
                f"(L1 and L2 skipped for flat document)"
            )

        return {
            "layer1": layer1_chunks,
            "layer2": layer2_chunks,
            "layer3": layer3_chunks,
            "total_chunks": len(layer1_chunks) + len(layer2_chunks) + len(layer3_chunks),
        }

    def _create_layer1_document(self, extracted_doc) -> List[Chunk]:
        """
        Layer 1: Document-level chunk (summary only).

        Purpose:
        - Global filtering during retrieval
        - Document identification
        - DRM prevention
        """

        # Use document summary if available, else truncate full text
        content = extracted_doc.document_summary or extracted_doc.full_text[:500]

        chunk = Chunk(
            chunk_id=f"{extracted_doc.document_id}_L1",
            content=content,
            raw_content=content,
            metadata=ChunkMetadata(
                chunk_id=f"{extracted_doc.document_id}_L1",
                layer=1,
                document_id=extracted_doc.document_id,
                title=extracted_doc.title,
                section_id=None,
                parent_chunk_id=None,
                page_number=0,
                char_start=0,
                char_end=len(content),
            ),
        )

        return [chunk]

    def _create_layer2_sections(self, extracted_doc) -> List[Chunk]:
        """
        Layer 2: Section-level chunks.

        Purpose:
        - Mid-level context
        - Section-specific queries
        - Context expansion when needed
        """

        chunks = []

        for section in extracted_doc.sections:
            # Use section summary if available, else section content
            content = section.summary or section.content

            chunk = Chunk(
                chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                content=content,
                raw_content=section.content,  # Always use raw content
                metadata=ChunkMetadata(
                    chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                    layer=2,
                    document_id=extracted_doc.document_id,
                    section_id=section.section_id,
                    parent_chunk_id=f"{extracted_doc.document_id}_L1",
                    page_number=section.page_number,
                    char_start=section.char_start,
                    char_end=section.char_end,
                    section_title=section.title,
                    section_path=section.path,
                    section_level=section.level,
                    section_depth=section.depth,
                ),
            )

            chunks.append(chunk)

        return chunks

    def _create_layer3_chunks(self, extracted_doc) -> List[Chunk]:
        """
        Layer 3: Chunk-level with Contextual Retrieval.

        CRITICAL: This is the PRIMARY chunking layer!

        Process (Contextual Retrieval):
        1. Split each section into 500-char chunks using RCTS
        2. Generate LLM-based context for each chunk (explains what chunk discusses)
        3. Prepend context to chunk for embedding
        4. Keep raw content (without context) for generation
        5. Fallback to basic chunking if context generation fails

        Based on:
        - Anthropic, 2024: Contextual Retrieval reduces failures by 67%
        - LegalBench-RAG: RCTS outperforms fixed-size
        """

        chunks = []

        # Get document summary for context generation
        doc_summary = extracted_doc.document_summary or ""

        # CONTEXTUAL RETRIEVAL mode
        if self.enable_contextual and self.context_generator:
            logger.info("Using Contextual Retrieval for Layer 3")
            chunks = self._create_layer3_contextual(extracted_doc, doc_summary)

        # Basic mode (no augmentation)
        else:
            logger.info("Using basic chunking for Layer 3 (no augmentation)")
            chunks = self._create_layer3_basic(extracted_doc)

        return chunks

    def _create_layer3_contextual(self, extracted_doc, doc_summary: str) -> List[Chunk]:
        """
        Create Layer 3 chunks with Contextual Retrieval.

        Generates LLM-based context for each chunk.
        """
        chunks = []
        chunk_counter = 0

        # Prepare all chunks with metadata for batch context generation
        chunks_to_contextualize = []

        for section in extracted_doc.sections:
            # Skip empty sections
            if not section.content.strip():
                continue

            # Split section into raw chunks
            raw_chunks = self.text_splitter.split_text(section.content)

            for idx, raw_chunk in enumerate(raw_chunks):
                chunk_counter += 1

                # Get surrounding chunks (if enabled in config)
                preceding_chunk = None
                following_chunk = None
                if (
                    self.config.context_config
                    and self.config.context_config.include_surrounding_chunks
                ):
                    num_surrounding = self.config.context_config.num_surrounding_chunks
                    # Get preceding chunk(s)
                    if idx > 0:
                        preceding_chunk = raw_chunks[idx - 1]
                    # Get following chunk(s)
                    if idx < len(raw_chunks) - 1:
                        following_chunk = raw_chunks[idx + 1]

                # Prepare metadata for context generation
                metadata = {
                    "document_summary": doc_summary,
                    "section_title": section.title,
                    "section_path": section.path,
                    "preceding_chunk": preceding_chunk,
                    "following_chunk": following_chunk,
                    "section": section,
                    "idx": idx,
                    "chunk_id": f"{extracted_doc.document_id}_L3_{section.section_id}_chunk_{idx}",
                }

                chunks_to_contextualize.append((raw_chunk, metadata))

        # Generate contexts in batch (parallel)
        logger.info(f"Generating contexts for {len(chunks_to_contextualize)} chunks...")

        try:
            chunk_contexts = self.context_generator.generate_contexts_batch(chunks_to_contextualize)

            # Create Chunk objects with contexts
            for (raw_chunk, metadata), context_result in zip(
                chunks_to_contextualize, chunk_contexts
            ):
                section = metadata["section"]
                idx = metadata["idx"]

                # Use context if successful, otherwise use raw chunk
                if context_result.success:
                    augmented_content = f"{context_result.context}\n\n{raw_chunk}"
                else:
                    # Fallback to basic (no augmentation)
                    logger.warning(
                        f"Context generation failed for chunk {metadata['chunk_id']}, "
                        f"using raw chunk"
                    )
                    augmented_content = raw_chunk

                chunk = Chunk(
                    chunk_id=metadata["chunk_id"],
                    content=augmented_content,  # For embedding (with context)
                    raw_content=raw_chunk,  # For generation (without context)
                    metadata=ChunkMetadata(
                        chunk_id=f"{extracted_doc.document_id}_L3_{section.section_id}_chunk_{idx}",
                        layer=3,
                        document_id=extracted_doc.document_id,
                        section_id=section.section_id,
                        parent_chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                        page_number=section.page_number,
                        char_start=section.char_start,
                        char_end=section.char_end,
                        section_title=section.title,
                        section_path=section.path,
                        section_level=section.level,
                        section_depth=section.depth,
                    ),
                )

                chunks.append(chunk)

        except Exception as e:
            logger.error(f"Batch context generation failed: {e}")
            if self.config.context_config and self.config.context_config.fallback_to_basic:
                logger.warning("Falling back to basic chunking due to context generation failure")
                return self._create_layer3_basic(extracted_doc)
            raise

        return chunks

    def _create_layer3_basic(self, extracted_doc) -> List[Chunk]:
        """
        Create Layer 3 chunks with basic RCTS chunking (no augmentation).

        Used as fallback when Contextual Retrieval is disabled or fails.
        """
        chunks = []

        for section in extracted_doc.sections:
            # Skip empty sections
            if not section.content.strip():
                continue

            # Split section into raw chunks
            raw_chunks = self.text_splitter.split_text(section.content)

            for idx, raw_chunk in enumerate(raw_chunks):
                chunk_id = f"{extracted_doc.document_id}_L3_{section.section_id}_chunk_{idx}"

                chunk = Chunk(
                    chunk_id=chunk_id,
                    content=raw_chunk,  # No augmentation
                    raw_content=raw_chunk,
                    metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        layer=3,
                        document_id=extracted_doc.document_id,
                        section_id=section.section_id,
                        parent_chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                        page_number=section.page_number,
                        char_start=section.char_start,
                        char_end=section.char_end,
                        section_title=section.title,
                        section_path=section.path,
                        section_level=section.level,
                        section_depth=section.depth,
                    ),
                )

                chunks.append(chunk)

        return chunks

    def get_chunking_stats(self, chunks_dict: Dict[str, List[Chunk]]) -> Dict:
        """
        Get statistics about the chunking results.

        Args:
            chunks_dict: Output from chunk_document()

        Returns:
            Dict with chunking statistics
        """

        stats = {
            "layer1_count": len(chunks_dict["layer1"]),
            "layer2_count": len(chunks_dict["layer2"]),
            "layer3_count": len(chunks_dict["layer3"]),
            "total_chunks": chunks_dict["total_chunks"],
        }

        # Layer 3 statistics (PRIMARY layer)
        if chunks_dict["layer3"]:
            layer3_sizes = [len(c.raw_content) for c in chunks_dict["layer3"]]
            stats["layer3_avg_size"] = sum(layer3_sizes) / len(layer3_sizes)
            stats["layer3_min_size"] = min(layer3_sizes)
            stats["layer3_max_size"] = max(layer3_sizes)

            # Context augmentation statistics
            context_sizes = [len(c.content) - len(c.raw_content) for c in chunks_dict["layer3"]]
            stats["context_avg_overhead"] = (
                sum(context_sizes) / len(context_sizes) if context_sizes else 0
            )

        return stats


# Example usage
if __name__ == "__main__":
    # This would be used after DoclingExtractorV2
    from config import ExtractionConfig, ChunkingConfig
    from docling_extractor_v2 import DoclingExtractorV2

    # Extract document
    extraction_config = ExtractionConfig(
        enable_smart_hierarchy=True, generate_summaries=True  # Required for Layer 1 and Layer 2
    )

    extractor = DoclingExtractorV2(extraction_config)
    result = extractor.extract("document.pdf")

    # Create multi-layer chunks with Contextual Retrieval
    chunking_config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=0,
        enable_contextual=True,  # Use Contextual Retrieval (RECOMMENDED)
    )

    chunker = MultiLayerChunker(
        config=chunking_config, api_key="your-api-key"  # For Anthropic/OpenAI
    )

    chunks = chunker.chunk_document(result)
    stats = chunker.get_chunking_stats(chunks)

    print(f"Layer 1: {stats['layer1_count']} chunks")
    print(f"Layer 2: {stats['layer2_count']} chunks")
    print(f"Layer 3: {stats['layer3_count']} chunks (PRIMARY)")
    print(f"Total: {stats['total_chunks']} chunks")
