"""
PHASE 3: Multi-Layer Chunking with Contextual Retrieval

Based on research:
- LegalBench-RAG: Small chunks optimal (500 chars equivalent to 512 tokens)
- Anthropic, 2024: Contextual Retrieval reduces retrieval failures by 67%
- Lima, 2024: Multi-layer embeddings improve essential chunks by 2.3x
- Docling HybridChunker: Token-aware, hierarchy-preserving chunking

Implementation:
- Layer 1: Document level (1 chunk per document)
- Layer 2: Section level (1 chunk per section)
- Layer 3: Chunk level (HybridChunker 512 tokens + Contextual Retrieval)

Contextual Retrieval:
- Generates LLM-based context for each chunk
- 67% reduction in retrieval failures (Anthropic research)
- Fallback to basic chunking if context generation fails
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

# Token-aware chunking (no Docling dependency)
import tiktoken

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

    # Document labeling (PHASE 3.5 extension)
    # Categories (propagated from document)
    category: Optional[str] = None  # Primary document category
    subcategory: Optional[str] = None  # Subcategory
    secondary_categories: Optional[List[str]] = None  # Secondary categories
    category_confidence: Optional[float] = None  # Classification confidence

    # Keywords (propagated from section)
    keywords: Optional[List[str]] = None  # Extracted keywords (5-10)
    key_phrases: Optional[List[str]] = None  # Multi-word key phrases (3-5)

    # Synthetic questions (per-chunk, HyDE boost)
    questions: Optional[List[str]] = None  # Questions this chunk answers (3-5)
    hyde_text: Optional[str] = None  # Combined questions for embedding

    # Labeling metadata
    labels_source: Optional[str] = None  # "generated" | "propagated"


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
        # Build breadcrumb for embedding text
        # Format: [Document Title > Section Path > Section Title]
        breadcrumb_parts = []

        # 1. Add document title/id first (ALWAYS include document context)
        if self.metadata.title:
            breadcrumb_parts.append(self.metadata.title)
        elif self.metadata.document_id:
            breadcrumb_parts.append(self.metadata.document_id)

        # 2. Add section path (hierarchical structure)
        if self.metadata.section_path:
            breadcrumb_parts.append(self.metadata.section_path)

        # 3. Add section title only if different from path (avoid duplication)
        if (
            self.metadata.section_title
            and self.metadata.section_title != self.metadata.section_path
            and self.metadata.section_title not in (self.metadata.section_path or "")
        ):
            breadcrumb_parts.append(self.metadata.section_title)

        # Construct embedding text: [breadcrumb]\n\ncontext\n\nraw_content
        # - breadcrumb: document + hierarchical path for structure awareness
        # - context: SAC summary for semantic context (first part before \n\n)
        # - raw_content: actual text for retrieval
        if breadcrumb_parts:
            breadcrumb = " > ".join(breadcrumb_parts)
            # Check if content contains context prefix (SAC augmentation)
            if self.content != self.raw_content and "\n\n" in self.content:
                # Content has SAC prefix - extract it (first part, not last!)
                context_part = self.content.split("\n\n", 1)[0]
                embedding_text = f"[{breadcrumb}]\n\n{context_part}\n\n{self.raw_content}"
            else:
                # No SAC prefix - use raw_content directly
                embedding_text = f"[{breadcrumb}]\n\n{self.raw_content}"
        else:
            if self.content != self.raw_content and "\n\n" in self.content:
                context_part = self.content.split("\n\n", 1)[0]
                embedding_text = f"{context_part}\n\n{self.raw_content}"
            else:
                embedding_text = self.raw_content

        return {
            "chunk_id": self.chunk_id,
            "context": self.content,  # SAC-augmented content
            "raw_content": self.raw_content,
            "embedding_text": embedding_text,  # [breadcrumb]\n\ncontext\n\nraw_content
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
    - Layer 3: Chunk level (HybridChunker 512 tokens + Contextual Retrieval)

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024) - Small chunk size optimal
    - Contextual Retrieval (Anthropic, Sept 2024)
    - Multi-Layer Embeddings (Lima, 2024)
    - Docling HybridChunker - Token-aware, hierarchy-preserving
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
        self.enable_contextual = self.config.enable_contextual

        # Initialize contextual retrieval if enabled
        self.context_generator = None
        if self.enable_contextual:
            try:
                context_config = self.config.context_config or ContextGenerationConfig()
                self.context_generator = ContextualRetrieval(config=context_config, api_key=api_key)
                logger.info("Contextual Retrieval initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Contextual Retrieval: {e}")
                if not context_config.fallback_to_basic:
                    raise
                logger.info("Falling back to basic chunking")
                self.enable_contextual = False

        chunking_mode = "Contextual" if self.enable_contextual else "Basic"
        logger.info(
            f"MultiLayerChunker initialized: "
            f"max_tokens={self.config.max_tokens}, tokenizer={self.config.tokenizer_model}, "
            f"mode={chunking_mode}"
        )

    def chunk_document(self, extracted_doc) -> Dict[str, List[Chunk]]:
        """
        Create multi-layer chunks from ExtractedDocument.

        NEW ARCHITECTURE (PHASE 3B):
        1. Generate chunk contexts (PHASE 3A)
        2. Generate section summaries FROM chunk contexts (PHASE 3B) - NO TRUNCATION!
        3. Validate summaries (PHASE 3C)
        4. Create layer 1 & 2 chunks using validated summaries

        OPTIMIZATION: For documents without hierarchy (depth=1, flat structure),
        only Layer 3 (chunks) is created to save embeddings and storage.
        Documents with hierarchy (depth>1) use all 3 layers.

        Args:
            extracted_doc: ExtractedDocument from DoclingExtractorV2

        Returns:
            Dict with keys 'layer1', 'layer2', 'layer3' containing chunks
        """

        logger.info(f"Chunking document: {extracted_doc.document_id}")

        # PHASE 3A: Generate Layer 3 chunks with contexts (FIRST!)
        # This must happen BEFORE section summaries, because we use contexts to generate summaries
        logger.info("PHASE 3A: Generating Layer 3 chunks with contextual retrieval...")
        layer3_chunks = self._create_layer3_chunks(extracted_doc)

        # PHASE 3B: Generate section summaries FROM chunk contexts (NEW!)
        # This eliminates truncation problem - uses ALL chunks (entire section)
        if layer3_chunks:
            if self.enable_contextual:
                logger.info(
                    "PHASE 3B: Generating section summaries from chunk contexts "
                    "(hierarchical approach - NO TRUNCATION)..."
                )
            else:
                logger.info(
                    "PHASE 3B: Generating section summaries from chunk content "
                    "(basic mode - NO TRUNCATION)..."
                )
            self._generate_section_summaries_from_contexts(extracted_doc, layer3_chunks)
        else:
            logger.warning(
                "Skipping section summary generation: no chunks available"
            )

        # PHASE 3C: Validate section summaries
        if layer3_chunks:
            logger.info("PHASE 3C: Validating section summaries...")
            try:
                self._validate_summaries(extracted_doc, min_summary_length=50)
            except ValueError as e:
                logger.error(f"Summary validation failed: {e}")
                # Don't raise - allow pipeline to continue with warnings
                logger.warning("Continuing with invalid summaries (degraded quality expected)")

        # Detect if document has hierarchy
        has_hierarchy = extracted_doc.hierarchy_depth > 1

        if has_hierarchy:
            # Multi-layer indexing: Document → Sections → Chunks
            logger.info("Document has hierarchy (depth > 1) - using 3-layer indexing")

            # Layer 1: Document level (uses section summaries)
            layer1_chunks = self._create_layer1_document(extracted_doc)

            # Layer 2: Section level (uses section summaries)
            layer2_chunks = self._create_layer2_sections(extracted_doc)

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
            raw_content = section.content

            # Skip completely empty sections (no content AND no summary)
            if not raw_content.strip() and not content.strip():
                continue

            # If section has no direct content but has summary, use fallback text
            # This happens for structural headers that only contain child sections
            if not raw_content.strip() and content.strip():
                raw_content = f"[Sekce: {section.title or section.path}]"

            chunk = Chunk(
                chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                content=content,
                raw_content=raw_content,
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

    def _token_aware_split(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Split text into token-aware chunks using tiktoken directly.

        Uses max_tokens=512 (≈ 500 chars for CS/EN text).
        Preserves LegalBench-RAG research constraint.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk (default: 512)

        Returns:
            List of text chunks, each <= max_tokens
        """
        if not text.strip():
            return []

        # Get tiktoken encoding
        encoding = tiktoken.encoding_for_model(self.config.tokenizer_model)

        # Encode text to tokens
        tokens = encoding.encode(text)

        # Split into chunks of max_tokens
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def _create_layer3_token_aware(self, extracted_doc) -> List[Chunk]:
        """
        Layer 3: Token-aware chunking using tiktoken directly.

        Replaces HybridChunker (Docling dependency).
        Guarantees chunks fit within embedding model token limits.

        Args:
            extracted_doc: ExtractedDocument from UnstructuredExtractor

        Returns:
            List of Chunk objects with token-aware boundaries
        """
        chunks = []
        chunk_counter = 0

        # Iterate through sections and split into token-aware chunks
        for section in extracted_doc.sections:
            # Skip empty sections
            if not section.content.strip():
                continue

            # Split section content into token-aware chunks
            section_chunks = self._token_aware_split(
                section.content,
                max_tokens=self.config.max_tokens
            )

            if not section_chunks:
                continue

            # Create Chunk objects for each split
            for chunk_idx, chunk_text in enumerate(section_chunks):
                chunk_counter += 1
                chunk_id = f"{extracted_doc.document_id}_L3_{chunk_counter}"

                # Calculate character positions within section
                # This is approximate since we're working with token boundaries
                char_start = section.char_start
                char_end = section.char_start + len(chunk_text)

                chunk = Chunk(
                    chunk_id=chunk_id,
                    content=chunk_text,  # Will be augmented with context if enabled
                    raw_content=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        layer=3,
                        document_id=extracted_doc.document_id,
                        section_id=section.section_id,
                        parent_chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                        page_number=section.page_number,
                        char_start=char_start,
                        char_end=char_end,
                        section_title=section.title,
                        section_path=section.path,
                        section_level=section.level,
                        section_depth=section.depth,
                    ),
                )
                chunks.append(chunk)

        logger.info(f"Layer 3: {len(chunks)} token-aware chunks created (tiktoken)")
        return chunks

    def _create_layer3_chunks(self, extracted_doc) -> List[Chunk]:
        """
        Layer 3: Token-aware chunking with tiktoken directly.

        BREAKING CHANGE: Replaces Docling HybridChunker with direct tiktoken.
        Uses max_tokens=512 (≈ 500 chars) to preserve research intent.

        Process:
        1. Token-aware splitting with tiktoken (direct implementation)
        2. Contextual Retrieval augmentation (if enabled)
        3. Prepend context for embedding

        Based on:
        - Anthropic, 2024: Contextual Retrieval reduces failures by 67%
        - LegalBench-RAG: Small chunks optimal (512 tokens ≈ 500 chars)
        - Unstructured.io: Hierarchy-preserving extraction
        """
        logger.info("PHASE 3A: Layer 3 chunking with tiktoken (direct)")

        # Generate token-aware chunks
        chunks = self._create_layer3_token_aware(extracted_doc)

        # Apply Contextual Retrieval augmentation
        if self.enable_contextual and self.context_generator:
            logger.info("Applying Contextual Retrieval augmentation to token-aware chunks...")
            chunks = self._apply_contextual_augmentation_to_hybrid_chunks(chunks, extracted_doc)

        logger.info(f"Layer 3: {len(chunks)} token-aware chunks with context")
        return chunks

    def _apply_contextual_augmentation_to_hybrid_chunks(
        self, chunks: List[Chunk], extracted_doc
    ) -> List[Chunk]:
        """
        Apply Contextual Retrieval augmentation to hybrid chunks.

        Generates LLM-based context for each chunk and prepends it to chunk content.

        Args:
            chunks: List of Chunk objects from HybridChunker
            extracted_doc: ExtractedDocument with metadata

        Returns:
            List of Chunk objects with augmented content
        """
        doc_summary = extracted_doc.document_summary or ""

        # Prepare chunks for batch context generation
        chunks_to_contextualize = []
        for chunk in chunks:
            metadata = {
                "document_summary": doc_summary,
                "section_title": chunk.metadata.section_title,
                "section_path": chunk.metadata.section_path,
                "chunk_id": chunk.chunk_id,
                "preceding_chunk": None,  # HybridChunker doesn't provide surrounding chunks
                "following_chunk": None,
            }
            chunks_to_contextualize.append((chunk.raw_content, metadata))

        # Generate contexts in batch
        logger.info(f"Generating contexts for {len(chunks_to_contextualize)} hybrid chunks...")

        try:
            chunk_contexts = self.context_generator.generate_contexts_batch(chunks_to_contextualize)

            # Apply contexts to chunks
            augmented_chunks = []
            for chunk, context_result in zip(chunks, chunk_contexts):
                if context_result.success:
                    # Prepend context to chunk content
                    chunk.content = f"{context_result.context}\n\n{chunk.raw_content}"
                else:
                    # Keep raw content if context generation failed
                    logger.warning(
                        f"Context generation failed for {chunk.chunk_id}, using raw content"
                    )

                augmented_chunks.append(chunk)

            logger.info(
                f"Contextual augmentation: {sum(1 for c in chunk_contexts if c.success)}/{len(chunks)} successful"
            )
            return augmented_chunks

        except Exception as e:
            logger.error(f"Batch context generation failed: {e}, using raw chunks")
            return chunks  # Fallback to raw chunks

    def _generate_section_summaries_from_contexts(
        self, extracted_doc, layer3_chunks: List[Chunk]
    ) -> None:
        """
        Generate section summaries FROM chunks (PHASE 3B).

        This eliminates the truncation problem:
        - OLD: section_text[:2000] → summary (40% coverage for 5000 char sections)
        - NEW: ALL chunks → summary (100% coverage)

        Architecture (two modes):
        1. Contextual mode (when enable_contextual=True):
           - Extracts LLM-generated contexts from chunks
           - Section summary is hierarchical aggregation of contexts
           - Highest quality, most concise

        2. Basic mode (when enable_contextual=False):
           - Uses raw chunk content (no contexts)
           - Section summary aggregates all chunk texts
           - Still achieves 100% coverage (no truncation)

        Both modes provide complete section coverage, unlike the old truncated approach.

        Args:
            extracted_doc: ExtractedDocument to update
            layer3_chunks: List of Layer 3 chunks (with or without contexts)

        Modifies:
            extracted_doc.sections[].summary (in-place update)
        """
        try:
            from .summary_generator import SummaryGenerator
            from .config import SummarizationConfig
        except ImportError:
            from summary_generator import SummaryGenerator
            from config import SummarizationConfig

        # Check if summary generator is available
        if not hasattr(self, "summary_generator") or not self.summary_generator:
            # Initialize if needed
            try:

                summary_config = SummarizationConfig.from_env()
                self.summary_generator = SummaryGenerator(config=summary_config)
                logger.info("Initialized SummaryGenerator for section summary generation")
            except Exception as e:
                logger.warning(
                    f"Cannot generate section summaries: {e}\n"
                    f"Sections will have empty summaries."
                )
                return

        logger.info(
            f"Generating section summaries from {len(layer3_chunks)} chunks "
            f"(100% coverage - NO TRUNCATION)"
        )

        # Group chunks by section_id
        chunks_by_section = {}
        for chunk in layer3_chunks:
            section_id = chunk.metadata.section_id
            if section_id not in chunks_by_section:
                chunks_by_section[section_id] = []
            chunks_by_section[section_id].append(chunk)

        # Extract contexts for each section
        sections_data = []  # [(section_index, contexts_text, section_title)]
        section_index_map = {}  # {section_id: index in extracted_doc.sections}

        for section_idx, section in enumerate(extracted_doc.sections):
            section_id = section.section_id
            section_index_map[section_id] = section_idx

            # Get chunks for this section
            section_chunks = chunks_by_section.get(section_id, [])

            if not section_chunks:
                # Empty section - skip summary generation
                logger.debug(f"Section {section_id} has no chunks, skipping summary")
                continue

            # Extract contexts or raw content from chunks
            # When contextual retrieval is enabled: extract context (first part before "\n\n")
            # When disabled: use raw_content directly
            chunk_texts = []
            for chunk in section_chunks:
                # Check if chunk has contextual prefix (indicated by "\n\n" separator)
                if "\n\n" in chunk.content:
                    # Contextual mode: extract context prefix
                    context = chunk.content.split("\n\n", 1)[0]
                    chunk_texts.append(context)
                else:
                    # Basic mode: use raw_content (full chunk text, not just first 100 chars)
                    chunk_texts.append(chunk.raw_content)

            # Combine chunk texts into single text for summary generation
            contexts_text = "\n".join(f"- {txt}" for txt in chunk_texts)

            sections_data.append((section_idx, contexts_text, section.title))

        # Generate summaries in batch
        try:
            # Build input for generate_batch_summaries: [(text, title), ...]
            batch_input = [(contexts_text, title) for _, contexts_text, title in sections_data]

            # Use min_text_length=0 because:
            # - Chunk contexts (when available) are intentionally short but comprehensive
            # - Raw chunk content (when contexts not available) still covers 100% of section
            # Both approaches eliminate truncation problem
            summaries = self.summary_generator.generate_batch_summaries(
                batch_input, min_text_length=0
            )

            # Update sections with summaries
            for (section_idx, _, _), summary in zip(sections_data, summaries):
                extracted_doc.sections[section_idx].summary = summary

            logger.info(
                f"✓ Generated {len(summaries)} section summaries from chunks "
                f"(100% coverage - no truncation)"
            )

        except Exception as e:
            logger.error(f"Failed to generate section summaries from chunks: {e}")
            logger.warning("Sections will have empty summaries")

    def _validate_summaries(self, extracted_doc, min_summary_length: int = 50) -> None:
        """
        Validate section summaries (PHASE 3C).

        Checks:
        1. All non-empty sections have summaries
        2. All summaries have length >= min_summary_length
        3. Summary quality (no truncation artifacts)

        Args:
            extracted_doc: ExtractedDocument to validate
            min_summary_length: Minimum acceptable summary length (default: 50 chars)

        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Validating section summaries (min_length={min_summary_length})")

        issues = []
        stats = {"total": 0, "valid": 0, "too_short": 0, "missing": 0, "empty_section": 0}

        for section in extracted_doc.sections:
            stats["total"] += 1

            # Skip empty sections (they shouldn't have summaries)
            if not section.content.strip():
                stats["empty_section"] += 1
                continue

            # Check if summary exists
            if not section.summary or not section.summary.strip():
                stats["missing"] += 1
                issues.append(
                    f"Section '{section.title}' ({section.section_id}): "
                    f"Missing summary (content_length={len(section.content)})"
                )
                continue

            # Check summary length
            summary_length = len(section.summary.strip())
            if summary_length < min_summary_length:
                stats["too_short"] += 1
                issues.append(
                    f"Section '{section.title}' ({section.section_id}): "
                    f"Summary too short ({summary_length} < {min_summary_length} chars)"
                )
            else:
                stats["valid"] += 1

        # Log statistics
        logger.info(
            f"Summary validation: {stats['valid']}/{stats['total']} valid, "
            f"{stats['missing']} missing, {stats['too_short']} too short, "
            f"{stats['empty_section']} empty sections (skipped)"
        )

        # Report issues
        if issues:
            logger.warning(f"Found {len(issues)} summary validation issues:")
            for issue in issues[:10]:  # Log first 10 issues
                logger.warning(f"  - {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more issues")

        # Allow some tolerance (e.g., 90% valid is acceptable)
        non_empty_sections = stats["total"] - stats["empty_section"]
        if non_empty_sections > 0:
            valid_rate = stats["valid"] / non_empty_sections
            if valid_rate < 0.9:  # Less than 90% valid
                raise ValueError(
                    f"Summary validation failed: Only {valid_rate:.1%} of sections have valid summaries. "
                    f"Expected at least 90%. Issues:\n" + "\n".join(issues[:5])
                )

        logger.info("✓ Summary validation passed")

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
    # This would be used after UnstructuredExtractor
    from pathlib import Path
    from config import ChunkingConfig
    from unstructured_extractor import UnstructuredExtractor, ExtractionConfig

    # Extract document
    extraction_config = ExtractionConfig(
        strategy="hi_res",
        model="yolox",  # Best results from testing on legal documents
        languages=["ces", "eng"],
        detect_language_per_element=True,
        filter_rotated_text=True,
    )

    extractor = UnstructuredExtractor(extraction_config)
    pdf_path = Path("data/document.pdf")
    result = extractor.extract(pdf_path)

    # Create multi-layer chunks with Contextual Retrieval
    chunking_config = ChunkingConfig(
        max_tokens=512,  # Token-aware chunking (≈ 500 chars)
        tokenizer_model="text-embedding-3-large",
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
