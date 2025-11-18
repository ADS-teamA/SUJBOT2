"""
TIER 3: Analysis & Insights Tools

Deep analysis tools (1-3s) for specialized tasks.
Use for entity analysis, timeline extraction, statistics.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool
from .utils import format_chunk_result

logger = logging.getLogger(__name__)


# ============================================================================
# TIER 3 TOOLS: Analysis & Insights
# ============================================================================


# timeline_view tool removed - primitive regex-based implementation
# Use filtered_search with filter_type='temporal' or knowledge graph entity extraction instead

# summarize_section tool removed - false advertising (returns chunks, not LLM summary)
# Use get_document_info(info_type='section_details') to get section chunks
# Use filtered_search with section filter for more control


# ============================================================================
# UNIFIED TOOLS (Consolidated from multiple similar tools)
# ============================================================================


class GetStatsInput(ToolInput):
    """Input for unified get_stats tool."""

    stat_scope: str = Field(
        ...,
        description="Statistics scope: 'corpus' (overall stats), 'index' (comprehensive index stats with embedding/cache info), 'document' (per-document stats), 'entity' (knowledge graph stats)",
    )
    include_cache_stats: bool = Field(
        False, description="Include embedding cache statistics (for 'index' scope)"
    )


@register_tool
class GetStatsTool(BaseTool):
    """Get corpus/index statistics."""

    name = "get_stats"
    description = "Get corpus/index stats"
    detailed_help = """
    Get statistics about corpus, index, documents, or entities.

    **Stat scopes:**
    - 'corpus': Overall document counts, sizes
    - 'index': Comprehensive FAISS index stats (layers, dimensions, cache)
    - 'document': Per-document statistics
    - 'entity': Knowledge graph entity statistics

    **When to use:**
    - "How many documents?"
    - "Corpus statistics"
    - "Index information"
    - System/debugging queries

    **Best practices:**
    - Use 'corpus' for quick document counts
    - Use 'index' for detailed FAISS information
    - Set include_cache_stats=true for embedding cache info
    - Fast metadata aggregation (no search required)

    **Method:** Metadata aggregation
    **Speed:** <100ms (metadata lookup only)
    """
    tier = 3
    input_schema = GetStatsInput

    def execute_impl(self, stat_scope: str, include_cache_stats: bool = False) -> ToolResult:
        try:
            stats = {}

            if stat_scope in ["corpus", "document", "index"]:
                # Get vector store statistics
                vs_stats = self.vector_store.get_stats()
                stats["vector_store"] = vs_stats

                # Get unique documents count from stats (FAISS provides this)
                if "documents" in vs_stats:
                    stats["unique_documents"] = vs_stats["documents"]

                # Try to get document list from FAISS metadata (if available)
                try:
                    from src.faiss_vector_store import FAISSVectorStore
                    if isinstance(self.vector_store.vector_store, FAISSVectorStore):
                        # Extract document IDs from layer1 metadata (document summaries)
                        unique_docs = set()
                        for metadata in self.vector_store.vector_store.metadata_layer1:
                            doc_id = metadata.get("document_id") or metadata.get("doc_id", "Unknown")
                            unique_docs.add(doc_id)
                        stats["document_list"] = sorted(list(unique_docs))
                except Exception as e:
                    # If we can't get document list, skip it (non-critical)
                    logger.debug(f"Could not extract document list: {e}")
                    stats["document_list"] = []

            if stat_scope in ["corpus", "entity", "index"]:
                # Get knowledge graph statistics
                if self.knowledge_graph:
                    from collections import Counter

                    entity_types = Counter()
                    relationship_types = Counter()

                    for entity in self.knowledge_graph.entities:
                        entity_types[entity.type] += 1

                    for rel in self.knowledge_graph.relationships:
                        relationship_types[rel.type] += 1

                    stats["knowledge_graph"] = {
                        "total_entities": len(self.knowledge_graph.entities),
                        "total_relationships": len(self.knowledge_graph.relationships),
                        "entity_types": {k.value: v for k, v in entity_types.items()},
                        "relationship_types": {k.value: v for k, v in relationship_types.items()},
                    }

                    if stat_scope == "index":
                        # Add top entities for index scope
                        stats["knowledge_graph"]["top_entities"] = [
                            {"id": e.id, "name": e.value, "type": e.type.value}
                            for e in sorted(
                                self.knowledge_graph.entities,
                                key=lambda x: x.confidence,
                                reverse=True,
                            )[:10]
                        ]
                else:
                    stats["knowledge_graph"] = None

            if stat_scope == "index":
                # Additional index-specific information
                stats["hybrid_search_enabled"] = stats.get("vector_store", {}).get(
                    "hybrid_enabled", False
                )

                # Embedding model information
                if self.embedder:
                    stats["embedding_model"] = {
                        "model_name": self.embedder.model_name,
                        "dimensions": self.embedder.dimensions,
                        "model_type": self.embedder.model_type,
                    }

                    # Cache statistics if requested
                    if include_cache_stats and hasattr(self.embedder, "get_cache_stats"):
                        stats["embedding_cache"] = self.embedder.get_cache_stats()

                # Document info
                stats["documents"] = {
                    "count": stats["unique_documents"],
                    "document_ids": stats["document_list"],
                }

                # System configuration
                stats["configuration"] = {
                    "reranking_enabled": hasattr(self, "reranker") and self.reranker is not None,
                    "context_assembler_enabled": hasattr(self, "context_assembler")
                    and self.context_assembler is not None,
                }

            if stat_scope not in ["corpus", "index", "document", "entity"]:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid stat_scope: {stat_scope}. Must be 'corpus', 'index', 'document', or 'entity'",
                )

            return ToolResult(
                success=True,
                data=stats,
                metadata={
                    "stat_scope": stat_scope,
                    "stat_categories": list(stats.keys()),
                },
            )

        except Exception as e:
            logger.error(f"Get stats failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


# ============================================================================
# DEFINITION ALIGNMENT (Legal Terminology Mapping)
# ============================================================================


class DefinitionAlignerInput(ToolInput):
    """Input for definition alignment tool."""

    term: str = Field(
        ...,
        description="Legal term to align (e.g., 'Consumer', 'Personal Data', 'Data Controller')"
    )
    context_document_id: Optional[str] = Field(
        None,
        description="Document ID providing context (e.g., contract being analyzed)"
    )
    reference_laws: Optional[List[str]] = Field(
        None,
        description="List of law document IDs to search for definitions (e.g., ['gdpr', 'ccpa'])"
    )
    similarity_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Minimum semantic similarity for term matching (0.0-1.0)"
    )
    include_related_terms: bool = Field(
        True,
        description="Include semantically related terms (synonyms, hypernyms)"
    )


@register_tool
class DefinitionAlignerTool(BaseTool):
    """
    Align legal definitions across documents using knowledge graph + embeddings.

    Solves the "term mismatch" problem: Law says "Consumer" but contract says "Client".
    Uses knowledge graph + pgvector semantic search to find equivalent terms.
    """

    name = "definition_aligner"
    description = "Align legal terminology across documents using graph and semantic search"
    detailed_help = """
    Align legal definitions to resolve terminology mismatches.

    **Use cases:**
    - Law requires "Data Controller" but contract uses "Data Custodian" → Are they equivalent?
    - Regulation mentions "Consumer" but documentation uses "Client" → Same entity?
    - Standard defines "Quality Management System" vs doc says "QMS" → Alignment?

    **How it works:**
    1. Search knowledge graph for term entity and its definitions
    2. Find related entities via relationships
    3. Use pgvector semantic search for similar terms (embedding similarity)
    4. Return alignment map with confidence scores

    **When to use:**
    - Compliance verification with terminology differences
    - Cross-document requirement mapping
    - Legal definition disambiguation

    **Method:** Knowledge graph traversal + pgvector semantic search
    **Speed:** 500-1000ms (graph query + embedding search)
    **Tier:** 3 (Analysis tool - use when terminology alignment needed)
    """
    tier = 3
    input_schema = DefinitionAlignerInput

    def execute_impl(
        self,
        term: str,
        context_document_id: Optional[str] = None,
        reference_laws: Optional[List[str]] = None,
        similarity_threshold: float = 0.75,
        include_related_terms: bool = True
    ) -> ToolResult:
        try:
            # Sanitize input to prevent injection attacks
            # Remove special characters that could affect graph queries or regex matching
            term_sanitized = re.sub(r'[^\w\s\-_]', '', term)
            if not term_sanitized or not term_sanitized.strip():
                return ToolResult(
                    success=False,
                    error="Invalid term: contains only special characters or is empty after sanitization"
                )

            alignments = []

            # PHASE 1: Knowledge Graph Search for Definitions
            # Find term entity in KG with related definitions (use sanitized term)
            kg_definitions = self._search_kg_definitions(term_sanitized, reference_laws)

            for kg_def in kg_definitions:
                alignments.append({
                    "source": "knowledge_graph",
                    "term": kg_def["term"],
                    "definition": kg_def["definition"],
                    "document": kg_def["document_id"],
                    "breadcrumb": kg_def.get("breadcrumb", "[Unknown]"),
                    "confidence": kg_def["confidence"],
                    "alignment_type": "exact_match" if kg_def["term"].lower() == term_sanitized.lower() else "graph_related"
                })

            # PHASE 2: Semantic Search via pgvector
            # Find semantically similar terms using embedding similarity (use sanitized term)
            if include_related_terms:
                semantic_matches = self._search_semantic_definitions(
                    term_sanitized,
                    context_document_id,
                    reference_laws,
                    similarity_threshold
                )

                for match in semantic_matches:
                    # Avoid duplicates from KG search
                    if not any(a["term"] == match["term"] and a["document"] == match["document_id"]
                              for a in alignments):
                        alignments.append({
                            "source": "semantic_search",
                            "term": match["term"],
                            "definition": match["definition"],
                            "document": match["document_id"],
                            "breadcrumb": match.get("breadcrumb", "[Unknown]"),
                            "confidence": match["similarity_score"],
                            "alignment_type": "semantic_similar"
                        })

            # Sort by confidence descending
            alignments.sort(key=lambda x: x["confidence"], reverse=True)

            # Generate alignment summary (use sanitized term)
            summary = self._generate_alignment_summary(term_sanitized, alignments)

            result_data = {
                "query_term": term_sanitized,  # Return sanitized term
                "original_term": term,  # Keep original for transparency
                "alignments": alignments,
                "summary": summary,
                "total_alignments": len(alignments),
                "filters": {
                    "context_document_id": context_document_id,
                    "reference_laws": reference_laws,
                    "similarity_threshold": similarity_threshold
                }
            }

            return ToolResult(
                success=True,
                data=result_data,
                metadata={
                    "alignments_found": len(alignments),
                    "sources": list(set(a["source"] for a in alignments)),
                    "top_confidence": alignments[0]["confidence"] if alignments else 0.0
                }
            )

        except Exception as e:
            logger.error(f"Definition alignment failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _search_kg_definitions(
        self,
        term: str,
        reference_laws: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge graph for term definitions.

        Uses graph traversal to find:
        1. Exact entity matches for term
        2. Related LEGAL_TERM entities
        3. Definition relationships

        Args:
            term: Legal term to search
            reference_laws: Filter to specific law documents

        Returns:
            List of definition dicts with confidence scores
        """
        if not self.knowledge_graph:
            logger.warning("Knowledge graph not available for definition alignment")
            return []

        definitions = []

        try:
            # Search for term entity (case-insensitive)
            term_lower = term.lower()
            term_entities = [
                e for e in self.knowledge_graph.entities
                if hasattr(e, 'type') and hasattr(e.type, 'value') and
                e.type.value == "legal_term" and term_lower in e.normalized_value.lower()
            ]

            # Filter by reference laws if specified
            if reference_laws:
                term_entities = [
                    e for e in term_entities
                    if any(law_id in e.metadata.get("document_ids", []) for law_id in reference_laws)
                ]

            # Extract definitions from entity metadata and relationships
            for entity in term_entities:
                # Check entity properties for definition
                definition_text = entity.properties.get("definition", "")
                if definition_text:
                    definitions.append({
                        "term": entity.value,
                        "definition": definition_text,
                        "document_id": entity.metadata.get("document_ids", ["Unknown"])[0],
                        "breadcrumb": entity.metadata.get("breadcrumb", "[Unknown]"),
                        "confidence": entity.confidence
                    })

                # Check relationships to definition entities
                if hasattr(self.knowledge_graph, 'get_outgoing_relationships'):
                    for rel in self.knowledge_graph.get_outgoing_relationships(entity.id):
                        if hasattr(rel.type, 'value') and rel.type.value == "definition_of":
                            target = self.knowledge_graph.get_entity(rel.target_entity_id)
                            if target and hasattr(target.type, 'value') and target.type.value == "definition":
                                definitions.append({
                                    "term": entity.value,
                                    "definition": target.value,
                                    "document_id": target.metadata.get("document_ids", ["Unknown"])[0],
                                    "breadcrumb": target.metadata.get("breadcrumb", "[Unknown]"),
                                    "confidence": min(entity.confidence, rel.confidence)
                                })

        except Exception as e:
            logger.warning(f"Knowledge graph search failed: {e}")

        return definitions

    def _search_semantic_definitions(
        self,
        term: str,
        context_document_id: Optional[str],
        reference_laws: Optional[List[str]],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar terms using pgvector.

        Embeds query term and searches vector store for similar chunks
        that contain legal definitions.

        Args:
            term: Legal term to search
            context_document_id: Context document for filtering
            reference_laws: Filter to specific law documents
            similarity_threshold: Minimum cosine similarity

        Returns:
            List of semantic match dicts with similarity scores
        """
        # Construct search query emphasizing definition context
        search_query = f'legal definition of "{term}"'

        # Search vector store
        try:
            search_results = self.vector_store.search(
                query=search_query,
                k=20,  # Get top 20 candidates
                layer=3  # Search chunk layer (most granular)
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        # Filter by document IDs if specified
        if reference_laws or context_document_id:
            allowed_docs = set(reference_laws or [])
            if context_document_id:
                allowed_docs.add(context_document_id)

            search_results = [
                r for r in search_results
                if r.get("doc_id") in allowed_docs
            ]

        # Extract definitions from chunks using heuristics
        definitions = []
        definition_patterns = [
            r'"([^"]+)"\s+means\s+([^.]+)',  # "X" means Y
            r'([A-Z][^:]+):\s+([^.]+)',      # Term: definition
            r'([A-Z][^(]+)\(([^)]+)\)',      # Term (definition)
        ]

        for result in search_results:
            chunk_text = result.get("text", "")
            similarity = result.get("score", 0.0)

            # Only consider chunks above similarity threshold
            if similarity < similarity_threshold:
                continue

            # Try to extract term and definition using patterns
            for pattern in definition_patterns:
                matches = re.findall(pattern, chunk_text)
                for match in matches:
                    if len(match) >= 2:
                        extracted_term = match[0].strip()
                        extracted_def = match[1].strip()

                        # Check if extracted term is related to query term
                        # (simple heuristic: shared words or substring)
                        if (term.lower() in extracted_term.lower() or
                            extracted_term.lower() in term.lower() or
                            len(set(term.lower().split()) & set(extracted_term.lower().split())) > 0):

                            definitions.append({
                                "term": extracted_term,
                                "definition": extracted_def,
                                "document_id": result.get("doc_id", "Unknown"),
                                "breadcrumb": result.get("breadcrumb", "[Unknown]"),
                                "similarity_score": similarity
                            })

        return definitions

    def _generate_alignment_summary(
        self,
        term: str,
        alignments: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable alignment summary.

        Args:
            term: Query term
            alignments: List of alignment dicts

        Returns:
            Summary string
        """
        if not alignments:
            return f"No definitions found for term '{term}'"

        top_alignment = alignments[0]
        summary_parts = [
            f"Term '{term}' aligned with {len(alignments)} definition(s).",
            f"Top match: '{top_alignment['term']}' ({top_alignment['confidence']:.2f} confidence)",
            f"Definition: {top_alignment['definition'][:150]}...",
            f"Source: {top_alignment['breadcrumb']}"
        ]

        # Check for conflicts (multiple high-confidence definitions with different meanings)
        high_conf_defs = [a for a in alignments if a["confidence"] > 0.85]
        if len(high_conf_defs) > 1:
            summary_parts.append(
                f"⚠️ WARNING: {len(high_conf_defs)} high-confidence definitions found. "
                "Manual review recommended for terminology conflicts."
            )

        return " ".join(summary_parts)
