"""
Extractor Agent - Document retrieval and information extraction.

Responsibilities:
1. Hybrid search for document retrieval (BM25 + Dense + RRF)
2. Context expansion around relevant chunks
3. Document metadata and summary retrieval
4. Citation preservation and provenance tracking
"""

import logging
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..core.state import DocumentMetadata
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("extractor")
class ExtractorAgent(BaseAgent):
    """
    Extractor Agent - Retrieves documents and chunks from vector store.

    Uses hybrid search (BM25 + Dense + RRF) for semantic retrieval,
    expands context around relevant chunks, and preserves full citations.
    """

    def __init__(self, config):
        """Initialize extractor with config."""
        super().__init__(config)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=config.api_key)

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("extractor")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        # Retrieval parameters
        self.default_k = 6  # Default number of chunks to retrieve
        self.max_k = 15  # Maximum for complex queries

        logger.info(f"ExtractorAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant documents and chunks.

        Args:
            state: Current workflow state with query

        Returns:
            Updated state with:
                - documents (List[DocumentMetadata])
                - citations (List[str])
                - agent_outputs["extractor"] with extracted content
        """
        query = state.get("query", "")
        complexity_score = state.get("complexity_score", 50)

        if not query:
            logger.error("No query provided in state")
            state["errors"] = state.get("errors", [])
            state["errors"].append("No query provided for extraction")
            return state

        logger.info(f"Extracting documents for query: {query[:100]}...")

        try:
            # Determine retrieval parameters based on complexity
            k = self._determine_k(complexity_score)

            # Step 1: Hybrid search for initial retrieval
            search_results = await self._hybrid_search(query, k=k)

            if not search_results:
                logger.warning("No search results found, trying fallback")
                search_results = await self._hybrid_search(query, k=k * 2)

            # Step 2: Expand context around top chunks
            expanded_results = await self._expand_context(search_results)

            # Step 3: Get document summaries
            document_metadata = await self._get_document_metadata(expanded_results)

            # Step 4: Extract citations
            citations = self._extract_citations(expanded_results)

            # Step 5: Format extraction output for next agents
            extraction_output = self._format_extraction_output(
                search_results=search_results,
                expanded_results=expanded_results,
                document_metadata=document_metadata,
                citations=citations
            )

            # Update state
            state["documents"] = state.get("documents", []) + document_metadata
            state["citations"] = state.get("citations", []) + citations
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["extractor"] = extraction_output

            logger.info(
                f"Extracted {len(search_results)} chunks from "
                f"{len(document_metadata)} documents"
            )

            return state

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Extraction error: {str(e)}")
            return state

    def _determine_k(self, complexity_score: int) -> int:
        """
        Determine number of chunks to retrieve based on complexity.

        Args:
            complexity_score: Query complexity (0-100)

        Returns:
            Number of chunks to retrieve (k)
        """
        if complexity_score < 30:
            return self.default_k  # Simple queries: 6 chunks
        elif complexity_score < 70:
            return min(10, self.max_k)  # Medium queries: 10 chunks
        else:
            return self.max_k  # Complex queries: 15 chunks

    async def _hybrid_search(
        self,
        query: str,
        k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (BM25 + Dense + RRF fusion).

        Args:
            query: Search query
            k: Number of results to retrieve

        Returns:
            List of search results with chunks and metadata
        """
        try:
            result = await self.tool_adapter.execute(
                tool_name="search",
                inputs={"query": query, "k": k},
                agent_name=self.config.name
            )

            if result["success"]:
                return result["data"]
            else:
                logger.error(f"Search failed: {result.get('error')}")
                return []

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            return []

    async def _expand_context(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Expand context around top-ranked chunks.

        Args:
            search_results: Initial search results

        Returns:
            Results with expanded context
        """
        if not search_results:
            return []

        expanded_results = []

        # Expand context for top 3 chunks
        for result in search_results[:3]:
            try:
                chunk_id = result.get("chunk_id")
                if not chunk_id:
                    expanded_results.append(result)
                    continue

                # Get expanded context
                expansion_result = await self.tool_adapter.execute(
                    tool_name="get_chunk_context",
                    inputs={
                        "chunk_id": chunk_id,
                        "context_size": 2  # 2 chunks before/after
                    },
                    agent_name=self.config.name
                )

                if expansion_result["success"]:
                    # Merge expanded context with original result
                    result_with_context = {
                        **result,
                        "expanded_context": expansion_result["data"]
                    }
                    expanded_results.append(result_with_context)
                else:
                    expanded_results.append(result)

            except Exception as e:
                logger.warning(f"Context expansion failed for chunk: {e}")
                expanded_results.append(result)

        # Add remaining results without expansion
        expanded_results.extend(search_results[3:])

        return expanded_results

    async def _get_document_metadata(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[DocumentMetadata]:
        """
        Retrieve document metadata and summaries.

        Args:
            search_results: Search results with document references

        Returns:
            List of DocumentMetadata objects
        """
        if not search_results:
            return []

        # Extract unique document IDs
        doc_ids = set()
        for result in search_results:
            doc_id = result.get("document_id")
            if doc_id:
                doc_ids.add(doc_id)

        metadata_list = []

        # Retrieve metadata for each document
        for doc_id in doc_ids:
            try:
                meta_result = await self.tool_adapter.execute(
                    tool_name="get_document_info",
                    inputs={"document_id": doc_id},
                    agent_name=self.config.name
                )

                if meta_result["success"]:
                    doc_info = meta_result["data"]
                    metadata = DocumentMetadata(
                        document_id=doc_id,
                        filename=doc_info.get("filename", ""),
                        document_type=doc_info.get("document_type", ""),
                        num_pages=doc_info.get("num_pages", 0),
                        summary=doc_info.get("summary", ""),
                        layer=doc_info.get("layer", 1)
                    )
                    metadata_list.append(metadata)

            except Exception as e:
                logger.warning(f"Failed to get metadata for {doc_id}: {e}")

        return metadata_list

    def _extract_citations(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract and format citations from search results.

        Args:
            search_results: Search results with metadata

        Returns:
            List of formatted citations
        """
        citations = []

        for result in search_results:
            try:
                doc_id = result.get("document_id", "unknown")
                filename = result.get("filename", doc_id)
                section = result.get("section_path", "")
                page = result.get("page_number", "")

                # Format citation
                citation_parts = [f"Doc: {filename}"]
                if section:
                    citation_parts.append(f"Section: {section}")
                if page:
                    citation_parts.append(f"Page: {page}")

                citation = "[" + ", ".join(citation_parts) + "]"
                citations.append(citation)

            except Exception as e:
                logger.warning(f"Failed to format citation: {e}")

        return citations

    def _format_extraction_output(
        self,
        search_results: List[Dict[str, Any]],
        expanded_results: List[Dict[str, Any]],
        document_metadata: List[DocumentMetadata],
        citations: List[str]
    ) -> Dict[str, Any]:
        """
        Format extraction output for downstream agents.

        Args:
            search_results: Initial search results
            expanded_results: Results with expanded context
            document_metadata: Document metadata list
            citations: Formatted citations

        Returns:
            Formatted extraction output dict
        """
        return {
            "num_chunks_retrieved": len(search_results),
            "num_documents": len(document_metadata),
            "chunks": expanded_results,
            "document_summaries": [
                {
                    "document_id": meta.document_id,
                    "filename": meta.filename,
                    "summary": meta.summary,
                    "num_pages": meta.num_pages
                }
                for meta in document_metadata
            ],
            "citations": citations,
            "retrieval_method": "hybrid_search_with_context_expansion"
        }
