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

from ..core.agent_base import BaseAgent
from ..core.agent_initializer import initialize_agent
from ..core.agent_registry import register_agent
from ..core.state import DocumentMetadata

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

        # Initialize common components (provider, prompts, tools)
        components = initialize_agent(config, "extractor")
        self.provider = components.provider
        self.system_prompt = components.system_prompt
        self.tool_adapter = components.tool_adapter

        # Retrieval parameters
        self.default_k = 6  # Default number of chunks to retrieve
        self.max_k = 10  # Maximum for complex queries (search tool limit)

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant documents and chunks (AUTONOMOUS).

        LLM autonomously decides search strategy and document retrieval.

        Args:
            state: Current workflow state with query

        Returns:
            Updated state with extracted documents and chunks
        """
        query = state.get("query", "")

        if not query:
            logger.error("No query provided in state")
            state["errors"] = state.get("errors", [])
            state["errors"].append("No query provided for extraction")
            return state

        logger.info(f"Running autonomous document extraction for: {query[:100]}...")

        try:
            # Determine max iterations based on query complexity
            # (set by orchestrator during routing, defaults to 50 = medium)
            complexity_score = state.get("complexity_score", 50)
            if complexity_score < 30:
                # Simple queries: "Co je X?" - 1 search should suffice
                max_iterations = 3
            elif complexity_score < 70:
                # Medium queries: may need search + expand_context
                max_iterations = 5
            else:
                # Complex queries: multi-document, comparative analysis
                max_iterations = 8

            logger.info(
                f"Complexity-aware iterations: complexity={complexity_score}, "
                f"max_iterations={max_iterations}"
            )

            # Run autonomous tool calling loop
            # LLM decides: search parameters, which documents to fetch, whether to expand context
            result = await self._run_autonomous_tool_loop(
                system_prompt=self.system_prompt,
                state=state,
                max_iterations=max_iterations
            )

            # Parse result from autonomous loop
            final_answer = result.get("final_answer", "")
            tool_calls = result.get("tool_calls", [])
            agent_cost = result.get("total_tool_cost_usd", 0.0)

            # Extract chunk_ids from tool results (CRITICAL for report generation)
            # Tool results contain data=[{chunk_id: ..., content: ...}, ...]
            all_chunk_ids = []
            all_citations = []  # Keep breadcrumb citations for context
            all_chunks_data = []  # Store full chunk data for downstream agents

            for tool_call in tool_calls:
                tool_name = tool_call.get("tool", "unknown")
                tool_result = tool_call.get("result", {})
                if not isinstance(tool_result, dict):
                    logger.debug(f"Skipping non-dict tool result from {tool_name}: {type(tool_result)}")
                    continue
                # Extract chunk_ids from data field (list of chunk dicts)
                data = tool_result.get("data", [])
                if not isinstance(data, list):
                    logger.debug(f"Tool {tool_name} returned non-list data: {type(data)}")
                    continue
                for chunk in data:
                    if not isinstance(chunk, dict):
                        logger.debug(f"Skipping non-dict chunk from {tool_name}: {type(chunk)}")
                        continue
                    chunk_id = chunk.get("chunk_id")
                    if not chunk_id:
                        logger.debug(f"Skipping chunk without chunk_id from {tool_name}")
                        continue
                    if chunk_id not in all_chunk_ids:
                        all_chunk_ids.append(chunk_id)
                        all_chunks_data.append(chunk)

                # Also keep breadcrumb citations for context (backwards compatibility)
                citations = tool_result.get("citations", [])
                if isinstance(citations, list):
                    all_citations.extend(citations)

            logger.info(
                f"Extracted {len(all_chunk_ids)} unique chunk_ids and {len(all_citations)} "
                f"breadcrumb citations from {len(tool_calls)} tool calls"
            )

            # Store extraction output WITH CITATIONS
            extraction_output = {
                "analysis": final_answer,
                "tool_calls_made": [t["tool"] for t in tool_calls],
                "chunk_ids": all_chunk_ids,  # PRIMARY: for \cite{chunk_id} format
                "chunks_data": all_chunks_data,  # Full chunk data for synthesis
                "citations": all_citations,  # SECONDARY: breadcrumb citations for context
                "iterations": result.get("iterations", 0),
                "retrieval_method": "autonomous_llm_driven",
                "total_tool_cost_usd": agent_cost
            }

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["extractor"] = extraction_output

            logger.info(
                f"Autonomous extraction complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}"
            )

            # Set final answer if this is only agent in sequence
            agent_sequence = state.get("agent_sequence", [])
            if agent_sequence == ["extractor"]:
                logger.info("Extractor is only agent - using autonomous answer")
                state["final_answer"] = final_answer

            return state

        except Exception as e:
            logger.error(f"Autonomous extraction failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Extraction error: {str(e)}")
            return state


    # Old hardcoded methods removed - autonomous pattern handles everything via LLM
