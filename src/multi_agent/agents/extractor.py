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

        # Initialize provider (auto-detects from model name: claude/gpt/gemini)
        try:
            from src.agent.providers.factory import create_provider

            self.provider = create_provider(model=config.model)
            logger.info(f"Initialized provider for model: {config.model}")
        except Exception as e:
            logger.error(f"Failed to create provider: {e}")
            raise ValueError(
                f"Failed to initialize LLM provider for model {config.model}. "
                f"Ensure API keys are configured in environment and model name is valid."
            ) from e

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("extractor")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        # Retrieval parameters
        self.default_k = 6  # Default number of chunks to retrieve
        self.max_k = 10  # Maximum for complex queries (search tool limit)

        logger.info(f"ExtractorAgent initialized with model: {config.model}")

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
            # Run autonomous tool calling loop
            # LLM decides: search parameters, which documents to fetch, whether to expand context
            result = await self._run_autonomous_tool_loop(
                system_prompt=self.system_prompt,
                state=state,
                max_iterations=10
            )

            # Parse result from autonomous loop
            final_answer = result.get("final_answer", "")
            tool_calls = result.get("tool_calls", [])
            agent_cost = result.get("total_tool_cost_usd", 0.0)

            # Store extraction output
            extraction_output = {
                "analysis": final_answer,
                "tool_calls_made": [t["tool"] for t in tool_calls],
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
