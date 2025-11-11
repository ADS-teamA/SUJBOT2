"""
Classifier Agent - Content categorization and organization.

Responsibilities:
1. Document type classification (Contract, Policy, Report, etc.)
2. Domain identification (Legal, Technical, Financial, etc.)
3. Complexity assessment
4. Language detection and sensitivity classification
"""

import logging
from typing import Any, Dict

from anthropic import Anthropic

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("classifier")
class ClassifierAgent(BaseAgent):
    """
    Classifier Agent - Categorizes documents and content.

    Classifies along multiple dimensions: document type, domain,
    complexity, language, and sensitivity level.
    """

    def __init__(self, config):
        """Initialize classifier with config."""
        super().__init__(config)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=config.api_key)

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("classifier")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"ClassifierAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify documents and content.

        Args:
            state: Current workflow state with extracted documents

        Returns:
            Updated state with classification results
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping classification")
            return state

        logger.info("Classifying extracted documents...")

        try:
            # Get document list for classification context
            doc_list_result = await self.tool_adapter.execute(
                tool_name="get_document_list",
                inputs={},
                agent_name=self.config.name
            )

            # Classify documents based on extracted content
            classification = await self._classify_content(
                query=query,
                extractor_output=extractor_output,
                doc_list=doc_list_result.get("data", []) if doc_list_result["success"] else []
            )

            # Update state with classification
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["classifier"] = classification

            logger.info(
                f"Classification complete: type={classification.get('document_type')}, "
                f"domain={classification.get('domain')}, "
                f"confidence={classification.get('confidence')}%"
            )

            return state

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Classification error: {str(e)}")
            return state

    async def _classify_content(
        self,
        query: str,
        extractor_output: Dict[str, Any],
        doc_list: list
    ) -> Dict[str, Any]:
        """
        Classify content using LLM.

        Args:
            query: User query
            extractor_output: Extracted documents and chunks
            doc_list: List of available documents

        Returns:
            Classification result dict
        """
        # Prepare classification prompt
        chunks_summary = self._summarize_chunks(extractor_output.get("chunks", []))
        doc_summaries = extractor_output.get("document_summaries", [])

        user_message = f"""Classify the following content:

Query: {query}

Extracted Documents:
{self._format_doc_summaries(doc_summaries)}

Content Summary:
{chunks_summary}

Provide classification in the JSON format specified in the system prompt."""

        # Call Anthropic API
        try:
            api_params = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.system_prompt,
                "messages": [{"role": "user", "content": user_message}]
            }

            if self.config.enable_prompt_caching:
                api_params["system"] = [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]

            response = self.client.messages.create(**api_params)
            response_text = response.content[0].text

            # Parse classification from response
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group(0))
                return classification
            else:
                raise ValueError("No JSON found in classification response")

        except Exception as e:
            logger.error(f"Classification LLM call failed: {e}", exc_info=True)
            return self._get_fallback_classification()

    def _summarize_chunks(self, chunks: list) -> str:
        """Summarize chunks for classification prompt."""
        if not chunks:
            return "No chunks available"

        summary_lines = []
        for i, chunk in enumerate(chunks[:5], 1):  # First 5 chunks
            text = chunk.get("text", "")[:200]  # First 200 chars
            summary_lines.append(f"{i}. {text}...")

        return "\n".join(summary_lines)

    def _format_doc_summaries(self, doc_summaries: list) -> str:
        """Format document summaries for prompt."""
        if not doc_summaries:
            return "No documents"

        lines = []
        for doc in doc_summaries:
            lines.append(
                f"- {doc.get('filename')}: {doc.get('summary', 'No summary')[:150]}"
            )

        return "\n".join(lines)

    def _get_fallback_classification(self) -> Dict[str, Any]:
        """Fallback classification if LLM fails."""
        return {
            "document_type": "unknown",
            "domain": "unknown",
            "complexity": "medium",
            "language": "mixed",
            "sensitivity": "internal",
            "confidence": 30,
            "tags": [],
            "reasoning": "Fallback classification due to processing error"
        }
