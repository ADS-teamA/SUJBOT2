"""
Multi-provider entity labeler for chunk metadata enrichment.

This TransformComponent extracts entities from chunks using LLM providers,
adding entity labels to node metadata for improved retrieval filtering.

Supports:
- OpenAI models (gpt-4o-mini, gpt-4o, etc.) - default
- Gemini models (gemini-2.5-flash, etc.)
- Anthropic Claude models (claude-haiku-4-5, claude-sonnet-4, etc.)

Entity types are aligned with GeminiKGExtractor for consistency:
- Legal: regulation, legal_provision, requirement, permit
- Organizations: organization, person
- Technical: facility, system, safety_function, isotope, dose_limit
- Other: topic, date, location
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
from json_repair import repair_json
from llama_index.core.schema import BaseNode, TransformComponent
from pydantic import Field

logger = logging.getLogger(__name__)

# Entity types from src/graph/gemini_kg_extractor.py (lines 26-48)
# Kept in sync with KG extraction for consistent entity modeling
ENTITY_TYPES = [
    # Legal entities
    "regulation",       # Laws, decrees, regulations
    "legal_provision",  # Specific provisions (paragraphs, articles)
    "requirement",      # Requirements from legal provisions
    "permit",           # Permits, licenses, authorizations
    # Organizations and people
    "organization",     # Authorities, institutions, companies
    "person",           # People mentioned in documents
    # Technical entities (nuclear/technical documents)
    "facility",         # Facilities, objects (nuclear power plant)
    "system",           # Technical systems (cooling system, I&C)
    "safety_function",  # Safety functions
    "isotope",          # Radioactive isotopes (U-235, Cs-137)
    "dose_limit",       # Radiation dose limits
    # Other
    "topic",            # Main document topics
    "date",             # Important dates (effective dates, deadlines)
    "location",         # Places
]

# Language-aware entity extraction prompt
# NOTE: topics removed - use Labeling Pipeline categories instead (SSOT)
# IMPORTANT: Entity names must be in the SAME language as the document
ENTITY_EXTRACTION_PROMPT = """Extract entities from the following legal/technical document text.

## Entity types:
{entity_types}

## Text to analyze:
{text}

## Instructions:
1. Identify ALL significant entities (specific instances, not general concepts)
2. For each entity provide: name (entity name), type (from types above), confidence (0.0-1.0)
3. List all entity types found in the text

## CRITICAL LANGUAGE REQUIREMENT:
- Entity names (the "name" field) MUST be in the SAME LANGUAGE as the source text above
- If the text is in Czech, entity names must be in Czech (e.g., "SÚJB", "vyhláška č. 422/2016 Sb.")
- If the text is in English, entity names must be in English
- DO NOT translate entity names - keep them exactly as they appear in the text

## Return ONLY valid JSON in this format:
{{
  "entities": [
    {{"name": "entity name in document language", "type": "entity_type", "confidence": 0.9}}
  ],
  "types": ["type1", "type2"]
}}
"""


class GeminiEntityLabeler(TransformComponent):
    """
    Extract entities from chunks using Gemini 2.5 Flash.

    Adds to node.metadata:
        - entities: List of entity dicts with name, type, confidence
        - entity_types: List of unique entity types found

    NOTE: topics removed for SSOT - use Labeling Pipeline categories instead.

    Example:
        >>> labeler = GeminiEntityLabeler(batch_size=10, min_confidence=0.6)
        >>> labeled_nodes = labeler(nodes)
        >>> print(labeled_nodes[0].metadata["entities"])
        [{"name": "SÚJB", "type": "organization", "confidence": 0.95}]
    """

    # Pydantic field declarations (required for BaseModel)
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model to use for entity extraction"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of nodes to process per batch"
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for entities"
    )
    max_text_length: int = Field(
        default=4000,
        ge=1,
        description="Maximum text length to send to Gemini"
    )

    # Private field for lazy model initialization
    _model: Optional[Union[genai.GenerativeModel, Any]] = None
    _provider: Optional[str] = None

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        batch_size: int = 10,
        min_confidence: float = 0.6,
        max_text_length: int = 4000,
    ):
        """
        Initialize the entity labeler with automatic provider detection.

        Args:
            model_name: Model name (gemini-2.5-flash, gpt-4o-mini, etc.)
            batch_size: Number of nodes to process per batch
            min_confidence: Minimum confidence threshold for entities
            max_text_length: Maximum text length to send to LLM

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            min_confidence=min_confidence,
            max_text_length=max_text_length,
        )

        # Detect provider from model name
        from src.agent.providers.factory import detect_provider_from_model
        self._provider = detect_provider_from_model(model_name)
        logger.info(f"Entity labeler initialized: model={model_name}, provider={self._provider}")

    @property
    def model(self) -> Any:
        """Lazy initialization of model based on provider."""
        if self._model is None:
            import os

            if self._provider == "google":
                if not os.getenv("GOOGLE_API_KEY"):
                    raise RuntimeError(
                        "GOOGLE_API_KEY environment variable not set. "
                        "Required for Gemini entity labeling. Set it in .env file."
                    )
                self._model = genai.GenerativeModel(self.model_name)
            elif self._provider == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError(
                        "OPENAI_API_KEY environment variable not set. "
                        "Required for GPT entity labeling. Set it in .env file."
                    )
                from openai import OpenAI
                self._model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            elif self._provider == "anthropic":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    raise RuntimeError(
                        "ANTHROPIC_API_KEY environment variable not set. "
                        "Required for Claude entity labeling. Set it in .env file."
                    )
                import anthropic
                self._model = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            else:
                raise ValueError(f"Unsupported provider for entity labeling: {self._provider}")
        return self._model

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """
        Add entity labels to node metadata.

        Args:
            nodes: List of nodes to process

        Returns:
            List of nodes with entity metadata added
        """
        if not nodes:
            return nodes

        logger.info(f"Entity labeling: {len(nodes)} nodes with {self.model_name}")

        # Track failures for summary
        failure_count = 0

        # Process in batches
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i : i + self.batch_size]
            batch_failures = self._process_batch(batch)
            failure_count += batch_failures

            if i > 0 and i % (self.batch_size * 5) == 0:
                logger.info(f"Entity labeling progress: {i}/{len(nodes)} nodes")

        # Summary logging with failure rate
        if failure_count > 0:
            failure_rate = failure_count / len(nodes)
            if failure_rate > 0.5:
                logger.error(
                    f"Entity labeling DEGRADED: {failure_count}/{len(nodes)} nodes failed "
                    f"({failure_rate:.0%}). Check API keys and rate limits."
                )
            else:
                logger.warning(
                    f"Entity labeling completed with {failure_count} failures "
                    f"out of {len(nodes)} nodes ({failure_rate:.0%})"
                )
        else:
            logger.info(f"Entity labeling complete: {len(nodes)} nodes processed successfully")

        return nodes

    def _process_batch(self, nodes: List[BaseNode]) -> int:
        """Process batch of nodes with LLM provider.

        Returns:
            Number of failures in this batch.
        """
        failures = 0
        for node in nodes:
            try:
                # Get text content from node
                text = self._get_node_text(node)
                if not text or len(text.strip()) < 50:
                    # Skip very short or empty texts
                    node.metadata["entities"] = []
                    node.metadata["entity_types"] = []
                    continue

                # Extract entities
                entities = self._extract_entities(text)

                # Filter by confidence and add to metadata
                filtered_entities = [
                    e for e in entities.get("entities", [])
                    if e.get("confidence", 0) >= self.min_confidence
                ]

                node.metadata["entities"] = filtered_entities
                node.metadata["entity_types"] = entities.get("types", [])

            except Exception as e:
                # Preserve node even if labeling fails
                failures += 1
                node_id = getattr(node, 'id_', 'unknown')
                logger.warning(
                    f"Entity extraction failed for node {node_id}: {e}",
                    exc_info=True
                )
                node.metadata["entity_extraction_error"] = str(e)
                node.metadata["entities"] = []
                node.metadata["entity_types"] = []

        return failures

    def _get_node_text(self, node: BaseNode) -> str:
        """Extract text content from node."""
        # Try different attributes
        if hasattr(node, "text") and node.text:
            return node.text
        if hasattr(node, "get_content"):
            return node.get_content()
        return ""

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Call LLM (Gemini, OpenAI, or Anthropic) for entity extraction."""
        # Truncate text if too long
        truncated_text = text[: self.max_text_length]
        if len(text) > self.max_text_length:
            truncated_text += "..."

        # Build prompt
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            text=truncated_text,
        )

        # Call appropriate API
        if self._provider == "google":
            response = self.model.generate_content(prompt)
            response_text = response.text
        elif self._provider == "openai":
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            response_text = response.choices[0].message.content
        elif self._provider == "anthropic":
            response = self.model.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

        # Parse response
        return self._parse_response(response_text)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response with repair fallback."""
        default = {"entities": [], "types": []}

        if not text:
            return default

        try:
            # Try to extract JSON from response
            json_text = self._extract_json(text)
            if json_text:
                return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.debug(f"Initial JSON parse failed: {e}")

        # Try JSON repair
        try:
            repaired = repair_json(text)
            if repaired:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.debug(f"JSON repair also failed: {e}")

        logger.warning(f"Failed to parse entity extraction response (length={len(text)})")
        return default

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text."""
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return text[start : end + 1]

        return None
