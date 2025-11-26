"""
Gemini 2.5 Flash entity labeler for chunk metadata enrichment.

This TransformComponent extracts entities from chunks using Gemini 2.5 Flash,
adding entity labels to node metadata for improved retrieval filtering.

Entity types are aligned with GeminiKGExtractor for consistency:
- Legal: regulation, legal_provision, requirement, permit
- Organizations: organization, person
- Technical: facility, system, safety_function, isotope, dose_limit
- Other: topic, date, location
"""

import json
import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from json_repair import repair_json
from llama_index.core.schema import BaseNode, TransformComponent

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

# Prompt optimized for Czech legal/technical documents
ENTITY_EXTRACTION_PROMPT = """Extrahuj entity z následujícího textu českého právního/technického dokumentu.

## Typy entit:
{entity_types}

## Text k analýze:
{text}

## Pokyny:
1. Identifikuj VŠECHNY významné entity
2. Pro každou entitu uveď: name (název), type (typ z výše uvedených), confidence (0.0-1.0)
3. Uveď také seznam hlavních typů entit a témat v dokumentu

## Vrať POUZE validní JSON ve formátu:
{{
  "entities": [
    {{"name": "název entity", "type": "typ_entity", "confidence": 0.9}}
  ],
  "types": ["typ1", "typ2"],
  "topics": ["téma1", "téma2"]
}}
"""


class GeminiEntityLabeler(TransformComponent):
    """
    Extract entities from chunks using Gemini 2.5 Flash.

    Adds to node.metadata:
        - entities: List of entity dicts with name, type, confidence
        - entity_types: List of unique entity types found
        - topics: List of topics identified in the chunk

    Example:
        >>> labeler = GeminiEntityLabeler(batch_size=10, min_confidence=0.6)
        >>> labeled_nodes = labeler(nodes)
        >>> print(labeled_nodes[0].metadata["entities"])
        [{"name": "SÚJB", "type": "organization", "confidence": 0.95}]
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        batch_size: int = 10,
        min_confidence: float = 0.6,
        max_text_length: int = 4000,
    ):
        """
        Initialize the entity labeler.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
            batch_size: Number of nodes to process per batch
            min_confidence: Minimum confidence threshold for entities
            max_text_length: Maximum text length to send to Gemini
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.max_text_length = max_text_length

        # Initialize Gemini model
        self._model: Optional[genai.GenerativeModel] = None

    @property
    def model(self) -> genai.GenerativeModel:
        """Lazy initialization of Gemini model."""
        if self._model is None:
            self._model = genai.GenerativeModel(self.model_name)
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

        # Process in batches
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i : i + self.batch_size]
            self._process_batch(batch)

            if i > 0 and i % (self.batch_size * 5) == 0:
                logger.info(f"Entity labeling progress: {i}/{len(nodes)} nodes")

        logger.info(f"Entity labeling complete: {len(nodes)} nodes processed")
        return nodes

    def _process_batch(self, nodes: List[BaseNode]) -> None:
        """Process batch of nodes with Gemini."""
        for node in nodes:
            try:
                # Get text content from node
                text = self._get_node_text(node)
                if not text or len(text.strip()) < 50:
                    # Skip very short or empty texts
                    node.metadata["entities"] = []
                    node.metadata["entity_types"] = []
                    node.metadata["topics"] = []
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
                node.metadata["topics"] = entities.get("topics", [])

            except Exception as e:
                # Preserve node even if labeling fails
                logger.warning(f"Entity extraction failed for node: {e}")
                node.metadata["entity_extraction_error"] = str(e)
                node.metadata["entities"] = []
                node.metadata["entity_types"] = []
                node.metadata["topics"] = []

    def _get_node_text(self, node: BaseNode) -> str:
        """Extract text content from node."""
        # Try different attributes
        if hasattr(node, "text") and node.text:
            return node.text
        if hasattr(node, "get_content"):
            return node.get_content()
        return ""

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Call Gemini for entity extraction."""
        # Truncate text if too long
        truncated_text = text[: self.max_text_length]
        if len(text) > self.max_text_length:
            truncated_text += "..."

        # Build prompt
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            text=truncated_text,
        )

        # Call Gemini
        response = self.model.generate_content(prompt)

        # Parse response
        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response with repair fallback."""
        default = {"entities": [], "types": [], "topics": []}

        if not text:
            return default

        try:
            # Try to extract JSON from response
            json_text = self._extract_json(text)
            if json_text:
                return json.loads(json_text)
        except json.JSONDecodeError:
            pass

        # Try JSON repair
        try:
            repaired = repair_json(text)
            if repaired:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass

        logger.warning("Failed to parse entity extraction response")
        return default

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text."""
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return text[start : end + 1]

        return None
