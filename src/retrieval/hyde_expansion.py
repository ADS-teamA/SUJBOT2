"""
HyDE + Query Expansion Generator

Generates:
1. HyDE (Hypothetical Document Embeddings) - a synthetic document answering the query
2. Query Expansions - 2 alternative phrasings of the query

Single LLM call for efficiency.

Research:
- HyDE: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- Query Expansion: Standard IR technique for vocabulary mismatch
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HyDEExpansionResult:
    """Result of HyDE + expansion generation."""

    original_query: str
    hyde_document: str
    expansions: List[str]  # Exactly 2 expansions

    @property
    def all_queries(self) -> List[str]:
        """Return all query variants (hyde + expansions)."""
        return [self.hyde_document] + self.expansions


class HyDEExpansionGenerator:
    """
    Generate HyDE document and query expansions in a single LLM call.

    The HyDE document is a hypothetical passage that would answer the query.
    Expansions are paraphrases that help with vocabulary mismatch.

    Example:
        >>> generator = HyDEExpansionGenerator(client)
        >>> result = generator.generate("What is the safety margin for reactor VR-1?")
        >>> print(result.hyde_document)
        "The safety margin for reactor VR-1 is defined as..."
        >>> print(result.expansions)
        ["How is the safety factor calculated for VR-1?", ...]
    """

    def __init__(
        self,
        client,  # DeepInfraClient
        prompt_path: Optional[str] = None,
    ):
        """
        Initialize generator.

        Args:
            client: DeepInfraClient instance
            prompt_path: Path to prompt template (defaults to prompts/hyde_expansion.txt)
        """
        self.client = client

        # Load prompt template
        if prompt_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            prompt_path = project_root / "prompts" / "hyde_expansion.txt"

        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, path: str) -> str:
        """Load prompt template from file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Prompt file not found: {path}. Using default template.")
            return self._default_prompt_template()

        with open(path, "r", encoding="utf-8") as f:
            template = f.read()

        logger.debug(f"Loaded prompt template from {path}")
        return template

    def _default_prompt_template(self) -> str:
        """Fallback prompt template."""
        return """Given the following query, generate:
1. A hypothetical document (2-3 sentences) that would directly answer this query
2. Two alternative phrasings of the query

Query: {query}

Respond in EXACTLY this format:
HYDE: [Your hypothetical document]
EXP1: [First alternative phrasing]
EXP2: [Second alternative phrasing]

Write in the SAME LANGUAGE as the query."""

    def generate(self, query: str) -> HyDEExpansionResult:
        """
        Generate HyDE document and query expansions.

        Args:
            query: Original user query

        Returns:
            HyDEExpansionResult with hyde_document and expansions
        """
        # Format prompt
        prompt = self.prompt_template.format(query=query)

        # Generate with LLM
        try:
            response = self.client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,  # Some creativity for diverse expansions
            )

            # Parse response
            result = self._parse_response(query, response)
            return result

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            # Fallback: return original query as all variants
            return HyDEExpansionResult(
                original_query=query,
                hyde_document=query,  # Fallback to original
                expansions=[query, query],  # Fallback to original
            )

    def _parse_response(self, query: str, response: str) -> HyDEExpansionResult:
        """
        Parse LLM response into structured result.

        Expected format:
            HYDE: [hypothetical document]
            EXP1: [expansion 1]
            EXP2: [expansion 2]
        """
        hyde_doc = query  # Default fallback
        exp1 = query
        exp2 = query

        # Parse HYDE
        hyde_match = re.search(r"HYDE:\s*(.+?)(?=\nEXP1:|$)", response, re.DOTALL)
        if hyde_match:
            hyde_doc = hyde_match.group(1).strip()

        # Parse EXP1
        exp1_match = re.search(r"EXP1:\s*(.+?)(?=\nEXP2:|$)", response, re.DOTALL)
        if exp1_match:
            exp1 = exp1_match.group(1).strip()

        # Parse EXP2
        exp2_match = re.search(r"EXP2:\s*(.+?)$", response, re.DOTALL)
        if exp2_match:
            exp2 = exp2_match.group(1).strip()

        # Log parsed result
        logger.debug(
            f"Parsed HyDE result:\n"
            f"  HYDE: {hyde_doc[:100]}...\n"
            f"  EXP1: {exp1}\n"
            f"  EXP2: {exp2}"
        )

        return HyDEExpansionResult(
            original_query=query,
            hyde_document=hyde_doc,
            expansions=[exp1, exp2],
        )
