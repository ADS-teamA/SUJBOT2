"""
Document category extractor with dynamic taxonomy generation.

Extracts document-level categories using LLM:
- Dynamic taxonomy: LLM creates document-specific categories
- Single LLM call per document (smart propagation)
- Categories propagate to sections and chunks

This is LLM-driven (CLAUDE.md compliance) - no hardcoded category rules.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.exceptions import APIKeyError, ProviderError, is_recoverable
from src.extraction_models import DocumentSection, ExtractedDocument

logger = logging.getLogger(__name__)


# Prompt for dynamic taxonomy generation
TAXONOMY_PROMPT = """Analyzuj tento dokument a vytvoř taxonomii kategorií.

## Shrnutí dokumentu:
{document_summary}

## Vzorky sekcí (náhodně vybrané):
{section_samples}

## Úkol:
1. Identifikuj 3-7 hlavních kategorií relevantních PRO TENTO DOKUMENT
2. Pro každou kategorii vytvoř 2-4 subkategorie
3. Klasifikuj celý dokument do nejrelevantnější kategorie

## Pravidla:
- Kategorie musí být specifické pro obsah dokumentu
- Používej české názvy kategorií
- Subkategorie musí být podmnožinou hlavní kategorie
- Confidence musí být 0.0-1.0

## Vrať POUZE validní JSON:
{{
  "taxonomy": {{
    "hlavní_kategorie_1": ["subkategorie_1a", "subkategorie_1b"],
    "hlavní_kategorie_2": ["subkategorie_2a", "subkategorie_2b"]
  }},
  "document_classification": {{
    "primary_category": "hlavní_kategorie",
    "primary_subcategory": "subkategorie",
    "secondary_categories": ["sekundární_kategorie_1"],
    "confidence": 0.95
  }}
}}
"""


@dataclass
class DocumentTaxonomy:
    """Document-specific category taxonomy."""

    taxonomy: Dict[str, List[str]]  # {category: [subcategories]}
    primary_category: str
    primary_subcategory: Optional[str]
    secondary_categories: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "taxonomy": self.taxonomy,
            "primary_category": self.primary_category,
            "primary_subcategory": self.primary_subcategory,
            "secondary_categories": self.secondary_categories,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentTaxonomy":
        """Create from dictionary."""
        classification = data.get("document_classification", {})
        return cls(
            taxonomy=data.get("taxonomy", {}),
            primary_category=classification.get("primary_category", "unknown"),
            primary_subcategory=classification.get("primary_subcategory"),
            secondary_categories=classification.get("secondary_categories", []),
            confidence=classification.get("confidence", 0.0),
        )

    @classmethod
    def default(cls) -> "DocumentTaxonomy":
        """Return default taxonomy on failure."""
        return cls(
            taxonomy={"general": ["uncategorized"]},
            primary_category="general",
            primary_subcategory="uncategorized",
            secondary_categories=[],
            confidence=0.0,
        )


class DocumentCategoryExtractor:
    """
    Extract document-level categories with dynamic taxonomy.

    Uses a single LLM call per document to:
    1. Generate document-specific category taxonomy
    2. Classify the document into the taxonomy

    Categories are then propagated to sections and chunks.

    Example:
        >>> extractor = DocumentCategoryExtractor(model_name="gpt-4o-mini")
        >>> taxonomy = await extractor.extract_taxonomy(document)
        >>> print(taxonomy.primary_category)
        "nuclear_safety"
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        use_dynamic_categories: bool = True,
        fixed_categories: Optional[List[str]] = None,
        sample_sections: int = 10,
        max_section_length: int = 500,
    ):
        """
        Initialize category extractor.

        Args:
            model_name: LLM model to use
            use_dynamic_categories: If True, LLM creates taxonomy; else uses fixed
            fixed_categories: Fixed category list (used if use_dynamic_categories=False)
            sample_sections: Number of sections to sample for context
            max_section_length: Max chars per section sample
        """
        self.model_name = model_name
        self.use_dynamic_categories = use_dynamic_categories
        self.fixed_categories = fixed_categories or [
            "legal",
            "technical",
            "regulatory",
            "safety",
            "administrative",
            "operational",
        ]
        self.sample_sections = sample_sections
        self.max_section_length = max_section_length

        # LLM client (lazy init)
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            import os

            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise APIKeyError(
                    "OPENAI_API_KEY not set. Required for category extraction.",
                    details={"component": "DocumentCategoryExtractor"}
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _sample_sections_text(
        self, sections: List[DocumentSection], n: int
    ) -> str:
        """
        Sample sections for taxonomy generation context.

        Samples evenly across document to capture diversity.

        Args:
            sections: All document sections
            n: Number of sections to sample

        Returns:
            Formatted section samples text
        """
        if not sections:
            return "(No sections available)"

        # Sample evenly across document
        step = max(1, len(sections) // n)
        sampled = sections[::step][:n]

        samples = []
        for i, section in enumerate(sampled):
            title = section.title or f"Section {i + 1}"
            content = (section.content or "")[:self.max_section_length]
            if len(section.content or "") > self.max_section_length:
                content += "..."
            samples.append(f"### {title}\n{content}")

        return "\n\n".join(samples)

    def _build_prompt(self, document: ExtractedDocument) -> str:
        """
        Build taxonomy generation prompt.

        Args:
            document: Extracted document

        Returns:
            Formatted prompt
        """
        section_samples = self._sample_sections_text(
            document.sections, self.sample_sections
        )

        return TAXONOMY_PROMPT.format(
            document_summary=document.document_summary or "(No summary available)",
            section_samples=section_samples,
        )

    def _parse_response(self, response_text: str) -> DocumentTaxonomy:
        """
        Parse LLM response into DocumentTaxonomy.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed taxonomy or default
        """
        try:
            # Extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start >= 0 and end > start:
                json_str = response_text[start : end + 1]
                data = json.loads(json_str)
                return DocumentTaxonomy.from_dict(data)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse taxonomy response: {e}")

        # Try json_repair
        try:
            from json_repair import repair_json

            repaired = repair_json(response_text)
            if repaired:
                data = json.loads(repaired)
                return DocumentTaxonomy.from_dict(data)
        except Exception as e:
            logger.warning(f"JSON repair also failed: {e}")

        return DocumentTaxonomy.default()

    def extract_taxonomy_sync(
        self, document: ExtractedDocument
    ) -> DocumentTaxonomy:
        """
        Extract taxonomy synchronously (real-time API).

        Args:
            document: Extracted document

        Returns:
            Document taxonomy with categories
        """
        prompt = self._build_prompt(document)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            response_text = response.choices[0].message.content or ""
            return self._parse_response(response_text)

        except APIKeyError:
            # Re-raise API key errors - these are not recoverable
            raise
        except Exception as e:
            # Check if recoverable before falling back
            if not is_recoverable(e):
                raise
            # Wrap OpenAI errors as ProviderError for upstream handling
            import openai
            if isinstance(e, (openai.APIError, openai.APIConnectionError, openai.RateLimitError)):
                raise ProviderError(
                    f"Category extraction API error: {e}",
                    details={"model": self.model_name},
                    cause=e
                )
            logger.error(f"Category extraction failed: {e}", exc_info=True)
            return DocumentTaxonomy.default()

    async def extract_taxonomy_async(
        self, document: ExtractedDocument
    ) -> DocumentTaxonomy:
        """
        Extract taxonomy asynchronously.

        Args:
            document: Extracted document

        Returns:
            Document taxonomy with categories
        """
        import asyncio

        # Run sync call in executor (OpenAI client is not async-native)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.extract_taxonomy_sync, document
        )

    def create_batch_request(
        self, document: ExtractedDocument, custom_id: str
    ) -> Dict[str, Any]:
        """
        Create Batch API request for category extraction.

        Args:
            document: Extracted document
            custom_id: Unique request ID

        Returns:
            Batch API request dictionary
        """
        prompt = self._build_prompt(document)

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000,
            },
        }

    def propagate_to_section(
        self, taxonomy: DocumentTaxonomy, section: DocumentSection
    ) -> Dict[str, Any]:
        """
        Propagate document categories to section.

        Args:
            taxonomy: Document taxonomy
            section: Section to label

        Returns:
            Category metadata for section
        """
        return {
            "category": taxonomy.primary_category,
            "subcategory": taxonomy.primary_subcategory,
            "secondary_categories": taxonomy.secondary_categories,
            "category_confidence": taxonomy.confidence,
            "category_source": "propagated_from_document",
        }

    def propagate_to_chunk(
        self, taxonomy: DocumentTaxonomy, section_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Propagate document categories to chunk.

        Args:
            taxonomy: Document taxonomy
            section_path: Optional section path for context

        Returns:
            Category metadata for chunk
        """
        return {
            "category": taxonomy.primary_category,
            "subcategory": taxonomy.primary_subcategory,
            "secondary_categories": taxonomy.secondary_categories,
            "category_confidence": taxonomy.confidence,
            "category_source": "propagated_from_document",
        }
