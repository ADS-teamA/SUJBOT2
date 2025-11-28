"""
Section-level keyword extractor.

Extracts keywords at section level and propagates to chunks:
- 5-10 keywords per section
- 3-5 key phrases (multi-word terms)
- Uses document category for context
- Supports batch processing via OpenAI Batch API

Keywords are propagated from sections to their child chunks.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.exceptions import APIKeyError, ProviderError, is_recoverable
from src.extraction_models import DocumentSection

logger = logging.getLogger(__name__)


KEYWORD_PROMPT = """Extrahuj klíčová slova z této sekce dokumentu.

## Sekce: {section_title}
## Kategorie dokumentu: {document_category}

## Text sekce:
{section_text}

## Úkol:
1. Extrahuj 5-10 KLÍČOVÝCH SLOV (jednoslovné pojmy)
2. Extrahuj 3-5 KLÍČOVÝCH FRÁZÍ (víceslovné výrazy)
3. Klíčová slova musí být relevantní pro obsah sekce
4. Preferuj české pojmy, ale zachovej odborné anglické termíny

## Vrať POUZE validní JSON:
{{
  "keywords": ["klíčové", "slovo", "další", "pojmy"],
  "key_phrases": ["víceslovný výraz", "odborný termín"]
}}
"""


@dataclass
class SectionKeywords:
    """Keywords extracted from a section."""

    section_id: str
    keywords: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_id": self.section_id,
            "keywords": self.keywords,
            "key_phrases": self.key_phrases,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], section_id: str = "") -> "SectionKeywords":
        """Create from dictionary."""
        return cls(
            section_id=section_id,
            keywords=data.get("keywords", []),
            key_phrases=data.get("key_phrases", []),
        )

    @classmethod
    def default(cls, section_id: str = "") -> "SectionKeywords":
        """Return default on failure."""
        return cls(section_id=section_id, keywords=[], key_phrases=[])


class SectionKeywordExtractor:
    """
    Extract keywords at section level.

    Processes sections in batches, optionally using OpenAI Batch API.
    Keywords are propagated to child chunks.

    Example:
        >>> extractor = SectionKeywordExtractor(model_name="gpt-4o-mini")
        >>> keywords = await extractor.extract_batch(sections, "nuclear_safety")
        >>> print(keywords["section_1"].keywords)
        ["radiation", "safety", "limits"]
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_keywords: int = 10,
        max_phrases: int = 5,
        max_text_length: int = 3000,
    ):
        """
        Initialize keyword extractor.

        Args:
            model_name: LLM model to use
            max_keywords: Maximum keywords per section
            max_phrases: Maximum key phrases per section
            max_text_length: Maximum text length to send to LLM
        """
        self.model_name = model_name
        self.max_keywords = max_keywords
        self.max_phrases = max_phrases
        self.max_text_length = max_text_length

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
                    "OPENAI_API_KEY not set. Required for keyword extraction.",
                    details={"component": "SectionKeywordExtractor"}
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _build_prompt(
        self, section: DocumentSection, document_category: str
    ) -> str:
        """
        Build keyword extraction prompt.

        Args:
            section: Section to extract keywords from
            document_category: Document-level category for context

        Returns:
            Formatted prompt
        """
        # Truncate text if needed
        text = section.content or ""
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length] + "..."

        return KEYWORD_PROMPT.format(
            section_title=section.title or "Untitled Section",
            document_category=document_category,
            section_text=text,
        )

    def _parse_response(
        self, response_text: str, section_id: str
    ) -> SectionKeywords:
        """
        Parse LLM response into SectionKeywords.

        Args:
            response_text: Raw LLM response
            section_id: Section identifier

        Returns:
            Parsed keywords or default
        """
        try:
            # Extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start >= 0 and end > start:
                json_str = response_text[start : end + 1]
                data = json.loads(json_str)

                keywords = data.get("keywords", [])
                key_phrases = data.get("key_phrases", [])

                # Limit counts
                keywords = keywords[: self.max_keywords]
                key_phrases = key_phrases[: self.max_phrases]

                return SectionKeywords(
                    section_id=section_id,
                    keywords=keywords,
                    key_phrases=key_phrases,
                )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse keywords for {section_id}: {e}")

        # Try json_repair
        try:
            from json_repair import repair_json

            repaired = repair_json(response_text)
            if repaired:
                data = json.loads(repaired)
                return SectionKeywords.from_dict(data, section_id)
        except Exception as e:
            logger.warning(f"JSON repair also failed for {section_id}: {e}")

        return SectionKeywords.default(section_id)

    def extract_sync(
        self, section: DocumentSection, document_category: str
    ) -> SectionKeywords:
        """
        Extract keywords synchronously (real-time API).

        Args:
            section: Section to process
            document_category: Document category for context

        Returns:
            Extracted keywords
        """
        if not section.content or len(section.content.strip()) < 50:
            return SectionKeywords.default(section.section_id or "")

        prompt = self._build_prompt(section, document_category)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            response_text = response.choices[0].message.content or ""
            return self._parse_response(response_text, section.section_id or "")

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
                    f"Keyword extraction API error: {e}",
                    details={"model": self.model_name, "section_id": section.section_id},
                    cause=e
                )
            logger.error(f"Keyword extraction failed for {section.section_id}: {e}", exc_info=True)
            return SectionKeywords.default(section.section_id or "")

    async def extract_async(
        self, section: DocumentSection, document_category: str
    ) -> SectionKeywords:
        """
        Extract keywords asynchronously.

        Args:
            section: Section to process
            document_category: Document category for context

        Returns:
            Extracted keywords
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.extract_sync, section, document_category
        )

    def extract_batch_sync(
        self, sections: List[DocumentSection], document_category: str
    ) -> Dict[str, SectionKeywords]:
        """
        Extract keywords for multiple sections synchronously.

        Args:
            sections: Sections to process
            document_category: Document category for context

        Returns:
            Dict mapping section_id -> SectionKeywords
        """
        results = {}

        for section in sections:
            section_id = section.section_id or str(id(section))
            keywords = self.extract_sync(section, document_category)
            results[section_id] = keywords

        return results

    def create_batch_request(
        self,
        section: DocumentSection,
        document_category: str,
        custom_id: str,
    ) -> Dict[str, Any]:
        """
        Create Batch API request for keyword extraction.

        Args:
            section: Section to process
            document_category: Document category
            custom_id: Unique request ID

        Returns:
            Batch API request dictionary
        """
        prompt = self._build_prompt(section, document_category)

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500,
            },
        }

    def create_batch_requests(
        self, sections: List[DocumentSection], document_category: str
    ) -> List[Dict[str, Any]]:
        """
        Create Batch API requests for all sections.

        Args:
            sections: Sections to process
            document_category: Document category

        Returns:
            List of batch requests
        """
        requests = []

        for section in sections:
            if not section.content or len(section.content.strip()) < 50:
                continue

            section_id = section.section_id or str(id(section))
            request = self.create_batch_request(
                section, document_category, f"kw_{section_id}"
            )
            requests.append(request)

        return requests

    def propagate_to_chunk(
        self,
        section_keywords: SectionKeywords,
        chunk_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Propagate section keywords to chunk.

        Optionally filters keywords based on chunk content.

        Args:
            section_keywords: Keywords from parent section
            chunk_text: Optional chunk text for filtering

        Returns:
            Keyword metadata for chunk
        """
        keywords = section_keywords.keywords.copy()
        key_phrases = section_keywords.key_phrases.copy()

        # Optionally filter to keywords that appear in chunk
        if chunk_text:
            chunk_lower = chunk_text.lower()
            keywords = [kw for kw in keywords if kw.lower() in chunk_lower]
            key_phrases = [kp for kp in key_phrases if kp.lower() in chunk_lower]

            # If filtering removed all keywords, keep originals
            if not keywords and section_keywords.keywords:
                keywords = section_keywords.keywords[:5]
            if not key_phrases and section_keywords.key_phrases:
                key_phrases = section_keywords.key_phrases[:2]

        return {
            "keywords": keywords,
            "key_phrases": key_phrases,
            "keywords_source": "propagated_from_section",
        }
