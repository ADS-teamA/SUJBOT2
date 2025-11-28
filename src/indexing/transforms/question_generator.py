"""
Chunk-level synthetic question generator for HyDE retrieval boost.

Generates questions that each chunk can answer:
- 3-5 natural questions per chunk
- Questions are added to embedding_text (+20-30% retrieval precision)
- Uses Batch API for cost efficiency (50% savings)

This is the most expensive labeling step (per-chunk), but has the highest
impact on retrieval quality.

HyDE Boost Mechanism:
- User queries are QUESTIONS
- Chunk content is STATEMENTS
- Synthetic questions BRIDGE this semantic gap
- Embedding similarity: query ↔ questions > query ↔ statements
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.exceptions import APIKeyError, ProviderError, is_recoverable
from src.utils.cache import LRUCache

logger = logging.getLogger(__name__)


QUESTION_PROMPT = """Vytvoř otázky, které tento chunk dokumentu zodpovídá.

## Chunk textu:
{chunk_text}

## Kontext:
- Dokument: {document_title}
- Sekce: {section_path}
- Kategorie: {category}

## Úkol:
Vytvoř 3-5 přirozených otázek v ČEŠTINĚ, které by uživatel mohl položit
a tento chunk by na ně odpověděl.

## Pravidla:
- Otázky musí být specifické pro obsah chunku
- Používej přirozený jazyk (jak by se ptal člověk)
- Typy otázek: co, jak, kdy, kdo, proč, jaký
- Otázky musí být zodpověditelné tímto chunkem

## Vrať POUZE validní JSON:
{{
  "questions": [
    "Jaká je ...?",
    "Kdy nastává ...?",
    "Kdo je odpovědný za ...?",
    "Jak se provádí ...?",
    "Proč je důležité ...?"
  ]
}}
"""


@dataclass
class ChunkQuestions:
    """Synthetic questions for a chunk."""

    chunk_id: str
    questions: List[str] = field(default_factory=list)
    hyde_text: str = ""  # Combined questions for embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "questions": self.questions,
            "hyde_text": self.hyde_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], chunk_id: str = "") -> "ChunkQuestions":
        """Create from dictionary."""
        questions = data.get("questions", [])
        hyde_text = " ".join(questions) if questions else ""
        return cls(
            chunk_id=chunk_id,
            questions=questions,
            hyde_text=hyde_text,
        )

    @classmethod
    def default(cls, chunk_id: str = "") -> "ChunkQuestions":
        """Return default on failure."""
        return cls(chunk_id=chunk_id, questions=[], hyde_text="")


class ChunkQuestionGenerator:
    """
    Generate synthetic questions for chunks (HyDE boost).

    Each chunk gets 3-5 questions that it can answer.
    Questions are combined into hyde_text for embedding augmentation.

    Example:
        >>> generator = ChunkQuestionGenerator(model_name="gpt-4o-mini")
        >>> questions = await generator.generate_batch(chunks, context)
        >>> print(questions["chunk_1"].hyde_text)
        "Jaká je limita dávky? Kdo kontroluje radiační ochranu?"
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_questions: int = 5,
        max_text_length: int = 2000,
        cache_enabled: bool = True,
        cache_size: int = 1000,
    ):
        """
        Initialize question generator.

        Args:
            model_name: LLM model to use
            max_questions: Maximum questions per chunk
            max_text_length: Maximum text length to send to LLM
            cache_enabled: Enable content-hash caching
            cache_size: Maximum cache entries
        """
        self.model_name = model_name
        self.max_questions = max_questions
        self.max_text_length = max_text_length

        # Cache (content-hash based)
        self._cache: Optional[LRUCache[ChunkQuestions]] = None
        if cache_enabled:
            self._cache = LRUCache[ChunkQuestions](
                max_size=cache_size, name="question_generator_cache"
            )

        # LLM client (lazy init)
        self._client = None

        # Statistics
        self._processed_count = 0
        self._cache_hits = 0

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            import os

            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise APIKeyError(
                    "OPENAI_API_KEY not set. Required for question generation.",
                    details={"component": "ChunkQuestionGenerator"}
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _get_cache_key(self, text: str) -> str:
        """Generate content-hash cache key."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def _build_prompt(
        self,
        chunk_text: str,
        document_title: str,
        section_path: str,
        category: str,
    ) -> str:
        """
        Build question generation prompt.

        Args:
            chunk_text: Chunk content
            document_title: Document title
            section_path: Section path/breadcrumb
            category: Document category

        Returns:
            Formatted prompt
        """
        # Truncate text if needed
        text = chunk_text
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length] + "..."

        return QUESTION_PROMPT.format(
            chunk_text=text,
            document_title=document_title or "Unknown",
            section_path=section_path or "Unknown",
            category=category or "general",
        )

    def _parse_response(
        self, response_text: str, chunk_id: str
    ) -> ChunkQuestions:
        """
        Parse LLM response into ChunkQuestions.

        Args:
            response_text: Raw LLM response
            chunk_id: Chunk identifier

        Returns:
            Parsed questions or default
        """
        try:
            # Extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start >= 0 and end > start:
                json_str = response_text[start : end + 1]
                data = json.loads(json_str)

                questions = data.get("questions", [])
                questions = questions[: self.max_questions]

                # Build hyde_text (combined questions for embedding)
                hyde_text = " ".join(questions)

                return ChunkQuestions(
                    chunk_id=chunk_id,
                    questions=questions,
                    hyde_text=hyde_text,
                )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse questions for {chunk_id}: {e}")

        # Try json_repair
        try:
            from json_repair import repair_json

            repaired = repair_json(response_text)
            if repaired:
                data = json.loads(repaired)
                return ChunkQuestions.from_dict(data, chunk_id)
        except Exception as e:
            logger.warning(f"JSON repair also failed for {chunk_id}: {e}")

        return ChunkQuestions.default(chunk_id)

    def generate_sync(
        self,
        chunk_id: str,
        chunk_text: str,
        document_title: str,
        section_path: str,
        category: str,
    ) -> ChunkQuestions:
        """
        Generate questions synchronously (real-time API).

        Args:
            chunk_id: Chunk identifier
            chunk_text: Chunk content
            document_title: Document title
            section_path: Section path
            category: Document category

        Returns:
            Generated questions
        """
        if not chunk_text or len(chunk_text.strip()) < 50:
            return ChunkQuestions.default(chunk_id)

        # Check cache
        cache_key = self._get_cache_key(chunk_text)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                # Update chunk_id (cache is content-based)
                cached.chunk_id = chunk_id
                return cached

        prompt = self._build_prompt(
            chunk_text, document_title, section_path, category
        )

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,  # Slightly higher for variety
                max_tokens=400,
            )
            response_text = response.choices[0].message.content or ""
            questions = self._parse_response(response_text, chunk_id)

            # Cache result
            if self._cache and questions.questions:
                self._cache.set(cache_key, questions)

            self._processed_count += 1
            return questions

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
                    f"Question generation API error: {e}",
                    details={"model": self.model_name, "chunk_id": chunk_id},
                    cause=e
                )
            logger.error(f"Question generation failed for {chunk_id}: {e}", exc_info=True)
            return ChunkQuestions.default(chunk_id)

    async def generate_async(
        self,
        chunk_id: str,
        chunk_text: str,
        document_title: str,
        section_path: str,
        category: str,
    ) -> ChunkQuestions:
        """
        Generate questions asynchronously.

        Args:
            chunk_id: Chunk identifier
            chunk_text: Chunk content
            document_title: Document title
            section_path: Section path
            category: Document category

        Returns:
            Generated questions
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_sync,
            chunk_id,
            chunk_text,
            document_title,
            section_path,
            category,
        )

    def create_batch_request(
        self,
        chunk_id: str,
        chunk_text: str,
        document_title: str,
        section_path: str,
        category: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Create Batch API request for question generation.

        Args:
            chunk_id: Chunk identifier
            chunk_text: Chunk content
            document_title: Document title
            section_path: Section path
            category: Document category

        Returns:
            Batch API request dictionary or None if cached/skipped
        """
        if not chunk_text or len(chunk_text.strip()) < 50:
            return None

        # Check cache
        cache_key = self._get_cache_key(chunk_text)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return None  # Skip - already cached

        prompt = self._build_prompt(
            chunk_text, document_title, section_path, category
        )

        return {
            "custom_id": f"q_{chunk_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 400,
            },
        }

    def process_batch_response(
        self,
        custom_id: str,
        response_text: str,
        chunk_text: str,
    ) -> ChunkQuestions:
        """
        Process single response from Batch API.

        Args:
            custom_id: Request ID (q_chunk_id)
            response_text: LLM response
            chunk_text: Original chunk text (for caching)

        Returns:
            Parsed questions
        """
        # Extract chunk_id from custom_id
        chunk_id = custom_id[2:] if custom_id.startswith("q_") else custom_id

        questions = self._parse_response(response_text, chunk_id)

        # Cache result
        if self._cache and questions.questions and chunk_text:
            cache_key = self._get_cache_key(chunk_text)
            self._cache.set(cache_key, questions)

        self._processed_count += 1
        return questions

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        cache_stats = self._cache.get_stats() if self._cache else {"enabled": False}
        return {
            "processed": self._processed_count,
            "cache_hits": self._cache_hits,
            "cache": cache_stats,
        }

    def augment_embedding_text(
        self,
        original_embedding_text: str,
        questions: ChunkQuestions,
    ) -> str:
        """
        Augment embedding text with synthetic questions (HyDE boost).

        Args:
            original_embedding_text: Original chunk embedding text
            questions: Generated questions

        Returns:
            Augmented embedding text
        """
        if not questions.hyde_text:
            return original_embedding_text

        # Append questions to embedding text
        return f"{original_embedding_text}\n\n[Otázky: {questions.hyde_text}]"
