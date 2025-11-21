"""
HyDE (Hypothetical Document Embeddings) Generator

Generates hypothetical documents that answer queries for improved zero-shot retrieval.

Research basis:
- Gao et al. (2022): "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- Strategy: Generate hypothetical answer → Embed hypothetical answer → Search

Benefits:
- +15-30% recall for zero-shot queries (no training data)
- Bridges vocabulary gap between query and documents
- Works best for: factual questions, domain-specific terminology
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HyDEResult:
    """Result from HyDE generation."""

    original_query: str
    hypothetical_docs: List[str]  # Multiple hypothetical documents
    generation_method: str  # "llm" or "none"
    model_used: Optional[str] = None
    cost_estimate: float = 0.0


class HyDEGenerator:
    """
    Generate hypothetical documents using LLM for better retrieval.

    Strategy (Hybrid HyDE):
    1. For original query: Generate 3 hypothetical documents (multi-hypothesis averaging)
    2. For paraphrases: Generate 1 hypothetical document each (efficiency)

    Design Decisions:
    - Direct I/O loading: Simple Path.read_text() for prompts
    - Template substitution: {query} placeholder for domain-specific prompts
    - Multi-hypothesis averaging: 3 docs for original query (reduces variance)

    Features:
    - Configurable LLM provider (OpenAI/Anthropic)
    - Domain-specific prompts via template files
    - Graceful fallback on errors (returns empty list)
    - Cost tracking via global tracker
    - Prompt file caching (loaded once at init)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        num_hypotheses: int = 3,
    ):
        """
        Initialize HyDE generator.

        Args:
            provider: "openai" or "anthropic"
            model: LLM model name (default: gpt-4o-mini for stability/cost)
            anthropic_api_key: Anthropic API key (if provider="anthropic")
            openai_api_key: OpenAI API key (if provider="openai")
            num_hypotheses: Number of hypothetical docs to generate (default: 3)
        """
        self.provider = provider.lower()
        self.model = model
        self.num_hypotheses = num_hypotheses

        # Initialize LLM client
        if self.provider == "openai":
            if not openai_api_key:
                raise ValueError("openai_api_key required when provider='openai'")
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=openai_api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

        elif self.provider == "anthropic":
            if not anthropic_api_key:
                raise ValueError("anthropic_api_key required when provider='anthropic'")
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=anthropic_api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'")

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

        logger.info(
            f"HyDEGenerator initialized: provider={self.provider}, "
            f"model={self.model}, num_hypotheses={num_hypotheses}"
        )

    def _load_prompt_template(self) -> str:
        """
        Load HyDE prompt template from file.

        Uses direct I/O loading (simple Path.read_text()).

        Returns:
            Prompt template string with {query} placeholder
        """
        # Default prompt file: prompts/hyde_prompt.txt
        project_root = Path(__file__).parent.parent.parent
        prompt_file = project_root / "prompts" / "hyde_prompt.txt"

        try:
            if not prompt_file.exists():
                logger.warning(
                    f"HyDE prompt file not found: {prompt_file}. Using fallback prompt."
                )
                return self._get_fallback_prompt()

            template = prompt_file.read_text(encoding="utf-8").strip()

            # Validate template has {query} placeholder
            if "{query}" not in template:
                logger.warning(
                    f"HyDE prompt template missing {{query}} placeholder. Using fallback prompt."
                )
                return self._get_fallback_prompt()

            logger.debug(f"Loaded HyDE prompt template from: {prompt_file}")
            return template

        except Exception as e:
            logger.error(f"Failed to load HyDE prompt template: {e}. Using fallback prompt.")
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Get fallback HyDE prompt template."""
        return """Write a short, factual passage that answers this question: {query}

The passage should be 2-3 sentences long, use domain-specific terminology, and sound like it came from a real document. Do not include explanations or meta-commentary. Write only the passage itself."""

    def generate(
        self,
        query: str,
        num_docs: Optional[int] = None,
    ) -> HyDEResult:
        """
        Generate hypothetical documents for query.

        Args:
            query: Original user query
            num_docs: Number of hypothetical docs to generate (default: self.num_hypotheses)

        Returns:
            HyDEResult with hypothetical documents

        Note:
            - num_docs=0: Returns empty list (skip HyDE)
            - num_docs=1: Generates 1 hypothetical document
            - num_docs=3: Generates 3 hypothetical documents (multi-hypothesis averaging)
        """
        # Use configured default if not specified
        if num_docs is None:
            num_docs = self.num_hypotheses

        # Optimization: Skip generation if num_docs == 0
        if num_docs == 0:
            logger.debug(f"Skipping HyDE generation (num_docs=0): {query}")
            return HyDEResult(
                original_query=query,
                hypothetical_docs=[],
                generation_method="none",
            )

        # Generate hypothetical documents via LLM
        try:
            hypothetical_docs = self._generate_hypothetical_docs(query, num_docs)

            logger.info(
                f"Generated {len(hypothetical_docs)} hypothetical docs for query '{query[:50]}...'"
            )

            return HyDEResult(
                original_query=query,
                hypothetical_docs=hypothetical_docs,
                generation_method="llm",
                model_used=self.model,
            )

        except Exception as e:
            logger.warning(
                f"HyDE generation failed ({type(e).__name__}: {e}), returning empty result"
            )

            # Graceful fallback
            return HyDEResult(
                original_query=query,
                hypothetical_docs=[],
                generation_method="fallback",
            )

    def _generate_hypothetical_docs(
        self,
        query: str,
        num_docs: int,
    ) -> List[str]:
        """
        Generate hypothetical documents using LLM.

        Strategy:
        - Use prompt template with {query} substitution
        - Request num_docs hypothetical documents
        - Parse response (split by delimiter '---' or double newlines)

        Returns:
            List of hypothetical documents
        """
        # Build prompt from template
        prompt = self._build_hyde_prompt(query, num_docs)

        # Call LLM based on provider
        if self.provider == "openai":
            hypothetical_text = self._call_openai(prompt)
        else:  # anthropic
            hypothetical_text = self._call_anthropic(prompt)

        # Parse response (split by delimiter or newlines)
        docs = self._parse_hypothetical_docs(hypothetical_text, num_docs)

        # Log warning if we got fewer docs than requested
        if len(docs) < num_docs:
            logger.warning(
                f"Generated {len(docs)}/{num_docs} hypothetical docs. "
                f"LLM response may have been incomplete."
            )

        return docs[:num_docs]  # Return up to num_docs

    def _build_hyde_prompt(self, query: str, num_docs: int) -> str:
        """Build HyDE prompt from template."""
        # Substitute {query} placeholder
        prompt = self.prompt_template.replace("{query}", query)

        # Add instruction for multiple documents if needed
        if num_docs > 1:
            prompt += (
                f"\n\nGenerate {num_docs} different hypothetical passages, "
                f"separated by '---' on a new line."
            )

        return prompt

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # GPT-5/o-series models have special requirements
        if self.model.startswith(("gpt-5", "o1-", "o3-")):
            params["max_completion_tokens"] = 500
        else:
            params["max_tokens"] = 500
            params["temperature"] = 0.7

        response = self.client.chat.completions.create(**params)
        text = response.choices[0].message.content or ""

        # Track cost
        try:
            from src.cost_tracker import get_global_tracker

            tracker = get_global_tracker()
            tracker.track_llm(
                "openai",
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
        except Exception:
            pass  # Non-critical

        return text

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        # Track cost
        try:
            from src.cost_tracker import get_global_tracker

            tracker = get_global_tracker()
            tracker.track_llm(
                "anthropic", self.model, response.usage.input_tokens, response.usage.output_tokens
            )
        except Exception:
            pass  # Non-critical

        return text

    def _parse_hypothetical_docs(self, text: str, num_docs: int) -> List[str]:
        """
        Parse hypothetical documents from LLM response.

        Strategy:
        - Split by '---' delimiter (for multi-doc responses)
        - Fallback: split by double newlines
        - Filter out empty strings
        - Return up to num_docs
        """
        # Split by delimiter
        if "---" in text:
            docs = [doc.strip() for doc in text.split("---")]
        else:
            # Fallback: split by double newlines
            docs = [doc.strip() for doc in text.split("\n\n") if doc.strip()]

        # Filter empty strings
        docs = [doc for doc in docs if doc]

        return docs
