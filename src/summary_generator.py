"""
PHASE 2: Generic Summary Generation

Based on Reuter et al., 2024 (Summary-Augmented Chunking):
- Generic summaries OUTPERFORM expert-guided summaries
- Optimal length: 150 characters (±20 tolerance)
- Models: Claude Sonnet 4.5 (default), Claude Haiku 4.5, or gpt-4o-mini

Supports:
- Claude: claude-sonnet-4.5, claude-haiku-4.5 (via Anthropic API)
- OpenAI: gpt-4o-mini, gpt-4o (via OpenAI API)

Configuration is loaded from centralized config.py.
"""

import logging
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized configuration
try:
    from .config import SummarizationConfig, resolve_model_alias
    from .cost_tracker import get_global_tracker
    from .utils.security import sanitize_error
    from .utils.api_clients import APIClientFactory
    from .utils.batch_api import BatchAPIClient, BatchRequest
except ImportError:
    from config import SummarizationConfig, resolve_model_alias
    from cost_tracker import get_global_tracker
    from utils.security import sanitize_error
    from utils.api_clients import APIClientFactory
    from utils.batch_api import BatchAPIClient, BatchRequest

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generate generic summaries for documents and sections.

    Based on research:
    - Reuter et al., 2024: Generic > Expert-guided for retrieval
    - Optimal length: configured in SummarizationConfig (default 150 chars)
    - Style: Broad semantic alignment, not overfitted to narrow features

    Configuration is centralized in config.py.
    """

    def __init__(self, config: Optional[SummarizationConfig] = None, api_key: Optional[str] = None):
        """
        Initialize summary generator.

        Args:
            config: SummarizationConfig instance (uses defaults if None)
            api_key: API key (or set ANTHROPIC_API_KEY/OPENAI_API_KEY env var)
        """
        # Use provided config or create default
        self.config = config or SummarizationConfig()

        # Resolve model alias (e.g., "haiku" -> "claude-haiku-4-5-20251001")
        self.model = resolve_model_alias(self.config.model)

        # Extract config values for convenience
        self.max_chars = self.config.max_chars
        self.tolerance = self.config.tolerance
        self.max_workers = self.config.max_workers
        self.min_text_length = self.config.min_text_length
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens
        self.enable_prompt_batching = self.config.enable_prompt_batching
        self.batch_size = self.config.batch_size
        self.use_batch_api = self.config.use_batch_api
        self.batch_api_poll_interval = self.config.batch_api_poll_interval
        self.batch_api_timeout = self.config.batch_api_timeout

        # Initialize cost tracker
        self.tracker = get_global_tracker()

        # Detect provider from model name
        if "claude" in self.model.lower():
            self.provider = "claude"
            self._init_claude(api_key)
        elif "gpt" in self.model.lower():
            self.provider = "openai"
            self._init_openai(api_key)
        else:
            raise ValueError(
                f"Unsupported model: {self.model}. "
                f"Supported: claude-sonnet-4.5, claude-haiku-4.5, gpt-4o-mini, gpt-4o"
            )

        logger.info(
            f"SummaryGenerator initialized: provider={self.provider}, "
            f"model={self.model}, max_chars={self.max_chars}"
        )

    def _init_claude(self, api_key: Optional[str]):
        """Initialize Anthropic Claude client using centralized factory."""
        self.client = APIClientFactory.create_anthropic(api_key=api_key)
        logger.info("Claude client initialized")

    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client using centralized factory."""
        self.client = APIClientFactory.create_openai(api_key=api_key)
        logger.info("OpenAI client initialized")

    def generate_document_summary(
        self, document_text: str = None, section_summaries: list[str] = None
    ) -> str:
        """
        Generate generic document-level summary.

        Two modes:
        1. Hierarchical (RECOMMENDED): Summarize section summaries
           - Works for documents of ANY size
           - No truncation, uses all content
           - Better coverage and accuracy

        2. Direct (FALLBACK): Summarize document text
           - Limited to first 5000 chars
           - May miss important content
           - Use only if section summaries unavailable

        Args:
            document_text: Full document text (fallback mode)
            section_summaries: List of section summaries (hierarchical mode)

        Returns:
            Generic summary (~150 chars)
        """

        # Mode 1: Hierarchical summarization (RECOMMENDED)
        if section_summaries:
            # Filter out empty summaries
            valid_summaries = [s for s in section_summaries if s and s.strip()]

            if valid_summaries:
                logger.info(
                    f"Using hierarchical summarization ({len(valid_summaries)} section summaries)"
                )
                return self._generate_from_section_summaries(valid_summaries)

        # Mode 2: Direct summarization (FALLBACK)
        if document_text:
            logger.warning("Using direct summarization (limited to first 5000 chars)")
            # Truncate to first 5000 chars for efficiency
            text_preview = document_text[:30000]

            prompt = f"""You are an expert document summarizer.

Summarize the following document text. Focus on extracting the most important entities,
core purpose, and key topics.

CRITICAL: The summary MUST be EXACTLY {self.max_chars} characters or less (current limit: {self.max_chars} chars).
This is a hard constraint. The summary should be optimized for providing context to smaller text chunks.

Output only the summary text, nothing else.

Document:
{text_preview}

Summary (max {self.max_chars} characters):"""

            try:
                if self.provider == "claude":
                    summary = self._generate_with_claude(prompt)
                else:  # openai
                    summary = self._generate_with_openai(prompt)

                # Check length and retry if too long
                if len(summary) > self.max_chars + self.tolerance:
                    logger.warning(
                        f"Summary too long ({len(summary)} chars), regenerating with stricter limit"
                    )
                    return self.generate_document_summary_strict(text_preview)

                logger.debug(f"Generated document summary: {len(summary)} chars")
                return summary

            except Exception as e:
                logger.error(f"Failed to generate document summary: {sanitize_error(e)}")
                # Fallback: simple truncation
                return document_text[: self.max_chars].strip() + "..."

        # No input provided
        raise ValueError("Either document_text or section_summaries must be provided")

    def _generate_from_section_summaries(self, section_summaries: list[str]) -> str:
        """
        Generate document summary from section summaries (hierarchical approach).

        This is the RECOMMENDED method for large documents:
        - No truncation needed
        - Covers entire document through section summaries
        - More accurate and comprehensive

        Args:
            section_summaries: List of section-level summaries

        Returns:
            Document-level summary
        """

        # Combine section summaries
        combined_text = "\n".join(f"- {s}" for s in section_summaries)

        prompt = f"""You are an expert document summarizer.

You are given summaries of different sections from a document.
Create a unified document-level summary that captures the main theme and purpose.

CRITICAL: The summary MUST be EXACTLY {self.max_chars} characters or less (current limit: {self.max_chars} chars).
This is a hard constraint. The summary should be optimized for providing global context to text chunks.

Output only the summary text, nothing else.

Section summaries:
{combined_text}

Document summary (max {self.max_chars} characters):"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt)
            else:  # openai
                summary = self._generate_with_openai(prompt)

            # Check length and retry if too long
            if len(summary) > self.max_chars + self.tolerance:
                logger.warning(
                    f"Summary too long ({len(summary)} chars), regenerating with stricter limit"
                )
                # Retry with stricter limit
                target_chars = int(self.max_chars * 0.9)
                prompt_strict = f"""CRITICAL: Summarize in EXACTLY {target_chars} characters or LESS.

{combined_text}

Summary (STRICT LIMIT: {target_chars} characters):"""

                if self.provider == "claude":
                    summary = self._generate_with_claude(prompt_strict)
                else:
                    summary = self._generate_with_openai(prompt_strict)

                summary = summary[: self.max_chars]  # Hard truncate

            logger.debug(f"Generated hierarchical document summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate hierarchical summary: {sanitize_error(e)}")
            # Fallback: combine first chars of each section summary
            fallback = " ".join(section_summaries)
            return fallback[: self.max_chars].strip() + "..."

    def _generate_with_claude(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text using Claude API.

        Uses temperature and max_tokens from config unless overridden.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Track cost
        self.tracker.track_llm(
            provider="anthropic",
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            operation="summary",
        )

        return response.content[0].text.strip()

    def _generate_with_openai(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text using OpenAI API.

        Uses temperature and max_tokens from config unless overridden.
        """
        # GPT-5 and O-series models use max_completion_tokens instead of max_tokens
        # GPT-5 models only support temperature=1.0 (default)
        tokens_param = max_tokens or self.max_tokens
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,  # GPT-5 only supports default temperature
                max_completion_tokens=tokens_param,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=tokens_param,
            )

        # Track cost
        self.tracker.track_llm(
            provider="openai",
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            operation="summary",
        )

        return response.choices[0].message.content.strip()

    def generate_document_summary_strict(self, document_text: str) -> str:
        """
        Generate summary with stricter length constraint.

        Called when first attempt exceeds max_chars.
        """

        text_preview = document_text[:5000]
        target_chars = int(self.max_chars * 0.9)  # Aim for 90% of max

        prompt = f"""CRITICAL: Summarize this document in EXACTLY {target_chars} characters or LESS.
Be extremely concise.

Document:
{text_preview}

Summary (STRICT LIMIT: {target_chars} characters):"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt)
            else:  # openai
                summary = self._generate_with_openai(prompt)

            return summary[: self.max_chars]  # Hard truncate if needed

        except Exception as e:
            logger.error(f"Failed to generate strict summary: {sanitize_error(e)}")
            return document_text[: self.max_chars].strip() + "..."

    def generate_section_summary(self, section_text: str, section_title: str = "") -> str:
        """
        Generate generic section-level summary.

        Args:
            section_text: Section content
            section_title: Section title (for context)

        Returns:
            Generic summary (~150 chars)
        """

        # For sections, use shorter preview
        text_preview = section_text[:2000]

        title_context = f"Section title: {section_title}\n\n" if section_title else ""

        prompt = f"""Summarize this document section concisely.

{title_context}Section content:
{text_preview}

CRITICAL: Provide a summary that is EXACTLY {self.max_chars} characters or less (current limit: {self.max_chars} chars).
Focus on the main topic and key information. This is a hard constraint.

Summary (max {self.max_chars} characters):"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt)
            else:  # openai
                summary = self._generate_with_openai(prompt)

            # Check length
            if len(summary) > self.max_chars + self.tolerance:
                summary = summary[: self.max_chars]

            logger.debug(f"Generated section summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate section summary: {sanitize_error(e)}")
            # Fallback
            return section_text[: self.max_chars].strip() + "..."

    # ===== OpenAI Batch API (centralized in utils.batch_api) =====

    def _generate_batch_summaries_with_openai_batch(
        self, filtered_texts: list[tuple[int, str, str]]  # [(index, text, title), ...]
    ) -> dict[int, str]:
        """
        Generate summaries using OpenAI Batch API (50% cheaper, async).

        Uses centralized BatchAPIClient for all batch processing logic.

        Args:
            filtered_texts: List of (index, text, title) tuples

        Returns:
            Dict mapping index to summary

        Raises:
            Exception if batch processing fails
        """
        logger.info(
            f"Using OpenAI Batch API: {len(filtered_texts)} sections "
            f"(50% cost savings, async processing)"
        )

        # Create batch API client
        batch_client = BatchAPIClient(
            openai_client=self.client, logger_instance=logger, cost_tracker=self.tracker
        )

        # Define request creation function
        def create_request(item: tuple[int, str, str], idx: int) -> BatchRequest:
            section_idx, text, title = item

            # Truncate text for efficiency
            text_preview = text[:2000]
            title_context = f"Section title: {title}\n\n" if title else ""

            prompt = f"""Summarize this document section concisely.

{title_context}Section content:
{text_preview}

CRITICAL: Provide a summary that is EXACTLY {self.max_chars} characters or less (current limit: {self.max_chars} chars).
Focus on the main topic and key information. This is a hard constraint.

Summary (max {self.max_chars} characters):"""

            # Build request body with model-specific parameters
            # GPT-5 and O-series models use max_completion_tokens instead of max_tokens
            # GPT-5 models only support temperature=1.0 (default)
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }

            if self.model.startswith(("gpt-5", "o1-", "o3-", "o4-")):
                # GPT-5/o-series parameters
                body["max_completion_tokens"] = self.max_tokens
                body["temperature"] = 1.0  # GPT-5 only supports default temperature
            else:
                # GPT-4 and earlier parameters
                body["max_tokens"] = self.max_tokens
                body["temperature"] = self.temperature

            return BatchRequest(
                custom_id=f"section_{section_idx}",
                method="POST",
                url="/v1/chat/completions",
                body=body,
            )

        # Define response parsing function
        def parse_response(response: dict) -> str:
            """Extract and validate summary from API response."""
            summary = response["choices"][0]["message"]["content"].strip()

            # Enforce length limit
            if len(summary) > self.max_chars + self.tolerance:
                summary = summary[: self.max_chars]

            return summary

        # Process batch using centralized client
        try:
            results_by_custom_id = batch_client.process_batch(
                items=filtered_texts,
                create_request_fn=create_request,
                parse_response_fn=parse_response,
                poll_interval=self.batch_api_poll_interval,
                timeout_hours=self.batch_api_timeout // 3600,
                operation="summary",
                model=self.model,
            )

            # Map results back to section indices
            summaries_map = {}
            for section_idx, text, title in filtered_texts:
                custom_id = f"section_{section_idx}"
                if custom_id in results_by_custom_id:
                    summaries_map[section_idx] = results_by_custom_id[custom_id]
                else:
                    # Fallback: truncate text
                    logger.warning(f"No summary for section {section_idx}, using fallback")
                    summaries_map[section_idx] = text[: self.max_chars].strip() + "..."

            logger.info(f"✓ Batch API succeeded: {len(summaries_map)} summaries generated")
            return summaries_map

        except Exception as e:
            logger.error(f"Batch API failed: {sanitize_error(e)}, falling back to parallel mode")
            raise

    def _batch_summarize_with_prompt(
        self, sections: list[tuple[int, str, str]]  # [(index, text, title), ...]
    ) -> list[tuple[int, str]]:
        """
        Generate summaries for multiple sections in ONE API call using JSON output.

        This is 10-15× faster than individual API calls per section:
        - 50 sections: 50 API calls × 300ms = 15s → 4 API calls × 300ms = 1.2s

        Args:
            sections: List of (original_index, text, title) tuples

        Returns:
            List of (index, summary) tuples for matching with original order
        """
        import json

        if not sections:
            return []

        # Truncate texts for prompt (keep manageable input size)
        max_text_preview = 500  # chars per section
        sections_data = []
        for idx, text, title in sections:
            sections_data.append({"index": idx, "title": title, "content": text[:max_text_preview]})

        # Create JSON input
        sections_json = json.dumps(sections_data, ensure_ascii=False, indent=2)

        # Build prompt
        prompt = f"""You are an expert document summarizer. Generate concise summaries for each section below.

CRITICAL CONSTRAINTS:
- Each summary MUST be EXACTLY {self.max_chars} characters or less
- Output MUST be valid JSON array
- Preserve the exact order and index numbers

Input sections (JSON):
{sections_json}

Output format (JSON array):
[
  {{"index": 0, "summary": "..."}},
  {{"index": 1, "summary": "..."}},
  ...
]

Generate summaries (MUST be valid JSON):"""

        try:
            # Generate with LLM
            if self.provider == "claude":
                response_text = self._generate_with_claude(
                    prompt, max_tokens=self.max_tokens * len(sections)
                )
            else:  # openai
                response_text = self._generate_with_openai(
                    prompt, max_tokens=self.max_tokens * len(sections)
                )

            # Parse JSON response
            # Try to extract JSON from response (handle cases where LLM adds explanation)
            response_text = response_text.strip()

            # Find JSON array (starts with [ and ends with ])
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]")

            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                logger.warning(
                    "No JSON array found in response, falling back to sequential processing"
                )
                raise ValueError("Invalid JSON response")

            json_text = response_text[start_idx : end_idx + 1]
            summaries_json = json.loads(json_text)

            if not isinstance(summaries_json, list):
                raise ValueError("Response is not a JSON array")

            # Extract summaries
            results = []
            for item in summaries_json:
                if not isinstance(item, dict):
                    continue
                if "index" not in item or "summary" not in item:
                    continue

                idx = item["index"]
                summary = item["summary"].strip()

                # Enforce length limit (hard truncate if needed)
                if len(summary) > self.max_chars + self.tolerance:
                    summary = summary[: self.max_chars]

                results.append((idx, summary))

            # Verify we got summaries for all sections
            if len(results) != len(sections):
                logger.warning(
                    f"Got {len(results)} summaries for {len(sections)} sections, "
                    f"falling back to sequential processing"
                )
                raise ValueError("Incomplete summaries")

            logger.debug(f"Batch summarized {len(results)} sections in one API call")
            return results

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, falling back to sequential processing")
            raise
        except Exception as e:
            logger.warning(
                f"Batch summarization failed: {e}, falling back to sequential processing"
            )
            raise

    def generate_batch_summaries(
        self, texts: list[tuple[str, str]],  # [(text, title), ...]
        min_text_length: int | None = None,  # Override min_text_length (None = use self.min_text_length)
    ) -> list[str]:
        """
        Generate summaries for multiple sections.

        Two modes:
        1. Prompt Batching (NEW, 10-15× faster):
           - Groups sections into batches of 10-20
           - One API call per batch with JSON output
           - Automatic fallback to parallel mode on errors

        2. Parallel Mode (legacy, still used as fallback):
           - Uses ThreadPoolExecutor for concurrent API requests
           - One API call per section

        Args:
            texts: List of (text, title) tuples
            min_text_length: Override minimum text length (None = use self.min_text_length)

        Returns:
            List of summaries (in original order)
        """
        # Use override or default
        effective_min_length = min_text_length if min_text_length is not None else self.min_text_length

        # Filter out tiny sections
        filtered_texts = []
        skip_indices = []
        for i, (text, title) in enumerate(texts):
            if len(text.strip()) < effective_min_length:
                skip_indices.append(i)
                logger.debug(f"Skipping tiny section '{title}' ({len(text)} chars)")
            else:
                filtered_texts.append((i, text, title))

        if not filtered_texts:
            logger.warning("All sections too small, skipping summary generation")
            return [""] * len(texts)

        summaries_map = {}
        failures = []

        # MODE 1: OpenAI Batch API (NEW - 50% cost savings, async)
        if self.use_batch_api and self.provider == "openai":
            try:
                summaries_map = self._generate_batch_summaries_with_openai_batch(filtered_texts)
                # Success! Skip other modes
                logger.info(f"✓ Batch API succeeded: {len(summaries_map)} summaries generated")

            except Exception as e:
                logger.warning(f"Batch API failed ({e}), falling back to parallel mode")
                summaries_map = {}  # Clear partial results
                # Fall through to MODE 2 (parallel)

        # MODE 2: Prompt Batching (JSON batching - disabled by default)
        if not summaries_map and self.enable_prompt_batching:
            logger.info(
                f"Using prompt batching: {len(filtered_texts)} sections → "
                f"{(len(filtered_texts) + self.batch_size - 1) // self.batch_size} API calls "
                f"(batch_size={self.batch_size})"
            )

            # Split into batches
            batches = []
            for i in range(0, len(filtered_texts), self.batch_size):
                batch = filtered_texts[i : i + self.batch_size]
                batches.append(batch)

            # Process batches in parallel for maximum speed
            def process_batch(batch_idx_and_batch):
                batch_idx, batch = batch_idx_and_batch
                try:
                    # Summarize entire batch in one API call
                    batch_results = self._batch_summarize_with_prompt(batch)
                    logger.debug(
                        f"Batch {batch_idx}/{len(batches)}: "
                        f"Generated {len(batch_results)} summaries in one API call"
                    )
                    return (True, batch_results, [])  # Success, results, no failures
                except Exception as e:
                    # Fallback: Process this batch with parallel mode
                    logger.warning(f"Batch {batch_idx} failed ({e}), falling back to parallel mode")

                    batch_results = []
                    batch_failures = []

                    def generate_one(idx: int, text: str, title: str) -> tuple[int, str, bool]:
                        try:
                            summary = self.generate_section_summary(text, title)
                            return (idx, summary, True)
                        except Exception as e2:
                            logger.error(
                                f"Failed to generate summary for '{title}': {sanitize_error(e2)}"
                            )
                            fallback = text[: self.max_chars].strip() + "..."
                            return (idx, fallback, False)

                    # Process failed batch sections in parallel
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = [
                            executor.submit(generate_one, idx, text, title)
                            for idx, text, title in batch
                        ]

                        for future in as_completed(futures):
                            idx, summary, success = future.result()
                            batch_results.append((idx, summary))
                            if not success:
                                batch_failures.append(idx)

                    return (False, batch_results, batch_failures)  # Fallback used

            # Use ThreadPoolExecutor to process batches in parallel
            max_concurrent_batches = min(5, len(batches))  # Limit concurrent batch requests
            with ThreadPoolExecutor(max_workers=max_concurrent_batches) as executor:
                batch_futures = [
                    executor.submit(process_batch, (idx, batch))
                    for idx, batch in enumerate(batches, 1)
                ]

                for future in as_completed(batch_futures):
                    success, batch_results, batch_failures = future.result()

                    # Add results to summaries_map
                    for idx, summary in batch_results:
                        summaries_map[idx] = summary

                    # Track failures
                    failures.extend(batch_failures)

        # MODE 3: Parallel Mode (fallback - one API call per section)
        if not summaries_map:
            logger.info(
                f"Using parallel mode: {len(filtered_texts)} sections → "
                f"{len(filtered_texts)} API calls"
            )

            def generate_one(idx: int, text: str, title: str) -> tuple[int, str, bool]:
                """Generate one summary. Returns (idx, summary, success)."""
                try:
                    summary = self.generate_section_summary(text, title)
                    return (idx, summary, True)  # Success
                except Exception as e:
                    logger.error(f"Failed to generate summary for '{title}': {sanitize_error(e)}")
                    fallback = text[: self.max_chars].strip() + "..."
                    return (idx, fallback, False)  # Failure

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(generate_one, idx, text, title)
                    for idx, text, title in filtered_texts
                ]

                for future in as_completed(futures):
                    idx, summary, success = future.result()
                    summaries_map[idx] = summary
                    if not success:
                        failures.append(idx)

        # Report failures if any
        if failures:
            failure_rate = len(failures) / len(filtered_texts) * 100
            logger.warning(
                f"Summary generation: {len(failures)}/{len(filtered_texts)} failed ({failure_rate:.1f}%)\n"
                f"Failed sections will use truncated content (reduced quality)"
            )

            # High failure rate warning
            if failure_rate > 25:
                logger.error(
                    f"HIGH FAILURE RATE: {failure_rate:.1f}% of summaries failed!\n"
                    f"This will significantly impact RAG quality.\n"
                    f"Common causes:\n"
                    f"  - API quota exhausted\n"
                    f"  - Invalid API key\n"
                    f"  - Network connectivity issues\n"
                    f"  - Model unavailable"
                )

        # Build result list in original order
        result = []
        for i in range(len(texts)):
            if i in skip_indices:
                result.append("")  # Empty summary for tiny sections
            else:
                result.append(summaries_map[i])

        successful = len(summaries_map) - len(failures)
        logger.info(
            f"Generated {successful}/{len(summaries_map)} summaries successfully "
            f"(skipped {len(skip_indices)} tiny sections)"
        )
        return result


# Example usage
if __name__ == "__main__":
    # Test
    generator = SummaryGenerator()

    test_doc = """
    This is a technical specification document for a nuclear power plant reactor core.
    It describes the primary cooling system, safety features, and operational parameters.
    The document covers emergency shutdown procedures, redundancy systems, and regulatory
    compliance requirements for the VVER-1200 reactor type.
    """

    summary = generator.generate_document_summary(test_doc)
    print(f"Summary ({len(summary)} chars): {summary}")
