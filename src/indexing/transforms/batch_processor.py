"""
OpenAI Batch API processor for labeling pipeline.

Wraps the existing BatchAPIClient from src/utils/batch_api.py
for document labeling operations (categories, keywords, questions).

Features:
- 50% cost savings compared to real-time API
- Automatic batching and JSONL formatting
- Polling with configurable timeout
- Integration with labeling extractors
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from src.config_schema import LabelingConfig
from src.extraction_models import DocumentSection, ExtractedDocument
from src.indexing.transforms.category_extractor import (
    DocumentCategoryExtractor,
    DocumentTaxonomy,
)
from src.indexing.transforms.keyword_extractor import (
    SectionKeywordExtractor,
    SectionKeywords,
)
from src.indexing.transforms.question_generator import (
    ChunkQuestionGenerator,
    ChunkQuestions,
)
from src.multi_layer_chunker import Chunk
from src.utils.batch_api import BatchAPIClient, BatchRequest

logger = logging.getLogger(__name__)


class LabelingBatchProcessor:
    """
    Process labeling requests via OpenAI Batch API.

    Combines category, keyword, and question extraction into
    efficient batch operations with 50% cost savings.

    Example:
        >>> processor = LabelingBatchProcessor(config)
        >>> result = await processor.process_document(document, chunks)
        >>> labeled_chunks = processor.apply_result(chunks, result)
    """

    def __init__(
        self,
        config: Optional[LabelingConfig] = None,
        cost_tracker=None,
    ):
        """
        Initialize batch processor.

        Args:
            config: Labeling configuration
            cost_tracker: Optional cost tracker
        """
        self.config = config or LabelingConfig()
        self.cost_tracker = cost_tracker

        # Initialize extractors
        self.category_extractor = DocumentCategoryExtractor(
            model_name=self.config.model,
            use_dynamic_categories=self.config.use_dynamic_categories,
        )
        self.keyword_extractor = SectionKeywordExtractor(
            model_name=self.config.model,
            max_keywords=self.config.max_keywords_per_chunk,
        )
        self.question_generator = ChunkQuestionGenerator(
            model_name=self.config.model,
            max_questions=self.config.max_questions_per_chunk,
            cache_enabled=self.config.cache_enabled,
            cache_size=self.config.cache_size,
        )

        # Batch API client (lazy init)
        self._batch_client: Optional[BatchAPIClient] = None

    def _get_batch_client(self) -> BatchAPIClient:
        """Get or create Batch API client."""
        if self._batch_client is None:
            import os

            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Required for Batch API."
                )

            openai_client = OpenAI(api_key=api_key)
            self._batch_client = BatchAPIClient(
                openai_client, logger, self.cost_tracker
            )

        return self._batch_client

    def create_category_request(
        self, document: ExtractedDocument
    ) -> BatchRequest:
        """
        Create category extraction batch request.

        Args:
            document: Document to classify

        Returns:
            BatchRequest for category extraction
        """
        request_dict = self.category_extractor.create_batch_request(
            document, "cat_doc"
        )
        return BatchRequest(
            custom_id=request_dict["custom_id"],
            method=request_dict["method"],
            url=request_dict["url"],
            body=request_dict["body"],
        )

    def create_keyword_requests(
        self,
        sections: List[DocumentSection],
        document_category: str,
    ) -> List[BatchRequest]:
        """
        Create keyword extraction batch requests.

        Args:
            sections: Sections to extract keywords from
            document_category: Document category for context

        Returns:
            List of BatchRequests
        """
        request_dicts = self.keyword_extractor.create_batch_requests(
            sections, document_category
        )
        return [
            BatchRequest(
                custom_id=r["custom_id"],
                method=r["method"],
                url=r["url"],
                body=r["body"],
            )
            for r in request_dicts
        ]

    def create_question_requests(
        self,
        chunks: List[Chunk],
        document_title: str,
        document_category: str,
    ) -> List[BatchRequest]:
        """
        Create question generation batch requests.

        Args:
            chunks: Chunks to generate questions for
            document_title: Document title
            document_category: Document category

        Returns:
            List of BatchRequests (excludes cached chunks)
        """
        requests = []

        for chunk in chunks:
            request_dict = self.question_generator.create_batch_request(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.raw_content or chunk.content,
                document_title=document_title,
                section_path=chunk.metadata.section_path or "",
                category=document_category,
            )

            if request_dict:  # None if cached
                requests.append(
                    BatchRequest(
                        custom_id=request_dict["custom_id"],
                        method=request_dict["method"],
                        url=request_dict["url"],
                        body=request_dict["body"],
                    )
                )

        return requests

    def process_keywords_batch(
        self,
        sections: List[DocumentSection],
        document_category: str,
    ) -> Dict[str, SectionKeywords]:
        """
        Process keyword extraction via Batch API.

        Args:
            sections: Sections to process
            document_category: Document category

        Returns:
            Dict mapping section_id -> SectionKeywords
        """
        logger.info(f"Processing {len(sections)} sections via Batch API...")

        # Create requests
        requests = self.keyword_extractor.create_batch_requests(
            sections, document_category
        )

        if not requests:
            logger.info("No keyword requests to process (all empty sections)")
            return {}

        # Submit batch
        batch_client = self._get_batch_client()

        def create_request_fn(item: Dict, idx: int) -> BatchRequest:
            return BatchRequest(
                custom_id=item["custom_id"],
                method=item["method"],
                url=item["url"],
                body=item["body"],
            )

        def parse_response_fn(response: Dict) -> str:
            return response.get("choices", [{}])[0].get("message", {}).get(
                "content", ""
            )

        results = batch_client.process_batch(
            items=requests,
            create_request_fn=create_request_fn,
            parse_response_fn=parse_response_fn,
            poll_interval=self.config.batch_api_poll_interval,
            timeout_hours=self.config.batch_api_timeout_hours,
            operation="keyword_extraction",
            model=self.config.model,
        )

        # Parse results
        section_keywords = {}
        for custom_id, response_text in results.items():
            if response_text:
                # Extract section_id from custom_id (kw_section_id)
                section_id = custom_id[3:] if custom_id.startswith("kw_") else custom_id
                keywords = self.keyword_extractor._parse_response(
                    response_text, section_id
                )
                section_keywords[section_id] = keywords

        logger.info(f"Extracted keywords for {len(section_keywords)} sections")
        return section_keywords

    def process_questions_batch(
        self,
        chunks: List[Chunk],
        document_title: str,
        document_category: str,
    ) -> Dict[str, ChunkQuestions]:
        """
        Process question generation via Batch API.

        Args:
            chunks: Chunks to process
            document_title: Document title
            document_category: Document category

        Returns:
            Dict mapping chunk_id -> ChunkQuestions
        """
        logger.info(f"Processing {len(chunks)} chunks via Batch API...")

        # Build chunk lookup for caching
        chunk_texts = {
            chunk.chunk_id: chunk.raw_content or chunk.content for chunk in chunks
        }

        # Create requests (excludes cached)
        requests = []
        for chunk in chunks:
            request_dict = self.question_generator.create_batch_request(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk_texts[chunk.chunk_id],
                document_title=document_title,
                section_path=chunk.metadata.section_path or "",
                category=document_category,
            )
            if request_dict:
                requests.append(request_dict)

        # Get cached results
        chunk_questions = {}
        cache_hits = self.question_generator._cache_hits

        # Submit batch if there are uncached requests
        if requests:
            batch_client = self._get_batch_client()

            def create_request_fn(item: Dict, idx: int) -> BatchRequest:
                return BatchRequest(
                    custom_id=item["custom_id"],
                    method=item["method"],
                    url=item["url"],
                    body=item["body"],
                )

            def parse_response_fn(response: Dict) -> str:
                return response.get("choices", [{}])[0].get("message", {}).get(
                    "content", ""
                )

            results = batch_client.process_batch(
                items=requests,
                create_request_fn=create_request_fn,
                parse_response_fn=parse_response_fn,
                poll_interval=self.config.batch_api_poll_interval,
                timeout_hours=self.config.batch_api_timeout_hours,
                operation="question_generation",
                model=self.config.model,
            )

            # Parse results
            for custom_id, response_text in results.items():
                if response_text:
                    # Extract chunk_id from custom_id (q_chunk_id)
                    chunk_id = custom_id[2:] if custom_id.startswith("q_") else custom_id
                    chunk_text = chunk_texts.get(chunk_id, "")
                    questions = self.question_generator.process_batch_response(
                        custom_id, response_text, chunk_text
                    )
                    chunk_questions[chunk_id] = questions

        # Add cached results back
        if self.question_generator._cache:
            for chunk_id, chunk_text in chunk_texts.items():
                if chunk_id not in chunk_questions:
                    cache_key = self.question_generator._get_cache_key(chunk_text)
                    cached = self.question_generator._cache.get(cache_key)
                    if cached:
                        cached.chunk_id = chunk_id
                        chunk_questions[chunk_id] = cached

        new_cache_hits = self.question_generator._cache_hits - cache_hits
        logger.info(
            f"Generated questions for {len(chunk_questions)} chunks "
            f"({new_cache_hits} from cache)"
        )
        return chunk_questions

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "question_generator": self.question_generator.get_stats(),
        }
