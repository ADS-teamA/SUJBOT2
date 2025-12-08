"""
Base class for all document labeling transformers.

Provides common functionality:
- LRUCache integration (content-hash based deduplication)
- OpenAI Batch API support (50% cost savings)
- Graceful error handling with fallbacks
- JSON parsing with repair

All labelers are LLM-driven (CLAUDE.md compliance) - no hardcoded rules.
"""

import hashlib
import json
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, TypeVar

from json_repair import repair_json
from llama_index.core.schema import BaseNode, TransformComponent

from src.utils.cache import LRUCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseLabeler(TransformComponent):
    """
    Base class for all document labeling transformers.

    Provides:
    - Content-hash based caching (LRUCache)
    - Batch processing support
    - JSON response parsing with repair fallback
    - Graceful error handling

    All subclasses must implement:
    - _get_prompt_template(): Return the LLM prompt template
    - _parse_response(): Parse LLM response into metadata dict
    - _get_metadata_keys(): Return list of metadata keys this labeler adds
    - _get_default_metadata(): Return default metadata on failure

    Example:
        >>> class MyLabeler(BaseLabeler):
        ...     def _get_prompt_template(self) -> str:
        ...         return "Analyze: {text}"
        ...     def _parse_response(self, response: str) -> Dict:
        ...         return json.loads(response)
        ...     def _get_metadata_keys(self) -> List[str]:
        ...         return ["my_key"]
        ...     def _get_default_metadata(self) -> Dict:
        ...         return {"my_key": None}
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        batch_size: int = 50,
        cache_enabled: bool = True,
        cache_size: int = 1000,
        name: Optional[str] = None,
    ):
        """
        Initialize the labeler.

        Args:
            model_name: LLM model to use (default: gpt-4o-mini)
            batch_size: Number of items to process per batch
            cache_enabled: Enable content-hash based caching
            cache_size: Maximum cache entries
            name: Optional name for logging
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.name = name or self.__class__.__name__

        # Initialize cache
        self._cache: Optional[LRUCache[Dict[str, Any]]] = None
        if cache_enabled:
            self._cache = LRUCache[Dict[str, Any]](
                max_size=cache_size, name=f"{self.name}_cache"
            )

        # Statistics
        self._processed_count = 0
        self._cache_hits = 0
        self._errors = 0

    @abstractmethod
    def _get_prompt_template(self) -> str:
        """
        Return the LLM prompt template.

        Template should use {text} and optionally {context} placeholders.

        Returns:
            Prompt template string
        """
        pass

    @abstractmethod
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into metadata dictionary.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed metadata dictionary
        """
        pass

    @abstractmethod
    def _get_metadata_keys(self) -> List[str]:
        """
        Return list of metadata keys this labeler adds.

        Returns:
            List of metadata key names
        """
        pass

    @abstractmethod
    def _get_default_metadata(self) -> Dict[str, Any]:
        """
        Return default metadata on failure.

        Returns:
            Default metadata dictionary
        """
        pass

    def _get_cache_key(self, text: str) -> str:
        """
        Generate content-hash cache key.

        Args:
            text: Text content to hash

        Returns:
            MD5 hash of content (first 16 chars)
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def _get_node_text(self, node: BaseNode) -> str:
        """
        Extract text content from node.

        Args:
            node: LlamaIndex node

        Returns:
            Text content or empty string
        """
        if hasattr(node, "text") and node.text:
            return node.text
        if hasattr(node, "get_content"):
            return node.get_content()
        return ""

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON object from text.

        Args:
            text: Text potentially containing JSON

        Returns:
            Extracted JSON string or None
        """
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return text[start : end + 1]
        return None

    def _parse_json_response(self, text: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JSON response with repair fallback.

        Args:
            text: Raw response text
            default: Default value on failure

        Returns:
            Parsed dictionary or default
        """
        if not text:
            return default

        # Try direct JSON extraction
        try:
            json_text = self._extract_json(text)
            if json_text:
                return json.loads(json_text)
        except json.JSONDecodeError:
            logger.debug(f"{self.name}: Initial JSON parse failed, trying repair")

        # Try JSON repair
        try:
            repaired = repair_json(text)
            if repaired:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.debug(f"{self.name}: JSON repair also failed: {e}")

        logger.warning(f"{self.name}: Failed to parse response (length={len(text)})")
        return default

    def _apply_metadata(self, node: BaseNode, metadata: Dict[str, Any]) -> None:
        """
        Apply extracted metadata to node.

        Args:
            node: Node to update
            metadata: Metadata to apply
        """
        for key in self._get_metadata_keys():
            if key in metadata:
                node.metadata[key] = metadata[key]

    def _handle_error(self, node: BaseNode, error: Exception) -> None:
        """
        Handle labeling error gracefully.

        Args:
            node: Node that failed
            error: Exception that occurred
        """
        node_id = getattr(node, "id_", "unknown")
        logger.warning(f"{self.name}: Failed for node {node_id}: {error}")

        # Apply default metadata
        default = self._get_default_metadata()
        for key, value in default.items():
            node.metadata[key] = value

        # Mark error
        node.metadata[f"{self.name.lower()}_error"] = str(error)
        self._errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get labeler statistics.

        Returns:
            Statistics dictionary
        """
        cache_stats = self._cache.get_stats() if self._cache else {"enabled": False}
        return {
            "name": self.name,
            "processed": self._processed_count,
            "cache_hits": self._cache_hits,
            "errors": self._errors,
            "cache": cache_stats,
        }


class SyncLabeler(BaseLabeler):
    """
    Synchronous labeler for immediate LLM calls.

    Use this for real-time labeling without Batch API.
    Subclasses must implement _call_llm() for their provider.
    """

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM synchronously.

        Args:
            prompt: Prompt to send

        Returns:
            LLM response text
        """
        pass

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """
        Process nodes with synchronous LLM calls.

        Args:
            nodes: List of nodes to process

        Returns:
            Nodes with metadata added
        """
        if not nodes:
            return nodes

        logger.info(f"{self.name}: Processing {len(nodes)} nodes with {self.model_name}")

        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i : i + self.batch_size]
            self._process_batch_sync(batch)

            if i > 0 and i % (self.batch_size * 5) == 0:
                logger.info(f"{self.name}: Progress {i}/{len(nodes)} nodes")

        logger.info(
            f"{self.name}: Complete - {len(nodes)} processed, "
            f"{self._cache_hits} cache hits, {self._errors} errors"
        )
        return nodes

    def _process_batch_sync(self, nodes: List[BaseNode]) -> None:
        """
        Process batch of nodes synchronously.

        Args:
            nodes: Batch of nodes
        """
        for node in nodes:
            try:
                text = self._get_node_text(node)
                if not text or len(text.strip()) < 50:
                    self._apply_metadata(node, self._get_default_metadata())
                    continue

                # Check cache
                cache_key = self._get_cache_key(text)
                if self._cache:
                    cached = self._cache.get(cache_key)
                    if cached is not None:
                        self._apply_metadata(node, cached)
                        self._cache_hits += 1
                        self._processed_count += 1
                        continue

                # Call LLM
                prompt = self._get_prompt_template().format(
                    text=text[:4000],  # Truncate if needed
                    context=node.metadata.get("section_path", ""),
                )
                response = self._call_llm(prompt)
                metadata = self._parse_response(response)

                # Apply and cache
                self._apply_metadata(node, metadata)
                if self._cache:
                    self._cache.set(cache_key, metadata)
                self._processed_count += 1

            except Exception as e:
                self._handle_error(node, e)


class AsyncLabeler(BaseLabeler):
    """
    Asynchronous labeler for Batch API processing.

    Use this for batch processing with OpenAI Batch API (50% cost savings).
    Subclasses must implement _create_batch_request() and _submit_batch().
    """

    @abstractmethod
    async def _create_batch_requests(
        self, nodes: List[BaseNode]
    ) -> List[Dict[str, Any]]:
        """
        Create batch requests for all nodes.

        Args:
            nodes: Nodes to process

        Returns:
            List of batch request dictionaries
        """
        pass

    @abstractmethod
    async def _submit_and_wait(
        self, requests: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Submit batch and wait for completion.

        Args:
            requests: Batch requests

        Returns:
            Dict mapping custom_id -> response text
        """
        pass

    async def process_async(
        self, nodes: List[BaseNode], **kwargs: Any
    ) -> List[BaseNode]:
        """
        Process nodes with Batch API.

        Args:
            nodes: List of nodes to process

        Returns:
            Nodes with metadata added
        """
        if not nodes:
            return nodes

        logger.info(
            f"{self.name}: Processing {len(nodes)} nodes via Batch API with {self.model_name}"
        )

        # Filter nodes that need processing (not in cache)
        nodes_to_process = []
        cache_results = {}

        for node in nodes:
            text = self._get_node_text(node)
            if not text or len(text.strip()) < 50:
                self._apply_metadata(node, self._get_default_metadata())
                continue

            cache_key = self._get_cache_key(text)
            if self._cache:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    cache_results[id(node)] = cached
                    self._cache_hits += 1
                    continue

            nodes_to_process.append(node)

        # Apply cached results
        for node in nodes:
            if id(node) in cache_results:
                self._apply_metadata(node, cache_results[id(node)])
                self._processed_count += 1

        if not nodes_to_process:
            logger.info(f"{self.name}: All {len(nodes)} nodes served from cache")
            return nodes

        logger.info(
            f"{self.name}: {len(nodes_to_process)} nodes need LLM calls "
            f"({self._cache_hits} cache hits)"
        )

        # Create and submit batch
        try:
            requests = await self._create_batch_requests(nodes_to_process)
            responses = await self._submit_and_wait(requests)

            # Process responses
            for node in nodes_to_process:
                node_id = getattr(node, "id_", str(id(node)))
                response = responses.get(node_id)

                if response:
                    try:
                        metadata = self._parse_response(response)
                        self._apply_metadata(node, metadata)

                        # Cache result
                        text = self._get_node_text(node)
                        if self._cache and text:
                            cache_key = self._get_cache_key(text)
                            self._cache.set(cache_key, metadata)

                        self._processed_count += 1
                    except Exception as e:
                        self._handle_error(node, e)
                else:
                    self._handle_error(
                        node, ValueError(f"No response for node {node_id}")
                    )

        except Exception as e:
            logger.error(f"{self.name}: Batch processing failed: {e}")
            # Apply defaults to all unprocessed nodes
            for node in nodes_to_process:
                self._handle_error(node, e)

        logger.info(
            f"{self.name}: Complete - {self._processed_count} processed, "
            f"{self._cache_hits} cache hits, {self._errors} errors"
        )
        return nodes
