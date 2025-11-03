"""
OpenAI Batch API client for MY_SUJBOT pipeline.

Provides centralized Batch API processing to eliminate duplicate code:
- summary_generator.py: Lines 424-694 (271 lines)
- contextual_retrieval.py: Lines 512-696 (185 lines)

Total duplication eliminated: 450+ lines

Batch API features:
- 50% cost savings compared to real-time API
- Automatic request batching and JSONL formatting
- Polling with configurable timeout (default: 12 hours)
- Error handling and fallback support
- Cost tracking integration

Usage:
    from src.utils import BatchAPIClient

    client = BatchAPIClient(openai_client, logger, tracker)

    results = client.process_batch(
        items=sections,
        create_request_fn=lambda item, idx: BatchRequest(...),
        parse_response_fn=lambda response: response['choices'][0]['message']['content'],
        operation="summary"
    )
"""

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .security import sanitize_error

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Single request in a batch."""

    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI Batch API format."""
        return {
            "custom_id": self.custom_id,
            "method": self.method,
            "url": self.url,
            "body": self.body,
        }


class BatchAPIClient:
    """
    Generic OpenAI Batch API client.

    Handles the complete batch processing flow:
    1. Create batch requests (JSONL format)
    2. Submit batch to OpenAI
    3. Poll for completion (with timeout)
    4. Parse and return results

    Features:
    - 50% cost savings vs real-time API
    - Configurable timeout (default: 12 hours)
    - Automatic cost tracking
    - Detailed error messages
    """

    def __init__(
        self, openai_client, logger_instance: Optional[logging.Logger] = None, cost_tracker=None
    ):
        """
        Initialize Batch API client.

        Args:
            openai_client: OpenAI client instance
            logger_instance: Optional logger (uses module logger if not provided)
            cost_tracker: Optional CostTracker instance for cost tracking
        """
        self.client = openai_client
        self.logger = logger_instance or logger
        self.tracker = cost_tracker

    def process_batch(
        self,
        items: List[Any],
        create_request_fn: Callable[[Any, int], BatchRequest],
        parse_response_fn: Callable[[Dict[str, Any]], Any],
        poll_interval: int = 5,
        timeout_hours: int = 12,
        operation: str = "batch",
        model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Process items using OpenAI Batch API.

        Args:
            items: List of items to process
            create_request_fn: Function to create BatchRequest from item and index
            parse_response_fn: Function to parse API response into result
            poll_interval: Seconds between status checks (default: 5)
            timeout_hours: Maximum hours to wait (default: 12)
            operation: Operation name for cost tracking
            model: Model name for cost tracking

        Returns:
            Dict mapping custom_id -> parsed result

        Raises:
            Exception: If batch processing fails

        Example:
            >>> def create_summary_request(section, idx):
            >>>     return BatchRequest(
            >>>         custom_id=f"summary_{idx}",
            >>>         method="POST",
            >>>         url="/v1/chat/completions",
            >>>         body={"model": "gpt-4o-mini", "messages": [...]}
            >>>     )
            >>>
            >>> results = client.process_batch(
            >>>     items=sections,
            >>>     create_request_fn=create_summary_request,
            >>>     parse_response_fn=lambda r: r['choices'][0]['message']['content']
            >>> )
        """
        self.logger.info(f"Processing {len(items)} items via Batch API...")

        # Step 1: Create batch requests
        requests = self._create_batch_requests(items, create_request_fn)

        # Step 2: Submit batch
        batch_id = self._submit_batch(requests)

        # Step 3: Poll for completion
        completed_batch = self._poll_batch(
            batch_id, poll_interval=poll_interval, timeout_seconds=timeout_hours * 3600
        )

        # Step 4: Parse results
        results = self._parse_batch_results(
            completed_batch, parse_response_fn, operation=operation, model=model
        )

        return results

    def _create_batch_requests(
        self, items: List[Any], create_request_fn: Callable[[Any, int], BatchRequest]
    ) -> List[BatchRequest]:
        """
        Create batch requests from items.

        Args:
            items: List of items to process
            create_request_fn: Function to create BatchRequest from item and index

        Returns:
            List of BatchRequest objects
        """
        self.logger.info(f"Creating {len(items)} batch requests...")

        requests = []
        for idx, item in enumerate(items):
            try:
                request = create_request_fn(item, idx)
                requests.append(request)
            except Exception as e:
                self.logger.error(f"Failed to create request for item {idx}: {sanitize_error(e)}")
                raise

        self.logger.info(f"✓ Created {len(requests)} batch requests")
        return requests

    def _submit_batch(self, requests: List[BatchRequest]) -> str:
        """
        Submit batch to OpenAI.

        Args:
            requests: List of BatchRequest objects

        Returns:
            Batch ID

        Raises:
            Exception: If submission fails
        """
        self.logger.info(f"Submitting batch with {len(requests)} requests...")

        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            temp_path = f.name
            for idx, request in enumerate(requests):
                json_line = json.dumps(request.to_dict())
                f.write(json_line + "\n")

                # Debug: Log first request to verify format
                if idx == 0:
                    self.logger.debug(f"First batch request: {json_line[:200]}...")

        try:
            # Debug: Log file size
            file_size = Path(temp_path).stat().st_size
            self.logger.debug(
                f"Batch file created: {temp_path} ({file_size} bytes, {len(requests)} lines)"
            )

            # Upload file
            self.logger.debug(f"Uploading batch file to OpenAI...")
            with open(temp_path, "rb") as f:
                batch_file = self.client.files.create(file=f, purpose="batch")

            self.logger.debug(f"File uploaded: {batch_file.id}")

            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            self.logger.info(f"✓ Batch submitted: {batch.id} (file: {batch_file.id})")
            return batch.id

        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except FileNotFoundError:
                pass  # File already deleted, no action needed
            except PermissionError as e:
                logger.warning(f"Could not delete temp file {temp_path}: Permission denied")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {type(e).__name__}")

    def _poll_batch(
        self, batch_id: str, poll_interval: int = 5, timeout_seconds: int = 43200  # 12 hours
    ):
        """
        Poll batch until completion.

        Args:
            batch_id: Batch ID to poll
            poll_interval: Seconds between checks
            timeout_seconds: Maximum seconds to wait

        Returns:
            Completed batch object

        Raises:
            TimeoutError: If batch doesn't complete within timeout
            Exception: If batch fails or is cancelled
        """
        self.logger.info(
            f"Polling batch {batch_id} "
            f"(interval: {poll_interval}s, timeout: {timeout_seconds // 3600}h)"
        )

        start_time = time.time()
        last_status = None

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Batch {batch_id} timeout after {elapsed / 3600:.1f} hours. "
                    f"Consider using smaller batches or fast mode."
                )

            # Get batch status
            batch = self.client.batches.retrieve(batch_id)

            # Log progress if status changed
            if batch.status != last_status:
                if batch.request_counts:
                    completed = batch.request_counts.completed or 0
                    total = batch.request_counts.total or 0
                    progress = f"{completed}/{total}" if total > 0 else "unknown"
                else:
                    progress = "unknown"

                self.logger.info(
                    f"Batch {batch_id}: {batch.status} "
                    f"(progress: {progress}, elapsed: {elapsed / 60:.1f}m)"
                )
                last_status = batch.status

            # Check completion status
            if batch.status == "completed":
                self.logger.info(f"✓ Batch {batch_id} completed")
                return batch

            elif batch.status == "failed":
                error_msg = getattr(batch, "errors", "Unknown error")
                raise Exception(f"Batch {batch_id} failed: {error_msg}")

            elif batch.status == "expired":
                raise Exception(f"Batch {batch_id} expired (exceeded 24h window)")

            elif batch.status == "cancelled":
                raise Exception(f"Batch {batch_id} was cancelled")

            # Sleep before next poll
            time.sleep(poll_interval)

    def _parse_batch_results(
        self,
        batch,
        parse_response_fn: Callable[[Dict[str, Any]], Any],
        operation: str = "batch",
        model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Parse batch results.

        Args:
            batch: Completed batch object
            parse_response_fn: Function to parse API response
            operation: Operation name for cost tracking
            model: Model name for cost tracking

        Returns:
            Dict mapping custom_id -> parsed result
        """
        self.logger.info(f"Parsing batch results from {batch.id}...")

        # Check batch completion status (ALWAYS log for debugging)
        request_counts = getattr(batch, 'request_counts', None)
        if request_counts:
            completed = getattr(request_counts, 'completed', 0) or 0
            failed = getattr(request_counts, 'failed', 0) or 0
            total = getattr(request_counts, 'total', 0) or 0
            self.logger.warning(
                f"⚠️  Batch request_counts: completed={completed}, failed={failed}, total={total}"
            )

            # Critical: If total=0, batch processed NO requests!
            if total == 0:
                self.logger.error(
                    f"🔴 BATCH PROCESSED 0 REQUESTS! This indicates malformed batch input. "
                    f"Batch ID: {batch.id}, Status: {batch.status}"
                )
        else:
            self.logger.warning(f"⚠️  Batch has NO request_counts attribute!")

        # Download output file
        output_file_id = batch.output_file_id
        if not output_file_id:
            # Check if there's an error file instead
            error_file_id = getattr(batch, 'error_file_id', None)
            if error_file_id:
                # Download and log errors
                try:
                    error_content = self.client.files.content(error_file_id)
                    error_str = error_content.read().decode("utf-8")
                    self.logger.error(f"Batch {batch.id} has errors:\n{error_str[:500]}")
                except Exception as e:
                    self.logger.error(f"Failed to read error file: {e}")

            raise Exception(
                f"Batch has no output file. "
                f"Status: {batch.status}, "
                f"Request counts: {getattr(batch, 'request_counts', 'unknown')}"
            )

        file_content = self.client.files.content(output_file_id)
        content_str = file_content.read().decode("utf-8")

        # Parse JSONL
        results = {}
        total_input_tokens = 0
        total_output_tokens = 0

        for line_num, line in enumerate(content_str.strip().split("\n"), 1):
            try:
                response = json.loads(line)

                # Extract custom_id
                custom_id = response.get("custom_id")
                if not custom_id:
                    self.logger.warning(f"Line {line_num}: Missing custom_id, skipping")
                    continue

                # Check for errors
                if response.get("error"):
                    error_msg = response["error"]
                    self.logger.error(
                        f"Request {custom_id} failed: {sanitize_error(str(error_msg))}"
                    )
                    results[custom_id] = None
                    continue

                # Parse response
                response_body = response.get("response", {}).get("body", {})
                if not response_body:
                    self.logger.warning(f"Request {custom_id}: Empty response body")
                    results[custom_id] = None
                    continue

                # Track tokens
                usage = response_body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                # Parse result using custom function
                try:
                    parsed_result = parse_response_fn(response_body)

                    # Debug: Log parsed result details
                    if line_num <= 3:  # Only log first 3 for debugging
                        self.logger.debug(
                            f"Request {custom_id}: "
                            f"in={input_tokens} out={output_tokens} "
                            f"result_len={len(parsed_result) if parsed_result else 0}"
                        )

                    results[custom_id] = parsed_result
                except Exception as e:
                    self.logger.error(
                        f"Failed to parse response for {custom_id}: {sanitize_error(e)}"
                    )
                    results[custom_id] = None

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse line {line_num}: {sanitize_error(e)}")
                continue

        # Track costs if tracker available
        if self.tracker:
            try:
                self.tracker.track_llm(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    operation=operation,
                )
            except Exception as e:
                self.logger.warning(f"Failed to track costs: {sanitize_error(e)}")

        self.logger.info(
            f"✓ Parsed {len(results)} results "
            f"({total_input_tokens} input tokens, {total_output_tokens} output tokens)"
        )

        return results


# Example usage
if __name__ == "__main__":
    print("=== Batch API Client Example ===\n")

    # This is a demonstration - requires actual OpenAI client
    print("Note: This example requires OPENAI_API_KEY to run")
    print("\nExample usage:")
    print(
        """
    from src.utils import BatchAPIClient, APIClientFactory

    # Create clients
    openai_client = APIClientFactory.create_openai()
    batch_client = BatchAPIClient(openai_client, logger)

    # Define request creator
    def create_summary_request(section, idx):
        return BatchRequest(
            custom_id=f"summary_{idx}",
            method="POST",
            url="/v1/chat/completions",
            body={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Summarize this text:"},
                    {"role": "user", "content": section['text']}
                ]
            }
        )

    # Process batch
    results = batch_client.process_batch(
        items=sections,
        create_request_fn=create_summary_request,
        parse_response_fn=lambda r: r['choices'][0]['message']['content'],
        operation="summary"
    )

    # Use results
    for custom_id, summary in results.items():
        print(f"{custom_id}: {summary}")
    """
    )
