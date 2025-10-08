"""
Batch Processing for SUJBOT2

This module handles batch processing of multiple compliance checks,
with support for parallel execution, error handling, and progress tracking.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Callable, AsyncIterator
from pathlib import Path

from .models import (
    ComplianceCheckRequest,
    ComplianceReport,
    BatchResult,
    AnalysisProgress,
    AnalysisStage,
)
from .exceptions import BatchProcessingError


logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process multiple compliance checks in batch.

    Handles parallel execution with concurrency control, error recovery,
    and progress tracking for batch operations.

    Example:
        >>> checker = ComplianceChecker(config_path="config.yaml")
        >>> batch_processor = BatchProcessor(checker)
        >>> requests = [...]
        >>> result = await batch_processor.process_batch(requests)
    """

    def __init__(self, compliance_checker):
        """
        Initialize batch processor.

        Args:
            compliance_checker: ComplianceChecker instance to use
        """
        self.checker = compliance_checker
        self.logger = logging.getLogger(__name__)

    async def process_batch(
        self,
        batch_requests: List[ComplianceCheckRequest],
        batch_id: Optional[str] = None,
        max_parallel: int = 3,
        progress_callback: Optional[Callable] = None,
        continue_on_error: bool = True,
    ) -> BatchResult:
        """
        Process batch of compliance checks.

        Args:
            batch_requests: List of ComplianceCheckRequest
            batch_id: Optional batch identifier (generated if not provided)
            max_parallel: Maximum concurrent checks
            progress_callback: Optional callback for individual check progress
            continue_on_error: Continue processing if individual checks fail

        Returns:
            BatchResult with all reports and errors

        Raises:
            BatchProcessingError: If critical batch processing error occurs

        Example:
            >>> requests = [
            ...     ComplianceCheckRequest(
            ...         contract_path="contract1.pdf",
            ...         law_paths=["law.pdf"]
            ...     ),
            ...     ComplianceCheckRequest(
            ...         contract_path="contract2.pdf",
            ...         law_paths=["law.pdf"]
            ...     ),
            ... ]
            >>> result = await batch_processor.process_batch(requests)
            >>> print(f"Success rate: {result.success_rate:.1%}")
        """
        batch_id = batch_id or str(uuid.uuid4())
        start_time = datetime.now()

        self.logger.info(f"Starting batch processing: {batch_id} ({len(batch_requests)} requests)")

        reports: List[ComplianceReport] = []
        errors: List[dict] = []

        # Process requests with concurrency control
        semaphore = asyncio.Semaphore(max_parallel)

        async def process_request(index: int, request: ComplianceCheckRequest):
            """Process a single request with error handling."""
            async with semaphore:
                try:
                    self.logger.info(f"[{batch_id}] Processing request {index + 1}/{len(batch_requests)}")

                    # Wrap progress callback to add batch context
                    async def wrapped_progress_callback(progress):
                        if progress_callback:
                            # Add batch context to progress
                            if isinstance(progress, AnalysisProgress):
                                progress.metadata["batch_id"] = batch_id
                                progress.metadata["batch_index"] = index
                                progress.metadata["batch_total"] = len(batch_requests)
                            await progress_callback(progress)

                    # Run compliance check
                    report = await self.checker.check_compliance(
                        request,
                        progress_callback=wrapped_progress_callback if progress_callback else None
                    )

                    self.logger.info(f"[{batch_id}] Request {index + 1} completed: {report.total_issues} issues found")
                    return ("success", index, report)

                except Exception as e:
                    error_msg = f"Request {index + 1} failed: {str(e)}"
                    self.logger.error(f"[{batch_id}] {error_msg}")

                    if not continue_on_error:
                        raise BatchProcessingError(
                            f"Batch processing aborted after error in request {index + 1}",
                            batch_id=batch_id
                        ) from e

                    return ("error", index, {"request_index": index, "error": str(e), "type": type(e).__name__})

        # Create tasks for all requests
        tasks = [process_request(i, req) for i, req in enumerate(batch_requests)]

        # Process all requests
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Separate successful reports from errors
        for result_type, index, result in results:
            if result_type == "success":
                reports.append(result)
            else:
                errors.append(result)

        end_time = datetime.now()

        # Create batch result
        batch_result = BatchResult(
            batch_id=batch_id,
            total_requests=len(batch_requests),
            successful=len(reports),
            failed=len(errors),
            reports=reports,
            errors=errors,
            start_time=start_time,
            end_time=end_time,
            metadata={
                "max_parallel": max_parallel,
                "continue_on_error": continue_on_error,
            }
        )

        self.logger.info(
            f"Batch processing complete: {batch_id} "
            f"({batch_result.successful}/{batch_result.total_requests} successful, "
            f"duration: {batch_result.duration_seconds:.1f}s)"
        )

        return batch_result

    async def process_batch_streaming(
        self,
        batch_requests: List[ComplianceCheckRequest],
        batch_id: Optional[str] = None,
        max_parallel: int = 3,
    ) -> AsyncIterator[tuple[int, ComplianceReport]]:
        """
        Process batch with streaming results.

        Yields results as soon as each request completes, rather than
        waiting for the entire batch.

        Args:
            batch_requests: List of ComplianceCheckRequest
            batch_id: Optional batch identifier
            max_parallel: Maximum concurrent checks

        Yields:
            Tuple of (request_index, ComplianceReport)

        Example:
            >>> async for index, report in batch_processor.process_batch_streaming(requests):
            ...     print(f"Request {index} complete: {report.total_issues} issues")
        """
        batch_id = batch_id or str(uuid.uuid4())

        self.logger.info(f"Starting streaming batch processing: {batch_id}")

        semaphore = asyncio.Semaphore(max_parallel)

        async def check_with_semaphore(index: int, req: ComplianceCheckRequest):
            """Process request with semaphore."""
            async with semaphore:
                try:
                    report = await self.checker.check_compliance(req)
                    return (index, report, None)
                except Exception as e:
                    self.logger.error(f"[{batch_id}] Request {index} failed: {e}")
                    return (index, None, e)

        # Create tasks
        tasks = [
            asyncio.create_task(check_with_semaphore(i, req))
            for i, req in enumerate(batch_requests)
        ]

        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            index, report, error = await coro
            if report:
                yield (index, report)
            elif error:
                # Could yield error information if desired
                self.logger.error(f"[{batch_id}] Request {index} failed: {error}")

    async def process_directory(
        self,
        contract_dir: Path,
        law_paths: List[str],
        mode: str = "exhaustive",
        max_parallel: int = 3,
        contract_pattern: str = "*.pdf",
    ) -> BatchResult:
        """
        Process all contracts in a directory.

        Convenient method for batch processing all contracts in a folder
        against the same set of laws.

        Args:
            contract_dir: Directory containing contract PDFs
            law_paths: List of paths to law PDFs (same for all contracts)
            mode: Compliance check mode
            max_parallel: Maximum concurrent checks
            contract_pattern: Glob pattern for contract files

        Returns:
            BatchResult

        Example:
            >>> result = await batch_processor.process_directory(
            ...     contract_dir=Path("contracts/"),
            ...     law_paths=["laws/zakon_89_2012.pdf"],
            ...     mode="sample"
            ... )
        """
        contract_dir = Path(contract_dir)

        if not contract_dir.exists():
            raise BatchProcessingError(
                f"Contract directory not found: {contract_dir}",
                batch_id="dir_" + str(uuid.uuid4())
            )

        # Find all contract files
        contract_files = list(contract_dir.glob(contract_pattern))

        if not contract_files:
            self.logger.warning(f"No contracts found in {contract_dir} matching {contract_pattern}")
            return BatchResult(
                batch_id="dir_" + str(uuid.uuid4()),
                total_requests=0,
                successful=0,
                failed=0,
                reports=[],
                errors=[],
            )

        self.logger.info(f"Found {len(contract_files)} contracts in {contract_dir}")

        # Create requests
        requests = [
            ComplianceCheckRequest(
                contract_path=str(contract_file),
                law_paths=law_paths,
                mode=mode,
                metadata={"source_directory": str(contract_dir)}
            )
            for contract_file in contract_files
        ]

        # Process batch
        return await self.process_batch(
            requests,
            batch_id="dir_" + str(uuid.uuid4()),
            max_parallel=max_parallel
        )

    def export_batch_results(
        self,
        batch_result: BatchResult,
        output_dir: Path,
        format: str = "json"
    ) -> None:
        """
        Export all batch results to files.

        Args:
            batch_result: BatchResult to export
            output_dir: Directory to save results
            format: Export format (json, markdown, html, pdf)

        Example:
            >>> batch_processor.export_batch_results(
            ...     batch_result,
            ...     output_dir=Path("reports/batch_001/"),
            ...     format="json"
            ... )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Exporting batch results to {output_dir}")

        # Export individual reports
        for i, report in enumerate(batch_result.reports):
            output_path = output_dir / f"report_{i:03d}.{format}"
            self.checker.export_report(report, str(output_path), format=format)

        # Export batch summary
        summary_path = output_dir / f"batch_summary.json"
        import json
        with open(summary_path, "w") as f:
            json.dump({
                "batch_id": batch_result.batch_id,
                "total_requests": batch_result.total_requests,
                "successful": batch_result.successful,
                "failed": batch_result.failed,
                "success_rate": batch_result.success_rate,
                "duration_seconds": batch_result.duration_seconds,
                "errors": batch_result.errors,
                "metadata": batch_result.metadata,
            }, f, indent=2)

        self.logger.info(f"Batch results exported: {len(batch_result.reports)} reports, 1 summary")


__all__ = [
    "BatchProcessor",
]
