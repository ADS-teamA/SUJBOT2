"""
Main API Interface for SUJBOT2

This module provides the primary ComplianceChecker class, which orchestrates
all components for legal compliance checking with a clean async API.
"""

import asyncio
import logging
import uuid
import os
from pathlib import Path
from typing import Optional, Dict, List, Callable, AsyncIterator
from datetime import datetime

from .config import Config, load_config
from .models import (
    ComplianceCheckRequest,
    ComplianceReport,
    ComplianceIssue,
    ComplianceMode,
    IndexingProgress,
    AnalysisProgress,
    IndexingStage,
    AnalysisStage,
    QueryResponse,
    ProcessedQuery,
    Source,
    RetrievalStrategy,
    SeverityLevel,
)
from .exceptions import (
    DocumentNotFoundError,
    IndexingError,
    ComplianceCheckError,
    APIKeyError,
)
from .batch_processor import BatchProcessor


logger = logging.getLogger(__name__)


class ComplianceChecker:
    """
    Main API for legal compliance checking.

    This is the primary entry point for SUJBOT2. It orchestrates
    all components (document reading, indexing, retrieval, analysis, knowledge graph)
    to provide a simple, high-level async API.

    Features:
    - Async/await interface
    - Progress callbacks for long-running operations
    - Batch processing support
    - Interactive query interface
    - Knowledge graph generation
    - Multiple export formats

    Usage:
        >>> checker = ComplianceChecker(config_path="config.yaml")
        >>> request = ComplianceCheckRequest(
        ...     contract_path="contract.pdf",
        ...     law_paths=["law.pdf"]
        ... )
        >>> report = await checker.check_compliance(request)
        >>> print(f"Compliance score: {report.overall_compliance_score:.2%}")

    Attributes:
        config: Configuration object
        document_reader: DocumentReader instance (initialized on demand)
        indexing_pipeline: IndexingPipeline instance (initialized on demand)
        hybrid_retriever: HybridRetriever instance (initialized on demand)
        cross_doc_retriever: ComparativeRetriever instance (initialized on demand)
        compliance_analyzer: ComplianceAnalyzer instance (initialized on demand)
        query_processor: QueryProcessor instance (initialized on demand)
        knowledge_graph: LegalKnowledgeGraph instance (built on demand)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize compliance checker.

        Args:
            config_path: Path to YAML config file
            config: Dict config (overrides file)

        Raises:
            ConfigurationError: If config is invalid
            APIKeyError: If Claude API key is missing

        Example:
            >>> # From config file
            >>> checker = ComplianceChecker(config_path="config.yaml")
            >>>
            >>> # From dict
            >>> checker = ComplianceChecker(config={
            ...     "llm": {"api_key": "sk-..."},
            ...     "retrieval": {"hybrid_alpha": 0.8}
            ... })
        """
        # Load configuration
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = load_config(config_path=config_path, config=config)

        # Setup logging
        self._setup_logging()

        # Initialize components (lazy loading - initialized on first use)
        self._document_reader = None
        self._indexing_pipeline = None
        self._hybrid_retriever = None
        self._cross_doc_retriever = None
        self._compliance_analyzer = None
        self._query_processor = None
        self.knowledge_graph = None

        # Document storage (document_id -> document metadata)
        self._indexed_documents: Dict[str, Dict] = {}

        # Batch processor
        self._batch_processor = None

        self.logger = logging.getLogger(__name__)
        self.logger.info("ComplianceChecker initialized")

    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_level = self.config.get("api.logging.level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Lazy-loading properties for components

    @property
    def document_reader(self):
        """Get or initialize DocumentReader."""
        if self._document_reader is None:
            # TODO: Import and initialize DocumentReader
            # from .document_reader import DocumentReader
            # self._document_reader = DocumentReader(self.config)
            self.logger.warning("DocumentReader not yet implemented - placeholder")
            self._document_reader = "DocumentReader placeholder"
        return self._document_reader

    @property
    def indexing_pipeline(self):
        """Get or initialize IndexingPipeline."""
        if self._indexing_pipeline is None:
            # TODO: Import and initialize IndexingPipeline
            # from .indexing_pipeline import IndexingPipeline
            # self._indexing_pipeline = IndexingPipeline(self.config)
            self.logger.warning("IndexingPipeline not yet implemented - placeholder")
            self._indexing_pipeline = "IndexingPipeline placeholder"
        return self._indexing_pipeline

    @property
    def hybrid_retriever(self):
        """Get or initialize HybridRetriever."""
        if self._hybrid_retriever is None:
            # TODO: Import and initialize HybridRetriever
            # from .hybrid_retriever import HybridRetriever
            # self._hybrid_retriever = HybridRetriever(self.config)
            self.logger.warning("HybridRetriever not yet implemented - placeholder")
            self._hybrid_retriever = "HybridRetriever placeholder"
        return self._hybrid_retriever

    @property
    def cross_doc_retriever(self):
        """Get or initialize ComparativeRetriever."""
        if self._cross_doc_retriever is None:
            # TODO: Import and initialize ComparativeRetriever
            # from .comparative_retriever import ComparativeRetriever
            # self._cross_doc_retriever = ComparativeRetriever(self.config)
            self.logger.warning("ComparativeRetriever not yet implemented - placeholder")
            self._cross_doc_retriever = "ComparativeRetriever placeholder"
        return self._cross_doc_retriever

    @property
    def compliance_analyzer(self):
        """Get or initialize ComplianceAnalyzer."""
        if self._compliance_analyzer is None:
            # TODO: Import and initialize ComplianceAnalyzer
            # from .compliance_analyzer import ComplianceAnalyzer
            # from anthropic import Anthropic
            # self._compliance_analyzer = ComplianceAnalyzer(
            #     self.config,
            #     self.cross_doc_retriever,
            #     llm_client=Anthropic(api_key=self.config.get("llm.api_key"))
            # )
            self.logger.warning("ComplianceAnalyzer not yet implemented - placeholder")
            self._compliance_analyzer = "ComplianceAnalyzer placeholder"
        return self._compliance_analyzer

    @property
    def query_processor(self):
        """Get or initialize QueryProcessor."""
        if self._query_processor is None:
            # TODO: Import and initialize QueryProcessor
            # from .query_processor import QueryProcessor
            # self._query_processor = QueryProcessor(self.config)
            self.logger.warning("QueryProcessor not yet implemented - placeholder")
            self._query_processor = "QueryProcessor placeholder"
        return self._query_processor

    @property
    def batch_processor(self) -> BatchProcessor:
        """Get or initialize BatchProcessor."""
        if self._batch_processor is None:
            self._batch_processor = BatchProcessor(self)
        return self._batch_processor

    # Main API methods

    async def check_compliance(
        self,
        request: ComplianceCheckRequest,
        progress_callback: Optional[Callable] = None
    ) -> ComplianceReport:
        """
        Perform complete compliance check.

        This is the main method for checking if a contract complies with
        legal requirements. It orchestrates all steps: indexing, retrieval,
        analysis, and reporting.

        Args:
            request: ComplianceCheckRequest with contract and law paths
            progress_callback: Optional async callback(progress: AnalysisProgress)

        Returns:
            ComplianceReport with all findings

        Raises:
            DocumentNotFoundError: If file not found
            IndexingError: If indexing fails
            ComplianceCheckError: If analysis fails

        Example:
            >>> async def progress(p):
            ...     print(f"{p.stage}: {p.message} ({p.progress:.0%})")
            >>>
            >>> request = ComplianceCheckRequest(
            ...     contract_path="smlouva.pdf",
            ...     law_paths=["zakon_89_2012.pdf"],
            ...     mode="exhaustive"
            ... )
            >>> report = await checker.check_compliance(request, progress_callback=progress)
        """
        try:
            self.logger.info(f"Starting compliance check: {request.contract_path}")

            # Step 1: Index contract
            self.logger.info("Indexing contract...")
            contract_id = await self.index_document(
                request.contract_path,
                document_type="contract",
                progress_callback=progress_callback
            )

            # Step 2: Index laws
            self.logger.info(f"Indexing {len(request.law_paths)} law documents...")
            law_ids = []
            for law_path in request.law_paths:
                law_id = await self.index_document(
                    law_path,
                    document_type="law_code",
                    progress_callback=progress_callback
                )
                law_ids.append(law_id)

            # Step 3: Build knowledge graph (optional)
            if request.generate_graph:
                if progress_callback:
                    await progress_callback(AnalysisProgress(
                        stage=AnalysisStage.GRAPH_BUILDING,
                        progress=0.3,
                        message="Building knowledge graph..."
                    ))

                self.logger.info("Building knowledge graph...")
                self.knowledge_graph = await self.build_knowledge_graph(
                    [contract_id] + law_ids
                )

            # Step 4: Run compliance analysis
            if progress_callback:
                await progress_callback(AnalysisProgress(
                    stage=AnalysisStage.ANALYSIS,
                    progress=0.5,
                    message="Running compliance analysis..."
                ))

            self.logger.info("Running compliance analysis...")

            # TODO: Once components are implemented, this will do actual analysis
            # For now, return a placeholder report
            report = await self._create_placeholder_report(
                contract_id,
                law_ids,
                request.mode
            )

            if progress_callback:
                await progress_callback(AnalysisProgress(
                    stage=AnalysisStage.COMPLETE,
                    progress=1.0,
                    message="Compliance check complete",
                    issues_found=report.total_issues
                ))

            self.logger.info(f"Compliance check complete: {report.total_issues} issues found")
            return report

        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}")
            raise ComplianceCheckError(f"Compliance check failed: {e}") from e

    async def index_document(
        self,
        document_path: str,
        document_type: str = "contract",
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Index a single document.

        Parses document, chunks content, generates embeddings, and stores
        in vector and keyword indexes.

        Args:
            document_path: Path to PDF document
            document_type: 'contract' | 'law_code' | 'regulation'
            progress_callback: Optional callback for progress

        Returns:
            document_id: Unique identifier for indexed document

        Raises:
            DocumentNotFoundError: If file doesn't exist
            IndexingError: If indexing fails

        Example:
            >>> document_id = await checker.index_document(
            ...     "documents/contract.pdf",
            ...     document_type="contract"
            ... )
            >>> print(f"Indexed: {document_id}")
        """
        if not Path(document_path).exists():
            raise DocumentNotFoundError(path=document_path)

        self.logger.info(f"Indexing document: {document_path}")

        # Generate document ID
        document_id = str(uuid.uuid4())
        doc_name = Path(document_path).name

        try:
            # Step 1: Parse document
            if progress_callback:
                await progress_callback(IndexingProgress(
                    document_id=document_id,
                    document_name=doc_name,
                    stage=IndexingStage.PARSING,
                    progress=0.0,
                    message="Parsing document..."
                ))

            # TODO: Actual document parsing
            # document = await self.document_reader.read_document(document_path, document_type)

            # Step 2: Chunk document
            if progress_callback:
                await progress_callback(IndexingProgress(
                    document_id=document_id,
                    document_name=doc_name,
                    stage=IndexingStage.CHUNKING,
                    progress=0.25,
                    message="Chunking document..."
                ))

            # TODO: Actual chunking
            # chunks = await self.indexing_pipeline.chunk_document(document)

            # Step 3: Generate embeddings and index
            if progress_callback:
                await progress_callback(IndexingProgress(
                    document_id=document_id,
                    document_name=doc_name,
                    stage=IndexingStage.EMBEDDING,
                    progress=0.5,
                    message="Generating embeddings..."
                ))

            # TODO: Actual indexing
            # await self.indexing_pipeline.index_chunks(chunks, document_id, document_type)

            # Step 4: Complete
            if progress_callback:
                await progress_callback(IndexingProgress(
                    document_id=document_id,
                    document_name=doc_name,
                    stage=IndexingStage.COMPLETE,
                    progress=1.0,
                    message="Indexing complete"
                ))

            # Store document metadata
            self._indexed_documents[document_id] = {
                "path": document_path,
                "type": document_type,
                "indexed_at": datetime.now(),
            }

            self.logger.info(f"Document indexed: {document_id}")
            return document_id

        except Exception as e:
            self.logger.error(f"Indexing failed for {document_path}: {e}")
            raise IndexingError(f"Failed to index document: {e}") from e

    async def query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> QueryResponse:
        """
        Ask a question about indexed documents.

        Retrieves relevant provisions and generates an answer using Claude.

        Args:
            query: Natural language question
            document_ids: Filter to specific documents (None = all)
            top_k: Number of results to return

        Returns:
            QueryResponse with answer and sources

        Example:
            >>> response = await checker.query(
            ...     "What are the contractor's obligations under §89?",
            ...     top_k=5
            ... )
            >>> print(response.answer)
            >>> for source in response.sources:
            ...     print(f"  - {source.legal_reference}: {source.content[:100]}")
        """
        self.logger.info(f"Processing query: {query}")

        # TODO: Actual query processing
        # For now, return placeholder response
        processed_query = ProcessedQuery(
            original_query=query,
            normalized_query=query.lower().strip(),
            retrieval_strategy=RetrievalStrategy.HYBRID,
            sub_queries=[],
            query_type="factual",
            expanded_terms=[],
        )

        response = QueryResponse(
            query=query,
            answer="[Placeholder answer - query processing not yet implemented]",
            sources=[],
            processed_query=processed_query,
            confidence=0.0,
        )

        return response

    async def build_knowledge_graph(
        self,
        document_ids: List[str]
    ):
        """
        Build knowledge graph for indexed documents.

        Creates a graph representation of legal provisions and their
        relationships (references, semantic similarities, hierarchies).

        Args:
            document_ids: Documents to include in graph

        Returns:
            LegalKnowledgeGraph instance

        Example:
            >>> kg = await checker.build_knowledge_graph([contract_id, law_id])
            >>> print(f"Graph has {kg.node_count} nodes")
        """
        self.logger.info(f"Building knowledge graph for {len(document_ids)} documents")

        # TODO: Actual graph building
        # For now, return placeholder
        self.logger.warning("Knowledge graph building not yet implemented")
        return None

    async def batch_check_compliance(
        self,
        requests: List[ComplianceCheckRequest],
        max_parallel: int = 3
    ) -> AsyncIterator[ComplianceReport]:
        """
        Process multiple compliance checks in parallel.

        Yields reports as they complete, rather than waiting for all to finish.

        Args:
            requests: List of ComplianceCheckRequest
            max_parallel: Max concurrent checks

        Yields:
            ComplianceReport for each request

        Example:
            >>> requests = [...]
            >>> async for report in checker.batch_check_compliance(requests):
            ...     print(f"Report ready: {report.report_id}")
        """
        semaphore = asyncio.Semaphore(max_parallel)

        async def check_with_semaphore(req):
            async with semaphore:
                return await self.check_compliance(req)

        tasks = [check_with_semaphore(req) for req in requests]

        for task in asyncio.as_completed(tasks):
            report = await task
            yield report

    def export_report(
        self,
        report: ComplianceReport,
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export compliance report to file.

        Args:
            report: ComplianceReport to export
            output_path: Output file path
            format: 'json' | 'markdown' | 'html' | 'pdf'

        Example:
            >>> checker.export_report(report, "report.json", format="json")
            >>> checker.export_report(report, "report.md", format="markdown")
        """
        self.logger.info(f"Exporting report to {output_path} (format: {format})")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._export_json(report, output_path)
        elif format == "markdown":
            self._export_markdown(report, output_path)
        elif format == "html":
            self._export_html(report, output_path)
        elif format == "pdf":
            self._export_pdf(report, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Report exported to {output_path}")

    def _export_json(self, report: ComplianceReport, output_path: Path):
        """Export report as JSON."""
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    def _export_markdown(self, report: ComplianceReport, output_path: Path):
        """Export report as Markdown."""
        lines = [
            f"# Compliance Report: {report.report_id}",
            "",
            f"**Generated:** {report.timestamp.isoformat()}",
            f"**Mode:** {report.mode.value}",
            f"**Overall Compliance Score:** {report.overall_compliance_score:.2%}",
            "",
            "## Summary",
            "",
            report.summary,
            "",
            f"## Issues Found: {report.total_issues}",
            "",
        ]

        for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW]:
            issues = [i for i in report.issues if i.severity == severity]
            if issues:
                lines.append(f"### {severity.value.upper()} ({len(issues)})")
                lines.append("")
                for issue in issues:
                    lines.extend([
                        f"#### {issue.title}",
                        "",
                        issue.description,
                        "",
                        f"**Law Provision:** {issue.law_provision.legal_reference}",
                        "",
                        "**Recommendations:**",
                        "",
                    ])
                    for rec in issue.recommendations:
                        lines.append(f"- {rec}")
                    lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _export_html(self, report: ComplianceReport, output_path: Path):
        """Export report as HTML."""
        # TODO: Implement HTML export with proper template
        self.logger.warning("HTML export not yet fully implemented")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<html><body><h1>Compliance Report (placeholder)</h1></body></html>")

    def _export_pdf(self, report: ComplianceReport, output_path: Path):
        """Export report as PDF."""
        # TODO: Implement PDF export (could use reportlab or convert from HTML)
        self.logger.warning("PDF export not yet implemented")
        raise NotImplementedError("PDF export not yet implemented")

    async def _create_placeholder_report(
        self,
        contract_id: str,
        law_ids: List[str],
        mode: ComplianceMode
    ) -> ComplianceReport:
        """Create a placeholder report for testing."""
        return ComplianceReport(
            report_id=str(uuid.uuid4()),
            contract_id=contract_id,
            law_ids=law_ids,
            mode=mode,
            overall_compliance_score=0.85,
            total_issues=0,
            issues=[],
            summary="Compliance analysis complete. (Placeholder report - components not yet implemented)",
            timestamp=datetime.now(),
            metadata={
                "placeholder": True,
                "note": "This is a placeholder report. Actual analysis will be implemented once all components are ready."
            }
        )


__all__ = [
    "ComplianceChecker",
]
