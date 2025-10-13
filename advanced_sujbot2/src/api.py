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
            from .document_reader import LegalDocumentReader
            self._document_reader = LegalDocumentReader()
            self.logger.info("DocumentReader initialized")
        return self._document_reader

    @property
    def indexing_pipeline(self):
        """Get or initialize IndexingPipeline."""
        if self._indexing_pipeline is None:
            from .embeddings import LegalEmbedder, EmbeddingConfig
            from .indexing import MultiDocumentVectorStore, VectorStoreConfig
            from .chunker import LegalChunkingPipeline, ChunkingConfig

            # Initialize components
            embedding_config = EmbeddingConfig(**self.config.get('embeddings', {}))
            embedder = LegalEmbedder(embedding_config)

            vector_config = VectorStoreConfig(**self.config.get('indexing', {}))
            vector_store = MultiDocumentVectorStore(embedder, vector_config)

            chunking_config = ChunkingConfig(**self.config.get('chunking', {}))
            chunker = LegalChunkingPipeline(chunking_config)

            # Create pipeline wrapper
            self._indexing_pipeline = {
                'chunker': chunker,
                'embedder': embedder,
                'vector_store': vector_store
            }
            self.logger.info("IndexingPipeline initialized")
        return self._indexing_pipeline

    @property
    def hybrid_retriever(self):
        """Get or initialize HybridRetriever."""
        if self._hybrid_retriever is None:
            from .hybrid_retriever import create_hybrid_retriever, RetrievalConfig

            # Get vector store and embedder from indexing pipeline
            pipeline = self.indexing_pipeline
            vector_store = pipeline['vector_store']
            embedder = pipeline['embedder']

            # Create config
            retrieval_config = RetrievalConfig(**self.config.get('retrieval', {}))

            # Initialize retriever
            self._hybrid_retriever = create_hybrid_retriever(
                vector_store=vector_store,
                embedder=embedder,
                config=retrieval_config
            )
            self.logger.info("HybridRetriever initialized")
        return self._hybrid_retriever

    @property
    def cross_doc_retriever(self):
        """Get or initialize ComparativeRetriever."""
        if self._cross_doc_retriever is None:
            from .cross_doc_retrieval import create_comparative_retriever

            # Get vector store and embedder from indexing pipeline
            pipeline = self.indexing_pipeline
            vector_store = pipeline['vector_store']
            embedder = pipeline['embedder']

            # Create retriever
            cross_doc_config = self.config.get('cross_document', {})
            self._cross_doc_retriever = create_comparative_retriever(
                vector_store=vector_store,
                embedder=embedder,
                config=cross_doc_config
            )
            self.logger.info("ComparativeRetriever initialized")
        return self._cross_doc_retriever

    @property
    def compliance_analyzer(self):
        """Get or initialize ComplianceAnalyzer."""
        if self._compliance_analyzer is None:
            from .compliance_analyzer import ComplianceAnalyzer
            from anthropic import Anthropic

            # Get API key from environment or config
            api_key = os.environ.get('CLAUDE_API_KEY') or self.config.get('llm', {}).get('api_key')
            if not api_key:
                raise APIKeyError("Claude API key not found. Set CLAUDE_API_KEY environment variable.")

            # Create LLM client
            llm_client = Anthropic(api_key=api_key)

            # Initialize analyzer
            compliance_config = self.config.get('compliance', {})
            self._compliance_analyzer = ComplianceAnalyzer(
                config=compliance_config,
                cross_doc_retriever=self.cross_doc_retriever,
                llm_client=llm_client
            )
            self.logger.info("ComplianceAnalyzer initialized")
        return self._compliance_analyzer

    @property
    def query_processor(self):
        """Get or initialize QueryProcessor."""
        if self._query_processor is None:
            try:
                from .query_processor import QueryProcessor

                # Get API key
                api_key = os.environ.get('CLAUDE_API_KEY') or self.config.get('llm', {}).get('api_key')

                # Create config dict for query processor
                query_config = self.config.get('query_processing', {})
                query_config['claude_api_key'] = api_key

                self._query_processor = QueryProcessor(query_config)
                self.logger.info("QueryProcessor initialized")
            except ImportError:
                self.logger.warning("QueryProcessor not found - needs to be copied from multi-agent/src/")
                self._query_processor = None
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

            # Get chunks from indexed documents
            pipeline = self.indexing_pipeline
            vector_store = pipeline['vector_store']

            # Get contract chunks
            contract_metadata = vector_store.metadata_stores.get(contract_id, {})
            contract_chunks = list(contract_metadata.values())

            # Get law chunks
            law_chunks = []
            for law_id in law_ids:
                law_metadata = vector_store.metadata_stores.get(law_id, {})
                law_chunks.extend(law_metadata.values())

            # Run compliance analysis
            report = await self.compliance_analyzer.analyze_compliance(
                contract_chunks=contract_chunks,
                law_chunks=law_chunks,
                contract_id=contract_id,
                law_ids=law_ids,
                mode=request.mode.value
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

            document = await self.document_reader.read_document(document_path, document_type)

            # Step 2: Chunk document
            if progress_callback:
                await progress_callback(IndexingProgress(
                    document_id=document_id,
                    document_name=doc_name,
                    stage=IndexingStage.CHUNKING,
                    progress=0.25,
                    message="Chunking document..."
                ))

            pipeline = self.indexing_pipeline
            chunks = await pipeline['chunker'].chunk(document)

            # Step 3: Generate embeddings and index
            if progress_callback:
                await progress_callback(IndexingProgress(
                    document_id=document_id,
                    document_name=doc_name,
                    stage=IndexingStage.EMBEDDING,
                    progress=0.5,
                    message="Generating embeddings..."
                ))

            await pipeline['vector_store'].add_document(
                chunks=chunks,
                document_id=document_id,
                document_type=document_type,
                metadata={'path': document_path, 'name': doc_name}
            )

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

        # Step 1: Process query if processor available
        processed_query = None
        if self.query_processor:
            try:
                processed_query = await self.query_processor.process(query)
                self.logger.info(f"Query processed: intent={processed_query.intent.value}, "
                                f"complexity={processed_query.complexity.value}")
            except Exception as e:
                self.logger.warning(f"Query processing failed, continuing with simple retrieval: {e}")

        # Simple fallback if no processor or processing failed
        if not processed_query:
            processed_query = ProcessedQuery(
                original_query=query,
                normalized_query=query.lower().strip(),
                retrieval_strategy=RetrievalStrategy.HYBRID,
                sub_queries=[],
                query_type="factual",
                expanded_terms=[],
            )

        # Step 2: Retrieve relevant chunks
        self.logger.info(f"Retrieving chunks (top_k={top_k})...")
        retrieval_results = await self.hybrid_retriever.search(
            query=query,
            top_k=top_k,
            document_ids=document_ids
        )

        # Convert to Source objects
        sources = []
        for result in retrieval_results:
            sources.append(Source(
                document_id=result.chunk.document_id,
                chunk_id=result.chunk.chunk_id,
                content=result.chunk.content,
                legal_reference=result.chunk.legal_reference or "N/A",
                score=result.score,
                metadata=result.chunk.metadata
            ))

        # Step 3: Generate answer using LLM
        self.logger.info("Generating answer with LLM...")
        try:
            from anthropic import Anthropic

            # Get API key
            api_key = os.environ.get('CLAUDE_API_KEY') or self.config.get('llm', {}).get('api_key')
            if not api_key:
                raise APIKeyError("Claude API key not found for answer generation")

            client = Anthropic(api_key=api_key)

            # Build context from sources
            context_parts = []
            for i, source in enumerate(sources[:5], 1):  # Use top 5 sources
                ref = source.legal_reference if source.legal_reference != "N/A" else "Provision"
                context_parts.append(f"[{i}] {ref}:\n{source.content}\n")

            context = "\n".join(context_parts)

            # Generate answer
            prompt = f"""Based on the following legal provisions, answer the question.

Question: {query}

Legal Provisions:
{context}

Provide a clear, accurate answer based on the provisions above. Include references to specific provisions in your answer."""

            response_message = client.messages.create(
                model=self.config.get('llm', {}).get('main_model', 'claude-sonnet-4-5-20250929'),
                max_tokens=self.config.get('llm', {}).get('max_tokens', 4000),
                temperature=self.config.get('llm', {}).get('temperature', 0.1),
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response_message.content[0].text
            confidence = min(sources[0].score if sources else 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            answer = f"Retrieved {len(sources)} relevant provisions, but answer generation failed: {e}"
            confidence = 0.0

        response = QueryResponse(
            query=query,
            answer=answer,
            sources=sources,
            processed_query=processed_query,
            confidence=confidence,
        )

        self.logger.info(f"Query complete: {len(sources)} sources, confidence={confidence:.2f}")
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

        from .knowledge_graph import GraphBuilder, ReferenceLinker, LegalKnowledgeGraph

        # Get documents
        documents = []
        pipeline = self.indexing_pipeline
        vector_store = pipeline['vector_store']

        for doc_id in document_ids:
            if doc_id in self._indexed_documents:
                doc_metadata = self._indexed_documents[doc_id]
                # Create mock document structure for graph building
                # In production, we'd load the full document structure
                documents.append({
                    'document_id': doc_id,
                    'document_type': doc_metadata['type'],
                    'chunks': list(vector_store.metadata_stores.get(doc_id, {}).values())
                })

        # Build graph
        builder = GraphBuilder()
        kg = LegalKnowledgeGraph()

        # Add documents to graph (simplified - would need full document structure)
        for doc in documents:
            for chunk in doc['chunks']:
                from .knowledge_graph import GraphNode, NodeType
                node = GraphNode(
                    node_id=chunk.chunk_id,
                    node_type=NodeType.CHUNK,
                    content=chunk.content,
                    legal_reference=chunk.legal_reference,
                    hierarchy_path=chunk.hierarchy_path,
                    hierarchy_level=0,
                    metadata=chunk.metadata
                )
                kg.add_node(node)

        # Link references
        ref_linker = ReferenceLinker()
        ref_linker.link_references(kg)

        self.logger.info(f"Knowledge graph built: {kg.node_count} nodes, {kg.edge_count} edges")
        return kg

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
