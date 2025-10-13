"""Service for chat/query operations with streaming support."""
from typing import AsyncIterator, List, Optional
import json
import asyncio
import logging

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.models.query import QuerySource
from app.services.rag_pipeline import get_rag_pipeline
from app.services.intent_router import IntentRouter, PipelineRoute
from app.rag.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class ChatService:
    """Service for chat and query operations with RAG pipeline integration."""

    def __init__(self):
        """Initialize chat service with RAG pipeline and intent router."""
        # Use Haiku model for faster, cheaper responses
        self.model = "claude-3-5-haiku-20241022"
        self.rag_pipeline = get_rag_pipeline()
        self.client = AsyncAnthropic(api_key=settings.CLAUDE_API_KEY) if settings.CLAUDE_API_KEY else None

        # Initialize Intent Router for intelligent message routing
        self.intent_router = IntentRouter(api_key=settings.CLAUDE_API_KEY) if settings.CLAUDE_API_KEY else None

    async def _categorize_documents(self, document_ids: List[str]) -> tuple[List[str], List[str]]:
        """
        Categorize documents into contracts and laws.

        Args:
            document_ids: List of document IDs

        Returns:
            (contract_ids, law_ids)
        """
        contract_ids = []
        law_ids = []

        # First, ensure document info is loaded from database
        try:
            if hasattr(self.rag_pipeline.vector_store, 'load_document_info_from_db'):
                await self.rag_pipeline.vector_store.load_document_info_from_db()
        except Exception as e:
            logger.warning(f"Failed to load document info from DB: {e}")

        for doc_id in document_ids:
            # Get document info from vector store
            doc_info = self.rag_pipeline.vector_store.get_document_info(doc_id)

            if doc_info:
                doc_type = doc_info.get('document_type', '')
                if doc_type == 'contract':
                    contract_ids.append(doc_id)
                elif doc_type in ['law_code', 'regulation']:
                    law_ids.append(doc_id)
                else:
                    # Unknown type - try to infer from ID
                    if 'contract' in doc_id.lower() or 'smlouva' in doc_id.lower():
                        contract_ids.append(doc_id)
                    elif 'law' in doc_id.lower() or 'zakon' in doc_id.lower() or 'sb' in doc_id.lower():
                        law_ids.append(doc_id)
                    else:
                        # Default to contract
                        contract_ids.append(doc_id)
            else:
                # Document not in store - try to infer from ID
                if 'contract' in doc_id.lower() or 'smlouva' in doc_id.lower():
                    contract_ids.append(doc_id)
                elif 'law' in doc_id.lower() or 'zakon' in doc_id.lower() or 'sb' in doc_id.lower():
                    law_ids.append(doc_id)

        return contract_ids, law_ids

    async def process_query_stream(
        self,
        query: str,
        document_ids: List[str],
        language: str = "cs"
    ) -> AsyncIterator[str]:
        """
        Process query and stream response chunks with intelligent routing.

        Uses Intent Router to automatically determine if this is a simple query
        or requires compliance analysis pipeline.

        Args:
            query: User query
            document_ids: List of document IDs to search
            language: Response language (cs/en)

        Yields:
            Response chunks as they arrive
        """
        logger.info(f"Streaming query: {query[:100]}... for {len(document_ids)} documents")

        try:
            # Step 1: Categorize documents (contracts vs laws)
            contract_ids, law_ids = await self._categorize_documents(document_ids)
            logger.info(f"Categorized: {len(contract_ids)} contracts, {len(law_ids)} laws")

            # Step 2: Route message using Intent Router
            if self.intent_router:
                routing_decision = await self.intent_router.route(
                    message=query,
                    available_contract_ids=contract_ids,
                    available_law_ids=law_ids,
                    language=language
                )

                logger.info(
                    f"Routing decision: intent={routing_decision.intent.value}, "
                    f"pipeline={routing_decision.pipeline.value}, "
                    f"confidence={routing_decision.confidence:.2f}"
                )

                # NOTE: Document validation disabled - always use all available documents
                # if routing_decision.user_message:
                #     # Missing required documents
                #     yield routing_decision.user_message
                #     return

                # Handle multi-intent queries (sequential execution)
                if routing_decision.is_multi_intent:
                    logger.info(
                        f"Multi-intent detected: {len(routing_decision.intents)} intents, "
                        f"executing sequentially..."
                    )

                    # Execute each pipeline in sequence
                    for idx, (intent, pipeline) in enumerate(
                        zip(routing_decision.intents, routing_decision.pipelines),
                        1
                    ):
                        if idx > 1:
                            # Add separator between pipelines
                            if language == "cs":
                                yield f"\n\n---\n\n"
                            else:
                                yield f"\n\n---\n\n"

                        logger.info(f"Executing pipeline {idx}/{len(routing_decision.intents)}: {pipeline.value}")

                        # Execute specific pipeline
                        if pipeline == PipelineRoute.SIMPLE_CHAT:
                            # Simple query pipeline
                            async for chunk in self._process_simple_query_stream(
                                query, document_ids, language
                            ):
                                yield chunk

                        elif pipeline == PipelineRoute.COMPLIANCE_ANALYSIS:
                            # Compliance analysis pipeline
                            async for chunk in self._process_compliance_stream(
                                query, contract_ids, law_ids, language, routing_decision.parameters
                            ):
                                yield chunk

                        elif pipeline == PipelineRoute.CROSS_DOCUMENT_QUERY:
                            # Cross-document query pipeline
                            async for chunk in self._process_cross_document_query_stream(
                                query, contract_ids, law_ids, language
                            ):
                                yield chunk

                        elif pipeline == PipelineRoute.GREETING_HANDLER:
                            # Greeting handler
                            if language == "cs":
                                yield "Dobrý den! Jsem váš právní asistent. "
                            else:
                                yield "Hello! I'm your legal assistant. "

                    return

                # Single intent - route to appropriate pipeline
                if routing_decision.pipeline == PipelineRoute.COMPLIANCE_ANALYSIS:
                    # Use compliance analysis pipeline
                    async for chunk in self._process_compliance_stream(
                        query, contract_ids, law_ids, language, routing_decision.parameters
                    ):
                        yield chunk
                    return

                elif routing_decision.pipeline == PipelineRoute.CROSS_DOCUMENT_QUERY:
                    # Use cross-document query pipeline
                    async for chunk in self._process_cross_document_query_stream(
                        query, contract_ids, law_ids, language
                    ):
                        yield chunk
                    return

                elif routing_decision.pipeline == PipelineRoute.GREETING_HANDLER:
                    # Handle greeting
                    if language == "cs":
                        yield "Dobrý den! Jsem váš právní asistent. Mohu vám pomoci s analýzou právních dokumentů, kontrolou souladu smluv se zákony, nebo odpovědět na vaše právní dotazy. Jak vám mohu pomoci?"
                    else:
                        yield "Hello! I'm your legal assistant. I can help you with legal document analysis, contract-law compliance checking, or answer your legal questions. How can I help you?"
                    return

                # Fall through to simple chat if not handled above
                elif routing_decision.pipeline == PipelineRoute.SIMPLE_CHAT:
                    # Use simple query pipeline
                    async for chunk in self._process_simple_query_stream(
                        query, document_ids, language
                    ):
                        yield chunk
                    return

            # Step 3: Default to simple query pipeline (or if no router available)
            # Retrieve relevant chunks via RAG pipeline
            retrieval_results = await self.rag_pipeline.query(
                query=query,
                document_ids=document_ids,
                top_k=None,  # Use config default (retrieval.top_k = 20)
                rerank=True,
                language=language
            )

            # Build context from retrieved chunks
            context = self._build_context_from_results(retrieval_results.get("results", []))

            # Build prompt for Claude
            prompt = self._build_legal_prompt(query, context, language)

            # Stream response from Claude API
            if self.client:
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
            else:
                # Fallback if no API key
                yield "CHYBA: Claude API klíč není nastaven. Zkontrolujte .env soubor."
                return

            # Yield sources after answer
            sources = []
            for result in retrieval_results.get("results", []):
                sources.append({
                    "legal_reference": result.get("metadata", {}).get("section", ""),
                    "content": result.get("content", "")[:200],
                    "document_id": result.get("document_id", ""),
                    "confidence": result.get("score", 0.0),
                    "page": result.get("metadata", {}).get("page")
                })

            if sources:
                sources_json = json.dumps({
                    "type": "sources",
                    "sources": sources
                }, ensure_ascii=False)
                yield "\n\n" + sources_json

        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"\n\nCHYBA: {str(e)}"

    async def process_query(
        self,
        query: str,
        document_ids: List[str],
        language: str = "cs",
        rerank: bool = True,
        top_k: Optional[int] = None
    ) -> dict:
        """
        Process query with full RAG pipeline (hybrid retrieval + reranking).

        Args:
            query: User query
            document_ids: List of document IDs to search
            language: Query language ('cs' or 'en')
            rerank: Whether to apply cross-encoder reranking
            top_k: Number of results to return

        Returns:
            Query response with answer and sources

        Raises:
            RetrievalError: If retrieval fails
        """
        import time
        start_time = time.time()

        try:
            logger.info(f"Processing query: {query[:100]}... for {len(document_ids)} documents")

            # Execute hybrid retrieval with RAG pipeline
            # This includes:
            # 1. Query processing and expansion
            # 2. Semantic search (FAISS + BGE-M3)
            # 3. BM25 keyword search
            # 4. Hybrid fusion (alpha=0.7)
            # 5. Cross-encoder reranking
            retrieval_results = await self.rag_pipeline.query(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                rerank=rerank,
                language=language
            )

            # Convert RAG results to QuerySource format
            sources = []
            for result in retrieval_results.get("results", []):
                sources.append(
                    QuerySource(
                        legal_reference=result.get("metadata", {}).get("section", ""),
                        content=result.get("content", ""),
                        document_id=result.get("document_id", ""),
                        confidence=result.get("score", 0.0),
                        page=result.get("metadata", {}).get("page"),
                        chunk_id=result.get("chunk_id")
                    )
                )

            # TODO: Integrate Claude API for answer synthesis
            # For now, return retrieved sources
            # In full implementation, this would:
            # 1. Build context from top sources
            # 2. Call Claude API with prompt template
            # 3. Stream or return complete answer
            answer = self._generate_answer_from_sources(query, sources, language)

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Query processed in {processing_time}ms: "
                f"{len(sources)} sources retrieved, reranked={rerank}"
            )

            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "processing_time_ms": processing_time,
                "metadata": {
                    "document_count": len(document_ids),
                    "source_count": len(sources),
                    "reranked": rerank,
                    "language": language
                }
            }

        except RetrievalError as e:
            logger.error(f"Retrieval failed: {e}")
            raise

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise

    def _generate_answer_from_sources(
        self,
        query: str,
        sources: List[QuerySource],
        language: str
    ) -> str:
        """
        Generate answer from retrieved sources.

        TODO: Replace with Claude API integration for full answer synthesis.

        Args:
            query: Original query
            sources: Retrieved sources
            language: Response language

        Returns:
            Generated answer
        """
        if not sources:
            if language == "cs":
                return "Nenalezeny žádné relevantní zdroje pro váš dotaz."
            else:
                return "No relevant sources found for your query."

        # Temporary: return summary of sources
        # In production, use Claude API with prompt template
        if language == "cs":
            answer = f"Na základě {len(sources)} relevantních zdrojů:\n\n"
        else:
            answer = f"Based on {len(sources)} relevant sources:\n\n"

        for i, source in enumerate(sources[:3], 1):
            ref = source.legal_reference or f"Zdroj {i}"
            answer += f"{i}. {ref} (skóre: {source.confidence:.2f})\n"
            answer += f"   {source.content[:200]}...\n\n"

        return answer

    def _build_context_from_results(self, results: List[dict]) -> str:
        """Build context string from retrieval results."""
        if not results:
            return ""

        context = ""
        for i, result in enumerate(results, 1):
            section = result.get("metadata", {}).get("section", f"Source {i}")
            content = result.get("content", "")
            score = result.get("score", 0.0)

            context += f"\n[{i}] {section} (relevance: {score:.2f})\n"
            context += f"{content}\n"

        return context

    async def _process_compliance_stream(
        self,
        query: str,
        contract_ids: List[str],
        law_ids: List[str],
        language: str,
        parameters: dict
    ) -> AsyncIterator[str]:
        """
        Process compliance analysis using Advanced Compliance Pipeline and stream results.

        Uses the new pipeline with:
        - Pre-filtering (skip irrelevant chunks)
        - Multi-round retrieval (2 rounds for better recall)
        - Haiku quick filter (fast screening)
        - Sonnet deep analysis (escalated cases only)

        Args:
            query: User query
            contract_ids: Contract document IDs
            law_ids: Law document IDs
            language: Response language
            parameters: Routing parameters (mode, focus, etc.)

        Yields:
            Compliance analysis results as text stream with status updates
        """
        try:
            if not contract_ids or not law_ids:
                if language == "cs":
                    yield "Pro kontrolu souladu potřebuji alespoň jednu smlouvu a jeden zákon."
                else:
                    yield "For compliance checking I need at least one contract and one law."
                return

            # Get chunks from indexed documents
            contract_chunks = []
            for contract_id in contract_ids:
                chunks = await self.rag_pipeline.vector_store.get_document_chunks(contract_id)
                contract_chunks.extend(chunks)

            if not contract_chunks:
                if language == "cs":
                    yield "CHYBA: Nenalezeny žádné chunky pro smlouvy.\n"
                else:
                    yield "ERROR: No chunks found for contracts.\n"
                return

            logger.info(f"Retrieved {len(contract_chunks)} contract chunks for compliance analysis")

            # Initialize Advanced Compliance Pipeline
            from app.rag.advanced_compliance_pipeline import AdvancedCompliancePipeline

            pipeline_config = {
                "pre_filter": {
                    "min_chunk_size": 50,
                    "max_chunk_size": 5000
                },
                "multi_round": {
                    "num_rounds": 2,
                    "final_top_k": 5
                },
                "haiku": {
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "high_confidence_threshold": 0.85,
                    "low_confidence_threshold": 0.60
                },
                "sonnet": {
                    "max_tokens": 3000,
                    "temperature": 0.1
                }
            }

            pipeline = AdvancedCompliancePipeline(
                llm_client=self.client,
                comparative_retriever=self.rag_pipeline.cross_doc_retriever,
                config=pipeline_config
            )

            # Stream results from advanced pipeline
            # The pipeline's analyze_compliance_stream() method already emits
            # status updates with __STATUS__ markers
            # Supports multiple laws - will analyze against ALL provided laws
            async for chunk in pipeline.analyze_compliance_stream(
                contract_chunks=contract_chunks,
                law_ids=law_ids,  # Analyze against ALL laws
                contract_id=contract_ids[0],
                language=language
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Compliance analysis failed: {e}", exc_info=True)
            if language == "cs":
                yield f"\n\nCHYBA při analýze souladu: {str(e)}"
            else:
                yield f"\n\nERROR in compliance analysis: {str(e)}"

    async def _process_simple_query_stream(
        self,
        query: str,
        document_ids: List[str],
        language: str
    ) -> AsyncIterator[str]:
        """
        Process simple query and stream response.

        Uses RAG pipeline to retrieve relevant chunks and Claude API to generate answer.

        Args:
            query: User query
            document_ids: List of document IDs to search
            language: Response language

        Yields:
            Response chunks and sources as JSON
        """
        try:
            # Step 1: Send pipeline status - Retrieving
            if language == "cs":
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "simple_query",
                    "stage": "retrieval",
                    "stage_name": "Retrieval",
                    "step": 1,
                    "total_steps": 3,
                    "progress": 33,
                    "message": "🔍 Vyhledávám relevantní dokumenty..."
                })
            else:
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "simple_query",
                    "stage": "retrieval",
                    "stage_name": "Retrieval",
                    "step": 1,
                    "total_steps": 3,
                    "progress": 33,
                    "message": "🔍 Retrieving relevant documents..."
                })
            yield f"__STATUS__{status_msg}__STATUS__\n"

            # Retrieve relevant chunks via RAG pipeline
            retrieval_results = await self.rag_pipeline.query(
                query=query,
                document_ids=document_ids,
                top_k=None,  # Use config default (retrieval.top_k = 20)
                rerank=True,
                language=language
            )

            # Build context from retrieved chunks
            context = self._build_context_from_results(retrieval_results.get("results", []))

            # Build prompt for Claude
            prompt = self._build_legal_prompt(query, context, language)

            # Step 2: Send pipeline status - Generating
            if language == "cs":
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "simple_query",
                    "stage": "generation",
                    "stage_name": "Generation",
                    "step": 2,
                    "total_steps": 3,
                    "progress": 66,
                    "message": "✨ Generuji odpověď..."
                })
            else:
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "simple_query",
                    "stage": "generation",
                    "stage_name": "Generation",
                    "step": 2,
                    "total_steps": 3,
                    "progress": 66,
                    "message": "✨ Generating answer..."
                })
            yield f"__STATUS__{status_msg}__STATUS__\n"

            # Stream response from Claude API
            if self.client:
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
            else:
                # Fallback if no API key
                if language == "cs":
                    yield "CHYBA: Claude API klíč není nastaven. Zkontrolujte .env soubor."
                else:
                    yield "ERROR: Claude API key is not set. Check .env file."
                return

            # Step 3: Send pipeline status - Finalizing
            if language == "cs":
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "simple_query",
                    "stage": "finalization",
                    "stage_name": "Finalization",
                    "step": 3,
                    "total_steps": 3,
                    "progress": 100,
                    "message": "✅ Dokončuji..."
                })
            else:
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "simple_query",
                    "stage": "finalization",
                    "stage_name": "Finalization",
                    "step": 3,
                    "total_steps": 3,
                    "progress": 100,
                    "message": "✅ Finalizing..."
                })
            yield f"__STATUS__{status_msg}__STATUS__\n"

            # Yield sources after answer
            sources = []
            for result in retrieval_results.get("results", []):
                sources.append({
                    "legal_reference": result.get("metadata", {}).get("section", ""),
                    "content": result.get("content", "")[:200],
                    "document_id": result.get("document_id", ""),
                    "confidence": result.get("score", 0.0),
                    "page": result.get("metadata", {}).get("page")
                })

            if sources:
                sources_json = json.dumps({
                    "type": "sources",
                    "sources": sources
                }, ensure_ascii=False)
                yield "\n\n" + sources_json

        except Exception as e:
            logger.error(f"Simple query processing failed: {e}", exc_info=True)
            if language == "cs":
                yield f"\n\nCHYBA: {str(e)}"
            else:
                yield f"\n\nERROR: {str(e)}"

    async def _process_cross_document_query_stream(
        self,
        query: str,
        contract_ids: List[str],
        law_ids: List[str],
        language: str
    ) -> AsyncIterator[str]:
        """
        Process cross-document query using cross-document retrieval.

        Matches clauses from contracts with provisions from laws using:
        - Explicit reference matching (§ citations)
        - Semantic similarity matching
        - Structural matching

        Args:
            query: User query
            contract_ids: Contract document IDs
            law_ids: Law document IDs
            language: Response language

        Yields:
            Response chunks with cross-document matches
        """
        try:
            if not contract_ids and not law_ids:
                if language == "cs":
                    yield "Pro srovnání dokumentů potřebuji alespoň dva dokumenty."
                else:
                    yield "For document comparison I need at least two documents."
                return

            # Step 1: Send pipeline status - Starting cross-document analysis
            if language == "cs":
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "cross_document",
                    "stage": "initialization",
                    "stage_name": "Initialization",
                    "step": 1,
                    "total_steps": 3,
                    "progress": 33,
                    "message": "🔍 Zahajuji mezidokumentovou analýzu..."
                })
            else:
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "cross_document",
                    "stage": "initialization",
                    "stage_name": "Initialization",
                    "step": 1,
                    "total_steps": 3,
                    "progress": 33,
                    "message": "🔍 Starting cross-document analysis..."
                })
            yield f"__STATUS__{status_msg}__STATUS__\n"

            # Step 2: Send pipeline status - Matching documents
            if language == "cs":
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "cross_document",
                    "stage": "matching",
                    "stage_name": "Matching",
                    "step": 2,
                    "total_steps": 3,
                    "progress": 66,
                    "message": "🔗 Hledám souvislosti mezi dokumenty..."
                })
            else:
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "cross_document",
                    "stage": "matching",
                    "stage_name": "Matching",
                    "step": 2,
                    "total_steps": 3,
                    "progress": 66,
                    "message": "🔗 Finding relationships between documents..."
                })
            yield f"__STATUS__{status_msg}__STATUS__\n"

            # Use cross-document retrieval from RAG pipeline
            cross_doc_retriever = self.rag_pipeline.cross_doc_retriever

            if contract_ids and law_ids:
                # Contract-Law comparison
                matches = cross_doc_retriever.find_related_clauses(
                    source_doc_id=contract_ids[0],
                    target_doc_ids=law_ids,
                    top_k=10,
                    similarity_threshold=0.7
                )

                if not matches:
                    if language == "cs":
                        yield "❌ Nenalezeny žádné příbuzné klauzule mezi dokumenty.\n"
                    else:
                        yield "❌ No related clauses found between documents.\n"
                    return

                # Build context from matches
                context = ""
                if language == "cs":
                    context += "## Nalezené shody mezi smlouvou a zákonem:\n\n"
                else:
                    context += "## Found matches between contract and law:\n\n"

                for i, match in enumerate(matches[:5], 1):
                    source_section = match.get("source_section", f"Zdroj {i}")
                    target_section = match.get("target_section", "")
                    source_content = match.get("source_content", "")
                    target_content = match.get("target_content", "")
                    similarity = match.get("similarity_score", 0.0)
                    match_type = match.get("match_type", "semantic")

                    if language == "cs":
                        context += f"### Shoda {i} (relevance: {similarity:.2f}, typ: {match_type})\n\n"
                        context += f"**Smlouva ({source_section}):**\n{source_content}\n\n"
                        context += f"**Zákon ({target_section}):**\n{target_content}\n\n"
                    else:
                        context += f"### Match {i} (relevance: {similarity:.2f}, type: {match_type})\n\n"
                        context += f"**Contract ({source_section}):**\n{source_content}\n\n"
                        context += f"**Law ({target_section}):**\n{target_content}\n\n"

            else:
                # Generic multi-document comparison
                # Use hybrid retrieval on all documents
                all_doc_ids = contract_ids + law_ids
                retrieval_results = await self.rag_pipeline.query(
                    query=query,
                    document_ids=all_doc_ids,
                    top_k=10,
                    rerank=True,
                    language=language
                )

                context = self._build_context_from_results(retrieval_results.get("results", []))

            # Step 3: Send pipeline status - Generating analysis
            if language == "cs":
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "cross_document",
                    "stage": "generation",
                    "stage_name": "Generation",
                    "step": 3,
                    "total_steps": 3,
                    "progress": 100,
                    "message": "✨ Generuji analýzu..."
                })
            else:
                status_msg = json.dumps({
                    "type": "pipeline_status",
                    "pipeline": "cross_document",
                    "stage": "generation",
                    "stage_name": "Generation",
                    "step": 3,
                    "total_steps": 3,
                    "progress": 100,
                    "message": "✨ Generating analysis..."
                })
            yield f"__STATUS__{status_msg}__STATUS__\n"

            # Build prompt for cross-document analysis
            if language == "cs":
                prompt = f"""Jsi právní expert na srovnávání právních dokumentů.

<dotaz_uživatele>
{query}
</dotaz_uživatele>

<dokumenty_a_shody>
{context}
</dokumenty_a_shody>

<instrukce>
1. Analyzuj vztahy a shody mezi dokumenty
2. Upozorni na důležité podobnosti a rozdíly
3. Pokud jde o smlouvu a zákon, upozorni na:
   - Klauzule které odpovídají zákonným požadavkům
   - Klauzule které se odchylují od zákona
   - Chybějící požadavky
4. Odpověz přesně na dotaz uživatele
5. Používej konkrétní citace z dokumentů
</instrukce>

Odpověď:"""
            else:
                prompt = f"""You are a legal expert in comparing legal documents.

<user_query>
{query}
</user_query>

<documents_and_matches>
{context}
</documents_and_matches>

<instructions>
1. Analyze relationships and matches between documents
2. Point out important similarities and differences
3. If comparing contract with law, highlight:
   - Clauses that meet legal requirements
   - Clauses that deviate from law
   - Missing requirements
4. Answer the user's query precisely
5. Use specific citations from documents
</instructions>

Answer:"""

            # Stream response from Claude API
            if self.client:
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
            else:
                if language == "cs":
                    yield "CHYBA: Claude API klíč není nastaven."
                else:
                    yield "ERROR: Claude API key is not set."

        except Exception as e:
            logger.error(f"Cross-document query failed: {e}", exc_info=True)
            if language == "cs":
                yield f"\n\nCHYBA při mezidokumentové analýze: {str(e)}"
            else:
                yield f"\n\nERROR in cross-document analysis: {str(e)}"

    def _build_legal_prompt(self, query: str, context: str, language: str) -> str:
        """Build adaptive prompt for Claude API with legal context and tone matching."""
        if language == "cs":
            return f"""Jsi užitečný právní asistent specializující se na český právní systém.

<question>
{query}
</question>

<legal_documents>
{context if context else "Žádné relevantní dokumenty nebyly nalezeny."}
</legal_documents>

<instructions>
DŮLEŽITÉ - Přizpůsob se typu dotazu:

1. **Pokud je dotaz neformální nebo obecný** (jako "ahoj", "dobrý den", "jak se máš"):
   - Odpověz přirozeně a přátelsky
   - Představ se jako právní asistent
   - Nabídni pomoc s právními otázkami
   - Nevynucuj právní jazyk tam, kde to nedává smysl

2. **Pokud je dotaz právní**:
   - Odpověz přesně a konkrétně na základě dokumentů
   - Cituj konkrétní paragrafy a ustanovení ze zdrojů
   - Používej profesionální právní jazyk
   - Strukturuj odpověď přehledně

3. **Pokud informace není v dokumentech**:
   - Jasně uveď, že odpověď není v poskytnutých dokumentech
   - Nenabízej informace mimo poskytnuté zdroje
   - Doporuč upřesnit dotaz nebo nahrát relevantní dokumenty

4. **Vždy používej český jazyk a přiměřený tón** odpovídající uživatelskému dotazu
</instructions>

Odpověď:"""
        else:
            return f"""You are a helpful legal assistant specializing in Czech law.

<question>
{query}
</question>

<legal_documents>
{context if context else "No relevant documents found."}
</legal_documents>

<instructions>
IMPORTANT - Adapt to the query type:

1. **If the query is informal or general** (like "hello", "hi", "how are you"):
   - Respond naturally and friendly
   - Introduce yourself as a legal assistant
   - Offer help with legal questions
   - Don't force legal language where it doesn't make sense

2. **If the query is legal**:
   - Answer precisely and specifically based on documents
   - Cite specific paragraphs and provisions from sources
   - Use professional legal language
   - Structure your answer clearly

3. **If information is not in the documents**:
   - Clearly state that the answer is not in the provided documents
   - Don't offer information outside the provided sources
   - Recommend clarifying the query or uploading relevant documents

4. **Always use appropriate tone** matching the user's query
</instructions>

Answer:"""
