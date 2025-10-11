"""Service for chat/query operations with streaming support."""
from typing import AsyncIterator, List, Optional
import json
import asyncio
import logging

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.models.query import QuerySource
from app.services.rag_pipeline import get_rag_pipeline
from app.rag.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class ChatService:
    """Service for chat and query operations with RAG pipeline integration."""

    def __init__(self):
        """Initialize chat service with RAG pipeline."""
        # Use Haiku model for faster, cheaper responses
        self.model = "claude-3-5-haiku-20241022"
        self.rag_pipeline = get_rag_pipeline()
        self.client = AsyncAnthropic(api_key=settings.CLAUDE_API_KEY) if settings.CLAUDE_API_KEY else None

    async def process_query_stream(
        self,
        query: str,
        document_ids: List[str],
        language: str = "cs"
    ) -> AsyncIterator[str]:
        """
        Process query and stream response chunks with real Claude API.

        Args:
            query: User query
            document_ids: List of document IDs to search
            language: Response language (cs/en)

        Yields:
            Response chunks as they arrive
        """
        logger.info(f"Streaming query: {query[:100]}... for {len(document_ids)} documents")

        try:
            # 1. Retrieve relevant chunks via RAG pipeline
            retrieval_results = await self.rag_pipeline.query(
                query=query,
                document_ids=document_ids,
                top_k=5,
                rerank=True,
                language=language
            )

            # 2. Build context from retrieved chunks
            context = self._build_context_from_results(retrieval_results.get("results", []))

            # 3. Build prompt for Claude
            prompt = self._build_legal_prompt(query, context, language)

            # 4. Stream response from Claude API
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

            # 5. Yield sources after answer
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
