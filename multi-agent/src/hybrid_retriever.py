"""
Hybrid Retrieval System for Document Analyzer
Combines multiple retrieval strategies for optimal performance
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

from vector_store_faiss import FAISSVectorStore as VectorStore, SearchResult, create_vector_store
from indexing_pipeline import DocumentChunker, ChunkingStrategy

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval"""
    hybrid_alpha: float = 0.7  # Weight for semantic search (0.7 = 70% semantic, 30% keyword)
    top_k: int = 20
    rerank_top_k: int = 5
    min_score: float = 0.3
    enable_reranking: bool = True
    enable_query_decomposition: bool = True
    max_sub_queries: int = 5
    enable_query_expansion: bool = True


@dataclass
class QueryDecomposition:
    """Decomposed query for multi-aspect search"""
    original_query: str
    sub_queries: List[str]
    query_type: str  # factual, analytical, comparison, temporal
    keywords: List[str]
    entities: List[str]


class KeywordExtractor:
    """Extract keywords and entities from queries"""

    def __init__(self):
        self.stop_words = set([
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'what', 'how', 'when', 'where', 'why', 'who'
        ])

    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove punctuation and convert to lowercase
        clean_query = re.sub(r'[^\w\s]', '', query.lower())

        # Split into words and filter
        words = clean_query.split()
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]

        # Find multi-word entities (capitalized sequences in original)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, query)

        return list(set(keywords + [e.lower() for e in entities]))

    def extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        # Simple pattern-based entity extraction
        # Can be enhanced with spaCy or other NER models

        patterns = {
            'dates': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b',
            'numbers': r'\b\d+(?:\.\d+)?%?\b',
            'capitalized': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'quoted': r'"([^"]*)"',
        }

        entities = []
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, query)
            entities.extend(matches)

        return list(set(entities))


class QueryDecomposer:
    """Decompose complex queries into sub-queries"""

    def __init__(self):
        self.keyword_extractor = KeywordExtractor()

    def decompose(self, query: str) -> QueryDecomposition:
        """Decompose query into components"""

        # Determine query type
        query_type = self._classify_query(query)

        # Extract components
        keywords = self.keyword_extractor.extract_keywords(query)
        entities = self.keyword_extractor.extract_entities(query)

        # Generate sub-queries based on type
        sub_queries = self._generate_sub_queries(query, query_type, keywords, entities)

        return QueryDecomposition(
            original_query=query,
            sub_queries=sub_queries,
            query_type=query_type,
            keywords=keywords,
            entities=entities
        )

    def _classify_query(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(word in query_lower for word in ['when', 'timeline', 'history', 'evolution']):
            return 'temporal'
        elif any(word in query_lower for word in ['analyze', 'evaluate', 'assess', 'impact']):
            return 'analytical'
        elif any(word in query_lower for word in ['list', 'enumerate', 'what are', 'types of']):
            return 'enumerative'
        else:
            return 'factual'

    def _generate_sub_queries(self, query: str, query_type: str,
                            keywords: List[str], entities: List[str]) -> List[str]:
        """Generate sub-queries based on query type"""
        sub_queries = [query]  # Always include original

        if query_type == 'comparison':
            # Split comparison queries
            comparison_terms = ['compare', 'versus', 'vs', 'difference between']
            for term in comparison_terms:
                if term in query.lower():
                    parts = re.split(term, query, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        sub_queries.append(f"What is {parts[0].strip()}?")
                        sub_queries.append(f"What is {parts[1].strip()}?")

        elif query_type == 'temporal':
            # Add time-focused variations
            for entity in entities:
                sub_queries.append(f"Timeline of {entity}")
                sub_queries.append(f"History of {entity}")

        elif query_type == 'analytical':
            # Break down analytical questions
            if 'impact' in query.lower():
                for entity in entities[:2]:  # Limit to avoid explosion
                    sub_queries.append(f"What is {entity}?")
                    sub_queries.append(f"Effects of {entity}")

        elif query_type == 'enumerative':
            # Focus on listing and categorization
            main_topic = ' '.join(keywords[:3])
            sub_queries.append(f"Types of {main_topic}")
            sub_queries.append(f"Categories of {main_topic}")

        # Limit number of sub-queries
        return sub_queries[:5]


class QueryExpander:
    """Expand queries with synonyms and related terms"""

    def __init__(self):
        # Simple synonym dictionary - can be enhanced with WordNet
        self.synonyms = {
            'analyze': ['examine', 'evaluate', 'assess', 'study'],
            'create': ['make', 'build', 'develop', 'generate'],
            'improve': ['enhance', 'optimize', 'upgrade', 'refine'],
            'problem': ['issue', 'challenge', 'difficulty', 'obstacle'],
            'solution': ['answer', 'resolution', 'fix', 'remedy'],
            'method': ['approach', 'technique', 'strategy', 'procedure'],
            'result': ['outcome', 'consequence', 'effect', 'finding'],
            'important': ['significant', 'critical', 'essential', 'crucial'],
        }

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms"""
        expanded_terms = []
        words = query.lower().split()

        for word in words:
            expanded_terms.append(word)
            # Add synonyms if available
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word][:2])  # Limit synonyms

        # Create expanded query without duplicates
        expanded = ' '.join(list(dict.fromkeys(expanded_terms)))
        return expanded


class BM25Searcher:
    """BM25-based keyword search"""

    def __init__(self, documents: List[str] = None):
        self.documents = documents or []
        self.tokenizer = None
        self.bm25 = None
        if documents:
            self._build_index(documents)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()

    def _build_index(self, documents: List[str]):
        """Build BM25 index"""
        if not documents:
            logger.warning("No documents to index for BM25")
            self.bm25 = None
            return

        tokenized_docs = [self._tokenize(doc) for doc in documents]
        # Filter out empty tokenized docs to prevent division by zero
        tokenized_docs = [doc for doc in tokenized_docs if doc]

        if not tokenized_docs:
            logger.warning("All documents resulted in empty tokens for BM25")
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"Built BM25 index with {len(tokenized_docs)} documents")

    def update_index(self, documents: List[str]):
        """Update BM25 index with new documents"""
        self.documents = documents
        self._build_index(documents)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using BM25"""
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices with scores
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

        return results


class HybridRetriever:
    """Main hybrid retrieval system"""

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize retrieval config
        retrieval_config = self.config.get('retrieval', {})
        self.retrieval_config = RetrievalConfig(
            hybrid_alpha=retrieval_config.get('hybrid_alpha', 0.7),
            top_k=retrieval_config.get('top_k', 20),
            rerank_top_k=retrieval_config.get('rerank_top_k', 5),
            min_score=retrieval_config.get('min_score', 0.3),
            enable_reranking=retrieval_config.get('enable_reranking', True),
            enable_query_decomposition=retrieval_config.get('enable_query_decomposition', True),
            max_sub_queries=retrieval_config.get('max_sub_queries', 5),
            enable_query_expansion=retrieval_config.get('enable_query_expansion', True)
        )

        # Initialize components
        self.vector_store = create_vector_store(self.config)
        self.bm25_searcher = BM25Searcher()
        self.query_decomposer = QueryDecomposer()
        self.query_expander = QueryExpander()

        # Initialize reranker if enabled
        if self.retrieval_config.enable_reranking:
            reranker_model = retrieval_config.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info(f"Loading reranker model: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
        else:
            self.reranker = None

        # Cache for document chunks
        self.document_chunks = []
        self.chunk_texts = []

    async def index_documents(self, chunks: List[Any]):
        """Index document chunks for retrieval (both vector + BM25)"""
        # Store chunks for BM25
        self.document_chunks = chunks
        self.chunk_texts = [chunk.content for chunk in chunks]

        # Build BM25 index
        self.bm25_searcher.update_index(self.chunk_texts)

        # Add to vector store
        await self.vector_store.add_documents(chunks)

        logger.info(f"Indexed {len(chunks)} chunks for hybrid retrieval")

    async def index_documents_bm25_only(self, chunks: List[Any]):
        """Index document chunks ONLY in BM25 (vector store already done by pipeline)"""
        # Store chunks for BM25
        self.document_chunks = chunks
        self.chunk_texts = [chunk.content for chunk in chunks]

        # Build BM25 index only
        self.bm25_searcher.update_index(self.chunk_texts)

        logger.info(f"Indexed {len(chunks)} chunks in BM25 (vector store skipped - already indexed)")

    async def retrieve(self, query: str, filters: Dict = None) -> List[SearchResult]:
        """Perform hybrid retrieval"""

        # Query decomposition if enabled
        if self.retrieval_config.enable_query_decomposition:
            decomposition = self.query_decomposer.decompose(query)
            queries = decomposition.sub_queries[:self.retrieval_config.max_sub_queries]
            logger.info(f"Decomposed into {len(queries)} sub-queries")
        else:
            queries = [query]

        # Query expansion if enabled
        if self.retrieval_config.enable_query_expansion:
            expanded_queries = [self.query_expander.expand_query(q) for q in queries]
        else:
            expanded_queries = queries

        # Collect results from all queries
        all_results = []

        for expanded_query in expanded_queries:
            # Semantic search
            semantic_results = await self._semantic_search(expanded_query, filters)

            # Keyword search
            keyword_results = self._keyword_search(expanded_query)

            # Combine results
            combined = self._combine_results(semantic_results, keyword_results)
            all_results.extend(combined)

        # Deduplicate by chunk_id
        unique_results = {}
        for result in all_results:
            if result.chunk_id not in unique_results or result.score > unique_results[result.chunk_id].score:
                unique_results[result.chunk_id] = result

        results = list(unique_results.values())

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        # Take top-k
        results = results[:self.retrieval_config.top_k]

        # Rerank if enabled
        if self.retrieval_config.enable_reranking and self.reranker and results:
            results = await self._rerank_results(query, results)
            # After reranking, scores are from cross-encoder (can be negative)
            # Don't apply min_score filter to reranked results - reranker already selected best ones
        else:
            # Only filter by minimum score if not reranking
            results = [r for r in results if r.score >= self.retrieval_config.min_score]

        return results

    async def _semantic_search(self, query: str, filters: Dict = None) -> List[SearchResult]:
        """Perform semantic search using vector store"""
        return await self.vector_store.search(
            query=query,
            top_k=self.retrieval_config.top_k,
            filters=filters
        )

    def _keyword_search(self, query: str) -> List[SearchResult]:
        """Perform keyword search using BM25"""
        if not self.bm25_searcher.bm25:
            return []

        bm25_results = self.bm25_searcher.search(query, self.retrieval_config.top_k)

        # Convert to SearchResult format
        results = []
        for idx, score in bm25_results:
            if idx < len(self.document_chunks):
                chunk = self.document_chunks[idx]
                results.append(SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=score / 100.0,  # Normalize BM25 scores
                    metadata=chunk.metadata
                ))

        return results

    def _combine_results(self, semantic_results: List[SearchResult],
                        keyword_results: List[SearchResult]) -> List[SearchResult]:
        """Combine semantic and keyword search results"""
        alpha = self.retrieval_config.hybrid_alpha

        # Create score dictionaries
        semantic_scores = {r.chunk_id: r.score for r in semantic_results}
        keyword_scores = {r.chunk_id: r.score for r in keyword_results}

        # Get all unique chunk IDs
        all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())

        # Combine scores
        combined_results = []
        for chunk_id in all_chunk_ids:
            sem_score = semantic_scores.get(chunk_id, 0)
            key_score = keyword_scores.get(chunk_id, 0)

            # Weighted combination
            combined_score = alpha * sem_score + (1 - alpha) * key_score

            # Get the result object (prefer semantic for metadata)
            result = None
            for r in semantic_results:
                if r.chunk_id == chunk_id:
                    result = r
                    break
            if not result:
                for r in keyword_results:
                    if r.chunk_id == chunk_id:
                        result = r
                        break

            if result:
                # Create new result with combined score
                combined_results.append(SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=combined_score,
                    metadata=result.metadata
                ))

        return combined_results

    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using cross-encoder"""
        if not results:
            return results

        # Prepare query-document pairs
        pairs = [[query, r.content] for r in results]

        # Get reranking scores
        logger.info(f"Reranking {len(results)} results")
        rerank_scores = await asyncio.to_thread(self.reranker.predict, pairs)

        # Normalize reranker scores (cross-encoder returns logits, can be negative)
        # Use min-max normalization to scale to 0-1 range
        import numpy as np
        rerank_scores_array = np.array(rerank_scores, dtype=np.float32)

        # Min-max normalization
        min_score = float(rerank_scores_array.min())
        max_score = float(rerank_scores_array.max())

        if max_score - min_score > 1e-6:  # Avoid division by zero
            normalized_scores = (rerank_scores_array - min_score) / (max_score - min_score)
        else:
            # All scores are the same, assign 0.5
            normalized_scores = np.full_like(rerank_scores_array, 0.5)

        # Update scores and sort
        for i, result in enumerate(results):
            result.score = float(normalized_scores[i])

        results.sort(key=lambda x: x.score, reverse=True)

        # Return top-k after reranking
        return results[:self.retrieval_config.rerank_top_k]

    async def get_relevant_context(self, query: str, max_tokens: int = 4000) -> str:
        """Get relevant context for a query within token limit"""
        results = await self.retrieve(query)

        if not results:
            return ""

        # Initialize tokenizer
        encoding = tiktoken.get_encoding("cl100k_base")

        context_parts = []
        total_tokens = 0

        for result in results:
            # Count tokens in this chunk
            chunk_tokens = len(encoding.encode(result.content))

            # Check if adding this chunk exceeds limit
            if total_tokens + chunk_tokens > max_tokens:
                # Try to add partial chunk
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only add if meaningful
                    truncated = encoding.decode(encoding.encode(result.content)[:remaining_tokens])
                    context_parts.append(truncated)
                break

            context_parts.append(result.content)
            total_tokens += chunk_tokens

        return "\n\n".join(context_parts)

    async def search_with_feedback(self, query: str, relevance_feedback: List[str] = None) -> List[SearchResult]:
        """Search with relevance feedback for query refinement"""

        # Initial search
        results = await self.retrieve(query)

        if not relevance_feedback:
            return results

        # Pseudo-relevance feedback: use top results to expand query
        # Rocchio algorithm simplified
        relevant_docs = relevance_feedback[:3]  # Use top relevant docs

        # Extract keywords from relevant documents
        keyword_extractor = KeywordExtractor()
        expansion_terms = []

        for doc in relevant_docs:
            keywords = keyword_extractor.extract_keywords(doc)
            expansion_terms.extend(keywords[:5])  # Top 5 keywords per doc

        # Create expanded query
        expanded_query = f"{query} {' '.join(set(expansion_terms))}"

        # Re-search with expanded query
        logger.info(f"Searching with feedback-expanded query")
        return await self.retrieve(expanded_query)


# CLI interface
async def main():
    """CLI for hybrid retriever"""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Document Retrieval")
    parser.add_argument("action", choices=["search", "test"],
                       help="Action to perform")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")

    args = parser.parse_args()

    # Initialize retriever
    retriever = HybridRetriever(args.config)

    if args.action == "search":
        if not args.query:
            print("Error: --query required for search")
            return

        results = await retriever.retrieve(args.query)

        print(f"\nFound {len(results)} results for: {args.query}\n")
        for i, result in enumerate(results[:args.top_k], 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   Content: {result.content[:200]}...")
            print(f"   Metadata: {result.metadata}\n")

    elif args.action == "test":
        # Test with sample data
        from .indexing_pipeline import DocumentChunk

        test_chunks = [
            DocumentChunk(
                chunk_id="1",
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"doc": "test1"}
            ),
            DocumentChunk(
                chunk_id="2",
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"doc": "test1"}
            ),
            DocumentChunk(
                chunk_id="3",
                content="Natural language processing enables computers to understand text.",
                metadata={"doc": "test2"}
            )
        ]

        await retriever.index_documents(test_chunks)

        test_query = "What is machine learning and AI?"
        results = await retriever.retrieve(test_query)

        print(f"Test query: {test_query}")
        print(f"Results: {len(results)} found")
        for result in results:
            print(f"  - Score: {result.score:.4f} | {result.content[:50]}...")


if __name__ == "__main__":
    asyncio.run(main())