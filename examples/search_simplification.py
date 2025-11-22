"""
Example Simplifications for search.py

This file demonstrates the proposed simplifications for the SearchTool class.
These examples preserve exact functionality while improving readability.
"""


class SearchToolSimplified:
    """Example of simplified SearchTool methods."""

    def _build_method_name(self, num_expands: int, use_graph_boost: bool) -> str:
        """
        Build method name from active features.

        This replaces the complex nested ternary operator with a clear,
        sequential approach that's easier to understand and maintain.

        Args:
            num_expands: Number of query expansions used
            use_graph_boost: Whether graph boosting was applied

        Returns:
            Method name string (e.g., "hybrid+expansion+graph+rerank")
        """
        components = ["hybrid"]

        if num_expands > 1:
            components.append("expansion")

        if use_graph_boost:
            components.append("graph")

        if self.reranker:
            components.append("rerank")

        return "+".join(components)

    def _search_with_hyde(
        self, query: str, hyde_docs: list, candidates_k: int, use_graph_boost: bool
    ) -> list:
        """
        Execute HyDE-enhanced search.

        This method consolidates the duplicate HyDE search logic that appears
        in both the graph boost path and standard path, eliminating ~60 lines
        of duplicate code.

        Args:
            query: Original search query
            hyde_docs: List of hypothetical documents
            candidates_k: Number of candidates to retrieve
            use_graph_boost: Whether to use graph-enhanced retrieval

        Returns:
            List of chunks with HyDE tags
        """
        chunks = []

        for hyde_idx, hyde_doc in enumerate(hyde_docs, 1):
            # Embed the hypothetical document
            hyde_embedding = self.embedder.embed_texts([hyde_doc])[0]

            # Choose search method based on graph boost availability
            if use_graph_boost and self.graph_retriever:
                hyde_chunks = self._search_with_graph_boost(
                    query, hyde_embedding, candidates_k
                )
            else:
                hyde_chunks = self._search_standard(
                    hyde_embedding, candidates_k
                )

            # Tag chunks with HyDE metadata
            for chunk in hyde_chunks:
                chunk["_source_query"] = f"{query} (HyDE {hyde_idx})"
                chunk["_hyde_doc_index"] = hyde_idx

            chunks.extend(hyde_chunks)
            self._log_hyde_results(hyde_idx, len(hyde_docs), len(hyde_chunks))

        return chunks

    def _search_with_graph_boost(
        self, query: str, query_embedding, candidates_k: int
    ) -> list:
        """Execute graph-enhanced search with fallback."""
        try:
            results = self.graph_retriever.search(
                query=query,
                query_embedding=query_embedding,
                k=candidates_k,
                enable_graph_boost=True,
            )
            chunks = results.get("layer3", [])

            if not chunks:
                # Fallback to standard search
                return self._search_standard(query_embedding, candidates_k)

            return chunks

        except Exception as e:
            logger.warning(f"Graph search failed: {e}, using standard search")
            return self._search_standard(query_embedding, candidates_k)

    def _search_standard(self, query_embedding, candidates_k: int) -> list:
        """Execute standard hierarchical search."""
        results = self.vector_store.hierarchical_search(
            query_embedding=query_embedding,
            k_layer3=candidates_k,
        )
        return results["layer3"]

    def _log_hyde_results(self, hyde_idx: int, total_docs: int, chunks_retrieved: int):
        """Log HyDE search results."""
        logger.info(
            f"HyDE doc {hyde_idx}/{total_docs} retrieved {chunks_retrieved} chunks"
        )

    def _process_search_results(
        self, query: str, candidates_k: int, use_graph_boost: bool, hyde_docs: list
    ) -> tuple:
        """
        Process search with optional HyDE and graph boost.

        This method replaces the complex nested if/else blocks with a cleaner
        flow that handles HyDE and standard searches uniformly.

        Returns:
            Tuple of (chunks, search_metadata)
        """
        all_chunks = []
        search_metadata = []

        # Process HyDE searches if available
        if hyde_docs:
            hyde_chunks = self._search_with_hyde(
                query, hyde_docs, candidates_k, use_graph_boost
            )
            all_chunks.extend(hyde_chunks)

        # Always search with original query for comparison
        query_embedding = self.embedder.embed_texts([query])[0]

        if use_graph_boost:
            chunks = self._search_with_graph_boost(
                query, query_embedding, candidates_k
            )
        else:
            chunks = self._search_standard(query_embedding, candidates_k)

        # Tag chunks with source query
        for chunk in chunks:
            if "_source_query" not in chunk:
                chunk["_source_query"] = query

        all_chunks.extend(chunks)

        # Track metadata
        search_metadata.append({
            "query": query,
            "chunks_retrieved": len(chunks),
            "graph_boost_applied": use_graph_boost,
        })

        return all_chunks, search_metadata


# Example of how to use the simplified method in actual metadata building:
def build_metadata_simplified(self, **kwargs):
    """Example of simplified metadata building."""
    # Instead of nested ternary:
    # "method": (
    #     "hybrid+expansion+graph+rerank"
    #     if num_expands > 1 and use_graph_boost and self.reranker
    #     else "hybrid+graph+rerank"
    #     if use_graph_boost and self.reranker
    #     else ...
    # )

    # Use the clear method:
    return {
        "query": kwargs["query"],
        "k": kwargs["k"],
        "method": self._build_method_name(
            kwargs["num_expands"],
            kwargs["use_graph_boost"]
        ),
        # ... other metadata fields
    }