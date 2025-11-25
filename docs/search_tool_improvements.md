# Search Tool Improvement Proposals

Based on the analysis of `src/agent/tools/search.py`, `src/agent/rag_confidence.py`, and the benchmark results (`bm25_only_hyde_exp0_k100.json`), I propose the following improvements to the search capabilities.

## 1. Fix HyDE Integration in `bm25_only` Mode
**Observation:** The benchmark configuration showed `use_hyde=True` with `search_method="bm25_only"`, but the code for `_execute_bm25_only` does not utilize the generated hypothetical documents. HyDE is currently only used in `dense_only` and `hybrid` modes (via `_get_query_embedding`).
**Proposal:**
- **Option A:** Explicitly log a warning that HyDE is ignored in `bm25_only` mode.
- **Option B (Enhancement):** Use the generated HyDE document to extract keywords for BM25 expansion (similar to Query Expansion but using the hypothetical answer's terms).

## 2. Adaptive "Auto" Search Strategy
**Observation:** The tool relies on the caller to specify `num_expands`, `use_hyde`, `enable_graph_boost`, etc. Agents often default to "safe" but suboptimal settings (e.g., `k=10`, no expansion).
**Proposal:**
- Implement an `auto` mode for `search_method` and other parameters.
- Use a lightweight heuristic or LLM call to classify the query:
    - **Entity-centric?** (e.g., "Who is the CEO of X?") → Enable `graph_boost`.
    - **Ambiguous/Abstract?** (e.g., "safety culture") → Enable `use_hyde` or `num_expands=1`.
    - **Specific/Keyword-heavy?** (e.g., "error code 503") → Prefer `bm25_only` or standard `hybrid`.
- This reduces the cognitive load on the calling agent and improves performance on varied query types.

## 3. Deep Graph Path Integration
**Observation:** Currently, `graph_boost` works by finding chunks that mention entities in the query and boosting their scores. It does not explicitly return the *relationship path* that justifies the relevance.
**Proposal:**
- Enhance `GraphEnhancedRetriever` (and `SearchTool`) to return the **traversal path** as metadata.
- Example output: "Chunk X is relevant because it mentions 'Entity A', which is 'superseded_by' 'Entity B' in your query."
- This provides "Explainable Retrieval" which is crucial for the RAG answer generation step.

## 4. Dynamic RRF Weights
**Observation:** `_rrf_fusion` uses a standard unweighted formula: `score = sum(1 / (k + rank))`.
**Proposal:**
- Allow weighted RRF where BM25 and Dense ranks contribute differently.
- For example, if `search_method="hybrid_keyword_bias"`, weight BM25 ranks higher.
- This helps in domains where exact terminology match is more important than semantic relatedness.

## 5. Iterative Refinement (Self-Correction)
**Observation:** The tool calculates `RAGConfidenceScore` but simply returns it.
**Proposal:**
- Implement an internal "retry loop" if confidence is `VERY LOW`.
- If the initial search yields low confidence, the tool could automatically:
    1.  Trigger Query Expansion (if not already used).
    2.  Increase `k`.
    3.  Try a different `search_method`.
- This makes the tool more robust without requiring the agent to implement the loop.

## 6. Performance Optimizations
**Observation:** Query Expansion and HyDE generation are sequential and synchronous.
**Proposal:**
- If the environment allows, parallelize the LLM calls for Query Expansion and HyDE generation.
- Implement caching for `_get_query_embedding` and `_get_query_expander` results to speed up repeated searches (common in multi-turn conversations).

## 7. Structured "Reasoning" Output
**Observation:** The tool returns a list of chunks.
**Proposal:**
- Add a top-level `summary` field to the `ToolResult`.
- Example: "Found 15 results. High confidence due to strong consensus between BM25 and Dense. Graph boost identified 2 key entities."
- This helps the Orchestrator agent make better decisions about whether to stop searching or continue.
