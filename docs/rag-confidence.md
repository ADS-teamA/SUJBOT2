# RAG Confidence Scoring

**Status:** ✅ Implemented (2025-10-29)  
**Location:** `src/agent/rag_confidence.py`  
**Integration:** Automatic in `search` tool + dedicated `assess_retrieval_confidence` tool

---

## Overview

RAG Confidence Scoring evaluates the **quality and reliability of retrieval results** before they're sent to the LLM. This provides early detection of retrieval failures and enables actionable improvements.

### Why RAG Confidence > LLM Confidence?

| Aspect | LLM Confidence | RAG Confidence |
|--------|----------------|----------------|
| **Speed** | 2-3s (15 samples) | <10ms (score analysis) |
| **Cost** | $0.01-0.05 per query | FREE (no API calls) |
| **Reliability** | Depends on LLM randomness | Based on retrieval scores |
| **Actionability** | "Low confidence" → ? | "BM25-Dense disagree" → improve query |
| **Debugging** | Hard to debug | Clear score breakdown |
| **Real-time** | No (too slow) | Yes (instant) |

---

## Architecture

### Confidence Metrics

The system evaluates retrieval quality using **7 key metrics**:

#### 1. Score-Based Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Top Score** | `max(rerank_score or rrf_score)` | How confident is the best match? |
| **Score Gap** | `top_score - second_score` | Is there a clear winner? |
| **Score Spread** | `std_dev(scores)` | Are results diverse or clustered? |
| **Consensus** | `count(score > 0.75)` | How many chunks agree? |

#### 2. Retrieval Method Agreement

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **BM25-Dense Correlation** | `corr(bm25_scores, dense_scores)` | Do keyword and semantic agree? |
| **Reranker Impact** | `1 - corr(rerank_scores, rrf_scores)` | Did reranker change ranking? |

#### 3. Context Quality

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Document Diversity** | `unique(document_ids) / k` | Are results from multiple docs? |
| **Graph Support** | `any(graph_boost > 0)` | Does knowledge graph support this? |

### Overall Confidence Formula

```python
confidence = (
    0.30 * top_score_norm +           # Most important
    0.20 * score_gap_norm +           # Clear winner?
    0.15 * consensus_norm +           # Multiple high-confidence chunks?
    0.15 * bm25_dense_agreement +     # Methods agree?
    0.10 * low_spread_norm +          # Consistent scores?
    0.05 * graph_support_norm +       # Knowledge graph confirms?
    0.05 * low_diversity_norm         # Single source?
)
```

**Weights based on:**
- RAGAS framework (context precision/recall)
- Legal compliance research (confidence thresholds)
- Hybrid search best practices

---

## Confidence Thresholds

Based on legal compliance research:

| Confidence | Range | Interpretation | Action |
|------------|-------|----------------|--------|
| **HIGH** | ≥0.85 | Strong retrieval confidence | ✅ Safe for automated response |
| **MEDIUM** | 0.70-0.84 | Moderate confidence | ⚠️ Review recommended |
| **LOW** | 0.50-0.69 | Weak retrieval | ⚠️ Mandatory review |
| **VERY LOW** | <0.50 | Poor retrieval | 🚨 Expert review required |

---

## Usage

### Automatic Scoring (Default)

The `search` tool automatically scores confidence for every query:

```python
# User query
> What are the requirements for waste disposal?

# Agent uses search tool
search(query="requirements for waste disposal", k=5)

# Returns results with confidence metadata
{
  "data": [...],  # Formatted chunks
  "citations": [
    "⚠️ MEDIUM - Moderate confidence, review recommended",  # Added if flagged
    "[1] doc1: Section 3.2",
    "[2] doc1: Section 4.1"
  ],
  "metadata": {
    "rag_confidence": {
      "overall_confidence": 0.78,
      "top_score": 0.82,
      "score_gap": 0.08,
      "interpretation": "MEDIUM - Moderate confidence, review recommended",
      "should_flag_for_review": true,
      "details": {...}
    }
  }
}
```

### Dedicated Tool

Use `assess_retrieval_confidence` for explicit confidence checking:

```python
# After search
search_results = search(query="...", k=5)
chunk_ids = [chunk["chunk_id"] for chunk in search_results["data"]]

# Assess confidence
confidence = assess_retrieval_confidence(chunk_ids=chunk_ids)

# Returns detailed breakdown
{
  "overall_confidence": 0.78,
  "top_score": 0.82,
  "score_gap": 0.08,
  "bm25_dense_agreement": 0.65,
  "graph_support": false,
  "interpretation": "MEDIUM - Moderate confidence, review recommended",
  "recommendations": [
    "MODERATE: Medium confidence. Review recommended for critical use cases.",
    "No knowledge graph support. Consider using multi_hop_search for graph-based retrieval."
  ]
}
```

---

## Example Scenarios

### High Confidence Example

```json
{
  "overall_confidence": 0.92,
  "top_score": 0.95,
  "score_gap": 0.12,
  "score_spread": 0.08,
  "consensus_count": 5,
  "bm25_dense_agreement": 0.87,
  "reranker_impact": 0.15,
  "graph_support": true,
  "document_diversity": 0.17,
  "interpretation": "HIGH - Strong retrieval confidence",
  "should_flag_for_review": false
}
```

**Analysis:**
- ✅ Top score is very high (0.95)
- ✅ Clear winner (gap = 0.12)
- ✅ BM25 and dense agree (0.87 correlation)
- ✅ Knowledge graph supports results
- ✅ Results from single document (focused)
- **→ Safe to use for automated response**

---

### Low Confidence Example

```json
{
  "overall_confidence": 0.48,
  "top_score": 0.62,
  "score_gap": 0.03,
  "score_spread": 0.18,
  "consensus_count": 0,
  "bm25_dense_agreement": 0.32,
  "reranker_impact": 0.45,
  "graph_support": false,
  "document_diversity": 0.83,
  "interpretation": "VERY LOW - Poor retrieval, expert review required",
  "should_flag_for_review": true
}
```

**Analysis:**
- ❌ Top score is mediocre (0.62)
- ❌ No clear winner (gap = 0.03)
- ❌ BM25 and dense disagree (0.32 correlation)
- ❌ High score spread (0.18 - inconsistent)
- ❌ Results from 5 different documents (scattered)
- **→ Flag for human review, don't auto-answer**

**Recommendations:**
- Try query expansion: `search(query="...", num_expands=3-5)`
- Try exact match: `exact_match_search(query="...", search_type="keywords")`
- Try graph search: `multi_hop_search(query="...", max_hops=2)`

---

## Integration Points

### 1. Search Tool (`src/agent/tools/search.py`)

Automatically scores confidence after retrieval:

```python
# After reranking
confidence_scorer = RAGConfidenceScorer()
confidence = confidence_scorer.score_retrieval(chunks, query=query)

# Add to metadata
result_metadata["rag_confidence"] = confidence.to_dict()

# Add warning to citations if flagged
if confidence.should_flag:
    citations.insert(0, f"⚠️ {confidence.interpretation}")
```

### 2. Dedicated Tool (`src/agent/tools/filtered_search.py`)

Provides explicit confidence assessment:

```python
@register_tool
class AssessRetrievalConfidenceTool(BaseTool):
    """Assess confidence of retrieval results."""

    name = "assess_retrieval_confidence"

    def execute_impl(self, chunk_ids: List[str]) -> ToolResult:
        # Find chunks by ID
        chunks = self._get_chunks_by_ids(chunk_ids)
        
        # Score confidence
        scorer = RAGConfidenceScorer()
        confidence = scorer.score_retrieval(chunks)
        
        # Add recommendations
        recommendations = self._generate_recommendations(confidence)
        
        return ToolResult(
            success=True,
            data=confidence.to_dict(),
            metadata={"recommendations": recommendations}
        )
```

### 3. Web UI (Future)

Display confidence badges in search results:

```html
<!-- High confidence -->
<span class="badge badge-success">
  🟢 High Confidence (0.92)
</span>

<!-- Medium confidence -->
<span class="badge badge-warning">
  🟡 Medium Confidence (0.78) - Review Recommended
</span>

<!-- Low confidence -->
<span class="badge badge-danger">
  🔴 Low Confidence (0.48) - Expert Review Required
</span>
```

---

## Testing

Comprehensive test suite: `tests/agent/test_rag_confidence.py`

```bash
# Run tests
uv run pytest tests/agent/test_rag_confidence.py -v

# Expected: 12 passed
```

**Test coverage:**
- ✅ Empty results
- ✅ High/medium/low confidence scenarios
- ✅ BM25-Dense agreement calculation
- ✅ Reranker impact calculation
- ✅ Document diversity calculation
- ✅ Graph support detection
- ✅ Score extraction priority
- ✅ Serialization to dict
- ✅ Custom thresholds
- ✅ Retrieval methods analysis

---

## Performance

| Operation | Time | Cost |
|-----------|------|------|
| Score calculation | <10ms | FREE |
| Metadata lookup | <5ms | FREE |
| Total overhead | <15ms | FREE |

**Impact on search latency:**
- Search without confidence: ~200ms
- Search with confidence: ~215ms (+7.5%)
- **Negligible overhead for significant value**

---

## Future Enhancements

1. **Adaptive Retrieval**
   - If confidence < 0.70, automatically trigger query expansion
   - If BM25-Dense disagree, try both separately and merge

2. **Confidence-Based Caching**
   - Cache high-confidence results longer
   - Invalidate low-confidence results faster

3. **User Feedback Loop**
   - Track user corrections on low-confidence results
   - Retrain confidence thresholds based on feedback

4. **Confidence Trends**
   - Track confidence over time per document
   - Identify documents with consistently low retrieval quality

---

## References

- **RAGAS Framework:** Context precision/recall metrics
- **Legal Compliance Research:** Confidence thresholds (≥90% automated, 70-89% review, <70% expert)
- **Hybrid Search Best Practices:** BM25-Dense agreement as quality signal
- **LegalBench-RAG:** Reranker impact analysis

---

## API Reference

### `RAGConfidenceScorer`

```python
class RAGConfidenceScorer:
    def __init__(
        self,
        high_confidence_threshold: float = 0.85,
        medium_confidence_threshold: float = 0.70,
        low_confidence_threshold: float = 0.50,
        consensus_threshold: float = 0.75,
    ):
        """Initialize RAG confidence scorer with custom thresholds."""
        
    def score_retrieval(
        self,
        chunks: List[Dict],
        query: Optional[str] = None
    ) -> RAGConfidenceScore:
        """Score confidence of RAG retrieval results."""
```

### `RAGConfidenceScore`

```python
@dataclass
class RAGConfidenceScore:
    overall_confidence: float          # 0-1, higher = more confident
    top_score: float                   # Best retrieval score
    score_gap: float                   # Gap between top and second
    score_spread: float                # Standard deviation of scores
    consensus_count: int               # Number of high-confidence chunks
    bm25_dense_agreement: float        # Correlation between BM25 and dense
    reranker_impact: float             # How much reranker changed ranking
    graph_support: bool                # Knowledge graph support
    document_diversity: float          # Diversity of source documents
    interpretation: str                # Human-readable level
    should_flag: bool                  # Whether to flag for review
    details: Dict                      # Detailed breakdown
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
```

---

## CLI Display

The RAG confidence score is displayed **after the agent's response** and **before the cost summary**.

### Display Format

```
📊 RAG Confidence: [emoji] [interpretation] ([score])
```

- **Emoji**: `✓` for high confidence (green), `⚠️` for low confidence (yellow warning)
- **Interpretation**: Human-readable confidence level
- **Score**: Numerical confidence (0.00 to 1.00)

### Example Outputs

**High Confidence (0.92):**
```
> What is the moderator in VR-1 reactor?

A: The VR-1 reactor uses light water as the moderator...

📊 RAG Confidence: ✓ HIGH - Strong retrieval confidence (0.92)

💰 This message: $0.0162
```

**Low Confidence (0.58):**
```
> What is the maximum operating temperature for component XYZ-123?

A: Based on the available documentation, component specifications suggest...

📊 RAG Confidence: ⚠️ LOW - Weak retrieval, mandatory review (0.58)

💰 This message: $0.0138
```

### When Confidence is Displayed

Confidence is displayed when:
1. The agent uses the `search` or `exact_match_search` tool
2. The retrieval returns results with scores

Confidence is **not** displayed for commands (`/help`) or metadata tools (`get_document_list`).

---

## Troubleshooting

### Issue: Confidence Score Not Displayed

**Symptom:** No `📊 RAG Confidence` line in output.

**Cause:** Agent used a tool without confidence scoring (e.g., `exact_match_search` before the fix, or metadata tools).

**Solution:** Both `search` and `exact_match_search` now include confidence scoring. If using other tools like `get_document_list`, confidence doesn't apply.

### Issue: Confidence Shows "Unknown" or 0.00

**Possible causes:**
1. Empty results (no chunks retrieved)
2. Missing scores in chunks
3. Exception during confidence calculation

**Debug:** Enable debug mode with `--debug` flag and check logs for `RAG confidence scoring failed`.

### Tools with RAG Confidence Scoring

**✅ With Confidence:**
- `search` - Hybrid search with BM25 + Dense + reranking
- `exact_match_search` - Fast BM25 keyword search
- `assess_retrieval_confidence` - Dedicated confidence tool

**❌ Without Confidence (not applicable):**
- `get_document_list`, `get_document_info`, `expand_context`, etc.

---

## Related Documentation

- **Implementation**: [`src/agent/rag_confidence.py`](../src/agent/rag_confidence.py)
- **Search Tool**: [`src/agent/tools/search.py`](../src/agent/tools/search.py)
- **Assessment Tool**: [`src/agent/tools/filtered_search.py`](../src/agent/tools/filtered_search.py)
- **CLI Integration**: [`src/agent/cli.py`](../src/agent/cli.py)
- **Tests**: [`tests/agent/test_rag_confidence.py`](../tests/agent/test_rag_confidence.py)

