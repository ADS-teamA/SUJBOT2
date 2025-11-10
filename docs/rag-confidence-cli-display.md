# RAG Confidence Score - CLI Display

This document shows how RAG confidence scores are displayed in the CLI after implementing the feature.

---

## Display Location

The RAG confidence score is displayed **after the agent's response** and **before the cost summary**, making it easy to see the retrieval quality for each query.

---

## Display Format

```
📊 RAG Confidence: [emoji] [interpretation] ([score])
```

- **Emoji**: 
  - `✓` for high confidence (green)
  - `⚠️` for low confidence (yellow warning)
- **Interpretation**: Human-readable confidence level
- **Score**: Numerical confidence (0.00 to 1.00)

---

## Example Outputs

### Example 1: High Confidence Query

```
╭─────────────────────────────────────────────────────────────╮
│              RAG Agent - Document Assistant                 │
│  Type your question or use /help for commands              │
╰─────────────────────────────────────────────────────────────╯

> What is the moderator in VR-1 reactor?

A: The VR-1 reactor uses light water as the moderator and light water 
as the reflector, or alternatively nuclear-grade graphite or beryllium 
[Doc: BZ_VR1, Section: Active zone of VR-1 reactor].

📊 RAG Confidence: ✓ HIGH - Strong retrieval confidence (0.92)

💰 This message: $0.0162
  Input (new): 6,135 tokens
  Output: 91 tokens

Session total: $0.0162 (6,226 tokens)
```

**Interpretation**: The system found highly relevant chunks with strong agreement between BM25 and dense retrieval. The answer is reliable.

---

### Example 2: Medium Confidence Query

```
> What are the safety procedures for emergency shutdown?

A: The emergency shutdown procedures include manual SCRAM activation, 
automatic safety systems, and coolant circulation protocols [Doc: Safety_Manual, 
Section: Emergency Procedures].

📊 RAG Confidence: ⚠️ MEDIUM - Moderate confidence, review recommended (0.75)

💰 This message: $0.0145
  Input (new): 5,892 tokens
  Output: 78 tokens

Session total: $0.0307 (12,196 tokens)
```

**Interpretation**: The retrieval found relevant information, but there's some uncertainty. The answer is likely correct but should be verified for critical applications.

---

### Example 3: Low Confidence Query

```
> What is the maximum operating temperature for component XYZ-123?

A: Based on the available documentation, component specifications suggest 
operating temperatures should not exceed standard limits. However, specific 
data for XYZ-123 was not found in the indexed documents.

📊 RAG Confidence: ⚠️ LOW - Weak retrieval, mandatory review (0.58)

💰 This message: $0.0138
  Input (new): 5,654 tokens
  Output: 65 tokens

Session total: $0.0445 (17,915 tokens)
```

**Interpretation**: The retrieval struggled to find relevant information. The answer may be incomplete or based on weak evidence. Manual verification required.

---

### Example 4: Very Low Confidence Query

```
> What is the quantum efficiency of the flux capacitor?

A: I could not find specific information about quantum efficiency of flux 
capacitors in the available documentation. The search returned general 
information about reactor components, but nothing directly relevant to 
your query.

📊 RAG Confidence: ⚠️ VERY LOW - Poor retrieval, expert review required (0.32)

💰 This message: $0.0125
  Input (new): 5,234 tokens
  Output: 58 tokens

Session total: $0.0570 (23,261 tokens)
```

**Interpretation**: The retrieval failed to find relevant information. The query may be outside the scope of indexed documents, or the terminology doesn't match. Expert review essential.

---

### Example 5: Query Without Search (No Confidence)

```
> /help

📖 Available Commands:
  /help, /h        - Show this help
  /model, /m       - List available models or switch model
  /stats, /s       - Show tool execution and cost statistics
  ...

💰 This message: $0.0000
  Input (new): 0 tokens
  Output: 0 tokens

Session total: $0.0570 (23,261 tokens)
```

**Note**: No confidence score is displayed because no search tool was used (command execution only).

---

## Color Coding

The confidence score is color-coded for quick visual assessment:

- **Green** (`✓`): High confidence - retrieval quality is strong
- **Yellow** (`⚠️`): Low/Medium/Very Low confidence - review recommended

This makes it easy to spot queries that may need additional verification at a glance.

---

## When Confidence is Displayed

RAG confidence is displayed when:
1. The agent uses the `search` tool (or any tool that performs retrieval)
2. The retrieval returns results with scores
3. The confidence scorer successfully analyzes the results

Confidence is **not** displayed when:
- No search tool is used (e.g., `/help` command)
- Tools like `get_document_list` or `get_document_info` are used (no retrieval scoring)
- The query is answered from cached knowledge without retrieval

---

## Integration with Streaming Mode

The confidence display works in both streaming and non-streaming modes:

### Streaming Mode (Default)
```
> What is RAG?

[Using search...]
[✓ RAG Confidence: HIGH - Strong retrieval confidence (0.88)]

A: RAG (Retrieval-Augmented Generation) is a technique that combines...

📊 RAG Confidence: ✓ HIGH - Strong retrieval confidence (0.88)

💰 This message: $0.0156
```

**Note**: In streaming mode, confidence is shown **twice**:
1. During tool execution (blue `[...]` notification)
2. After the response (green/yellow summary line)

### Non-Streaming Mode
```
> What is RAG?

A: RAG (Retrieval-Augmented Generation) is a technique that combines...

📊 RAG Confidence: ✓ HIGH - Strong retrieval confidence (0.88)

💰 This message: $0.0156
```

**Note**: In non-streaming mode, confidence is shown **once** after the response.

---

## Technical Details

### Implementation
- **Location**: `src/agent/cli.py` (lines 363-382)
- **Data Source**: `agent.get_latest_rag_confidence()` from tool call history
- **Scoring**: Automatic via `RAGConfidenceScorer` in search tool
- **Display Logic**: Color-coded based on `should_flag_for_review` flag

### Confidence Metrics
The displayed score is a weighted combination of 7 metrics:
1. Top Score (30%)
2. Score Gap (20%)
3. Consensus Count (15%)
4. BM25-Dense Agreement (15%)
5. Score Spread (10%)
6. Graph Support (5%)
7. Document Diversity (5%)

See [`docs/rag-confidence.md`](rag-confidence.md) for detailed metric explanations.

---

## Benefits

1. **Immediate Feedback**: Users see retrieval quality instantly
2. **Trust Calibration**: Users know when to trust vs verify answers
3. **Quality Assurance**: Low confidence flags potential issues
4. **Debugging Aid**: Helps identify queries that need better indexing
5. **Transparency**: Makes RAG system behavior visible to users

---

## Future Enhancements

Potential improvements to the display:

1. **Detailed Breakdown**: Show individual metric scores on demand
2. **Historical Tracking**: Track confidence trends over session
3. **Recommendations**: Suggest query improvements for low confidence
4. **Confidence Threshold Alerts**: Configurable warning levels
5. **Export to Logs**: Save confidence scores for analysis

---

## Related Documentation

- **Core Implementation**: [`src/agent/rag_confidence.py`](../src/agent/rag_confidence.py)
- **Detailed Metrics**: [`docs/rag-confidence.md`](rag-confidence.md)
- **Search Tool Integration**: [`src/agent/tools/tier1_basic.py`](../src/agent/tools/tier1_basic.py)
- **CLI Implementation**: [`src/agent/cli.py`](../src/agent/cli.py)

