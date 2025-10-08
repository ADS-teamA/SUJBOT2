# Query Processor Documentation

## Overview

The Query Processor is a sophisticated query analysis and decomposition system designed for legal compliance analysis. It transforms complex natural language queries into structured, analyzable components that enable precise document retrieval and compliance checking.

**Key Capabilities:**
- Intent classification (gap analysis, conflict detection, risk assessment, etc.)
- Complexity assessment (simple, moderate, complex, expert)
- Entity extraction (legal references, dates, obligations, prohibitions)
- Query decomposition into targeted sub-questions
- Query expansion with synonyms and related terms
- Automatic retrieval strategy selection

## Architecture

```
┌─────────────────────────────────────┐
│  User Query                         │
│  "Najdi všechna slabá místa ve      │
│   smlouvě, která se neshodují se    │
│   zákonem"                          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  QueryClassifier                    │
│  - Pattern-based intent detection   │
│  - LLM fallback (Claude Haiku)      │
│  - Complexity scoring               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  EntityExtractor                    │
│  - Legal refs (§89, Článek 5)       │
│  - Dates and deadlines              │
│  - Obligations/prohibitions         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LegalQuestionDecomposer            │
│  - Strategy selection               │
│  - Sub-question generation (Haiku)  │
│  - Dependency analysis              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  QueryExpander                      │
│  - Synonym expansion                │
│  - Legal term variants              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ProcessedQuery                     │
│  - Intent + complexity              │
│  - Entities + sub-queries           │
│  - Expanded terms                   │
│  - Retrieval strategy               │
└─────────────────────────────────────┘
```

## Components

### 1. Query Classifier

Classifies query intent and assesses complexity.

**Intent Types:**
- **Compliance Analysis:**
  - `GAP_ANALYSIS`: Finding missing requirements
  - `CONFLICT_DETECTION`: Finding contradictions
  - `RISK_ASSESSMENT`: Identifying legal risks
  - `COMPLIANCE_CHECK`: Verifying compliance

- **Information Retrieval:**
  - `FACTUAL`: Simple fact retrieval
  - `COMPARISON`: Comparing provisions
  - `DEFINITION`: Defining legal terms
  - `EXPLANATION`: Explaining provisions

- **Analytical:**
  - `ENUMERATION`: Listing instances
  - `RELATIONSHIP`: Understanding relationships
  - `CONSEQUENCE`: Understanding implications

**Complexity Levels:**
- `SIMPLE`: Single fact, direct answer
- `MODERATE`: Multiple facts, some reasoning
- `COMPLEX`: Multi-step analysis required
- `EXPERT`: Deep legal reasoning, multi-document

**Classification Strategy:**
1. **Pattern-based** (fast): Regex matching against intent patterns
2. **LLM-based** (fallback): Claude Haiku for nuanced understanding
3. **Complexity scoring**: Linguistic heuristics (word count, question marks, conditionals)

### 2. Entity Extractor

Extracts structured entities from queries using regex patterns.

**Legal References:**
```python
# Examples:
"§89"                      → Normalized: "§89"
"§89 odst. 2"              → Normalized: "§89 odst. 2"
"§89 odst. 2 písm. a)"     → Normalized: "§89 odst. 2 písm. a)"
"zákon č. 89/2012 Sb."     → Normalized: "Zákon č. 89/2012 Sb."
"Článek 5.2"               → Normalized: "Článek 5.2"
```

**Temporal Entities:**
```python
# Examples:
"15.10.2024"               → Type: date
"do 30 dní"                → Type: date
"termín"                   → Type: date (keyword)
```

**Obligations/Prohibitions:**
```python
# Obligations: musí, povinen, má povinnost, must, shall
# Prohibitions: nesmí, zakázáno, prohibited, must not
```

### 3. Legal Question Decomposer

Breaks complex queries into targeted sub-questions using Claude Haiku.

**Decomposition Strategies:**

| Intent | Strategy | Sub-Questions |
|--------|----------|---------------|
| `GAP_ANALYSIS` | `requirement_based` | 1. Legal requirements<br>2. Contract specifications<br>3. Missing items |
| `CONFLICT_DETECTION` | `provision_pairing` | 1. Contract provisions<br>2. Law provisions<br>3. Conflicts |
| `RISK_ASSESSMENT` | `risk_category` | 1. Legal risks<br>2. Contractual risks<br>3. Financial risks |
| `COMPARISON` | `entity_separation` | 1. First entity<br>2. Second entity<br>3. Comparison |
| `COMPLIANCE_CHECK` | `clause_by_clause` | 1. Legal requirements<br>2. Contract clauses<br>3. Compliance check |

**Sub-Query Properties:**
- `sub_query_id`: Unique identifier (e.g., "sq_1", "sq_2")
- `text`: Sub-question text
- `intent`: Classified intent
- `priority`: Execution priority (1-5)
- `depends_on`: List of prerequisite sub-query IDs
- `retrieval_strategy`: "hybrid" | "cross_document" | "graph_aware"
- `target_document_types`: ["contract", "law_code"]

### 4. Query Expander

Expands query terms with synonyms and related legal terms.

**Synonym Dictionary:**
```python
{
    "smlouva": ["kontrakt", "dohoda", "ujednání", "contract"],
    "zákon": ["legislation", "předpis", "legal code"],
    "povinnost": ["závazek", "obligation", "duty"],
    "zákaz": ["prohibice", "prohibition", "ban"],
    "dodavatel": ["kontraktorem", "supplier", "zhotovitel"],
    "odpovědnost": ["liability", "ručení", "accountability"],
    # ... more terms
}
```

## Usage

### Basic Usage

```python
from src.query_processor import QueryProcessor
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize processor
processor = QueryProcessor(config)

# Process query
query = "Najdi všechny konflikty mezi smlouvou a zákonem č. 89/2012 Sb."
processed = await processor.process(query)

# Access results
print(f"Intent: {processed.intent.value}")
print(f"Complexity: {processed.complexity.value}")
print(f"Entities: {len(processed.entities)}")

if processed.requires_decomposition:
    print(f"\nSub-queries ({len(processed.sub_queries)}):")
    for sq in processed.sub_queries:
        print(f"  {sq.priority}. {sq.text}")
        print(f"     Strategy: {sq.retrieval_strategy}")
```

### Running Tests

```bash
# Full test suite
python test_query_processor.py --mode full

# Feature-specific tests
python test_query_processor.py --mode features

# Interactive testing
python test_query_processor.py --mode interactive

# Unit tests with pytest
pytest tests/test_query_processor.py -v
```

## Configuration

Add to `config.yaml`:

```yaml
query_processing:
  # LLM for classification and decomposition
  llm_model: "claude-3-5-haiku-20241022"
  llm_temperature: 0.3
  llm_max_tokens: 1000

  # Decomposition settings
  enable_decomposition: true
  max_sub_queries: 5
  decomposition_complexity_threshold: "moderate"  # simple | moderate | complex

  # Entity extraction
  extract_legal_references: true
  extract_temporal_entities: true
  extract_obligations: true

  # Query expansion
  enable_query_expansion: true
  max_synonyms_per_term: 3

  # Retrieval strategy selection
  auto_select_strategy: true

  # Logging
  verbose_logging: false
```

## Examples

### Example 1: Gap Analysis

**Query:**
```
"Které povinné body ze zákona č. 89/2012 Sb. chybí v této smlouvě?"
```

**Output:**
```python
ProcessedQuery(
    intent=QueryIntent.GAP_ANALYSIS,
    complexity=QueryComplexity.COMPLEX,
    entities=[
        ExtractedEntity(type="legal_ref", value="zákon č. 89/2012 Sb.",
                       normalized="Zákon č. 89/2012 Sb.")
    ],
    requires_decomposition=True,
    decomposition_strategy="requirement_based",
    sub_queries=[
        SubQuery(id="sq_1", text="Jaké jsou povinné body podle zákona č. 89/2012 Sb.?",
                intent=QueryIntent.FACTUAL, strategy="hybrid", targets=["law_code"]),
        SubQuery(id="sq_2", text="Které body obsahuje smlouva?",
                intent=QueryIntent.FACTUAL, strategy="hybrid", targets=["contract"]),
        SubQuery(id="sq_3", text="Které povinné body ve smlouvě chybí?",
                intent=QueryIntent.GAP_ANALYSIS, strategy="cross_document",
                targets=["contract", "law_code"], depends_on=["sq_1", "sq_2"])
    ],
    retrieval_strategy="cross_document"
)
```

### Example 2: Conflict Detection

**Query:**
```
"Najdi konflikty mezi smlouvou a §89 odst. 2."
```

**Output:**
```python
ProcessedQuery(
    intent=QueryIntent.CONFLICT_DETECTION,
    complexity=QueryComplexity.COMPLEX,
    entities=[
        ExtractedEntity(type="legal_ref", value="§89 odst. 2",
                       normalized="§89 odst. 2")
    ],
    sub_queries=[
        SubQuery(id="sq_1", text="Co specifikuje smlouva ohledně předmětu §89 odst. 2?",
                strategy="hybrid", targets=["contract"]),
        SubQuery(id="sq_2", text="Co vyžaduje §89 odst. 2?",
                strategy="hybrid", targets=["law_code"]),
        SubQuery(id="sq_3", text="Kde smlouva odporuje §89 odst. 2?",
                strategy="cross_document", depends_on=["sq_1", "sq_2"])
    ]
)
```

### Example 3: Simple Factual Query

**Query:**
```
"Co je termín dokončení?"
```

**Output:**
```python
ProcessedQuery(
    intent=QueryIntent.FACTUAL,
    complexity=QueryComplexity.SIMPLE,
    entities=[
        ExtractedEntity(type="date", value="termín", normalized="termín")
    ],
    requires_decomposition=False,
    sub_queries=[],
    expanded_terms={
        "termín": ["lhůta", "deadline", "time limit"]
    },
    retrieval_strategy="hybrid"
)
```

## Integration with Retrieval Pipeline

The processed query integrates with the retrieval system:

```python
# 1. Process query
processed = await query_processor.process(user_query)

# 2. Route to appropriate retrieval strategy
if processed.retrieval_strategy == "cross_document":
    # Use cross-document retrieval for compliance checks
    results = await cross_document_retriever.retrieve(processed)
elif processed.retrieval_strategy == "graph_aware":
    # Use graph-based retrieval for relationship queries
    results = await graph_retriever.retrieve(processed)
else:
    # Use hybrid retrieval for standard queries
    results = await hybrid_retriever.retrieve(processed)

# 3. If decomposed, process sub-queries
if processed.requires_decomposition:
    sub_results = []
    for sub_query in processed.sub_queries:
        # Check dependencies
        if all(dep_completed(dep) for dep in sub_query.depends_on):
            result = await retrieve_with_strategy(
                sub_query.text,
                sub_query.retrieval_strategy,
                sub_query.target_document_types
            )
            sub_results.append(result)

    # Synthesize final answer from sub-results
    final_answer = await answer_synthesizer.synthesize(
        processed, sub_results
    )
```

## Performance

**Typical Processing Times:**

| Component | Average Time | Notes |
|-----------|-------------|-------|
| Pattern classification | <0.1s | Regex-based, fast |
| LLM classification | 0.5-1.5s | Claude Haiku call |
| Entity extraction | <0.1s | Regex-based |
| Decomposition | 1-2s | Claude Haiku call |
| Query expansion | <0.05s | Dictionary lookup |
| **Total pipeline** | **1-3s** | Depends on complexity |

**Optimization Strategies:**

1. **Pattern-first classification**: Avoids LLM calls for obvious queries
2. **Parallel execution**: Classification and entity extraction run concurrently
3. **Caching**: LLM responses cached for repeated queries
4. **Selective decomposition**: Simple queries skip decomposition

## Limitations and Future Work

**Current Limitations:**

1. **Czech-English hybrid**: Patterns and synonyms support Czech and English but not other languages
2. **Static synonym dictionary**: Would benefit from dynamic expansion using embeddings
3. **Simple dependency inference**: Could use more sophisticated dependency graph analysis
4. **No multi-turn context**: Each query processed independently

**Future Enhancements:**

1. **Few-shot learning**: Provide examples in prompts for better classification
2. **Dependency DAG**: Build explicit dependency graph for parallel sub-query execution
3. **Query rewriting**: Rewrite ambiguous queries for clarity
4. **Multi-turn understanding**: Maintain conversation context, resolve anaphora
5. **Legal ontology integration**: Use legal taxonomy for term expansion
6. **Fine-tuned models**: Domain-specific embeddings for better entity recognition

## API Reference

### QueryProcessor

```python
class QueryProcessor:
    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None)
    async def process(self, query: str) -> ProcessedQuery
```

### ProcessedQuery

```python
@dataclass
class ProcessedQuery:
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    requires_decomposition: bool
    entities: List[ExtractedEntity]
    sub_queries: List[SubQuery]
    expanded_terms: Dict[str, List[str]]
    retrieval_strategy: str
    processing_time: float
    decomposition_strategy: Optional[str]
```

### QueryIntent

```python
class QueryIntent(Enum):
    GAP_ANALYSIS = "gap_analysis"
    CONFLICT_DETECTION = "conflict_detection"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    FACTUAL = "factual"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    ENUMERATION = "enumeration"
    RELATIONSHIP = "relationship"
    CONSEQUENCE = "consequence"
```

### QueryComplexity

```python
class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"
```

## Troubleshooting

### Issue: Classification always returns FACTUAL

**Cause**: Patterns not matching, LLM fallback not working
**Solution**:
- Check if API key is set
- Enable verbose logging to see classification path
- Review pattern matches in debug output

### Issue: No entities extracted

**Cause**: Query doesn't contain recognizable legal references
**Solution**:
- Check if entity extraction is enabled in config
- Verify regex patterns match your legal citation format
- Add custom patterns if needed

### Issue: Decomposition not triggering

**Cause**: Complexity threshold too high or decomposition disabled
**Solution**:
- Lower `decomposition_complexity_threshold` in config
- Set `enable_decomposition: true`
- Check if query intent is in decomposition list

### Issue: Slow processing

**Cause**: Multiple LLM calls for classification and decomposition
**Solution**:
- Reduce LLM calls by relying on pattern-based classification
- Set `auto_select_strategy: false` to skip strategy selection
- Disable decomposition for simple queries

## License

Part of the Advanced SUJBOT2 project.

## Author

Advanced SUJBOT2 Team
Date: 2025-10-08
