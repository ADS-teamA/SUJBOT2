# 08. Query Processing Specification

## 1. Purpose

**Objective**: Analyze, classify, and decompose user queries to optimize retrieval and enable sophisticated legal compliance analysis.

**Why Query Processing?**
- Legal compliance queries are often complex and multi-faceted
- Single retrieval pass insufficient for comprehensive analysis
- Different query types require different retrieval strategies
- Sub-query decomposition enables divide-and-conquer approach
- Entity extraction focuses retrieval on specific provisions

**Key Capabilities**:
1. **Query Classification** - Identify query intent (gap analysis, conflict detection, risk assessment, etc.)
2. **Legal Question Decomposition** - Break complex queries into targeted sub-questions
3. **Entity Extraction** - Extract legal references, parties, dates, obligations
4. **Query Expansion** - Add synonyms and related terms for better recall
5. **Compliance-Aware Routing** - Direct queries to appropriate retrieval strategies

---

## 2. Query Processing Architecture

### High-Level Flow

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
│  Query Classifier                   │
│  - Intent detection                 │
│  - Complexity analysis              │
│  → Query type + complexity score    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Entity Extractor                   │
│  - Legal references (§89)           │
│  - Document types                   │
│  - Temporal constraints             │
│  → Extracted entities               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Question Decomposer                │
│  - Generate sub-questions           │
│  - Assign retrieval strategies      │
│  → List of sub-queries              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Query Expander                     │
│  - Add synonyms                     │
│  - Add related terms                │
│  → Expanded query variants          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Processed Query                    │
│  - Original + sub-queries           │
│  - Entities                         │
│  - Expansions                       │
│  - Retrieval strategy               │
└─────────────────────────────────────┘
```

### Component Interaction

```
QueryProcessor
├── QueryClassifier
│   └── Detects intent and complexity
├── EntityExtractor
│   ├── LegalReferenceExtractor (§, články)
│   ├── TemporalExtractor (dates, deadlines)
│   └── ObligationExtractor (must, shall, may)
├── LegalQuestionDecomposer
│   ├── SubQuestionGenerator (Claude Haiku)
│   └── StrategyAssigner
└── QueryExpander
    ├── SynonymExpander
    └── LegalTermExpander
```

---

## 3. Data Structures

### 3.1 Query Types and Intents

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime

class QueryIntent(Enum):
    """High-level query intent."""
    # Compliance analysis
    GAP_ANALYSIS = "gap_analysis"           # Missing requirements in contract
    CONFLICT_DETECTION = "conflict_detection"  # Contract contradicts law
    RISK_ASSESSMENT = "risk_assessment"     # Identify legal risks
    COMPLIANCE_CHECK = "compliance_check"   # Is contract compliant?

    # Information retrieval
    FACTUAL = "factual"                     # What is X?
    COMPARISON = "comparison"               # Compare X vs Y
    DEFINITION = "definition"               # Define legal term
    EXPLANATION = "explanation"             # Explain provision

    # Analytical
    ENUMERATION = "enumeration"             # List all X
    RELATIONSHIP = "relationship"           # How does X relate to Y?
    CONSEQUENCE = "consequence"             # What happens if X?

class QueryComplexity(Enum):
    """Query complexity level."""
    SIMPLE = "simple"           # Single fact, direct answer
    MODERATE = "moderate"       # Multiple facts, some reasoning
    COMPLEX = "complex"         # Multi-step analysis, synthesis required
    EXPERT = "expert"           # Deep legal reasoning, multi-document

@dataclass
class ExtractedEntity:
    """Entity extracted from query."""
    entity_type: str  # legal_ref | party | date | obligation | prohibition | term
    value: str        # Actual text
    normalized: Optional[str] = None  # Normalized form (e.g., "§89 odst. 2")
    confidence: float = 1.0
    span: tuple[int, int] = (0, 0)  # Character span in query

@dataclass
class ProcessedQuery:
    """Result of query processing."""
    # Original query
    original_query: str

    # Classification
    intent: QueryIntent
    complexity: QueryComplexity
    requires_decomposition: bool

    # Extracted entities
    entities: List[ExtractedEntity]

    # Sub-queries (if decomposed)
    sub_queries: List['SubQuery'] = field(default_factory=list)

    # Query expansion
    expanded_terms: Dict[str, List[str]] = field(default_factory=dict)  # {term: [synonyms]}

    # Retrieval strategy
    retrieval_strategy: str = "hybrid"  # hybrid | cross_document | graph_aware

    # Metadata
    processing_time: float = 0.0
    decomposition_strategy: Optional[str] = None

@dataclass
class SubQuery:
    """Sub-question generated from complex query."""
    sub_query_id: str
    text: str
    intent: QueryIntent
    priority: int  # 1 (high) to 5 (low)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # IDs of prerequisite sub-queries

    # Retrieval settings
    retrieval_strategy: str = "hybrid"
    target_document_types: List[str] = field(default_factory=list)  # ['law_code', 'contract']
    focus_entities: List[ExtractedEntity] = field(default_factory=list)

    # Execution state
    executed: bool = False
    result: Optional[str] = None
```

---

## 4. Query Classifier

### 4.1 Intent Detection

**Approach**: Pattern matching + LLM-based classification (Claude Haiku)

**Patterns**:
```python
INTENT_PATTERNS = {
    QueryIntent.GAP_ANALYSIS: [
        r"chybí|scházející|nepokryto|nevyhovuje",
        r"gap|missing|absent",
        r"nesplňuje požadavky"
    ],
    QueryIntent.CONFLICT_DETECTION: [
        r"konflikt|rozpor|nesoulad|neshoduje se",
        r"conflict|contradiction|inconsistent",
        r"odchyluje se od|porušuje"
    ],
    QueryIntent.RISK_ASSESSMENT: [
        r"riziko|nebezpečí|hrozba",
        r"risk|danger|vulnerability",
        r"slabá místa|problematické"
    ],
    QueryIntent.COMPARISON: [
        r"porovnej|srovnej|rozdíl",
        r"compare|versus|vs\.?|difference"
    ],
    QueryIntent.ENUMERATION: [
        r"všechny?|vyjmenuj|seznam",
        r"list all|enumerate|find all"
    ]
}
```

### 4.2 Implementation

```python
import re
from anthropic import Anthropic

class QueryClassifier:
    """Classify query intent and complexity."""

    def __init__(self, llm_client: Anthropic):
        self.llm = llm_client
        self.intent_patterns = INTENT_PATTERNS

    async def classify(self, query: str) -> tuple[QueryIntent, QueryComplexity]:
        """
        Classify query intent and complexity.

        Returns:
            (intent, complexity)
        """
        # First: fast pattern-based classification
        pattern_intent = self._classify_by_patterns(query)

        if pattern_intent and self._is_high_confidence_match(query, pattern_intent):
            intent = pattern_intent
        else:
            # Fallback: LLM-based classification
            intent = await self._classify_with_llm(query)

        # Determine complexity
        complexity = self._assess_complexity(query)

        return intent, complexity

    def _classify_by_patterns(self, query: str) -> Optional[QueryIntent]:
        """Match query against intent patterns."""
        query_lower = query.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        return None

    def _is_high_confidence_match(self, query: str, intent: QueryIntent) -> bool:
        """Check if pattern match is confident enough."""
        # Simple heuristic: at least 2 patterns match
        query_lower = query.lower()
        patterns = self.intent_patterns[intent]

        match_count = sum(
            1 for pattern in patterns
            if re.search(pattern, query_lower)
        )

        return match_count >= 2

    async def _classify_with_llm(self, query: str) -> QueryIntent:
        """
        Classify using Claude Haiku for nuanced understanding.
        """
        prompt = f"""Classify the intent of this legal query.

Query: "{query}"

Possible intents:
- gap_analysis: Finding missing requirements in contract
- conflict_detection: Finding contradictions between contract and law
- risk_assessment: Identifying legal risks
- compliance_check: Verifying compliance
- factual: Simple fact retrieval
- comparison: Comparing two provisions
- definition: Defining a legal term
- explanation: Explaining a provision
- enumeration: Listing all instances of something
- relationship: Understanding relationships between provisions
- consequence: Understanding implications

Respond with ONLY the intent name, nothing else."""

        response = await self.llm.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=50,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        intent_str = response.content[0].text.strip().lower()

        # Map to enum
        try:
            return QueryIntent(intent_str)
        except ValueError:
            # Default fallback
            return QueryIntent.FACTUAL

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """
        Assess query complexity based on linguistic features.
        """
        # Heuristics
        word_count = len(query.split())
        has_multiple_questions = query.count("?") > 1 or " a " in query.lower()
        has_comparison = any(word in query.lower() for word in ["porovnej", "compare", "rozdíl", "vs"])
        has_enumeration = any(word in query.lower() for word in ["všechny", "all", "list", "vyjmenuj"])
        has_conditional = any(word in query.lower() for word in ["pokud", "if", "when", "když"])

        # Score complexity
        complexity_score = 0

        if word_count > 30:
            complexity_score += 2
        elif word_count > 15:
            complexity_score += 1

        if has_multiple_questions:
            complexity_score += 2

        if has_comparison or has_enumeration:
            complexity_score += 1

        if has_conditional:
            complexity_score += 1

        # Map score to complexity
        if complexity_score >= 4:
            return QueryComplexity.EXPERT
        elif complexity_score >= 3:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
```

---

## 5. Entity Extractor

### 5.1 Legal Reference Extraction

```python
import re
from typing import List

class LegalReferenceExtractor:
    """Extract legal references from query."""

    # Regex patterns for Czech legal citations
    PATTERNS = {
        "paragraph_full": r"§\s*(\d+)\s*(?:odst\.\s*(\d+))?\s*(?:písm\.\s*([a-z]))?",
        "paragraph_simple": r"paragraf(?:u)?\s+(\d+)",
        "article": r"[Čč]lánek\s+(\d+)(?:\.(\d+))?",
        "law_reference": r"[Zz]ákon(?:a)?\s+č\.\s*(\d+)/(\d{4})\s*Sb\.",
        "section": r"(?:[Čč]ást|ČÁST)\s+([IVX]+)",
        "chapter": r"(?:[Hh]lava|HLAVA)\s+([IVX]+)"
    }

    def extract(self, query: str) -> List[ExtractedEntity]:
        """Extract all legal references from query."""
        entities = []

        for ref_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, query):
                # Construct normalized reference
                if ref_type == "paragraph_full":
                    para_num, subsec, letter = match.groups()
                    normalized = f"§{para_num}"
                    if subsec:
                        normalized += f" odst. {subsec}"
                    if letter:
                        normalized += f" písm. {letter}"
                elif ref_type == "article":
                    article_num, point = match.groups()
                    normalized = f"Článek {article_num}"
                    if point:
                        normalized += f".{point}"
                elif ref_type == "law_reference":
                    law_num, year = match.groups()
                    normalized = f"Zákon č. {law_num}/{year} Sb."
                else:
                    normalized = match.group(0)

                entity = ExtractedEntity(
                    entity_type="legal_ref",
                    value=match.group(0),
                    normalized=normalized,
                    confidence=0.95,
                    span=(match.start(), match.end())
                )
                entities.append(entity)

        return entities

class TemporalExtractor:
    """Extract dates and deadlines."""

    PATTERNS = {
        "absolute_date": r"(\d{1,2}\.\s*\d{1,2}\.\s*\d{4})",
        "relative_date": r"(do|od|před|po)\s+(\d+)\s+(dn(?:í|ů|y)|měsíc(?:ů|e)?|rok(?:ů|y)?)",
        "deadline_keyword": r"termín|lhůta|deadline"
    }

    def extract(self, query: str) -> List[ExtractedEntity]:
        """Extract temporal entities."""
        entities = []

        for date_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, query):
                entity = ExtractedEntity(
                    entity_type="date",
                    value=match.group(0),
                    normalized=match.group(0),  # Would normalize in real impl
                    confidence=0.85,
                    span=(match.start(), match.end())
                )
                entities.append(entity)

        return entities

class ObligationExtractor:
    """Extract obligations and prohibitions."""

    OBLIGATION_KEYWORDS = [
        "musí", "povinen", "má povinnost", "must", "shall", "required"
    ]

    PROHIBITION_KEYWORDS = [
        "nesmí", "zakázáno", "prohibited", "must not", "shall not"
    ]

    def extract(self, query: str) -> List[ExtractedEntity]:
        """Extract obligation/prohibition mentions."""
        entities = []
        query_lower = query.lower()

        # Check obligations
        for keyword in self.OBLIGATION_KEYWORDS:
            if keyword in query_lower:
                idx = query_lower.find(keyword)
                entity = ExtractedEntity(
                    entity_type="obligation",
                    value=keyword,
                    normalized="OBLIGATION",
                    confidence=0.9,
                    span=(idx, idx + len(keyword))
                )
                entities.append(entity)

        # Check prohibitions
        for keyword in self.PROHIBITION_KEYWORDS:
            if keyword in query_lower:
                idx = query_lower.find(keyword)
                entity = ExtractedEntity(
                    entity_type="prohibition",
                    value=keyword,
                    normalized="PROHIBITION",
                    confidence=0.9,
                    span=(idx, idx + len(keyword))
                )
                entities.append(entity)

        return entities

class EntityExtractor:
    """Main entity extraction coordinator."""

    def __init__(self):
        self.legal_ref_extractor = LegalReferenceExtractor()
        self.temporal_extractor = TemporalExtractor()
        self.obligation_extractor = ObligationExtractor()

    def extract(self, query: str) -> List[ExtractedEntity]:
        """Extract all entities from query."""
        entities = []

        entities.extend(self.legal_ref_extractor.extract(query))
        entities.extend(self.temporal_extractor.extract(query))
        entities.extend(self.obligation_extractor.extract(query))

        # Sort by span position
        entities.sort(key=lambda e: e.span[0])

        return entities
```

---

## 6. Legal Question Decomposer

### 6.1 Decomposition Strategies

**Strategy Selection**:
- **Gap Analysis** → Generate sub-questions for each requirement category
- **Conflict Detection** → Separate sub-questions for contract clauses vs law provisions
- **Comparison** → Break into "What is X?", "What is Y?", "Compare X vs Y"
- **Enumeration** → Single focused query for listing

### 6.2 Implementation

```python
class LegalQuestionDecomposer:
    """Decompose complex legal queries into sub-questions."""

    def __init__(self, llm_client: Anthropic):
        self.llm = llm_client

    async def decompose(
        self,
        query: ProcessedQuery
    ) -> List[SubQuery]:
        """
        Decompose query into sub-questions if needed.

        Returns:
            List of SubQuery objects
        """
        # Simple queries don't need decomposition
        if query.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
            if not self._requires_decomposition(query):
                return []

        # Select decomposition strategy
        strategy = self._select_strategy(query.intent)

        # Generate sub-questions using LLM
        sub_questions = await self._generate_sub_questions(query, strategy)

        # Assign retrieval strategies
        sub_queries = self._assign_strategies(sub_questions, query)

        return sub_queries

    def _requires_decomposition(self, query: ProcessedQuery) -> bool:
        """Determine if query needs decomposition."""
        # Always decompose compliance queries
        if query.intent in [
            QueryIntent.GAP_ANALYSIS,
            QueryIntent.CONFLICT_DETECTION,
            QueryIntent.RISK_ASSESSMENT
        ]:
            return True

        # Decompose complex comparisons
        if query.intent == QueryIntent.COMPARISON and query.complexity != QueryComplexity.SIMPLE:
            return True

        # Decompose enumerations with analysis
        if query.intent == QueryIntent.ENUMERATION and "analyze" in query.original_query.lower():
            return True

        return False

    def _select_strategy(self, intent: QueryIntent) -> str:
        """Select decomposition strategy based on intent."""
        strategies = {
            QueryIntent.GAP_ANALYSIS: "requirement_based",
            QueryIntent.CONFLICT_DETECTION: "provision_pairing",
            QueryIntent.RISK_ASSESSMENT: "risk_category",
            QueryIntent.COMPARISON: "entity_separation",
            QueryIntent.COMPLIANCE_CHECK: "clause_by_clause"
        }

        return strategies.get(intent, "generic")

    async def _generate_sub_questions(
        self,
        query: ProcessedQuery,
        strategy: str
    ) -> List[str]:
        """
        Generate sub-questions using Claude Haiku.
        """
        prompt = self._build_decomposition_prompt(query, strategy)

        response = await self.llm.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response (expects numbered list)
        sub_questions = self._parse_sub_questions(response.content[0].text)

        return sub_questions

    def _build_decomposition_prompt(
        self,
        query: ProcessedQuery,
        strategy: str
    ) -> str:
        """Build prompt for sub-question generation."""
        strategy_instructions = {
            "requirement_based": """Break down into sub-questions covering:
1. What are the legal requirements?
2. What does the contract specify?
3. Which requirements are missing or incomplete?""",

            "provision_pairing": """Break down into sub-questions:
1. What are the relevant contract provisions?
2. What are the corresponding law provisions?
3. Where do they conflict?""",

            "risk_category": """Break down into risk categories:
1. Legal compliance risks
2. Contractual obligation risks
3. Financial/penalty risks
4. Operational risks""",

            "entity_separation": """Break down the comparison:
1. Describe the first entity
2. Describe the second entity
3. Compare them directly""",

            "generic": """Break down into logical sub-questions that:
- Can be answered independently
- Together cover the full query
- Are specific and focused"""
        }

        instructions = strategy_instructions.get(strategy, strategy_instructions["generic"])

        # Extract entity context
        entities_str = ", ".join([e.normalized or e.value for e in query.entities])

        prompt = f"""Decompose this complex legal query into focused sub-questions.

Query: "{query.original_query}"

Intent: {query.intent.value}
Complexity: {query.complexity.value}
Extracted entities: {entities_str if entities_str else "none"}

Strategy: {strategy}
{instructions}

Requirements:
- Generate 2-5 sub-questions
- Make each sub-question specific and answerable
- Maintain legal terminology
- Consider both contract and law documents
- Order by logical dependency (prerequisites first)

Output format (numbered list):
1. [Sub-question 1]
2. [Sub-question 2]
...

Sub-questions:"""

        return prompt

    def _parse_sub_questions(self, response: str) -> List[str]:
        """Parse numbered list of sub-questions."""
        lines = response.strip().split("\n")
        sub_questions = []

        for line in lines:
            # Match numbered list items
            match = re.match(r"^\s*\d+\.\s*(.+)$", line)
            if match:
                sub_questions.append(match.group(1).strip())

        return sub_questions

    def _assign_strategies(
        self,
        sub_questions: List[str],
        original_query: ProcessedQuery
    ) -> List[SubQuery]:
        """
        Assign retrieval strategies to each sub-question.
        """
        sub_queries = []

        for i, sq_text in enumerate(sub_questions):
            # Determine target documents
            target_docs = self._infer_target_documents(sq_text)

            # Determine retrieval strategy
            retrieval_strategy = self._infer_retrieval_strategy(sq_text, original_query)

            # Determine priority (earlier = higher)
            priority = i + 1

            # Check for dependencies
            depends_on = self._infer_dependencies(i, sub_questions)

            sub_query = SubQuery(
                sub_query_id=f"sq_{i+1}",
                text=sq_text,
                intent=self._classify_sub_query_intent(sq_text),
                priority=priority,
                depends_on=depends_on,
                retrieval_strategy=retrieval_strategy,
                target_document_types=target_docs,
                focus_entities=original_query.entities
            )

            sub_queries.append(sub_query)

        return sub_queries

    def _infer_target_documents(self, sub_question: str) -> List[str]:
        """Infer which document types to search."""
        sq_lower = sub_question.lower()

        targets = []

        if any(word in sq_lower for word in ["smlouva", "contract", "článek"]):
            targets.append("contract")

        if any(word in sq_lower for word in ["zákon", "law", "§", "paragraf"]):
            targets.append("law_code")

        # Default: search both
        if not targets:
            targets = ["contract", "law_code"]

        return targets

    def _infer_retrieval_strategy(
        self,
        sub_question: str,
        original_query: ProcessedQuery
    ) -> str:
        """Infer retrieval strategy for sub-question."""
        sq_lower = sub_question.lower()

        # Cross-document if comparing or checking compliance
        if any(word in sq_lower for word in ["porovnej", "compare", "odpovídá", "complies", "konflikt"]):
            return "cross_document"

        # Graph-aware if asking about relationships
        if any(word in sq_lower for word in ["souvisí", "related", "reference", "navazuje"]):
            return "graph_aware"

        # Default: hybrid
        return "hybrid"

    def _infer_dependencies(
        self,
        index: int,
        sub_questions: List[str]
    ) -> List[str]:
        """
        Infer if this sub-question depends on previous ones.
        Simple heuristic: comparison questions depend on descriptive questions.
        """
        if index == 0:
            return []

        sq_lower = sub_questions[index].lower()

        # If this is a comparison/analysis, it depends on previous descriptive questions
        if any(word in sq_lower for word in ["compare", "porovnej", "analyze", "konflikt", "rozdíl"]):
            # Depends on all previous questions
            return [f"sq_{i+1}" for i in range(index)]

        return []

    def _classify_sub_query_intent(self, sub_question: str) -> QueryIntent:
        """Quick classification of sub-query intent."""
        sq_lower = sub_question.lower()

        if any(word in sq_lower for word in ["what", "co je", "jaké jsou"]):
            return QueryIntent.FACTUAL

        if any(word in sq_lower for word in ["compare", "porovnej"]):
            return QueryIntent.COMPARISON

        if any(word in sq_lower for word in ["all", "všechny", "list"]):
            return QueryIntent.ENUMERATION

        return QueryIntent.FACTUAL
```

---

## 7. Query Expander

### 7.1 Expansion Strategies

**Synonym Expansion**: Add Czech/English synonyms
- povinnost → závazek, duty, obligation
- zákaz → prohibice, prohibition

**Legal Term Expansion**: Related legal concepts
- smlouva → kontrakt, dohoda, ujednání
- odpovědnost → liability, ručení

**Contextual Expansion**: Domain-specific terms
- dodavatel → kontraktory, supplier, zhotovitel

### 7.2 Implementation

```python
class QueryExpander:
    """Expand query with synonyms and related terms."""

    def __init__(self):
        # Load synonym dictionary
        self.synonyms = self._load_synonyms()

    def expand(self, query: str, entities: List[ExtractedEntity]) -> Dict[str, List[str]]:
        """
        Expand query terms with synonyms.

        Returns:
            Dict mapping original terms to expansions
        """
        # Extract key terms (nouns, legal terms)
        key_terms = self._extract_key_terms(query, entities)

        # Expand each term
        expanded = {}
        for term in key_terms:
            synonyms = self._get_synonyms(term)
            if synonyms:
                expanded[term] = synonyms

        return expanded

    def _extract_key_terms(
        self,
        query: str,
        entities: List[ExtractedEntity]
    ) -> Set[str]:
        """Extract key terms worth expanding."""
        # Start with entity values
        key_terms = {e.value for e in entities if e.entity_type not in ["date", "legal_ref"]}

        # Add nouns (simple heuristic: words 5+ chars, not stopwords)
        stopwords = {"který", "který", "která", "které", "tento", "tato", "jsou"}
        words = query.lower().split()

        for word in words:
            if len(word) >= 5 and word not in stopwords:
                key_terms.add(word)

        return key_terms

    def _get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for term."""
        term_lower = term.lower()
        return self.synonyms.get(term_lower, [])

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym dictionary."""
        # Hardcoded for demo; would load from file
        return {
            "smlouva": ["kontrakt", "dohoda", "ujednání", "contract"],
            "zákon": ["legislation", "předpis", "legal code"],
            "povinnost": ["závazek", "obligation", "duty"],
            "zákaz": ["prohibice", "prohibition", "ban"],
            "dodavatel": ["kontraktorem", "supplier", "zhotovitel"],
            "odpovědnost": ["liability", "ručení", "accountability"],
            "termín": ["lhůta", "deadline", "time limit"],
            "sankce": ["penalty", "pokuta", "trest"]
        }
```

---

## 8. Query Processor (Main Orchestrator)

```python
import asyncio
import time
from anthropic import Anthropic

class QueryProcessor:
    """Main query processing pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize components
        llm_client = Anthropic(api_key=config["claude_api_key"])

        self.classifier = QueryClassifier(llm_client)
        self.entity_extractor = EntityExtractor()
        self.decomposer = LegalQuestionDecomposer(llm_client)
        self.expander = QueryExpander()

        self.logger = logging.getLogger(__name__)

    async def process(self, query: str) -> ProcessedQuery:
        """
        Process user query through full pipeline.

        Returns:
            ProcessedQuery with all analysis
        """
        start_time = time.time()

        self.logger.info(f"Processing query: {query}")

        # 1. Classify intent and complexity
        intent, complexity = await self.classifier.classify(query)
        self.logger.info(f"Classified as {intent.value} (complexity: {complexity.value})")

        # 2. Extract entities
        entities = self.entity_extractor.extract(query)
        self.logger.info(f"Extracted {len(entities)} entities")

        # 3. Build ProcessedQuery object
        processed_query = ProcessedQuery(
            original_query=query,
            intent=intent,
            complexity=complexity,
            requires_decomposition=False,
            entities=entities,
            sub_queries=[],
            expanded_terms={},
            retrieval_strategy="hybrid"
        )

        # 4. Decompose if needed
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            sub_queries = await self.decomposer.decompose(processed_query)

            if sub_queries:
                processed_query.requires_decomposition = True
                processed_query.sub_queries = sub_queries
                processed_query.decomposition_strategy = self.decomposer._select_strategy(intent)
                self.logger.info(f"Decomposed into {len(sub_queries)} sub-questions")

        # 5. Expand query terms
        expanded_terms = self.expander.expand(query, entities)
        processed_query.expanded_terms = expanded_terms
        if expanded_terms:
            self.logger.info(f"Expanded {len(expanded_terms)} terms")

        # 6. Select retrieval strategy
        processed_query.retrieval_strategy = self._select_retrieval_strategy(processed_query)

        processed_query.processing_time = time.time() - start_time
        self.logger.info(f"Query processing complete in {processed_query.processing_time:.2f}s")

        return processed_query

    def _select_retrieval_strategy(self, query: ProcessedQuery) -> str:
        """Select retrieval strategy based on query characteristics."""
        # Cross-document for compliance queries
        if query.intent in [
            QueryIntent.GAP_ANALYSIS,
            QueryIntent.CONFLICT_DETECTION,
            QueryIntent.COMPLIANCE_CHECK,
            QueryIntent.COMPARISON
        ]:
            return "cross_document"

        # Graph-aware for relationship queries
        if query.intent == QueryIntent.RELATIONSHIP:
            return "graph_aware"

        # Default: hybrid
        return "hybrid"
```

---

## 9. Configuration

### config.yaml

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
  auto_select_strategy: true  # Automatically choose retrieval strategy

  # Logging
  verbose_logging: false
```

---

## 10. Usage Examples

### 10.1 Basic Processing

```python
from src.query_processor import QueryProcessor

# Initialize
processor = QueryProcessor(config)

# Process query
query = "Najdi všechna slabá místa ve smlouvě, která se neshodují se zákonem č. 89/2012 Sb."
processed = await processor.process(query)

# Inspect results
print(f"Intent: {processed.intent}")
print(f"Complexity: {processed.complexity}")
print(f"Entities: {[e.normalized for e in processed.entities]}")

if processed.requires_decomposition:
    print(f"\nSub-queries ({len(processed.sub_queries)}):")
    for sq in processed.sub_queries:
        print(f"  {sq.priority}. {sq.text}")
        print(f"     Strategy: {sq.retrieval_strategy}, Targets: {sq.target_document_types}")
```

### 10.2 Compliance Query

```python
query = "Je smlouva o výstavbě v souladu s požadavky zákona č. 89/2012 Sb.?"
processed = await processor.process(query)

# Output:
# Intent: COMPLIANCE_CHECK
# Complexity: COMPLEX
# Requires decomposition: True
#
# Sub-queries:
#   1. Jaké jsou požadavky zákona č. 89/2012 Sb.?
#      Strategy: hybrid, Targets: ['law_code']
#   2. Co specifikuje smlouva o výstavbě?
#      Strategy: hybrid, Targets: ['contract']
#   3. Splňuje smlouva všechny zákonné požadavky?
#      Strategy: cross_document, Targets: ['contract', 'law_code']
```

### 10.3 Gap Analysis Query

```python
query = "Které povinné body ze zákona chybí v této smlouvě?"
processed = await processor.process(query)

# Output:
# Intent: GAP_ANALYSIS
# Decomposition strategy: requirement_based
#
# Sub-queries:
#   1. Jaké jsou povinné body podle zákona?
#   2. Které body obsahuje smlouva?
#   3. Které povinné body ve smlouvě chybí?
```

---

## 11. Testing

### 11.1 Unit Tests

```python
import pytest

@pytest.mark.asyncio
async def test_query_classification():
    """Test intent classification."""
    classifier = QueryClassifier(llm_client)

    # Conflict detection
    query = "Najdi konflikty mezi smlouvou a zákonem"
    intent, complexity = await classifier.classify(query)
    assert intent == QueryIntent.CONFLICT_DETECTION

    # Gap analysis
    query = "Co chybí ve smlouvě oproti zákonu?"
    intent, complexity = await classifier.classify(query)
    assert intent == QueryIntent.GAP_ANALYSIS

def test_entity_extraction():
    """Test legal reference extraction."""
    extractor = LegalReferenceExtractor()

    query = "Podle §89 odst. 2 písm. a) zákona č. 89/2012 Sb."
    entities = extractor.extract(query)

    assert len(entities) == 2
    assert entities[0].normalized == "§89 odst. 2 písm. a)"
    assert entities[1].normalized == "Zákon č. 89/2012 Sb."

@pytest.mark.asyncio
async def test_query_decomposition():
    """Test question decomposition."""
    decomposer = LegalQuestionDecomposer(llm_client)

    processed_query = ProcessedQuery(
        original_query="Najdi všechny konflikty",
        intent=QueryIntent.CONFLICT_DETECTION,
        complexity=QueryComplexity.COMPLEX,
        requires_decomposition=True,
        entities=[]
    )

    sub_queries = await decomposer.decompose(processed_query)

    assert len(sub_queries) >= 2
    assert any("contract" in sq.text.lower() for sq in sub_queries)
    assert any("law" in sq.text.lower() or "zákon" in sq.text.lower() for sq in sub_queries)
```

### 11.2 Integration Tests

```python
@pytest.mark.asyncio
async def test_full_query_processing():
    """Test complete query processing pipeline."""
    processor = QueryProcessor(config)

    query = "Je smlouva v souladu s §89?"
    processed = await processor.process(query)

    # Check classification
    assert processed.intent == QueryIntent.COMPLIANCE_CHECK

    # Check entity extraction
    assert any(e.entity_type == "legal_ref" for e in processed.entities)

    # Check decomposition (if complex)
    if processed.requires_decomposition:
        assert len(processed.sub_queries) > 0
        assert processed.retrieval_strategy == "cross_document"
```

---

## 12. Performance Considerations

### 12.1 Optimization Strategies

**Caching**:
```python
# Cache LLM classification results
from functools import lru_cache

@lru_cache(maxsize=1000)
async def classify_cached(query: str) -> tuple[QueryIntent, QueryComplexity]:
    return await classifier.classify(query)
```

**Parallel Processing**:
```python
# Run classification and entity extraction in parallel
intent_task = classifier.classify(query)
entity_task = asyncio.to_thread(entity_extractor.extract, query)

intent, entities = await asyncio.gather(intent_task, entity_task)
```

**Selective Decomposition**:
```python
# Skip decomposition for simple queries to save LLM calls
if complexity == QueryComplexity.SIMPLE:
    return []  # No decomposition
```

### 12.2 Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Classification | <0.5s | Pattern-based |
| Classification (LLM) | <1.5s | Haiku call |
| Entity extraction | <0.1s | Regex-based |
| Decomposition | <2s | Haiku call, 2-5 sub-queries |
| Query expansion | <0.1s | Dictionary lookup |
| **Full pipeline** | **<3s** | Including all components |

---

## 13. Future Enhancements

### 13.1 Advanced Techniques

**Few-Shot Learning for Classification**:
- Provide examples in prompt for better intent detection
- Maintain example bank for each intent type

**Dependency Graph for Sub-Queries**:
- Build explicit DAG of sub-query dependencies
- Execute in parallel where possible

**Query Rewriting**:
- Rewrite ambiguous queries for clarity
- Expand abbreviations, resolve pronouns

**Multi-Turn Query Understanding**:
- Maintain conversation context
- Resolve anaphora ("it", "that", "these")

### 13.2 Domain-Specific Improvements

**Legal Ontology Integration**:
- Use legal taxonomy for term expansion
- Recognize hierarchical relationships (obligation > specific obligation types)

**Contract-Specific Templates**:
- Recognize standard contract clauses
- Map to legal requirements automatically

**Compliance Query Patterns**:
- Pre-built templates for common compliance checks
- "Does X comply with Y?" → standardized decomposition

---

## 14. Summary

**Key Takeaways**:

1. **Query Classification**: Pattern-based + LLM for accurate intent detection
2. **Entity Extraction**: Regex patterns for Czech legal references (§, články, zákony)
3. **Question Decomposition**: Break complex compliance queries into focused sub-questions
4. **Strategy Selection**: Automatic routing to appropriate retrieval strategies
5. **Query Expansion**: Synonyms and related terms for better recall

**Integration with Pipeline**:
```
QueryProcessor → ProcessedQuery → HybridRetriever / ComparativeRetriever → Reranker → ComplianceAnalyzer
```

**Next Specifications**:
- See [09. Compliance Analyzer](09_compliance_analyzer.md) for using processed queries
- See [05. Hybrid Retrieval](05_hybrid_retrieval.md) for retrieval strategies
- See [06. Cross-Document Retrieval](06_cross_document_retrieval.md) for compliance checks

---

**Page Count**: ~18 pages
**Last Updated**: 2025-10-08
**Status**: Complete ✅
