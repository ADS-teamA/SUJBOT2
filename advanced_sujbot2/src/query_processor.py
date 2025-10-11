"""
Query Processing Module for Legal Compliance Analysis

This module provides comprehensive query processing capabilities including:
- Intent classification and complexity assessment
- Entity extraction (legal references, dates, obligations)
- Question decomposition for complex queries
- Query expansion with synonyms
- Retrieval strategy selection

Author: Advanced SUJBOT2
Date: 2025-10-08
"""

import re
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class QueryIntent(Enum):
    """High-level query intent."""
    # Compliance analysis
    GAP_ANALYSIS = "gap_analysis"
    CONFLICT_DETECTION = "conflict_detection"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"

    # Information retrieval
    FACTUAL = "factual"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    EXPLANATION = "explanation"

    # Analytical
    ENUMERATION = "enumeration"
    RELATIONSHIP = "relationship"
    CONSEQUENCE = "consequence"


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
    span: Tuple[int, int] = (0, 0)  # Character span in query


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
    target_document_types: List[str] = field(default_factory=list)
    focus_entities: List[ExtractedEntity] = field(default_factory=list)

    # Execution state
    executed: bool = False
    result: Optional[str] = None


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
    sub_queries: List[SubQuery] = field(default_factory=list)

    # Query expansion
    expanded_terms: Dict[str, List[str]] = field(default_factory=dict)

    # Retrieval strategy
    retrieval_strategy: str = "hybrid"

    # Metadata
    processing_time: float = 0.0
    decomposition_strategy: Optional[str] = None


# ============================================================================
# Query Classifier
# ============================================================================

# Intent detection patterns
INTENT_PATTERNS = {
    QueryIntent.GAP_ANALYSIS: [
        r"chybí|scházející|nepokryto|nevyhovuje",
        r"gap|missing|absent",
        r"nesplňuje požadavky",
        r"které.*chybí"
    ],
    QueryIntent.CONFLICT_DETECTION: [
        r"konflikt|rozpor|nesoulad|neshoduje se",
        r"conflict|contradiction|inconsistent",
        r"odchyluje se od|porušuje",
        r"v rozporu"
    ],
    QueryIntent.RISK_ASSESSMENT: [
        r"riziko|nebezpečí|hrozba",
        r"risk|danger|vulnerability",
        r"slabá místa|problematické"
    ],
    QueryIntent.COMPLIANCE_CHECK: [
        r"v souladu|splňuje|odpovídá",
        r"complies|compliance|conforms",
        r"je.*v souladu"
    ],
    QueryIntent.COMPARISON: [
        r"porovnej|srovnej|rozdíl",
        r"compare|versus|vs\.?|difference",
        r"oproti|na rozdíl"
    ],
    QueryIntent.ENUMERATION: [
        r"všechny?|vyjmenuj|seznam",
        r"list all|enumerate|find all",
        r"které.*jsou"
    ],
    QueryIntent.DEFINITION: [
        r"co je|definice|význam",
        r"what is|define|meaning"
    ],
    QueryIntent.EXPLANATION: [
        r"vysvětli|jak funguje|proč",
        r"explain|how does|why"
    ]
}


class QueryClassifier:
    """Classify query intent and complexity."""

    def __init__(self, llm_client: AsyncAnthropic, config: Dict[str, Any]):
        self.llm = llm_client
        self.config = config
        self.intent_patterns = INTENT_PATTERNS
        self.model = config.get("query_processing", {}).get("llm_model", "claude-3-5-haiku-20241022")
        self.temperature = config.get("query_processing", {}).get("llm_temperature", 0.0)
        self.max_tokens = config.get("query_processing", {}).get("llm_max_tokens", 50)

    async def classify(self, query: str) -> Tuple[QueryIntent, QueryComplexity]:
        """
        Classify query intent and complexity.

        Args:
            query: User query string

        Returns:
            Tuple of (intent, complexity)
        """
        # First: fast pattern-based classification
        pattern_intent = self._classify_by_patterns(query)

        if pattern_intent and self._is_high_confidence_match(query, pattern_intent):
            intent = pattern_intent
            logger.debug(f"Pattern-based classification: {intent.value}")
        else:
            # Fallback: LLM-based classification
            intent = await self._classify_with_llm(query)
            logger.debug(f"LLM-based classification: {intent.value}")

        # Determine complexity
        complexity = self._assess_complexity(query)
        logger.debug(f"Complexity assessed as: {complexity.value}")

        return intent, complexity

    def _classify_by_patterns(self, query: str) -> Optional[QueryIntent]:
        """Match query against intent patterns."""
        query_lower = query.lower()

        # Score each intent based on pattern matches
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        # Return highest scoring intent
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]

        return None

    def _is_high_confidence_match(self, query: str, intent: QueryIntent) -> bool:
        """Check if pattern match is confident enough."""
        query_lower = query.lower()
        patterns = self.intent_patterns[intent]

        match_count = sum(
            1 for pattern in patterns
            if re.search(pattern, query_lower)
        )

        return match_count >= 2

    async def _classify_with_llm(self, query: str) -> QueryIntent:
        """Classify using Claude Haiku for nuanced understanding."""
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

        try:
            response = await self.llm.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            intent_str = response.content[0].text.strip().lower()

            # Map to enum
            try:
                return QueryIntent(intent_str)
            except ValueError:
                logger.warning(f"Unknown intent from LLM: {intent_str}, defaulting to FACTUAL")
                return QueryIntent.FACTUAL

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return QueryIntent.FACTUAL

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity based on linguistic features."""
        # Heuristics
        word_count = len(query.split())
        has_multiple_questions = query.count("?") > 1 or " a " in query.lower()
        has_comparison = any(word in query.lower() for word in ["porovnej", "compare", "rozdíl", "vs", "oproti"])
        has_enumeration = any(word in query.lower() for word in ["všechny", "all", "list", "vyjmenuj"])
        has_conditional = any(word in query.lower() for word in ["pokud", "if", "when", "když"])
        has_multi_entity = query.count("§") > 1 or (query.count("zákon") > 1)

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

        if has_multi_entity:
            complexity_score += 1

        # Map score to complexity
        if complexity_score >= 5:
            return QueryComplexity.EXPERT
        elif complexity_score >= 3:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE


# ============================================================================
# Entity Extractors
# ============================================================================

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
                elif ref_type == "paragraph_simple":
                    para_num = match.group(1)
                    normalized = f"§{para_num}"
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

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        query_config = config.get("query_processing", {})

        self.extract_legal_refs = query_config.get("extract_legal_references", True)
        self.extract_temporal = query_config.get("extract_temporal_entities", True)
        self.extract_obligations = query_config.get("extract_obligations", True)

        self.legal_ref_extractor = LegalReferenceExtractor()
        self.temporal_extractor = TemporalExtractor()
        self.obligation_extractor = ObligationExtractor()

    def extract(self, query: str) -> List[ExtractedEntity]:
        """Extract all entities from query."""
        entities = []

        if self.extract_legal_refs:
            entities.extend(self.legal_ref_extractor.extract(query))

        if self.extract_temporal:
            entities.extend(self.temporal_extractor.extract(query))

        if self.extract_obligations:
            entities.extend(self.obligation_extractor.extract(query))

        # Sort by span position and remove duplicates
        entities.sort(key=lambda e: e.span[0])

        # Remove overlapping entities (keep higher confidence)
        filtered_entities = []
        for entity in entities:
            if not any(self._overlaps(entity, existing) for existing in filtered_entities):
                filtered_entities.append(entity)

        return filtered_entities

    def _overlaps(self, e1: ExtractedEntity, e2: ExtractedEntity) -> bool:
        """Check if two entities overlap in text."""
        return not (e1.span[1] <= e2.span[0] or e2.span[1] <= e1.span[0])


# ============================================================================
# Legal Question Decomposer
# ============================================================================

class LegalQuestionDecomposer:
    """Decompose complex legal queries into sub-questions."""

    def __init__(self, llm_client: AsyncAnthropic, config: Dict[str, Any]):
        self.llm = llm_client
        self.config = config
        self.model = config.get("query_processing", {}).get("llm_model", "claude-3-5-haiku-20241022")
        self.temperature = config.get("query_processing", {}).get("llm_temperature", 0.3)
        self.max_tokens = config.get("query_processing", {}).get("llm_max_tokens", 1000)
        self.max_sub_queries = config.get("query_processing", {}).get("max_sub_queries", 5)

    async def decompose(self, query: ProcessedQuery) -> List[SubQuery]:
        """
        Decompose query into sub-questions if needed.

        Args:
            query: Processed query object

        Returns:
            List of SubQuery objects
        """
        # Simple queries don't need decomposition
        if not self._requires_decomposition(query):
            return []

        # Select decomposition strategy
        strategy = self._select_strategy(query.intent)
        logger.info(f"Using decomposition strategy: {strategy}")

        # Generate sub-questions using LLM
        sub_questions = await self._generate_sub_questions(query, strategy)

        # Limit number of sub-queries
        sub_questions = sub_questions[:self.max_sub_queries]

        # Assign retrieval strategies
        sub_queries = self._assign_strategies(sub_questions, query)

        logger.info(f"Generated {len(sub_queries)} sub-queries")
        return sub_queries

    def _requires_decomposition(self, query: ProcessedQuery) -> bool:
        """Determine if query needs decomposition."""
        # Check if decomposition is enabled
        if not self.config.get("query_processing", {}).get("enable_decomposition", True):
            return False

        # Always decompose compliance queries
        if query.intent in [
            QueryIntent.GAP_ANALYSIS,
            QueryIntent.CONFLICT_DETECTION,
            QueryIntent.RISK_ASSESSMENT,
            QueryIntent.COMPLIANCE_CHECK
        ]:
            return True

        # Decompose complex comparisons
        if query.intent == QueryIntent.COMPARISON and query.complexity != QueryComplexity.SIMPLE:
            return True

        # Decompose enumerations with analysis
        if query.intent == QueryIntent.ENUMERATION and "analyz" in query.original_query.lower():
            return True

        # Decompose based on complexity
        threshold = self.config.get("query_processing", {}).get("decomposition_complexity_threshold", "moderate")
        complexity_levels = {
            "simple": [QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            "moderate": [QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            "complex": [QueryComplexity.EXPERT]
        }

        return query.complexity in complexity_levels.get(threshold, [])

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
        """Generate sub-questions using Claude Haiku."""
        prompt = self._build_decomposition_prompt(query, strategy)

        try:
            response = await self.llm.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response (expects numbered list)
            sub_questions = self._parse_sub_questions(response.content[0].text)

            return sub_questions

        except Exception as e:
            logger.error(f"Error generating sub-questions: {e}")
            return []

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

            "clause_by_clause": """Break down compliance check:
1. What are the key legal requirements?
2. What are the corresponding contract clauses?
3. Do they meet the requirements?""",

            "generic": """Break down into logical sub-questions that:
- Can be answered independently
- Together cover the full query
- Are specific and focused"""
        }

        instructions = strategy_instructions.get(strategy, strategy_instructions["generic"])

        # Extract entity context
        entities_str = ", ".join([e.normalized or e.value for e in query.entities]) if query.entities else "none"

        prompt = f"""Decompose this complex legal query into focused sub-questions.

Query: "{query.original_query}"

Intent: {query.intent.value}
Complexity: {query.complexity.value}
Extracted entities: {entities_str}

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
                sub_question = match.group(1).strip()
                # Remove quotes if present
                sub_question = sub_question.strip('"').strip("'")
                sub_questions.append(sub_question)

        return sub_questions

    def _assign_strategies(
        self,
        sub_questions: List[str],
        original_query: ProcessedQuery
    ) -> List[SubQuery]:
        """Assign retrieval strategies to each sub-question."""
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

        if any(word in sq_lower for word in ["smlouva", "contract", "článek", "klauzule"]):
            targets.append("contract")

        if any(word in sq_lower for word in ["zákon", "law", "§", "paragraf", "předpis"]):
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
        if any(word in sq_lower for word in ["porovnej", "compare", "odpovídá", "complies", "konflikt", "soulad"]):
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
        if any(word in sq_lower for word in ["compare", "porovnej", "analyz", "konflikt", "rozdíl", "soulad"]):
            # Depends on all previous questions
            return [f"sq_{i+1}" for i in range(index)]

        return []

    def _classify_sub_query_intent(self, sub_question: str) -> QueryIntent:
        """Quick classification of sub-query intent."""
        sq_lower = sub_question.lower()

        if any(word in sq_lower for word in ["what", "co je", "jaké jsou", "které jsou"]):
            return QueryIntent.FACTUAL

        if any(word in sq_lower for word in ["compare", "porovnej", "rozdíl"]):
            return QueryIntent.COMPARISON

        if any(word in sq_lower for word in ["all", "všechny", "list", "vyjmenuj"]):
            return QueryIntent.ENUMERATION

        if any(word in sq_lower for word in ["konflikt", "rozpor", "nesoulad"]):
            return QueryIntent.CONFLICT_DETECTION

        if any(word in sq_lower for word in ["soulad", "splňuje", "odpovídá"]):
            return QueryIntent.COMPLIANCE_CHECK

        return QueryIntent.FACTUAL


# ============================================================================
# Query Expander
# ============================================================================

class QueryExpander:
    """Expand query with synonyms and related terms."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_synonyms = config.get("query_processing", {}).get("max_synonyms_per_term", 3)
        # Load synonym dictionary
        self.synonyms = self._load_synonyms()

    def expand(self, query: str, entities: List[ExtractedEntity]) -> Dict[str, List[str]]:
        """
        Expand query terms with synonyms.

        Args:
            query: Query string
            entities: Extracted entities

        Returns:
            Dict mapping original terms to expansions
        """
        # Check if expansion is enabled
        if not self.config.get("query_processing", {}).get("enable_query_expansion", True):
            return {}

        # Extract key terms (nouns, legal terms)
        key_terms = self._extract_key_terms(query, entities)

        # Expand each term
        expanded = {}
        for term in key_terms:
            synonyms = self._get_synonyms(term)
            if synonyms:
                expanded[term] = synonyms[:self.max_synonyms]

        return expanded

    def _extract_key_terms(
        self,
        query: str,
        entities: List[ExtractedEntity]
    ) -> Set[str]:
        """Extract key terms worth expanding."""
        # Start with entity values (excluding legal refs and dates)
        key_terms = {
            e.value.lower() for e in entities
            if e.entity_type not in ["date", "legal_ref"]
        }

        # Add nouns (simple heuristic: words 5+ chars, not stopwords)
        stopwords = {
            "který", "která", "které", "tento", "tato", "jsou",
            "bude", "bylo", "byly", "byly", "jsem", "jste"
        }
        words = query.lower().split()

        for word in words:
            # Clean word
            word = re.sub(r'[^\w\s]', '', word)
            if len(word) >= 5 and word not in stopwords:
                key_terms.add(word)

        return key_terms

    def _get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for term."""
        term_lower = term.lower()
        return self.synonyms.get(term_lower, [])

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym dictionary."""
        # Hardcoded for demo; would load from file in production
        return {
            "smlouva": ["kontrakt", "dohoda", "ujednání", "contract"],
            "zákon": ["legislation", "předpis", "legal code"],
            "povinnost": ["závazek", "obligation", "duty"],
            "zákaz": ["prohibice", "prohibition", "ban"],
            "dodavatel": ["kontraktorem", "supplier", "zhotovitel"],
            "odpovědnost": ["liability", "ručení", "accountability"],
            "termín": ["lhůta", "deadline", "time limit"],
            "sankce": ["penalty", "pokuta", "trest"],
            "požadavek": ["requirement", "nárok", "podmínka"],
            "ustanovení": ["provision", "klauzule", "clause"],
            "soulad": ["compliance", "shoda", "conformity"],
            "konflikt": ["rozpor", "nesoulad", "contradiction"],
            "riziko": ["risk", "nebezpečí", "hrozba"]
        }


# ============================================================================
# Query Processor (Main Orchestrator)
# ============================================================================

class QueryProcessor:
    """Main query processing pipeline."""

    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize query processor.

        Args:
            config: Configuration dictionary
            api_key: Optional API key (uses env var if not provided)
        """
        self.config = config

        # Initialize LLM client
        import os
        api_key = api_key or os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise ValueError("API key required. Set CLAUDE_API_KEY environment variable.")

        llm_client = AsyncAnthropic(api_key=api_key)

        # Initialize components
        self.classifier = QueryClassifier(llm_client, config)
        self.entity_extractor = EntityExtractor(config)
        self.decomposer = LegalQuestionDecomposer(llm_client, config)
        self.expander = QueryExpander(config)

        self.logger = logging.getLogger(__name__)
        self.verbose = config.get("query_processing", {}).get("verbose_logging", False)

    async def process(self, query: str) -> ProcessedQuery:
        """
        Process user query through full pipeline.

        Args:
            query: User query string

        Returns:
            ProcessedQuery with all analysis
        """
        start_time = time.time()

        if self.verbose:
            self.logger.info(f"Processing query: {query}")

        # 1. Classify intent and complexity (in parallel with entity extraction)
        classify_task = self.classifier.classify(query)
        entities = self.entity_extractor.extract(query)

        intent, complexity = await classify_task

        if self.verbose:
            self.logger.info(f"Classified as {intent.value} (complexity: {complexity.value})")
            self.logger.info(f"Extracted {len(entities)} entities")

        # 2. Build ProcessedQuery object
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

        # 3. Decompose if needed
        sub_queries = await self.decomposer.decompose(processed_query)

        if sub_queries:
            processed_query.requires_decomposition = True
            processed_query.sub_queries = sub_queries
            processed_query.decomposition_strategy = self.decomposer._select_strategy(intent)
            if self.verbose:
                self.logger.info(f"Decomposed into {len(sub_queries)} sub-questions")

        # 4. Expand query terms
        expanded_terms = self.expander.expand(query, entities)
        processed_query.expanded_terms = expanded_terms
        if expanded_terms and self.verbose:
            self.logger.info(f"Expanded {len(expanded_terms)} terms")

        # 5. Select retrieval strategy
        processed_query.retrieval_strategy = self._select_retrieval_strategy(processed_query)

        processing_time = time.time() - start_time
        processed_query.processing_time = processing_time

        if self.verbose:
            self.logger.info(f"Query processing complete in {processing_time:.2f}s")

        return processed_query

    def _select_retrieval_strategy(self, query: ProcessedQuery) -> str:
        """Select retrieval strategy based on query characteristics."""
        # Check if auto-selection is enabled
        if not self.config.get("query_processing", {}).get("auto_select_strategy", True):
            return "hybrid"

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


# ============================================================================
# Utility Functions
# ============================================================================

def print_processed_query(pq: ProcessedQuery) -> None:
    """Pretty print a processed query for debugging."""
    print(f"\n{'='*60}")
    print(f"Query: {pq.original_query}")
    print(f"{'='*60}")
    print(f"Intent: {pq.intent.value}")
    print(f"Complexity: {pq.complexity.value}")
    print(f"Retrieval Strategy: {pq.retrieval_strategy}")
    print(f"Processing Time: {pq.processing_time:.2f}s")

    if pq.entities:
        print(f"\nEntities ({len(pq.entities)}):")
        for entity in pq.entities:
            print(f"  - {entity.entity_type}: {entity.value} → {entity.normalized}")

    if pq.expanded_terms:
        print(f"\nExpanded Terms ({len(pq.expanded_terms)}):")
        for term, synonyms in pq.expanded_terms.items():
            print(f"  - {term}: {', '.join(synonyms)}")

    if pq.requires_decomposition and pq.sub_queries:
        print(f"\nSub-queries ({len(pq.sub_queries)}):")
        print(f"Decomposition Strategy: {pq.decomposition_strategy}")
        for sq in pq.sub_queries:
            print(f"\n  {sq.priority}. {sq.text}")
            print(f"     ID: {sq.sub_query_id}")
            print(f"     Intent: {sq.intent.value}")
            print(f"     Strategy: {sq.retrieval_strategy}")
            print(f"     Targets: {sq.target_document_types}")
            if sq.depends_on:
                print(f"     Depends on: {', '.join(sq.depends_on)}")

    print(f"{'='*60}\n")
