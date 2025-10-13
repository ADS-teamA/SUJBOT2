"""
Intent Router Agent - Intelligent Message Router for Legal Compliance System

This agent automatically analyzes user messages and routes them to the appropriate
pipeline based on intent classification. It leverages existing indexed documents
and RAG infrastructure for optimal performance.

Key Features:
- Automatic intent classification (simple query vs. compliance check)
- Document requirement analysis (which documents are needed)
- Pipeline selection (query, compliance, gap analysis)
- Parameter extraction for downstream services

Industry Best Practices (2025 AI Legal Tech Standards):
- Multi-layered discrepancy detection
- Automated conflict identification
- Risk severity scoring
- Recommendation generation
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


# ============================================================================
# Intent Types
# ============================================================================

class UserIntent(Enum):
    """User message intent classification"""

    # Simple interactions
    GREETING = "greeting"  # "ahoj", "dobrý den"
    SIMPLE_QUERY = "simple_query"  # "Co říká zákon o..."

    # Compliance operations
    COMPLIANCE_CHECK = "compliance_check"  # "Je smlouva v souladu se zákonem?"
    GAP_ANALYSIS = "gap_analysis"  # "Co chybí ve smlouvě?"
    CONFLICT_DETECTION = "conflict_detection"  # "Kde jsou rozpory?"
    RISK_ASSESSMENT = "risk_assessment"  # "Jaká jsou rizika?"

    # Comparison operations
    CONTRACT_LAW_COMPARISON = "contract_law_comparison"  # "Porovnej smlouvu se zákonem"
    MULTI_DOCUMENT_COMPARISON = "multi_document_comparison"  # "Porovnej tyto smlouvy"

    # Information retrieval
    DEFINITION_LOOKUP = "definition_lookup"  # "Co je záruční doba?"
    REFERENCE_LOOKUP = "reference_lookup"  # "Najdi §89"

    # Unknown/unclear
    UNCLEAR = "unclear"


class PipelineRoute(Enum):
    """Target pipeline for processing"""
    SIMPLE_CHAT = "simple_chat"  # Standard RAG query
    COMPLIANCE_ANALYSIS = "compliance_analysis"  # Full compliance checking
    CROSS_DOCUMENT_QUERY = "cross_document_query"  # Cross-document retrieval
    GREETING_HANDLER = "greeting_handler"  # Simple response


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class DocumentRequirement:
    """Specifies which documents are needed"""
    contracts_required: bool = False
    laws_required: bool = False
    min_contracts: int = 0
    min_laws: int = 0
    specific_document_types: List[str] = field(default_factory=list)

    def is_satisfied_by(self, contract_ids: List[str], law_ids: List[str]) -> bool:
        """Check if available documents satisfy requirements"""
        if self.contracts_required and len(contract_ids) < self.min_contracts:
            return False
        if self.laws_required and len(law_ids) < self.min_laws:
            return False
        return True

    def get_missing_description(self, contract_ids: List[str], law_ids: List[str], language: str = "cs") -> Optional[str]:
        """Get description of what documents are missing"""
        missing = []

        if self.contracts_required and len(contract_ids) < self.min_contracts:
            if language == "cs":
                missing.append(f"smlouvu (potřeba alespoň {self.min_contracts})")
            else:
                missing.append(f"contract (need at least {self.min_contracts})")

        if self.laws_required and len(law_ids) < self.min_laws:
            if language == "cs":
                missing.append(f"zákon (potřeba alespoň {self.min_laws})")
            else:
                missing.append(f"law (need at least {self.min_laws})")

        if not missing:
            return None

        if language == "cs":
            return f"Pro tento dotaz musíte nahrát: {', '.join(missing)}"
        else:
            return f"For this query you need to upload: {', '.join(missing)}"


@dataclass
class RoutingDecision:
    """Result of intent routing"""
    # Intent classification
    intent: UserIntent
    pipeline: PipelineRoute
    confidence: float

    # Multi-intent support (for sequential execution)
    is_multi_intent: bool = False
    intents: List[UserIntent] = field(default_factory=list)  # Ordered list
    pipelines: List[PipelineRoute] = field(default_factory=list)  # Ordered list

    # Document requirements
    document_requirement: DocumentRequirement = field(default_factory=DocumentRequirement)

    # Extracted parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Explanation
    reasoning: str = ""
    user_message: Optional[str] = None  # Message to show user if needed


# ============================================================================
# Intent Classifier
# ============================================================================

class IntentClassifier:
    """Classifies user intent using LLM (Claude) for accurate multi-intent detection"""

    def __init__(self, llm_client: AsyncAnthropic, model: str = "claude-3-5-haiku-20241022"):
        self.llm = llm_client
        self.model = model

    async def classify(
        self,
        message: str,
        language: str = "cs"
    ) -> Tuple[List[UserIntent], List[float]]:
        """
        Classify user message intent using LLM.

        Can detect multiple intents in complex queries.

        Returns:
            (intents_list, confidences_list) - ordered by execution priority
        """
        return await self._llm_classify(message, language)

    async def _llm_classify(
        self,
        message: str,
        language: str
    ) -> Tuple[List[UserIntent], List[float]]:
        """LLM-based multi-intent classification"""

        if language == "cs":
            prompt = f"""Analyzuj tento dotaz uživatele a urči jeho intent(y):

"{message}"

Možné intenty:
- greeting: Pozdrav, smalltalk
- simple_query: Jednoduchý dotaz na informaci z dokumentů (Jak...? Co...? Kde...?)
- compliance_check: Kontrola, zda smlouva odpovídá zákonu (Je v souladu? Splňuje požadavky?)
- gap_analysis: Hledání chybějících požadavků (Co chybí? Co není pokryto?)
- conflict_detection: Hledání rozporů mezi dokumenty (Kde jsou rozpory?)
- risk_assessment: Analýza rizik (Jaká jsou rizika?)
- contract_law_comparison: Porovnání smlouvy se zákonem
- definition_lookup: Vyhledání definice (Co je...? Co znamená...?)
- reference_lookup: Vyhledání konkrétního paragrafu (Najdi §89)

DŮLEŽITÉ:
- Pokud dotaz obsahuje VÍCE intentů (např. "Jak plánuje stavět? Je to v souladu se zákonem?"),
  vrať VŠECHNY intenty v pořadí, v jakém by měly být provedeny.
- První intent je ten, který získává informaci, druhý intent ji porovnává/analyzuje.

Formát odpovědi:
intent1_name confidence1 [intent2_name confidence2] [...]

Příklady:
- "Je smlouva v souladu se zákonem?" → compliance_check 0.95
- "Jak plánuje stavět dům? Je to v souladu?" → simple_query 0.9 compliance_check 0.9
- "Co je §89?" → definition_lookup 0.95
- "Ahoj" → greeting 0.99"""
        else:
            prompt = f"""Analyze this user query and determine its intent(s):

"{message}"

Possible intents:
- greeting: Greeting, small talk
- simple_query: Simple query for information from documents (How...? What...? Where...?)
- compliance_check: Check if contract complies with law (Does it comply? Meets requirements?)
- gap_analysis: Find missing requirements (What's missing? What's not covered?)
- conflict_detection: Find conflicts between documents (Where are conflicts?)
- risk_assessment: Analyze risks (What are the risks?)
- contract_law_comparison: Compare contract with law
- definition_lookup: Look up definition (What is...? What does... mean?)
- reference_lookup: Look up specific paragraph (Find §89)

IMPORTANT:
- If the query contains MULTIPLE intents (e.g., "How do they plan to build? Is it compliant?"),
  return ALL intents in the order they should be executed.
- First intent gathers information, second intent compares/analyzes it.

Response format:
intent1_name confidence1 [intent2_name confidence2] [...]

Examples:
- "Is the contract compliant with law?" → compliance_check 0.95
- "How do they plan to build? Is it compliant?" → simple_query 0.9 compliance_check 0.9
- "What is §89?" → definition_lookup 0.95
- "Hello" → greeting 0.99"""

        try:
            response = await self.llm.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response: "intent1 conf1 intent2 conf2 ..."
            response_text = response.content[0].text.strip()
            parts = response_text.split()

            intents = []
            confidences = []

            # Parse pairs of (intent, confidence)
            i = 0
            while i < len(parts) - 1:
                intent_str = parts[i]
                try:
                    confidence = float(parts[i + 1])

                    # Map to enum
                    try:
                        intent = UserIntent(intent_str)
                        intents.append(intent)
                        confidences.append(confidence)
                    except ValueError:
                        logger.warning(f"Unknown intent from LLM: {intent_str}")

                    i += 2
                except (ValueError, IndexError):
                    # Not a valid confidence, skip
                    i += 1

            # If no valid intents parsed, return UNCLEAR
            if not intents:
                logger.warning(f"No valid intents parsed from LLM: {response_text}")
                return [UserIntent.UNCLEAR], [0.5]

            return intents, confidences

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return [UserIntent.UNCLEAR], [0.5]


# ============================================================================
# Document Analyzer
# ============================================================================

class DocumentAnalyzer:
    """Analyzes which documents are needed for a given intent"""

    REQUIREMENTS_MAP = {
        UserIntent.GREETING: DocumentRequirement(
            contracts_required=False,
            laws_required=False
        ),
        UserIntent.SIMPLE_QUERY: DocumentRequirement(
            contracts_required=False,
            laws_required=False,
            min_contracts=0,
            min_laws=0
        ),
        UserIntent.COMPLIANCE_CHECK: DocumentRequirement(
            contracts_required=True,
            laws_required=True,
            min_contracts=1,
            min_laws=1
        ),
        UserIntent.GAP_ANALYSIS: DocumentRequirement(
            contracts_required=True,
            laws_required=True,
            min_contracts=1,
            min_laws=1
        ),
        UserIntent.CONFLICT_DETECTION: DocumentRequirement(
            contracts_required=True,
            laws_required=True,
            min_contracts=1,
            min_laws=1
        ),
        UserIntent.RISK_ASSESSMENT: DocumentRequirement(
            contracts_required=True,
            laws_required=True,
            min_contracts=1,
            min_laws=1
        ),
        UserIntent.CONTRACT_LAW_COMPARISON: DocumentRequirement(
            contracts_required=True,
            laws_required=True,
            min_contracts=1,
            min_laws=1
        ),
        UserIntent.MULTI_DOCUMENT_COMPARISON: DocumentRequirement(
            contracts_required=False,
            laws_required=False,
            min_contracts=0,
            min_laws=0
        ),
        UserIntent.DEFINITION_LOOKUP: DocumentRequirement(
            contracts_required=False,
            laws_required=False
        ),
        UserIntent.REFERENCE_LOOKUP: DocumentRequirement(
            contracts_required=False,
            laws_required=False
        ),
        UserIntent.UNCLEAR: DocumentRequirement(
            contracts_required=False,
            laws_required=False
        ),
    }

    def get_requirements(self, intent: UserIntent) -> DocumentRequirement:
        """Get document requirements for an intent"""
        return self.REQUIREMENTS_MAP.get(intent, DocumentRequirement())


# ============================================================================
# Pipeline Router
# ============================================================================

class PipelineRouter:
    """Routes intent to appropriate pipeline"""

    INTENT_TO_PIPELINE = {
        UserIntent.GREETING: PipelineRoute.GREETING_HANDLER,
        UserIntent.SIMPLE_QUERY: PipelineRoute.SIMPLE_CHAT,
        UserIntent.COMPLIANCE_CHECK: PipelineRoute.COMPLIANCE_ANALYSIS,
        UserIntent.GAP_ANALYSIS: PipelineRoute.COMPLIANCE_ANALYSIS,
        UserIntent.CONFLICT_DETECTION: PipelineRoute.COMPLIANCE_ANALYSIS,
        UserIntent.RISK_ASSESSMENT: PipelineRoute.COMPLIANCE_ANALYSIS,
        UserIntent.CONTRACT_LAW_COMPARISON: PipelineRoute.CROSS_DOCUMENT_QUERY,
        UserIntent.MULTI_DOCUMENT_COMPARISON: PipelineRoute.CROSS_DOCUMENT_QUERY,
        UserIntent.DEFINITION_LOOKUP: PipelineRoute.SIMPLE_CHAT,
        UserIntent.REFERENCE_LOOKUP: PipelineRoute.SIMPLE_CHAT,
        UserIntent.UNCLEAR: PipelineRoute.SIMPLE_CHAT,
    }

    def route(self, intent: UserIntent) -> PipelineRoute:
        """Select pipeline for intent"""
        return self.INTENT_TO_PIPELINE.get(intent, PipelineRoute.SIMPLE_CHAT)

    def extract_parameters(self, intent: UserIntent, message: str) -> Dict[str, Any]:
        """Extract parameters for pipeline"""
        params = {}

        if intent in [UserIntent.COMPLIANCE_CHECK, UserIntent.GAP_ANALYSIS,
                      UserIntent.CONFLICT_DETECTION, UserIntent.RISK_ASSESSMENT]:
            # Compliance analysis parameters
            params["mode"] = "exhaustive"  # Could be "quick" or "exhaustive"
            params["detect_conflicts"] = True
            params["detect_gaps"] = True
            params["calculate_risk"] = True

            if intent == UserIntent.GAP_ANALYSIS:
                params["focus"] = "gaps"
            elif intent == UserIntent.CONFLICT_DETECTION:
                params["focus"] = "conflicts"
            elif intent == UserIntent.RISK_ASSESSMENT:
                params["focus"] = "risks"

        return params


# ============================================================================
# Intent Router - Main Orchestrator
# ============================================================================

class IntentRouter:
    """
    Main Intent Router Agent

    Analyzes user messages and routes them to the appropriate pipeline.
    Leverages existing indexed documents and RAG infrastructure.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Intent Router"""
        import os
        api_key = api_key or os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise ValueError("CLAUDE_API_KEY required for Intent Router")

        self.llm = AsyncAnthropic(api_key=api_key)

        # Initialize components
        self.classifier = IntentClassifier(self.llm)
        self.doc_analyzer = DocumentAnalyzer()
        self.pipeline_router = PipelineRouter()

    async def route(
        self,
        message: str,
        available_contract_ids: List[str],
        available_law_ids: List[str],
        language: str = "cs"
    ) -> RoutingDecision:
        """
        Route user message to appropriate pipeline(s).

        Supports multi-intent queries - if multiple intents detected,
        they will be executed sequentially.

        Args:
            message: User message
            available_contract_ids: List of uploaded contract IDs
            available_law_ids: List of uploaded law IDs
            language: Message language ('cs' or 'en')

        Returns:
            RoutingDecision with intent(s), pipeline(s), and parameters
        """
        # Step 1: Classify intent(s) using LLM
        intents, confidences = await self.classifier.classify(message, language)

        if len(intents) == 1:
            # Single intent - simple case
            intent = intents[0]
            confidence = confidences[0]

            logger.info(f"Classified intent: {intent.value} (confidence: {confidence:.2f})")

            # Analyze document requirements
            doc_requirement = self.doc_analyzer.get_requirements(intent)

            # Check if requirements are satisfied
            requirements_met = doc_requirement.is_satisfied_by(
                available_contract_ids,
                available_law_ids
            )

            # Select pipeline
            pipeline = self.pipeline_router.route(intent)

            # Extract parameters
            parameters = self.pipeline_router.extract_parameters(intent, message)

            # Build routing decision
            decision = RoutingDecision(
                intent=intent,
                pipeline=pipeline,
                confidence=confidence,
                is_multi_intent=False,
                intents=[intent],
                pipelines=[pipeline],
                document_requirement=doc_requirement,
                parameters=parameters
            )

            # Add user message if requirements not met
            if not requirements_met:
                decision.user_message = doc_requirement.get_missing_description(
                    available_contract_ids,
                    available_law_ids,
                    language
                )
                decision.reasoning = "Missing required documents"
            else:
                decision.reasoning = f"Routed to {pipeline.value} based on {intent.value}"

            return decision

        else:
            # Multiple intents - sequential execution
            logger.info(f"Classified {len(intents)} intents: {[i.value for i in intents]}")

            # Get pipelines for all intents
            pipelines = [self.pipeline_router.route(intent) for intent in intents]

            # Merge document requirements (use most restrictive)
            doc_requirements = [self.doc_analyzer.get_requirements(intent) for intent in intents]
            merged_requirement = self._merge_requirements(doc_requirements)

            # Check if requirements are satisfied
            requirements_met = merged_requirement.is_satisfied_by(
                available_contract_ids,
                available_law_ids
            )

            # Extract parameters for all intents
            all_parameters = {}
            for intent in intents:
                params = self.pipeline_router.extract_parameters(intent, message)
                all_parameters.update(params)

            # Build multi-intent routing decision
            decision = RoutingDecision(
                intent=intents[0],  # Primary intent
                pipeline=pipelines[0],  # Primary pipeline
                confidence=confidences[0],
                is_multi_intent=True,
                intents=intents,
                pipelines=pipelines,
                document_requirement=merged_requirement,
                parameters=all_parameters
            )

            # Add user message if requirements not met
            if not requirements_met:
                decision.user_message = merged_requirement.get_missing_description(
                    available_contract_ids,
                    available_law_ids,
                    language
                )
                decision.reasoning = "Missing required documents"
            else:
                intent_names = ", ".join([i.value for i in intents])
                pipeline_names = ", ".join([p.value for p in pipelines])
                decision.reasoning = f"Multi-intent query: {intent_names} → Sequential execution: {pipeline_names}"

            return decision

    def _merge_requirements(
        self,
        requirements: List[DocumentRequirement]
    ) -> DocumentRequirement:
        """Merge multiple document requirements (use most restrictive)"""
        merged = DocumentRequirement()

        for req in requirements:
            if req.contracts_required:
                merged.contracts_required = True
                merged.min_contracts = max(merged.min_contracts, req.min_contracts)
            if req.laws_required:
                merged.laws_required = True
                merged.min_laws = max(merged.min_laws, req.min_laws)

        return merged

    def should_use_compliance_pipeline(self, decision: RoutingDecision) -> bool:
        """Helper: Should we use compliance analysis pipeline?"""
        return decision.pipeline == PipelineRoute.COMPLIANCE_ANALYSIS

    def should_use_cross_document_query(self, decision: RoutingDecision) -> bool:
        """Helper: Should we use cross-document query?"""
        return decision.pipeline == PipelineRoute.CROSS_DOCUMENT_QUERY


# ============================================================================
# Convenience Functions
# ============================================================================

async def route_message(
    message: str,
    available_contract_ids: List[str],
    available_law_ids: List[str],
    language: str = "cs",
    api_key: Optional[str] = None
) -> RoutingDecision:
    """
    Convenience function to route a message.

    Example:
        >>> decision = await route_message(
        ...     "Je smlouva v souladu se zákonem č. 89/2012 Sb.?",
        ...     contract_ids=["contract_001"],
        ...     law_ids=["law_89_2012"]
        ... )
        >>> print(decision.intent)  # UserIntent.COMPLIANCE_CHECK
        >>> print(decision.pipeline)  # PipelineRoute.COMPLIANCE_ANALYSIS
    """
    router = IntentRouter(api_key=api_key)
    return await router.route(
        message,
        available_contract_ids,
        available_law_ids,
        language
    )
