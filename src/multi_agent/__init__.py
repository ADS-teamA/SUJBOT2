"""
Multi-Agent System for SUJBOT2

LangGraph-based multi-agent orchestration system implementing research-backed
patterns from Harvey AI, Definely, and academic papers (L-MARS, PAKTON, MASLegalBench).

Architecture:
- 8 specialized agents with distinct responsibilities
- Adaptive complexity-based routing
- Per-agent model configuration (cost optimization)
- Distributed tool access (3-5 tools per agent)
- PostgreSQL checkpointing for conversation persistence
- 3-level prompt caching (90% cost savings)
- LangSmith integration for observability

Agents:
1. Orchestrator - Root coordinator
2. Extractor - Document extraction and retrieval
3. Classifier - Content classification
4. Compliance - Regulatory compliance checking
5. Risk Verifier - Risk assessment and verification
6. Citation Auditor - Citation validation
7. Gap Synthesizer - Knowledge gap analysis
8. Report Generator - Final report generation

Research Constraints Preserved:
- Hierarchical document summaries (unchanged)
- Token-aware chunking (512 tokens, unchanged)
- Generic summaries (150 chars, unchanged)
- Multi-layer embeddings (3 FAISS indexes, unchanged)
- Hybrid search (BM25+Dense+RRF, unchanged)
"""

from .core.state import MultiAgentState, AgentState
from .core.agent_base import BaseAgent, AgentConfig, AgentTier, AgentRole
from .core.agent_registry import AgentRegistry, get_agent_registry

__version__ = "2.0.0"
__all__ = [
    "MultiAgentState",
    "AgentState",
    "BaseAgent",
    "AgentConfig",
    "AgentTier",
    "AgentRole",
    "AgentRegistry",
    "get_agent_registry",
]
