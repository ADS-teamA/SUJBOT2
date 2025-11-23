"""
Multi-Agent Runner - Main CLI interface for multi-agent system.

Orchestrates:
1. Configuration loading
2. System initialization (caching, checkpointing, LangSmith)
3. Agent registry setup
4. Complexity analysis and routing
5. Workflow execution
6. Result formatting

Replaces the old single-agent CLI (src/agent/cli.py).
"""

import logging
import asyncio
import json
from typing import Optional, Dict, Any, AsyncGenerator
from pathlib import Path
from dotenv import load_dotenv

from .core.state import MultiAgentState, ExecutionPhase
from .core.agent_registry import AgentRegistry
from .routing.complexity_analyzer import ComplexityAnalyzer
from .routing.workflow_builder import WorkflowBuilder
from .checkpointing import create_checkpointer, StateManager
from .caching import create_cache_manager
from .observability import setup_langsmith
from .hitl.config import HITLConfig
from .hitl.quality_detector import QualityDetector
from .hitl.clarification_generator import ClarificationGenerator
from .core.event_bus import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class MultiAgentRunner:
    """
    Main runner for multi-agent system.

    Coordinates all components to execute multi-agent workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-agent runner.

        Args:
            config: Full configuration dict with multi_agent section
        """
        self.config = config
        self.multi_agent_config = config.get("multi_agent", {})

        # Initialize components
        self.checkpointer = None
        self.state_manager = None
        self.cache_manager = None
        self.langsmith = None
        self.agent_registry = None
        self.complexity_analyzer = None
        self.workflow_builder = None
        self.vector_store = None  # Store vector store for orchestrator

        # HITL components
        self.hitl_config = None
        self.quality_detector = None
        self.clarification_generator = None

        logger.info("MultiAgentRunner initialized")

    async def initialize(self) -> bool:
        """
        Initialize all systems.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing multi-agent system...")

        try:
            # 1. Initialize LangSmith (observability)
            self.langsmith = setup_langsmith(self.multi_agent_config)

            # 2. Initialize checkpointing (state persistence)
            self.checkpointer = create_checkpointer(self.multi_agent_config)
            self.state_manager = StateManager(self.checkpointer)

            # 3. Initialize caching (prompt caching for cost savings)
            self.cache_manager = create_cache_manager(self.multi_agent_config)

            # 4. Initialize tool registry with RAG components (BEFORE agents)
            await self._initialize_tools()

            # 4.5. Initialize agent registry
            self.agent_registry = AgentRegistry()

            # Register all 8 agents (orchestrator needs vector_store from tools)
            await self._register_agents()

            # 5. Initialize routing components
            routing_config = self.multi_agent_config.get("routing", {})
            self.complexity_analyzer = ComplexityAnalyzer(routing_config)

            # 6. Initialize HITL components (if enabled)
            clarification_config = self.multi_agent_config.get("clarification", {})
            if clarification_config.get("enabled", False):
                logger.info("Initializing HITL clarification system...")

                # Load HITL config
                self.hitl_config = HITLConfig.from_dict(clarification_config)

                # Initialize quality detector
                self.quality_detector = QualityDetector(self.hitl_config)

                # Initialize clarification generator
                api_key = self.config.get("api_keys", {}).get("anthropic_api_key", "")
                self.clarification_generator = ClarificationGenerator(
                    config=self.hitl_config, api_key=api_key
                )

                logger.info("HITL clarification system initialized")
            else:
                logger.info("HITL clarification system disabled")

            # 7. Initialize workflow builder (with HITL components if enabled)
            self.workflow_builder = WorkflowBuilder(
                agent_registry=self.agent_registry,
                checkpointer=self.checkpointer.get_saver() if self.checkpointer else None,
                hitl_config=self.hitl_config,
                quality_detector=self.quality_detector,
                clarification_generator=self.clarification_generator,
            )

            logger.info("Multi-agent system initialized successfully")

            return True

        except Exception as e:
            from .core.error_tracker import track_error, ErrorSeverity

            error_id = track_error(
                error=e,
                severity=ErrorSeverity.CRITICAL,
                context={"phase": "initialization", "config_keys": list(self.multi_agent_config.keys())}
            )

            error_msg = (
                f"[{error_id}] Failed to initialize multi-agent system: {type(e).__name__}: {e}. "
                f"Check: (1) API keys are set in .env file (ANTHROPIC_API_KEY or OPENAI_API_KEY), (2) PostgreSQL is running (if checkpointing enabled), "
                f"(3) all agent configs are present, (4) dependencies are installed."
            )

            logger.error(error_msg, exc_info=True)

            # Store error for health endpoint
            self.initialization_error = {
                "error_id": error_id,
                "error": str(e),
                "type": type(e).__name__,
                "message": f"Multi-agent system failed to initialize [{error_id}]. Check logs for details."
            }

            # CRITICAL: Raise instead of returning False
            # This prevents caller from using uninitialized runner
            raise RuntimeError(error_msg) from e

    async def _register_agents(self) -> None:
        """Register all agents with registry."""
        from .agents import (
            OrchestratorAgent,
            ExtractorAgent,
            ClassifierAgent,
            RequirementExtractorAgent,
            ComplianceAgent,
            RiskVerifierAgent,
            CitationAuditorAgent,
            GapSynthesizerAgent,
        )

        # Get agent configs
        orchestrator_config = self.multi_agent_config.get("orchestrator", {})
        agents_config = self.multi_agent_config.get("agents", {})

        # Register orchestrator (special - used for initial analysis)
        orchestrator_cfg = self._build_agent_config("orchestrator", orchestrator_config)
        orchestrator = OrchestratorAgent(
            orchestrator_cfg,
            vector_store=self.vector_store,
            agent_registry=self.agent_registry
        )
        self.agent_registry.register(orchestrator)

        # Register other agents
        agent_classes = {
            "extractor": ExtractorAgent,
            "classifier": ClassifierAgent,
            "requirement_extractor": RequirementExtractorAgent,
            "compliance": ComplianceAgent,
            "risk_verifier": RiskVerifierAgent,
            "citation_auditor": CitationAuditorAgent,
            "gap_synthesizer": GapSynthesizerAgent,
        }

        for agent_name, agent_class in agent_classes.items():
            agent_cfg = self._build_agent_config(agent_name, agents_config.get(agent_name, {}))
            agent = agent_class(agent_cfg)
            self.agent_registry.register(agent)

        logger.info(f"Registered {len(agent_classes) + 1} agents")

    async def _initialize_tools(self) -> None:
        """Initialize tool registry with RAG components."""
        from ..agent.tools import get_registry
        # Tool modules are auto-imported via tools/__init__.py
        from ..storage import load_vector_store_adapter
        from ..embedding_generator import EmbeddingGenerator
        from ..reranker import CrossEncoderReranker
        from ..agent.config import ToolConfig
        from ..agent.providers.factory import create_provider
        import os

        logger.info("Initializing tool registry with RAG components...")

        # Get storage backend from config
        storage_config = self.config.get("storage", {})
        backend = storage_config.get("backend", "postgresql")

        # Initialize LLM provider for tools (HyDE, Synthesis)
        # Use same model as orchestrator or default to Claude
        tool_model = self.multi_agent_config.get("orchestrator", {}).get("model", "claude-3-5-sonnet-20241022")
        api_key = self.config.get("api_keys", {}).get("anthropic_api_key")

        try:
            self.llm_provider = create_provider(model=tool_model, api_key=api_key)
            logger.info(f"Initialized LLM provider for tools: {tool_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider for tools: {e}. HyDE and Synthesis will be disabled.")
            self.llm_provider = None

        try:
            # Load vector store adapter (PostgreSQL or FAISS)
            if backend == "postgresql":
                # PostgreSQL backend - load from database
                connection_string = os.getenv(
                    storage_config.get("postgresql", {}).get("connection_string_env", "DATABASE_URL")
                )
                if not connection_string:
                    logger.error(
                        "PostgreSQL connection string not found in environment. "
                        "Set DATABASE_URL in .env file."
                    )
                    registry = get_registry()
                    logger.info("Tool registry initialized with 0 tools (no database connection)")
                    return

                logger.info("Loading PostgreSQL vector store adapter...")
                vector_store = await load_vector_store_adapter(
                    backend="postgresql",
                    connection_string=connection_string,
                    pool_size=storage_config.get("postgresql", {}).get("pool_size", 20),
                    dimensions=storage_config.get("postgresql", {}).get("dimensions", 3072)
                )
                logger.info("PostgreSQL adapter loaded successfully")

            else:  # FAISS backend
                vector_store_path = Path(self.config.get("vector_store_path", "vector_db"))

                # Check if FAISS files exist
                if not vector_store_path.exists():
                    logger.warning(
                        f"Vector store not found at {vector_store_path}. "
                        f"Tools requiring vector search will be unavailable. "
                        f"Run indexing pipeline first: uv run python run_pipeline.py data/"
                    )
                    registry = get_registry()
                    logger.info("Tool registry initialized with 0 tools (no vector store)")
                    return

                logger.info(f"Loading FAISS vector store adapter from {vector_store_path}")
                vector_store = await load_vector_store_adapter(
                    backend="faiss",
                    path=str(vector_store_path)
                )
                logger.info("FAISS adapter loaded successfully")

            self.vector_store = vector_store  # Store for orchestrator

            # Initialize embedder (avoid circular dependency - construct config directly)
            from ..config import EmbeddingConfig, ModelConfig

            # Get model info from already-loaded config (self.config was loaded at MultiAgentRunner init)
            # We can't call get_config() here because it might be in the middle of loading
            # Instead, use the root config that was already loaded
            model_name = self.config.get("models", {}).get("embedding_model", "bge-m3")
            model_provider = self.config.get("models", {}).get("embedding_provider", "huggingface")

            # Create EmbeddingConfig with provider/model explicitly set
            # This avoids triggering __post_init__ which would call get_config() recursively
            embedding_config = EmbeddingConfig(
                provider=model_provider,
                model=model_name,
                batch_size=64,
                normalize=True,
                enable_multi_layer=True,
                cache_enabled=True,
                cache_max_size=1000
            )
            embedder = EmbeddingGenerator(embedding_config)
            logger.info(f"Embedder initialized: {model_provider}/{model_name}")

            # Initialize reranker (conditional on config - check agent_tools.enable_reranking)
            agent_tools_config_temp = self.config.get("agent_tools", {})
            enable_reranking = agent_tools_config_temp.get("enable_reranking", True)

            reranker = None
            if enable_reranking:
                try:
                    reranker = CrossEncoderReranker()
                    logger.info("Reranker initialized (enable_reranking=True)")
                except Exception as e:
                    logger.warning(f"Reranker unavailable: {e}. Tools will use base retrieval.")
                    reranker = None
            else:
                logger.info("Reranking DISABLED in config (enable_reranking=False)")

            # Knowledge graph (optional) - supports both Neo4j and JSON backends
            knowledge_graph = None
            kg_config = self.config.get("knowledge_graph", {})
            kg_backend = kg_config.get("backend", "simple")

            if kg_config.get("enable", True):
                try:
                    if kg_backend == "neo4j":
                        # Load from Neo4j database
                        from ..graph.config import Neo4jConfig
                        from ..graph.neo4j_manager import Neo4jManager
                        from ..graph.models import KnowledgeGraph, Entity
                        from datetime import datetime

                        neo4j_cfg = self.config.get("neo4j", {})
                        neo4j_config = Neo4jConfig(
                            uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
                            username=neo4j_cfg.get("username", "neo4j"),
                            password=neo4j_cfg.get("password", ""),
                            database=neo4j_cfg.get("database", "neo4j"),
                        )

                        # Use try/finally to ensure Neo4j connection is always closed
                        neo4j_manager = Neo4jManager(neo4j_config)

                        try:
                            # Query all entities from Neo4j
                            cypher = "MATCH (e:Entity) RETURN e"
                            result = neo4j_manager.execute(cypher)

                            entities = []
                            for record in result:
                                node = record["e"]

                                # Validate required fields
                                entity_id = node.get("id", "")
                                entity_type_str = node.get("type", "")

                                if not entity_id or not entity_type_str:
                                    logger.warning(f"Skipping malformed Neo4j entity: id={entity_id}, type={entity_type_str}")
                                    continue

                                # Convert Neo4j node to Entity
                                # FIX: Convert string type to EntityType enum (was causing AttributeError)
                                try:
                                    from src.graph.models import EntityType
                                    entity_type = EntityType(entity_type_str)
                                except (ValueError, KeyError):
                                    logger.warning(f"Unknown entity type '{entity_type_str}', using UNKNOWN")
                                    entity_type = EntityType.UNKNOWN

                                # FIX: source_chunk_ids should be List[str], not set (type annotation mismatch)
                                source_chunk_ids = node.get("source_chunk_ids", [])
                                if not isinstance(source_chunk_ids, list):
                                    source_chunk_ids = [] if source_chunk_ids is None else list(source_chunk_ids)

                                entities.append(Entity(
                                    id=entity_id,
                                    type=entity_type,  # ✓ Now EntityType enum
                                    value=node.get("value", ""),
                                    normalized_value=node.get("normalized_value", node.get("value", "")),
                                    confidence=node.get("confidence", 1.0),
                                    source_chunk_ids=source_chunk_ids,  # ✓ Now List[str]
                                    document_id=node.get("document_id", ""),
                                    first_mention_chunk_id=node.get("first_mention_chunk_id"),
                                    extraction_method=node.get("extraction_method"),
                                ))

                            # Create KnowledgeGraph from Neo4j entities
                            knowledge_graph = KnowledgeGraph(
                                source_document_id="neo4j_unified",
                                created_at=datetime.now(),
                            )
                            knowledge_graph.entities = entities
                            logger.info(f"✓ KG loaded from Neo4j: {len(entities)} entities")

                        finally:
                            # FIX: Always close Neo4j connection, even if exception occurs (resource leak)
                            neo4j_manager.close()

                    else:
                        # Load from JSON file (simple backend)
                        kg_path = Path("vector_db/unified_kg.json")
                        if not kg_path.exists():
                            kg_path = Path("output/knowledge_graph.json")  # Fallback

                        if kg_path.exists():
                            from ..graph.models import KnowledgeGraph
                            knowledge_graph = KnowledgeGraph.load_json(str(kg_path))
                            logger.info(f"✓ KG loaded from {kg_path.name}: {len(knowledge_graph.entities)} entities")

                except Exception as e:
                    logger.warning(f"Failed to load knowledge graph from {kg_backend}: {e}")

            # Context assembler (optional)
            context_assembler = None
            try:
                from ..context_assembly import ContextAssembler
                context_assembler = ContextAssembler()
                logger.info("Context assembler initialized")
            except Exception as e:
                logger.warning(f"Context assembler unavailable: {e}")

            # Tool config from config.json (not hardcoded defaults)
            agent_tools_config = self.config.get("agent_tools", {})
            tool_config = ToolConfig(
                default_k=agent_tools_config.get("default_k", 6),
                enable_reranking=agent_tools_config.get("enable_reranking", True),
                reranker_candidates=agent_tools_config.get("reranker_candidates", 50),
                reranker_model=agent_tools_config.get("reranker_model", "bge-reranker-large"),
                enable_graph_boost=agent_tools_config.get("enable_graph_boost", True),
                graph_boost_weight=agent_tools_config.get("graph_boost_weight", 0.3),
                max_document_compare=agent_tools_config.get("max_document_compare", 3),
                compliance_threshold=agent_tools_config.get("compliance_threshold", 0.7),
                context_window=agent_tools_config.get("context_window", 2),
                lazy_load_reranker=agent_tools_config.get("lazy_load_reranker", False),
                lazy_load_graph=agent_tools_config.get("lazy_load_graph", True),
                cache_embeddings=agent_tools_config.get("cache_embeddings", True),
                hyde_num_hypotheses=agent_tools_config.get("hyde_num_hypotheses", 3),
                query_expansion_provider=agent_tools_config.get("query_expansion_provider", "openai"),
                query_expansion_model=agent_tools_config.get("query_expansion_model", "gpt-4o-mini"),
            )

            # Initialize tools in registry
            registry = get_registry()

            registry.initialize_tools(
                vector_store=vector_store,
                embedder=embedder,
                reranker=reranker,
                graph_retriever=None,  # TODO: Add graph retriever if needed
                knowledge_graph=knowledge_graph,
                context_assembler=None,  # TODO: Add context assembler if needed
                llm_provider=self.llm_provider,
                config=tool_config,  # Use the full config from above, not minimal config
            )

            # Log results
            total_tools = len(registry)
            unavailable = registry.get_unavailable_tools() if hasattr(registry, 'get_unavailable_tools') else []
            available = total_tools - len(unavailable)

            if unavailable:
                logger.info(f"Tool registry initialized: {available}/{total_tools} tools available")
                logger.info(f"Unavailable tools: {list(unavailable.keys())}")
            else:
                logger.info(f"Tool registry initialized: {total_tools} tools available")

        except Exception as e:
            logger.error(f"Failed to initialize tool registry: {e}", exc_info=True)
            raise ValueError(
                f"Tool initialization failed: {e}. "
                f"Ensure vector store/database is configured and dependencies are installed."
            )

    def _build_agent_config(self, agent_name: str, config: Dict[str, Any]):
        """Build agent config object."""
        from .core.agent_base import AgentConfig, AgentRole, AgentTier

        # Map agent names to their roles and tiers
        agent_metadata = {
            "orchestrator": {"role": AgentRole.ORCHESTRATE, "tier": AgentTier.ORCHESTRATOR},
            "extractor": {"role": AgentRole.EXTRACT, "tier": AgentTier.WORKER},
            "classifier": {"role": AgentRole.CLASSIFY, "tier": AgentTier.SPECIALIST},
            "requirement_extractor": {"role": AgentRole.EXTRACT, "tier": AgentTier.SPECIALIST},
            "compliance": {"role": AgentRole.VERIFY, "tier": AgentTier.SPECIALIST},
            "risk_verifier": {"role": AgentRole.VERIFY, "tier": AgentTier.SPECIALIST},
            "citation_auditor": {"role": AgentRole.AUDIT, "tier": AgentTier.WORKER},
            "gap_synthesizer": {"role": AgentRole.SYNTHESIZE, "tier": AgentTier.SPECIALIST},
        }

        if agent_name not in agent_metadata:
            raise ValueError(f"Unknown agent name: {agent_name}")

        metadata = agent_metadata[agent_name]

        return AgentConfig(
            name=agent_name,
            role=metadata["role"],
            tier=metadata["tier"],
            model=config.get("model", "claude-sonnet-4-5-20250929"),
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.3),
            timeout_seconds=config.get("timeout_seconds", 30),
            tools=config.get("tools", []),
            enable_prompt_caching=config.get("enable_prompt_caching", True),
            api_key=self.config.get("api_keys", {}).get("anthropic_api_key", ""),
        )

    async def run_query(self, query: str, stream_progress: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run query through multi-agent system.

        Args:
            query: User query
            stream_progress: If True, yields intermediate progress updates

        Yields:
            Dict events with type 'progress', 'tool_call', or 'final'.
            Final event has type='final' and contains final_answer and metadata.
        """
        logger.info(f"Running query: {query[:100]}...")

        # Create thread ID
        thread_id = self.state_manager.create_thread_id()

        # Initialize state
        state = MultiAgentState(
            query=query,
            execution_phase=ExecutionPhase.ROUTING,
            agent_sequence=[],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        try:
            # Initialize EventBus for real-time progress streaming
            event_bus = EventBus(max_queue_size=1000)

            # Step 1: Run orchestrator for complexity analysis
            logger.info("Step 1: Analyzing query complexity...")
            orchestrator = self.agent_registry.get_agent("orchestrator")

            state_dict = state.model_dump()
            # Inject EventBus into state (internal infrastructure, excluded from serialization)
            # Note: Use "event_bus" key (no underscore) to match MultiAgentState field name
            state_dict["event_bus"] = event_bus

            # DEBUG: Verify query before orchestrator
            logger.info(f"Query before orchestrator: {state_dict.get('query', 'MISSING')}")

            # Execute orchestrator and MERGE result into existing state (don't overwrite)
            # Run concurrently to allow event streaming during orchestrator execution
            orchestrator_task = asyncio.create_task(orchestrator.execute(state_dict))

            # Stream events while orchestrator is running
            while not orchestrator_task.done():
                # Check for events with short timeout
                events = await event_bus.get_pending_events(timeout=0.1)
                for event in events:
                    if stream_progress:
                        yield self._convert_event_to_sse(event)
                
                # Small sleep to prevent busy loop if no events
                if not events:
                    await asyncio.sleep(0.1)

            # Get result (re-raise exception if failed)
            orchestrator_result = await orchestrator_task
            state_dict.update(orchestrator_result)

            # DEBUG: Verify query after orchestrator
            logger.info(f"Query after orchestrator: {state_dict.get('query', 'MISSING')}")

            # Extract and preserve EventBus before converting back to Pydantic model
            # (event_bus field has exclude=True, so it won't be in model_dump() output)
            preserved_event_bus = state_dict.pop("event_bus", None)

            # Update state
            state = MultiAgentState(**state_dict)

            # DEBUG: Verify query after state reconstruction
            logger.info(f"Query after reconstruction: {state.query if state.query else 'EMPTY STRING'}")

            # Restore EventBus for later use (will be re-injected before workflow execution)
            # CRITICAL: Always ensure event_bus is valid (never None) to prevent AttributeError
            if preserved_event_bus:
                event_bus = preserved_event_bus
            else:
                # EventBus was lost during orchestrator execution (shouldn't happen but defensive)
                logger.warning("EventBus was lost during orchestrator execution, recreating")
                event_bus = EventBus(max_queue_size=1000)

            # Step 2: Check if orchestrator provided direct answer (for greetings/simple queries)
            # When no agents are needed, orchestrator returns final_answer directly
            if hasattr(state, 'final_answer') and state.final_answer and not state.agent_sequence:
                logger.info("Orchestrator provided direct answer without agents")
                # Get accurate cost from CostTracker (model-specific pricing)
                from src.cost_tracker import get_global_tracker
                tracker = get_global_tracker()
                total_cost_usd = tracker.get_total_cost()
                total_cost_cents = total_cost_usd * 100.0

                direct_result = {
                    "success": True,
                    "final_answer": state.final_answer,
                    "complexity_score": state.complexity_score or 0,
                    "query_type": state.query_type.value if hasattr(state.query_type, "value") else str(state.query_type or "direct"),
                    "agent_sequence": [],
                    "documents": [],
                    "citations": [],
                    "total_cost_cents": total_cost_cents,
                    "errors": [],
                }
                # Always yield result (function is now always a generator)
                yield {"type": "final", **direct_result}
                return  # End generator

            # Step 3: Build workflow from agent sequence
            agent_sequence = state.agent_sequence
            logger.info(f"Step 3: Building workflow with sequence: {agent_sequence}")

            if not agent_sequence:
                logger.error("Empty agent sequence without final_answer from orchestrator")

                # Get accurate cost from CostTracker (model-specific pricing)
                from src.cost_tracker import get_global_tracker
                tracker = get_global_tracker()
                total_cost_usd = tracker.get_total_cost()
                total_cost_cents = total_cost_usd * 100.0

                error_result = {
                    "success": False,
                    "final_answer": "Query routing failed: Orchestrator did not provide agent sequence or final answer. Please try rephrasing your query.",
                    "complexity_score": 0,
                    "query_type": "error",
                    "agent_sequence": [],
                    "documents": [],
                    "citations": [],
                    "total_cost_cents": total_cost_cents,
                    "errors": ["Empty agent sequence without direct answer"],
                }
                yield {"type": "final", **error_result}
                return

            workflow = self.workflow_builder.build_workflow(
                agent_sequence=agent_sequence, enable_parallel=True
            )

            # Step 3: Execute workflow with streaming
            logger.info("Step 3: Executing workflow with streaming...")
            state.execution_phase = ExecutionPhase.AGENT_EXECUTION

            # Re-inject EventBus into state dict for workflow execution
            # Note: MultiAgentState has field named "event_bus" (without underscore)
            # to comply with Pydantic V2 (no leading underscores allowed)
            state_dict = state.model_dump()
            state_dict["event_bus"] = event_bus

            # DEBUG: Verify query is in state_dict
            logger.info(f"State dict query before workflow: {state_dict.get('query', 'MISSING')}")

            # Run workflow with streaming to get intermediate state updates
            config = {"configurable": {"thread_id": thread_id}}
            final_result = None

            if self.langsmith and self.langsmith.is_enabled():
                with self.langsmith.trace_workflow(f"query_{thread_id}"):
                    async for state_chunk in workflow.astream(state_dict, config):
                        # state_chunk is a dict with node results
                        # Keep the latest chunk as final result
                        final_result = state_chunk
                        logger.debug(f"Workflow state update: {list(state_chunk.keys())}")

                        # If streaming progress, yield intermediate state
                        if stream_progress:
                            # Extract actual state from node dict
                            # astream() returns {"node_name": {state}}, need to unwrap
                            actual_state = next(iter(state_chunk.values())) if state_chunk else {}

                            # Extract current agent from actual state and emit AGENT_START event
                            current_agent = actual_state.get("current_agent")
                            if current_agent and event_bus:  # Defensive check to prevent AttributeError
                                await event_bus.emit(
                                    event_type=EventType.AGENT_START,
                                    data={"agent": current_agent},
                                    agent_name=current_agent
                                )

                            # Yield pending events from EventBus
                            events = await event_bus.get_pending_events(timeout=0.0) if event_bus else []
                            for event in events:
                                yield self._convert_event_to_sse(event)
            else:
                async for state_chunk in workflow.astream(state_dict, config):
                    final_result = state_chunk
                    logger.debug(f"Workflow state update: {list(state_chunk.keys())}")

                    # If streaming progress, yield intermediate state
                    if stream_progress:
                        # Extract actual state from node dict
                        # astream() returns {"node_name": {state}}, need to unwrap
                        actual_state = next(iter(state_chunk.values())) if state_chunk else {}

                        # Extract current agent from actual state and emit AGENT_START event
                        current_agent = actual_state.get("current_agent")
                        if current_agent and event_bus:  # Defensive check to prevent AttributeError
                            await event_bus.emit(
                                event_type=EventType.AGENT_START,
                                data={"agent": current_agent},
                                agent_name=current_agent
                            )

                        # Yield pending events from EventBus
                        events = await event_bus.get_pending_events(timeout=0.0) if event_bus else []
                        for event in events:
                            yield self._convert_event_to_sse(event)

            if not final_result:
                raise RuntimeError("Workflow produced no output")

            # Extract result - astream() returns dict where keys are node names
            # The final state is in the last yielded chunk
            # Get the actual state from within the node dict
            if final_result:
                # Extract state from node dict (astream returns {"node_name": {state}})
                # Get the first (and should be only) value from the dict
                result = next(iter(final_result.values())) if final_result else {}
            else:
                result = {}

            # Debug: Log what we got from astream
            logger.info(f"Final result node keys: {list(final_result.keys())}")
            logger.info(f"Extracted state keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
            logger.info(f"Final answer in result: {result.get('final_answer', 'NOT FOUND')[:100] if result.get('final_answer') else 'NOT FOUND'}")

            # Step 4: Extract final answer
            final_answer = result.get("final_answer", "No answer generated")

            logger.info("Query execution completed successfully")

            # Get accurate cost from CostTracker (model-specific pricing)
            from src.cost_tracker import get_global_tracker
            tracker = get_global_tracker()
            total_cost_usd = tracker.get_total_cost()
            total_cost_cents = total_cost_usd * 100.0

            # Build final result dict
            final_result_dict = {
                "success": True,
                "final_answer": final_answer,
                "complexity_score": result.get("complexity_score", 0),
                "query_type": result.get("query_type", "unknown"),
                "agent_sequence": result.get("agent_sequence", []),
                "documents": result.get("documents", []),
                "citations": result.get("citations", []),
                "total_cost_cents": total_cost_cents,
                "errors": result.get("errors", []),
            }

            # Always yield final result (function is now always a generator)
            yield {"type": "final", **final_result_dict}
            return

        except Exception as e:
            # Check if this is a GraphInterrupt (HITL clarification needed)
            if e.__class__.__name__ == "GraphInterrupt":
                logger.info("HITL: Workflow interrupted for clarification")

                # Extract interrupt data
                interrupt_value = None
                if hasattr(e, "args") and len(e.args) > 0:
                    # GraphInterrupt contains tuple of Interrupt objects
                    interrupts = e.args[0] if isinstance(e.args[0], tuple) else (e.args[0],)
                    if interrupts and hasattr(interrupts[0], "value"):
                        interrupt_value = interrupts[0].value

                if interrupt_value and interrupt_value.get("type") == "clarification_needed":
                    # Yield clarification result
                    yield {
                        "type": "final",
                        "success": False,
                        "clarification_needed": True,
                        "thread_id": thread_id,
                        "questions": interrupt_value.get("questions", []),
                        "quality_metrics": interrupt_value.get("quality_metrics", {}),
                        "original_query": query,
                        "complexity_score": state.complexity_score,
                        "query_type": state.query_type.value if hasattr(state.query_type, "value") else str(state.query_type),
                        "agent_sequence": state.agent_sequence,  # Needed for resume
                    }
                    return
                # If not clarification interrupt, fall through to error handling below

            # Handle all other exceptions (including unknown GraphInterrupt types)
            from .core.error_tracker import track_error, ErrorSeverity

            error_id = track_error(
                error=e,
                severity=ErrorSeverity.HIGH,
                context={
                    "query": query[:200] if query else "",
                    "complexity_score": state.complexity_score if hasattr(state, 'complexity_score') else None,
                    "agent_sequence": state.agent_sequence if hasattr(state, 'agent_sequence') else [],
                }
            )

            logger.error(
                f"[{error_id}] Query execution failed: {type(e).__name__}: {e}",
                exc_info=True
            )

            # Build actionable error message
            error_message = f"Query processing failed [{error_id}]. "

            # Add specific guidance based on error type
            if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                error_message += "The query is taking too long. Try simplifying your question or breaking it into smaller parts."
            elif "API" in str(e) or "rate limit" in str(e).lower():
                error_message += "API service is temporarily unavailable. Please try again in a few moments."
            elif isinstance(e, MemoryError):
                error_message += "The query requires too much memory. Please contact support."
            else:
                error_message += f"Error: {type(e).__name__}. {str(e)[:200]}"

            yield {
                "type": "final",
                "success": False,
                "final_answer": error_message,
                "errors": [f"[{error_id}] {str(e)}"],
                "error_id": error_id,
                "error_type": type(e).__name__,
            }
            return

    async def resume_with_clarification(
        self, thread_id: str, user_response: str, original_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resume interrupted workflow with user clarification.

        Args:
            thread_id: Thread ID from interrupted workflow
            user_response: User's clarification response
            original_state: Original state dict from before interrupt

        Returns:
            Dict with final_answer and metadata
        """
        logger.info(f"Resuming workflow {thread_id} with user clarification...")

        try:
            # Load original state and add user clarification
            state_dict = dict(original_state)
            state_dict["user_clarification"] = user_response

            # Rebuild workflow (must use same agent sequence as before)
            agent_sequence = state_dict.get("agent_sequence", [])
            logger.info(f"Rebuilding workflow with sequence: {agent_sequence}")

            workflow = self.workflow_builder.build_workflow(
                agent_sequence=agent_sequence, enable_parallel=True
            )

            # Resume workflow with same thread_id (will load from checkpoint)
            logger.info("Resuming workflow execution...")

            if self.langsmith and self.langsmith.is_enabled():
                with self.langsmith.trace_workflow(f"resume_{thread_id}"):
                    result = await workflow.ainvoke(state_dict, {"thread_id": thread_id})
            else:
                result = await workflow.ainvoke(state_dict, {"thread_id": thread_id})

            # Extract final answer
            final_answer = result.get("final_answer", "No answer generated")

            logger.info("Resumed workflow completed successfully")

            # Get accurate cost from CostTracker (model-specific pricing)
            from src.cost_tracker import get_global_tracker
            tracker = get_global_tracker()
            total_cost_usd = tracker.get_total_cost()
            total_cost_cents = total_cost_usd * 100.0

            return {
                "success": True,
                "final_answer": final_answer,
                "complexity_score": result.get("complexity_score", 0),
                "query_type": result.get("query_type", "unknown"),
                "agent_sequence": result.get("agent_sequence", []),
                "documents": result.get("documents", []),
                "citations": result.get("citations", []),
                "total_cost_cents": total_cost_cents,
                "errors": result.get("errors", []),
                "enriched_query": result.get("enriched_query"),
            }

        except Exception as e:
            logger.error(f"Resume failed: {e}", exc_info=True)

            return {
                "success": False,
                "final_answer": f"Error resuming workflow: {str(e)}",
                "errors": [str(e)],
            }

    def _convert_event_to_sse(self, event: Event) -> Dict[str, Any]:
        """
        Convert EventBus event to SSE format for frontend.

        Args:
            event: Event from EventBus

        Returns:
            Dict in SSE format expected by frontend
        """
        if event.event_type == EventType.AGENT_START:
            return {
                "type": "agent_start",
                "agent": event.agent_name,
                "timestamp": event.timestamp.isoformat()
            }
        elif event.event_type == EventType.AGENT_COMPLETE:
            return {
                "type": "agent_complete",
                "agent": event.agent_name,
                "timestamp": event.timestamp.isoformat()
            }
        elif event.event_type == EventType.TOOL_CALL_START:
            return {
                "type": "tool_call",
                "agent": event.agent_name,
                "tool": event.tool_name,
                "status": "running",
                "timestamp": event.timestamp.isoformat()
            }
        elif event.event_type == EventType.TOOL_CALL_COMPLETE:
            return {
                "type": "tool_call",
                "agent": event.agent_name,
                "tool": event.tool_name,
                "status": "completed" if event.data.get("success") else "failed",
                "timestamp": event.timestamp.isoformat()
            }
        else:
            # Unknown event type - pass through with basic formatting
            return {
                "type": event.event_type.value,
                "data": event.data,
                "timestamp": event.timestamp.isoformat()
            }

    def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        logger.info("Shutting down multi-agent system...")

        if self.checkpointer:
            self.checkpointer.close()

        if self.langsmith:
            self.langsmith.disable()

        logger.info("Multi-agent system shut down")


async def main():
    """Main CLI entry point."""
    import argparse

    # Load environment variables from .env file
    load_dotenv()

    # Set up argument parser
    parser = argparse.ArgumentParser(description="SUJBOT2 Multi-Agent System")
    parser.add_argument("--config", type=str, help="Path to config.json", default="config.json")
    parser.add_argument("--query", type=str, help="Query to execute")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize runner
    runner = MultiAgentRunner(config)

    if not await runner.initialize():
        logger.error("Failed to initialize multi-agent system")
        return

    # Run query or interactive mode
    if args.query:
        # Single query mode - consume async generator
        result = None
        async for event in runner.run_query(args.query):
            if event.get("type") == "final":
                result = event
                break

        if not result:
            print("Error: No result returned from query")
            return

        print("\n" + "=" * 80)
        print("FINAL ANSWER:")
        print("=" * 80)
        print(result["final_answer"])
        print("\n" + "=" * 80)
        print(f"Complexity: {result.get('complexity_score', 'N/A')}")
        print(f"Query Type: {result.get('query_type', 'N/A')}")
        print(f"Agent Sequence: {', '.join(result.get('agent_sequence', []))}")
        print(f"Cost: ${result.get('total_cost_cents', 0) / 100:.4f}")
        print("=" * 80)

    elif args.interactive:
        # Interactive mode
        print("SUJBOT2 Multi-Agent System (Interactive Mode)")
        print("Type 'quit' or 'exit' to quit")
        print("=" * 80)

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() in ["quit", "exit"]:
                    break

                if not query:
                    continue

                # Consume async generator
                result = None
                async for event in runner.run_query(query):
                    if event.get("type") == "final":
                        result = event
                        break

                if result:
                    print("\n" + "-" * 80)
                    print(result["final_answer"])
                    print("-" * 80)
                else:
                    print("\nError: No result returned")

            except KeyboardInterrupt:
                print("\nInterrupted")
                break

            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)

    else:
        print("Please specify --query or --interactive mode")

    # Shutdown
    runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
